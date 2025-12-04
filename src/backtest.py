#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple portfolio backtest for the current workflow.
It reads optimizer-produced weights, aligns them with adjusted close prices,
computes daily returns = Σ (weights * asset_returns), subtracts trading costs
based on turnover, and outputs daily performance plus summary statistics.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import yaml

# 使用统一的路径管理
from src.config.path import ROOT_DIR, SETTINGS_FILE, OUTPUT_BACKTESTS_DIR, get_path

SETTINGS = SETTINGS_FILE


def load_settings(path = SETTINGS_FILE) -> Dict:
    """加载配置文件，支持绝对路径和相对路径"""
    if isinstance(path, str):
        path = get_path(path) if not os.path.isabs(path) else Path(path)
    else:
        path = get_path(str(path)) if not path.is_absolute() else path
    
    with open(path, "r") as f:
        return yaml.safe_load(f)


def risk_analysis(returns: pd.Series, periods_per_year: int = 252) -> Dict[str, float]:
    """Compute basic risk metrics for a daily return series."""
    mean_ret = returns.mean()
    std_ret = returns.std(ddof=1)
    annualized = mean_ret * periods_per_year
    info_ratio = mean_ret / std_ret * np.sqrt(periods_per_year) if std_ret != 0 else 0.0
    equity_curve = (1 + returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1
    max_drawdown = float(drawdown.min())
    return {
        "mean": float(mean_ret),
        "std": float(std_ret),
        "annualized_return": float(annualized),
        "information_ratio": float(info_ratio),
        "max_drawdown": max_drawdown,
    }


def run_backtest(cfg: Dict):
    print("[Backtest] Loading weights and prices...")
    weights_path = get_path(cfg["paths"]["portfolio_path"])
    
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Portfolio weights not found: {weights_path}. "
            "Run `python src/optimizer.py --optimize` first."
        )

    weights = pd.read_parquet(weights_path).sort_index()
    weights.index = pd.to_datetime(weights.index)
    weights = weights.fillna(0.0)
    
    prices_path = get_path(cfg["paths"]["prices_parquet"])
    prices = pd.read_parquet(prices_path)
    prices.index = pd.MultiIndex.from_tuples(prices.index, names=["date", "ticker"])
    
    # 修复：去除重复索引（保留最后一个）
    if prices.index.duplicated().any():
        print(f"[Warning] Found {prices.index.duplicated().sum()} duplicate (date, ticker) entries, removing duplicates...")
        prices = prices[~prices.index.duplicated(keep='last')]
        print(f"[OK] Removed duplicates, remaining: {len(prices)} rows")
    
    close_prices = prices["Adj Close"].unstack("ticker").sort_index()

    common_dates = weights.index.intersection(close_prices.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between weights and prices.")

    weights = weights.loc[common_dates].fillna(0.0)
    close_prices = close_prices.loc[common_dates].ffill().bfill()
    returns = close_prices.pct_change(fill_method=None).fillna(0.0)

    weight_sum = weights.sum(axis=1).replace(0, np.nan)
    weights_norm = weights.div(weight_sum, axis=0).fillna(0.0)

    # Align signals: use weights from previous day to trade today
    weights_shifted = weights_norm.shift(1).iloc[1:]
    returns = returns.iloc[1:]

    gross_returns = (weights_shifted * returns).sum(axis=1)

    # Turnover approximation: 0.5 * Σ|w_t - w_{t-1}|, first day = Σ|w_0|
    turnover = weights_shifted.diff().abs().sum(axis=1).fillna(0.0)
    turnover = 0.5 * turnover

    exchange_cfg = cfg.get("backtest", {}).get("exchange_kwargs", {})
    fee_rate = exchange_cfg.get("open_cost", 0.0005) + exchange_cfg.get("close_cost", 0.0015)
    cost_rate = turnover * fee_rate
    net_returns = gross_returns - cost_rate

    equity_curve = (1 + net_returns).cumprod()
    
    # 计算基准收益（所有股票等权重）
    benchmark_returns = returns.mean(axis=1).iloc[1:]
    benchmark_equity = (1 + benchmark_returns).cumprod()
    
    # 统一字段名：添加 strategy_return, benchmark_return, nav, benchmark_nav, timestamp
    daily = pd.DataFrame(
        {
            "strategy_return": net_returns,  # 统一字段名
            "benchmark_return": benchmark_returns,  # 基准收益
            "gross_return": gross_returns,
            "net_return": net_returns,  # 保留旧字段名兼容
            "turnover": turnover,
            "cost": cost_rate,
            "nav": equity_curve * 100,  # NAV 以 100 为基准
            "benchmark_nav": benchmark_equity * 100,
            "equity_curve": equity_curve,
            "timestamp": pd.to_datetime(net_returns.index).astype(int) // 10**9,  # Unix timestamp
        }
    )

    metrics = risk_analysis(net_returns)
    total_return = float(equity_curve.iloc[-1] - 1.0)

    summary = {
        "start_date": str(daily.index[0].date()),
        "end_date": str(daily.index[-1].date()),
        "num_days": int(len(daily)),
        "total_return": total_return,
        "annualized_return": metrics["annualized_return"],
        "sharpe_ratio": metrics["information_ratio"],
        "max_drawdown": metrics["max_drawdown"],
        "avg_turnover": float(turnover.mean()),
        "avg_cost": float(cost_rate.mean()),
    }

    output_dir = OUTPUT_BACKTESTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(output_dir / "daily_returns.parquet")
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[Backtest] Summary")
    print(f"  Start: {summary['start_date']}  End: {summary['end_date']}  Days: {summary['num_days']}")
    print(f"  Total Return: {summary['total_return']*100:.2f}%")
    print(f"  Annualized Return: {summary['annualized_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown: {summary['max_drawdown']*100:.2f}%")
    print(f"  Avg Turnover: {summary['avg_turnover']:.4f}")
    print(f"  Avg Cost: {summary['avg_cost']:.6f}")
    print(f"[OK] Saved daily returns and summary to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="Run backtest with current weights")
    args = parser.parse_args()
    
    if not args.run:
        parser.print_help()
        return
    
    cfg = load_settings()
    run_backtest(cfg)


if __name__ == "__main__":
    main()

