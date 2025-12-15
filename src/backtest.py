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
        if not os.path.isabs(path):
            path = get_path(path)
        else:
            path = Path(path)
    elif isinstance(path, Path):
        if not path.is_absolute():
            path = get_path(str(path))
        # else: path is already absolute, no change needed
    
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
        "max_drawdown": float(max_drawdown),
    }


def run_backtest(cfg: Dict):
    """Run a backtest using optimizer-produced weights and price data."""
    weights_dir = cfg["paths"]["output_portfolios"]
    prices_path = cfg["paths"]["data_processed"] / "prices.parquet"
    output_dir = cfg["paths"]["output_backtests"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prices
    prices = pd.read_parquet(prices_path)
    prices = prices.set_index(["date", "ticker"])["adj_close"].unstack("ticker")

    # Find weight files
    weight_files = sorted(Path(weights_dir).glob("weights_*.parquet"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files found in {weights_dir}")

    # Process each weight file
    all_returns = []
    all_dates = []

    for weight_file in weight_files:
        date_str = weight_file.stem.replace("weights_", "")
        date = pd.to_datetime(date_str)

        # Load weights
        weights_df = pd.read_parquet(weight_file)
        if "ticker" in weights_df.columns:
            weights_df = weights_df.set_index("ticker")
        weights = weights_df["weight"]

        # Get prices for this date and next trading day
        if date not in prices.index:
            continue
        current_prices = prices.loc[date]
        next_dates = prices.index[prices.index > date]
        if len(next_dates) == 0:
            continue
        next_date = next_dates[0]
        next_prices = prices.loc[next_date]

        # Compute returns for assets we hold
        asset_returns = (next_prices / current_prices - 1).fillna(0.0)

        # Align weights with asset returns
        common_tickers = weights.index.intersection(asset_returns.index)
        if len(common_tickers) == 0:
            continue

        weights_aligned = weights.loc[common_tickers]
        returns_aligned = asset_returns.loc[common_tickers]

        # Portfolio return = sum(weights * returns)
        portfolio_return = (weights_aligned * returns_aligned).sum()

        # Estimate trading cost (simplified: assume 0.1% per trade on turnover)
        # Turnover = sum(abs(weight_change))
        # For simplicity, assume full turnover on rebalancing days
        turnover = 1.0  # Full rebalance
        trading_cost = turnover * 0.001  # 0.1% cost
        net_return = portfolio_return - trading_cost

        all_returns.append(net_return)
        all_dates.append(date)

    if not all_returns:
        raise ValueError("No returns computed - check data alignment")

    # Create returns series
    returns_series = pd.Series(all_returns, index=all_dates, name="returns")
    returns_series = returns_series.sort_index()

    # Compute risk metrics
    risk_metrics = risk_analysis(returns_series)

    # Save results
    output_file = output_dir / "backtest_results.json"
    results = {
        "returns": returns_series.to_dict(),
        "summary": risk_metrics,
    }
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Backtest complete. Results saved to {output_file}")
    print(f"Annualized return: {risk_metrics['annualized_return']:.2%}")
    print(f"Information ratio: {risk_metrics['information_ratio']:.2f}")
    print(f"Max drawdown: {risk_metrics['max_drawdown']:.2%}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run portfolio backtest")
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    args = parser.parse_args()
    
    cfg = load_settings(args.config) if args.config else load_settings()
    run_backtest(cfg)
