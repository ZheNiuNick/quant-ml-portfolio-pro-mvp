#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate comprehensive backtest visualizations + factor contribution summary.

Usage:
    python scripts/generate_performance_report.py \
        --weights outputs/portfolios/weights.parquet \
        --prices data/processed/prices.parquet \
        --daily outputs/backtests/daily_returns.parquet \
        --shap outputs/reports/shap_top5.json

Outputs (saved under outputs/backtests/):
    - performance_overview.png  (equity, drawdown, rolling Sharpe, daily returns)
    - return_histogram.png
    - strategy_vs_benchmark.png
    - excess_return_curve.png
    - rolling_alpha_beta.png
    - excess_return_hist.png
    - monthly_performance.csv
    - analysis_summary.json
    - factor_contribution.csv  (cross-sectional IC / contribution for top SHAP factors)
"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_daily_returns(path: Path) -> pd.DataFrame:
    daily = pd.read_parquet(path)
    return ensure_datetime(daily)


def generate_performance_overview(daily: pd.DataFrame, out_dir: Path):
    net = daily["net_return"]
    equity = (1 + net).cumprod()
    max_equity = equity.cummax()
    drawdown = equity / max_equity - 1

    rolling_win = 60
    rolling_return = net.rolling(rolling_win).mean() * 252
    rolling_vol = net.rolling(rolling_win).std(ddof=0) * np.sqrt(252)
    rolling_sharpe = (rolling_return / rolling_vol).replace([np.inf, -np.inf], np.nan)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)
    axes[0].plot(equity.index, equity, label="Equity Curve")
    axes[0].set_title("Equity Curve")
    axes[0].set_ylabel("Growth (x)")
    axes[0].legend()

    axes[1].fill_between(drawdown.index, drawdown, 0, color="red", alpha=0.4)
    axes[1].set_title("Drawdown")
    axes[1].set_ylabel("Drawdown")

    axes[2].plot(rolling_sharpe.index, rolling_sharpe, label="Rolling Sharpe (60d)")
    axes[2].axhline(0, color="black", linewidth=0.5)
    axes[2].set_title("Rolling Sharpe (60 trading days)")
    axes[2].legend()

    axes[3].bar(net.index, net, width=1.0, color="gray")
    axes[3].set_title("Daily Net Returns")
    axes[3].set_ylabel("Return")
    axes[3].set_xlabel("Date")

    plt.tight_layout()
    plt.savefig(out_dir / "performance_overview.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.hist(net, bins=50, color="steelblue", alpha=0.8)
    ax2.set_title("Distribution of Daily Net Returns")
    ax2.set_xlabel("Return")
    ax2.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "return_histogram.png", dpi=150)
    plt.close(fig2)


def load_prices(path: Path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    data.index = pd.MultiIndex.from_tuples(data.index, names=["date", "ticker"])
    close = data["Adj Close"].unstack("ticker").sort_index()
    close = close.ffill().bfill()
    return close


def generate_benchmark_charts(net: pd.Series, prices: pd.DataFrame, out_dir: Path):
    bench_ret = prices.pct_change(fill_method=None).fillna(0.0).mean(axis=1)
    common_idx = net.index.intersection(bench_ret.index)
    net = net.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]

    strategy_curve = (1 + net).cumprod()
    benchmark_curve = (1 + bench_ret).cumprod()
    excess_ret = net - bench_ret
    excess_curve = (1 + excess_ret).cumprod()

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(strategy_curve.index, strategy_curve, label="Strategy")
    ax.plot(benchmark_curve.index, benchmark_curve, linestyle="--", label="Equal-Weight Benchmark")
    ax.set_title("Cumulative Return vs Benchmark")
    ax.set_ylabel("Growth (x)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "strategy_vs_benchmark.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ax2.plot(excess_curve.index, excess_curve, color="purple")
    ax2.set_title("Cumulative Excess Return (Strategy - Benchmark)")
    ax2.set_ylabel("Excess Growth (x)")
    plt.tight_layout()
    plt.savefig(out_dir / "excess_return_curve.png", dpi=150)
    plt.close(fig2)

    window = 60
    alphas = []
    betas = []
    idx_vals = []
    for i in range(window, len(net) + 1):
        r = net.iloc[i - window : i]
        b = bench_ret.iloc[i - window : i]
        X = np.vstack([np.ones(len(b)), b.values]).T
        coef = np.linalg.lstsq(X, r.values, rcond=None)[0]
        alpha_daily = coef[0]
        beta = coef[1]
        alphas.append(alpha_daily * 252)
        betas.append(beta)
        idx_vals.append(net.index[i - 1])

    rolling_alpha = pd.Series(alphas, index=idx_vals)
    rolling_beta = pd.Series(betas, index=idx_vals)

    fig3, ax3 = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax3[0].plot(rolling_alpha.index, rolling_alpha, label="Rolling Alpha (annualized)")
    ax3[0].axhline(0, color="black", linewidth=0.5)
    ax3[0].set_ylabel("Alpha")
    ax3[0].legend()

    ax3[1].plot(rolling_beta.index, rolling_beta, color="orange", label="Rolling Beta")
    ax3[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax3[1].set_ylabel("Beta")
    ax3[1].set_xlabel("Date")
    ax3[1].legend()

    plt.tight_layout()
    plt.savefig(out_dir / "rolling_alpha_beta.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.hist(excess_ret, bins=50, color="darkgreen", alpha=0.8)
    ax4.set_title("Distribution of Daily Excess Returns")
    ax4.set_xlabel("Strategy - Benchmark Return")
    ax4.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(out_dir / "excess_return_hist.png", dpi=150)
    plt.close(fig4)

    monthly_returns = (1 + net).resample("ME").prod() - 1
    monthly_excess = (1 + excess_ret).resample("ME").prod() - 1
    monthly_bench = (1 + bench_ret).resample("ME").prod() - 1
    pd.DataFrame(
        {
            "strategy_return": monthly_returns,
            "excess_return": monthly_excess,
            "benchmark_return": monthly_bench,
        }
    ).to_csv(out_dir / "monthly_performance.csv")

    return bench_ret


def save_summary(net: pd.Series, out_dir: Path):
    equity = (1 + net).cumprod()
    drawdown = equity / equity.cummax() - 1
    summary = {
        "start": str(net.index[0].date()),
        "end": str(net.index[-1].date()),
        "days": int(len(net)),
        "total_return": float(equity.iloc[-1] - 1),
        "annualized_return": float(net.mean() * 252),
        "annualized_vol": float(net.std(ddof=1) * np.sqrt(252)),
        "sharpe": float(net.mean() / net.std(ddof=1) * np.sqrt(252) if net.std(ddof=1) != 0 else 0.0),
        "max_drawdown": float(drawdown.min()),
    }
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


def factor_contribution(factor_path: Path, prices_path: Path, shap_path: Path, out_dir: Path):
    if not shap_path.exists():
        print(f"[Warn] SHAP file not found: {shap_path}, skip factor contribution.")
        return
    with open(shap_path, "r") as f:
        shap_data = json.load(f)
    top_factors: List[str] = shap_data.get("top5_features", [])
    if not top_factors:
        print("[Warn] No top features in shap file.")
        return

    factor_store = pd.read_parquet(factor_path)
    factor_store.index = pd.MultiIndex.from_tuples(factor_store.index, names=["date", "ticker"])
    factor_store = factor_store.sort_index()

    prices = pd.read_parquet(prices_path)
    prices.index = pd.MultiIndex.from_tuples(prices.index, names=["date", "ticker"])
    prices = prices.sort_index()

    close = prices["Adj Close"].unstack("ticker").sort_index()
    fwd_ret = close.shift(-1) / close - 1.0
    fwd_ret = fwd_ret.stack().rename("fwd_return")

    df = factor_store.join(fwd_ret, how="inner")
    df = df.dropna(subset=["fwd_return"])

    results = []
    for factor in top_factors:
        if factor not in df.columns:
            continue
        factor_series = df[factor]
        corr_daily = (
            pd.concat([factor_series, df["fwd_return"]], axis=1)
            .groupby(level="date")
            .apply(lambda g: g[factor].corr(g["fwd_return"], method="spearman"))
        )
        results.append(
            {
                "factor": factor,
                "daily_ic_mean": corr_daily.mean(),
                "daily_ic_std": corr_daily.std(ddof=1),
                "coverage_days": int(corr_daily.count()),
            }
        )

    if results:
        pd.DataFrame(results).to_csv(out_dir / "factor_contribution.csv", index=False)
    else:
        print("[Warn] No factors matched for contribution analysis.")


def main():
    parser = argparse.ArgumentParser(description="Generate backtest report visuals and factor contribution.")
    parser.add_argument("--daily", default="outputs/backtests/daily_returns.parquet")
    parser.add_argument("--prices", default="data/processed/prices.parquet")
    parser.add_argument("--weights", default="outputs/portfolios/weights.parquet", help="reserved for future use")
    parser.add_argument("--factor-store", default="data/factors/factor_store.parquet")
    parser.add_argument("--shap", default="outputs/reports/shap_top5.json")
    args = parser.parse_args()

    out_dir = Path("outputs/backtests")
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = load_daily_returns(Path(args.daily))
    generate_performance_overview(daily, out_dir)
    net = daily["net_return"]

    prices = load_prices(Path(args.prices))
    bench_ret = generate_benchmark_charts(net, prices, out_dir)

    save_summary(net, out_dir)

    factor_contribution(
        Path(args.factor_store),
        Path(args.prices),
        Path(args.shap),
        out_dir,
    )

    print("[OK] Report generation complete.")


if __name__ == "__main__":
    main()

