#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过滤因子库，只保留市值前100股票的因子

使用方法：
    python scripts/filter_factors_to_top100.py

功能：
1. 加载因子库
2. 过滤到前100股票
3. 保存过滤后的因子库
"""

import sys
from pathlib import Path
import pandas as pd

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.path import get_path

def load_top100_tickers() -> list:
    """加载市值前100股票列表"""
    top100_file = get_path("data/top100_stocks.txt")
    if top100_file.exists():
        with open(top100_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"[Info] 加载前100股票列表: {len(tickers)} 只股票")
        return tickers
    else:
        raise FileNotFoundError(f"找不到前100股票文件: {top100_file}")

def main():
    print("=" * 60)
    print("过滤因子库到前100股票")
    print("=" * 60)
    
    # 加载前100股票列表
    top100_tickers = load_top100_tickers()
    
    # 加载因子库
    factors_path = Path("data/factors/factor_store.parquet")
    if not factors_path.exists():
        raise FileNotFoundError(f"因子库不存在: {factors_path}")
    
    print(f"\n[1] 加载因子库...")
    factors = pd.read_parquet(factors_path)
    
    # 确保索引格式正确
    if isinstance(factors.index, pd.MultiIndex):
        factors.index = pd.MultiIndex.from_tuples(
            [(pd.to_datetime(d), t) for d, t in factors.index],
            names=["date", "ticker"]
        )
    
    original_tickers = factors.index.get_level_values("ticker").nunique()
    original_rows = len(factors)
    
    print(f"  原始因子库:")
    print(f"    股票数: {original_tickers}")
    print(f"    总行数: {original_rows}")
    print(f"    因子数: {len(factors.columns)}")
    print(f"    日期范围: {factors.index.get_level_values('date').min().date()} 到 {factors.index.get_level_values('date').max().date()}")
    
    # 过滤到前100股票
    print(f"\n[2] 过滤到前100股票...")
    available_tickers = set(factors.index.get_level_values("ticker").unique())
    valid_tickers = [t for t in top100_tickers if t in available_tickers]
    
    if len(valid_tickers) < len(top100_tickers):
        missing = set(top100_tickers) - available_tickers
        print(f"  [Warn] {len(missing)} 只前100股票在因子库中不存在: {sorted(list(missing))[:10]}...")
    
    filtered_factors = factors.loc[
        factors.index.get_level_values("ticker").isin(valid_tickers)
    ]
    
    filtered_tickers = filtered_factors.index.get_level_values("ticker").nunique()
    filtered_rows = len(filtered_factors)
    
    print(f"  过滤后因子库:")
    print(f"    股票数: {filtered_tickers}")
    print(f"    总行数: {filtered_rows}")
    print(f"    因子数: {len(filtered_factors.columns)}")
    print(f"    日期范围: {filtered_factors.index.get_level_values('date').min().date()} 到 {filtered_factors.index.get_level_values('date').max().date()}")
    
    # 备份原文件
    backup_path = factors_path.parent / f"{factors_path.stem}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    print(f"\n[3] 备份原因子库...")
    factors.to_parquet(backup_path)
    print(f"  ✓ 已备份到: {backup_path}")
    
    # 保存过滤后的因子库
    print(f"\n[4] 保存过滤后的因子库...")
    filtered_factors.to_parquet(factors_path)
    print(f"  ✓ 已保存到: {factors_path}")
    
    print("\n" + "=" * 60)
    print("✓ 过滤完成！")
    print("=" * 60)
    print(f"  减少股票数: {original_tickers} → {filtered_tickers} (减少 {original_tickers - filtered_tickers} 只)")
    print(f"  减少行数: {original_rows:,} → {filtered_rows:,} (减少 {original_rows - filtered_rows:,} 行)")
    print(f"  备份文件: {backup_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 过滤失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

