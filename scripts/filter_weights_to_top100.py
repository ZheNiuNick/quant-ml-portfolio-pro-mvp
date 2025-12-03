#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过滤权重文件，只保留市值前100股票的权重

使用方法：
    python scripts/filter_weights_to_top100.py

功能：
1. 加载权重文件
2. 过滤到前100股票
3. 保存过滤后的权重文件
"""

import sys
from pathlib import Path
import pandas as pd

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

def load_top100_tickers() -> list:
    """加载市值前100股票列表"""
    top100_file = Path("data/top100_stocks.txt")
    if top100_file.exists():
        with open(top100_file, "r") as f:
            tickers = [line.strip() for line in f if line.strip()]
        print(f"[Info] 加载前100股票列表: {len(tickers)} 只股票")
        return tickers
    else:
        raise FileNotFoundError(f"找不到前100股票文件: {top100_file}")

def main():
    print("=" * 60)
    print("过滤权重文件到前100股票")
    print("=" * 60)
    
    # 加载前100股票列表
    top100_tickers = load_top100_tickers()
    
    # 加载权重文件
    weights_path = Path("outputs/portfolios/weights.parquet")
    if not weights_path.exists():
        raise FileNotFoundError(f"权重文件不存在: {weights_path}")
    
    print(f"\n[1] 加载权重文件...")
    weights = pd.read_parquet(weights_path)
    weights.index = pd.to_datetime(weights.index)
    
    original_tickers = weights.shape[1]
    original_dates = len(weights)
    
    print(f"  原始权重文件:")
    print(f"    股票数: {original_tickers}")
    print(f"    日期数: {original_dates}")
    print(f"    日期范围: {weights.index.min().date()} 到 {weights.index.max().date()}")
    
    # 过滤到前100股票
    print(f"\n[2] 过滤到前100股票...")
    available_tickers = set(weights.columns)
    valid_tickers = [t for t in top100_tickers if t in available_tickers]
    
    if len(valid_tickers) < len(top100_tickers):
        missing = set(top100_tickers) - available_tickers
        print(f"  [Warn] {len(missing)} 只前100股票在权重文件中不存在: {sorted(list(missing))[:10]}...")
    
    # 只保留前100股票的列
    filtered_weights = weights[valid_tickers].copy()
    
    # 重新归一化（确保每日权重和为1）
    filtered_weights = filtered_weights.div(filtered_weights.sum(axis=1), axis=0).fillna(0.0)
    
    filtered_tickers = filtered_weights.shape[1]
    
    print(f"  过滤后权重文件:")
    print(f"    股票数: {filtered_tickers}")
    print(f"    日期数: {len(filtered_weights)}")
    print(f"    日期范围: {filtered_weights.index.min().date()} 到 {filtered_weights.index.max().date()}")
    
    # 检查最新日期的持仓
    latest_date = filtered_weights.index.max()
    latest_weights = filtered_weights.loc[latest_date]
    latest_positions = (latest_weights > 0).sum()
    print(f"    最新日期 ({latest_date.date()}) 持仓数: {latest_positions}")
    
    # 备份原文件
    backup_path = weights_path.parent / f"{weights_path.stem}_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    print(f"\n[3] 备份原权重文件...")
    weights.to_parquet(backup_path)
    print(f"  ✓ 已备份到: {backup_path}")
    
    # 保存过滤后的权重文件
    print(f"\n[4] 保存过滤后的权重文件...")
    filtered_weights.to_parquet(weights_path)
    print(f"  ✓ 已保存到: {weights_path}")
    
    print("\n" + "=" * 60)
    print("✓ 过滤完成！")
    print("=" * 60)
    print(f"  减少股票数: {original_tickers} → {filtered_tickers} (减少 {original_tickers - filtered_tickers} 只)")
    print(f"  备份文件: {backup_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ 过滤失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

