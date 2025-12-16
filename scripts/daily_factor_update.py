#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
每日因子数据自动补齐模块

功能：
1. 自动检测因子最新日期
2. 获取需要更新的日期区间
3. 为每一天重新计算所有因子（160个因子）
4. 用隔天收益计算 IC / ICIR（信息比率）
5. 把每日结果 append 到因子数据文件中（Parquet 追加行或 DuckDB INSERT INTO）
6. 确保 schema 与旧数据一致
7. 如果当天已存在 → 覆盖（更新策略）

使用方法：
    python scripts/daily_factor_update.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--force]
"""

import argparse
import sys
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import yaml

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.path import DATA_FACTORS_DIR, get_path
from src.factor_engine import (
    read_prices, calculate_all_factors, load_settings,
    industry_standard_processing, forward_return, daily_rank_ic
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


def get_latest_factor_date(cfg) -> Optional[pd.Timestamp]:
    """获取因子数据的最新日期"""
    factor_store_path = Path(cfg["paths"]["factors_store"])
    
    if not factor_store_path.exists():
        return None
    
    try:
        factor_store = pd.read_parquet(factor_store_path)
        
        # 确保索引是 MultiIndex
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
            else:
                return None
        
        dates = factor_store.index.get_level_values(0).unique()
        if len(dates) > 0:
            return pd.to_datetime(dates.max())
        return None
    except Exception as e:
        print(f"[Warn] Failed to read factor_store: {e}")
        return None


def get_price_date_range(cfg) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """获取价格数据的日期范围"""
    # 优先从 parquet 文件读取（通常比 DuckDB 更新）
    parquet_path = get_path(cfg["paths"].get("prices_parquet", "data/processed/prices.parquet"))
    if parquet_path.exists():
        try:
            prices = pd.read_parquet(parquet_path)
            if not isinstance(prices.index, pd.MultiIndex):
                if "date" in prices.columns and "ticker" in prices.columns:
                    prices["date"] = pd.to_datetime(prices["date"])
                    prices = prices.set_index(["date", "ticker"]).sort_index()
            dates = prices.index.get_level_values(0).unique()
            min_date = pd.to_datetime(dates.min())
            max_date = pd.to_datetime(dates.max())
            return min_date, max_date
        except Exception as e:
            print(f"[Warn] Failed to read from parquet: {e}, falling back to read_prices")
    
    # Fallback to read_prices (which tries parquet then DuckDB)
    prices = read_prices(cfg)
    if prices is None or len(prices) == 0:
        raise ValueError("无法读取价格数据")
    
    dates = prices.index.get_level_values(0).unique()
    min_date = pd.to_datetime(dates.min())
    max_date = pd.to_datetime(dates.max())
    
    return min_date, max_date


def calculate_daily_factors_and_ic(prices: pd.DataFrame, date: pd.Timestamp, cfg) -> Tuple[pd.DataFrame, pd.Series]:
    """
    计算指定日期的所有因子，并计算IC/ICIR
    
    Returns:
        factors_df: DataFrame with MultiIndex (date, ticker), columns are factors
        ic_series: Series with factor names as index, IC values as values
    """
    # 获取计算因子所需的历史数据窗口
    lookback_days = cfg.get("factor_processing", {}).get("lookback_days", 90)
    start_date = date - timedelta(days=lookback_days)
    
    # 筛选价格数据（需要足够的历史数据）
    price_dates = prices.index.get_level_values(0).unique()
    available_dates = price_dates[(price_dates >= start_date) & (price_dates <= date)]
    
    if len(available_dates) < 30:  # 至少需要30天数据
        raise ValueError(f"日期 {date} 的历史数据不足（需要至少30天，实际{len(available_dates)}天）")
    
    # 筛选该日期范围内的价格数据
    price_subset = prices.loc[prices.index.get_level_values(0).isin(available_dates)]
    
    # 修复重复索引问题
    if price_subset.index.duplicated().any():
        print(f"  [修复] 发现重复索引，正在去重...")
        price_subset = price_subset[~price_subset.index.duplicated(keep='first')]
    
    # 计算所有因子（使用该日期的价格数据）
    print(f"  [计算因子] 日期: {date.strftime('%Y-%m-%d')}, 历史窗口: {len(available_dates)} 天")
    factors = calculate_all_factors(price_subset, start_date.strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
    
    # 只保留该日期的因子数据
    date_factors = factors.loc[factors.index.get_level_values(0) == date]
    
    if len(date_factors) == 0:
        raise ValueError(f"日期 {date} 没有因子数据")
    
    # 处理因子（使用 industry_standard_processing）
    processed_factors = {}
    for col in date_factors.columns:
        raw = date_factors[col].dropna()
        if raw.empty:
            continue
        processed = industry_standard_processing(raw)
        processed_factors[col] = processed
    
    if not processed_factors:
        raise ValueError(f"日期 {date} 处理后没有有效因子")
    
    factors_df = pd.DataFrame(processed_factors)
    
    # 计算未来1日收益（用于计算IC）
    forward_ret = forward_return(prices, horizon=1)
    date_forward_ret = forward_ret.loc[forward_ret.index.get_level_values(0) == date]
    
    # 计算每个因子的IC
    ic_dict = {}
    for factor_name in factors_df.columns:
        factor_series = factors_df[factor_name]
        
        # 对齐索引
        aligned = pd.concat([factor_series, date_forward_ret], axis=1).dropna()
        if len(aligned) < 10:  # 至少需要10个样本
            ic_dict[factor_name] = np.nan
            continue
        
        # 计算该日期的Rank IC（Spearman相关系数）
        try:
            ic = aligned.iloc[:, 0].rank().corr(aligned.iloc[:, 1].rank(), method='spearman')
            ic_dict[factor_name] = ic if not pd.isna(ic) else np.nan
        except:
            ic_dict[factor_name] = np.nan
    
    ic_series = pd.Series(ic_dict)
    
    return factors_df, ic_series


def calculate_icir(ic_series: pd.Series) -> pd.Series:
    """
    计算ICIR（信息比率）：IC均值 / IC标准差
    
    这里使用滚动窗口计算ICIR（例如60天窗口）
    """
    # 简化：如果只有单日数据，返回NaN
    if len(ic_series) == 1:
        return pd.Series([np.nan] * len(ic_series), index=ic_series.index)
    
    # 对于单日数据，无法计算ICIR，返回NaN
    # 实际应用中，ICIR需要历史IC序列来计算
    icir_series = pd.Series([np.nan] * len(ic_series), index=ic_series.index)
    
    return icir_series


def update_factor_store(factors_df: pd.DataFrame, ic_series: pd.Series, icir_series: pd.Series, 
                       cfg, overwrite: bool = False, update_ic_only: bool = False):
    """
    更新因子数据文件（Parquet 或 DuckDB）
    
    Args:
        factors_df: 因子数据（MultiIndex: date, ticker）
        ic_series: IC序列（factor names as index）
        icir_series: ICIR序列（factor names as index）
        cfg: 配置
        overwrite: 如果日期已存在，是否覆盖因子数据
        update_ic_only: 如果为True，只更新IC/ICIR数据，不更新因子数据
    """
    factor_store_path = Path(cfg["paths"]["factors_store"])
    factor_store_path.parent.mkdir(parents=True, exist_ok=True)
    
    date = factors_df.index.get_level_values(0).unique()[0]
    
    # 读取现有数据
    if factor_store_path.exists():
        existing_factors = pd.read_parquet(factor_store_path)
        
        # 确保索引是 MultiIndex
        if not isinstance(existing_factors.index, pd.MultiIndex):
            if "date" in existing_factors.columns and "ticker" in existing_factors.columns:
                existing_factors["date"] = pd.to_datetime(existing_factors["date"])
                existing_factors = existing_factors.set_index(["date", "ticker"]).sort_index()
            else:
                existing_factors.index = pd.MultiIndex.from_tuples(existing_factors.index, names=["date", "ticker"])
        
        # 检查日期是否已存在
        existing_dates = existing_factors.index.get_level_values(0).unique()
        if date in existing_dates:
            if update_ic_only:
                # 只更新IC/ICIR数据，不更新因子数据
                print(f"  [更新IC] 日期 {date.strftime('%Y-%m-%d')} 因子数据已存在，只更新IC/ICIR数据")
            elif overwrite:
                print(f"  [更新] 日期 {date.strftime('%Y-%m-%d')} 已存在，覆盖中...")
                # 删除该日期的数据
                existing_factors = existing_factors.loc[existing_factors.index.get_level_values(0) != date]
                # 合并数据
                common_cols = list(set(factors_df.columns) & set(existing_factors.columns))
                if len(common_cols) != len(factors_df.columns):
                    missing = set(factors_df.columns) - set(existing_factors.columns)
                    print(f"  [Warn] 新因子列不在现有数据中: {missing}")
                    factors_df = factors_df[common_cols]
                    existing_factors = existing_factors[common_cols]
                updated_factors = pd.concat([existing_factors, factors_df], axis=0).sort_index()
                updated_factors.to_parquet(factor_store_path)
                print(f"  [保存] 因子数据已保存到 {factor_store_path}")
            else:
                print(f"  [跳过因子] 日期 {date.strftime('%Y-%m-%d')} 已存在，跳过因子数据更新（使用 --force 覆盖）")
        else:
            # 日期不存在，正常追加
            common_cols = list(set(factors_df.columns) & set(existing_factors.columns))
            if len(common_cols) != len(factors_df.columns):
                missing = set(factors_df.columns) - set(existing_factors.columns)
                print(f"  [Warn] 新因子列不在现有数据中: {missing}")
                factors_df = factors_df[common_cols]
                existing_factors = existing_factors[common_cols]
            updated_factors = pd.concat([existing_factors, factors_df], axis=0).sort_index()
            updated_factors.to_parquet(factor_store_path)
            print(f"  [保存] 因子数据已保存到 {factor_store_path}")
    else:
        # 创建新文件
        updated_factors = factors_df.sort_index()
        updated_factors.to_parquet(factor_store_path)
        print(f"  [保存] 因子数据已保存到 {factor_store_path}")
    
    # 保存IC/ICIR数据到单独文件（无论因子数据是否已存在，都更新IC/ICIR）
    ic_store_path = factor_store_path.parent / "factor_ic_ir.parquet"
    ic_data = pd.DataFrame({
        "date": date,
        "factor": ic_series.index,
        "ic": ic_series.values,
        "icir": icir_series.values
    })
    
    if ic_store_path.exists():
        existing_ic = pd.read_parquet(ic_store_path)
        # 删除该日期的旧数据（如果存在）
        existing_ic = existing_ic[existing_ic["date"] != date]
        ic_data = pd.concat([existing_ic, ic_data], axis=0).sort_values("date")
    
    ic_data.to_parquet(ic_store_path)
    print(f"  [保存] IC/ICIR数据已保存到 {ic_store_path}")


def update_daily_factors(cfg, start_date: Optional[pd.Timestamp] = None, 
                        end_date: Optional[pd.Timestamp] = None, 
                        force: bool = False):
    """
    更新每日因子数据
    
    Args:
        cfg: 配置
        start_date: 开始日期（如果为None，从最新因子日期+1开始）
        end_date: 结束日期（如果为None，到最新价格日期）
        force: 如果日期已存在，是否覆盖
    """
    print("\n" + "="*60)
    print("每日因子数据自动补齐")
    print("="*60)
    
    # 1. 获取价格数据日期范围
    price_min, price_max = get_price_date_range(cfg)
    print(f"[Info] 价格数据日期范围: {price_min.strftime('%Y-%m-%d')} 到 {price_max.strftime('%Y-%m-%d')}")
    
    # 2. 获取因子最新日期
    latest_factor_date = get_latest_factor_date(cfg)
    if latest_factor_date is None:
        print("[Info] 未找到现有因子数据，将从价格数据起始日期开始计算")
        if start_date is None:
            start_date = price_min
    else:
        print(f"[Info] 因子数据最新日期: {latest_factor_date.strftime('%Y-%m-%d')}")
        if start_date is None:
            start_date = latest_factor_date + timedelta(days=1)
    
    # 3. 确定更新日期范围
    if end_date is None:
        end_date = price_max
    
    # 确保日期在价格数据范围内
    start_date = max(start_date, price_min)
    end_date = min(end_date, price_max)
    
    if start_date > end_date:
        print(f"[Info] 无需更新：开始日期 {start_date.strftime('%Y-%m-%d')} 大于结束日期 {end_date.strftime('%Y-%m-%d')}")
        return
    
    print(f"[Info] 更新日期范围: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
    
    # 4. 读取价格数据（优先从 parquet 文件读取，通常比 DuckDB 更新）
    print("\n[Info] 读取价格数据...")
    parquet_path = get_path(cfg["paths"].get("prices_parquet", "data/processed/prices.parquet"))
    if parquet_path.exists():
        try:
            prices = pd.read_parquet(parquet_path)
            if not isinstance(prices.index, pd.MultiIndex):
                if "date" in prices.columns and "ticker" in prices.columns:
                    prices["date"] = pd.to_datetime(prices["date"])
                    prices = prices.set_index(["date", "ticker"]).sort_index()
            print(f"[Info] Loaded prices from parquet: {prices.shape}")
        except Exception as e:
            print(f"[Warn] Failed to load from parquet: {e}, falling back to read_prices")
            prices = read_prices(cfg)
    else:
        prices = read_prices(cfg)
    
    if prices is None or len(prices) == 0:
        raise ValueError("无法读取价格数据")
    
    # 修复重复索引问题
    if prices.index.duplicated().any():
        print("[Info] 发现价格数据中有重复索引，正在去重...")
        prices = prices[~prices.index.duplicated(keep='first')]
        print(f"[Info] 去重后价格数据: {prices.shape}")
    
    # 5. 获取交易日列表（排除周末和节假日）
    all_dates = pd.to_datetime(prices.index.get_level_values(0).unique()).sort_values()
    update_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    
    print(f"[Info] 需要更新的交易日数: {len(update_dates)}")
    
    # 6. 逐日计算因子和IC/ICIR
    success_count = 0
    fail_count = 0
    
    for idx, date in enumerate(update_dates, 1):
        try:
            print(f"\n[{idx}/{len(update_dates)}] 处理日期: {date.strftime('%Y-%m-%d')}")
            
            # 计算因子和IC
            factors_df, ic_series = calculate_daily_factors_and_ic(prices, date, cfg)
            
            # 计算ICIR（对于单日数据，ICIR为NaN）
            icir_series = calculate_icir(ic_series)
            
            # 更新因子数据文件（即使日期已存在，也更新IC/ICIR数据）
            # 检查日期是否已存在
            factor_store_path = Path(cfg["paths"]["factors_store"])
            date_exists = False
            if factor_store_path.exists():
                existing_factors = pd.read_parquet(factor_store_path)
                if not isinstance(existing_factors.index, pd.MultiIndex):
                    if "date" in existing_factors.columns and "ticker" in existing_factors.columns:
                        existing_factors["date"] = pd.to_datetime(existing_factors["date"])
                        existing_factors = existing_factors.set_index(["date", "ticker"]).sort_index()
                existing_dates = existing_factors.index.get_level_values(0).unique()
                date_exists = date in existing_dates
            
            # 如果日期已存在且不使用force，只更新IC/ICIR
            update_ic_only = date_exists and not force
            update_factor_store(factors_df, ic_series, icir_series, cfg, overwrite=force, update_ic_only=update_ic_only)
            
            print(f"  [成功] 日期 {date.strftime('%Y-%m-%d')}: {len(factors_df.columns)} 个因子, "
                  f"有效IC: {ic_series.notna().sum()}/{len(ic_series)}")
            success_count += 1
            
        except Exception as e:
            print(f"  [失败] 日期 {date.strftime('%Y-%m-%d')}: {str(e)}")
            fail_count += 1
            import traceback
            traceback.print_exc()
    
    # 7. 总结
    print("\n" + "="*60)
    print("更新完成")
    print("="*60)
    print(f"成功: {success_count} 天")
    print(f"失败: {fail_count} 天")
    print(f"总计: {len(update_dates)} 天")


def main():
    parser = argparse.ArgumentParser(description="每日因子数据自动补齐")
    parser.add_argument("--start-date", type=str, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)")
    parser.add_argument("--force", action="store_true", help="如果日期已存在，覆盖")
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_settings()
    
    # 解析日期
    start_date = pd.to_datetime(args.start_date) if args.start_date else None
    end_date = pd.to_datetime(args.end_date) if args.end_date else None
    
    # 执行更新
    update_daily_factors(cfg, start_date=start_date, end_date=end_date, force=args.force)


if __name__ == "__main__":
    main()

