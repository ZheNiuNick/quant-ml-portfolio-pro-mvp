#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
为现有因子数据生成IC/ICIR数据

这个脚本用于为已经存在的因子数据生成IC/ICIR数据，即使因子数据已经是最新的。
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# 抑制警告
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*ConstantInputWarning.*')
warnings.filterwarnings('ignore', message='.*correlation coefficient is not defined.*')

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.path import DATA_FACTORS_DIR, ROOT_DIR, get_path
from src.factor_engine import read_prices, forward_return, daily_rank_ic, load_settings

def generate_ic_ir_for_existing_factors(cfg, start_date=None, end_date=None):
    """为现有因子数据生成IC/ICIR数据"""
    print("\n" + "="*60)
    print("为现有因子数据生成IC/ICIR数据")
    print("="*60)
    
    # 读取因子数据
    factor_store_rel_path = cfg["paths"]["factors_store"]
    # 如果已经是绝对路径，直接使用；否则基于项目根目录解析
    if Path(factor_store_rel_path).is_absolute():
        factor_store_path = Path(factor_store_rel_path)
    else:
        factor_store_path = (ROOT_DIR / factor_store_rel_path).resolve()
    
    if not factor_store_path.exists():
        print(f"[错误] 因子数据文件不存在: {factor_store_path}")
        return
    
    print(f"[Info] 读取因子数据: {factor_store_path}")
    factor_store = pd.read_parquet(factor_store_path)
    
    # 确保索引是 MultiIndex
    if not isinstance(factor_store.index, pd.MultiIndex):
        if "date" in factor_store.columns and "ticker" in factor_store.columns:
            factor_store["date"] = pd.to_datetime(factor_store["date"])
            factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        else:
            print("[错误] 因子数据格式不正确")
            return
    
    # 获取日期范围
    all_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
    
    if start_date:
        start_date = pd.to_datetime(start_date)
        all_dates = all_dates[all_dates >= start_date]
    
    if end_date:
        end_date = pd.to_datetime(end_date)
        all_dates = all_dates[all_dates <= end_date]
    
    print(f"[Info] 需要处理的日期数: {len(all_dates)}")
    print(f"[Info] 日期范围: {all_dates.min()} 到 {all_dates.max()}")
    
    # 读取价格数据
    print("\n[Info] 读取价格数据...")
    prices = read_prices(cfg)
    if prices is None or len(prices) == 0:
        print("[错误] 无法读取价格数据")
        return
    
    # 修复重复索引
    if prices.index.duplicated().any():
        print("[Info] 发现价格数据中有重复索引，正在去重...")
        prices = prices[~prices.index.duplicated(keep='first')]
    
    # 计算未来1日收益
    print("[Info] 计算未来1日收益...")
    forward_ret = forward_return(prices, horizon=1)
    
    # 存储所有IC/ICIR数据
    all_ic_data = []
    
    # 逐日计算IC
    for idx, date in enumerate(all_dates, 1):
        try:
            print(f"\n[{idx}/{len(all_dates)}] 处理日期: {date.strftime('%Y-%m-%d')}")
            
            # 获取该日期的因子数据
            date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date]
            if len(date_factors) == 0:
                print(f"  [跳过] 该日期没有因子数据")
                continue
            
            # 获取该日期的未来收益
            date_forward_ret = forward_ret.loc[forward_ret.index.get_level_values(0) == date]
            
            if len(date_forward_ret) == 0:
                print(f"  [跳过] 该日期没有未来收益数据")
                continue
            
            # 计算每个因子的IC
            ic_dict = {}
            for factor_name in date_factors.columns:
                factor_series = date_factors[factor_name]
                
                # 对齐索引
                aligned = pd.concat([factor_series, date_forward_ret], axis=1).dropna()
                if len(aligned) < 10:  # 至少需要10个样本
                    ic_dict[factor_name] = np.nan
                    continue
                
                # 计算该日期的Rank IC（Spearman相关系数）
                try:
                    # 检查是否有足够的变异性
                    factor_vals = aligned.iloc[:, 0]
                    ret_vals = aligned.iloc[:, 1]
                    
                    # 如果因子值或收益值都是常数，无法计算IC
                    if factor_vals.nunique() < 2 or ret_vals.nunique() < 2:
                        ic_dict[factor_name] = np.nan
                        continue
                    
                    ic = factor_vals.rank().corr(ret_vals.rank(), method='spearman')
                    ic_dict[factor_name] = ic if not pd.isna(ic) else np.nan
                except Exception as e:
                    ic_dict[factor_name] = np.nan
            
            # 计算ICIR（需要历史IC序列）
            # 对于单日数据，ICIR为NaN，但我们可以计算历史ICIR
            ic_series = pd.Series(ic_dict)
            valid_ic_count = ic_series.notna().sum()
            
            # 保存IC数据
            for factor_name, ic_value in ic_dict.items():
                all_ic_data.append({
                    "date": date,
                    "factor": factor_name,
                    "ic": ic_value if not pd.isna(ic_value) else 0.0,
                    "icir": np.nan  # 单日数据无法计算ICIR
                })
            
            # 只在显示进度时输出成功信息
            if idx % 100 == 0 or idx == 1 or idx == len(all_dates):
                print(f"  [成功] 有效IC: {valid_ic_count}/{len(ic_dict)}")
            
        except Exception as e:
            # 只在显示进度时输出失败信息
            if idx % 100 == 0 or idx == 1 or idx == len(all_dates):
                print(f"  [失败] {str(e)}")
            # 记录错误但继续处理
            pass
    
    # 计算ICIR（使用60天滚动窗口）
    # Note: ICIR is calculated per-factor using rolling window, but stored in the same file
    # The API endpoint will recalculate ICIR using daily aggregated IC (across all factors)
    print("\n[Info] ICIR将在API端点使用60天滚动窗口实时计算（基于每日聚合IC）")
    print("[Info] 此处只保存每日IC数据，ICIR留空（由API计算）")
    ic_df = pd.DataFrame(all_ic_data)
    
    # ICIR不在这里计算，因为API端点是按日期聚合所有因子的IC后计算滚动ICIR
    # 这样可以与Rolling ICIR (60D)的计算方式保持一致
    # 将所有ICIR设置为NaN（由API计算）
    if 'icir' not in ic_df.columns:
        ic_df['icir'] = np.nan
        
        # 保存IC/ICIR数据
        ic_store_path = factor_store_path.parent / "factor_ic_ir.parquet"
        ic_df.to_parquet(ic_store_path)
        print(f"\n[成功] IC/ICIR数据已保存到 {ic_store_path}")
        print(f"[Info] 总记录数: {len(ic_df)}")
        print(f"[Info] 有效IC数: {ic_df['ic'].notna().sum()}")
    print(f"[Info] 日期范围: {ic_df['date'].min()} 到 {ic_df['date'].max()}")
    print(f"[Info] 交易天数: {ic_df['date'].nunique()}")
    if ic_df['date'].nunique() < 60:
        print(f"[提示] 当前只有 {ic_df['date'].nunique()} 个交易日，Rolling IC/ICIR/t-stat需要至少60个交易日")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="为现有因子数据生成IC/ICIR数据")
    parser.add_argument("--start-date", type=str, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="结束日期 (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_settings()
    
    # 执行生成
    generate_ic_ir_for_existing_factors(
        cfg,
        start_date=args.start_date,
        end_date=args.end_date
    )

