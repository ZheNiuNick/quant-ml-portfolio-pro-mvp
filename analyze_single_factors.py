#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单因子IC/ICIR分析
计算每个因子与未来收益的IC和ICIR，用于特征筛选
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import yaml
from scipy import stats as scipy_stats

sys.path.insert(0, str(Path(__file__).parent))

from src.modeling import (
    apply_feature_lag,
    build_label_series,
    calculate_rank_ic_and_icir,
    load_settings,
)

def analyze_single_factors(cfg, use_qlib_label=False):
    """
    分析每个因子的单因子IC和ICIR
    
    Args:
        cfg: 配置字典
        use_qlib_label: 是否使用QLib标签定义
    """
    print("="*60)
    print("单因子IC/ICIR分析")
    print("="*60)
    
    # 加载因子数据
    factor_file = Path(cfg["paths"]["factors_store"])
    if not factor_file.exists():
        print(f"[ERROR] 因子文件不存在: {factor_file}")
        return None
    
    print(f"\n[1] 加载因子数据...")
    X = pd.read_parquet(factor_file)
    
    # Handle different index formats
    if isinstance(X.index, pd.MultiIndex):
        # Already MultiIndex, just ensure names are correct
        if X.index.names != ["date", "ticker"]:
            X.index.names = ["date", "ticker"]
    else:
        # Try to convert to MultiIndex
        try:
            # If index is tuples, convert directly
            if all(isinstance(idx, tuple) and len(idx) == 2 for idx in X.index[:10]):
                X.index = pd.MultiIndex.from_tuples(X.index, names=["date", "ticker"])
            else:
                # Index is not tuples, need to reconstruct from data
                # This shouldn't happen if factor_engine.py is correct, but handle it anyway
                raise ValueError(f"Unexpected index format: {type(X.index)}, first few values: {X.index[:5].tolist()}")
        except (ValueError, TypeError) as e:
            print(f"[ERROR] 无法转换索引格式: {e}")
            print(f"  索引类型: {type(X.index)}")
            print(f"  索引名称: {X.index.names if hasattr(X.index, 'names') else 'N/A'}")
            print(f"  索引前5个值: {X.index[:5].tolist()}")
            raise
    
    X = X.sort_index()
    print(f"    因子数据: {X.shape[0]:,} 行 × {X.shape[1]} 列")
    
    # 加载价格数据并计算标签
    print(f"\n[2] 加载价格数据并计算标签...")
    prices = pd.read_parquet(cfg["paths"]["prices_parquet"])
    
    # Handle different index formats
    if isinstance(prices.index, pd.MultiIndex):
        if prices.index.names != ["date", "ticker"]:
            prices.index.names = ["date", "ticker"]
    else:
        try:
            if all(isinstance(idx, tuple) and len(idx) == 2 for idx in prices.index[:10]):
                prices.index = pd.MultiIndex.from_tuples(prices.index, names=["date", "ticker"])
            else:
                raise ValueError(f"Unexpected price index format: {type(prices.index)}")
        except (ValueError, TypeError) as e:
            print(f"[ERROR] 无法转换价格数据索引格式: {e}")
            raise
    
    prices = prices.sort_index()
    
    # 对齐索引
    idx = X.index.intersection(prices.index)
    X = X.reindex(idx)

    feature_lag_days = int(cfg.get("model", {}).get("feature_lag_days", 0))
    if feature_lag_days > 0:
        print(f"    应用特征滞后: {feature_lag_days} 天")
        X = apply_feature_lag(X, feature_lag_days)
    
    # 计算标签
    label_cfg = cfg.get("model", {}).get("label_options", {})
    if use_qlib_label:
        print("    使用QLib标签: Ref($close, -2)/Ref($close, -1) - 1")
    else:
        print(f"    使用自定义标签: horizon_days={label_cfg.get('horizon_days', 1)}, "
              f"shift_days={label_cfg.get('shift_days', 0)}")
    y = build_label_series(prices, idx, cfg, use_qlib_label=use_qlib_label).reindex(idx)
    
    print(f"    对齐后数据: {len(X):,} 行")
    print(f"    标签缺失率: {y.isna().sum() / len(y) * 100:.2f}%")
    
    # 计算每个因子的IC和ICIR
    print(f"\n[3] 计算每个因子的IC和ICIR...")
    print(f"    共 {len(X.columns)} 个因子需要分析...")
    
    factor_stats = {}
    total_factors = len(X.columns)
    
    for idx, factor_name in enumerate(X.columns, 1):
        if idx % 20 == 0 or idx == 1 or idx == total_factors:
            progress = idx / total_factors * 100
            print(f"    进度: {idx}/{total_factors} ({progress:.1f}%) - {factor_name}")
        
        # 获取因子值和标签
        factor_series = X[factor_name]
        
        # 计算IC和ICIR
        rank_ic, rank_icir = calculate_rank_ic_and_icir(factor_series, y)
        
        # 计算IC统计
        df_eval = pd.DataFrame({"factor": factor_series, "y": y}).dropna()
        if len(df_eval) > 0:
            # 按日期计算每日IC
            daily_ics = []
            for date in df_eval.index.get_level_values(0).unique():
                date_data = df_eval.xs(date, level=0)
                if len(date_data) >= 2:
                    factor_ranks = date_data["factor"].rank(method="first")
                    y_ranks = date_data["y"].rank(method="first")
                    if factor_ranks.nunique() > 1 and y_ranks.nunique() > 1:
                        ic = factor_ranks.corr(y_ranks, method="spearman")
                        if not np.isnan(ic):
                            daily_ics.append(ic)
            
            if len(daily_ics) > 0:
                ic_mean = np.mean(daily_ics)
                ic_std = np.std(daily_ics)
                ic_ir = ic_mean / ic_std if ic_std > 0 else 0.0
                
                # 1. IC胜率（IC > 0 的比例）
                ic_win_rate = sum(1 for ic in daily_ics if ic > 0) / len(daily_ics) if len(daily_ics) > 0 else 0.0
                
                # 2. IC显著性检验（t-test，检验IC是否显著不为0）
                if len(daily_ics) > 1:
                    t_stat, p_value = scipy_stats.ttest_1samp(daily_ics, 0)
                    ic_significant = p_value < 0.05
                else:
                    t_stat, p_value = 0.0, 1.0
                    ic_significant = False
                
                # 3. IC的绝对值统计（用于评估信号强度）
                ic_abs_mean = np.mean([abs(ic) for ic in daily_ics])
                
                # 4. IC的最大值和最小值
                ic_max = float(np.max(daily_ics))
                ic_min = float(np.min(daily_ics))
                
                factor_stats[factor_name] = {
                    "mean_ic": float(ic_mean),
                    "std_ic": float(ic_std),
                    "icir": float(ic_ir),
                    "ic_win_rate": float(ic_win_rate),  # IC胜率
                    "ic_t_stat": float(t_stat),  # t统计量
                    "ic_p_value": float(p_value),  # p值
                    "ic_significant": bool(ic_significant),  # 是否显著
                    "ic_abs_mean": float(ic_abs_mean),  # 平均绝对IC
                    "ic_max": float(ic_max),  # 最大IC
                    "ic_min": float(ic_min),  # 最小IC
                    "valid_days": len(daily_ics),
                    "total_samples": len(df_eval)
                }
            else:
                factor_stats[factor_name] = {
                    "mean_ic": 0.0,
                    "std_ic": 0.0,
                    "icir": 0.0,
                    "ic_win_rate": 0.0,
                    "ic_t_stat": 0.0,
                    "ic_p_value": 1.0,
                    "ic_significant": False,
                    "ic_abs_mean": 0.0,
                    "ic_max": 0.0,
                    "ic_min": 0.0,
                    "valid_days": 0,
                    "total_samples": len(df_eval)
                }
        else:
            factor_stats[factor_name] = {
                "mean_ic": 0.0,
                "std_ic": 0.0,
                "icir": 0.0,
                "ic_win_rate": 0.0,
                "ic_t_stat": 0.0,
                "ic_p_value": 1.0,
                "ic_significant": False,
                "ic_abs_mean": 0.0,
                "ic_max": 0.0,
                "ic_min": 0.0,
                "valid_days": 0,
                "total_samples": 0
            }
    
    # 保存结果
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存详细结果（包含所有因子）
    summary_path = output_dir / "single_factor_summary.json"
    with open(summary_path, "w") as f:
        json.dump(factor_stats, f, indent=2)
    print(f"\n[4] 详细结果已保存: {summary_path}")
    
    # 生成统计报告
    print(f"\n[5] 因子统计报告:")
    icir_values = [v["icir"] for v in factor_stats.values()]
    mean_ic_values = [v["mean_ic"] for v in factor_stats.values()]
    win_rates = [v["ic_win_rate"] for v in factor_stats.values() if v["valid_days"] > 0]
    significant_count = sum(1 for v in factor_stats.values() if v.get("ic_significant", False))
    
    print(f"    总因子数: {len(factor_stats)}")
    print(f"    平均ICIR: {np.mean(icir_values):.6f}")
    print(f"    平均IC: {np.mean(mean_ic_values):.6f}")
    if win_rates:
        print(f"    平均IC胜率: {np.mean(win_rates):.2%}")
    print(f"    显著因子数: {significant_count} 个 ({significant_count/len(factor_stats)*100:.1f}%)")
    
    # 按ICIR排序
    sorted_factors = sorted(factor_stats.items(), key=lambda x: abs(x[1]["icir"]), reverse=True)
    
    print(f"\n    前10个因子（按|ICIR|排序）:")
    for i, (factor, stats) in enumerate(sorted_factors[:10], 1):
        sig_mark = "✓" if stats.get("ic_significant", False) else " "
        win_mark = "✓" if stats.get("ic_win_rate", 0) > 0.6 else " "
        print(f"      {i:2d}. {factor:20s} - IC: {stats['mean_ic']:7.4f}, ICIR: {stats['icir']:7.4f}, "
              f"胜率: {stats.get('ic_win_rate', 0):5.1%} {win_mark}, 显著: {sig_mark}")
    
    # 统计不同ICIR阈值的因子数量（符合业内标准）
    print(f"\n    不同ICIR阈值的因子数量（业内标准）:")
    thresholds = [0.01, 0.02, 0.05, 0.5, 1.0]
    for threshold in thresholds:
        count = sum(1 for v in factor_stats.values() if abs(v["icir"]) > threshold)
        print(f"      |ICIR| > {threshold:4.2f}: {count:3d} 个因子 ({count/len(factor_stats)*100:5.1f}%)")
    
    # 综合筛选（符合业内标准）
    print(f"\n    综合筛选（符合业内标准）:")
    print(f"      筛选条件:")
    print(f"        - |IC均值| > 0.02")
    print(f"        - |ICIR| > 0.5")
    print(f"        - IC胜率 > 60%")
    print(f"        - IC显著 (p < 0.05)")
    
    good_factors_strict = []
    good_factors_moderate = []
    
    for factor, stats in factor_stats.items():
        # 严格筛选（符合业内标准）
        conditions_strict = [
            abs(stats['mean_ic']) > 0.02,
            abs(stats['icir']) > 0.5,
            stats.get('ic_win_rate', 0) > 0.6,
            stats.get('ic_significant', False),
        ]
        if all(conditions_strict):
            good_factors_strict.append(factor)
        
        # 中等筛选（适合学生项目，降低阈值）
        conditions_moderate = [
            abs(stats['mean_ic']) > 0.005,
            abs(stats['icir']) > 0.05,
            stats.get('ic_win_rate', 0) > 0.50,
        ]
        if all(conditions_moderate):
            good_factors_moderate.append(factor)
    
    print(f"      严格筛选（全部条件）: {len(good_factors_strict)} 个因子")
    if len(good_factors_strict) > 0:
        print(f"        因子列表: {', '.join(good_factors_strict[:10])}")
        if len(good_factors_strict) > 10:
            print(f"        ... 还有 {len(good_factors_strict) - 10} 个")
    
    print(f"      中等筛选（适合学生项目）: {len(good_factors_moderate)} 个因子")
    if len(good_factors_moderate) > 0:
        print(f"        因子列表: {', '.join(good_factors_moderate[:10])}")
        if len(good_factors_moderate) > 10:
            print(f"        ... 还有 {len(good_factors_moderate) - 10} 个")
    
    # 保存筛选建议（包含综合筛选结果）
    recommendations = {
        "total_factors": len(factor_stats),
        "thresholds": {},
        "industry_standard": {
            "strict": {
                "count": len(good_factors_strict),
                "factors": good_factors_strict,
                "criteria": {
                    "min_ic": 0.02,
                    "min_icir": 0.5,
                    "min_win_rate": 0.6,
                    "require_significant": True
                }
            },
            "moderate": {
                "count": len(good_factors_moderate),
                "factors": good_factors_moderate,
                "criteria": {
                    "min_ic": 0.005,
                    "min_icir": 0.05,
                    "min_win_rate": 0.50,
                    "require_significant": False
                }
            }
        }
    }
    for threshold in [0.01, 0.02, 0.05, 0.5, 1.0]:
        good_factors = [k for k, v in factor_stats.items() if abs(v["icir"]) > threshold]
        recommendations["thresholds"][f"icir_{threshold}"] = {
            "count": len(good_factors),
            "factors": good_factors[:20]  # 只保存前20个
        }
    
    rec_path = output_dir / "factor_selection_recommendations.json"
    with open(rec_path, "w") as f:
        json.dump(recommendations, f, indent=2)
    print(f"\n[6] 筛选建议已保存: {rec_path}")
    
    print("\n" + "="*60)
    print("✅ 单因子分析完成！")
    print("="*60)
    print("\n使用建议:")
    print("  1. 查看 outputs/reports/single_factor_summary.json 了解每个因子的详细统计")
    print("  2. 查看 outputs/reports/factor_selection_recommendations.json 了解筛选建议")
    print("  3. 根据项目需求选择筛选标准:")
    print("     - 严格筛选（业内标准）: 使用 industry_standard.strict 中的因子")
    print("     - 中等筛选（学生项目）: 使用 industry_standard.moderate 中的因子")
    print("     - 自定义筛选: 在 config/settings.yaml 中设置:")
    print("       model:")
    print("         filter_by_icir: true")
    print("         min_icir: 0.02  # 根据分析结果调整阈值")
    print("  4. 重新训练模型，将只使用筛选后的因子")
    
    return factor_stats

if __name__ == "__main__":
    cfg = load_settings()
    use_qlib_label = cfg.get("model", {}).get("use_qlib_label", False)
    analyze_single_factors(cfg, use_qlib_label=use_qlib_label)

