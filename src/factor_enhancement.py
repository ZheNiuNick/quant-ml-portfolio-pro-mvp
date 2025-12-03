#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子增强模块：降维、因子质量改进、信息保留分析
业内常用方法：
1. PCA降维
2. 因子分析（Factor Analysis）
3. 因子中性化（行业/市值）
4. 因子组合优化
5. 信息保留分析
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.decomposition import PCA, FactorAnalysis
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[Warn] sklearn not available, PCA/FA features disabled")

# ------------------------
# 信息保留分析
# ------------------------
def analyze_information_preservation(raw: pd.Series, processed: pd.Series) -> Dict:
    """
    分析处理前后因子的信息保留情况
    
    Returns:
        dict with correlation, rank_correlation, information_ratio
    """
    # 对齐索引
    common_idx = raw.index.intersection(processed.index)
    raw_aligned = raw.reindex(common_idx).dropna()
    processed_aligned = processed.reindex(common_idx).dropna()
    final_idx = raw_aligned.index.intersection(processed_aligned.index)
    
    if len(final_idx) < 100:
        return {
            "correlation": 0.0,
            "rank_correlation": 0.0,
            "information_ratio": 0.0,
            "samples": len(final_idx)
        }
    
    raw_final = raw_aligned.reindex(final_idx)
    processed_final = processed_aligned.reindex(final_idx)
    
    # Pearson相关性
    corr = raw_final.corr(processed_final)
    
    # Spearman秩相关性（更关注排序信息）
    rank_corr = raw_final.rank().corr(processed_final.rank())
    
    # 信息比率（保留的方差比例）
    info_ratio = processed_final.std() / raw_final.std() if raw_final.std() > 0 else 0.0
    
    return {
        "correlation": float(corr),
        "rank_correlation": float(rank_corr),
        "information_ratio": float(info_ratio),
        "samples": len(final_idx)
    }


def compare_processing_methods(factor_store: pd.DataFrame, 
                               sample_factor: str = None) -> pd.DataFrame:
    """
    对比不同处理方法的信息保留情况
    
    Returns:
        DataFrame with comparison results
    """
    if sample_factor is None:
        # 选择IC最高的因子
        sample_factor = factor_store.columns[0]
    
    if sample_factor not in factor_store.columns:
        raise ValueError(f"Factor {sample_factor} not found")
    
    raw = factor_store[sample_factor].dropna()
    
    results = []
    
    # 方法1：QLib风格（只处理异常值）
    from src.factor_engine import qlib_style_processing
    qlib_proc = qlib_style_processing(raw)
    qlib_info = analyze_information_preservation(raw, qlib_proc)
    results.append({
        "method": "qlib_style",
        "factor": sample_factor,
        **qlib_info
    })
    
    # 方法2：轻量处理（Winsorize + Z-score）
    from src.factor_engine import light_processing
    light_proc = light_processing(raw)
    light_info = analyze_information_preservation(raw, light_proc)
    results.append({
        "method": "light_processing",
        "factor": sample_factor,
        **light_info
    })
    
    # 方法3：只Winsorize（不标准化）
    from src.factor_engine import winsorize_by_group
    winsorized = winsorize_by_group(raw, 0.01, 0.99, level="date")
    winsor_info = analyze_information_preservation(raw, winsorized)
    results.append({
        "method": "winsorize_only",
        "factor": sample_factor,
        **winsor_info
    })
    
    # 方法4：只Z-score（不Winsorize）
    from src.factor_engine import zscore_by_group
    zscored = zscore_by_group(raw, level="date")
    zscore_info = analyze_information_preservation(raw, zscored)
    results.append({
        "method": "zscore_only",
        "factor": sample_factor,
        **zscore_info
    })
    
    return pd.DataFrame(results)


# ------------------------
# 降维方法
# ------------------------
def apply_pca_reduction(factor_store: pd.DataFrame,
                        n_components: int = None,
                        explained_variance_threshold: float = 0.95,
                        cfg: dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    使用PCA降维，保留主要信息
    
    Args:
        factor_store: 因子数据框
        n_components: 主成分数量（如果为None，使用explained_variance_threshold）
        explained_variance_threshold: 累计解释方差阈值
        cfg: 配置字典
    
    Returns:
        (reduced_factors, pca_info)
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for PCA")
    
    # 按日期分组，每日横截面标准化
    dates = factor_store.index.get_level_values("date").unique()
    
    # 准备数据：按日期分组处理
    reduced_list = []
    pca_info = {
        "n_components": [],
        "explained_variance": [],
        "dates": []
    }
    
    for date in dates:
        date_data = factor_store.xs(date, level=0)
        
        # 移除全NaN列
        date_data = date_data.loc[:, date_data.notna().any()]
        
        if date_data.empty or len(date_data.columns) < 2:
            continue
        
        # 填充缺失值（使用中位数）
        date_data = date_data.fillna(date_data.median())
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(date_data.values)
        
        # 确定主成分数量
        if n_components is None:
            pca_temp = PCA()
            pca_temp.fit(X_scaled)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            n_comp = np.argmax(cumsum_var >= explained_variance_threshold) + 1
            n_comp = min(n_comp, len(date_data.columns), len(date_data) - 1)
        else:
            n_comp = min(n_components, len(date_data.columns), len(date_data) - 1)
        
        if n_comp < 1:
            continue
        
        # 应用PCA
        pca = PCA(n_components=n_comp)
        X_reduced = pca.fit_transform(X_scaled)
        
        # 创建DataFrame
        reduced_df = pd.DataFrame(
            X_reduced,
            index=date_data.index,
            columns=[f"PCA_{i+1}" for i in range(n_comp)]
        )
        reduced_list.append(reduced_df)
        
        # 记录信息
        pca_info["n_components"].append(n_comp)
        pca_info["explained_variance"].append(float(pca.explained_variance_ratio_.sum()))
        pca_info["dates"].append(date)
    
    if not reduced_list:
        return pd.DataFrame(), pca_info
    
    # 合并结果
    reduced_factors = pd.concat(reduced_list).sort_index()
    
    # 汇总信息
    pca_info["avg_n_components"] = float(np.mean(pca_info["n_components"]))
    pca_info["avg_explained_variance"] = float(np.mean(pca_info["explained_variance"]))
    
    return reduced_factors, pca_info


def apply_factor_analysis(factor_store: pd.DataFrame,
                         n_factors: int = 20,
                         cfg: dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    使用因子分析降维
    
    Args:
        factor_store: 因子数据框
        n_factors: 因子数量
        cfg: 配置字典
    
    Returns:
        (reduced_factors, fa_info)
    """
    if not HAS_SKLEARN:
        raise ImportError("sklearn required for Factor Analysis")
    
    dates = factor_store.index.get_level_values("date").unique()
    
    reduced_list = []
    fa_info = {
        "n_factors": [],
        "dates": []
    }
    
    for date in dates:
        date_data = factor_store.xs(date, level=0)
        date_data = date_data.loc[:, date_data.notna().any()]
        
        if date_data.empty or len(date_data.columns) < n_factors:
            continue
        
        # 填充缺失值
        date_data = date_data.fillna(date_data.median())
        
        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(date_data.values)
        
        # 因子分析
        n_fact = min(n_factors, len(date_data.columns), len(date_data) - 1)
        if n_fact < 1:
            continue
        
        fa = FactorAnalysis(n_components=n_fact, random_state=42)
        X_reduced = fa.fit_transform(X_scaled)
        
        # 创建DataFrame
        reduced_df = pd.DataFrame(
            X_reduced,
            index=date_data.index,
            columns=[f"FA_{i+1}" for i in range(n_fact)]
        )
        reduced_list.append(reduced_df)
        
        fa_info["n_factors"].append(n_fact)
        fa_info["dates"].append(date)
    
    if not reduced_list:
        return pd.DataFrame(), fa_info
    
    reduced_factors = pd.concat(reduced_list).sort_index()
    fa_info["avg_n_factors"] = float(np.mean(fa_info["n_factors"]))
    
    return reduced_factors, fa_info


# ------------------------
# 因子质量改进
# ------------------------
def improve_factor_quality(factor_store: pd.DataFrame,
                          prices: pd.DataFrame,
                          cfg: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    改进因子质量：移除低质量因子，保留高IC因子
    
    Args:
        factor_store: 因子数据框
        prices: 价格数据框
        cfg: 配置字典
    
    Returns:
        (improved_factors, quality_info)
    """
    from src.factor_engine import forward_return
    from src.factor_engine import daily_rank_ic
    
    # 计算未来收益
    forward_ret = forward_return(prices, horizon=1)
    
    # 计算每个因子的IC
    factor_ics = {}
    for col in factor_store.columns:
        factor_series = factor_store[col]
        ic_series = daily_rank_ic(factor_series, forward_ret)
        if len(ic_series) > 0:
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            factor_ics[col] = {
                "mean_ic": float(mean_ic),
                "icir": float(icir),
                "valid_days": len(ic_series)
            }
    
    # 根据配置筛选（注意：cfg已经是improve_quality的配置字典）
    min_ic = cfg.get("min_ic", 0.0)
    min_icir = cfg.get("min_icir", 0.0)
    min_win_rate = cfg.get("min_win_rate", 0.0)
    
    good_factors = []
    quality_info = {
        "total_factors": len(factor_store.columns),
        "removed": [],
        "kept": []
    }
    
    for col, stats in factor_ics.items():
        # 计算IC胜率
        factor_series = factor_store[col]
        ic_series = daily_rank_ic(factor_series, forward_ret)
        win_rate = (ic_series > 0).sum() / len(ic_series) if len(ic_series) > 0 else 0.0
        
        # 筛选条件
        if (abs(stats["mean_ic"]) >= min_ic and 
            abs(stats["icir"]) >= min_icir and
            win_rate >= min_win_rate):
            good_factors.append(col)
            quality_info["kept"].append({
                "factor": col,
                "mean_ic": stats["mean_ic"],
                "icir": stats["icir"],
                "win_rate": float(win_rate)
            })
        else:
            quality_info["removed"].append({
                "factor": col,
                "mean_ic": stats["mean_ic"],
                "icir": stats["icir"],
                "win_rate": float(win_rate),
                "reason": "low_quality"
            })
    
    improved_factors = factor_store[good_factors] if good_factors else factor_store.iloc[:, 0:0]
    quality_info["n_kept"] = len(good_factors)
    quality_info["n_removed"] = len(quality_info["removed"])
    
    return improved_factors, quality_info


# ------------------------
# 因子组合优化
# ------------------------
def optimize_factor_combination(factor_store: pd.DataFrame,
                               prices: pd.DataFrame,
                               method: str = "ic_weighted",
                               cfg: dict = None) -> Tuple[pd.DataFrame, Dict]:
    """
    优化因子组合：根据IC加权组合因子
    
    Args:
        factor_store: 因子数据框
        prices: 价格数据框
        method: 组合方法 ("ic_weighted", "equal_weight", "icir_weighted")
        cfg: 配置字典
    
    Returns:
        (combined_factors, combination_info)
    """
    from src.factor_engine import forward_return, daily_rank_ic
    
    forward_ret = forward_return(prices, horizon=1)
    
    # 计算每个因子的IC
    factor_ics = {}
    for col in factor_store.columns:
        factor_series = factor_store[col]
        ic_series = daily_rank_ic(factor_series, forward_ret)
        if len(ic_series) > 0:
            mean_ic = ic_series.mean()
            std_ic = ic_series.std()
            icir = mean_ic / std_ic if std_ic > 0 else 0.0
            factor_ics[col] = {
                "mean_ic": float(mean_ic),
                "icir": float(icir)
            }
    
    # 按日期组合
    dates = factor_store.index.get_level_values("date").unique()
    combined_list = []
    combination_info = {
        "method": method,
        "n_factors": len(factor_store.columns),
        "weights": {}
    }
    
    for date in dates:
        date_data = factor_store.xs(date, level=0)
        
        if date_data.empty:
            continue
        
        # 计算权重
        if method == "ic_weighted":
            weights = {col: abs(factor_ics.get(col, {}).get("mean_ic", 0.0)) 
                      for col in date_data.columns}
        elif method == "icir_weighted":
            weights = {col: abs(factor_ics.get(col, {}).get("icir", 0.0)) 
                      for col in date_data.columns}
        else:  # equal_weight
            weights = {col: 1.0 for col in date_data.columns}
        
        # 归一化权重
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(weights) for k in weights.keys()}
        
        # 加权组合
        combined = pd.Series(0.0, index=date_data.index)
        for col, weight in weights.items():
            if col in date_data.columns:
                combined += date_data[col] * weight
        
        combined_list.append(combined.to_frame(name="COMBINED"))
        
        # 记录权重（只记录一次）
        if date == dates[0]:
            combination_info["weights"] = weights
    
    if not combined_list:
        return pd.DataFrame(), combination_info
    
    combined_factors = pd.concat(combined_list).sort_index()
    return combined_factors, combination_info


# ------------------------
# 主函数
# ------------------------
def enhance_factors(factor_store: pd.DataFrame,
                   prices: pd.DataFrame,
                   cfg: dict) -> Tuple[pd.DataFrame, Dict]:
    """
    综合因子增强流程
    
    Returns:
        (enhanced_factors, enhancement_info)
    """
    enhancement_cfg = cfg.get("factor_enhancement", {})
    enabled = enhancement_cfg.get("enabled", False)
    
    if not enabled:
        return factor_store, {"enabled": False}
    
    info = {"steps": []}
    current_factors = factor_store.copy()
    
    # 步骤1：改进因子质量
    if enhancement_cfg.get("improve_quality", {}).get("enabled", False):
        print("[Enhancement] Step 1: Improving factor quality...")
        current_factors, quality_info = improve_factor_quality(
            current_factors, prices, enhancement_cfg.get("improve_quality", {})
        )
        info["steps"].append({"step": "improve_quality", **quality_info})
        print(f"  Kept {quality_info['n_kept']} factors, removed {quality_info['n_removed']}")
    
    # 步骤2：PCA降维
    if enhancement_cfg.get("pca_reduction", {}).get("enabled", False):
        print("[Enhancement] Step 2: Applying PCA reduction...")
        pca_cfg = enhancement_cfg.get("pca_reduction", {})
        reduced_factors, pca_info = apply_pca_reduction(
            current_factors,
            n_components=pca_cfg.get("n_components"),
            explained_variance_threshold=pca_cfg.get("explained_variance_threshold", 0.95),
            cfg=cfg
        )
        if not reduced_factors.empty:
            current_factors = reduced_factors
            info["steps"].append({"step": "pca_reduction", **pca_info})
            print(f"  Reduced to {pca_info['avg_n_components']:.1f} components on average")
    
    # 步骤3：因子组合
    if enhancement_cfg.get("factor_combination", {}).get("enabled", False):
        print("[Enhancement] Step 3: Optimizing factor combination...")
        comb_cfg = enhancement_cfg.get("factor_combination", {})
        combined_factors, comb_info = optimize_factor_combination(
            current_factors,
            prices,
            method=comb_cfg.get("method", "ic_weighted"),
            cfg=cfg
        )
        if not combined_factors.empty:
            # 将组合因子添加到原始因子
            current_factors = current_factors.join(combined_factors, how="outer")
            info["steps"].append(comb_info)
            print(f"  Added combined factor using {comb_info['method']} method")
    
    info["final_n_factors"] = len(current_factors.columns)
    info["original_n_factors"] = len(factor_store.columns)
    
    return current_factors, info

