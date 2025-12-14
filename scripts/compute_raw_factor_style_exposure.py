#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compute Raw Factor Style Exposure for Factor Distribution Visualization

This script computes each raw factor's exposure to constructed Barra style factors
using covariance. This is used ONLY for visualization/diagnostics in Factor Distribution.

Key principle:
- Raw factors (including Alpha1-Alpha101) are inputs
- Style factors are constructed from raw factors
- Raw factor exposure to styles is computed via: exposure_k = cov(factor_value, style_score_k)
- This is for coloring/diagnostics ONLY, not classification truth
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import argparse
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.path import DATA_FACTORS_DIR, ROOT_DIR, SETTINGS_FILE, OUTPUT_DIR, get_path
from src.factor_engine import read_prices, load_settings as load_factor_settings
from src.barra_style_mapper import BarraStyleMapper


def compute_raw_factor_style_exposure(window_days: int = 60):
    """
    Compute raw factor exposure to Barra style factors using covariance.
    
    This is used for Factor Distribution visualization (color coding by dominant style).
    
    Args:
        window_days: Number of trading days to use for computing exposure (default 60)
    """
    print("\n" + "=" * 60)
    print(f"计算原始因子对风格因子的暴露度 (窗口: {window_days} 天)...")
    print("=" * 60)
    
    try:
        # Load factor data
        factor_cfg = load_factor_settings(str(SETTINGS_FILE))
        factor_store_rel_path = factor_cfg["paths"].get("factors_store", "data/factors/factor_store.parquet")
        if Path(factor_store_rel_path).is_absolute():
            factor_store_path = Path(factor_store_rel_path)
        else:
            factor_store_path = (ROOT_DIR / factor_store_rel_path).resolve()
        
        if not factor_store_path.exists():
            print(f"❌ 文件不存在: {factor_store_path}")
            return False
        
        print(f"📖 读取因子数据: {factor_store_path}")
        factor_store = pd.read_parquet(factor_store_path)
        
        if not isinstance(factor_store.index, pd.MultiIndex):
            if "date" in factor_store.columns and "ticker" in factor_store.columns:
                factor_store["date"] = pd.to_datetime(factor_store["date"])
                factor_store = factor_store.set_index(["date", "ticker"]).sort_index()
        
        # Get available dates
        available_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
        
        if len(available_dates) < window_days:
            print(f"⚠️  可用日期 ({len(available_dates)}) 少于窗口大小 ({window_days})，使用全部日期")
            window_dates = available_dates
        else:
            # Use most recent window_days
            window_dates = available_dates[-window_days:]
        
        window_start = window_dates[0]
        window_end = window_dates[-1]
        print(f"📅 使用日期窗口: {window_start.date()} 到 {window_end.date()} ({len(window_dates)} 天)")
        
        # Initialize mapper
        mapper = BarraStyleMapper()
        
        # Step 1: Estimate fixed PCA/orthogonalization structure from the window
        print("\n🔧 估计固定 PCA 和正交化结构...")
        fixed_structure = _estimate_fixed_structure(mapper, factor_store, window_dates)
        
        if fixed_structure is None:
            print("⚠️  无法估计固定结构，将使用每日计算")
            use_fixed_structure = False
        else:
            print(f"✅ 固定结构估计完成: {len(fixed_structure.get('style_factor_order', []))} 个风格因子")
            use_fixed_structure = True
        
        # Step 2: For each date, compute style factor scores
        print("\n📊 计算风格因子得分...")
        style_scores_by_date = {}
        
        for date in window_dates:
            try:
                date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date]
                if isinstance(date_factors.index, pd.MultiIndex):
                    date_factors = date_factors.reset_index(level='date', drop=True)
                
                if date_factors.empty:
                    continue
                
                # Compute style factor scores for this date
                style_scores = mapper.compute_style_factor_scores(
                    date_factors,
                    date,
                    use_fixed_structure=use_fixed_structure,
                    fixed_structure=fixed_structure if use_fixed_structure else None
                )
                
                if not style_scores.empty:
                    style_scores_by_date[date] = style_scores
                    
            except Exception as e:
                warnings.warn(f"Failed to compute style scores for {date}: {e}")
                continue
        
        if len(style_scores_by_date) == 0:
            print("❌ 无法计算任何日期的风格因子得分")
            return False
        
        print(f"✅ 成功计算 {len(style_scores_by_date)} 个日期的风格因子得分")
        
        # Step 3: For each raw factor, compute exposure to each style factor using covariance
        all_factor_names = factor_store.columns.tolist()
        print(f"\n📊 计算 {len(all_factor_names)} 个原始因子的风格暴露度...")
        
        style_names = mapper.CANONICAL_STYLES
        results = []
        
        for idx, factor_name in enumerate(all_factor_names, 1):
            if idx % 50 == 0 or idx == 1 or idx == len(all_factor_names):
                print(f"  进度: {idx}/{len(all_factor_names)} - {factor_name}")
            
            # Collect factor values and style scores across all dates
            factor_values_list = []
            style_scores_list = []
            valid_dates = []
            
            for date in window_dates:
                if date not in style_scores_by_date:
                    continue
                
                try:
                    # Get factor values for this date
                    date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date]
                    if isinstance(date_factors.index, pd.MultiIndex):
                        date_factors = date_factors.reset_index(level='date', drop=True)
                    
                    if factor_name not in date_factors.columns:
                        continue
                    
                    factor_values = date_factors[factor_name].dropna()
                    style_scores = style_scores_by_date[date]
                    
                    # Align by common tickers
                    common_tickers = factor_values.index.intersection(style_scores.index)
                    if len(common_tickers) < 10:  # Need at least 10 observations
                        continue
                    
                    factor_values_aligned = factor_values.loc[common_tickers]
                    style_scores_aligned = style_scores.loc[common_tickers]
                    
                    factor_values_list.append(factor_values_aligned)
                    style_scores_list.append(style_scores_aligned)
                    valid_dates.append(date)
                    
                except Exception as e:
                    warnings.warn(f"Failed to align data for {factor_name} on {date}: {e}")
                    continue
            
            if len(factor_values_list) == 0:
                # No valid data, assign to Custom with zero exposures
                exposures = {style: 0.0 for style in style_names}
                dominant_style = "Custom"
            else:
                # Compute covariance-based exposure for each style
                exposures = {}
                for style in style_names:
                    style_exposures = []
                    
                    for factor_vals, style_sc in zip(factor_values_list, style_scores_list):
                        if style not in style_sc.columns:
                            continue
                        
                        # Compute covariance (exposure) for this date
                        try:
                            # Normalize both series (z-score) before computing covariance
                            factor_z = (factor_vals - factor_vals.mean()) / (factor_vals.std() + 1e-8)
                            style_z = (style_sc[style] - style_sc[style].mean()) / (style_sc[style].std() + 1e-8)
                            
                            # Covariance = correlation when both are normalized
                            cov = factor_z.corr(style_z)
                            if not pd.isna(cov):
                                style_exposures.append(cov)
                        except:
                            pass
                    
                    # Average exposure across dates
                    if len(style_exposures) > 0:
                        exposures[style] = float(np.mean(style_exposures))
                    else:
                        exposures[style] = 0.0
                
                # Assign dominant style based on max absolute exposure
                dominant_style = mapper.assign_dominant_style(exposures, min_abs_exposure=0.05)
            
            # Build result row
            result_row = {
                'factor': factor_name,
                'dominant_style': dominant_style,
                'window_start': window_start,
                'window_end': window_end,
                'n_days': len(valid_dates)
            }
            
            # Add style exposures
            for style in style_names:
                result_row[f'style_exposure_{style}'] = exposures.get(style, 0.0)
            
            results.append(result_row)
        
        # Create DataFrame
        attribution_df = pd.DataFrame(results)
        
        # Save to parquet
        output_path = DATA_FACTORS_DIR / "raw_factor_style_exposure.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        attribution_df.to_parquet(output_path, index=False)
        
        print(f"\n✅ 原始因子风格暴露度数据已保存: {output_path}")
        print(f"\n📊 风格分布:")
        style_counts = attribution_df['dominant_style'].value_counts()
        for style, count in style_counts.items():
            print(f"  {style}: {count} 个因子")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"\n❌ 错误: {e}")
        print(traceback.format_exc())
        return False


def _estimate_fixed_structure(mapper: BarraStyleMapper, 
                              factor_store: pd.DataFrame,
                              window_dates: pd.DatetimeIndex) -> Optional[Dict]:
    """
    Estimate fixed PCA and orthogonalization structure from a date window.
    
    Returns:
        Dictionary with keys: pca_models, pca_feature_names, style_factor_order
    """
    try:
        # Use the most recent date in the window as reference
        reference_date = window_dates[-1]
        reference_factors = factor_store.loc[factor_store.index.get_level_values(0) == reference_date]
        if isinstance(reference_factors.index, pd.MultiIndex):
            reference_factors = reference_factors.reset_index(level='date', drop=True)
        
        # Compute style factor scores for reference date
        style_scores = mapper.compute_style_factor_scores(reference_factors, reference_date, use_fixed_structure=False)
        
        if style_scores.empty:
            return None
        
        # Get PCA models from barra_model
        pca_models = mapper.barra_model.pca_models.copy()
        
        # Store feature names for each bucket
        pca_feature_names = {}
        taxonomy = mapper.barra_model.factor_taxonomy
        
        for bucket_name in taxonomy.keys():
            bucket_factor_names = []
            for pattern in taxonomy[bucket_name]:
                # Include ALL factors matching pattern (including Alpha if pattern matches)
                bucket_factor_names.extend([f for f in reference_factors.columns if pattern in f])
            if bucket_name in pca_models and len(bucket_factor_names) > 0:
                pca_feature_names[bucket_name] = bucket_factor_names
        
        fixed_structure = {
            'pca_models': pca_models,
            'pca_feature_names': pca_feature_names,
            'style_factor_order': style_scores.columns.tolist(),
            'ortho_template': style_scores
        }
        
        return fixed_structure
        
    except Exception as e:
        warnings.warn(f"Failed to estimate fixed structure: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute raw factor style exposure for visualization")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size in trading days (default: 60)")
    
    args = parser.parse_args()
    
    success = compute_raw_factor_style_exposure(window_days=args.window)
    sys.exit(0 if success else 1)

