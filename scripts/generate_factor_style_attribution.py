#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Factor Style Attribution Data

This script computes the dominant Barra style for each factor (including Alpha factors)
by computing their average exposure to style factors over a rolling window.

Output: data/factors/factor_style_attribution.parquet
Schema: factor | dominant_style | style_exposure_{style} columns | window_start | window_end | n_days
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.path import DATA_FACTORS_DIR, ROOT_DIR, SETTINGS_FILE, get_path
from src.factor_engine import read_prices, load_settings as load_factor_settings
from src.barra_style_mapper import BarraStyleMapper


def generate_factor_style_attribution(window_days: int = 60):
    """
    Generate factor style attribution using Barra-style classification.
    
    Args:
        window_days: Number of trading days to use for averaging exposures (default 60)
    """
    print("\n" + "=" * 60)
    print(f"生成因子风格归因数据 (窗口: {window_days} 天)...")
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
        
        # Step 2: For each factor, compute average style exposures across the window
        all_factor_names = factor_store.columns.tolist()
        print(f"\n📊 计算 {len(all_factor_names)} 个因子的风格归因...")
        
        results = []
        style_names = mapper.CANONICAL_STYLES
        
        for idx, factor_name in enumerate(all_factor_names, 1):
            if idx % 50 == 0 or idx == 1 or idx == len(all_factor_names):
                print(f"  进度: {idx}/{len(all_factor_names)} - {factor_name}")
            
            # Collect exposures for this factor across all dates in window
            factor_exposures_by_date = []
            
            for date in window_dates:
                try:
                    # Get factor values for this date
                    date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date]
                    if isinstance(date_factors.index, pd.MultiIndex):
                        date_factors = date_factors.reset_index(level='date', drop=True)
                    
                    if factor_name not in date_factors.columns:
                        continue
                    
                    factor_values = date_factors[factor_name].dropna()
                    if len(factor_values) < 10:  # Need at least 10 observations
                        continue
                    
                    # Compute style factor scores for this date
                    style_scores = mapper.compute_style_factor_scores(
                        date_factors,
                        date,
                        use_fixed_structure=use_fixed_structure,
                        fixed_structure=fixed_structure if use_fixed_structure else None
                    )
                    
                    if style_scores.empty:
                        continue
                    
                    # Compute exposures
                    exposures = mapper.compute_factor_style_exposure(
                        factor_name,
                        factor_values,
                        style_scores,
                        method='regression'
                    )
                    
                    factor_exposures_by_date.append(exposures)
                    
                except Exception as e:
                    warnings.warn(f"Failed to compute exposure for {factor_name} on {date}: {e}")
                    continue
            
            if len(factor_exposures_by_date) == 0:
                # No valid exposures, assign to Custom
                avg_exposures = {style: 0.0 for style in style_names}
                dominant_style = "Custom"
            else:
                # Average exposures across dates
                exposure_df = pd.DataFrame(factor_exposures_by_date)
                avg_exposures = exposure_df.mean().to_dict()
                
                # Assign dominant style
                dominant_style = mapper.assign_dominant_style(avg_exposures, min_abs_exposure=0.05)
            
            # Build result row
            result_row = {
                'factor': factor_name,
                'dominant_style': dominant_style,
                'window_start': window_start,
                'window_end': window_end,
                'n_days': len(window_dates)
            }
            
            # Add style exposures
            for style in style_names:
                result_row[f'style_exposure_{style}'] = avg_exposures.get(style, 0.0)
            
            results.append(result_row)
        
        # Create DataFrame
        attribution_df = pd.DataFrame(results)
        
        # Save to parquet
        output_path = DATA_FACTORS_DIR / "factor_style_attribution.parquet"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        attribution_df.to_parquet(output_path, index=False)
        
        print(f"\n✅ 因子风格归因数据已保存: {output_path}")
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
        
        # Store feature names for each bucket (we need to extract them from the computation)
        # For simplicity, we'll store the bucket factors that were used
        pca_feature_names = {}
        taxonomy = mapper.barra_model.factor_taxonomy
        
        for bucket_name in taxonomy.keys():
            bucket_factor_names = []
            for pattern in taxonomy[bucket_name]:
                bucket_factor_names.extend([f for f in reference_factors.columns if pattern in f and not f.startswith('Alpha')])
            if bucket_name in pca_models and len(bucket_factor_names) > 0:
                pca_feature_names[bucket_name] = bucket_factor_names
        
        fixed_structure = {
            'pca_models': pca_models,
            'pca_feature_names': pca_feature_names,
            'style_factor_order': style_scores.columns.tolist(),
            'ortho_template': style_scores  # Store as template for reference
        }
        
        return fixed_structure
        
    except Exception as e:
        warnings.warn(f"Failed to estimate fixed structure: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate factor style attribution data")
    parser.add_argument("--window", type=int, default=60, help="Rolling window size in trading days (default: 60)")
    
    args = parser.parse_args()
    
    success = generate_factor_style_attribution(window_days=args.window)
    sys.exit(0 if success else 1)

