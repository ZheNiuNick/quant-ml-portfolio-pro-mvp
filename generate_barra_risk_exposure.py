#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Barra-style Risk Exposure Data

This script implements a proper Barra-style multi-factor risk model following:
1. Factor taxonomy and grouping
2. Cross-sectional normalization (winsorize, z-score)
3. PCA-based dimensionality reduction within style buckets
4. Orthogonalization across style factors
5. Alpha separation (alpha factors excluded from risk model)
6. Risk decomposition output
"""

import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import warnings

# 使用统一的路径管理
sys.path.insert(0, str(Path(__file__).parent))

from src.config.path import SETTINGS_FILE, OUTPUT_DIR, DATA_FACTORS_DIR, ROOT_DIR, get_path, OUTPUT_PORTFOLIOS_DIR
from src.factor_engine import read_prices, load_settings as load_factor_settings
from src.barra_risk_model import BarraRiskModel, compute_portfolio_risk_decomposition

SETTINGS = SETTINGS_FILE
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_barra_risk_exposure():
    """Generate Barra-style risk exposure data"""
    print("\n" + "=" * 60)
    print("生成 Barra-style 风险暴露数据...")
    print("=" * 60)
    
    try:
        # 1. Load data
        factor_cfg = load_factor_settings(str(SETTINGS))
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
        
        # Load prices for forward returns
        print("📖 读取价格数据...")
        if "paths" in factor_cfg and "prices_parquet" in factor_cfg["paths"]:
            parquet_path = factor_cfg["paths"]["prices_parquet"]
            factor_cfg["paths"]["prices_parquet"] = str(get_path(parquet_path))
        
        prices = read_prices(factor_cfg)
        if prices is None or len(prices) == 0:
            print("❌ 价格数据不存在或为空")
            return False
        
        # Load portfolio weights
        portfolio_path_rel = factor_cfg["paths"].get("portfolio_path", "outputs/portfolios/weights.parquet")
        if Path(portfolio_path_rel).is_absolute():
            portfolio_path = Path(portfolio_path_rel)
        else:
            portfolio_path = (ROOT_DIR / portfolio_path_rel).resolve()
        
        weights_df = None
        if portfolio_path.exists():
            print(f"📖 读取投资组合权重: {portfolio_path}")
            weights_df = pd.read_parquet(portfolio_path)
            weights_df.index = pd.to_datetime(weights_df.index)
        else:
            print("⚠️  权重文件不存在，将只计算因子暴露度（不计算组合风险分解）")
        
        # 2. Initialize Barra model
        model = BarraRiskModel(winsorize_percentile=0.025, pca_variance_threshold=0.5)
        
        # Classify factors
        factor_names = factor_store.columns.tolist()
        classified = model.classify_factors(factor_names)
        
        print(f"\n📊 因子分类:")
        for bucket, factors in classified.items():
            if bucket != 'Unclassified':
                print(f"  {bucket}: {len(factors)} factors")
        if 'Unclassified' in classified:
            print(f"  Unclassified: {len(classified['Unclassified'])} factors")
        
        # Get dates to process (last 30 trading days)
        available_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
        dates_to_process = available_dates[-30:]
        
        print(f"\n📅 处理 {len(dates_to_process)} 个日期...")
        
        results = {}
        factor_returns_history = []  # For covariance estimation
        
        for i, date_obj in enumerate(dates_to_process, 1):
            if i % 10 == 0:
                print(f"  处理进度: {i}/{len(dates_to_process)}")
            
            try:
                # Get date-specific data
                date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date_obj]
                if isinstance(date_factors.index, pd.MultiIndex):
                    date_factors = date_factors.reset_index(level='date', drop=True)
                
                # Get portfolio weights for this date (or closest previous date)
                portfolio_weights = None
                if weights_df is not None:
                    if date_obj in weights_df.index:
                        weight_date = date_obj
                    else:
                        available_dates_before = weights_df.index[weights_df.index <= date_obj]
                        if len(available_dates_before) > 0:
                            weight_date = available_dates_before.max()
                        else:
                            weight_date = None
                    
                    if weight_date is not None:
                        portfolio_weights_series = weights_df.loc[weight_date].fillna(0.0)
                        portfolio_weights_series = portfolio_weights_series[portfolio_weights_series > 0]
                        if len(portfolio_weights_series) > 0:
                            portfolio_weights = portfolio_weights_series / portfolio_weights_series.sum()
                
                # Step 1: Normalize factors within each bucket
                normalized_bucket_factors = {}
                for bucket_name, factor_patterns in model.factor_taxonomy.items():
                    bucket_factor_names = []
                    for pattern in factor_patterns:
                        bucket_factor_names.extend([f for f in date_factors.columns if pattern in f])
                    
                    if len(bucket_factor_names) == 0:
                        continue
                    
                    bucket_data = date_factors[bucket_factor_names].copy()
                    
                    # Winsorize and normalize each factor
                    normalized_factors = []
                    valid_factor_names = []
                    for factor_name in bucket_factor_names:
                        if factor_name not in bucket_data.columns:
                            continue
                        
                        # Extract factor values as Series
                        factor_col = bucket_data[factor_name]
                        if isinstance(factor_col, pd.DataFrame):
                            # If it's a DataFrame, take the first column
                            factor_values = factor_col.iloc[:, 0].dropna()
                        else:
                            factor_values = factor_col.dropna()
                        
                        if not isinstance(factor_values, pd.Series) or len(factor_values) == 0:
                            continue
                        
                        # Check standard deviation
                        std_val = factor_values.std()
                        if isinstance(std_val, pd.Series):
                            std_val = std_val.iloc[0] if len(std_val) > 0 else np.nan
                        std_val = float(std_val) if not pd.isna(std_val) else np.nan
                        
                        if pd.isna(std_val) or std_val < 1e-8:
                            continue
                        
                        # Winsorize
                        winsorized = model.winsorize_cross_sectional(factor_values, date_obj)
                        # Z-score normalize
                        normalized = model.zscore_normalize_cross_sectional(winsorized)
                        normalized_factors.append(normalized)
                        valid_factor_names.append(factor_name)
                    
                    if len(normalized_factors) > 0:
                        bucket_df = pd.concat(normalized_factors, axis=1)
                        bucket_df.columns = valid_factor_names
                        normalized_bucket_factors[bucket_name] = bucket_df
                
                # Step 2: Reduce dimension within each bucket (PCA)
                style_factors_dict = {}
                for bucket_name, bucket_data in normalized_bucket_factors.items():
                    try:
                        pc1, variance_explained = model.reduce_dimension_within_bucket(bucket_data, bucket_name)
                        if isinstance(pc1, pd.Series) and len(pc1) > 0:
                            if not pc1.isna().all():
                                style_factors_dict[bucket_name] = pc1
                    except Exception as e:
                        warnings.warn(f"Failed to reduce dimension for {bucket_name}: {e}")
                        continue
                
                if len(style_factors_dict) == 0:
                    continue
                
                # Combine into DataFrame
                style_factors_df = pd.DataFrame(style_factors_dict)
                style_factors_df = style_factors_df.dropna()
                
                if style_factors_df.empty:
                    continue
                
                # Step 3: Orthogonalize style factors
                style_factors_ortho = model.orthogonalize_style_factors(style_factors_df)
                
                # Step 4: Compute portfolio exposures
                portfolio_exposures = {}
                if portfolio_weights is not None:
                    for style_name in style_factors_ortho.columns:
                        style_values = style_factors_ortho[style_name]
                        common_tickers = portfolio_weights.index.intersection(style_values.index)
                        if len(common_tickers) > 0:
                            weights_aligned = portfolio_weights.loc[common_tickers] / portfolio_weights.loc[common_tickers].sum()
                            style_aligned = style_values.loc[common_tickers]
                            portfolio_exposures[style_name] = float((weights_aligned * style_aligned).sum())
                else:
                    # If no portfolio weights, use median exposure as portfolio exposure
                    for style_name in style_factors_ortho.columns:
                        portfolio_exposures[style_name] = float(style_factors_ortho[style_name].median())
                
                # Step 5: Estimate factor returns using forward returns (if available)
                # Get forward returns for this date
                forward_returns = None
                if date_obj < dates_to_process[-1]:
                    next_date_idx = list(dates_to_process).index(date_obj) + 1
                    if next_date_idx < len(dates_to_process):
                        next_date = dates_to_process[next_date_idx]
                        next_prices = prices.loc[prices.index.get_level_values(0) == next_date]
                        if isinstance(next_prices.index, pd.MultiIndex):
                            next_prices = next_prices.reset_index(level='date', drop=True)
                        
                        current_prices = prices.loc[prices.index.get_level_values(0) == date_obj]
                        if isinstance(current_prices.index, pd.MultiIndex):
                            current_prices = current_prices.reset_index(level='date', drop=True)
                        
                        if 'Adj Close' in next_prices.columns and 'Adj Close' in current_prices.columns:
                            forward_returns = (next_prices['Adj Close'] / current_prices['Adj Close'] - 1.0)
                            forward_returns = forward_returns.dropna()
                
                # Step 6: Estimate factor returns using cross-sectional regression
                factor_returns = None
                if forward_returns is not None and len(forward_returns) > 0:
                    factor_returns = model.estimate_factor_returns(style_factors_ortho, forward_returns)
                
                # Step 7: Compute risk contribution
                # Use factor covariance if we have factor returns history, otherwise use variance
                style_variances = {}
                total_variance = 0.0
                for style_name in style_factors_ortho.columns:
                    style_values = style_factors_ortho[style_name]
                    if portfolio_weights is not None:
                        # Portfolio-weighted variance
                        common_tickers = portfolio_weights.index.intersection(style_values.index)
                        if len(common_tickers) > 0:
                            weights_aligned = portfolio_weights.loc[common_tickers] / portfolio_weights.loc[common_tickers].sum()
                            style_aligned = style_values.loc[common_tickers]
                            weighted_mean = (weights_aligned * style_aligned).sum()
                            variance = ((weights_aligned * (style_aligned - weighted_mean) ** 2).sum())
                            style_variances[style_name] = float(variance)
                            total_variance += variance
                    else:
                        # Simple variance
                        variance_val = style_values.var()
                        if isinstance(variance_val, pd.Series):
                            variance_val = variance_val.iloc[0] if len(variance_val) > 0 else 0.0
                        variance = float(variance_val) if not (pd.isna(variance_val) or np.isnan(variance_val)) else 0.0
                        style_variances[style_name] = variance
                        total_variance += variance
                
                # Step 8: Estimate specific risk (idiosyncratic risk)
                specific_risk = None
                if forward_returns is not None and factor_returns is not None and len(factor_returns) > 0:
                    # Estimate specific risk from regression residuals
                    common_tickers = style_factors_ortho.index.intersection(forward_returns.index)
                    if len(common_tickers) > 0:
                        X = style_factors_ortho.loc[common_tickers]
                        y = forward_returns.loc[common_tickers]
                        predicted = (X * factor_returns).sum(axis=1)
                        residuals = y - predicted
                        specific_risk = float(residuals.std())
                
                # Step 9: Compute risk contributions
                # Risk contribution = factor variance / (factor variance + specific variance)
                # For now, use factor variance proportion
                risk_contributions = {}
                specific_risk_contribution = 0.0
                
                if total_variance > 0:
                    # Factor risk contributions
                    for style_name, variance in style_variances.items():
                        risk_contributions[style_name] = float((variance / total_variance) * 100)
                    
                    # If we have specific risk, adjust contributions
                    if specific_risk is not None and specific_risk > 0:
                        # Assume equal weight portfolio for specific risk calculation
                        n_stocks = len(style_factors_ortho.index)
                        if n_stocks > 0:
                            specific_variance = (specific_risk ** 2) / n_stocks  # Diversification effect
                            total_variance_with_specific = total_variance + specific_variance
                            
                            # Adjust contributions
                            factor_risk_pct = (total_variance / total_variance_with_specific) * 100
                            specific_risk_contribution = (specific_variance / total_variance_with_specific) * 100
                            
                            # Scale factor contributions
                            for style_name in risk_contributions:
                                risk_contributions[style_name] = risk_contributions[style_name] * (factor_risk_pct / 100)
                else:
                    for style_name in style_factors_ortho.columns:
                        risk_contributions[style_name] = 0.0
                
                # Sort by risk contribution
                sorted_styles = sorted(risk_contributions.items(), key=lambda x: x[1], reverse=True)
                
                results[str(date_obj.date())] = {
                    "factors": [s[0] for s in sorted_styles],
                    "exposures": [round(portfolio_exposures.get(s[0], 0.0), 4) for s in sorted_styles],
                    "risk_contributions": [round(s[1], 2) for s in sorted_styles],
                    "specific_risk_contribution": round(specific_risk_contribution, 2),
                    "specific_risk": round(specific_risk, 4) if specific_risk is not None else None,
                    "pca_variance_explained": {
                        k: float(v) for k, v in model.pca_variance_explained.items()
                    }
                }
                
            except Exception as e:
                print(f"  ⚠️  日期 {date_obj} 处理失败: {e}")
                import traceback
                # Only print first few lines to avoid spam
                tb_lines = traceback.format_exc().split('\n')
                for line in tb_lines[:10]:
                    if line.strip():
                        print(f"    {line}")
                continue
        
        # Save results
        output_file = OUTPUT_DIR / "factor_exposure.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        file_size = output_file.stat().st_size / (1024 * 1024)
        print(f"\n✅ 生成成功: {output_file} ({file_size:.2f} MB, {len(results)} 个日期)")
        
        # Print summary
        if len(results) > 0:
            latest_date = max(results.keys())
            latest_data = results[latest_date]
            print(f"\n📊 最新日期 ({latest_date}) 摘要:")
            print(f"  风格因子数: {len(latest_data['factors'])}")
            print(f"  风险贡献总和: {sum(latest_data['risk_contributions']):.2f}%")
            print(f"  前5个风格因子:")
            for i, (f, e, rc) in enumerate(zip(
                latest_data['factors'][:5],
                latest_data['exposures'][:5],
                latest_data['risk_contributions'][:5]
            ), 1):
                print(f"    {i}. {f}: 暴露度={e:.4f}, 风险贡献={rc:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🚀 开始生成 Barra-style 风险暴露数据...")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    
    if generate_barra_risk_exposure():
        print("\n✅ 完成！")
    else:
        print("\n❌ 生成失败")
        sys.exit(1)

