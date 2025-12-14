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
            unclassified_count = len(classified['Unclassified'])
            print(f"  Unclassified: {unclassified_count} factors (will be included in style construction via Custom bucket)")
            # Add Unclassified factors to Custom bucket for style construction
            if 'Custom' not in classified:
                classified['Custom'] = []
            classified['Custom'].extend(classified['Unclassified'])
        
        # Get dates to process (last 30 trading days)
        available_dates = pd.to_datetime(factor_store.index.get_level_values(0).unique()).sort_values()
        dates_to_process = available_dates[-30:]
        
        # Use rolling window (60-252 days) to estimate fixed PCA and orthogonalization structure
        # This ensures factor definitions are stable over time (not recomputed daily)
        rolling_window_days = 126  # Use 126 days (approx 6 months) as default, can be 60-252
        structure_end_date = dates_to_process[0]  # Use date before processing window
        structure_start_date = available_dates[available_dates <= structure_end_date][-rolling_window_days] if len(available_dates[available_dates <= structure_end_date]) >= rolling_window_days else available_dates[0]
        structure_dates = available_dates[(available_dates >= structure_start_date) & (available_dates < structure_end_date)]
        
        print(f"\n📅 处理 {len(dates_to_process)} 个日期...")
        print(f"🔧 使用滚动窗口 ({len(structure_dates)} 天, {structure_start_date.date()} 到 {structure_end_date.date()}) 估计固定PCA和正交化结构")
        
        # Step 0: Estimate fixed PCA and orthogonalization structure from rolling window
        fixed_pca_models = {}  # Store PCA models for each bucket
        fixed_ortho_matrix = None  # Store orthogonalization transformation matrix
        fixed_style_factor_order = None  # Store the order of style factors after orthogonalization
        
        if len(structure_dates) >= 20:  # Need at least 20 days to estimate structure
            print("  估计固定因子结构...")
            
            # Collect factor data from structure window (use most recent date as reference)
            reference_date = structure_dates[-1]
            reference_factors = factor_store.loc[factor_store.index.get_level_values(0) == reference_date]
            if isinstance(reference_factors.index, pd.MultiIndex):
                reference_factors = reference_factors.reset_index(level='date', drop=True)
            
            # Estimate PCA structure for each bucket
            normalized_bucket_factors_ref = {}
            fixed_pca_feature_names = {}  # Store feature names for each bucket's PCA model
            
            for bucket_name, factor_patterns in model.factor_taxonomy.items():
                bucket_factor_names = []
                for pattern in factor_patterns:
                    # Include ALL factors matching pattern (Alpha factors are raw factors too)
                    bucket_factor_names.extend([f for f in reference_factors.columns if pattern in f])
                
                if len(bucket_factor_names) == 0:
                    continue
                
                bucket_data_ref = reference_factors[bucket_factor_names].copy()
                
                # Normalize reference factors
                normalized_factors_ref = []
                valid_factor_names_ref = []
                for factor_name in bucket_factor_names:
                    if factor_name not in bucket_data_ref.columns:
                        continue
                    factor_col = bucket_data_ref[factor_name]
                    # Extract as Series (handle both Series and DataFrame cases)
                    if isinstance(factor_col, pd.DataFrame):
                        factor_values = factor_col.iloc[:, 0].dropna()
                    else:
                        factor_values = factor_col.dropna()
                    
                    if not isinstance(factor_values, pd.Series) or len(factor_values) == 0:
                        continue
                    std_val = factor_values.std()
                    if isinstance(std_val, pd.Series):
                        std_val = std_val.iloc[0] if len(std_val) > 0 else np.nan
                    std_val = float(std_val) if not pd.isna(std_val) else np.nan
                    if pd.isna(std_val) or std_val < 1e-8:
                        continue
                    
                    # Winsorize and normalize
                    winsorized = model.winsorize_cross_sectional(factor_values, reference_date)
                    normalized = model.zscore_normalize_cross_sectional(winsorized)
                    normalized_factors_ref.append(normalized)
                    valid_factor_names_ref.append(factor_name)
                
                if len(normalized_factors_ref) > 0:
                    bucket_df_ref = pd.concat(normalized_factors_ref, axis=1)
                    bucket_df_ref.columns = valid_factor_names_ref
                    normalized_bucket_factors_ref[bucket_name] = bucket_df_ref
                    
                    # Estimate PCA for this bucket (store model and feature names)
                    try:
                        pc1_ref, variance_explained_ref = model.reduce_dimension_within_bucket(bucket_df_ref, bucket_name)
                        if bucket_name in model.pca_models:
                            fixed_pca_models[bucket_name] = model.pca_models[bucket_name]
                            fixed_pca_feature_names[bucket_name] = valid_factor_names_ref  # Store feature names separately
                    except Exception as e:
                        warnings.warn(f"Failed to estimate PCA structure for {bucket_name}: {e}")
                        continue
            
            # Estimate orthogonalization structure
            if len(normalized_bucket_factors_ref) > 0:
                style_factors_dict_ref = {}
                for bucket_name, bucket_data_ref in normalized_bucket_factors_ref.items():
                    if bucket_name in fixed_pca_models and bucket_name in fixed_pca_feature_names:
                        try:
                            # Apply fixed PCA model
                            pca_model = fixed_pca_models[bucket_name]
                            expected_features = fixed_pca_feature_names[bucket_name]
                            # Align bucket data with expected features
                            available_features = [f for f in expected_features if f in bucket_data_ref.columns]
                            if len(available_features) == len(expected_features):
                                bucket_data_aligned = bucket_data_ref[available_features]
                                bucket_data_clean = bucket_data_aligned.dropna()
                                if len(bucket_data_clean) > 0:
                                    pc1_values = pca_model.transform(bucket_data_clean.values)
                                    pc1_ref = pd.Series(pc1_values.flatten(), index=bucket_data_clean.index)
                                    if not pc1_ref.isna().all():
                                        style_factors_dict_ref[bucket_name] = pc1_ref
                        except Exception as e:
                            warnings.warn(f"Failed to apply PCA structure for {bucket_name}: {e}")
                            continue
                
                if len(style_factors_dict_ref) > 0:
                    style_factors_df_ref = pd.DataFrame(style_factors_dict_ref)
                    style_factors_df_ref = style_factors_df_ref.dropna()
                    
                    if not style_factors_df_ref.empty:
                        # Estimate orthogonalization using reference data
                        style_factors_ortho_ref = model.orthogonalize_style_factors(style_factors_df_ref)
                        fixed_style_factor_order = list(style_factors_ortho_ref.columns)
                        
                        # Compute orthogonalization transformation matrix
                        # We need to store how to transform from original style factors to orthogonal ones
                        # For simplicity, we store the reference orthogonalized factors as a template
                        # In practice, we'll use the same Gram-Schmidt process with same factor order
                        print(f"  ✅ 固定因子结构估计完成: {len(fixed_style_factor_order)} 个风格因子, {len(fixed_pca_models)} 个PCA模型")
        else:
            warnings.warn(f"Insufficient data for structure estimation ({len(structure_dates)} days), falling back to daily computation")
        
        results = {}
        factor_returns_history = []  # List to store factor returns DataFrames for covariance estimation
        
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
                # Use classified factors (includes Unclassified → Custom mapping)
                normalized_bucket_factors = {}
                for bucket_name, factor_names_list in classified.items():
                    if bucket_name == 'Unclassified':
                        continue  # Already mapped to Custom
                    
                    # Use the actual classified factor names (not pattern matching again)
                    bucket_factor_names = [f for f in factor_names_list if f in date_factors.columns]
                    
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
                
                # Step 2: Apply fixed PCA structure (use pre-estimated PCA loadings from rolling window)
                # This ensures consistent factor definitions across all dates
                style_factors_dict = {}
                
                if len(fixed_pca_models) > 0:
                    # Use fixed PCA models (estimated from rolling window)
                    for bucket_name, bucket_data in normalized_bucket_factors.items():
                        if bucket_name in fixed_pca_models and bucket_name in fixed_pca_feature_names:
                            try:
                                pca_model = fixed_pca_models[bucket_name]
                                expected_features = fixed_pca_feature_names[bucket_name]
                                # Align bucket data with expected features (must match exactly)
                                available_features = [f for f in expected_features if f in bucket_data.columns]
                                if len(available_features) == len(expected_features):
                                    bucket_data_aligned = bucket_data[available_features]
                                    bucket_data_clean = bucket_data_aligned.dropna()
                                    if len(bucket_data_clean) > 0:
                                        # Transform using fixed PCA model
                                        pc1_values = pca_model.transform(bucket_data_clean.values)
                                        pc1 = pd.Series(pc1_values.flatten(), index=bucket_data_clean.index)
                                        if not pc1.isna().all():
                                            style_factors_dict[bucket_name] = pc1
                                # If features don't match, skip this bucket for this date
                            except Exception as e:
                                # Fallback to daily computation if fixed PCA fails
                                try:
                                    pc1, variance_explained = model.reduce_dimension_within_bucket(bucket_data, bucket_name)
                                    if isinstance(pc1, pd.Series) and len(pc1) > 0:
                                        if not pc1.isna().all():
                                            style_factors_dict[bucket_name] = pc1
                                except:
                                    # Skip this bucket if both methods fail
                                    continue
                else:
                    # Fallback: daily PCA computation (if structure estimation failed)
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
                
                # Step 3: Apply fixed orthogonalization structure
                # Use the same factor order and Gram-Schmidt process as estimated in structure window
                # This ensures consistent orthogonalization across all dates
                if fixed_style_factor_order is not None:
                    # Reorder style factors to match fixed order
                    existing_factors = [f for f in fixed_style_factor_order if f in style_factors_df.columns]
                    if len(existing_factors) > 0:
                        style_factors_df_reordered = style_factors_df[existing_factors]
                        # Apply orthogonalization with same factor order (ensures consistency)
                        style_factors_ortho = model.orthogonalize_style_factors(style_factors_df_reordered)
                    else:
                        # No matching factors, use standard orthogonalization
                        style_factors_ortho = model.orthogonalize_style_factors(style_factors_df)
                else:
                    # No fixed structure, use standard orthogonalization
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
                
                # Step 6: Build factor covariance matrix from HISTORICAL factor returns (before estimating current)
                # 
                # CRITICAL: Risk must be driven by factor return covariance, NOT by cross-sectional factor variance
                # 
                # Why factor return covariance?
                # 1. Risk measures uncertainty in future returns, not cross-sectional dispersion of factor values
                # 2. Cross-sectional variance of factor values does not capture how factor returns co-vary over time
                # 3. Two factors can have high cross-sectional variance but low return covariance (low risk)
                # 4. Barra models use: portfolio_variance = b^T Σ_f b where Σ_f = Cov(factor_returns over time)
                # 
                # Formula: Σ_f = Cov(factor_returns) where factor returns are estimated via cross-sectional regression
                # IMPORTANT: Use only factor returns from PREVIOUS dates (no look-ahead bias)
                factor_covariance = None
                if len(factor_returns_history) >= 2:
                    # Combine all historical factor returns (from previous dates only)
                    factor_returns_df_all = pd.concat(factor_returns_history, axis=0)
                    # Estimate covariance matrix using EWMA (exponentially weighted moving average)
                    factor_covariance = model.estimate_factor_covariance(factor_returns_df_all, method='ewma', lambda_param=0.94)
                
                # Step 7: Estimate factor returns using cross-sectional regression
                # Factor returns are estimated using forward returns: r_{i,t+1} = Σ_k (β_{i,k,t} × f_{k,t+1}) + ε_{i,t+1}
                # Note: Factor exposures (β) are from time t, stock returns (r) are from time t+1 (no look-ahead bias)
                factor_returns = None
                if forward_returns is not None and len(forward_returns) > 0:
                    factor_returns = model.estimate_factor_returns(style_factors_ortho, forward_returns)
                    
                    # Store factor returns for NEXT date's covariance estimation (not used for current date)
                    if factor_returns is not None and len(factor_returns) > 0:
                        factor_returns_df = pd.DataFrame([factor_returns], index=[date_obj])
                        factor_returns_history.append(factor_returns_df)
                
                # Step 8: Estimate specific risk (idiosyncratic risk) from regression residuals
                # Specific risk = Var(ε) where ε = r - Σ(β × factor_returns) from regression
                # For portfolio: σ²_specific = Σ_i (w_i² × σ²_i)
                specific_risk_per_stock = None
                specific_risk_series = None
                if forward_returns is not None and factor_returns is not None and len(factor_returns) > 0:
                    # Estimate specific risk using regression residuals
                    common_tickers = style_factors_ortho.index.intersection(forward_returns.index)
                    if len(common_tickers) > 0 and len(common_tickers) > len(style_factors_ortho.columns):
                        X = style_factors_ortho.loc[common_tickers]
                        y = forward_returns.loc[common_tickers]
                        
                        # Align factor returns with style factor columns
                        factor_returns_aligned = factor_returns.reindex(X.columns).fillna(0.0)
                        predicted = (X * factor_returns_aligned).sum(axis=1)
                        residuals = y - predicted
                        
                        # Store residuals as Series for per-stock specific risk
                        specific_risk_series = residuals
                        
                        # Average specific risk (standard deviation of residuals across all stocks)
                        # This is a cross-sectional measure for this date
                        specific_risk_per_stock = residuals.std()
                
                # Step 9: Compute portfolio risk decomposition using correct Barra formula
                # Portfolio variance = b^T Σ_f b + σ²_specific
                # Risk contribution of factor k: RC_k = b_k × (Σ_f b)_k
                # where b = portfolio factor exposures, Σ_f = factor covariance matrix
                risk_contributions = {}
                specific_risk_contribution = 0.0
                total_portfolio_variance = 0.0
                
                if portfolio_exposures and factor_covariance is not None and len(factor_covariance) > 0:
                    # Convert portfolio exposures to vector b (aligned with factor covariance)
                    style_names = list(portfolio_exposures.keys())
                    
                    # Ensure factor covariance columns/index match style names
                    common_factors = [f for f in style_names if f in factor_covariance.index and f in factor_covariance.columns]
                    if len(common_factors) > 0:
                        # Extract sub-matrix for common factors
                        factor_cov_subset = factor_covariance.loc[common_factors, common_factors]
                        portfolio_exposure_vector = np.array([portfolio_exposures[f] for f in common_factors])
                        
                        # Compute portfolio variance: b^T Σ_f b
                        factor_variance = portfolio_exposure_vector @ factor_cov_subset.values @ portfolio_exposure_vector
                        
                        # Compute specific risk variance: σ²_specific = Σ_i (w_i² × σ²_i)
                        # For portfolio, we aggregate using portfolio weights
                        if specific_risk_series is not None and portfolio_weights is not None:
                            # Align specific risk with portfolio weights
                            common_tickers_spec = portfolio_weights.index.intersection(specific_risk_series.index)
                            if len(common_tickers_spec) > 0:
                                # Portfolio-specific variance = Σ_i (w_i² × σ²_i)
                                # where σ²_i is the squared residual for stock i
                                weights_aligned_spec = portfolio_weights.loc[common_tickers_spec]
                                weights_aligned_spec = weights_aligned_spec / weights_aligned_spec.sum()  # Normalize
                                specific_var_aligned = (specific_risk_series.loc[common_tickers_spec] ** 2)
                                portfolio_specific_variance = (weights_aligned_spec ** 2 * specific_var_aligned).sum()
                            else:
                                # Fallback: use average specific variance
                                portfolio_specific_variance = ((portfolio_weights ** 2).sum() * (specific_risk_per_stock ** 2)) if specific_risk_per_stock is not None else 0.0
                        else:
                            portfolio_specific_variance = 0.0
                        
                        # Total portfolio variance
                        total_portfolio_variance = factor_variance + portfolio_specific_variance
                        
                        if total_portfolio_variance > 1e-10:
                            # Compute risk contributions: RC_k = b_k × (Σ_f b)_k
                            # (Σ_f b) is the marginal contribution vector
                            marginal_contributions = factor_cov_subset.values @ portfolio_exposure_vector
                            
                            # Risk contribution for each factor: RC_k = b_k × (Σ_f b)_k
                            # Note: RC_k can be negative if exposure and marginal contribution have opposite signs
                            # This indicates the factor is reducing portfolio risk (hedging effect)
                            # However, the absolute value |RC_k| represents the magnitude of risk impact
                            for idx, factor_name in enumerate(common_factors):
                                rc_k = portfolio_exposure_vector[idx] * marginal_contributions[idx]
                                risk_contributions[factor_name] = float((rc_k / total_portfolio_variance) * 100)
                                
                                # Note: Negative risk contribution means:
                                # - The factor exposure (b_k) and marginal contribution ((Σ_f b)_k) have opposite signs
                                # - This factor is acting as a hedge, reducing portfolio variance
                                # - The magnitude |RC_k| still represents the factor's impact on risk
                            
                            # Specific risk contribution
                            if portfolio_specific_variance > 0:
                                specific_risk_contribution = float((portfolio_specific_variance / total_portfolio_variance) * 100)
                            
                            # Validate: Risk contributions should sum to 100% (approximately)
                            # Note: Individual contributions can be negative if a factor reduces portfolio risk
                            total_rc = sum(risk_contributions.values()) + specific_risk_contribution
                            if abs(total_rc - 100.0) > 0.5:  # Allow 0.5% tolerance for numerical errors
                                warnings.warn(f"[{date_obj}] Risk contributions sum to {total_rc:.2f}% instead of 100%")
                            
                            # Additional validation: verify total variance calculation
                            computed_total_var = sum([rc / 100.0 * total_portfolio_variance for rc in risk_contributions.values()]) + \
                                               (specific_risk_contribution / 100.0 * total_portfolio_variance)
                            if abs(computed_total_var - total_portfolio_variance) > 1e-6:
                                warnings.warn(f"[{date_obj}] Variance decomposition mismatch: {computed_total_var:.6f} vs {total_portfolio_variance:.6f}")
                        else:
                            # Zero or negative variance, set all contributions to 0
                            for factor_name in common_factors:
                                risk_contributions[factor_name] = 0.0
                else:
                    # No factor covariance or exposures available, set all to 0
                    for style_name in style_factors_ortho.columns:
                        risk_contributions[style_name] = 0.0
                
                # REGRESSION GUARD: Ensure we only save style factor names, not raw factor names
                # Valid style factors are the keys in style_factors_ortho.columns (bucket names)
                valid_style_factor_names = set(style_factors_ortho.columns)
                
                # Filter risk_contributions to only include style factors
                filtered_risk_contributions = {
                    name: value for name, value in risk_contributions.items() 
                    if name in valid_style_factor_names
                }
                
                # Warn if any raw factors were found
                raw_factor_patterns = ['Alpha', 'AD', 'OBV', 'ADOSC', 'BB_', 'SMA_', 'EMA_', 'WMA_', 
                                     'DEMA_', 'RSI_', 'CCI_', 'STOCH', 'WILLR_', 'AROON', 'MACD',
                                     'MOM_', 'ROC_', 'MFI_', 'ATR_', 'NATR_', 'BOP', 'CUSTOM_']
                for factor_name in risk_contributions.keys():
                    if factor_name not in valid_style_factor_names:
                        is_raw = any(pattern in factor_name for pattern in raw_factor_patterns)
                        if is_raw:
                            warnings.warn(f"[REGRESSION GUARD] Raw factor '{factor_name}' detected in risk_contributions! "
                                        f"Only style factors should be present. Valid style factors: {valid_style_factor_names}")
                
                # Sort by risk contribution
                sorted_styles = sorted(filtered_risk_contributions.items(), key=lambda x: x[1], reverse=True)
                
                # Convert specific_risk_per_stock to scalar for output
                specific_risk_scalar = float(specific_risk_per_stock) if specific_risk_per_stock is not None else None
                
                results[str(date_obj.date())] = {
                    "factors": [s[0] for s in sorted_styles],  # Style factor names only
                    "exposures": [round(portfolio_exposures.get(s[0], 0.0), 4) for s in sorted_styles],
                    "risk_contributions": [round(s[1], 2) for s in sorted_styles],
                    "specific_risk_contribution": round(specific_risk_contribution, 2),
                    "specific_risk": round(specific_risk_scalar, 4) if specific_risk_scalar is not None else None,
                    "total_portfolio_variance": round(total_portfolio_variance, 6) if total_portfolio_variance > 0 else None,
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

