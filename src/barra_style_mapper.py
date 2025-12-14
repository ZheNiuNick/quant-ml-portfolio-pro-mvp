#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BarraStyleMapper: Single Source of Truth for Factor-to-Style Classification

This module provides unified factor classification logic used across:
- Multi-Factor Risk Exposure (Barra-style)
- Factor Clusters Analysis
- Alpha style attribution

Key features:
1. Canonical Barra-style categories
2. Raw factor to style bucket mapping (for non-Alpha factors)
3. Style factor score computation (winsorize + zscore + PCA + orthogonalization)
4. Factor style exposure computation (for Alpha factors via regression)
5. Dominant style assignment based on max absolute exposure
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Import BarraRiskModel for reusing normalization and PCA logic
from src.barra_risk_model import BarraRiskModel


class BarraStyleMapper:
    """
    Unified mapper for factor-to-style classification using Barra methodology.
    
    This class provides:
    - Canonical style definitions
    - Raw factor classification (for non-Alpha factors)
    - Style factor score computation
    - Factor style exposure computation (for Alpha factors)
    - Dominant style assignment
    """
    
    CANONICAL_STYLES = [
        "Price/Level",
        "Trend",
        "Momentum",
        "Volatility",
        "Liquidity",
        "Quality/Stability",
        "Custom"
    ]
    
    def __init__(self, winsorize_percentile: float = 0.025, 
                 pca_variance_threshold: float = 0.5):
        """
        Initialize BarraStyleMapper
        
        Args:
            winsorize_percentile: Percentile for winsorization (default 0.025 = 2.5%)
            pca_variance_threshold: Minimum variance explained by PC1 (default 0.5 = 50%)
        """
        self.winsorize_percentile = winsorize_percentile
        self.pca_variance_threshold = pca_variance_threshold
        
        # Use BarraRiskModel for normalization and PCA logic
        self.barra_model = BarraRiskModel(
            winsorize_percentile=winsorize_percentile,
            pca_variance_threshold=pca_variance_threshold
        )
        
        # Cache for style factor scores (computed once per date)
        self._style_factor_cache = {}
        self._fixed_structure = None  # Store fixed PCA/orthogonalization structure
    
    def get_raw_factor_to_style_bucket(self, factor_name: str) -> Optional[str]:
        """
        Classify a raw factor into a style bucket using pattern matching.
        
        This is used for initial classification during style construction.
        Note: Alpha factors are treated as raw factors and can match patterns if applicable.
        If no pattern matches, the factor goes to 'Unclassified' bucket.
        
        Args:
            factor_name: Name of the factor (including Alpha factors)
            
        Returns:
            Style bucket name, or None if not matched (will go to Unclassified)
        """
        taxonomy = self.barra_model.factor_taxonomy
        
        # Pattern matching (same logic as BarraRiskModel.classify_factors)
        # Alpha factors are treated the same as other raw factors
        for bucket_name, patterns in taxonomy.items():
            for pattern in patterns:
                if pattern in factor_name:
                    return bucket_name
        
        # Not matched - return None (caller should treat as Unclassified)
        return None
    
    def compute_style_factor_scores(self, date_factors_df: pd.DataFrame,
                                   date: pd.Timestamp,
                                   use_fixed_structure: bool = False,
                                   fixed_structure: Optional[Dict] = None) -> pd.DataFrame:
        """
        Compute style factor scores for a given date using Barra methodology.
        
        Steps:
        1. Classify raw factors into style buckets
        2. Winsorize and z-score normalize within each bucket
        3. Apply PCA within each bucket (or use fixed structure)
        4. Orthogonalize style factors using Gram-Schmidt
        
        Args:
            date_factors_df: DataFrame of raw factor values (stocks x factors) for a single date
            date: Date timestamp
            use_fixed_structure: If True, use pre-computed PCA/orthogonalization structure
            fixed_structure: Pre-computed structure dict with keys:
                - pca_models: Dict[bucket_name, PCA_model]
                - pca_feature_names: Dict[bucket_name, List[feature_names]]
                - style_factor_order: List of style names after orthogonalization
                - ortho_template: DataFrame of reference orthogonalized factors (for order)
        
        Returns:
            DataFrame of style factor scores (stocks x style_factors)
        """
        if date_factors_df.empty:
            return pd.DataFrame()
        
        # Step 1: Classify factors into buckets (include ALL factors including Alpha)
        taxonomy = self.barra_model.factor_taxonomy
        bucket_factors = {}
        
        for bucket_name, patterns in taxonomy.items():
            bucket_factor_names = []
            for pattern in patterns:
                # Include ALL factors matching pattern (Alpha factors are raw factors too)
                bucket_factor_names.extend([f for f in date_factors_df.columns if pattern in f])
            if len(bucket_factor_names) > 0:
                bucket_factors[bucket_name] = bucket_factor_names
        
        # Step 2: Normalize factors within each bucket
        normalized_bucket_factors = {}
        
        for bucket_name, factor_names in bucket_factors.items():
            bucket_data = date_factors_df[factor_names].copy()
            
            # Normalize each factor
            normalized_factors = []
            valid_factor_names = []
            
            for factor_name in factor_names:
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
                
                # Check variance
                std_val = factor_values.std()
                if isinstance(std_val, pd.Series):
                    std_val = std_val.iloc[0] if len(std_val) > 0 else np.nan
                std_val = float(std_val) if not pd.isna(std_val) else np.nan
                if pd.isna(std_val) or std_val < 1e-8:
                    continue
                
                # Winsorize and normalize (cross-sectional across stocks for this date)
                winsorized = self.barra_model.winsorize_cross_sectional(factor_values, date)
                normalized = self.barra_model.zscore_normalize_cross_sectional(winsorized)
                normalized_factors.append(normalized)
                valid_factor_names.append(factor_name)
            
            if len(normalized_factors) > 0:
                bucket_df = pd.concat(normalized_factors, axis=1)
                bucket_df.columns = valid_factor_names
                normalized_bucket_factors[bucket_name] = bucket_df
        
        # Step 3: Apply PCA within each bucket
        style_factors_dict = {}
        
        if use_fixed_structure and fixed_structure is not None:
            # Use fixed PCA structure
            pca_models = fixed_structure.get('pca_models', {})
            pca_feature_names = fixed_structure.get('pca_feature_names', {})
            
            for bucket_name, bucket_df in normalized_bucket_factors.items():
                if bucket_name in pca_models and bucket_name in pca_feature_names:
                    try:
                        pca_model = pca_models[bucket_name]
                        expected_features = pca_feature_names[bucket_name]
                        available_features = [f for f in expected_features if f in bucket_df.columns]
                        
                        if len(available_features) == len(expected_features):
                            bucket_data_aligned = bucket_df[available_features]
                            bucket_data_clean = bucket_data_aligned.dropna()
                            if len(bucket_data_clean) > 0:
                                pc1_values = pca_model.transform(bucket_data_clean.values)
                                pc1_series = pd.Series(pc1_values.flatten(), index=bucket_data_clean.index)
                                if not pc1_series.isna().all():
                                    style_factors_dict[bucket_name] = pc1_series
                    except Exception as e:
                        warnings.warn(f"Failed to apply fixed PCA for {bucket_name}: {e}")
                        # Fall back to computing on-the-fly
                        try:
                            pc1_series, _ = self.barra_model.reduce_dimension_within_bucket(bucket_df, bucket_name)
                            if not pc1_series.empty:
                                style_factors_dict[bucket_name] = pc1_series
                        except:
                            pass
        else:
            # Compute PCA on-the-fly
            for bucket_name, bucket_df in normalized_bucket_factors.items():
                try:
                    pc1_series, _ = self.barra_model.reduce_dimension_within_bucket(bucket_df, bucket_name)
                    if not pc1_series.empty:
                        style_factors_dict[bucket_name] = pc1_series
                except Exception as e:
                    warnings.warn(f"Failed to compute PCA for {bucket_name}: {e}")
                    continue
        
        if len(style_factors_dict) == 0:
            return pd.DataFrame()
        
        # Step 4: Orthogonalize style factors
        style_factors_df = pd.DataFrame(style_factors_dict)
        style_factors_df = style_factors_df.dropna()
        
        if style_factors_df.empty:
            return pd.DataFrame()
        
        if use_fixed_structure and fixed_structure is not None:
            # Use fixed orthogonalization order (Gram-Schmidt with same order)
            style_factor_order = fixed_structure.get('style_factor_order', style_factors_df.columns.tolist())
            # Ensure we have all required style factors
            available_styles = [s for s in style_factor_order if s in style_factors_df.columns]
            if len(available_styles) > 0:
                style_factors_df_reordered = style_factors_df[available_styles].copy()
                style_factors_ortho = self.barra_model.orthogonalize_style_factors(style_factors_df_reordered)
            else:
                style_factors_ortho = self.barra_model.orthogonalize_style_factors(style_factors_df)
        else:
            style_factors_ortho = self.barra_model.orthogonalize_style_factors(style_factors_df)
        
        return style_factors_ortho
    
    def compute_factor_style_exposure(self, factor_name: str,
                                     factor_values: pd.Series,
                                     style_factor_scores: pd.DataFrame,
                                     method: str = 'regression') -> Dict[str, float]:
        """
        Compute exposure of a factor (including Alpha) to each style factor.
        
        This is used to classify Alpha factors into dominant styles.
        
        Args:
            factor_name: Name of the factor
            factor_values: Series of factor values (stocks x 1) for a single date
            style_factor_scores: DataFrame of style factor scores (stocks x styles)
            method: Method to compute exposure ('regression' or 'correlation')
        
        Returns:
            Dictionary mapping style names to exposure coefficients
        """
        # Align data
        common_idx = factor_values.index.intersection(style_factor_scores.index)
        if len(common_idx) == 0:
            return {style: 0.0 for style in style_factor_scores.columns}
        
        y = factor_values.loc[common_idx].dropna()
        X = style_factor_scores.loc[y.index].dropna()
        
        # Further align
        common_idx_final = y.index.intersection(X.index)
        if len(common_idx_final) < 5:  # Need at least 5 observations
            return {style: 0.0 for style in style_factor_scores.columns}
        
        y = y.loc[common_idx_final]
        X = X.loc[common_idx_final]
        
        exposures = {}
        
        if method == 'regression':
            # OLS regression: factor = X * beta + epsilon
            try:
                from sklearn.linear_model import LinearRegression
                reg = LinearRegression(fit_intercept=True)
                reg.fit(X.values, y.values)
                
                for i, style in enumerate(X.columns):
                    exposures[style] = float(reg.coef_[i])
            except Exception as e:
                warnings.warn(f"Regression failed for {factor_name}: {e}, falling back to correlation")
                method = 'correlation'
        
        if method == 'correlation':
            # Correlation method as fallback
            for style in X.columns:
                try:
                    corr = y.corr(X[style])
                    exposures[style] = float(corr) if not pd.isna(corr) else 0.0
                except:
                    exposures[style] = 0.0
        
        # Ensure all styles are present
        for style in style_factor_scores.columns:
            if style not in exposures:
                exposures[style] = 0.0
        
        return exposures
    
    def assign_dominant_style(self, exposure_dict: Dict[str, float],
                             min_abs_exposure: float = 0.05) -> str:
        """
        Assign dominant style based on maximum absolute exposure.
        
        CRITICAL: Uses abs(exposure) to find dominant style, as negative correlations
        are still meaningful (reverse factor).
        
        Args:
            exposure_dict: Dictionary mapping style names to exposure coefficients
            min_abs_exposure: Minimum absolute exposure threshold (default 0.05)
                            If max abs exposure < threshold, assign to Custom
        
        Returns:
            Dominant style name
        """
        if not exposure_dict:
            return "Custom"
        
        # Compute absolute exposures
        abs_exposures = {style: abs(exposure) for style, exposure in exposure_dict.items()}
        
        # Find maximum
        if not abs_exposures or max(abs_exposures.values()) < min_abs_exposure:
            return "Custom"
        
        dominant_style = max(abs_exposures.items(), key=lambda x: x[1])[0]
        return dominant_style

