#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Barra-style Multi-Factor Risk Model

This module implements a proper Barra-style risk model following industry best practices:
1. Factor taxonomy: Classify raw factors into style buckets
2. Cross-sectional normalization: Winsorize, z-score, neutralize
3. Dimensionality reduction: PCA within each style bucket
4. Orthogonalization: Ensure style factors are approximately orthogonal
5. Alpha separation: Alpha signals excluded from risk factors
6. Risk model estimation: Factor returns, covariance, specific risk
7. Risk decomposition: Factor exposure, risk contribution, specific risk

Design principles:
- No raw factors appear directly in risk exposure
- Avoid multicollinearity through PCA and orthogonalization
- Stable, explainable methods suitable for professional quant interviews
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class BarraRiskModel:
    """
    Barra-style Multi-Factor Risk Model
    
    Factors are grouped into style buckets:
    1. Price/Level: Price levels and Bollinger Bands
    2. Trend: Moving averages and trend indicators
    3. Momentum: Momentum, RSI, Stochastic, etc.
    4. Volatility: ATR, volatility measures
    5. Liquidity: OBV, AD, ADOSC, MFI
    6. Quality/Stability: Balance of Power and similar
    """
    
    def __init__(self, winsorize_percentile: float = 0.025, pca_variance_threshold: float = 0.5):
        """
        Initialize Barra risk model
        
        Args:
            winsorize_percentile: Percentile for winsorization (default 0.025 = 2.5% tail)
            pca_variance_threshold: Minimum variance explained by PC1 (default 0.5 = 50%)
        """
        self.winsorize_percentile = winsorize_percentile
        self.pca_variance_threshold = pca_variance_threshold
        
        # Define factor taxonomy (excluding Alpha factors - they are alpha signals, not risk factors)
        self.factor_taxonomy = self._define_factor_taxonomy()
        
        # Store intermediate results
        self.style_factors = None
        self.factor_covariance = None
        self.specific_risk = None
        self.pca_models = {}
        self.pca_variance_explained = {}
    
    def _define_factor_taxonomy(self) -> Dict[str, List[str]]:
        """
        Define factor taxonomy - classify raw factors into style buckets for construction.
        
        CRITICAL: Alpha factors (Alpha1-Alpha101) are RAW FACTORS, not alpha signals.
        They participate in building style factors just like RSI, MACD, etc.
        Unclassified factors (including Alpha factors that don't match patterns) go to 'Unclassified'
        and will still participate in style construction.
        
        Returns:
            Dictionary mapping style bucket names to lists of factor patterns
        """
        return {
            'Price/Level': [
                'VWAP', 'AVGPRICE', 'MEDPRICE', 'TYPPRICE', 'WCLPRICE',
                'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER'
            ],
            'Trend': [
                'SMA_', 'EMA_', 'WMA_', 'DEMA_', 'TRIMA_', 'TEMA_',
                'MACD', 'MACD_SIGNAL', 'MACD_HIST'
            ],
            'Momentum': [
                'MOM_', 'ROC_', 'RSI_', 'STOCH', 'WILLR_', 'CCI_',
                'AROON', 'ADX_', 'DX_', 'PLUS_DI_', 'MINUS_DI_', 
                'ADXR_', 'APO', 'ULTOSC'
            ],
            'Volatility': [
                'ATR_', 'NATR_', 'TRANGE', 'STDDEV_', 'BB_WIDTH'
            ],
            'Liquidity': [
                'OBV', 'AD', 'ADOSC', 'MFI_'
            ],
            'Quality/Stability': [
                'BOP'
            ],
            'Custom': [
                'CUSTOM_RS', 'CUSTOM_LAR', 'CUSTOM_PMS', 'CUSTOM_VAR', 'CUSTOM_PPF'
            ]
        }
    
    def classify_factors(self, factor_names: List[str]) -> Dict[str, List[str]]:
        """
        Classify raw factors into style buckets for construction.
        
        CRITICAL: Alpha factors (Alpha1-Alpha101) are RAW FACTORS, not alpha signals.
        They participate in building style factors just like RSI, MACD, etc.
        
        Args:
            factor_names: List of raw factor names (including Alpha factors)
            
        Returns:
            Dictionary mapping style bucket names to lists of classified factors
        """
        classified = {bucket: [] for bucket in self.factor_taxonomy.keys()}
        classified['Unclassified'] = []
        
        # Include ALL factors (including Alpha factors) - they are all raw factors
        for factor_name in factor_names:
            classified_flag = False
            for bucket_name, patterns in self.factor_taxonomy.items():
                for pattern in patterns:
                    if pattern in factor_name:
                        classified[bucket_name].append(factor_name)
                        classified_flag = True
                        break
                if classified_flag:
                    break
            
            if not classified_flag:
                # Alpha factors and other unclassified factors go to Unclassified
                # They will be included in style construction but not assigned to a specific style bucket
                classified['Unclassified'].append(factor_name)
        
        # Remove empty buckets
        classified = {k: v for k, v in classified.items() if v}
        
        return classified
    
    def winsorize_cross_sectional(self, factor_values: pd.Series, date: pd.Timestamp) -> pd.Series:
        """
        Winsorize factor values cross-sectionally for a given date
        
        Args:
            factor_values: Series of factor values for all stocks on a date
            date: Date timestamp
            
        Returns:
            Winsorized Series
        """
        q_low = self.winsorize_percentile
        q_high = 1 - self.winsorize_percentile
        
        lower_bound = factor_values.quantile(q_low)
        upper_bound = factor_values.quantile(q_high)
        
        return factor_values.clip(lower=lower_bound, upper=upper_bound)
    
    def zscore_normalize_cross_sectional(self, factor_values: pd.Series) -> pd.Series:
        """
        Z-score normalize factor values cross-sectionally
        
        Args:
            factor_values: Series of factor values (already winsorized)
            
        Returns:
            Z-score normalized Series
        """
        mean = factor_values.mean()
        std = factor_values.std()
        
        if std > 1e-8:
            return (factor_values - mean) / std
        else:
            return pd.Series(0.0, index=factor_values.index)
    
    def neutralize_factor(self, factor_values: pd.Series, 
                         market_cap: Optional[pd.Series] = None,
                         industry_dummies: Optional[pd.DataFrame] = None) -> pd.Series:
        """
        Neutralize factor against market cap and/or industry
        
        Args:
            factor_values: Series of factor values (already normalized)
            market_cap: Optional Series of market cap values
            industry_dummies: Optional DataFrame of industry dummy variables
            
        Returns:
            Neutralized factor values
        """
        y = factor_values.values.reshape(-1, 1)
        X_list = []
        
        # Add market cap if available
        if market_cap is not None and len(market_cap) == len(factor_values):
            # Log market cap
            log_mcap = np.log(market_cap.values + 1e-10).reshape(-1, 1)
            X_list.append(log_mcap)
        
        # Add industry dummies if available
        if industry_dummies is not None:
            X_list.append(industry_dummies.values)
        
        if len(X_list) == 0:
            # No neutralization needed
            return factor_values
        
        # Combine features
        X = np.hstack(X_list) if len(X_list) > 1 else X_list[0]
        
        # Fit OLS regression: factor = X * beta + residual
        # Residual is the neutralized factor
        try:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, y)
            residuals = y - reg.predict(X)
            return pd.Series(residuals.flatten(), index=factor_values.index)
        except Exception as e:
            # If regression fails, return original values
            warnings.warn(f"Neutralization failed: {e}, returning original values")
            return factor_values
    
    def reduce_dimension_within_bucket(self, bucket_factors: pd.DataFrame, 
                                      bucket_name: str) -> Tuple[pd.Series, float]:
        """
        Apply PCA to reduce dimensionality within a style bucket
        Returns the first principal component (PC1) as the representative style factor
        
        Args:
            bucket_factors: DataFrame of normalized factors for this bucket (stocks x factors)
            bucket_name: Name of the style bucket
            
        Returns:
            Tuple of (PC1 as Series, variance explained by PC1)
        """
        if bucket_factors.empty or len(bucket_factors.columns) == 0:
            return pd.Series(dtype=float), 0.0
        
        # Remove any columns with all NaN or constant values
        valid_cols = []
        for col in bucket_factors.columns:
            col_data = bucket_factors[col].dropna()
            if len(col_data) > 0:
                std_val = col_data.std()
                # Convert to scalar if Series
                if isinstance(std_val, pd.Series):
                    std_val = std_val.iloc[0] if len(std_val) > 0 else np.nan
                std_val = float(std_val) if not (pd.isna(std_val) or np.isnan(std_val)) else np.nan
                if not np.isnan(std_val) and std_val > 1e-8:
                    valid_cols.append(col)
        
        if len(valid_cols) == 0:
            return pd.Series(dtype=float), 0.0
        
        bucket_factors_clean = bucket_factors[valid_cols].dropna()
        
        if len(bucket_factors_clean) == 0:
            return pd.Series(dtype=float), 0.0
        
        if len(valid_cols) == 1:
            # If only one factor, return it directly (already normalized)
            single_col = bucket_factors_clean.iloc[:, 0]
            return single_col, 1.0  # 100% variance explained by definition
        
        try:
            # Apply PCA
            pca = PCA(n_components=1)
            pc1_values = pca.fit_transform(bucket_factors_clean.values)
            variance_explained = pca.explained_variance_ratio_[0]
            
            # Check if PC1 explains sufficient variance
            if variance_explained < self.pca_variance_threshold:
                # Fall back to weighted average (equal weights)
                warnings.warn(
                    f"PC1 for {bucket_name} explains only {variance_explained:.1%} variance "
                    f"(threshold: {self.pca_variance_threshold:.1%}), using weighted average"
                )
                pc1_series = bucket_factors_clean.mean(axis=1)
                variance_explained = 1.0  # Weighted average explains 100% by definition
            else:
                pc1_series = pd.Series(pc1_values.flatten(), index=bucket_factors_clean.index)
            
            # Store PCA model for later use
            self.pca_models[bucket_name] = pca
            self.pca_variance_explained[bucket_name] = variance_explained
            
            return pc1_series, variance_explained
            
        except Exception as e:
            # Fall back to weighted average if PCA fails
            warnings.warn(f"PCA failed for {bucket_name}: {e}, using weighted average")
            pc1_series = bucket_factors_clean.mean(axis=1)
            self.pca_variance_explained[bucket_name] = 1.0
            return pc1_series, 1.0
    
    def orthogonalize_style_factors(self, style_factors: pd.DataFrame) -> pd.DataFrame:
        """
        Orthogonalize style factors using Gram-Schmidt process
        
        This ensures style factors are approximately orthogonal, reducing multicollinearity
        
        Args:
            style_factors: DataFrame of style factors (stocks x style buckets)
            
        Returns:
            Orthogonalized style factors DataFrame
        """
        if style_factors.empty:
            return style_factors
        
        # Remove NaN rows
        style_factors_clean = style_factors.dropna()
        
        if len(style_factors_clean) == 0:
            return style_factors
        
        # Get factor values as matrix
        Q = style_factors_clean.values.copy()
        factor_names = style_factors_clean.columns.tolist()
        
        # Gram-Schmidt orthogonalization
        n_factors = Q.shape[1]
        Q_ortho = np.zeros_like(Q)
        
        for i in range(n_factors):
            # Start with original factor
            q = Q[:, i].copy()
            
            # Subtract projections onto previous orthogonal factors
            for j in range(i):
                q = q - np.dot(q, Q_ortho[:, j]) * Q_ortho[:, j]
            
            # Normalize
            norm = np.linalg.norm(q)
            if norm > 1e-8:
                Q_ortho[:, i] = q / norm
            else:
                Q_ortho[:, i] = q
        
        # Create DataFrame with same index and columns
        orthogonalized = pd.DataFrame(
            Q_ortho, 
            index=style_factors_clean.index,
            columns=factor_names
        )
        
        # Reindex to original index (fill NaN for removed rows)
        orthogonalized = orthogonalized.reindex(style_factors.index)
        
        return orthogonalized
    
    def estimate_factor_returns(self, style_exposures: pd.DataFrame, 
                               returns: pd.Series) -> pd.Series:
        """
        Estimate factor returns using cross-sectional regression
        
        Returns = Style Exposures * Factor Returns + Residual
        
        Args:
            style_exposures: DataFrame of style factor exposures (stocks x style factors)
            returns: Series of stock returns
            
        Returns:
            Series of factor returns
        """
        # Align data - ensure same index
        common_idx = style_exposures.index.intersection(returns.index)
        
        if len(common_idx) == 0:
            # No common stocks, return zero factor returns
            return pd.Series(0.0, index=style_exposures.columns)
        
        X = style_exposures.loc[common_idx].values
        y = returns.loc[common_idx].values
        
        # Ensure same length
        if len(X) != len(y):
            min_len = min(len(X), len(y))
            X = X[:min_len]
            y = y[:min_len]
        
        # Remove NaN rows
        X_nan_mask = np.isnan(X).any(axis=1) if len(X.shape) > 1 else np.isnan(X)
        y_nan_mask = np.isnan(y)
        valid_mask = ~(X_nan_mask | y_nan_mask)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            return pd.Series(dtype=float, index=style_exposures.columns)
        
        try:
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, y)
            
            factor_returns = pd.Series(reg.coef_, index=style_exposures.columns)
            return factor_returns
        except Exception as e:
            warnings.warn(f"Factor return estimation failed: {e}")
            return pd.Series(0.0, index=style_exposures.columns)
    
    def estimate_factor_covariance(self, factor_returns_history: pd.DataFrame,
                                  method: str = 'ewma', lambda_param: float = 0.94) -> pd.DataFrame:
        """
        Estimate factor covariance matrix from historical factor returns
        
        Args:
            factor_returns_history: DataFrame of historical factor returns (dates x factors)
            method: Method for covariance estimation ('ewma' or 'sample')
            lambda_param: EWMA decay parameter (default 0.94, common in Barra models)
            
        Returns:
            Factor covariance matrix DataFrame
        """
        if factor_returns_history.empty:
            return pd.DataFrame()
        
        # Remove NaN
        factor_returns_clean = factor_returns_history.dropna()
        
        if len(factor_returns_clean) < 2:
            return pd.DataFrame(index=factor_returns_history.columns, 
                              columns=factor_returns_history.columns).fillna(0.0)
        
        if method == 'ewma':
            # Exponentially weighted moving average covariance
            cov_matrix = factor_returns_clean.ewm(alpha=1-lambda_param, adjust=False).cov()
            # Get the last covariance matrix (most recent)
            dates = cov_matrix.index.get_level_values(0).unique()
            if len(dates) > 0:
                return cov_matrix.loc[dates[-1]]
            else:
                return pd.DataFrame(index=factor_returns_history.columns,
                                  columns=factor_returns_history.columns).fillna(0.0)
        else:
            # Sample covariance
            return factor_returns_clean.cov()
    
    def estimate_specific_risk(self, style_exposures: pd.DataFrame,
                              returns: pd.Series,
                              factor_returns: pd.Series) -> pd.Series:
        """
        Estimate specific (idiosyncratic) risk from regression residuals
        
        Specific Risk = Var(Returns - Style Exposures * Factor Returns)
        
        Args:
            style_exposures: DataFrame of style factor exposures (stocks x style factors)
            returns: Series of stock returns
            factor_returns: Series of factor returns
            
        Returns:
            Series of specific risk (standard deviation) for each stock
        """
        # Align data
        common_idx = style_exposures.index.intersection(returns.index)
        X = style_exposures.loc[common_idx]
        y = returns.loc[common_idx]
        
        # Predict returns using factor model
        predicted_returns = (X * factor_returns).sum(axis=1)
        
        # Calculate residuals
        residuals = y - predicted_returns
        
        # Specific risk is the standard deviation of residuals
        specific_risk = residuals.std()
        
        # Return as constant series (same for all stocks in this cross-section)
        # In practice, specific risk can vary by stock, but for simplicity we use average
        return pd.Series(specific_risk, index=common_idx)


def compute_portfolio_risk_decomposition(
    portfolio_weights: pd.Series,
    style_exposures: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    specific_risk: float
) -> Dict:
    """
    Compute portfolio risk decomposition
    
    Args:
        portfolio_weights: Series of portfolio weights (ticker -> weight)
        style_exposures: DataFrame of style factor exposures (ticker x style factor)
        factor_covariance: Factor covariance matrix
        specific_risk: Specific risk (scalar or Series)
        
    Returns:
        Dictionary with:
        - factor_exposures: Portfolio exposure to each style factor
        - factor_risk_contributions: Risk contribution of each factor (%)
        - specific_risk_contribution: Specific risk contribution (%)
        - total_variance: Total portfolio variance
    """
    # Align data
    common_tickers = portfolio_weights.index.intersection(style_exposures.index)
    weights = portfolio_weights.loc[common_tickers]
    exposures = style_exposures.loc[common_tickers]
    
    # Normalize weights
    weights = weights / weights.sum()
    
    # Portfolio factor exposures (weighted average)
    portfolio_factor_exposures = (weights.values @ exposures.values).flatten()
    portfolio_factor_exposures = pd.Series(
        portfolio_factor_exposures, 
        index=exposures.columns
    )
    
    # Factor risk contribution: w' * X * F * X' * w
    # Where X is factor exposures, F is factor covariance
    factor_risk_contribution = portfolio_factor_exposures.values @ factor_covariance.values @ portfolio_factor_exposures.values
    
    # Specific risk contribution: w' * diag(specific_risk^2) * w
    if isinstance(specific_risk, pd.Series):
        specific_var = (weights.values ** 2) @ (specific_risk.loc[common_tickers].values ** 2)
    else:
        specific_var = (weights.values ** 2).sum() * (specific_risk ** 2)
    
    # Total variance
    total_variance = factor_risk_contribution + specific_var
    
    # Risk contributions as percentages
    factor_risk_pct = (factor_risk_contribution / total_variance * 100) if total_variance > 0 else 0.0
    specific_risk_pct = (specific_var / total_variance * 100) if total_variance > 0 else 0.0
    
    # Individual factor contributions
    factor_contributions = {}
    for i, factor_name in enumerate(exposures.columns):
        # Contribution of this factor
        factor_exposure_vec = np.zeros(len(exposures.columns))
        factor_exposure_vec[i] = portfolio_factor_exposures.iloc[i]
        contrib = factor_exposure_vec @ factor_covariance.values @ portfolio_factor_exposures.values
        factor_contributions[factor_name] = (contrib / total_variance * 100) if total_variance > 0 else 0.0
    
    return {
        'factor_exposures': portfolio_factor_exposures.to_dict(),
        'factor_risk_contributions': factor_contributions,
        'specific_risk_contribution': specific_risk_pct,
        'total_variance': total_variance,
        'total_risk': np.sqrt(total_variance) if total_variance > 0 else 0.0
    }


def build_barra_risk_model(
    factor_store: pd.DataFrame,
    prices: pd.DataFrame,
    dates: List[pd.Timestamp],
    portfolio_weights: Optional[Dict[pd.Timestamp, pd.Series]] = None,
    lookback_days: int = 60
) -> Dict:
    """
    Build Barra risk model for multiple dates
    
    Main pipeline:
    1. Classify factors into style buckets (exclude Alpha factors)
    2. For each date:
       a. Winsorize and normalize raw factors cross-sectionally
       b. Apply PCA within each style bucket
       c. Orthogonalize style factors
    3. Estimate factor returns using cross-sectional regression
    4. Estimate factor covariance matrix
    5. Estimate specific risk
    6. Compute portfolio risk decomposition (if portfolio weights provided)
    
    Args:
        factor_store: DataFrame with MultiIndex (date, ticker) and factor columns
        prices: DataFrame with MultiIndex (date, ticker) and price/volume columns
        dates: List of dates to process
        portfolio_weights: Optional dict mapping date -> Series of portfolio weights
        lookback_days: Number of days to look back for covariance estimation
        
    Returns:
        Dictionary with risk model results for each date
    """
    model = BarraRiskModel()
    
    # Classify factors
    factor_names = factor_store.columns.tolist()
    classified = model.classify_factors(factor_names)
    
    print(f"[Barra] Factor classification:")
    for bucket, factors in classified.items():
        print(f"  {bucket}: {len(factors)} factors")
    
    results = {}
    factor_returns_history = []
    
    for date in dates:
        try:
            # Get date-specific data
            date_factors = factor_store.loc[factor_store.index.get_level_values(0) == date]
            if isinstance(date_factors.index, pd.MultiIndex):
                date_factors = date_factors.reset_index(level='date', drop=True)
            
            date_prices = prices.loc[prices.index.get_level_values(0) == date]
            if isinstance(date_prices.index, pd.MultiIndex):
                date_prices = date_prices.reset_index(level='date', drop=True)
            
            if date_factors.empty:
                continue
            
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
                for factor_name in bucket_factor_names:
                    factor_values = bucket_data[factor_name].dropna()
                    if len(factor_values) == 0:
                        continue
                    
                    # Winsorize
                    winsorized = model.winsorize_cross_sectional(factor_values, date)
                    # Z-score normalize
                    normalized = model.zscore_normalize_cross_sectional(winsorized)
                    # No neutralization (no market cap or industry data available)
                    normalized_factors.append(normalized)
                
                if len(normalized_factors) > 0:
                    bucket_df = pd.concat(normalized_factors, axis=1)
                    bucket_df.columns = [f for f in bucket_factor_names if f in date_factors.columns]
                    normalized_bucket_factors[bucket_name] = bucket_df
            
            # Step 2: Reduce dimension within each bucket (PCA)
            style_factors_dict = {}
            for bucket_name, bucket_data in normalized_bucket_factors.items():
                pc1, variance_explained = model.reduce_dimension_within_bucket(bucket_data, bucket_name)
                if len(pc1) > 0:
                    style_factors_dict[bucket_name] = pc1
            
            if len(style_factors_dict) == 0:
                continue
            
            # Combine into DataFrame
            style_factors_df = pd.DataFrame(style_factors_dict)
            
            # Step 3: Orthogonalize style factors
            style_factors_ortho = model.orthogonalize_style_factors(style_factors_df)
            
            # Step 4: Estimate factor returns (if we have forward returns)
            # For now, we'll use current date's style factors as exposures
            # Factor returns estimation requires forward returns, which we'll skip for now
            
            # Store results
            results[str(date.date())] = {
                'style_factors': style_factors_ortho.to_dict('index'),  # ticker -> {style: value}
                'pca_variance_explained': model.pca_variance_explained.copy()
            }
            
            # Step 5: If portfolio weights provided, compute risk decomposition
            if portfolio_weights and date in portfolio_weights:
                weights = portfolio_weights[date]
                # Compute portfolio exposures to style factors
                portfolio_exposures = {}
                for style_name in style_factors_ortho.columns:
                    style_values = style_factors_ortho[style_name]
                    common_tickers = weights.index.intersection(style_values.index)
                    if len(common_tickers) > 0:
                        weights_aligned = weights.loc[common_tickers] / weights.loc[common_tickers].sum()
                        portfolio_exposures[style_name] = (weights_aligned * style_values.loc[common_tickers]).sum()
                
                results[str(date.date())]['portfolio_exposures'] = portfolio_exposures
            
        except Exception as e:
            print(f"[Barra] Error processing date {date}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results

