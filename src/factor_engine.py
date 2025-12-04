#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子引擎 - Alpha101 + TA-Lib + 自定义因子
- Alpha101: 101个WorldQuant经典因子
- TA-Lib: 50-80个精选技术指标
- 自定义: 5个针对S&P500优化的因子
- 总计: ~170-210个特征
"""

import argparse
from pathlib import Path
import json
from typing import Optional, Tuple

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:
    HAS_DUCKDB = False

import numpy as np
import pandas as pd
import warnings

# Import factor calculation modules
from src.alpha101 import calculate_alpha101_factors
from src.talib_factors import calculate_talib_factors
from src.custom_factors import calculate_custom_factors

# 使用统一的路径管理
from src.config.path import SETTINGS_FILE, DATA_PROCESSED_DIR, DATA_FACTORS_DIR, DUCKDB_DIR, get_path

SETTINGS = SETTINGS_FILE

# ------------------------
# Config & IO
# ------------------------
def load_settings(path=SETTINGS_FILE):
    import yaml
    path = get_path(path) if isinstance(path, str) and not Path(path).is_absolute() else Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def read_prices(cfg) -> pd.DataFrame:
    """
    Read daily OHLCV from parquet file or DuckDB.
    """
    # Try parquet file first
    parquet_path = get_path(cfg["paths"].get("prices_parquet", "data/processed/prices.parquet"), DATA_PROCESSED_DIR)
    if parquet_path.exists():
        try:
            df = pd.read_parquet(parquet_path)
            if not isinstance(df.index, pd.MultiIndex):
                if "date" in df.columns and "ticker" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index(["date", "ticker"]).sort_index()
                else:
                    df.index = pd.MultiIndex.from_tuples(df.index, names=["date", "ticker"])
            
            # Ensure proper index structure
            if isinstance(df.index, pd.MultiIndex):
                level_0 = df.index.get_level_values(0)
                level_1 = df.index.get_level_values(1)
                if pd.api.types.is_datetime64_any_dtype(level_0):
                    dates = pd.to_datetime(level_0, errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(level_1):
                        dates = pd.to_datetime(level_1, errors='coerce')
                        tickers = pd.Series(level_0).astype(str).values
                    else:
                        tickers = pd.Series(level_1).astype(str).values
                elif pd.api.types.is_datetime64_any_dtype(level_1):
                    dates = pd.to_datetime(level_1, errors='coerce')
                    tickers = pd.Series(level_0).astype(str).values
                else:
                    dates = pd.to_datetime(level_0, errors='coerce')
                    tickers = pd.Series(level_1).astype(str).values
                df.index = pd.MultiIndex.from_arrays([dates, tickers], names=["date", "ticker"])
            
            print(f"[Info] Loaded prices from {parquet_path}: {df.shape}")
            return df.sort_index()
        except Exception as e:
            print(f"[Warn] Failed to load from {parquet_path}: {e}")
    
    # Fallback to DuckDB
    if HAS_DUCKDB:
        try:
            con = duckdb.connect(cfg["database"]["duckdb_path"])
            df = con.execute("SELECT * FROM prices").df()
            con.close()
            if "date" in df.columns and "ticker" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index(["date", "ticker"]).sort_index()
            print(f"[Info] Loaded prices from DuckDB: {df.shape}")
            return df
        except Exception as e:
            print(f"[Warn] Failed to load from DuckDB: {e}")
    
    raise ValueError("Could not load price data from parquet or DuckDB")

# ------------------------
# Factor Calculation
# ------------------------
def calculate_all_factors(prices: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    计算所有因子：Alpha101 + TA-Lib + 自定义
    
    Args:
        prices: DataFrame with MultiIndex (date, ticker)
        start: Start date
        end: End date
    
    Returns:
        DataFrame with all factors
    """
    # Filter by date range
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    prices = prices.loc[(prices.index.get_level_values("date") >= start_date) & 
                        (prices.index.get_level_values("date") <= end_date)]
    
    if prices.empty:
        raise ValueError(f"No price data in range {start} to {end}")
    
    print(f"[Info] Calculating factors for {len(prices.index.get_level_values(1).unique())} tickers, "
          f"{len(prices.index.get_level_values(0).unique())} dates")
    
    all_factors = []
    
    # 1. Calculate Alpha101 factors
    print("[Info] Calculating Alpha101 factors (101 factors)...")
    try:
        alpha101 = calculate_alpha101_factors(prices)
        all_factors.append(alpha101)
        print(f"[OK] Alpha101: {len(alpha101.columns)} factors")
    except Exception as e:
        print(f"[Warn] Alpha101 calculation failed: {e}")
    
    # 2. Calculate TA-Lib factors
    print("[Info] Calculating TA-Lib factors (50-80 factors)...")
    try:
        talib_factors = calculate_talib_factors(prices)
        if not talib_factors.empty:
            all_factors.append(talib_factors)
            print(f"[OK] TA-Lib: {len(talib_factors.columns)} factors")
        else:
            print("[Warn] TA-Lib factors are empty")
    except Exception as e:
        print(f"[Warn] TA-Lib calculation failed: {e}")
    
    # 3. Calculate custom factors
    print("[Info] Calculating custom factors (5 factors)...")
    try:
        custom = calculate_custom_factors(prices)
        all_factors.append(custom)
        print(f"[OK] Custom: {len(custom.columns)} factors")
    except Exception as e:
        print(f"[Warn] Custom factors calculation failed: {e}")
        
    # Combine all factors
    if not all_factors:
        raise ValueError("No factors were calculated successfully")
        
    # Merge all factors on index
    result = all_factors[0]
    for df in all_factors[1:]:
        result = result.join(df, how="outer")
    
    result = result.sort_index()
    
    print(f"[OK] Total factors: {len(result.columns)}")
    print(f"[OK] Date range: {result.index.get_level_values(0).min()} to {result.index.get_level_values(0).max()}")
    print(f"[OK] Tickers: {len(result.index.get_level_values(1).unique())}")
        
    return result

# ------------------------
# Processing helpers
# ------------------------
def winsorize_by_group(s: pd.Series, lower=0.01, upper=0.99, level="date") -> pd.Series:
    """Winsorize by date group (legacy method, using quantiles)"""
    def _clip(x):
        ql, qh = x.quantile(lower), x.quantile(upper)
        return x.clip(ql, qh)
    return s.groupby(level=level).transform(_clip)

def mad_winsorize_by_group(s: pd.Series, n_mad: float = 3.0, level="date") -> pd.Series:
    """
    MAD-based winsorize by date group (industry standard)
    
    Args:
        s: Series with MultiIndex (date, ticker)
        n_mad: Number of MADs to clip (default 3.0)
        level: Group level (default "date")
    
    Returns:
        Winsorized Series
    """
    def _mad_clip(x):
        """Clip values beyond n_mad * MAD from median"""
        median = x.median()
        mad = (x - median).abs().median()
        
        # If MAD is 0, all values are the same, return as is
        if mad == 0 or not np.isfinite(mad):
            return x
        
        lower = median - n_mad * mad
        upper = median + n_mad * mad
        return x.clip(lower, upper)
    
    return s.groupby(level=level).transform(_mad_clip)
        
def rank_normalize_by_group(s: pd.Series, level="date") -> pd.Series:
    """
    Rank normalization by date group (industry standard for ranking models)
    
    Converts values to percentile ranks (0-1 scale), which is more robust
    to outliers and better suited for ranking models like lambdarank.
    
    Args:
        s: Series with MultiIndex (date, ticker)
        level: Group level (default "date")
    
    Returns:
        Rank-normalized Series (0-1 scale)
    """
    def _rank_norm(x):
        """Convert to percentile rank"""
        return x.rank(pct=True, method='first')
    
    return s.groupby(level=level).transform(_rank_norm)

def zscore_by_group(s: pd.Series, level="date") -> pd.Series:
    """Z-score normalization by date group (legacy method)"""
    def _z(x):
        std = x.std(ddof=0)
        return (x - x.mean()) / (std if std > 0 else 1.0)
    return s.groupby(level=level).transform(_z)

def qlib_style_processing(raw: pd.Series) -> pd.Series:
    """
    QLib style processing: only handle outliers, no normalization
    """
    x = raw.replace([np.inf, -np.inf], np.nan)
    x.name = raw.name
    return x

def light_processing(raw: pd.Series) -> pd.Series:
    """
    Light processing: Winsorize + Z-score (legacy method)
    """
    x = winsorize_by_group(raw, 0.01, 0.99, level="date")
    x = zscore_by_group(x, level="date")
    x.name = raw.name
    return x

def industry_standard_processing(raw: pd.Series) -> pd.Series:
    """
    Industry standard processing: MAD winsorize only (no normalization)
    
    This method:
    1. MAD winsorize: More robust to outliers than quantile-based
    2. Preserves original value distribution and information
    
    Note: Rank normalization is removed to preserve more information
    and avoid potential information loss that may cause negative Rank IC.
    
    Args:
        raw: Raw factor Series with MultiIndex (date, ticker)
    
    Returns:
        Processed Series (MAD winsorized, but not normalized)
    """
    # Step 1: MAD winsorize only (no rank normalization)
    x = mad_winsorize_by_group(raw, n_mad=3.0, level="date")
    
    # Note: Rank normalization removed to preserve information
    # If normalization is needed, it should be done during model training
    
    x.name = raw.name
    return x


def orthogonalize_factors(factor_store: pd.DataFrame,
                          method: str = "gram_schmidt",
                          order_by_ic: bool = True,
                          forward_ret: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Multi-step orthogonalization of factors (industry standard)
    
    Removes correlation between factors to reduce multicollinearity
    and improve model stability.
    
    Args:
        factor_store: DataFrame with MultiIndex (date, ticker), columns are factors
        method: "gram_schmidt" or "qr" (QR decomposition)
        order_by_ic: If True, order factors by IC before orthogonalization
        forward_ret: Forward return series for IC calculation (if order_by_ic=True)
    
    Returns:
        Orthogonalized factor DataFrame with same structure
    """
    from scipy.linalg import qr
    
    if factor_store.empty:
        return factor_store
    
    dates = factor_store.index.get_level_values("date").unique()
    orthogonalized_list = []
    
    print(f"\n[Info] Applying {method} orthogonalization to {len(factor_store.columns)} factors...")
    if order_by_ic and forward_ret is not None:
        print("[Info] Ordering factors by IC before orthogonalization...")
    
    # Calculate IC for ordering (if needed)
    factor_ics = {}
    if order_by_ic and forward_ret is not None:
        for col in factor_store.columns:
            factor_series = factor_store[col]
            ic_series = daily_rank_ic(factor_series, forward_ret)
            factor_ics[col] = ic_series.mean() if not ic_series.empty else 0.0
        
        # Sort factors by absolute IC (descending)
        sorted_cols = sorted(factor_store.columns, 
                           key=lambda x: abs(factor_ics.get(x, 0.0)), 
                           reverse=True)
    else:
        sorted_cols = list(factor_store.columns)
    
    # Process each date separately (cross-sectional orthogonalization)
    for date in dates:
        date_data = factor_store.xs(date, level=0)
        
        # Reorder columns if needed
        date_data = date_data[sorted_cols]
        
        # Remove rows with any NaN
        date_data = date_data.dropna()
        
        if date_data.empty or len(date_data) < 2:
            # If not enough data, keep original structure but fill with zeros
            # This ensures consistent column structure across all dates
            if date_data.empty:
                orth_index = pd.MultiIndex.from_product([[date], []], names=["date", "ticker"])
            else:
                if isinstance(date_data.index, pd.MultiIndex):
                    orth_index = date_data.index
                else:
                    orth_index = pd.MultiIndex.from_product([[date], date_data.index], names=["date", "ticker"])
            
            orth_df = pd.DataFrame(
                0.0,
                index=orth_index,
                columns=factor_store.columns
            )
            orthogonalized_list.append(orth_df)
            continue
        
        # Extract values as numpy array
        X = date_data.values
        
        # Handle constant columns (std = 0)
        stds = np.std(X, axis=0)
        valid_cols = stds > 1e-10
        if not valid_cols.any():
            # All columns are constant, return zeros with original structure
            if isinstance(date_data.index, pd.MultiIndex):
                orth_index = date_data.index
            else:
                orth_index = pd.MultiIndex.from_product([[date], date_data.index], names=["date", "ticker"])
            
            orth_df = pd.DataFrame(
                0.0,
                index=orth_index,
                columns=factor_store.columns
            )
            orthogonalized_list.append(orth_df)
            continue
        
        X_valid = X[:, valid_cols]
        valid_col_names = date_data.columns[valid_cols].tolist()
        
        if method == "qr":
            # QR decomposition method
            Q, R = qr(X_valid, mode='economic')
            X_orth = Q
        else:
            # Gram-Schmidt method (default)
            X_orth = np.zeros_like(X_valid)
            X_orth[:, 0] = X_valid[:, 0]
            
            for i in range(1, X_valid.shape[1]):
                # Project current column onto previous orthogonal columns
                proj = np.zeros(X_valid.shape[0])
                for j in range(i):
                    proj += np.dot(X_valid[:, i], X_orth[:, j]) / (np.dot(X_orth[:, j], X_orth[:, j]) + 1e-10) * X_orth[:, j]
                
                # Orthogonal component
                X_orth[:, i] = X_valid[:, i] - proj
        
        # Create DataFrame with original structure
        # Ensure index is MultiIndex with (date, ticker)
        if isinstance(date_data.index, pd.MultiIndex):
            orth_index = date_data.index
        else:
            # Create MultiIndex with date and ticker
            orth_index = pd.MultiIndex.from_product([[date], date_data.index], names=["date", "ticker"])
        
        orth_df = pd.DataFrame(
            X_orth,
            index=orth_index,
            columns=valid_col_names
        )
        
        # Add back constant columns (if any) as zeros
        if not valid_cols.all():
            const_cols = date_data.columns[~valid_cols].tolist()
            for col in const_cols:
                orth_df[col] = 0.0
        
        # Reorder to match original column order
        orth_df = orth_df[factor_store.columns]
        orthogonalized_list.append(orth_df)
    
    # Combine all dates
    if orthogonalized_list:
        result = pd.concat(orthogonalized_list, sort=False)
        # Ensure MultiIndex structure is preserved
        if not isinstance(result.index, pd.MultiIndex):
            # If somehow lost MultiIndex, try to reconstruct
            # This shouldn't happen, but just in case
            result.index = pd.MultiIndex.from_tuples(
                [(d, t) for d, df in zip(dates, orthogonalized_list) for t in df.index.get_level_values(-1)],
                names=["date", "ticker"]
            )
        result = result.sort_index()
        print(f"[OK] Orthogonalization complete: {len(result.columns)} factors, {len(result)} samples")
        return result
    else:
        print("[Warn] No data after orthogonalization, returning original")
        return factor_store


def _compute_factor_stability(series: pd.Series,
                              window: int,
                              min_periods: int) -> Tuple[float, float]:
    """
    计算某个因子的滚动稳定性指标：
      - 对每个 ticker 的时间序列计算 rolling std
      - 汇总所有 ticker 的 rolling std 中位数
      - coverage 表示有效 rolling 样本占比
    """
    panel = series.unstack("ticker").sort_index()
    if panel.empty:
        return np.nan, 0.0
    rolling_std = panel.rolling(window=window, min_periods=min_periods).std()
    values = rolling_std.values
    if values.size == 0:
        return np.nan, 0.0
    coverage = np.isfinite(values).sum() / values.size
    median_std = float(np.nanmedian(values)) if coverage > 0 else np.nan
    return median_std, coverage


def apply_stability_filter(factor_store: pd.DataFrame,
                           cfg: dict) -> tuple[pd.DataFrame, dict]:
    """
    根据滚动波动率剔除不稳定因子
    Returns:
        filtered_factor_store, removed_info
    """
    if not cfg.get("enabled", False):
        return factor_store, {}

    window = int(cfg.get("window", 60))
    threshold = float(cfg.get("threshold", 3.0))
    min_coverage = float(cfg.get("min_coverage", 0.25))
    min_periods = max(5, window // 2)

    keep_cols = []
    removed = {}

    print(f"\n[Info] Running factor stability filter "
          f"(window={window}, threshold={threshold}, min_coverage={min_coverage})...")

    for col in factor_store.columns:
        median_std, coverage = _compute_factor_stability(factor_store[col], window, min_periods)
        if np.isnan(median_std) or coverage < min_coverage or median_std > threshold:
            removed[col] = {
                "median_rolling_std": None if np.isnan(median_std) else float(median_std),
                "coverage": float(coverage),
            }
        else:
            keep_cols.append(col)

    if removed:
        print(f"[Filter] Removed {len(removed)} unstable factors; keeping {len(keep_cols)}")
    else:
        print("[Info] No factors removed by stability filter")

    filtered = factor_store[keep_cols] if keep_cols else factor_store.iloc[:, 0:0]
    return filtered, removed

# ------------------------
# Labels and metrics
# ------------------------
def forward_return(pr: pd.DataFrame, horizon: int = 1) -> pd.Series:
    """
    Calculate forward return: (price[t+horizon] / price[t]) - 1
    """
    close = pr["Adj Close"].unstack("ticker")
    fwd = (close.shift(-horizon) / close - 1).stack()
    fwd.name = f"fwd{horizon}"
    return fwd

def daily_rank_ic(factor: pd.Series, forward_ret: pd.Series) -> pd.Series:
    """Calculate daily Rank IC"""
    df = pd.concat([factor, forward_ret], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float)
    
    ic = df.groupby(level="date").apply(
        lambda x: x.iloc[:, 0].corr(x.iloc[:, 1], method="spearman")
    )
    return ic

# ------------------------
# Main builder
# ------------------------
def build_and_evaluate(cfg):
    """Build factor store from prices"""
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    # Load prices
    pr = read_prices(cfg)
    
    # Filter by universe if specified (for memory optimization)
    universe = cfg.get("data", {}).get("universe")
    if universe:
        # Filter prices to only include tickers in universe
        available_tickers = set(pr.index.get_level_values(1).unique())
        universe_tickers = set([str(t).strip() for t in universe])
        valid_tickers = list(universe_tickers & available_tickers)
        
        if len(valid_tickers) < len(universe_tickers):
            missing = universe_tickers - available_tickers
            print(f"[Warn] {len(missing)} tickers from universe not found in price data: {sorted(list(missing))[:10]}...")
        
        if valid_tickers:
            pr = pr.loc[pr.index.get_level_values(1).isin(valid_tickers)]
            print(f"[Info] Filtered to {len(valid_tickers)} tickers from universe (out of {len(universe)} requested)")
        else:
            raise ValueError(f"No valid tickers found in price data for the specified universe")
    
    # Get date range from config
    start = cfg.get("data", {}).get("start", "2010-01-01")
    end = cfg.get("data", {}).get("end", "2025-11-22")
    
    # Calculate all factors
    print("\n" + "="*60)
    print("Factor Engine: Alpha101 + TA-Lib + Custom")
    print("="*60)
    alphas = calculate_all_factors(pr, start, end)
    
    # Processing: Use industry standard method (MAD + Rank)
    processing_method = cfg.get("factor_processing", {}).get("method", "industry_standard")
    use_qlib_style = cfg.get("factor_processing", {}).get("qlib_style", False)
    
    print(f"\n[Info] Processing factors (method={processing_method}, qlib_style={use_qlib_style})...")
    processed_factors = {}
    
    total_factors = len(alphas.columns)
    
    # 处理所有因子
    for idx, col in enumerate(alphas.columns, 1):
        # 显示进度（每50个因子显示一次）
        if idx % 50 == 0 or idx == 1 or idx == total_factors:
            progress = idx / total_factors * 100
            print(f"  处理进度: {idx}/{total_factors} ({progress:.1f}%) - {col}")
        
        raw = alphas[col].dropna()
        if raw.empty:
            continue

        # Choose processing method
        if use_qlib_style:
            proc = qlib_style_processing(raw)
        elif processing_method == "industry_standard":
            proc = industry_standard_processing(raw)
        else:
            proc = light_processing(raw)  # Legacy method
        
        processed_factors[col] = proc
    
    # Combine processed factors
    factor_store = pd.DataFrame(processed_factors)
    factor_store = factor_store.sort_index()
    
    # Step 3: Apply orthogonalization (if enabled)
    ortho_cfg = cfg.get("factor_enhancement", {}).get("orthogonalization", {})
    if ortho_cfg.get("enabled", False):
        print(f"\n[Info] Applying orthogonalization...")
        # Calculate forward return for IC-based ordering
        forward_ret = forward_return(pr, horizon=1) if ortho_cfg.get("order_by_ic", True) else None
        
        factor_store = orthogonalize_factors(
            factor_store,
            method=ortho_cfg.get("method", "gram_schmidt"),
            order_by_ic=ortho_cfg.get("order_by_ic", True),
            forward_ret=forward_ret
        )

    # 应用稳定性过滤
    stability_cfg = cfg.get("factor_processing", {}).get("stability_filter", {})
    stability_report = {}
    if stability_cfg.get("enabled", False):
        factor_store, stability_report = apply_stability_filter(factor_store, stability_cfg)
        if factor_store.empty:
            raise ValueError("All factors were removed by the stability filter. "
                             "Consider relaxing threshold/min_coverage.")
        if stability_report:
            output_dir = Path("outputs/reports")
            output_dir.mkdir(parents=True, exist_ok=True)
            stability_path = output_dir / "factor_stability_report.json"
            with open(stability_path, "w") as f:
                json.dump(stability_report, f, indent=2)
            print(f"[Info] Stability report saved to {stability_path}")
    
    # 应用因子增强（降维、质量改进等）
    enhancement_cfg = cfg.get("factor_enhancement", {})
    if enhancement_cfg.get("enabled", False):
        try:
            from src.factor_enhancement import enhance_factors
            print(f"\n[Info] Applying factor enhancement...")
            factor_store, enhancement_info = enhance_factors(factor_store, pr, cfg)
            if enhancement_info.get("steps"):
                print(f"[Enhancement] Final factors: {enhancement_info.get('final_n_factors', len(factor_store.columns))} "
                      f"(original: {enhancement_info.get('original_n_factors', len(factor_store.columns))})")
                # 保存增强报告
                output_dir = Path("outputs/reports")
                output_dir.mkdir(parents=True, exist_ok=True)
                enhancement_path = output_dir / "factor_enhancement_report.json"
                with open(enhancement_path, "w") as f:
                    json.dump(enhancement_info, f, indent=2, default=str)
                print(f"[Info] Enhancement report saved to {enhancement_path}")
        except ImportError as e:
            print(f"[Warn] Factor enhancement not available: {e}")
        except Exception as e:
            print(f"[Warn] Factor enhancement failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Save factor store
    store_path = get_path(cfg["paths"]["factors_store"], DATA_FACTORS_DIR)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    factor_store.to_parquet(store_path)
    
    print(f"\n[OK] Factor store saved: {store_path}")
    print(f"[OK] Total factors: {len(factor_store.columns)}")
    print(f"[OK] Shape: {factor_store.shape}")
    print(f"[OK] Date range: {factor_store.index.get_level_values(0).min()} to {factor_store.index.get_level_values(0).max()}")

# ------------------------
# CLI
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build factor store")
    args = parser.parse_args()

    cfg = load_settings()
    if args.build:
        build_and_evaluate(cfg)

if __name__ == "__main__":
    main()
