#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fixed version — YFinance multi-index handling
Now fetches tickers from Qlib S&P500 to match factor_engine.py
"""
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
import duckdb
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("[WARN] requests not installed. S&P500 list will use fallback method.")

# 使用统一的路径管理
from src.config.path import SETTINGS_FILE, DATA_FACTORS_DIR, get_path

SETTINGS = SETTINGS_FILE

def load_settings(path=SETTINGS_FILE):
    import yaml
    path = get_path(path) if isinstance(path, str) and not Path(path).is_absolute() else Path(path)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def get_sp500_tickers_from_web():
    """
    Get S&P500 ticker list from web (Wikipedia or yfinance).
    Returns list of ticker symbols.
    """
    try:
        # Method 1: Try Wikipedia (most reliable)
        import requests
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            tables = pd.read_html(response.text)
            sp500_table = tables[0]  # First table contains S&P500 list
            tickers = sp500_table["Symbol"].tolist()
            # Clean tickers (remove dots, convert to standard format)
            tickers = [t.replace(".", "-") for t in tickers]
            print(f"[INFO] Retrieved {len(tickers)} S&P500 tickers from Wikipedia")
            return sorted(tickers)
    except Exception as e:
        print(f"[WARN] Wikipedia method failed: {e}")
    
    try:
        # Method 2: Use yfinance to get S&P500 list
        import yfinance as yf
        sp500 = yf.Ticker("^GSPC")
        # This doesn't directly give us the list, but we can try another approach
        # Actually, yfinance doesn't provide S&P500 list directly
        pass
    except Exception as e:
        print(f"[WARN] yfinance method failed: {e}")
    
    # Method 3: Use a hardcoded list of major S&P500 stocks (fallback)
    # This is a subset - in production, use Wikipedia or a data provider
    print("[WARN] Using fallback S&P500 list (subset of major stocks)")
    major_sp500 = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "V", "UNH",
        "JNJ", "WMT", "JPM", "MA", "PG", "XOM", "HD", "CVX", "ABBV", "PFE",
        "AVGO", "COST", "MRK", "PEP", "TMO", "CSCO", "ABT", "ACN", "ADBE", "NFLX",
        "CMCSA", "NKE", "DHR", "VZ", "TXN", "LIN", "PM", "BMY", "QCOM", "HON",
        "AMGN", "INTU", "AMAT", "RTX", "LOW", "GE", "SPGI", "BKNG", "DE", "CAT"
    ]
    return sorted(major_sp500)

def get_tickers_from_qlib(instruments="csi300", region="cn"):
    """
    Get ticker list from Qlib (CSI300 for CN, S&P500 for US).
    Tries multiple methods:
    1. From factor_store.parquet if exists
    2. From Qlib instruments file
    3. From Qlib data API
    4. For US market: from web (Wikipedia)
    
    Args:
        instruments: "csi300" (CN) or "sp500" (US)
        region: "cn" (China A-share) or "us" (US stocks)
    """
    # Method 1: Read from Qlib instruments file (most reliable, should be checked first)
    # This should be checked before factor_store to avoid using wrong region's data
    qlib_data_path = Path.home() / ".qlib" / "qlib_data" / f"{region}_data"
    instruments_file = qlib_data_path / "instruments" / f"{instruments}.txt"
    if instruments_file.exists():
        try:
            df = pd.read_csv(instruments_file, sep="\t", names=["symbol", "start_date", "end_date"])
            tickers = sorted(df["symbol"].unique().tolist())
            print(f"[INFO] Loaded {len(tickers)} tickers from Qlib instruments/{instruments}.txt")
            return tickers
        except Exception as e:
            print(f"[Warn] Could not read from {instruments_file}: {e}")
    
    # Method 2: Try to extract from existing factor_store (only if region matches)
    factor_store_path = get_path("data/factors/factor_store.parquet", DATA_FACTORS_DIR)
    if factor_store_path.exists():
        try:
            df = pd.read_parquet(factor_store_path)
            if isinstance(df.index, pd.MultiIndex) and "ticker" in df.index.names:
                tickers = sorted(df.index.get_level_values("ticker").unique().tolist())
                # Check if tickers match the expected region
                # CN stocks start with SH/SZ, US stocks are typically 1-5 letter codes
                if region == "cn":
                    # Check if these are CN stocks (SH/SZ prefix)
                    if any(t.startswith(("SH", "SZ")) for t in tickers[:10]):
                        print(f"[INFO] Extracted {len(tickers)} tickers from factor_store.parquet (CN market)")
                        return tickers
                elif region == "us":
                    # Check if these are US stocks (no SH/SZ prefix, typically 1-5 letters)
                    if not any(t.startswith(("SH", "SZ")) for t in tickers[:10]):
                        print(f"[INFO] Extracted {len(tickers)} tickers from factor_store.parquet (US market)")
                        return tickers
                    else:
                        print(f"[Warn] factor_store.parquet contains CN stocks but region is US, skipping...")
        except Exception as e:
            print(f"[Warn] Could not read tickers from factor_store: {e}")

    # Method 3: Use Qlib API (requires initialization)
    try:
        import qlib
        qlib.init(provider_uri=str(qlib_data_path), region=region)
        from qlib.data import D
        
        # Get actual ticker list by querying a small date range
        start_time = "2008-01-01" if region == "cn" else "2016-01-01"
        data = D.features(instruments, ["$close"], start_time=start_time, end_time=start_time, freq="day")
        if not data.empty:
            tickers = sorted(data.index.get_level_values("instrument").unique().tolist())
            print(f"[INFO] Loaded {len(tickers)} tickers via Qlib API ({instruments})")
            return tickers
    except Exception as e:
        print(f"[Warn] Could not get tickers via Qlib API: {e}")

    # Method 5: For US market S&P500, try web as last resort
    if region == "us" and instruments == "sp500":
        try:
            return get_sp500_tickers_from_web()
        except Exception as e:
            print(f"[WARN] Web method also failed: {e}")

    # Fallback: return None and let user know
    raise ValueError(
        f"Could not determine {instruments} ticker list. "
        f"Options:\n"
        f"1. Run factor_engine.py first to generate factor_store.parquet\n"
        f"2. Ensure Qlib data is installed at ~/.qlib/qlib_data/{region}_data\n"
        f"3. Or manually specify tickers in settings.yaml"
    )

# 向后兼容函数
def get_sp500_tickers_from_qlib():
    """向后兼容函数：获取 S&P500 tickers（US 市场）"""
    return get_tickers_from_qlib(instruments="sp500", region="us")

def fetch_daily_prices_from_qlib(instruments, start, end, out_path, region="cn"):
    """
    Fetch prices from QLib (for CN market CSI300 or US market S&P500).
    Returns DataFrame with MultiIndex (date, ticker).
    """
    import qlib
    from qlib.data import D
    from pathlib import Path
    
    qlib_data_path = Path.home() / ".qlib" / "qlib_data" / f"{region}_data"
    qlib.init(provider_uri=str(qlib_data_path), region=region)
    
    print(f"[Info] Fetching prices from QLib for {instruments} ({region} market)...")
    print(f"[Info] Date range: {start} to {end}")
    
    # 关键修复：先获取股票列表，然后传给D.features()
    # 因为某些版本的QLib不接受字符串作为instruments参数
    # 方法1：直接从instruments文件读取（最可靠）
    instruments_file = qlib_data_path / "instruments" / f"{instruments}.txt"
    if instruments_file.exists():
        df = pd.read_csv(instruments_file, sep="\t", names=["symbol", "start_date", "end_date"])
        instruments_list = sorted(df["symbol"].unique().tolist())
        print(f"[Info] Found {len(instruments_list)} instruments from {instruments_file}")
    else:
        # 方法2：尝试使用D.list_instruments()
        try:
            instruments_list = D.list_instruments(instruments=instruments, start_time=start, end_time=end, as_list=True)
            print(f"[Info] Found {len(instruments_list)} instruments via D.list_instruments() for {instruments}")
        except Exception as e:
            # 方法3：直接使用字符串（某些版本的QLib支持）
            print(f"[Warn] Could not get instruments list: {e}, trying to use {instruments} directly as string...")
            instruments_list = instruments
    
    # QLib price fields: $open, $high, $low, $close, $volume, $vwap
    fields = ["$open", "$high", "$low", "$close", "$volume", "$vwap"]
    
    # Fetch data using QLib API
    data = D.features(
        instruments=instruments_list,
        fields=fields,
        start_time=start,
        end_time=end,
        freq="day"
    )
    
    if data.empty:
        raise ValueError(f"No data returned from QLib for {instruments} ({region}) in range {start} to {end}")
    
    print(f"[Info] QLib returned data shape: {data.shape}")
    
    # Convert to our format: MultiIndex (date, ticker)
    # QLib returns: MultiIndex (datetime, instrument), columns = fields
    data.index.names = ["date", "ticker"]
    
    # Map QLib field names to our column names
    # QLib: $close -> Adj Close, $open -> Open, etc.
    column_mapping = {
        "$open": "Open",
        "$high": "High",
        "$low": "Low",
        "$close": "Close",
        "$volume": "Volume",
        "$vwap": "VWAP"
    }
    
    # Rename columns
    data = data.rename(columns=column_mapping)
    
    # For Chinese market, Close = Adj Close (no adjustment needed)
    if "$close" in column_mapping:
        data["Adj Close"] = data["Close"].copy()
    
    # Ensure proper column order
    desired_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    available_cols = [c for c in desired_cols if c in data.columns]
    data = data[available_cols]
    
    # Sort index
    data = data.sort_index()
    
    # Save
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out_path)
    print(f"[OK] Saved prices to {out_path} ({data.shape[0]} rows, {data.shape[1]} cols)")
    print(f"[OK] Date range: {data.index.get_level_values('date').min()} to {data.index.get_level_values('date').max()}")
    print(f"[OK] Tickers: {len(data.index.get_level_values('ticker').unique())} unique")
    
    return data

def fetch_daily_prices(tickers, start, end, out_path, region="us"):
    """
    Download prices for tickers, handling failures gracefully.
    For US market: uses yfinance
    For CN market: uses QLib (call fetch_daily_prices_from_qlib instead)
    Returns DataFrame with MultiIndex (date, ticker).
    """
    # 如果是CN市场，使用QLib而不是yfinance
    if region == "cn":
        raise ValueError(
            "For CN market, please use fetch_daily_prices_from_qlib() instead of fetch_daily_prices(). "
            "Or use data_pipeline.py with --fetch flag which will automatically use QLib for CN market."
        )
    
    # 1. Download MultiIndex dataframe (Field x Ticker)
    # yfinance can handle up to ~2000 tickers, but S&P500 is ~500, so batch if needed
    batch_size = 500
    if len(tickers) > batch_size:
        print(f"[Info] Downloading {len(tickers)} tickers in batches of {batch_size}...")
        dfs = []
        failed_tickers = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            batch_num = i // batch_size + 1
            print(f"[Info] Batch {batch_num}: {len(batch)} tickers")
            
            try:
                # Don't use group_by="ticker" as it can cause structure issues when many tickers fail
                df_batch = yf.download(batch, start=start, end=end, auto_adjust=False, progress=False)
                
                # Handle empty or invalid responses
                if df_batch.empty:
                    print(f"[Warn] Batch {batch_num}: No data returned")
                    failed_tickers.extend(batch)
                    continue
                
                # Debug: check structure
                print(f"[Debug] Batch {batch_num}: shape={df_batch.shape}, columns type={type(df_batch.columns)}")
                
                # Without group_by, yfinance returns columns as (field, ticker) MultiIndex
                if not isinstance(df_batch.columns, pd.MultiIndex):
                    # If not MultiIndex, create it
                    if len(batch) == 1:
                        # Single ticker: columns are just fields
                        df_batch.columns = pd.MultiIndex.from_product([df_batch.columns, batch], names=["field", "ticker"])
                    else:
                        # Multiple tickers but not MultiIndex - shouldn't happen, but handle it
                        print(f"[Warn] Batch {batch_num}: Expected MultiIndex columns but got {type(df_batch.columns)}")
                        # Try to infer structure from column names
                        # yfinance without group_by usually creates (field, ticker) even for multiple tickers
                        # But if all tickers failed except one, might return single level
                        failed_tickers.extend(batch)
                        continue
                
                # Verify MultiIndex structure
                if df_batch.columns.nlevels != 2:
                    print(f"[Warn] Batch {batch_num}: Expected 2 column levels, got {df_batch.columns.nlevels}, skipping")
                    failed_tickers.extend(batch)
                    continue
                
                # yfinance with group_by="ticker" returns columns as (field, ticker) or (ticker, field)
                # Let's check the actual structure
                col_levels = df_batch.columns.names
                print(f"[Debug] Batch {batch_num}: Column levels: {col_levels}, first few columns: {list(df_batch.columns[:5])}")
                
                # Check which level has more unique values (fields typically ~6, tickers ~500)
                level_0_unique = len(df_batch.columns.get_level_values(0).unique())
                level_1_unique = len(df_batch.columns.get_level_values(1).unique())
                
                print(f"[Debug] Batch {batch_num}: Level 0 unique values: {level_0_unique}, Level 1 unique values: {level_1_unique}")
                
                # Determine the structure: if level 0 has ~6 values and level 1 has many, it's (field, ticker)
                # If level 1 has ~6 and level 0 has many, it's (ticker, field)
                if level_0_unique <= 10 and level_1_unique > 100:
                    # Structure is (field, ticker) - correct!
                    df_batch.columns.names = ["field", "ticker"]
                    stack_level = "ticker"
                elif level_1_unique <= 10 and level_0_unique > 100:
                    # Structure is (ticker, field) - need to swap
                    print(f"[Info] Batch {batch_num}: Swapping column levels (ticker, field) -> (field, ticker)")
                    df_batch = df_batch.swaplevel(axis=1)
                    df_batch.columns.names = ["field", "ticker"]
                    stack_level = "ticker"
                else:
                    # Can't determine, assume standard (field, ticker)
                    print(f"[Warn] Batch {batch_num}: Ambiguous column structure, assuming (field, ticker)")
                    df_batch.columns.names = ["field", "ticker"]
                    stack_level = "ticker"
                
                # Stack ticker to index - this creates (date, ticker) MultiIndex
                try:
                    df_batch = df_batch.stack(stack_level, future_stack=True)
                except Exception as e:
                    print(f"[Error] Batch {batch_num}: Failed to stack {stack_level}: {e}")
                    print(f"[Debug] Column structure: {df_batch.columns}")
                    failed_tickers.extend(batch)
                    continue
                
                # Verify result structure
                if not isinstance(df_batch.index, pd.MultiIndex):
                    print(f"[Error] Batch {batch_num}: After stack, index is not MultiIndex")
                    failed_tickers.extend(batch)
                    continue
                
                if df_batch.index.nlevels != 2:
                    print(f"[Error] Batch {batch_num}: After stack, index has {df_batch.index.nlevels} levels, expected 2")
                    failed_tickers.extend(batch)
                    continue
                
                # Set index names
                df_batch.index.names = ["date", "ticker"]
                
                # Verify: tickers should be stock symbols, not field names
                unique_tickers = df_batch.index.get_level_values('ticker').unique()
                field_names = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'}
                
                # If most "tickers" are actually field names, we stacked the wrong level
                if len(unique_tickers) <= 10 and set(unique_tickers).issubset(field_names):
                    print(f"[Error] Batch {batch_num}: Stacked wrong level! 'Tickers' are field names: {unique_tickers}")
                    print(f"[Debug] Structure must be (ticker, field), need to stack 'field' level instead")
                    # Need to unstack and restack correctly
                    # The original structure was likely (ticker, field), not (field, ticker)
                    failed_tickers.extend(batch)
                    continue
                
                dfs.append(df_batch)
                print(f"[OK] Batch {batch_num}: {df_batch.shape[0]} rows, {len(unique_tickers)} tickers (sample: {list(unique_tickers[:5])})")
                
            except Exception as e:
                print(f"[Error] Batch {batch_num} failed: {e}")
                failed_tickers.extend(batch)
                continue
        
        if dfs:
            df = pd.concat(dfs).sort_index()
            # Remove duplicates if any
            df = df[~df.index.duplicated(keep='first')]
            
            # Final verification
            if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
                raise ValueError(f"Final DataFrame has wrong index structure: {type(df.index)}, levels={df.index.nlevels if isinstance(df.index, pd.MultiIndex) else 'N/A'}")
            
            if "ticker" not in df.index.names:
                raise ValueError(f"Final DataFrame index names: {df.index.names}, expected ['date', 'ticker']")
        else:
            raise ValueError("No data downloaded from any batch")
            
        if failed_tickers:
            print(f"[Warn] {len(failed_tickers)} tickers failed to download (likely delisted): {failed_tickers[:20]}...")
            successful_tickers = set(tickers) - set(failed_tickers)
            print(f"[Info] Successfully downloaded {len(successful_tickers)}/{len(tickers)} tickers")
            
    else:
        # Single batch download
        print(f"[Info] Downloading {len(tickers)} tickers...")
        df = yf.download(tickers, start=start, end=end, auto_adjust=False, progress=False)
        
        if df.empty:
            raise ValueError(f"No data downloaded for any of {len(tickers)} tickers")
        
        print(f"[Debug] Single batch: shape={df.shape}, columns type={type(df.columns)}, nlevels={df.columns.nlevels if hasattr(df.columns, 'nlevels') else 'N/A'}")
        
        # Handle single vs multiple tickers
        if not isinstance(df.columns, pd.MultiIndex):
            if len(tickers) == 1:
                df.columns = pd.MultiIndex.from_product([df.columns, tickers], names=["field", "ticker"])
            else:
                raise ValueError("Unexpected column structure from yfinance")
        
        # Normalize column names
        if df.columns.nlevels != 2:
            raise ValueError(f"Unexpected column levels: {df.columns.nlevels}, expected 2")
        
        df.columns.names = ["field", "ticker"]
        df = df.stack("ticker", future_stack=True)
        
        # Verify result
        if not isinstance(df.index, pd.MultiIndex) or df.index.nlevels != 2:
            raise ValueError(f"After stack, index structure wrong: {type(df.index)}, levels={df.index.nlevels if isinstance(df.index, pd.MultiIndex) else 'N/A'}")

    # 2. Ensure proper index structure
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"Expected MultiIndex, got {type(df.index)}")
    
    df.index.names = ["date", "ticker"]

    # 3. Sort and save
    df = df.sort_index()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"[OK] Saved prices to {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"[OK] Date range: {df.index.get_level_values('date').min()} to {df.index.get_level_values('date').max()}")
    print(f"[OK] Tickers: {len(df.index.get_level_values('ticker').unique())} unique")
    return df

def sanity_checks(df: pd.DataFrame):
    """Basic sanity checks on price data."""
    # Check Adj Close (most important for our use case)
    if "Adj Close" in df.columns:
        neg_adj_close = (df["Adj Close"] < 0).sum()
        if neg_adj_close > 0:
            print(f"[Warn] Adj Close contains {neg_adj_close} negative values (likely data quality issue)")
            # Replace negatives with NaN instead of failing
            df.loc[df["Adj Close"] < 0, "Adj Close"] = pd.NA
    
    # Check Volume (should be non-negative, but allow zeros)
    if "Volume" in df.columns:
        neg_volume = (df["Volume"] < 0).sum()
        if neg_volume > 0:
            print(f"[Warn] Volume contains {neg_volume} negative values, replacing with 0")
            df.loc[df["Volume"] < 0, "Volume"] = 0
    
    # Check Close (might have negatives due to data issues, but less critical than Adj Close)
    if "Close" in df.columns:
        neg_close = (df["Close"] < 0).sum()
        if neg_close > 0:
            print(f"[Warn] Close contains {neg_close} negative values (data quality issue, using Adj Close instead)")
    
    print("[OK] Sanity checks passed (with warnings if any)")

def load_to_duckdb(prices_parquet, db_path):
    # 确保路径是字符串
    prices_parquet_str = str(prices_parquet) if isinstance(prices_parquet, Path) else prices_parquet
    db_path_str = str(db_path) if isinstance(db_path, Path) else db_path
    
    con = duckdb.connect(db_path_str)
    con.execute("INSTALL parquet; LOAD parquet;")
    con.execute("CREATE OR REPLACE TABLE prices AS SELECT * FROM parquet_scan(?)", [prices_parquet_str])
    con.close()
    print(f"[OK] DuckDB table 'prices' created at {db_path_str}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true", help="fetch yahoo daily prices")
    parser.add_argument("--tickers", nargs="+", help="Override: specify tickers manually (otherwise uses Qlib S&P500)")
    args = parser.parse_args()
    cfg = load_settings()
    if args.fetch:
        # Get tickers: from args, config, or Qlib S&P500
        if args.tickers:
            tickers = args.tickers
            print(f"[Info] Using {len(tickers)} tickers from command line argument")
        elif cfg.get("data", {}).get("universe"):
            tickers = cfg["data"]["universe"]
            print(f"[Info] Using {len(tickers)} tickers from settings.yaml")
        else:
            # Default: get from Qlib (CSI300 for CN, S&P500 for US)
            region = cfg.get("data", {}).get("region", "cn")
            instruments = cfg.get("data", {}).get("instruments", "csi300" if region == "cn" else "sp500")
            tickers = get_tickers_from_qlib(instruments=instruments, region=region)
            print(f"[Info] Using {len(tickers)} tickers from Qlib {instruments} (matching factor_engine)")
        
        # Get date range
        # Use current date - 1 day as default end date to ensure we get latest available data
        from datetime import datetime, timedelta
        default_end = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        start = cfg.get("data", {}).get("start", "2010-01-01")
        end = cfg.get("data", {}).get("end", default_end)
        
        # 关键修复：根据市场选择数据源
        # CN市场：使用QLib（因为yfinance不支持中国A股）
        # US市场：使用yfinance
        region = cfg.get("data", {}).get("region", "cn")
        
        if region == "cn":
            # 对于CN市场，直接从QLib获取价格数据（与factor_engine一致）
            instruments = cfg.get("data", {}).get("instruments", "csi300")
            print(f"[Info] Using QLib for CN market ({instruments})")
            df = fetch_daily_prices_from_qlib(instruments, start, end, cfg["paths"]["prices_parquet"], region=region)
        else:
            # 对于US市场，使用yfinance
            print(f"[Info] Using yfinance for US market")
            df = fetch_daily_prices(tickers, start, end, cfg["paths"]["prices_parquet"], region=region)
        
        sanity_checks(df)
        load_to_duckdb(cfg["paths"]["prices_parquet"], cfg["database"]["duckdb_path"])

if __name__ == "__main__":
    main()
