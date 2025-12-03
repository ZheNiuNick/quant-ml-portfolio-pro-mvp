#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TA-Lib技术指标因子计算模块
精选50-80个常用的技术分析指标
"""

import numpy as np
import pandas as pd
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    print("[WARN] TA-Lib not available, will skip TA-Lib factors")


def calculate_talib_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    计算TA-Lib技术指标因子（50-80个精选指标）
    
    Args:
        prices: DataFrame with MultiIndex (date, ticker), columns: Open, High, Low, Close, Adj Close, Volume
    
    Returns:
        DataFrame with MultiIndex (date, ticker) and TA-Lib factor columns
    """
    if not HAS_TALIB:
        return pd.DataFrame()
    
    all_ticker_dfs = []
    tickers = prices.index.get_level_values("ticker").unique()
    total_tickers = len(tickers)
    
    print(f"  开始计算 {total_tickers} 只股票的TA-Lib因子...")
    
    # Process each ticker separately (TA-Lib works on 1D arrays)
    for ticker_idx, ticker in enumerate(tickers, 1):
        # 显示进度（每10只股票显示一次）
        if ticker_idx % 10 == 0 or ticker_idx == 1 or ticker_idx == total_tickers:
            progress = ticker_idx / total_tickers * 100
            print(f"    进度: {ticker_idx}/{total_tickers} ({progress:.1f}%) - {ticker}")
        ticker_data = prices.xs(ticker, level="ticker").sort_index()
        
        if len(ticker_data) < 50:  # Need enough data
            continue
        
        # Extract OHLCV arrays
        open_arr = ticker_data["Open"].values
        high_arr = ticker_data["High"].values
        low_arr = ticker_data["Low"].values
        close_arr = ticker_data["Adj Close"].values
        volume_arr = ticker_data["Volume"].values
        
        ticker_factors = {}
        
        # 1. Momentum Indicators (10个)
        ticker_factors["RSI_14"] = talib.RSI(close_arr, timeperiod=14)
        ticker_factors["RSI_6"] = talib.RSI(close_arr, timeperiod=6)
        ticker_factors["RSI_30"] = talib.RSI(close_arr, timeperiod=30)
        ticker_factors["MOM_10"] = talib.MOM(close_arr, timeperiod=10)
        ticker_factors["MOM_20"] = talib.MOM(close_arr, timeperiod=20)
        ticker_factors["ROC_10"] = talib.ROC(close_arr, timeperiod=10)
        ticker_factors["ROC_20"] = talib.ROC(close_arr, timeperiod=20)
        ticker_factors["CCI_14"] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=14)
        ticker_factors["CCI_20"] = talib.CCI(high_arr, low_arr, close_arr, timeperiod=20)
        ticker_factors["WILLR_14"] = talib.WILLR(high_arr, low_arr, close_arr, timeperiod=14)
        
        # 2. MACD系列 (3个)
        macd, macd_signal, macd_hist = talib.MACD(close_arr, fastperiod=12, slowperiod=26, signalperiod=9)
        ticker_factors["MACD"] = macd
        ticker_factors["MACD_SIGNAL"] = macd_signal
        ticker_factors["MACD_HIST"] = macd_hist
        
        # 3. Bollinger Bands (4个)
        bb_upper, bb_middle, bb_lower = talib.BBANDS(close_arr, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        ticker_factors["BB_UPPER"] = bb_upper
        ticker_factors["BB_MIDDLE"] = bb_middle
        ticker_factors["BB_LOWER"] = bb_lower
        ticker_factors["BB_WIDTH"] = (bb_upper - bb_lower) / bb_middle
        
        # 4. Moving Averages (8个)
        ticker_factors["SMA_5"] = talib.SMA(close_arr, timeperiod=5)
        ticker_factors["SMA_10"] = talib.SMA(close_arr, timeperiod=10)
        ticker_factors["SMA_20"] = talib.SMA(close_arr, timeperiod=20)
        ticker_factors["SMA_30"] = talib.SMA(close_arr, timeperiod=30)
        ticker_factors["EMA_12"] = talib.EMA(close_arr, timeperiod=12)
        ticker_factors["EMA_26"] = talib.EMA(close_arr, timeperiod=26)
        ticker_factors["WMA_20"] = talib.WMA(close_arr, timeperiod=20)
        ticker_factors["DEMA_20"] = talib.DEMA(close_arr, timeperiod=20)
        
        # 5. Volatility Indicators (5个)
        ticker_factors["ATR_14"] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=14)
        ticker_factors["ATR_20"] = talib.ATR(high_arr, low_arr, close_arr, timeperiod=20)
        ticker_factors["NATR_14"] = talib.NATR(high_arr, low_arr, close_arr, timeperiod=14)
        ticker_factors["TRANGE"] = talib.TRANGE(high_arr, low_arr, close_arr)
        ticker_factors["STDDEV_20"] = talib.STDDEV(close_arr, timeperiod=20, nbdev=1)
        
        # 6. Volume Indicators (6个)
        ticker_factors["OBV"] = talib.OBV(close_arr, volume_arr)
        ticker_factors["AD"] = talib.AD(high_arr, low_arr, close_arr, volume_arr)
        ticker_factors["ADOSC"] = talib.ADOSC(high_arr, low_arr, close_arr, volume_arr, fastperiod=3, slowperiod=10)
        ticker_factors["MFI_14"] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=14)
        ticker_factors["MFI_20"] = talib.MFI(high_arr, low_arr, close_arr, volume_arr, timeperiod=20)
        ticker_factors["VWAP"] = (high_arr + low_arr + close_arr) / 3  # Simplified VWAP (typical price)
        
        # 7. Trend Indicators (8个)
        ticker_factors["ADX_14"] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=14)
        ticker_factors["ADX_20"] = talib.ADX(high_arr, low_arr, close_arr, timeperiod=20)
        ticker_factors["ADXR_14"] = talib.ADXR(high_arr, low_arr, close_arr, timeperiod=14)
        aroon_up, aroon_down = talib.AROON(high_arr, low_arr, timeperiod=14)
        ticker_factors["AROON_UP"] = aroon_up
        ticker_factors["AROON_DOWN"] = aroon_down
        ticker_factors["AROONOSC"] = talib.AROONOSC(high_arr, low_arr, timeperiod=14)
        ticker_factors["DX_14"] = talib.DX(high_arr, low_arr, close_arr, timeperiod=14)
        ticker_factors["MINUS_DI_14"] = talib.MINUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
        ticker_factors["PLUS_DI_14"] = talib.PLUS_DI(high_arr, low_arr, close_arr, timeperiod=14)
        
        # 8. Oscillators (6个)
        slowk, slowd = talib.STOCH(high_arr, low_arr, close_arr, fastk_period=14, slowk_period=3, slowd_period=3)
        ticker_factors["STOCH_K"] = slowk
        ticker_factors["STOCH_D"] = slowd
        stochf_k, stochf_d = talib.STOCHF(high_arr, low_arr, close_arr, fastk_period=14, fastd_period=3)
        ticker_factors["STOCHF_K"] = stochf_k
        ticker_factors["STOCHF_D"] = stochf_d
        ticker_factors["ULTOSC"] = talib.ULTOSC(high_arr, low_arr, close_arr, timeperiod1=7, timeperiod2=14, timeperiod3=28)
        ticker_factors["APO"] = talib.APO(close_arr, fastperiod=12, slowperiod=26)
        
        # 9. Price Transform (4个)
        ticker_factors["AVGPRICE"] = talib.AVGPRICE(open_arr, high_arr, low_arr, close_arr)
        ticker_factors["MEDPRICE"] = talib.MEDPRICE(high_arr, low_arr)
        ticker_factors["TYPPRICE"] = talib.TYPPRICE(high_arr, low_arr, close_arr)
        ticker_factors["WCLPRICE"] = talib.WCLPRICE(high_arr, low_arr, close_arr)
        
        # 10. Math Transform (4个)
        ticker_factors["BOP"] = talib.BOP(open_arr, high_arr, low_arr, close_arr)
        ticker_factors["CDL2CROWS"] = talib.CDL2CROWS(open_arr, high_arr, low_arr, close_arr)
        ticker_factors["CDL3BLACKCROWS"] = talib.CDL3BLACKCROWS(open_arr, high_arr, low_arr, close_arr)
        ticker_factors["CDL3INSIDE"] = talib.CDL3INSIDE(open_arr, high_arr, low_arr, close_arr)
        
        # Convert to DataFrame
        ticker_df = pd.DataFrame(ticker_factors, index=ticker_data.index)
        ticker_df["ticker"] = ticker
        # Set ticker as second level, keep date as first level
        ticker_df = ticker_df.set_index("ticker", append=True)
        # Ensure correct order: date first, ticker second
        if ticker_df.index.names[0] != "date":
            ticker_df = ticker_df.swaplevel().sort_index()
        
        # Append to list
        all_ticker_dfs.append(ticker_df)
    
    # Combine all tickers
    if all_ticker_dfs:
        result = pd.concat(all_ticker_dfs, axis=0)
        result.index.names = ["date", "ticker"]
        result = result.sort_index()
        return result
    else:
        return pd.DataFrame()

