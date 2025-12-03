#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义因子模块
针对S&P500市场优化的5个自定义因子
"""

import numpy as np
import pandas as pd


def calculate_custom_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    计算5个针对S&P500优化的自定义因子
    
    Args:
        prices: DataFrame with MultiIndex (date, ticker), columns: Open, High, Low, Close, Adj Close, Volume
    
    Returns:
        DataFrame with MultiIndex (date, ticker) and custom factor columns
    """
    close = prices["Adj Close"].unstack("ticker")
    open_price = prices["Open"].unstack("ticker")
    high = prices["High"].unstack("ticker")
    low = prices["Low"].unstack("ticker")
    volume = prices["Volume"].unstack("ticker")
    
    factors = {}
    
    # 保存原始索引，确保所有因子都有相同的日期范围
    original_index = close.index
    
    # Custom Factor 1: 相对强度因子 (Relative Strength)
    # 计算股票相对于市场（使用所有股票的平均）的表现
    market_avg = close.mean(axis=1)
    # 使用div确保按行对齐
    custom_rs = close.div(market_avg, axis=0) - 1
    # 确保保留所有日期
    custom_rs = custom_rs.reindex(original_index)
    factors["CUSTOM_RS"] = custom_rs.stack()
    
    # Custom Factor 2: 流动性调整收益率 (Liquidity-Adjusted Return)
    # 考虑成交量的收益率，高成交量时收益率权重更高
    returns = close.pct_change()
    volume_ma = volume.rolling(20).mean()
    custom_lar = (returns * (volume / (volume_ma + 1e-10)))
    custom_lar = custom_lar.reindex(original_index)
    factors["CUSTOM_LAR"] = custom_lar.stack()
    
    # Custom Factor 3: 价格动量强度 (Price Momentum Strength)
    # 结合短期和长期动量的强度
    short_momentum = close / close.shift(5) - 1
    long_momentum = close / close.shift(20) - 1
    custom_pms = (short_momentum * long_momentum)
    custom_pms = custom_pms.reindex(original_index)
    factors["CUSTOM_PMS"] = custom_pms.stack()
    
    # Custom Factor 4: 波动率调整收益率 (Volatility-Adjusted Return)
    # 收益率除以波动率，得到风险调整后的收益
    returns_20d = close.pct_change(20)
    volatility_20d = close.pct_change().rolling(20).std()
    custom_var = (returns_20d / (volatility_20d + 1e-10))
    custom_var = custom_var.reindex(original_index)
    factors["CUSTOM_VAR"] = custom_var.stack()
    
    # Custom Factor 5: 价格位置因子 (Price Position Factor)
    # 当前价格在过去N天价格区间中的位置
    high_20d = high.rolling(20).max()
    low_20d = low.rolling(20).min()
    custom_ppf = ((close - low_20d) / (high_20d - low_20d + 1e-10))
    custom_ppf = custom_ppf.reindex(original_index)
    factors["CUSTOM_PPF"] = custom_ppf.stack()
    
    # Combine all factors
    df = pd.DataFrame(factors)
    df.index.names = ["date", "ticker"]
    df = df.sort_index()
    
    # 确保返回的DataFrame包含所有原始日期和股票
    # 重新索引到原始prices的索引
    original_multiindex = prices.index
    df = df.reindex(original_multiindex)
    
    return df

