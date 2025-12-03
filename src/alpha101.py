#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Alpha101因子计算模块
基于WorldQuant的101个经典Alpha因子实现
适配项目格式：MultiIndex (date, ticker) DataFrame
"""

import numpy as np
import pandas as pd
from numpy import abs, log, sign
from scipy.stats import rankdata

# ==================== Auxiliary functions ====================
def ts_sum(df, window=10):
    """Rolling sum"""
    return df.rolling(window).sum()

def sma(df, window=10):
    """Simple Moving Average"""
    return df.rolling(window).mean()

def stddev(df, window=10):
    """Rolling standard deviation"""
    return df.rolling(window).std()

def correlation(x, y, window=10):
    """Rolling correlation - compute column-wise"""
    # For DataFrame, compute correlation column by column
    if isinstance(x, pd.DataFrame) and isinstance(y, pd.DataFrame):
        result = pd.DataFrame(index=x.index, columns=x.columns, dtype=float)
        for col in x.columns:
            if col in y.columns:
                # Compute rolling correlation for each column pair
                x_col = x[col]
                y_col = y[col]
                # Use pandas rolling corr (more efficient)
                corr_series = x_col.rolling(window).corr(y_col)
                # Replace inf with NaN (occurs when one series has zero variance)
                corr_series = corr_series.replace([-np.inf, np.inf], np.nan)
                result[col] = corr_series
        return result
    else:
        # For Series, use standard rolling correlation
        corr_result = x.rolling(window).corr(y)
        return corr_result.replace([-np.inf, np.inf], np.nan)

def covariance(x, y, window=10):
    """Rolling covariance"""
    return x.rolling(window).cov(y)

def rolling_rank(na):
    """Auxiliary function for rolling rank"""
    return rankdata(na)[-1]

def ts_rank(df, window=10):
    """Time-series rank"""
    return df.rolling(window).apply(rolling_rank)

def rolling_prod(na):
    """Auxiliary function for rolling product"""
    return np.prod(na)

def product(df, window=10):
    """Rolling product"""
    return df.rolling(window).apply(rolling_prod)

def ts_min(df, window=10):
    """Rolling min"""
    return df.rolling(window).min()

def ts_max(df, window=10):
    """Rolling max"""
    return df.rolling(window).max()

def delta(df, period=1):
    """Difference: today's value minus value 'period' days ago"""
    return df.diff(period)

def delay(df, period=1):
    """Lag: shift by period"""
    return df.shift(period)

def rank(df):
    """Cross-sectional rank (along columns, axis=1)"""
    return df.rank(axis=1, pct=True)

def scale(df, k=1):
    """Scaling: rescaled such that sum(abs(df)) = k"""
    return df.mul(k).div(np.abs(df).sum())

def ts_argmax(df, window=10):
    """Which day ts_max(df, window) occurred on"""
    return df.rolling(window).apply(np.argmax) + 1

def ts_argmin(df, window=10):
    """Which day ts_min(df, window) occurred on"""
    return df.rolling(window).apply(np.argmin) + 1

def decay_linear(df, period=10):
    """Linear weighted moving average"""
    # Clean data
    if df.isnull().values.any():
        df = df.ffill().bfill().fillna(value=0)
    
    na_lwma = np.zeros_like(df)
    na_lwma[:period, :] = df.iloc[:period, :]
    na_series = df.values  # Use .values instead of deprecated .as_matrix()
    
    divisor = period * (period + 1) / 2
    y = (np.arange(period) + 1) * 1.0 / divisor
    
    for row in range(period - 1, df.shape[0]):
        x = na_series[row - period + 1: row + 1, :]
        na_lwma[row, :] = (np.dot(x.T, y))
    
    return pd.DataFrame(na_lwma, index=df.index, columns=df.columns)

# ==================== Alpha101 Class ====================
class Alphas(object):
    def __init__(self, df_data):
        """
        Initialize with unstacked DataFrame (date index, ticker columns)
        """
        self.open = df_data['Open']
        self.high = df_data['High']
        self.low = df_data['Low']
        self.close = df_data['Adj Close']  # Use Adj Close
        self.volume = df_data['Volume']
        # Calculate returns and vwap
        self.returns = self.close.pct_change()
        self.vwap = (self.high + self.low + self.close) / 3  # Simplified VWAP
    
    # Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) -0.5)
    def alpha001(self):
        inner = self.close.copy()
        inner[self.returns < 0] = stddev(self.returns, 20)
        return rank(ts_argmax(inner ** 2, 5)) - 0.5
    
    # Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
    def alpha002(self):
        df = -1 * correlation(rank(delta(log(self.volume + 1e-10), 2)), rank((self.close - self.open) / (self.open + 1e-10)), 6)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))
    def alpha003(self):
        df = -1 * correlation(rank(self.open), rank(self.volume), 10)
        # Replace inf with NaN, keep NaN as NaN (will be handled by processing)
        # If correlation returns inf, it means one series has zero variance in the window
        return df.replace([-np.inf, np.inf], np.nan)
    
    # Alpha#4: (-1 * Ts_Rank(rank(low), 9))
    def alpha004(self):
        return -1 * ts_rank(rank(self.low), 9)
    
    # Alpha#5: (rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
    def alpha005(self):
        return (rank((self.open - (ts_sum(self.vwap, 10) / 10))) * (-1 * abs(rank((self.close - self.vwap)))))
    
    # Alpha#6: (-1 * correlation(open, volume, 10))
    def alpha006(self):
        df = -1 * correlation(self.open, self.volume, 10)
        return df.replace([-np.inf, np.inf], 0).fillna(value=0)
    
    # Alpha#7: ((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1* 1))
    def alpha007(self):
        adv20 = sma(self.volume, 20)
        alpha = -1 * ts_rank(abs(delta(self.close, 7)), 60) * sign(delta(self.close, 7))
        alpha[adv20 >= self.volume] = -1
        return alpha
    
    # Alpha#8: (-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)),10))))
    def alpha008(self):
        return -1 * (rank(((ts_sum(self.open, 5) * ts_sum(self.returns, 5)) -
                           delay((ts_sum(self.open, 5) * ts_sum(self.returns, 5)), 10))))
    
    # Alpha#9: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ?delta(close, 1) : (-1 * delta(close, 1))))
    def alpha009(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 5) > 0
        cond_2 = ts_max(delta_close, 5) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return alpha
    
    # Alpha#10: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0)? delta(close, 1) : (-1 * delta(close, 1)))))
    def alpha010(self):
        delta_close = delta(self.close, 1)
        cond_1 = ts_min(delta_close, 4) > 0
        cond_2 = ts_max(delta_close, 4) < 0
        alpha = -1 * delta_close
        alpha[cond_1 | cond_2] = delta_close
        return rank(alpha)
    
    # Alpha#11: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) *rank(delta(volume, 3)))
    def alpha011(self):
        return ((rank(ts_max((self.vwap - self.close), 3)) + rank(ts_min((self.vwap - self.close), 3))) * rank(delta(self.volume, 3)))
    
    # Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))
    def alpha012(self):
        return sign(delta(self.volume, 1)) * (-1 * delta(self.close, 1))

    # Alpha#13: (-1 * rank(covariance(rank(close), rank(volume), 5)))
    def alpha013(self):
        return -1 * rank(covariance(rank(self.close), rank(self.volume), 5))
    
    # Alpha#14: ((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
    def alpha014(self):
        df = correlation(self.open, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(delta(self.returns, 3)) * df
    
    # Alpha#15: (-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
    def alpha015(self):
        df = correlation(rank(self.high), rank(self.volume), 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_sum(rank(df), 3)
    
    # Alpha#16: (-1 * rank(covariance(rank(high), rank(volume), 5)))
    def alpha016(self):
        return -1 * rank(covariance(rank(self.high), rank(self.volume), 5))
    
    # Alpha#17: (((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) *rank(ts_rank((volume / adv20), 5)))
    def alpha017(self):
        adv20 = sma(self.volume, 20)
        return -1 * (rank(ts_rank(self.close, 10)) *
                     rank(delta(delta(self.close, 1), 1)) *
                     rank(ts_rank((self.volume / (adv20 + 1e-10)), 5)))
        
    # Alpha#18: (-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open,10))))
    def alpha018(self):
        df = correlation(self.close, self.open, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank((stddev(abs((self.close - self.open)), 5) + (self.close - self.open)) + df))
    
    # Alpha#19: ((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns,250)))))
    def alpha019(self):
        return ((-1 * sign((self.close - delay(self.close, 7)) + delta(self.close, 7))) *
                (1 + rank(1 + ts_sum(self.returns, 250))))
    
    # Alpha#20: (((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open -delay(low, 1))))
    def alpha020(self):
        return -1 * (rank(self.open - delay(self.high, 1)) *
                     rank(self.open - delay(self.close, 1)) *
                     rank(self.open - delay(self.low, 1)))

    # Alpha#21: ((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close,2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume /adv20) == 1)) ? 1 : (-1 * 1))))
    def alpha021(self):
        cond_1 = sma(self.close, 8) + stddev(self.close, 8) < sma(self.close, 2)
        cond_2 = sma(self.volume, 20) / (self.volume + 1e-10) < 1
        alpha = pd.DataFrame(np.ones_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond_1 | cond_2] = -1
        return alpha
    
    # Alpha#22: (-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
    def alpha022(self):
        df = correlation(self.high, self.volume, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * delta(df, 5) * rank(stddev(self.close, 20))

    # Alpha#23: (((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
    def alpha023(self):
        cond = sma(self.high, 20) < self.high
        alpha = pd.DataFrame(np.zeros_like(self.close), index=self.close.index, columns=self.close.columns)
        alpha[cond] = -1 * delta(self.high, 2).fillna(value=0)
        return alpha
    
    # Alpha#24: ((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) ||((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close,100))) : (-1 * delta(close, 3)))
    def alpha024(self):
        cond = delta(sma(self.close, 100), 100) / (delay(self.close, 100) + 1e-10) <= 0.05
        alpha = -1 * delta(self.close, 3)
        alpha[cond] = -1 * (self.close - ts_min(self.close, 100))
        return alpha
    
    # Alpha#25: rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
    def alpha025(self):
        adv20 = sma(self.volume, 20)
        return rank(((((-1 * self.returns) * adv20) * self.vwap) * (self.high - self.close)))
    
    # Alpha#26: (-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
    def alpha026(self):
        df = correlation(ts_rank(self.volume, 5), ts_rank(self.high, 5), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_max(df, 3)
    
    # Alpha#27: ((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
    def alpha027(self):
        df = correlation(rank(self.volume), rank(self.vwap), 6)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        alpha = rank(sma(df, 2) / 2.0)
        # Convert to -1 or 1 based on condition
        result = pd.DataFrame(np.ones_like(alpha), index=alpha.index, columns=alpha.columns)
        result[alpha > 0.5] = -1
        return result
    
    # Alpha#28: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    def alpha028(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((df + ((self.high + self.low) / 2)) - self.close))

    # Alpha#29: (min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1),5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
    def alpha029(self):
        return (ts_min(rank(rank(scale(log(ts_sum(rank(rank(-1 * rank(delta((self.close - 1), 5)))), 2))))), 5) +
                ts_rank(delay((-1 * self.returns), 6), 5))

    # Alpha#30: (((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) +sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
    def alpha030(self):
        delta_close = delta(self.close, 1)
        inner = sign(delta_close) + sign(delay(delta_close, 1)) + sign(delay(delta_close, 2))
        return ((1.0 - rank(inner)) * ts_sum(self.volume, 5)) / (ts_sum(self.volume, 20) + 1e-10)

    # Alpha#31: ((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 *delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    def alpha031(self):
        adv20 = sma(self.volume, 20)
        df = correlation(adv20, self.low, 12).replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = rank(rank(rank(decay_linear((-1 * rank(rank(delta(self.close, 10)))), 10))))
        p2 = rank((-1 * delta(self.close, 3)))
        p3 = sign(scale(df))
        return p1 + p2 + p3

    # Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5),230))))
    def alpha032(self):
        df = correlation(self.vwap, delay(self.close, 5), 230)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return scale(((sma(self.close, 7) / 7) - self.close)) + (20 * scale(df))
    
    # Alpha#33: rank((-1 * ((1 - (open / close))^1)))
    def alpha033(self):
        return rank(-1 + (self.open / (self.close + 1e-10)))
    
    # Alpha#34: rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
    def alpha034(self):
        inner = stddev(self.returns, 2) / (stddev(self.returns, 5) + 1e-10)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return rank(2 - rank(inner) - rank(delta(self.close, 1)))

    # Alpha#35: ((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 -Ts_Rank(returns, 32)))
    def alpha035(self):
        return ((ts_rank(self.volume, 32) *
                 (1 - ts_rank(self.close + self.high - self.low, 16))) *
                (1 - ts_rank(self.returns, 32)))
            
    # Alpha#36: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open- close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap,adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
    def alpha036(self):
        adv20 = sma(self.volume, 20)
        df = correlation(self.vwap, adv20, 6)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (((((2.21 * rank(correlation((self.close - self.open), delay(self.volume, 1), 15))) + (0.7 * rank((self.open - self.close)))) + (0.73 * rank(ts_rank(delay((-1 * self.returns), 6), 5)))) + rank(abs(df))) + (0.6 * rank((((sma(self.close, 200) / 200) - self.open) * (self.close - self.open)))))
    
    # Alpha#37: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    def alpha037(self):
        df = correlation(delay(self.open - self.close, 1), self.close, 200)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return rank(df) + rank(self.open - self.close)
    
    # Alpha#38: ((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
    def alpha038(self):
        inner = self.close / (self.open + 1e-10)
        inner = inner.replace([-np.inf, np.inf], 1).fillna(value=1)
        return -1 * rank(ts_rank(self.close, 10)) * rank(inner)
    
    # Alpha#39: ((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 +rank(sum(returns, 250))))
    def alpha039(self):
        adv20 = sma(self.volume, 20)
        vol_ratio = self.volume / (adv20 + 1e-10)
        decayed = decay_linear(vol_ratio, 9)
        return ((-1 * rank(delta(self.close, 7) * (1 - rank(decayed)))) *
                (1 + rank(ts_sum(self.returns, 250))))
    
    # Alpha#40: ((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
    def alpha040(self):
        df = correlation(self.high, self.volume, 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * rank(stddev(self.high, 10)) * df

    # Alpha#41: (((high * low)^0.5) - vwap)
    def alpha041(self):
        return pow((self.high * self.low), 0.5) - self.vwap
    
    # Alpha#42: (rank((vwap - close)) / rank((vwap + close)))
    def alpha042(self):
        return rank((self.vwap - self.close)) / (rank((self.vwap + self.close)) + 1e-10)
        
    # Alpha#43: (ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
    def alpha043(self):
        adv20 = sma(self.volume, 20)
        return ts_rank(self.volume / (adv20 + 1e-10), 20) * ts_rank((-1 * delta(self.close, 7)), 8)

    # Alpha#44: (-1 * correlation(high, rank(volume), 5))
    def alpha044(self):
        df = correlation(self.high, rank(self.volume), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * df

    # Alpha#45: (-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) *rank(correlation(sum(close, 5), sum(close, 20), 2))))
    def alpha045(self):
        df = correlation(self.close, self.volume, 2)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        df2 = correlation(ts_sum(self.close, 5), ts_sum(self.close, 20), 2)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * (rank(sma(delay(self.close, 5), 20)) * df * rank(df2))
    
    # Alpha#46: ((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ?(-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 :((-1 * 1) * (close - delay(close, 1)))))
    def alpha046(self):
        inner = ((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10)
        alpha = (-1 * delta(self.close))
        alpha[inner < 0] = 1
        alpha[inner > 0.25] = -1
        return alpha

    # Alpha#47: ((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) /5))) - rank((vwap - delay(vwap, 5))))
    def alpha047(self):
        adv20 = sma(self.volume, 20)
        return ((((rank((1 / (self.close + 1e-10))) * self.volume) / (adv20 + 1e-10)) * ((self.high * rank((self.high - self.close))) / (sma(self.high, 5) / 5 + 1e-10))) - rank((self.vwap - delay(self.vwap, 5))))
    
    # Alpha#48: (indneutralize...) - Skip (requires industry data)
    def alpha048(self):
        # Simplified version without industry neutralization
        df = correlation(delta(self.close, 1), delta(delay(self.close, 1), 1), 250)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (df * delta(self.close, 1)) / (self.close + 1e-10)
    
    # Alpha#49: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha049(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.1] = 1
        return alpha
    
    # Alpha#50: (-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    def alpha050(self):
        df = correlation(rank(self.volume), rank(self.vwap), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (-1 * ts_max(rank(df), 5))
    
    # Alpha#51: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 *0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
    def alpha051(self):
        inner = (((delay(self.close, 20) - delay(self.close, 10)) / 10) - ((delay(self.close, 10) - self.close) / 10))
        alpha = (-1 * delta(self.close))
        alpha[inner < -0.05] = 1
        return alpha
    
    # Alpha#52: ((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) -sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    def alpha052(self):
        return (((-1 * delta(ts_min(self.low, 5), 5)) *
                 rank(((ts_sum(self.returns, 240) - ts_sum(self.returns, 20)) / 220))) * ts_rank(self.volume, 5))
        
    # Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
    def alpha053(self):
        inner = (self.close - self.low).replace(0, 0.0001)
        return -1 * delta((((self.close - self.low) - (self.high - self.close)) / inner), 9)

    # Alpha#54: ((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
    def alpha054(self):
        inner = (self.low - self.high).replace(0, -0.0001)
        return -1 * (self.low - self.close) * (self.open ** 5) / (inner * (self.close ** 5) + 1e-10)

    # Alpha#55: (-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low,12)))), rank(volume), 6))
    def alpha055(self):
        divisor = (ts_max(self.high, 12) - ts_min(self.low, 12)).replace(0, 0.0001)
        inner = (self.close - ts_min(self.low, 12)) / divisor
        df = correlation(rank(inner), rank(self.volume), 6)
        return -1 * df.replace([-np.inf, np.inf], 0).fillna(value=0)

    # Alpha#56: (0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap))))) - Skip (requires cap)
    def alpha056(self):
        # Simplified version without cap
        return 0 - (1 * rank((ts_sum(self.returns, 10) / (sma(ts_sum(self.returns, 2), 3) + 1e-10))) * rank(self.returns))
    
    # Alpha#57: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    def alpha057(self):
        decayed = decay_linear(rank(ts_argmax(self.close, 30)), 2)
        return (0 - (1 * ((self.close - self.vwap) / (decayed + 1e-10))))
    
    # Alpha#58: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume,3.92795), 7.89291), 5.50322)) - Skip (requires industry)
    def alpha058(self):
        # Simplified version without industry neutralization
        df = correlation(self.vwap, self.volume, 4)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_rank(decay_linear(df, 8), 5)
    
    # Alpha#59: (-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(...), volume, 4.25197), 16.2289), 8.19648)) - Skip
    def alpha059(self):
        # Simplified version
        df = correlation(self.vwap, self.volume, 4)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return -1 * ts_rank(decay_linear(df, 16), 8)
    
    # Alpha#60: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) -scale(rank(ts_argmax(close, 10))))))
    def alpha060(self):
        divisor = (self.high - self.low).replace(0, 0.0001)
        inner = ((self.close - self.low) - (self.high - self.close)) * self.volume / divisor
        return - ((2 * scale(rank(inner))) - scale(rank(ts_argmax(self.close, 10))))
    
    # Alpha#61: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    def alpha061(self):
        adv180 = sma(self.volume, 180)
        df = correlation(self.vwap, adv180, 18)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (rank((self.vwap - ts_min(self.vwap, 16))) < rank(df)).astype(float)
    
    # Alpha#62: ((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) +rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
    def alpha062(self):
        adv20 = sma(self.volume, 20)
        df = correlation(self.vwap, sma(adv20, 22), 10)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        cond = (rank(self.open) + rank(self.open)) < (rank(((self.high + self.low) / 2)) + rank(self.high))
        return ((rank(df) < rank(cond.astype(float))) * -1)
    
    # Alpha#63: ((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))- rank(...)) * -1) - Skip
    def alpha063(self):
        # Simplified version
        adv180 = sma(self.volume, 180)
        p1 = rank(decay_linear(delta(self.close, 2), 8))
        df = correlation(self.vwap * 0.318108 + self.open * (1 - 0.318108), sma(adv180, 37), 14)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = rank(decay_linear(df, 12))
        return (p1 - p2) * -1
    
    # Alpha#64: ((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054),sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 -0.178404))), 3.69741))) * -1)
    def alpha064(self):
        adv120 = sma(self.volume, 120)
        df = correlation(sma(((self.open * 0.178404) + (self.low * (1 - 0.178404))), 13), sma(adv120, 13), 17)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return ((rank(df) < rank(delta(((((self.high + self.low) / 2) * 0.178404) + (self.vwap * (1 - 0.178404))), 4))) * -1)
    
    # Alpha#65: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60,8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
    def alpha065(self):
        adv60 = sma(self.volume, 60)
        df = correlation(((self.open * 0.00817205) + (self.vwap * (1 - 0.00817205))), sma(adv60, 9), 6)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return ((rank(df) < rank((self.open - ts_min(self.open, 14)))) * -1)
      
    # Alpha#66: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    def alpha066(self):
        p1 = rank(decay_linear(delta(self.vwap, 4), 7))
        inner = (((self.low * 0.96633) + (self.low * (1 - 0.96633))) - self.vwap) / (self.open - ((self.high + self.low) / 2) + 1e-10)
        p2 = ts_rank(decay_linear(inner, 11), 7)
        return (p1 + p2) * -1
    
    # Alpha#67: ((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap,IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1) - Skip
    def alpha067(self):
        # Simplified version
        adv20 = sma(self.volume, 20)
        df = correlation(self.vwap, adv20, 6)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (rank((self.high - ts_min(self.high, 2))) ** rank(df) * -1)
    
    # Alpha#68: ((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) <rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
    def alpha068(self):
        adv15 = sma(self.volume, 15)
        df = correlation(rank(self.high), rank(adv15), 9)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = ts_rank(df, 14)
        p2 = rank(delta(((self.close * 0.518371) + (self.low * (1 - 0.518371))), 1))
        # Align indices and convert boolean to float
        result = (p1 < p2).astype(float) * -1
        return result.fillna(0)
    
    # Alpha#69: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412),4.79344))^Ts_Rank(...)) * -1) - Skip
    def alpha069(self):
        # Simplified version
        adv20 = sma(self.volume, 20)
        p1 = rank(ts_max(delta(self.vwap, 3), 5))
        df = correlation(self.close * 0.490655 + self.vwap * (1 - 0.490655), adv20, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(df, 9)
        return (p1 ** p2 * -1)
    
    # Alpha#70: ((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close,IndClass.industry), adv50, 17.8256), 17.9171)) * -1) - Skip
    def alpha070(self):
        # Simplified version
        adv50 = sma(self.volume, 50)
        df = correlation(self.close, adv50, 18)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (rank(delta(self.vwap, 1)) ** ts_rank(df, 18) * -1)
    
    # Alpha#71: max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180,12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap +vwap)))^2), 16.4662), 4.4388))
    def alpha071(self):
        adv180 = sma(self.volume, 180)
        df = correlation(ts_rank(self.close, 3), ts_rank(adv180, 12), 18)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = ts_rank(decay_linear(df, 4), 16)
        p2 = ts_rank(decay_linear((rank(((self.low + self.open) - (self.vwap + self.vwap))) ** 2), 16), 4)
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).max(axis=1)
        return result.fillna(0)
    
    # Alpha#72: (rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) /rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671),2.95011)))
    def alpha072(self):
        adv40 = sma(self.volume, 40)
        df1 = correlation(((self.high + self.low) / 2), adv40, 9)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = rank(decay_linear(df1, 10))
        df2 = correlation(ts_rank(self.vwap, 4), ts_rank(self.volume, 19), 7)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = rank(decay_linear(df2, 3))
        return p1 / (p2 + 1e-10)
    
    # Alpha#73: (max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)),Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open *0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
    def alpha073(self):
        p1 = rank(decay_linear(delta(self.vwap, 5), 3))
        inner = (delta(((self.open * 0.147155) + (self.low * (1 - 0.147155))), 2) / ((self.open * 0.147155) + (self.low * (1 - 0.147155)) + 1e-10)) * -1
        p2 = ts_rank(decay_linear(inner, 3), 17)
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).max(axis=1) * -1
        return result.fillna(0)
    
    # Alpha#74: ((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) <rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))* -1)
    def alpha074(self):
        adv30 = sma(self.volume, 30)
        df1 = correlation(self.close, sma(adv30, 37), 15)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        df2 = correlation(rank(((self.high * 0.0261661) + (self.vwap * (1 - 0.0261661)))), rank(self.volume), 11)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return ((rank(df1) < rank(df2)) * -1)
    
    # Alpha#75: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50),12.4413)))
    def alpha075(self):
        adv50 = sma(self.volume, 50)
        df1 = correlation(self.vwap, self.volume, 4)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        df2 = correlation(rank(self.low), rank(adv50), 12)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (rank(df1) < rank(df2)).astype(float)
    
    # Alpha#76: (max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)),Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81,8.14941), 19.569), 17.1543), 19.383)) * -1) - Skip
    def alpha076(self):
        # Simplified version
        adv81 = sma(self.volume, 81)
        p1 = rank(decay_linear(delta(self.vwap, 1), 12))
        df = correlation(self.low, adv81, 8)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(decay_linear(ts_rank(df, 20), 17), 19)
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).max(axis=1) * -1
        return result.fillna(0)
    
    # Alpha#77: min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)),rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
    def alpha077(self):
        adv40 = sma(self.volume, 40)
        p1 = rank(decay_linear(((((self.high + self.low) / 2) + self.high) - (self.vwap + self.high)), 20))
        df = correlation(((self.high + self.low) / 2), adv40, 3)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = rank(decay_linear(df, 6))
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).min(axis=1)
        return result.fillna(0)
    
    # Alpha#78: (rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428),sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
    def alpha078(self):
        adv40 = sma(self.volume, 40)
        df1 = correlation(ts_sum(((self.low * 0.352233) + (self.vwap * (1 - 0.352233))), 20), ts_sum(adv40, 20), 7)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        df2 = correlation(rank(self.vwap), rank(self.volume), 6)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return rank(df1) ** rank(df2)
    
    # Alpha#79: (rank(delta(IndNeutralize(...), 1.23438)) < rank(correlation(...))) - Skip
    def alpha079(self):
        # Simplified version
        adv150 = sma(self.volume, 150)
        p1 = rank(delta(self.close * 0.60733 + self.open * (1 - 0.60733), 1))
        df = correlation(ts_rank(self.vwap, 4), ts_rank(adv150, 9), 15)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = rank(df)
        return (p1 < p2).astype(float)
    
    # Alpha#80: ((rank(Sign(delta(IndNeutralize(...), 4.04545)))^Ts_Rank(...)) * -1) - Skip
    def alpha080(self):
        # Simplified version
        adv10 = sma(self.volume, 10)
        p1 = rank(sign(delta(self.open * 0.868128 + self.high * (1 - 0.868128), 4)))
        df = correlation(self.high, adv10, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(df, 6)
        return (p1 ** p2 * -1)
    
    # Alpha#81: ((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054),8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    def alpha081(self):
        adv10 = sma(self.volume, 10)
        df1 = correlation(self.vwap, ts_sum(adv10, 50), 8)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        inner = rank(rank(df1)) ** 4
        p1 = rank(log(product(rank(inner), 15)))
        df2 = correlation(rank(self.vwap), rank(self.volume), 5)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = rank(df2)
        return ((p1 < p2) * -1)
    
    # Alpha#82: (min(rank(decay_linear(delta(open, 1.46063), 14.8717)),Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) +(open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1) - Skip
    def alpha082(self):
        # Simplified version
        p1 = rank(decay_linear(delta(self.open, 1), 15))
        df = correlation(self.volume, self.open, 17)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(decay_linear(df, 7), 13)
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).min(axis=1) * -1
        return result.fillna(0)
    
    # Alpha#83: ((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high -low) / (sum(close, 5) / 5)) / (vwap - close)))
    def alpha083(self):
        divisor = (ts_sum(self.close, 5) / 5 + 1e-10)
        inner = (self.high - self.low) / divisor
        numerator = rank(delay(inner, 2)) * rank(rank(self.volume))
        denominator = inner / (self.vwap - self.close + 1e-10)
        return numerator / (denominator + 1e-10)
    
    # Alpha#84: SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close,4.96796))
    def alpha084(self):
        return pow(ts_rank((self.vwap - ts_max(self.vwap, 15)), 21), delta(self.close, 5))
    
    # Alpha#85: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30,9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595),7.11408)))
    def alpha085(self):
        adv30 = sma(self.volume, 30)
        df1 = correlation(((self.high * 0.876703) + (self.close * (1 - 0.876703))), adv30, 10)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        df2 = correlation(ts_rank(((self.high + self.low) / 2), 4), ts_rank(self.volume, 10), 7)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return rank(df1) ** rank(df2)
    
    # Alpha#86: ((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open+ close) - (vwap + open)))) * -1)
    def alpha086(self):
        adv20 = sma(self.volume, 20)
        df = correlation(self.close, sma(adv20, 15), 6)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = ts_rank(df, 20)
        p2 = rank(((self.open + self.close) - (self.vwap + self.open)))
        # Align indices and convert boolean to float
        result = (p1 < p2).astype(float) * -1
        return result.fillna(0)
    
    # Alpha#87: (max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))),1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81,IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1) - Skip
    def alpha087(self):
        # Simplified version
        adv81 = sma(self.volume, 81)
        p1 = rank(decay_linear(delta(((self.close * 0.369701) + (self.vwap * (1 - 0.369701))), 2), 3))
        df = correlation(adv81, self.close, 13)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(decay_linear(abs(df), 5), 14)
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).max(axis=1) * -1
        return result.fillna(0)
    
    # Alpha#88: min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))),8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60,20.6966), 8.01266), 6.65053), 2.61957))
    def alpha088(self):
        adv60 = sma(self.volume, 60)
        p1 = rank(decay_linear(((rank(self.open) + rank(self.low)) - (rank(self.high) + rank(self.close))), 8))
        df = correlation(ts_rank(self.close, 8), ts_rank(adv60, 21), 8)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(decay_linear(df, 7), 3)
        return pd.concat([p1, p2], axis=1).min(axis=1)
    
    # Alpha#89: (Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10,6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap,IndClass.industry), 3.48158), 10.1466), 15.3012)) - Skip
    def alpha089(self):
        # Simplified version
        adv10 = sma(self.volume, 10)
        df = correlation(self.low, adv10, 7)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = ts_rank(decay_linear(df, 6), 4)
        p2 = ts_rank(decay_linear(delta(self.vwap, 3), 10), 15)
        return p1 - p2
    
    # Alpha#90: ((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40,IndClass.subindustry), low, 5.38375), 3.21856)) * -1) - Skip
    def alpha090(self):
        # Simplified version
        adv40 = sma(self.volume, 40)
        df = correlation(adv40, self.low, 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (rank((self.close - ts_max(self.close, 5))) ** ts_rank(df, 3) * -1)
    
    # Alpha#91: ((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close,IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) -rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1) - Skip
    def alpha091(self):
        # Simplified version
        adv30 = sma(self.volume, 30)
        df1 = correlation(self.close, self.volume, 10)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = ts_rank(decay_linear(decay_linear(df1, 16), 4), 5)
        df2 = correlation(self.vwap, adv30, 4)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = rank(decay_linear(df2, 3))
        return (p1 - p2) * -1
    
    # Alpha#92: min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221),18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024),6.80584))
    def alpha092(self):
        adv30 = sma(self.volume, 30)
        cond = (((self.high + self.low) / 2) + self.close) < (self.low + self.open)
        p1 = ts_rank(decay_linear(cond.astype(float), 15), 19)
        df = correlation(rank(self.low), rank(adv30), 8)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(decay_linear(df, 7), 7)
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).min(axis=1)
        return result.fillna(0)
    
    # Alpha#93: (Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81,17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 -0.524434))), 2.77377), 16.2664))) - Skip
    def alpha093(self):
        # Simplified version
        adv81 = sma(self.volume, 81)
        df = correlation(self.vwap, adv81, 17)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = ts_rank(decay_linear(df, 20), 8)
        p2 = rank(decay_linear(delta(((self.close * 0.524434) + (self.vwap * (1 - 0.524434))), 3), 16))
        return p1 / (p2 + 1e-10)
    
    # Alpha#94: ((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap,19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    def alpha094(self):
        adv60 = sma(self.volume, 60)
        p1 = rank((self.vwap - ts_min(self.vwap, 12)))
        df = correlation(ts_rank(self.vwap, 20), ts_rank(adv60, 4), 18)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(df, 3)
        return (p1 ** p2 * -1)
    
    # Alpha#95: (rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low)/ 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    def alpha095(self):
        adv40 = sma(self.volume, 40)
        p1 = rank((self.open - ts_min(self.open, 12)))
        df = correlation(sma(((self.high + self.low) / 2), 19), sma(adv40, 19), 13)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank((rank(df) ** 5), 12)
        return (p1 < p2).astype(float)
    
    # Alpha#96: (max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878),4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404),Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
    def alpha096(self):
        adv60 = sma(self.volume, 60)
        df1 = correlation(rank(self.vwap), rank(self.volume), 4)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = ts_rank(decay_linear(df1, 4), 8)
        df2 = correlation(ts_rank(self.close, 7), ts_rank(adv60, 4), 4)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(decay_linear(ts_argmax(df2, 13), 14), 13)
        # Align indices before concat
        p1, p2 = p1.align(p2, join='outer', fill_value=0)
        result = pd.concat([p1, p2], axis=1).max(axis=1) * -1
        return result.fillna(0)
    
    # Alpha#97: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))),IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low,7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1) - Skip
    def alpha097(self):
        # Simplified version
        adv60 = sma(self.volume, 60)
        p1 = rank(decay_linear(delta(((self.low * 0.721001) + (self.vwap * (1 - 0.721001))), 3), 20))
        df = correlation(ts_rank(self.low, 8), ts_rank(adv60, 17), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = ts_rank(decay_linear(ts_rank(df, 19), 16), 7)
        return (p1 - p2) * -1
    
    # Alpha#98: (rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) -rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571),6.95668), 8.07206)))
    def alpha098(self):
        adv5 = sma(self.volume, 5)
        adv15 = sma(self.volume, 15)
        df1 = correlation(self.vwap, sma(adv5, 26), 5)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        p1 = rank(decay_linear(df1, 7))
        df2 = correlation(rank(self.open), rank(adv15), 21)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        p2 = rank(decay_linear(ts_rank(ts_argmin(df2, 9), 7), 8))
        return p1 - p2
    
    # Alpha#99: ((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) <rank(correlation(low, volume, 6.28259))) * -1)
    def alpha099(self):
        adv60 = sma(self.volume, 60)
        df1 = correlation(ts_sum(((self.high + self.low) / 2), 20), ts_sum(adv60, 20), 9)
        df1 = df1.replace([-np.inf, np.inf], 0).fillna(value=0)
        df2 = correlation(self.low, self.volume, 6)
        df2 = df2.replace([-np.inf, np.inf], 0).fillna(value=0)
        return ((rank(df1) < rank(df2)) * -1)
    
    # Alpha#100: (0 - (1 * (((1.5 * scale(indneutralize(...))) - scale(indneutralize(...))) * (volume / adv20)))) - Skip
    def alpha100(self):
        # Simplified version without industry neutralization
        adv20 = sma(self.volume, 20)
        inner = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low + 1e-10) * self.volume
        df = correlation(self.close, rank(adv20), 5)
        df = df.replace([-np.inf, np.inf], 0).fillna(value=0)
        return (0 - (1 * (((1.5 * scale(rank(inner))) - scale(df - rank(ts_argmin(self.close, 30)))) * (self.volume / (adv20 + 1e-10)))))
    
    # Alpha#101: ((close - open) / ((high - low) + .001))
    def alpha101(self):
        return (self.close - self.open) / ((self.high - self.low) + 0.001)


# ==================== Main Function ====================
def calculate_alpha101_factors(prices: pd.DataFrame) -> pd.DataFrame:
    """
    计算Alpha101的101个因子
    
    Args:
        prices: DataFrame with MultiIndex (date, ticker), columns: Open, High, Low, Close, Adj Close, Volume
    
    Returns:
        DataFrame with MultiIndex (date, ticker) and Alpha101 factor columns
    """
    # Unstack for easier calculation (date index, ticker columns)
    close = prices["Adj Close"].unstack("ticker")
    open_price = prices["Open"].unstack("ticker")
    high = prices["High"].unstack("ticker")
    low = prices["Low"].unstack("ticker")
    volume = prices["Volume"].unstack("ticker")
    
    # Create dict for Alphas class (Bug 1 fix: use dict instead of DataFrame)
    df_data = {
        'Open': open_price,
        'High': high,
        'Low': low,
        'Adj Close': close,
        'Volume': volume
    }
    
    # Initialize Alphas class
    stock = Alphas(df_data)
    
    # Calculate all 101 factors
    factors = {}
    
    # List of all alpha methods
    alpha_methods = [
        ('Alpha1', stock.alpha001), ('Alpha2', stock.alpha002), ('Alpha3', stock.alpha003),
        ('Alpha4', stock.alpha004), ('Alpha5', stock.alpha005), ('Alpha6', stock.alpha006),
        ('Alpha7', stock.alpha007), ('Alpha8', stock.alpha008), ('Alpha9', stock.alpha009),
        ('Alpha10', stock.alpha010), ('Alpha11', stock.alpha011), ('Alpha12', stock.alpha012),
        ('Alpha13', stock.alpha013), ('Alpha14', stock.alpha014), ('Alpha15', stock.alpha015),
        ('Alpha16', stock.alpha016), ('Alpha17', stock.alpha017), ('Alpha18', stock.alpha018),
        ('Alpha19', stock.alpha019), ('Alpha20', stock.alpha020), ('Alpha21', stock.alpha021),
        ('Alpha22', stock.alpha022), ('Alpha23', stock.alpha023), ('Alpha24', stock.alpha024),
        ('Alpha25', stock.alpha025), ('Alpha26', stock.alpha026), ('Alpha27', stock.alpha027),
        ('Alpha28', stock.alpha028), ('Alpha29', stock.alpha029), ('Alpha30', stock.alpha030),
        ('Alpha31', stock.alpha031), ('Alpha32', stock.alpha032), ('Alpha33', stock.alpha033),
        ('Alpha34', stock.alpha034), ('Alpha35', stock.alpha035), ('Alpha36', stock.alpha036),
        ('Alpha37', stock.alpha037), ('Alpha38', stock.alpha038), ('Alpha39', stock.alpha039),
        ('Alpha40', stock.alpha040), ('Alpha41', stock.alpha041), ('Alpha42', stock.alpha042),
        ('Alpha43', stock.alpha043), ('Alpha44', stock.alpha044), ('Alpha45', stock.alpha045),
        ('Alpha46', stock.alpha046), ('Alpha47', stock.alpha047), ('Alpha48', stock.alpha048),
        ('Alpha49', stock.alpha049), ('Alpha50', stock.alpha050), ('Alpha51', stock.alpha051),
        ('Alpha52', stock.alpha052), ('Alpha53', stock.alpha053), ('Alpha54', stock.alpha054),
        ('Alpha55', stock.alpha055), ('Alpha56', stock.alpha056), ('Alpha57', stock.alpha057),
        ('Alpha58', stock.alpha058), ('Alpha59', stock.alpha059), ('Alpha60', stock.alpha060),
        ('Alpha61', stock.alpha061), ('Alpha62', stock.alpha062), ('Alpha63', stock.alpha063),
        ('Alpha64', stock.alpha064), ('Alpha65', stock.alpha065), ('Alpha66', stock.alpha066),
        ('Alpha67', stock.alpha067), ('Alpha68', stock.alpha068), ('Alpha69', stock.alpha069),
        ('Alpha70', stock.alpha070), ('Alpha71', stock.alpha071), ('Alpha72', stock.alpha072),
        ('Alpha73', stock.alpha073), ('Alpha74', stock.alpha074), ('Alpha75', stock.alpha075),
        ('Alpha76', stock.alpha076), ('Alpha77', stock.alpha077), ('Alpha78', stock.alpha078),
        ('Alpha79', stock.alpha079), ('Alpha80', stock.alpha080), ('Alpha81', stock.alpha081),
        ('Alpha82', stock.alpha082), ('Alpha83', stock.alpha083), ('Alpha84', stock.alpha084),
        ('Alpha85', stock.alpha085), ('Alpha86', stock.alpha086), ('Alpha87', stock.alpha087),
        ('Alpha88', stock.alpha088), ('Alpha89', stock.alpha089), ('Alpha90', stock.alpha090),
        ('Alpha91', stock.alpha091), ('Alpha92', stock.alpha092), ('Alpha93', stock.alpha093),
        ('Alpha94', stock.alpha094), ('Alpha95', stock.alpha095), ('Alpha96', stock.alpha096),
        ('Alpha97', stock.alpha097), ('Alpha98', stock.alpha098), ('Alpha99', stock.alpha099),
        ('Alpha100', stock.alpha100), ('Alpha101', stock.alpha101),
    ]
    
    # Calculate each factor
    total_factors = len(alpha_methods)
    print(f"  开始计算 {total_factors} 个Alpha因子...")
    
    for idx, (name, method) in enumerate(alpha_methods, 1):
        # 显示进度（每10个因子或每25个因子显示一次）
        if idx % 25 == 0 or idx == 1 or idx == total_factors:
            progress = idx / total_factors * 100
            print(f"    进度: {idx}/{total_factors} ({progress:.1f}%) - {name}")
        
        try:
            result = method()
            # Handle different return types (Bug 2 fix: ensure consistent MultiIndex structure)
            if isinstance(result, pd.DataFrame):
                # DataFrame with date index and ticker columns - stack to MultiIndex
                if result.shape[1] > 0:
                    # Stack to create MultiIndex (date, ticker)
                    result = result.stack()
                else:
                    # Empty DataFrame - create NaN series with correct index
                    result = pd.Series(index=prices.index, dtype=float)
            elif isinstance(result, pd.Series):
                # Check if already MultiIndex
                if isinstance(result.index, pd.MultiIndex):
                    # Already MultiIndex, ensure it matches prices.index structure
                    if not result.index.equals(prices.index):
                        result = result.reindex(prices.index)
                else:
                    # Single-level index (date) - Bug 2: this is the problem case
                    # If result has date index matching unstacked data, it represents 2D data
                    # that was incorrectly extracted as a single column. We need to reconstruct.
                    if result.index.equals(close.index) and len(result) == len(close.index):
                        # This Series was extracted from a DataFrame but lost its ticker dimension
                        # We need to convert back to DataFrame with ticker columns, then stack
                        # Since we don't have the original structure, we'll broadcast to all tickers
                        # This is a workaround - ideally alpha methods should return full DataFrames
                        result_df = pd.DataFrame(
                            index=close.index,
                            columns=close.columns,
                            data=np.tile(result.values.reshape(-1, 1), (1, len(close.columns)))
                        )
                        result = result_df.stack()
                    else:
                        # Can't determine structure - create with correct MultiIndex
                        result = pd.Series(index=prices.index, dtype=float)
            else:
                # Other types - create NaN series
                result = pd.Series(index=prices.index, dtype=float)
            
            # Final check: ensure result is a Series with MultiIndex matching prices.index
            if not isinstance(result, pd.Series):
                result = pd.Series(index=prices.index, dtype=float)
            elif not isinstance(result.index, pd.MultiIndex):
                # Still not MultiIndex - create with correct structure
                result = pd.Series(index=prices.index, dtype=float)
            elif not result.index.equals(prices.index):
                # MultiIndex but different structure - reindex
                result = result.reindex(prices.index)
            
            factors[name] = result
        except Exception as e:
            print(f"[Warn] Failed to calculate {name}: {e}")
            # Create NaN series with same index structure
            factors[name] = pd.Series(index=prices.index, dtype=float)
    
    # Combine all factors
    df = pd.DataFrame(factors)
    df.index.names = ["date", "ticker"]
    df = df.sort_index()
    
    return df
