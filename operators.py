"""
operators.py - 因子计算算子模块
将FactorUtils的所有静态方法暴露为模块级函数，供factor文件通过 from panda_factor.operators import * 使用
"""
import numpy as np
import pandas as pd
from panda_factor.generate.factor_utils import FactorUtils

# --- 时间序列算子 ---
def DELAY(series, period=1): return FactorUtils.DELAY(series, period)
def DELTA(series, period=1): return FactorUtils.DELTA(series, period)
def DIFF(S, N=1): return FactorUtils.DIFF(S, N)
def REF(S, N=1): return FactorUtils.REF(S, N)
def SUM(series, window=20): return FactorUtils.SUM(series, window)
def PRODUCT(series, window=20): return FactorUtils.PRODUCT(series, window)

# --- 移动平均 ---
def MA(S, N): return FactorUtils.MA(S, N)
def EMA(S, N): return FactorUtils.EMA(S, N)
def SMA(S, N, M=1): return FactorUtils.SMA(S, N, M)
def WMA(S, N): return FactorUtils.WMA(S, N)
def DMA(S, A): return FactorUtils.DMA(S, A)
def DECAYLINEAR(S, d): return FactorUtils.DECAYLINEAR(S, d)
def DECAY_LINEAR(series, window=20): return FactorUtils.DECAY_LINEAR(series, window)

# --- 统计算子 ---
def STD(S, N): return FactorUtils.STD(S, N)
def STDDEV(series, window=20): return FactorUtils.STDDEV(series, window)
def CORRELATION(series1, series2, window=20): return FactorUtils.CORRELATION(series1, series2, window)
def COVARIANCE(series1, series2, window=20): return FactorUtils.COVARIANCE(series1, series2, window)
def AVEDEV(S, N): return FactorUtils.AVEDEV(S, N)
def SLOPE(S, N): return FactorUtils.SLOPE(S, N)

# --- 极值算子 ---
def MAX(series1, series2): return FactorUtils.MAX(series1, series2)
def MIN(series1, series2): return FactorUtils.MIN(series1, series2)
def HHV(S, N): return FactorUtils.HHV(S, N)
def LLV(S, N): return FactorUtils.LLV(S, N)
def HHVBARS(S, N): return FactorUtils.HHVBARS(S, N)
def LLVBARS(S, N): return FactorUtils.LLVBARS(S, N)
def TS_MAX(series, window=20): return FactorUtils.TS_MAX(series, window)
def TS_MIN(series, window=20): return FactorUtils.TS_MIN(series, window)
def TS_ARGMAX(series, window): return FactorUtils.TS_ARGMAX(series, window)
def TS_ARGMIN(series, window=20): return FactorUtils.TS_ARGMIN(series, window)
def TS_RANK(series, window=20): return FactorUtils.TS_RANK(series, window)
def TS_MEAN(series, window=20): return FactorUtils.TS_MEAN(series, window)

# --- 数学算子 ---
def ABS(series): return FactorUtils.ABS(series)
def LOG(series): return FactorUtils.LOG(series)
def SIGN(S): return FactorUtils.SIGN(S)
def SIGNEDPOWER(S, n): return FactorUtils.SIGNEDPOWER(S, n)
def POWER(series, power): return FactorUtils.POWER(series, power)
def RANK(series): return FactorUtils.RANK(series)
def SCALE(series): return FactorUtils.SCALE(series)
def RD(S, D=3): return FactorUtils.RD(S, D)

# --- 条件算子 ---
def IF(condition, true_value, false_value): return FactorUtils.IF(condition, true_value, false_value)
def COUNT(S, N): return FactorUtils.COUNT(S, N)
def EVERY(S, N): return FactorUtils.EVERY(S, N)
def EXIST(S, N): return FactorUtils.EXIST(S, N)
def SUMIF(S1, S2, N): return FactorUtils.SUMIF(S1, S2, N)
def CROSS(S1, S2): return FactorUtils.CROSS(S1, S2)
def LONGCROSS(S1, S2, N): return FactorUtils.LONGCROSS(S1, S2, N)
def BARSLAST(S): return FactorUtils.BARSLAST(S)
def BARSLASTCOUNT(S): return FactorUtils.BARSLASTCOUNT(S)

# --- 收益率算子 ---
def RETURNS(close, period=1): return FactorUtils.RETURNS(close, period)
def FUTURE_RETURNS(close, period=1): return FactorUtils.FUTURE_RETURNS(close, period)

# --- 成交量/价格算子 ---
def ADV(volume, window=20): return FactorUtils.ADV(volume, window)

def VWAP(close, volume, window=20):
    """Volume Weighted Average Price over rolling window"""
    close, volume = close.align(volume)
    result = pd.Series(index=close.index, dtype=float)
    for symbol in close.index.get_level_values('symbol').unique():
        mask = close.index.get_level_values('symbol') == symbol
        sc = close[mask]
        sv = volume[mask]
        pv_sum = (sc * sv).rolling(window=window, min_periods=1).sum()
        v_sum = sv.rolling(window=window, min_periods=1).sum()
        result[mask] = pv_sum / v_sum.replace(0, np.nan)
    return result

def OBV(close, volume): return FactorUtils.OBV(close, volume)
def MFI(high, low, close, volume, N=14): return FactorUtils.MFI(close, high, low, volume, N)

def CMF(high, low, close, volume, N=20):
    """Chaikin Money Flow"""
    clv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    mf = clv * volume
    cmf = mf.groupby(level='symbol').rolling(window=N, min_periods=1).sum().droplevel(0) / \
          volume.groupby(level='symbol').rolling(window=N, min_periods=1).sum().droplevel(0).replace(0, np.nan)
    return cmf

def VPT(close, volume):
    """Volume Price Trend"""
    ret = close.groupby(level='symbol').pct_change().fillna(0)
    vpt = (ret * volume).groupby(level='symbol').cumsum()
    return vpt

# --- 布林带算子 ---
def BOLL_UPPER(close, N=20, P=2):
    """Bollinger Band Upper"""
    mid = close.groupby(level='symbol').transform(lambda x: x.rolling(N, min_periods=1).mean())
    std = close.groupby(level='symbol').transform(lambda x: x.rolling(N, min_periods=1).std())
    return mid + P * std

def BOLL_LOWER(close, N=20, P=2):
    """Bollinger Band Lower"""
    mid = close.groupby(level='symbol').transform(lambda x: x.rolling(N, min_periods=1).mean())
    std = close.groupby(level='symbol').transform(lambda x: x.rolling(N, min_periods=1).std())
    return mid - P * std

def BOLL(close, N=20, P=2): return FactorUtils.BOLL(close, N, P)

# --- 技术指标 ---
def RSI(close, N=24): return FactorUtils.RSI(close, N)
def KDJ(close, high, low, N=9, M1=3, M2=3): return FactorUtils.KDJ(close, high, low, N, M1, M2)
def MACD(close, SHORT=12, LONG=26, M=9): return FactorUtils.MACD(close, SHORT, LONG, M)
def CCI(close, high, low, N=14): return FactorUtils.CCI(close, high, low, N)
def ATR(close, high, low, N=20): return FactorUtils.ATR(close, high, low, N)
def BIAS(close, L1=6, L2=12, L3=24): return FactorUtils.BIAS(close, L1, L2, L3)
def ROC(close, N=12): return FactorUtils.ROC(close, N)
def EXPMA(close, N1=12, N2=50): return FactorUtils.EXPMA(close, N1, N2)

__all__ = [
    'DELAY', 'DELTA', 'DIFF', 'REF', 'SUM', 'PRODUCT',
    'MA', 'EMA', 'SMA', 'WMA', 'DMA', 'DECAYLINEAR', 'DECAY_LINEAR',
    'STD', 'STDDEV', 'CORRELATION', 'COVARIANCE', 'AVEDEV', 'SLOPE',
    'MAX', 'MIN', 'HHV', 'LLV', 'HHVBARS', 'LLVBARS',
    'TS_MAX', 'TS_MIN', 'TS_ARGMAX', 'TS_ARGMIN', 'TS_RANK', 'TS_MEAN',
    'ABS', 'LOG', 'SIGN', 'SIGNEDPOWER', 'POWER', 'RANK', 'SCALE', 'RD',
    'IF', 'COUNT', 'EVERY', 'EXIST', 'SUMIF', 'CROSS', 'LONGCROSS',
    'BARSLAST', 'BARSLASTCOUNT',
    'RETURNS', 'FUTURE_RETURNS', 'ADV',
    'VWAP', 'OBV', 'MFI', 'CMF', 'VPT',
    'BOLL_UPPER', 'BOLL_LOWER', 'BOLL',
    'RSI', 'KDJ', 'MACD', 'CCI', 'ATR', 'BIAS', 'ROC', 'EXPMA',
]
