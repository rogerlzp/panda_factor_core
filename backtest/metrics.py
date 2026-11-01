"""
回测指标计算模块

提供各种回测性能指标的计算函数
"""

import pandas as pd
import numpy as np
from datetime import datetime


def calculate_annual_return(pv_df: pd.DataFrame) -> float:
    """
    计算年化收益率
    
    Args:
        pv_df: 净值曲线DataFrame，包含date和total_value列
        
    Returns:
        float: 年化收益率
    """
    if len(pv_df) < 2:
        return 0.0
    
    # 转换日期
    if isinstance(pv_df['date'].iloc[0], str):
        start_date = datetime.strptime(str(pv_df['date'].iloc[0]), '%Y%m%d')
        end_date = datetime.strptime(str(pv_df['date'].iloc[-1]), '%Y%m%d')
    else:
        start_date = pv_df['date'].iloc[0]
        end_date = pv_df['date'].iloc[-1]
    
    total_days = (end_date - start_date).days
    
    if total_days == 0:
        return 0.0
    
    total_return = (pv_df['total_value'].iloc[-1] - pv_df['total_value'].iloc[0]) / pv_df['total_value'].iloc[0]
    
    annual_return = (1 + total_return) ** (365 / total_days) - 1
    
    return annual_return


def calculate_max_drawdown(pv_df: pd.DataFrame) -> float:
    """
    计算最大回撤
    
    Args:
        pv_df: 净值曲线DataFrame，包含total_value列
        
    Returns:
        float: 最大回撤（正数）
    """
    if len(pv_df) < 2:
        return 0.0
    
    values = pv_df['total_value'].values
    cummax = np.maximum.accumulate(values)
    drawdown = (values - cummax) / cummax
    
    max_drawdown = drawdown.min()
    
    return abs(max_drawdown)


def calculate_sharpe_ratio(pv_df: pd.DataFrame, risk_free_rate: float = 0.03) -> float:
    """
    计算夏普比率
    
    Args:
        pv_df: 净值曲线DataFrame，包含total_value列
        risk_free_rate: 年化无风险收益率
        
    Returns:
        float: 夏普比率
    """
    if len(pv_df) < 2:
        return 0.0
    
    returns = pv_df['total_value'].pct_change().dropna()
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252  # 日度无风险收益
    
    if excess_returns.std() == 0:
        return 0.0
    
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
    
    return sharpe


def calculate_sortino_ratio(pv_df: pd.DataFrame, risk_free_rate: float = 0.03) -> float:
    """
    计算索提诺比率（只考虑下行风险）
    
    Args:
        pv_df: 净值曲线DataFrame，包含total_value列
        risk_free_rate: 年化无风险收益率
        
    Returns:
        float: 索提诺比率
    """
    if len(pv_df) < 2:
        return 0.0
    
    returns = pv_df['total_value'].pct_change().dropna()
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / 252
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    sortino = excess_returns.mean() / downside_returns.std() * np.sqrt(252)
    
    return sortino


def calculate_calmar_ratio(pv_df: pd.DataFrame) -> float:
    """
    计算卡玛比率（年化收益/最大回撤）
    
    Args:
        pv_df: 净值曲线DataFrame
        
    Returns:
        float: 卡玛比率
    """
    annual_return = calculate_annual_return(pv_df)
    max_dd = calculate_max_drawdown(pv_df)
    
    if max_dd == 0:
        return 0.0
    
    calmar = annual_return / max_dd
    
    return calmar


def calculate_win_rate(trades_df: pd.DataFrame) -> float:
    """
    计算胜率
    
    Args:
        trades_df: 交易记录DataFrame，包含action和profit列
        
    Returns:
        float: 胜率（0-1）
    """
    if len(trades_df) == 0:
        return 0.0
    
    sell_trades = trades_df[trades_df['action'] == 'sell']
    
    if len(sell_trades) == 0:
        return 0.0
    
    win_trades = sell_trades[sell_trades['profit'] > 0]
    win_rate = len(win_trades) / len(sell_trades)
    
    return win_rate


def calculate_profit_loss_ratio(trades_df: pd.DataFrame) -> float:
    """
    计算盈亏比（平均盈利/平均亏损）
    
    Args:
        trades_df: 交易记录DataFrame，包含action和profit列
        
    Returns:
        float: 盈亏比
    """
    if len(trades_df) == 0:
        return 0.0
    
    sell_trades = trades_df[trades_df['action'] == 'sell']
    
    if len(sell_trades) == 0:
        return 0.0
    
    win_trades = sell_trades[sell_trades['profit'] > 0]
    loss_trades = sell_trades[sell_trades['profit'] < 0]
    
    if len(loss_trades) == 0:
        return 0.0
    
    avg_win = win_trades['profit'].mean() if len(win_trades) > 0 else 0
    avg_loss = abs(loss_trades['profit'].mean())
    
    if avg_loss == 0:
        return 0.0
    
    pl_ratio = avg_win / avg_loss
    
    return pl_ratio


def calculate_all_metrics(pv_df: pd.DataFrame, trades_df: pd.DataFrame, initial_capital: float) -> dict:
    """
    计算所有回测指标
    
    Args:
        pv_df: 净值曲线DataFrame
        trades_df: 交易记录DataFrame
        initial_capital: 初始资金
        
    Returns:
        dict: 所有指标的字典
    """
    if len(pv_df) == 0:
        return {
            'initial_capital': initial_capital,
            'final_value': initial_capital,
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'total_trades': 0
        }
    
    final_value = pv_df['total_value'].iloc[-1]
    total_return = (final_value - initial_capital) / initial_capital
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'total_return': total_return,
        'annual_return': calculate_annual_return(pv_df),
        'max_drawdown': calculate_max_drawdown(pv_df),
        'sharpe_ratio': calculate_sharpe_ratio(pv_df),
        'sortino_ratio': calculate_sortino_ratio(pv_df),
        'calmar_ratio': calculate_calmar_ratio(pv_df),
        'win_rate': calculate_win_rate(trades_df),
        'profit_loss_ratio': calculate_profit_loss_ratio(trades_df),
        'total_trades': len(trades_df[trades_df['action'] == 'sell'])
    }
