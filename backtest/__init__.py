"""
回测模块

提供通用的回测框架，支持多种策略回测
"""

from .base import BaseBacktest
from .strategy import BaseStrategy, MultiFactorStrategy
from .metrics import (
    calculate_annual_return,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_win_rate,
    calculate_profit_loss_ratio
)

__all__ = [
    'BaseBacktest',
    'BaseStrategy',
    'MultiFactorStrategy',
    'calculate_annual_return',
    'calculate_max_drawdown',
    'calculate_sharpe_ratio',
    'calculate_sortino_ratio',
    'calculate_calmar_ratio',
    'calculate_win_rate',
    'calculate_profit_loss_ratio',
]
