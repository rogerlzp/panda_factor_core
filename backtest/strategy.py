"""
策略模块

提供策略基类和具体策略实现
"""

import pandas as pd
from datetime import datetime
from typing import Dict, List


class BaseStrategy:
    """策略基类"""
    
    def __init__(self, name: str = 'BaseStrategy'):
        """
        初始化策略
        
        Args:
            name: 策略名称
        """
        self.name = name
    
    def select_stocks(self, df_day: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        选股逻辑
        
        Args:
            df_day: 当日所有股票数据
            date: 当前日期（YYYYMMDD格式）
            
        Returns:
            DataFrame: 选中的股票
        """
        raise NotImplementedError("子类必须实现select_stocks方法")
    
    def should_sell(self, position: Dict, current_date: str) -> bool:
        """
        判断是否应该卖出
        
        Args:
            position: 持仓信息，包含：
                - symbol: 股票代码
                - shares: 持仓数量
                - buy_price: 买入价格
                - buy_date: 买入日期
                - cost: 买入成本
            current_date: 当前日期（YYYYMMDD格式）
            
        Returns:
            bool: 是否卖出
        """
        raise NotImplementedError("子类必须实现should_sell方法")
    
    def get_position_ratio(self, stock: pd.Series) -> float:
        """
        获取仓位比例（用于高开处理）
        
        Args:
            stock: 股票数据
            
        Returns:
            float: 仓位比例（0-1）
        """
        return 1.0  # 默认满仓


class MultiFactorStrategy(BaseStrategy):
    """多因子策略"""
    
    def __init__(
        self,
        factors: List[str],
        weights: Dict[str, float],
        top_n: int = 3,
        holding_period: int = 3,
        min_position: float = 0.9,
        min_price: float = 5.0,
        min_amount: float = 5000000,
        min_market_cap: float = 50e8,
        max_market_cap: float = 200e8,
        min_turnover: float = 10.0,
        max_open_return: float = 0.07,
        half_position_open: float = 0.03
    ):
        """
        初始化多因子策略
        
        Args:
            factors: 因子列表
            weights: 因子权重字典，如 {'price_pos_20d': 0.3, 'mom_20d': 0.2}
            top_n: 选股数量
            holding_period: 持有天数
            min_position: 最小价格位置（0-1）
            min_price: 最低价格
            min_amount: 最小成交额
            min_market_cap: 最小市值
            max_market_cap: 最大市值
            min_turnover: 最小换手率（%）
            max_open_return: 最大高开幅度，超过此值不买入
            half_position_open: 半仓高开幅度，超过此值半仓买入
        """
        super().__init__(name='MultiFactorStrategy')
        self.factors = factors
        self.weights = weights
        self.top_n = top_n
        self.holding_period = holding_period
        self.min_position = min_position
        self.min_price = min_price
        self.min_amount = min_amount
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_turnover = min_turnover
        self.max_open_return = max_open_return
        self.half_position_open = half_position_open
    
    def select_stocks(self, df_day: pd.DataFrame, date: str) -> pd.DataFrame:
        """
        多因子选股
        
        流程：
        1. 基础筛选（价格位置、市值、换手率等）
        2. 因子标准化（Min-Max归一化）
        3. 加权打分
        4. 排序并选择Top N
        """
        # 1. 基础筛选（沿用V67）
        conditions = (
            (df_day['price_position'].notna()) &
            (df_day['price_position'] >= self.min_position) &
            (df_day['close'] >= self.min_price) &
            (df_day['amount'] >= self.min_amount)
        )
        
        # 市值筛选（如果有market_cap列）
        if 'market_cap' in df_day.columns:
            conditions = conditions & (
                (df_day['market_cap'] >= self.min_market_cap) &
                (df_day['market_cap'] < self.max_market_cap)
            )
        
        # 换手率筛选（如果有turnover列）
        if 'turnover' in df_day.columns:
            conditions = conditions & (df_day['turnover'] >= self.min_turnover)
        
        df_filtered = df_day[conditions].copy()
        
        if len(df_filtered) == 0:
            return pd.DataFrame()
        
        # 2. 标准化因子值（Min-Max归一化到[0, 1]）
        for factor in self.factors:
            if factor not in df_filtered.columns:
                print(f"⚠️  因子 {factor} 不存在，跳过")
                continue
            
            values = df_filtered[factor]
            min_val = values.min()
            max_val = values.max()
            
            if max_val - min_val > 1e-8:
                df_filtered[f'{factor}_norm'] = (values - min_val) / (max_val - min_val)
            else:
                # 所有值相同，赋予中性分数
                df_filtered[f'{factor}_norm'] = 0.5
        
        # 3. 加权打分
        df_filtered['score'] = 0.0
        for factor, weight in self.weights.items():
            norm_col = f'{factor}_norm'
            if norm_col in df_filtered.columns:
                df_filtered['score'] += weight * df_filtered[norm_col]
        
        # 4. 排序并选择Top N
        df_selected = df_filtered.nlargest(self.top_n, 'score')
        
        return df_selected
    
    def should_sell(self, position: Dict, current_date: str) -> bool:
        """
        持有期满卖出
        
        Args:
            position: 持仓信息
            current_date: 当前日期（YYYYMMDD格式）
            
        Returns:
            bool: 是否卖出
        """
        buy_date = datetime.strptime(str(position['buy_date']), '%Y%m%d')
        current = datetime.strptime(str(current_date), '%Y%m%d')
        
        holding_days = (current - buy_date).days
        
        return holding_days >= self.holding_period
    
    def get_position_ratio(self, stock: pd.Series) -> float:
        """
        高开处理
        
        根据高开幅度决定仓位：
        - 高开 > max_open_return: 不买入（返回0）
        - 高开 > half_position_open: 半仓买入（返回0.5）
        - 其他: 满仓买入（返回1.0）
        
        Args:
            stock: 股票数据，需要包含open和pre_close列
            
        Returns:
            float: 仓位比例（0, 0.5, 或 1.0）
        """
        if 'open' not in stock or 'pre_close' not in stock:
            return 1.0
        
        # 计算高开幅度
        open_return = (stock['open'] - stock['pre_close']) / stock['pre_close']
        
        if open_return > self.max_open_return:
            return 0.0  # 高开过高，不买入
        elif open_return > self.half_position_open:
            return 0.5  # 半仓
        else:
            return 1.0  # 满仓


class V67Strategy(BaseStrategy):
    """
    V67策略（用于对比）
    
    单因子策略：只使用价格位置因子
    """
    
    def __init__(
        self,
        top_n: int = 3,
        holding_period: int = 3,
        min_position: float = 0.9,
        min_price: float = 5.0,
        min_amount: float = 5000000,
        min_market_cap: float = 50e8,
        max_market_cap: float = 200e8,
        min_turnover: float = 10.0,
        max_open_return: float = 0.07,
        half_position_open: float = 0.03
    ):
        """初始化V67策略"""
        super().__init__(name='V67Strategy')
        self.top_n = top_n
        self.holding_period = holding_period
        self.min_position = min_position
        self.min_price = min_price
        self.min_amount = min_amount
        self.min_market_cap = min_market_cap
        self.max_market_cap = max_market_cap
        self.min_turnover = min_turnover
        self.max_open_return = max_open_return
        self.half_position_open = half_position_open
    
    def select_stocks(self, df_day: pd.DataFrame, date: str) -> pd.DataFrame:
        """V67选股：只按价格位置排序"""
        conditions = (
            (df_day['price_position'].notna()) &
            (df_day['price_position'] >= self.min_position) &
            (df_day['close'] >= self.min_price) &
            (df_day['amount'] >= self.min_amount)
        )
        
        if 'market_cap' in df_day.columns:
            conditions = conditions & (
                (df_day['market_cap'] >= self.min_market_cap) &
                (df_day['market_cap'] < self.max_market_cap)
            )
        
        if 'turnover' in df_day.columns:
            conditions = conditions & (df_day['turnover'] >= self.min_turnover)
        
        df_filtered = df_day[conditions].copy()
        
        if len(df_filtered) == 0:
            return pd.DataFrame()
        
        # 按价格位置排序
        df_selected = df_filtered.nlargest(self.top_n, 'price_position')
        
        return df_selected
    
    def should_sell(self, position: Dict, current_date: str) -> bool:
        """持有期满卖出"""
        buy_date = datetime.strptime(str(position['buy_date']), '%Y%m%d')
        current = datetime.strptime(str(current_date), '%Y%m%d')
        
        holding_days = (current - buy_date).days
        
        return holding_days >= self.holding_period
    
    def get_position_ratio(self, stock: pd.Series) -> float:
        """高开处理"""
        if 'open' not in stock or 'pre_close' not in stock:
            return 1.0
        
        open_return = (stock['open'] - stock['pre_close']) / stock['pre_close']
        
        if open_return > self.max_open_return:
            return 0.0
        elif open_return > self.half_position_open:
            return 0.5
        else:
            return 1.0
