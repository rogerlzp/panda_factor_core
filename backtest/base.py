"""
回测基类模块

提供通用的回测引擎，支持多种策略
"""

import pandas as pd
from typing import Dict, Optional
from .metrics import calculate_all_metrics


class BaseBacktest:
    """通用回测引擎"""
    
    def __init__(self, initial_capital: float = 1000000, commission: float = 0.001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission: 手续费率（单边，买入和卖出各收一次）
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.cash = initial_capital
        self.positions = {}  # {symbol: Position}
        self.trades = []     # 交易记录
        self.portfolio_values = []  # 净值曲线
        self.blocked_t1_sells = 0
        self.blocked_limit_up_buys = 0
        self.blocked_limit_down_sells = 0

    def _get_limit_pct(self, symbol: str) -> float:
        """根据股票代码推断涨跌停幅度。"""
        symbol = str(symbol)
        if symbol.startswith(("688", "300", "301")):
            return 0.20
        return 0.10

    def _get_limit_prices(self, stock: pd.Series):
        """计算当日涨停/跌停价。"""
        if "pre_close" not in stock or pd.isna(stock["pre_close"]) or stock["pre_close"] <= 0:
            return None, None

        limit_pct = self._get_limit_pct(stock.get("symbol", ""))
        pre_close = float(stock["pre_close"])
        limit_up = round(pre_close * (1 + limit_pct), 2)
        limit_down = round(pre_close * (1 - limit_pct), 2)
        return limit_up, limit_down

    def _can_buy(self, stock: pd.Series) -> bool:
        """涨停时通常无法买入，采用开盘价是否触及涨停的保守判断。"""
        if "open" not in stock or pd.isna(stock["open"]):
            return True

        limit_up, _ = self._get_limit_prices(stock)
        if limit_up is None:
            return True

        return float(stock["open"]) < limit_up - 1e-8

    def _can_sell(self, stock: pd.Series) -> bool:
        """跌停时通常无法卖出，采用收盘价是否触及跌停的保守判断。"""
        if "close" not in stock or pd.isna(stock["close"]):
            return True

        _, limit_down = self._get_limit_prices(stock)
        if limit_down is None:
            return True

        return float(stock["close"]) > limit_down + 1e-8
        
    def run(
        self,
        df: pd.DataFrame,
        strategy,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict:
        """
        运行回测
        
        Args:
            df: 行情数据（包含因子），必须包含以下列：
                - date: 日期（YYYYMMDD格式字符串）
                - symbol: 股票代码
                - open: 开盘价
                - close: 收盘价
                - 其他因子列
            strategy: 策略对象（BaseStrategy子类实例）
            start_date: 开始日期（YYYYMMDD格式），None表示从最早日期开始
            end_date: 结束日期（YYYYMMDD格式），None表示到最晚日期
            
        Returns:
            dict: 回测结果，包含各项指标和净值曲线
        """
        # 1. 数据准备
        df = self._prepare_data(df, start_date, end_date)
        
        if len(df) == 0:
            print("⚠️  没有可用数据")
            return self._empty_result()
        
        dates = sorted(df['date'].unique())
        
        print(f"回测开始: {dates[0]} ~ {dates[-1]}, 共{len(dates)}个交易日")
        
        # 2. 按日期遍历
        for i, date in enumerate(dates):
            df_day = df[df['date'] == date]
            
            # 3. 卖出（先卖后买）
            self._process_sells(df_day, strategy, date)
            
            # 4. 买入
            self._process_buys(df_day, strategy, date)
            
            # 5. 记录净值
            self._record_portfolio_value(df_day, date)
            
            # 进度显示
            if (i + 1) % 50 == 0 or i == len(dates) - 1:
                print(f"  进度: {i+1}/{len(dates)} ({(i+1)/len(dates)*100:.1f}%)")
        
        # 6. 计算指标
        result = self.calculate_metrics()
        
        print(f"回测完成，共{result['total_trades']}笔交易")
        
        return result
    
    def _prepare_data(
        self,
        df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """准备回测数据"""
        df = df.copy()
        
        # 日期过滤
        if start_date:
            df = df[df['date'] >= start_date]
        if end_date:
            df = df[df['date'] <= end_date]
        
        # 排序
        df = df.sort_values(['date', 'symbol'])
        
        return df
    
    def _process_buys(self, df_day: pd.DataFrame, strategy, date: str):
        """处理买入"""
        # 1. 策略选股
        try:
            selected = strategy.select_stocks(df_day, date)
        except Exception as e:
            print(f"⚠️  {date} 选股失败: {e}")
            return
        
        if len(selected) == 0:
            return
        
        # 2. 计算每只股票的买入金额
        available_cash = self.cash
        buy_amount_per_stock = available_cash / len(selected)
        
        # 3. 执行买入
        for _, stock in selected.iterrows():
            symbol = stock['symbol']
            
            # 已持仓则跳过
            if symbol in self.positions:
                continue

            if not self._can_buy(stock):
                self.blocked_limit_up_buys += 1
                continue

            # 高开处理
            try:
                position_ratio = strategy.get_position_ratio(stock)
            except Exception as e:
                print(f"⚠️  {symbol} 仓位计算失败: {e}")
                position_ratio = 1.0
            
            if position_ratio == 0:
                continue  # 高开过高，不买入
            
            # 计算买入价格和数量
            buy_price = stock['open']
            buy_amount = buy_amount_per_stock * position_ratio
            shares = int(buy_amount / buy_price / 100) * 100  # 100股整数倍
            
            if shares == 0:
                continue
            
            # 扣除手续费
            cost = shares * buy_price * (1 + self.commission)
            
            if cost > self.cash:
                continue  # 资金不足
            
            # 更新持仓
            self.positions[symbol] = {
                'symbol': symbol,
                'shares': shares,
                'buy_price': buy_price,
                'buy_date': date,
                'cost': cost,
                'peak_price': buy_price
            }
            
            # 更新现金
            self.cash -= cost
            
            # 记录交易
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'buy',
                'price': buy_price,
                'shares': shares,
                'amount': cost
            })
    
    def _process_sells(self, df_day: pd.DataFrame, strategy, date: str):
        """处理卖出"""
        to_sell = []
        
        for symbol, position in self.positions.items():
            # T+1: 当天买入的仓位不能当天卖出
            if str(position.get('buy_date')) == str(date):
                self.blocked_t1_sells += 1
                continue

            stock_data = df_day[df_day['symbol'] == symbol]
            if len(stock_data) > 0:
                current_price = float(stock_data.iloc[0]['close'])
                if 'peak_price' in position:
                    if current_price > position['peak_price']:
                        position['peak_price'] = current_price
                else:
                    position['peak_price'] = max(current_price, float(position.get('buy_price', current_price)))

                stop_loss_pct = getattr(strategy, 'stop_loss_pct', None)
                trailing_stop_pct = getattr(strategy, 'trailing_stop_pct', None)
                if stop_loss_pct is not None and position.get('buy_price', 0) > 0:
                    ret = current_price / float(position['buy_price']) - 1.0
                    if ret <= -float(stop_loss_pct):
                        to_sell.append(symbol)
                        continue
                if trailing_stop_pct is not None and position.get('peak_price', 0) > 0:
                    trail = current_price / float(position['peak_price']) - 1.0
                    if trail <= -float(trailing_stop_pct):
                        to_sell.append(symbol)
                        continue

            # 检查是否应该卖出
            try:
                if strategy.should_sell(position, date):
                    to_sell.append(symbol)
            except Exception as e:
                print(f"⚠️  {symbol} 卖出判断失败: {e}")
        
        # 执行卖出
        for symbol in to_sell:
            position = self.positions[symbol]
            
            # 获取卖出价格
            stock_data = df_day[df_day['symbol'] == symbol]
            if len(stock_data) == 0:
                # 停牌或退市，按买入价卖出
                sell_price = position['buy_price']
            else:
                stock_row = stock_data.iloc[0]
                if not self._can_sell(stock_row):
                    self.blocked_limit_down_sells += 1
                    continue
                sell_price = stock_row['close']

            shares = position['shares']
            
            # 计算收益
            revenue = shares * sell_price * (1 - self.commission)
            profit = revenue - position['cost']
            
            # 更新现金
            self.cash += revenue
            
            # 记录交易
            self.trades.append({
                'date': date,
                'symbol': symbol,
                'action': 'sell',
                'price': sell_price,
                'shares': shares,
                'amount': revenue,
                'profit': profit,
                'return': profit / position['cost']
            })
            
            # 删除持仓
            del self.positions[symbol]
    
    def _record_portfolio_value(self, df_day: pd.DataFrame, date: str):
        """记录组合净值"""
        # 持仓市值
        position_value = 0
        for symbol, position in self.positions.items():
            stock_data = df_day[df_day['symbol'] == symbol]
            if len(stock_data) > 0:
                current_price = stock_data.iloc[0]['close']
                position_value += position['shares'] * current_price
            else:
                # 停牌，按买入价估值
                position_value += position['shares'] * position['buy_price']
        
        # 总净值
        total_value = self.cash + position_value
        self.portfolio_values.append({
            'date': date,
            'cash': self.cash,
            'position_value': position_value,
            'total_value': total_value
        })
    
    def calculate_metrics(self) -> Dict:
        """计算回测指标"""
        pv_df = pd.DataFrame(self.portfolio_values)
        trades_df = pd.DataFrame(self.trades)
        
        metrics = calculate_all_metrics(pv_df, trades_df, self.initial_capital)
        
        # 添加净值曲线和交易记录
        metrics['portfolio_values'] = pv_df
        metrics['trades'] = trades_df
        metrics['blocked_t1_sells'] = self.blocked_t1_sells
        metrics['blocked_limit_up_buys'] = self.blocked_limit_up_buys
        metrics['blocked_limit_down_sells'] = self.blocked_limit_down_sells
        
        return metrics
    
    def _empty_result(self) -> Dict:
        """返回空结果"""
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.initial_capital,
            'total_return': 0.0,
            'annual_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_loss_ratio': 0.0,
            'total_trades': 0,
            'blocked_t1_sells': self.blocked_t1_sells,
            'blocked_limit_up_buys': self.blocked_limit_up_buys,
            'blocked_limit_down_sells': self.blocked_limit_down_sells,
            'portfolio_values': pd.DataFrame(),
            'trades': pd.DataFrame()
        }
    
    def print_result(self, result: Dict):
        """打印回测结果"""
        print("\n" + "=" * 80)
        print("📊 回测结果")
        print("=" * 80)
        print(f"初始资金: {result['initial_capital']:,.0f}元")
        print(f"最终资金: {result['final_value']:,.0f}元")
        print(f"总收益率: {result['total_return']:.1%}")
        print(f"年化收益率: {result['annual_return']:.1%}")
        print(f"最大回撤: {result['max_drawdown']:.1%}")
        print(f"夏普比率: {result['sharpe_ratio']:.2f}")
        print(f"索提诺比率: {result['sortino_ratio']:.2f}")
        print(f"卡玛比率: {result['calmar_ratio']:.2f}")
        print(f"胜率: {result['win_rate']:.1%}")
        print(f"盈亏比: {result['profit_loss_ratio']:.2f}")
        print(f"总交易次数: {result['total_trades']}")
        print(f"T+1拦截卖出次数: {result.get('blocked_t1_sells', 0)}")
        print(f"涨停拦截买入次数: {result.get('blocked_limit_up_buys', 0)}")
        print(f"跌停拦截卖出次数: {result.get('blocked_limit_down_sells', 0)}")
        print("=" * 80)
