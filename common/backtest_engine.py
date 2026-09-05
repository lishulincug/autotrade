"""
回测引擎 - 通用回测框架
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class Trade:
    """单条交易记录"""
    date: pd.Timestamp
    symbol: str
    action: str  # 'buy' or 'sell'
    price: float
    shares: float
    commission: float = 0.0


@dataclass
class Position:
    """持仓信息"""
    symbol: str
    shares: float = 0.0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self):
        return self.shares * self.current_price

    @property
    def profit(self):
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def profit_pct(self):
        if self.avg_cost > 0:
            return (self.current_price - self.avg_cost) / self.avg_cost * 100
        return 0.0


class BacktestEngine:
    """
    通用回测引擎
    """

    def __init__(self, initial_cash: float = 1_000_000,
                 commission_rate: float = 0.0003,  # 万三手续费
                 slippage: float = 0.001):  # 千一滑点
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_rate = commission_rate
        self.slippage = slippage

        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Dict] = []

        self.current_date = None
        self.price_data = {}  # {symbol: DataFrame with close prices}

    def set_price_data(self, price_dict: Dict[str, pd.DataFrame]):
        """设置价格数据，key为symbol，value为包含close列的DataFrame"""
        self.price_data = {}
        for sym, df in price_dict.items():
            if 'close' not in df.columns:
                raise ValueError(f"{sym} 数据缺少 close 列")
            self.price_data[sym] = df.copy()

    def _get_price(self, symbol: str, date: pd.Timestamp,
                   price_type: str = 'close') -> Optional[float]:
        """获取指定日期的价格"""
        if symbol not in self.price_data:
            return None
        df = self.price_data[symbol]
        if date not in df.index:
            # 尝试找最近的一个交易日
            idx = df.index.get_indexer([date], method='pad')[0]
            if idx < 0:
                return None
            date = df.index[idx]
        try:
            return float(df.loc[date, price_type])
        except (KeyError, IndexError):
            return None

    def buy(self, symbol: str, date: pd.Timestamp,
            amount: Optional[float] = None,
            shares: Optional[float] = None) -> bool:
        """
        买入
        amount: 使用的现金金额（None则用shares参数）
        shares: 买入份额（与amount二选一）
        """
        price = self._get_price(symbol, date)
        if price is None:
            return False

        # 加入滑点
        exec_price = price * (1 + self.slippage)

        if amount is not None:
            # 按金额买入
            max_shares = amount / (exec_price * (1 + self.commission_rate))
            shares = int(max_shares / 100) * 100  # 按整手买入
            if shares <= 0:
                return False
        elif shares is None:
            return False

        total_cost = shares * exec_price
        commission = total_cost * self.commission_rate
        total_payment = total_cost + commission

        if self.cash < total_payment:
            # 调整到可用资金范围内
            max_possible = self.cash / (exec_price * (1 + self.commission_rate))
            shares = int(max_possible / 100) * 100
            if shares <= 0:
                return False
            total_cost = shares * exec_price
            commission = total_cost * self.commission_rate
            total_payment = total_cost + commission

        self.cash -= total_payment

        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)

        pos = self.positions[symbol]
        total_old_cost = pos.avg_cost * pos.shares
        total_new_shares = pos.shares + shares
        pos.avg_cost = (total_old_cost + total_cost) / total_new_shares if total_new_shares > 0 else 0
        pos.shares = total_new_shares
        pos.current_price = price

        self.trades.append(Trade(
            date=date, symbol=symbol, action='buy',
            price=exec_price, shares=shares, commission=commission
        ))
        return True

    def sell(self, symbol: str, date: pd.Timestamp,
             shares: Optional[float] = None,
             pct: Optional[float] = None) -> bool:
        """
        卖出
        shares: 卖出份额
        pct: 卖出持仓比例 (0~1)
        """
        if symbol not in self.positions:
            return False

        pos = self.positions[symbol]
        if pos.shares <= 0:
            return False

        if pct is not None:
            shares = int(pos.shares * pct / 100) * 100
        elif shares is None:
            shares = pos.shares  # 全卖

        shares = min(shares, pos.shares)
        if shares <= 0:
            return False

        price = self._get_price(symbol, date)
        if price is None:
            return False

        exec_price = price * (1 - self.slippage)
        total_revenue = shares * exec_price
        commission = total_revenue * self.commission_rate
        net_revenue = total_revenue - commission

        self.cash += net_revenue
        pos.shares -= shares
        if pos.shares == 0:
            pos.avg_cost = 0.0
        pos.current_price = price

        self.trades.append(Trade(
            date=date, symbol=symbol, action='sell',
            price=exec_price, shares=shares, commission=commission
        ))
        return True

    def sell_all(self, date: pd.Timestamp):
        """卖出所有持仓"""
        symbols = list(self.positions.keys())
        for sym in symbols:
            self.sell(sym, date)

    def update_positions_price(self, date: pd.Timestamp):
        """更新当日持仓价格"""
        for sym, pos in self.positions.items():
            price = self._get_price(sym, date)
            if price is not None:
                pos.current_price = price

    def get_total_value(self) -> float:
        """获取当前总资产"""
        position_value = sum(p.market_value for p in self.positions.values() if p.shares > 0)
        return self.cash + position_value

    def record_daily_value(self, date: pd.Timestamp):
        """记录每日净值"""
        self.update_positions_price(date)
        self.daily_values.append({
            'date': date,
            'cash': self.cash,
            'position_value': sum(p.market_value for p in self.positions.values()),
            'total_value': self.get_total_value(),
            'positions': {
                sym: {
                    'shares': pos.shares,
                    'price': pos.current_price,
                    'value': pos.market_value,
                    'profit_pct': pos.profit_pct
                } for sym, pos in self.positions.items() if pos.shares > 0
            }
        })

    def get_equity_curve(self) -> pd.DataFrame:
        """获取权益曲线"""
        if not self.daily_values:
            return pd.DataFrame()

        df = pd.DataFrame(self.daily_values)
        df.set_index('date', inplace=True)
        df['return'] = df['total_value'].pct_change()
        df['equity'] = df['total_value'] / self.initial_cash
        df['drawdown'] = (df['total_value'] / df['total_value'].cummax() - 1) * 100
        return df
