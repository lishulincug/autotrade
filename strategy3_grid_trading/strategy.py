"""
策略3：网格交易策略
======================
核心逻辑：基于均值回归原理，划定价格区间后机械高抛低吸，
无需预判方向，靠波动赚差价。

设置规则：
  1. 区间设定：按近 1 年价格的 ±15%~20% 划定上下轨
  2. 网格间距：宽基 ETF 设 2%~3%（波动率越高，间距越大）
  3. 交易规则：每下跌 1 格买入等额仓位，每上涨 1 格卖出对应仓位
  4. 仓位管理：总资金拆分为 10~15 份，3~5 份做底仓，剩余为网格流动资金
"""
import os
import sys
import math
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_broad_etf
from common.metrics import calculate_metrics, calculate_trade_stats, print_report


class GridTrader:
    """
    网格交易器
    """

    def __init__(self, symbol: str, grid_pct: float = 0.025,  # 单格2.5%
                 range_pct: float = 0.20,  # 上下各20%区间
                 total_parts: int = 12,  # 总资金分12份
                 base_parts: int = 4):  # 底仓4份
        self.symbol = symbol
        self.grid_pct = grid_pct
        self.range_pct = range_pct
        self.total_parts = total_parts
        self.base_parts = base_parts

        self.upper_bound = None
        self.lower_bound = None
        self.grid_levels = []  # 网格价位列表
        self.grid_states = {}  # {level: 'bought'/None}  记录每个网格的状态

        self.last_buy_level = None
        self.last_sell_level = None

    def setup_grid(self, anchor_price: float):
        """
        根据锚定价格设置网格
        """
        self.upper_bound = anchor_price * (1 + self.range_pct)
        self.lower_bound = anchor_price * (1 - self.range_pct)

        # 计算网格数量
        n_grids = int(self.range_pct * 2 / self.grid_pct)
        self.grid_levels = []
        for i in range(n_grids + 1):
            price = self.lower_bound * (1 + self.grid_pct) ** i
            if price <= self.upper_bound * 1.001:
                self.grid_levels.append(price)
        self.grid_levels.sort()

        # 初始化网格状态
        self.grid_states = {p: None for p in self.grid_levels}
        self.last_buy_level = None
        self.last_sell_level = None

    def rebalance_grid(self, new_anchor: float):
        """
        价格突破上下轨时，重建网格
        """
        # 保留已买入的网格信息（平移）
        old_states = {k: v for k, v in self.grid_states.items() if v == 'bought'}
        self.setup_grid(new_anchor)
        # 重置底仓标记（不实际交易）
        return old_states

    def get_action(self, current_price: float):
        """
        根据当前价格判断交易动作
        返回: ('buy', level_price) / ('sell', level_price) / (None, None)
        """
        if not self.grid_levels:
            return None, None

        # 检查是否突破上下轨
        if current_price > self.upper_bound or current_price < self.lower_bound:
            return 'rebuild', current_price

        # 找到当前价格位于哪两个网格之间
        buy_trigger = None
        sell_trigger = None

        for i, level in enumerate(self.grid_levels):
            if self.grid_states.get(level) is None:
                # 未持仓的网格：检查是否跌到此价位
                if current_price <= level and (self.last_buy_level is None or level < self.last_buy_level):
                    buy_trigger = level
                    break

        # 从高价往低找卖出点
        for level in reversed(self.grid_levels):
            if self.grid_states.get(level) == 'bought':
                # 已持仓的网格：检查是否涨到此价位+1格
                sell_level = level * (1 + self.grid_pct)
                if current_price >= sell_level:
                    # 找最接近的上方网格
                    candidates = [g for g in self.grid_levels if g >= sell_level]
                    if candidates:
                        target = min(candidates)
                        sell_trigger = (level, target)
                        break

        if buy_trigger is not None:
            return 'buy', buy_trigger
        elif sell_trigger is not None:
            return 'sell', sell_trigger
        return None, None

    def mark_buy(self, level):
        self.grid_states[level] = 'bought'
        self.last_buy_level = level

    def mark_sell(self, buy_level, sell_level):
        self.grid_states[buy_level] = None
        self.last_sell_level = sell_level


def run_strategy(initial_cash: float = 1_000_000,
                 grid_pct: float = 0.025,  # 单格2.5%
                 range_pct: float = 0.20,  # 区间±20%
                 total_parts: int = 12,
                 base_parts: int = 4,
                 start_date: str = '2020-06-01',
                 end_date: str = '2026-07-31'):
    """
    运行网格交易策略
    """
    print("正在加载宽基ETF数据...")
    etf_data = load_broad_etf()
    etf_code = '510300'
    etf_name = etf_data['etf_name'].iloc[0] if 'etf_name' in etf_data.columns else '沪深300ETF'
    print(f"标的: {etf_code} ({etf_name})")

    # 构建价格数据字典
    price_dict = {etf_code: etf_data}
    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0003, slippage=0.001)
    engine.set_price_data(price_dict)

    # 获取回测日期
    all_dates = [d for d in etf_data.index if start_date <= d.strftime('%Y-%m-%d') <= end_date]
    if len(all_dates) < 252:
        print("警告：回测期数据不足")

    print(f"\n回测区间: {start_date} ~ {end_date}")
    print(f"网格间距: {grid_pct * 100:.1f}%")
    print(f"价格区间: ±{range_pct * 100:.0f}%")
    print(f"资金分仓: 共{total_parts}份，底仓{base_parts}份，网格{total_parts - base_parts}份")
    print()

    # 初始化：取开始前1年的价格作为初始锚定
    warmup_data = etf_data.loc[etf_data.index < all_dates[0]]
    if len(warmup_data) >= 20:
        anchor_price = warmup_data['close'].iloc[-252:].mean() if len(warmup_data) >= 252 else warmup_data['close'].mean()
    else:
        anchor_price = etf_data.loc[all_dates[0], 'close']

    trader = GridTrader(
        symbol=etf_code,
        grid_pct=grid_pct,
        range_pct=range_pct,
        total_parts=total_parts,
        base_parts=base_parts
    )
    trader.setup_grid(anchor_price)

    # 每个网格的资金量
    cash_per_part = initial_cash / total_parts
    part_per_grid = 1  # 每格1份

    # 开仓日：建立底仓
    first_date = all_dates[0]
    base_shares = 0
    for i in range(base_parts):
        if engine.buy(etf_code, first_date, amount=cash_per_part):
            base_shares += int(cash_per_part / etf_data.loc[first_date, 'close'] / 100) * 100

    # 底仓对应标记到网格上
    current_close = etf_data.loc[first_date, 'close']
    near_levels = sorted([g for g in trader.grid_levels if g <= current_close])
    for i, lv in enumerate(near_levels[-base_parts:] if len(near_levels) >= base_parts else near_levels):
        trader.mark_buy(lv)

    print(f"[{first_date.strftime('%Y-%m-%d')}] 建仓完成")
    print(f"  网格区间: {trader.lower_bound:.3f} ~ {trader.upper_bound:.3f}")
    print(f"  网格数量: {len(trader.grid_levels)} 格")

    grid_rebuilds = 0

    # 逐日执行网格
    for idx, date in enumerate(all_dates):
        if idx == 0:
            engine.record_daily_value(date)
            continue

        current_price = etf_data.loc[date, 'close']
        action, data = trader.get_action(current_price)

        if action == 'rebuild':
            # 重建网格
            trader.rebalance_grid(current_price)
            grid_rebuilds += 1
            if grid_rebuilds <= 3:
                print(f"  [重建网格] {date.strftime('%Y-%m-%d')} 价格突破，新区间 "
                      f"{trader.lower_bound:.3f} ~ {trader.upper_bound:.3f}")
        elif action == 'buy':
            level_price = data
            buy_amount = cash_per_part * part_per_grid
            if engine.buy(etf_code, date, amount=buy_amount):
                trader.mark_buy(level_price)
        elif action == 'sell':
            buy_level, sell_level = data
            # 卖出对应份额
            sell_shares = int(cash_per_part / buy_level / 100) * 100
            if sell_shares > 0:
                if engine.sell(etf_code, date, shares=sell_shares):
                    trader.mark_sell(buy_level, sell_level)

        engine.record_daily_value(date)

    # 获取结果
    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="网格交易策略")
    trade_stats = calculate_trade_stats(engine.trades)
    trade_stats['网格重建次数'] = grid_rebuilds

    # 计算简单的买入持有收益作为对比
    buy_hold_final = initial_cash / etf_data.loc[all_dates[0], 'close'] * etf_data.loc[all_dates[-1], 'close']
    buy_hold_return = (buy_hold_final / initial_cash - 1) * 100
    trade_stats['买入持有收益率(%)'] = round(buy_hold_return, 2)
    trade_stats['超额收益(%)'] = round(metrics['累计收益率(%)'] - buy_hold_return, 2)

    print_report(metrics, trade_stats, title="策略3：网格交易策略")

    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    equity_df.to_csv(os.path.join(output_dir, 'grid_equity_curve.csv'))

    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
