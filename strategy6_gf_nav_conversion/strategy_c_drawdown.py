"""
策略6-C：广发约定净值转换 · 高点回撤止盈网格
============================================
核心逻辑：跟踪滚动窗口内的净值高点，当净值从高点回撤达到一定比例时买入
（越跌份额越大），当净值从买入成本上涨达到止盈比例时卖出止盈。

与策略A/B的区别：
  - 策略A/B用分位数划定买入/止盈区间
  - 策略C用"从高点回撤X%"作为买入触发，"从成本上涨Y%"作为止盈触发
  - 这是经典的"跌出来的机会，涨出来的利润"思路

机制说明：
  1. 资金存放在天天红B（广发货币基金，年化约2%），每日计息
  2. 计算N日滚动高点，当净值 ≤ 高点*(1-回撤比例) → 买入（越跌份额越大）
  3. 当净值 ≥ 持仓平均成本*(1+止盈比例) → 止盈
  4. 止盈后重置对应买入档

标的基金：广发中证红利ETF发起式联接C（021400）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_otc_fund_nav
from common.metrics import calculate_metrics, calculate_trade_stats, print_report
from strategy6_gf_nav_conversion.strategy import FUND_CODE, FUND_NAME, _save_gf_params, resolve_backtest_dates


def generate_drawdown_grid(current_high,
                           drawdown_levels=(0.03, 0.06, 0.09, 0.12, 0.15),
                           take_profit_levels=(0.03, 0.05, 0.08),
                           base_share=50000,
                           share_increment=20000,
                           sell_share=50000,
                           decimals=2):
    """
    根据当前滚动高点生成网格参数

    参数：
        current_high: 当前滚动窗口高点净值
        drawdown_levels: 买入回撤比例（从低到高，如0.03=3%）
        take_profit_levels: 止盈比例（从低到高）
        base_share: 首档买入份额
        share_increment: 每档递增份额
        sell_share: 每档止盈份额
        decimals: 约定净值保留小数位数（广发基金净值只保留2位）

    返回：
        buy_list: [{trigger_net_value, share, drawdown}, ...] 降序
        sell_list: [{trigger_net_value, share, take_profit}, ...] 升序（相对值）
    """
    buy_list = []
    sell_list = []

    # 买入档：从高点回撤不同比例
    for i, dd in enumerate(drawdown_levels):
        nv = round(current_high * (1 - dd), decimals)
        share = base_share + i * share_increment
        buy_list.append({"trigger_net_value": nv, "share": share, "drawdown": dd})

    # 止盈档：基于当前净值的上涨比例（实际止盈用持仓成本计算）
    for tp in take_profit_levels:
        # 这里的 trigger_net_value 是参考值，实际止盈以持仓成本为准
        sell_list.append({"trigger_net_value": 0, "share": sell_share, "take_profit": tp})

    return buy_list, sell_list


class DrawdownGridTrader:
    """高点回撤止盈网格交易器"""

    def __init__(self, drawdown_levels, take_profit_levels,
                 base_share, share_increment, sell_share, high_window=60, decimals=2):
        self.drawdown_levels = drawdown_levels
        self.take_profit_levels = take_profit_levels
        self.base_share = base_share
        self.share_increment = share_increment
        self.sell_share = sell_share
        self.high_window = high_window
        self.decimals = decimals
        self.buy_used = [False] * len(drawdown_levels)
        self.conversion_count = 0
        self.buy_count = 0
        self.sell_count = 0

    def get_rolling_high(self, nav_window):
        """获取滚动窗口高点"""
        return float(nav_window.max())

    def get_grid(self, current_high):
        return generate_drawdown_grid(
            current_high, self.drawdown_levels, self.take_profit_levels,
            self.base_share, self.share_increment, self.sell_share,
            decimals=self.decimals
        )

    def check_buys(self, current_nav, buy_list):
        """检查买入触发（净值 ≤ 高点*(1-回撤)）"""
        triggers = []
        for i, trigger in enumerate(buy_list):
            if not self.buy_used[i] and current_nav <= trigger['trigger_net_value']:
                triggers.append((i, trigger['share']))
        return triggers

    def check_sells(self, current_nav, avg_cost, available_shares):
        """检查止盈触发（净值 ≥ 成本*(1+止盈比例)）"""
        triggers = []
        remaining = available_shares
        for tp in self.take_profit_levels:
            if remaining <= 0:
                break
            target_nav = avg_cost * (1 + tp)
            if current_nav >= target_nav:
                sell_share = min(self.sell_share, remaining)
                triggers.append((tp, sell_share, target_nav))
                remaining -= sell_share
        return triggers

    def mark_buy_used(self, idx):
        self.buy_used[idx] = True
        self.buy_count += 1

    def reset_buys_above(self, sell_nav):
        """止盈后重置触发净值高于该卖出净值的买入档（即较浅的买入档）"""
        for i, trigger in enumerate(self.buy_triggers_cache):
            if trigger['trigger_net_value'] > sell_nav:
                self.buy_used[i] = False


def run_strategy(initial_cash: float = 1_000_000,
                 drawdown_levels=(0.03, 0.06, 0.09, 0.12, 0.15),
                 take_profit_levels=(0.03, 0.05, 0.08),
                 base_share: int = 50000,
                 share_increment: int = 20000,
                 sell_share: int = 50000,
                 high_window: int = 60,
                 money_fund_rate: float = 0.02,
                 start_date: str = None,
                 end_date: str = None,
                 nav_data: pd.DataFrame = None,
                 fund_code: str = FUND_CODE,
                 fund_name: str = FUND_NAME,
                 decimals: int = 2,
                 verbose: bool = True):
    """
    运行高点回撤止盈网格策略

    参数：
        drawdown_levels: 买入回撤比例（如0.03=从高点跌3%买入）
        take_profit_levels: 止盈比例（如0.03=成本涨3%止盈）
        base_share: 首档买入份额
        share_increment: 每档递增份额
        sell_share: 每档止盈份额
        high_window: 滚动高点窗口天数
        start_date: None=自动（跳过high_window日预热）
        fund_code/fund_name: 目标基金
    """
    if nav_data is None:
        if verbose:
            print("正在加载基金净值数据...")
        nav_data = load_otc_fund_nav(fund_code, fund_name,
                                     start_date='2019-01-01', end_date='2026-12-31',
                                     verbose=verbose)

    if verbose:
        print(f"标的: {fund_code} ({fund_name})")

    price_dict = {fund_code: nav_data}
    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0, slippage=0.0)
    engine.set_price_data(price_dict)

    # 自适应回测日期（跳过 high_window 日预热）
    all_dates, _, end_date = resolve_backtest_dates(
        nav_data, start_date, end_date, warmup_days=min(high_window, 60))
    if len(all_dates) < 30:
        if verbose:
            print("警告：回测期数据不足")

    trader = DrawdownGridTrader(
        drawdown_levels, take_profit_levels, base_share, share_increment,
        sell_share, high_window, decimals=decimals
    )

    daily_interest = 1 + money_fund_rate / 365
    final_buy_list = None
    final_sell_list = None

    if verbose:
        print(f"\n回测区间: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates)}日)")
        print(f"滚动高点窗口: {high_window} 日")
        print(f"买入回撤比例: {[f'{d*100:.0f}%' for d in drawdown_levels]}")
        print(f"止盈比例: {[f'{t*100:.0f}%' for t in take_profit_levels]}")
        print(f"天天红B年化: {money_fund_rate * 100:.1f}%")
        print(f"初始资金: {initial_cash:,.0f} 元\n")

    for idx, date in enumerate(all_dates):
        engine.cash *= daily_interest
        current_nav = nav_data.loc[date, 'close']

        # 滚动高点
        window_data = nav_data['close'].loc[:date].tail(high_window)
        if len(window_data) < 20:
            engine.record_daily_value(date)
            continue

        rolling_high = trader.get_rolling_high(window_data)
        buy_list, sell_list = trader.get_grid(rolling_high)
        trader.buy_triggers_cache = buy_list
        final_buy_list = buy_list

        # 先止盈（基于持仓成本）
        pos = engine.positions.get(fund_code)
        current_shares = pos.shares if pos and pos.shares > 0 else 0
        avg_cost = pos.avg_cost if pos and pos.shares > 0 else 0

        if current_shares > 0 and avg_cost > 0:
            sell_triggers = trader.check_sells(current_nav, avg_cost, current_shares)
            for tp, share, target_nav in sell_triggers:
                if engine.sell(fund_code, date, shares=share):
                    trader.sell_count += 1
                    trader.conversion_count += 1
                    trader.reset_buys_above(current_nav)
                    if verbose and trader.conversion_count <= 10:
                        print(f"  [{date.strftime('%Y-%m-%d')}] 止盈: 净值={current_nav:.4f} "
                              f"≥成本{avg_cost:.4f}*(1+{tp*100:.0f}%)={target_nav:.4f}, 份额={share}")

        # 再买入
        buy_triggers = trader.check_buys(current_nav, buy_list)
        for trigger_idx, share in buy_triggers:
            if engine.buy(fund_code, date, shares=share):
                trader.mark_buy_used(trigger_idx)
                trader.conversion_count += 1
                if verbose and trader.conversion_count <= 10:
                    dd = drawdown_levels[trigger_idx]
                    print(f"  [{date.strftime('%Y-%m-%d')}] 买入: 净值={current_nav:.4f} "
                          f"≤高点{rolling_high:.4f}*(1-{dd*100:.0f}%)={buy_list[trigger_idx]['trigger_net_value']}, "
                          f"份额={share}")

        engine.record_daily_value(date)

    # 生成止盈参数表（基于当前净值和最新持仓成本）
    pos = engine.positions.get(fund_code)
    avg_cost = pos.avg_cost if pos and pos.shares > 0 else current_nav
    sell_list_out = []
    for tp in take_profit_levels:
        sell_list_out.append({
            "trigger_net_value": round(avg_cost * (1 + tp), decimals),
            "share": sell_share
        })
    final_sell_list = sell_list_out

    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="高点回撤止盈网格")
    trade_stats = calculate_trade_stats(engine.trades)
    trade_stats['转换次数'] = trader.conversion_count
    trade_stats['买入触发次数'] = trader.buy_count
    trade_stats['止盈触发次数'] = trader.sell_count

    buy_hold_final = initial_cash / nav_data.loc[all_dates[0], 'close'] * nav_data.loc[all_dates[-1], 'close']
    buy_hold_return = (buy_hold_final / initial_cash - 1) * 100
    trade_stats['买入持有收益率(%)'] = round(buy_hold_return, 2)
    trade_stats['超额收益(%)'] = round(metrics['累计收益率(%)'] - buy_hold_return, 2)

    mf_final = initial_cash * (1 + money_fund_rate) ** (len(all_dates) / 365)
    mf_return = (mf_final / initial_cash - 1) * 100
    trade_stats['纯货币基金收益率(%)'] = round(mf_return, 2)

    if verbose:
        print_report(metrics, trade_stats, title="策略6-C：高点回撤止盈网格")
        if final_buy_list:
            print("\n【最新参数表】")
            print("====买入策略(天天红B→基金，净值≤高点*(1-回撤))====")
            for item in final_buy_list:
                print(f"  净值≤{item['trigger_net_value']} (回撤{item['drawdown']*100:.0f}%), 份额: {item['share']}")
            print("====止盈策略(基金→天天红B，净值≥成本*(1+止盈))====")
            for i, item in enumerate(final_sell_list):
                print(f"  净值≥{item['trigger_net_value']} (止盈{take_profit_levels[i]*100:.0f}%), 份额: {item['share']}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    if final_buy_list and final_sell_list:
        _save_gf_params(final_buy_list, final_sell_list, output_dir,
                        prefix='gf_drawdown_')

    engine.gf_buy_list = final_buy_list
    engine.gf_sell_list = final_sell_list

    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
