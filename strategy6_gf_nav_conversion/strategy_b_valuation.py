"""
策略6-B：广发约定净值转换 · 估值分位网格
============================================
核心逻辑：用滚动窗口计算当前净值在历史中的分位数（百分位排名），
当分位数处于低位（估值便宜）时买入，处于高位（估值昂贵）时止盈。

与策略A的区别：
  - 策略A用全历史静态分位数划定固定网格
  - 策略B用滚动窗口动态计算分位数，网格随市场变化自适应

机制说明：
  1. 资金存放在天天红B（广发货币基金，年化约2%），每日计息
  2. 滚动窗口N日，计算当前净值在窗口内的百分位
  3. 百分位 ≤ 买入阈值 → 买入（越便宜份额越大）
  4. 百分位 ≥ 止盈阈值 → 止盈
  5. 止盈后重置对应买入档

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


def generate_valuation_grid(nav_window: pd.Series,
                            buy_percentiles=(0.30, 0.20, 0.10),
                            sell_percentiles=(0.70, 0.80, 0.90),
                            base_share=50000,
                            share_increment=30000,
                            sell_share=50000,
                            decimals=2):
    """
    根据当前滚动窗口的净值分位数生成网格参数

    参数：
        nav_window: 滚动窗口内的净值序列
        buy_percentiles: 买入分位数阈值（从高到低，越便宜份额越大）
        sell_percentiles: 止盈分位数阈值（从低到高）
        base_share: 首档买入份额（最浅买入）
        share_increment: 每档递增份额
        sell_share: 每档止盈份额
        decimals: 约定净值保留小数位数（广发基金净值只保留2位）

    返回：
        buy_list: [{trigger_net_value, share}, ...] 降序
        sell_list: [{trigger_net_value, share}, ...] 升序
    """
    s = nav_window
    buy_list = []
    sell_list = []

    # 买入档：分位数从高到低（30%→20%→10%），对应净值从高到低
    for i, pct in enumerate(buy_percentiles):
        nv = round(float(s.quantile(pct)), decimals)
        share = base_share + i * share_increment
        buy_list.append({"trigger_net_value": nv, "share": share})

    # 止盈档：分位数从低到高（70%→80%→90%）
    for pct in sell_percentiles:
        nv = round(float(s.quantile(pct)), decimals)
        sell_list.append({"trigger_net_value": nv, "share": sell_share})

    return buy_list, sell_list


class ValuationGridTrader:
    """估值分位网格交易器"""

    def __init__(self, buy_percentiles, sell_percentiles,
                 base_share, share_increment, sell_share,
                 window=120, decimals=2):
        self.buy_percentiles = buy_percentiles
        self.sell_percentiles = sell_percentiles
        self.base_share = base_share
        self.share_increment = share_increment
        self.sell_share = sell_share
        self.window = window
        self.decimals = decimals
        self.buy_used = [False] * len(buy_percentiles)
        self.conversion_count = 0
        self.buy_count = 0
        self.sell_count = 0

    def compute_percentile(self, nav_window, current_nav):
        """计算当前净值在窗口内的百分位（0~1）"""
        return float((nav_window < current_nav).sum() / len(nav_window))

    def get_grid(self, nav_window):
        """根据当前窗口生成网格"""
        return generate_valuation_grid(
            nav_window, self.buy_percentiles, self.sell_percentiles,
            self.base_share, self.share_increment, self.sell_share,
            decimals=self.decimals
        )

    def check_buys(self, current_pct, buy_list):
        """检查买入触发（百分位 ≤ 阈值）"""
        triggers = []
        for i, trigger in enumerate(buy_list):
            if not self.buy_used[i]:
                # 买入阈值对应的是分位数；当前百分位 ≤ 该档分位时触发
                threshold_pct = self.buy_percentiles[i]
                if current_pct <= threshold_pct:
                    triggers.append((i, trigger['share']))
        return triggers

    def check_sells(self, current_pct, sell_list, available_shares):
        """检查止盈触发（百分位 ≥ 阈值）"""
        triggers = []
        remaining = available_shares
        for i, trigger in enumerate(sell_list):
            if remaining <= 0:
                break
            threshold_pct = self.sell_percentiles[i]
            if current_pct >= threshold_pct:
                sell_share = min(trigger['share'], remaining)
                triggers.append((i, sell_share))
                remaining -= sell_share
        return triggers

    def mark_buy_used(self, idx):
        self.buy_used[idx] = True
        self.buy_count += 1

    def reset_buys_below(self, sell_pct):
        """止盈后重置分位高于该止盈档的买入档（即更便宜的买入档）"""
        for i, pct in enumerate(self.buy_percentiles):
            if pct <= sell_pct:
                self.buy_used[i] = False


def run_strategy(initial_cash: float = 1_000_000,
                 buy_percentiles=(0.30, 0.20, 0.10),
                 sell_percentiles=(0.70, 0.80, 0.90),
                 base_share: int = 50000,
                 share_increment: int = 30000,
                 sell_share: int = 50000,
                 window: int = 120,
                 money_fund_rate: float = 0.02,
                 start_date: str = None,
                 end_date: str = None,
                 nav_data: pd.DataFrame = None,
                 fund_code: str = FUND_CODE,
                 fund_name: str = FUND_NAME,
                 decimals: int = 2,
                 verbose: bool = True):
    """
    运行估值分位网格策略

    参数：
        buy_percentiles: 买入分位数阈值（如0.30表示净值处于30%分位以下时买入）
        sell_percentiles: 止盈分位数阈值
        base_share: 首档买入份额
        share_increment: 每档递增份额
        sell_share: 每档止盈份额
        window: 滚动窗口天数
        money_fund_rate: 天天红B年化收益率
        start_date: None=自动（跳过window日预热）
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

    # 自适应回测日期（滚动窗口策略：跳过 window 日预热）
    all_dates, _, end_date = resolve_backtest_dates(
        nav_data, start_date, end_date, warmup_days=min(window, 60))
    if len(all_dates) < 30:
        if verbose:
            print("警告：回测期数据不足")

    trader = ValuationGridTrader(
        buy_percentiles, sell_percentiles, base_share, share_increment,
        sell_share, window, decimals=decimals
    )

    # 每日利息
    daily_interest = 1 + money_fund_rate / 365

    # 用于输出最终参数表的网格（用最后一日的窗口）
    final_buy_list = None
    final_sell_list = None

    if verbose:
        print(f"\n回测区间: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates)}日)")
        print(f"滚动窗口: {window} 日")
        print(f"买入分位阈值: {buy_percentiles}")
        print(f"止盈分位阈值: {sell_percentiles}")
        print(f"天天红B年化: {money_fund_rate * 100:.1f}%")
        print(f"初始资金: {initial_cash:,.0f} 元\n")

    for idx, date in enumerate(all_dates):
        engine.cash *= daily_interest
        current_nav = nav_data.loc[date, 'close']

        # 滚动窗口
        window_data = nav_data['close'].loc[:date].tail(window)
        if len(window_data) < 20:
            engine.record_daily_value(date)
            continue

        current_pct = trader.compute_percentile(window_data, current_nav)
        buy_list, sell_list = trader.get_grid(window_data)
        final_buy_list = buy_list
        final_sell_list = sell_list

        # 先止盈
        pos = engine.positions.get(fund_code)
        current_shares = pos.shares if pos and pos.shares > 0 else 0
        sell_triggers = trader.check_sells(current_pct, sell_list, current_shares)
        for trigger_idx, share in sell_triggers:
            if engine.sell(fund_code, date, shares=share):
                trader.sell_count += 1
                trader.conversion_count += 1
                trader.reset_buys_below(sell_percentiles[trigger_idx])
                if verbose and trader.conversion_count <= 10:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 止盈: 净值={current_nav:.4f} "
                          f"分位={current_pct:.2f}≥{sell_percentiles[trigger_idx]:.2f}, 份额={share}")

        # 再买入
        buy_triggers = trader.check_buys(current_pct, buy_list)
        for trigger_idx, share in buy_triggers:
            if engine.buy(fund_code, date, shares=share):
                trader.mark_buy_used(trigger_idx)
                trader.conversion_count += 1
                if verbose and trader.conversion_count <= 10:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 买入: 净值={current_nav:.4f} "
                          f"分位={current_pct:.2f}≤{buy_percentiles[trigger_idx]:.2f}, 份额={share}")

        engine.record_daily_value(date)

    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="估值分位网格")
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
        print_report(metrics, trade_stats, title="策略6-B：估值分位网格")
        if final_buy_list:
            print("\n【最新参数表（基于最近窗口）】")
            print("====买入策略(天天红B→基金，约定净值≤xxx)====")
            for item in final_buy_list:
                print(f"  净值≤{item['trigger_net_value']}, 份额: {item['share']}")
            print("====止盈策略(基金→天天红B，约定净值≥xxx)====")
            for item in final_sell_list:
                print(f"  净值≥{item['trigger_net_value']}, 份额: {item['share']}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    if final_buy_list:
        _save_gf_params(final_buy_list, final_sell_list, output_dir,
                        prefix='gf_valuation_')

    engine.gf_buy_list = final_buy_list
    engine.gf_sell_list = final_sell_list

    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
