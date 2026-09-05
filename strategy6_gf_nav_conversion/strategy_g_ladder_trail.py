"""
策略6-G：广发约定净值转换 · 阶梯止盈 + 移动止损映射
============================================
把「高点回撤/移动止损」映射为多档固定净值条件单：
  - 净值每上涨一个台阶，激活对应回撤止损档
  - 触及更高台阶后，作废更低止损档，挂上新止损

仅适合明确的牛市主升浪；震荡市会反复磨损，不建议长期启用。
启用 C 类＜7天惩罚费与买入档 ≥3% 间距。
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_otc_fund_nav
from common.metrics import calculate_metrics, calculate_trade_stats, print_report
from strategy6_gf_nav_conversion.strategy import (
    FUND_CODE, FUND_NAME, resolve_backtest_dates, _save_gf_params
)
from strategy6_gf_nav_conversion.gf_constraints import (
    enforce_buy_tier_gap, apply_share_jitter, HoldingLotTracker,
    buy_with_lot_track, sell_with_short_hold_penalty, enhance_save_rows
)


def generate_ladder(base_nav: float,
                    up_steps=(0.05, 0.10, 0.15, 0.20),
                    trail_pcts=(0.03, 0.04, 0.05, 0.06),
                    buy_drawdowns=(0.03, 0.06, 0.09),
                    base_share=60000,
                    share_increment=30000,
                    sell_share=None,
                    decimals=2,
                    jitter=True):
    """
    生成买入档 + 阶梯止损映射表。
    ladder: [{activate_nav, stop_nav, share}, ...] 升序
    """
    buy_list = []
    for i, dd in enumerate(buy_drawdowns):
        nv = round(base_nav * (1 - dd), decimals)
        share = base_share + i * share_increment
        buy_list.append({
            'trigger_net_value': nv,
            'share': float(share),
            'role': 'buy',
            'note': f'回撤建仓{-dd:.0%}',
        })
    buy_list = enforce_buy_tier_gap(buy_list, decimals=decimals)
    buy_list = apply_share_jitter(buy_list, seed_base=505, enabled=jitter)

    total_buy = sum(b['share'] for b in buy_list)
    if sell_share is None:
        sell_share = total_buy  # 触发止损时尽量清浮动仓

    ladder = []
    for up, trail in zip(up_steps, trail_pcts):
        activate = round(base_nav * (1 + up), decimals)
        stop = round(activate * (1 - trail), decimals)
        ladder.append({
            'activate_nav': activate,
            'stop_nav': stop,
            'share': round(float(sell_share), 2),
            'up': up,
            'trail': trail,
        })

    # 展示用 sell_list：当前可挂的全部止损位（实盘需随升档删低档）
    sell_list = []
    for i, step in enumerate(ladder):
        sell_list.append({
            'trigger_net_value': step['stop_nav'],
            'share': step['share'],
            'role': 'trail_stop',
            'note': f"台阶≥{step['activate_nav']}后生效止损",
            'zone': f"ladder_{i+1}",
        })
    sell_list = apply_share_jitter(sell_list, seed_base=606, enabled=jitter)
    # 同步 jitter 后的份额到 ladder
    for i, s in enumerate(sell_list):
        ladder[i]['share'] = s['share']

    return buy_list, sell_list, ladder


class LadderTrailTrader:
    def __init__(self, buy_list, ladder):
        self.buy_triggers = buy_list
        self.ladder = ladder
        self.buy_used = [False] * len(buy_list)
        self.active_level = -1  # -1=尚未激活任何止损
        self.conversion_count = 0
        self.buy_count = 0
        self.sell_count = 0

    def check_buys(self, current_nav):
        triggers = []
        for i, t in enumerate(self.buy_triggers):
            if not self.buy_used[i] and current_nav <= t['trigger_net_value']:
                triggers.append((i, t['share']))
        return triggers

    def update_ladder(self, current_nav):
        """净值创新高台阶时升档。"""
        new_level = self.active_level
        for i, step in enumerate(self.ladder):
            if current_nav >= step['activate_nav']:
                new_level = i
        if new_level > self.active_level:
            self.active_level = new_level
        return self.active_level

    def current_stop(self):
        if self.active_level < 0:
            return None
        return self.ladder[self.active_level]

    def check_stop(self, current_nav, available_shares):
        step = self.current_stop()
        if step is None or available_shares <= 0:
            return None
        if current_nav <= step['stop_nav']:
            return min(step['share'], available_shares)
        return None

    def mark_buy(self, idx):
        self.buy_used[idx] = True
        self.buy_count += 1

    def active_sell_list(self):
        """当前应保留在广发页面的唯一止损单。"""
        step = self.current_stop()
        if step is None:
            return []
        return [{
            'trigger_net_value': step['stop_nav'],
            'share': step['share'],
            'role': 'trail_stop',
            'note': f"当前有效止损(台阶≥{step['activate_nav']})",
            'zone': f"active_L{self.active_level+1}",
        }]


def run_strategy(initial_cash: float = 1_000_000,
                 up_steps=(0.05, 0.10, 0.15, 0.20),
                 trail_pcts=(0.03, 0.04, 0.05, 0.06),
                 buy_drawdowns=(0.03, 0.06, 0.09),
                 base_share: int = 60000,
                 share_increment: int = 30000,
                 money_fund_rate: float = 0.02,
                 start_date: str = None,
                 end_date: str = None,
                 nav_data: pd.DataFrame = None,
                 fund_code: str = FUND_CODE,
                 fund_name: str = FUND_NAME,
                 decimals: int = 2,
                 jitter: bool = True,
                 verbose: bool = True):
    """
    阶梯止盈+移动止损映射。仅适合主升浪行情，震荡市慎用。
    """
    if nav_data is None:
        if verbose:
            print("正在加载基金净值数据...")
        nav_data = load_otc_fund_nav(fund_code, fund_name,
                                     start_date='2019-01-01', end_date='2026-12-31',
                                     verbose=verbose)

    if verbose:
        print(f"标的: {fund_code} ({fund_name})")
        print("注意：本策略仅适合牛市主升浪，震荡市可能反复磨损。")

    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0, slippage=0.0)
    engine.set_price_data({fund_code: nav_data})

    all_dates, warmup_navs, end_date = resolve_backtest_dates(
        nav_data, start_date, end_date, warmup_days=60)
    if len(all_dates) < 30 and verbose:
        print("警告：回测期数据不足")

    base_nav = float(warmup_navs.iloc[-1]) if len(warmup_navs) else float(nav_data.loc[all_dates[0], 'close'])
    buy_list, sell_list_all, ladder = generate_ladder(
        base_nav,
        up_steps=up_steps,
        trail_pcts=trail_pcts,
        buy_drawdowns=buy_drawdowns,
        base_share=base_share,
        share_increment=share_increment,
        decimals=decimals,
        jitter=jitter,
    )
    trader = LadderTrailTrader(buy_list, ladder)
    tracker = HoldingLotTracker()
    daily_interest = 1 + money_fund_rate / 365
    stopped_out = False

    if verbose:
        print(f"\n回测区间: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"基准净值: {base_nav:.{decimals}f}")
        print("====买入====")
        for b in buy_list:
            print(f"  ≤{b['trigger_net_value']} 份额={b['share']}")
        print("====阶梯止损====")
        for step in ladder:
            print(f"  激活≥{step['activate_nav']} → 止损≤{step['stop_nav']} 份额={step['share']}")

    for date in all_dates:
        engine.cash *= daily_interest
        current_nav = float(nav_data.loc[date, 'close'])

        # 先升档
        trader.update_ladder(current_nav)

        pos = engine.positions.get(fund_code)
        current_shares = pos.shares if pos and pos.shares > 0 else 0.0

        # 止损
        if not stopped_out:
            stop_share = trader.check_stop(current_nav, current_shares)
            if stop_share and stop_share > 0:
                ok, _ = sell_with_short_hold_penalty(
                    engine, tracker, fund_code, date, stop_share)
                if ok:
                    trader.sell_count += 1
                    trader.conversion_count += 1
                    stopped_out = True
                    # 止损后重置买入档，允许下一轮
                    trader.buy_used = [False] * len(trader.buy_triggers)
                    trader.active_level = -1
                    stopped_out = False

        # 买入（无仓或未满档）
        for trigger_idx, share in trader.check_buys(current_nav):
            if buy_with_lot_track(engine, tracker, fund_code, date, share):
                trader.mark_buy(trigger_idx)
                trader.conversion_count += 1

        engine.record_daily_value(date)

    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="阶梯止盈+移动止损")
    trade_stats = calculate_trade_stats(engine.trades)
    trade_stats['转换次数'] = trader.conversion_count
    trade_stats['买入触发次数'] = trader.buy_count
    trade_stats['止盈触发次数'] = trader.sell_count
    trade_stats['当前止损档'] = trader.active_level + 1
    trade_stats['短期惩罚费(元)'] = round(tracker.total_penalty, 2)

    buy_hold_final = initial_cash / nav_data.loc[all_dates[0], 'close'] * nav_data.loc[all_dates[-1], 'close']
    buy_hold_return = (buy_hold_final / initial_cash - 1) * 100
    trade_stats['买入持有收益率(%)'] = round(buy_hold_return, 2)
    trade_stats['超额收益(%)'] = round(metrics['累计收益率(%)'] - buy_hold_return, 2)
    mf_final = initial_cash * (1 + money_fund_rate) ** (len(all_dates) / 365)
    trade_stats['纯货币基金收益率(%)'] = round((mf_final / initial_cash - 1) * 100, 2)

    # 输出：全部阶梯 + 当前有效止损
    active_sell = trader.active_sell_list()
    display_sell = active_sell if active_sell else sell_list_all

    if verbose:
        print_report(metrics, trade_stats, title="策略6-G：阶梯止盈+移动止损映射")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    _save_gf_params(buy_list, display_sell, output_dir, prefix='gf_ladder_trail_')
    pd.DataFrame(enhance_save_rows(buy_list, sell_list_all)).to_csv(
        os.path.join(output_dir, 'gf_ladder_trail_params.csv'),
        index=False, encoding='utf-8-sig')

    engine.gf_buy_list = buy_list
    engine.gf_sell_list = display_sell
    engine.gf_ladder = ladder
    engine.gf_active_level = trader.active_level
    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
