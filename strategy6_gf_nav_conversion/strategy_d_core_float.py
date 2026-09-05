"""
策略6-D：广发约定净值转换 · 底仓锁利 + 浮动网格
============================================
核心改进：解决纯动态网格在单边牛市卖飞的痛点。

持仓拆分：
  1. 底仓（约 40%）：低位买入后长期持有，不设止盈
  2. 浮动网格（约 60%）：高抛低吸，涨了分批卖、跌了分批买

极端行情：
  - 净值 ≤ 历史 Q20：额外加仓计入底仓
  - 净值 ≥ 历史 Q90：只卖浮动仓，底仓继续持有

启用 C 类＜7天 1.5% 惩罚费建模；买入档间距 ≥3%。
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


def generate_core_float_grid(net_value_series,
                             core_ratio=0.40,
                             float_ratio=0.60,
                             buy_tier_count=4,
                             sell_tier_count=3,
                             buy_low_q=0.15,
                             buy_high_q=0.70,
                             sell_low_q=0.75,
                             sell_high_q=0.90,
                             q20=0.20,
                             q90=0.90,
                             initial_cash=1_000_000,
                             ref_nav=None,
                             decimals=2,
                             jitter=True):
    """
    生成底仓+浮动网格参数。
    买入 4 档：最低档 role=core，其余 role=float；可选 Q20 额外底仓档。
    卖出总份额 = 浮动仓总份额。
    """
    s = pd.Series(net_value_series) if not isinstance(net_value_series, pd.Series) else net_value_series
    ref = float(ref_nav) if ref_nav is not None else float(s.iloc[-1])
    total_target_shares = initial_cash / max(ref, 1e-6)
    core_shares_budget = total_target_shares * core_ratio
    float_shares_budget = total_target_shares * float_ratio

    buy_high = round(float(s.quantile(buy_high_q)), decimals)
    buy_low = round(float(s.quantile(buy_low_q)), decimals)
    sell_low = round(float(s.quantile(sell_low_q)), decimals)
    sell_high = round(float(s.quantile(sell_high_q)), decimals)
    nv_q20 = round(float(s.quantile(q20)), decimals)
    nv_q90 = round(float(s.quantile(q90)), decimals)

    # 先建浮动档（较高净值），再建底仓档（更低净值），避免排序后 role 错位
    float_tiers = max(1, buy_tier_count - 1)
    # 浮动档落在 [buy_high, max(buy_low上移, q20上方)]
    float_low = max(buy_low, round(nv_q20 * 1.01, decimals))
    if float_low >= buy_high:
        float_low = round(buy_high * (1 - 0.03 * float_tiers), decimals)
    float_nvs = sorted(np.linspace(buy_high, float_low, float_tiers), reverse=True)

    float_weights = np.arange(1, float_tiers + 1, dtype=float)
    float_weights = float_weights / float_weights.sum()

    buy_list = []
    for i, nv in enumerate(float_nvs):
        share = float_shares_budget * float_weights[i]
        buy_list.append({
            'trigger_net_value': round(float(nv), decimals),
            'share': round(float(share), 2),
            'role': 'float',
            'note': '浮动网格',
        })

    # 主底仓：不低于 Q20，且低于最低浮动档
    main_core_nv = min(buy_low, round(float_nvs[-1] * (1 - 0.03), decimals))
    buy_list.append({
        'trigger_net_value': round(float(main_core_nv), decimals),
        'share': round(float(core_shares_budget * 0.75), 2),
        'role': 'core',
        'note': '底仓-不设止盈',
    })
    # Q20 额外底仓（若与主底仓过近则再下压一档）
    extra_nv = nv_q20
    if abs(extra_nv - main_core_nv) / max(main_core_nv, 1e-6) < 0.03:
        extra_nv = round(min(extra_nv, main_core_nv) * (1 - 0.03), decimals)
    buy_list.append({
        'trigger_net_value': round(float(extra_nv), decimals),
        'share': round(float(core_shares_budget * 0.25), 2),
        'role': 'core',
        'note': 'Q20极端加仓-底仓',
    })

    buy_list = enforce_buy_tier_gap(buy_list, decimals=decimals)
    # 按净值降序后：强制最低 2 档为 core，其余为 float（防 gap 调整打乱语义）
    buy_list = sorted(buy_list, key=lambda x: x['trigger_net_value'], reverse=True)
    n_core = 2
    for i, b in enumerate(buy_list):
        if i >= len(buy_list) - n_core:
            b['role'] = 'core'
            b['note'] = '底仓-不设止盈' if i == len(buy_list) - 1 else 'Q20极端加仓-底仓'
        else:
            b['role'] = 'float'
            b['note'] = '浮动网格'

    buy_list = apply_share_jitter(buy_list, seed_base=101, enabled=jitter)

    # 重新统计浮动份额（jitter 后）
    float_share_total = sum(b['share'] for b in buy_list if b.get('role') == 'float')

    sell_nvs = sorted(np.linspace(sell_low, sell_high, sell_tier_count))
    # 卖出份额均分浮动仓；最高档可略多
    sell_weights = np.linspace(1.0, 1.4, sell_tier_count)
    sell_weights = sell_weights / sell_weights.sum()
    sell_list = []
    for i, nv in enumerate(sell_nvs):
        share = float_share_total * sell_weights[i]
        sell_list.append({
            'trigger_net_value': round(float(nv), decimals),
            'share': round(float(share), 2),
            'role': 'float',
            'note': f'浮动止盈(Q90上限参考={nv_q90})',
            'zone': 'float_only',
        })
    sell_list = apply_share_jitter(sell_list, seed_base=202, enabled=jitter)

    meta = {
        'core_ratio': core_ratio,
        'float_ratio': float_ratio,
        'nv_q20': nv_q20,
        'nv_q90': nv_q90,
        'float_share_total': round(float_share_total, 2),
        'core_share_total': round(sum(b['share'] for b in buy_list if b.get('role') == 'core'), 2),
    }
    return buy_list, sell_list, meta


class CoreFloatTrader:
    """底仓+浮动网格交易器：止盈不卖底仓，重置只动浮动买入档。"""

    def __init__(self, buy_list, sell_list, meta):
        self.buy_triggers = buy_list
        self.sell_triggers = sell_list
        self.meta = meta
        self.buy_used = [False] * len(buy_list)
        self.sell_used = [False] * len(sell_list)
        self.conversion_count = 0
        self.buy_count = 0
        self.sell_count = 0
        self.core_shares_bought = 0.0
        self.float_shares_bought = 0.0
        self.float_shares_sold = 0.0

    def check_buys(self, current_nav):
        triggers = []
        for i, t in enumerate(self.buy_triggers):
            if not self.buy_used[i] and current_nav <= t['trigger_net_value']:
                triggers.append((i, t['share'], t.get('role', 'float')))
        return triggers

    def mark_buy(self, idx, share, role):
        self.buy_used[idx] = True
        self.buy_count += 1
        if role == 'core':
            self.core_shares_bought += share
        else:
            self.float_shares_bought += share

    def available_float_shares(self, total_shares):
        """可卖浮动仓 ≈ 总持仓 - 已买入底仓（底仓永不卖）。"""
        core_held = min(self.core_shares_bought, total_shares)
        return max(0.0, total_shares - core_held)

    def check_sells(self, current_nav, available_float):
        triggers = []
        remaining = available_float
        # 高估区也只卖浮动：与平时相同
        for i, t in enumerate(self.sell_triggers):
            if remaining <= 0:
                break
            if self.sell_used[i]:
                continue
            if current_nav >= t['trigger_net_value']:
                sell_share = min(t['share'], remaining)
                if sell_share > 0:
                    triggers.append((i, sell_share))
                    remaining -= sell_share
        return triggers

    def mark_sell(self, idx, share):
        self.sell_used[idx] = True
        self.sell_count += 1
        self.float_shares_sold += share

    def reset_float_buys_below(self, sell_nav):
        for i, t in enumerate(self.buy_triggers):
            if t.get('role') != 'float':
                continue
            if t['trigger_net_value'] < sell_nav:
                self.buy_used[i] = False


def run_strategy(initial_cash: float = 1_000_000,
                 core_ratio: float = 0.40,
                 float_ratio: float = 0.60,
                 buy_tier_count: int = 4,
                 sell_tier_count: int = 3,
                 money_fund_rate: float = 0.02,
                 start_date: str = None,
                 end_date: str = None,
                 nav_data: pd.DataFrame = None,
                 fund_code: str = FUND_CODE,
                 fund_name: str = FUND_NAME,
                 decimals: int = 2,
                 jitter: bool = True,
                 verbose: bool = True):
    if nav_data is None:
        if verbose:
            print("正在加载基金净值数据...")
        nav_data = load_otc_fund_nav(fund_code, fund_name,
                                     start_date='2019-01-01', end_date='2026-12-31',
                                     verbose=verbose)

    if verbose:
        print(f"标的: {fund_code} ({fund_name})")

    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0, slippage=0.0)
    engine.set_price_data({fund_code: nav_data})

    all_dates, warmup_navs, end_date = resolve_backtest_dates(
        nav_data, start_date, end_date, warmup_days=60)
    if len(all_dates) < 30 and verbose:
        print("警告：回测期数据不足")

    ref_nav = float(nav_data.loc[all_dates[0], 'close']) if len(all_dates) else float(warmup_navs.iloc[-1])
    buy_list, sell_list, meta = generate_core_float_grid(
        warmup_navs,
        core_ratio=core_ratio,
        float_ratio=float_ratio,
        buy_tier_count=buy_tier_count,
        sell_tier_count=sell_tier_count,
        initial_cash=initial_cash,
        ref_nav=ref_nav,
        decimals=decimals,
        jitter=jitter,
    )
    trader = CoreFloatTrader(buy_list, sell_list, meta)
    tracker = HoldingLotTracker()
    daily_interest = 1 + money_fund_rate / 365

    if verbose:
        print(f"\n回测区间: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"底仓/浮动: {core_ratio:.0%}/{float_ratio:.0%}")
        print(f"Q20={meta['nv_q20']}, Q90={meta['nv_q90']}")
        print("====买入====")
        for b in buy_list:
            print(f"  ≤{b['trigger_net_value']} 份额={b['share']} [{b['role']}] {b.get('note','')}")
        print("====止盈(仅浮动)====")
        for s in sell_list:
            print(f"  ≥{s['trigger_net_value']} 份额={s['share']}")

    for date in all_dates:
        engine.cash *= daily_interest
        current_nav = nav_data.loc[date, 'close']

        pos = engine.positions.get(fund_code)
        current_shares = pos.shares if pos and pos.shares > 0 else 0.0
        avail_float = trader.available_float_shares(current_shares)

        for trigger_idx, share in trader.check_sells(current_nav, avail_float):
            ok, _ = sell_with_short_hold_penalty(
                engine, tracker, fund_code, date, share)
            if ok:
                actual = engine.trades[-1].shares
                trader.mark_sell(trigger_idx, actual)
                trader.conversion_count += 1
                trader.reset_float_buys_below(sell_list[trigger_idx]['trigger_net_value'])
                # 止盈后允许同档再次触发（浮动循环）
                trader.sell_used[trigger_idx] = False

        for trigger_idx, share, role in trader.check_buys(current_nav):
            if buy_with_lot_track(engine, tracker, fund_code, date, share):
                actual = engine.trades[-1].shares
                trader.mark_buy(trigger_idx, actual, role)
                trader.conversion_count += 1

        engine.record_daily_value(date)

    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="底仓锁利+浮动网格")
    trade_stats = calculate_trade_stats(engine.trades)
    trade_stats['转换次数'] = trader.conversion_count
    trade_stats['买入触发次数'] = trader.buy_count
    trade_stats['止盈触发次数'] = trader.sell_count
    trade_stats['底仓份额'] = round(trader.core_shares_bought, 2)
    trade_stats['浮动买入份额'] = round(trader.float_shares_bought, 2)
    trade_stats['浮动卖出份额'] = round(trader.float_shares_sold, 2)
    trade_stats['短期惩罚费(元)'] = round(tracker.total_penalty, 2)

    buy_hold_final = initial_cash / nav_data.loc[all_dates[0], 'close'] * nav_data.loc[all_dates[-1], 'close']
    buy_hold_return = (buy_hold_final / initial_cash - 1) * 100
    trade_stats['买入持有收益率(%)'] = round(buy_hold_return, 2)
    trade_stats['超额收益(%)'] = round(metrics['累计收益率(%)'] - buy_hold_return, 2)
    mf_final = initial_cash * (1 + money_fund_rate) ** (len(all_dates) / 365)
    trade_stats['纯货币基金收益率(%)'] = round((mf_final / initial_cash - 1) * 100, 2)

    if verbose:
        print_report(metrics, trade_stats, title="策略6-D：底仓锁利+浮动网格")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    _save_gf_params(buy_list, sell_list, output_dir, prefix='gf_core_float_')
    # 增强 CSV（含角色）
    rows = enhance_save_rows(buy_list, sell_list)
    pd.DataFrame(rows).to_csv(
        os.path.join(output_dir, 'gf_core_float_params.csv'),
        index=False, encoding='utf-8-sig')

    engine.gf_buy_list = buy_list
    engine.gf_sell_list = sell_list
    engine.gf_meta = meta
    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
