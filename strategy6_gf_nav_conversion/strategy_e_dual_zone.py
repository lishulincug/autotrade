"""
策略6-E：广发约定净值转换 · 净值分位自适应双区
============================================
用基金自身近 3 年净值历史分位划分交易区间，每月切换规则：

  - 低估区（≤20分位）：只买不卖，买入份额放大，暂停止盈
  - 震荡区（20~80分位）：正常动态网格，买卖双向
  - 高估区（≥80分位）：只卖不买，分批止盈

比原 B 更能规避高位接盘/低位卖飞；完全不依赖外部 PE/PB。
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
    FUND_CODE, FUND_NAME, resolve_backtest_dates, _save_gf_params, generate_grid_params
)
from strategy6_gf_nav_conversion.gf_constraints import (
    enforce_buy_tier_gap, apply_share_jitter, HoldingLotTracker,
    buy_with_lot_track, sell_with_short_hold_penalty, nav_percentile, enhance_save_rows
)


ZONE_UNDER = 'under'   # 低估：只买
ZONE_MID = 'mid'       # 震荡：网格
ZONE_OVER = 'over'     # 高估：只卖


def classify_zone(pct: float, under=0.20, over=0.80) -> str:
    if pct <= under:
        return ZONE_UNDER
    if pct >= over:
        return ZONE_OVER
    return ZONE_MID


def build_zone_grid(nav_window: pd.Series,
                    zone: str,
                    buy_low_q=0.15,
                    buy_high_q=0.55,
                    sell_low_q=0.70,
                    sell_high_q=0.90,
                    buy_tiers=4,
                    sell_tiers=3,
                    base_share=50000,
                    share_increment=25000,
                    sell_share=50000,
                    under_share_mult=1.5,
                    decimals=2,
                    jitter=True):
    """按区间生成买卖档。低估放大买入、清空卖出；高估清空买入。"""
    if zone == ZONE_UNDER:
        buy_list, _ = generate_grid_params(
            nav_window,
            buy_low_quantile=buy_low_q,
            buy_high_quantile=min(buy_high_q + 0.15, 0.70),
            sell_low_quantile=sell_low_q,
            sell_high_quantile=sell_high_q,
            buy_tier_count=buy_tiers,
            sell_tier_count=sell_tiers,
            base_share=int(base_share * under_share_mult),
            share_increment=int(share_increment * under_share_mult),
            sell_share=sell_share,
            decimals=decimals,
        )
        for b in buy_list:
            b['role'] = 'buy'
            b['zone'] = zone
            b['note'] = '低估区只买'
            b['share'] = round(float(b['share']), 2)
        sell_list = []
    elif zone == ZONE_OVER:
        _, sell_list = generate_grid_params(
            nav_window,
            buy_low_quantile=buy_low_q,
            buy_high_quantile=buy_high_q,
            sell_low_quantile=sell_low_q,
            sell_high_quantile=sell_high_q,
            buy_tier_count=buy_tiers,
            sell_tier_count=sell_tiers,
            base_share=base_share,
            share_increment=share_increment,
            sell_share=sell_share,
            decimals=decimals,
        )
        buy_list = []
        for s in sell_list:
            s['role'] = 'sell'
            s['zone'] = zone
            s['note'] = '高估区只卖'
            s['share'] = round(float(s['share']), 2)
    else:
        buy_list, sell_list = generate_grid_params(
            nav_window,
            buy_low_quantile=buy_low_q,
            buy_high_quantile=buy_high_q,
            sell_low_quantile=sell_low_q,
            sell_high_quantile=sell_high_q,
            buy_tier_count=buy_tiers,
            sell_tier_count=sell_tiers,
            base_share=base_share,
            share_increment=share_increment,
            sell_share=sell_share,
            decimals=decimals,
        )
        for b in buy_list:
            b['role'] = 'buy'
            b['zone'] = zone
            b['note'] = '震荡区网格'
            b['share'] = round(float(b['share']), 2)
        for s in sell_list:
            s['role'] = 'sell'
            s['zone'] = zone
            s['note'] = '震荡区止盈'
            s['share'] = round(float(s['share']), 2)

    if buy_list:
        buy_list = enforce_buy_tier_gap(buy_list, decimals=decimals)
        buy_list = apply_share_jitter(buy_list, seed_base=303, enabled=jitter)
    if sell_list:
        sell_list = apply_share_jitter(sell_list, seed_base=404, enabled=jitter)
    return buy_list, sell_list


class DualZoneTrader:
    def __init__(self):
        self.buy_triggers = []
        self.sell_triggers = []
        self.buy_used = []
        self.sell_used = []
        self.zone = ZONE_MID
        self.percentile = 0.5
        self.conversion_count = 0
        self.buy_count = 0
        self.sell_count = 0

    def set_grid(self, buy_list, sell_list, zone, pct):
        # 换区时重置触发状态
        zone_changed = zone != self.zone
        self.buy_triggers = buy_list
        self.sell_triggers = sell_list
        self.zone = zone
        self.percentile = pct
        if zone_changed or len(self.buy_used) != len(buy_list):
            self.buy_used = [False] * len(buy_list)
        if zone_changed or len(self.sell_used) != len(sell_list):
            self.sell_used = [False] * len(sell_list)

    def check_buys(self, current_nav):
        if self.zone == ZONE_OVER:
            return []
        triggers = []
        for i, t in enumerate(self.buy_triggers):
            if not self.buy_used[i] and current_nav <= t['trigger_net_value']:
                triggers.append((i, t['share']))
        return triggers

    def check_sells(self, current_nav, available_shares):
        if self.zone == ZONE_UNDER:
            return []
        triggers = []
        remaining = available_shares
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

    def mark_buy(self, idx):
        self.buy_used[idx] = True
        self.buy_count += 1

    def mark_sell(self, idx):
        self.sell_used[idx] = True
        self.sell_count += 1

    def reset_buys_below(self, sell_nav):
        for i, t in enumerate(self.buy_triggers):
            if t['trigger_net_value'] < sell_nav:
                self.buy_used[i] = False


def _month_key(date) -> str:
    return f"{date.year}-{date.month:02d}"


def run_strategy(initial_cash: float = 1_000_000,
                 lookback_years: float = 3.0,
                 under_threshold: float = 0.20,
                 over_threshold: float = 0.80,
                 under_share_mult: float = 1.5,
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

    # 需要足够历史算近3年分位：warmup 至少 60，实际分位用 lookback
    all_dates, _, end_date = resolve_backtest_dates(
        nav_data, start_date, end_date, warmup_days=60)
    if len(all_dates) < 30 and verbose:
        print("警告：回测期数据不足")

    trader = DualZoneTrader()
    tracker = HoldingLotTracker()
    daily_interest = 1 + money_fund_rate / 365
    lookback_days = int(lookback_years * 252)

    current_month = None
    final_buy_list, final_sell_list = [], []
    zone_history = []

    if verbose:
        print(f"\n回测区间: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')}")
        print(f"分位回看≈{lookback_years}年，低估≤{under_threshold:.0%} / 高估≥{over_threshold:.0%}")

    for date in all_dates:
        engine.cash *= daily_interest
        current_nav = float(nav_data.loc[date, 'close'])

        hist = nav_data['close'].loc[:date].tail(lookback_days)
        if len(hist) < 40:
            engine.record_daily_value(date)
            continue

        pct = nav_percentile(hist, current_nav)
        zone = classify_zone(pct, under_threshold, over_threshold)
        mk = _month_key(date)

        # 每月首个交易日或跨区时重算网格
        need_refresh = (current_month != mk) or (zone != trader.zone) or not trader.buy_triggers and not trader.sell_triggers
        if need_refresh:
            current_month = mk
            buy_list, sell_list = build_zone_grid(
                hist, zone,
                under_share_mult=under_share_mult,
                decimals=decimals,
                jitter=jitter,
            )
            trader.set_grid(buy_list, sell_list, zone, pct)
            final_buy_list, final_sell_list = buy_list, sell_list
            zone_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'percentile': round(pct, 4),
                'zone': zone,
                'buy_tiers': len(buy_list),
                'sell_tiers': len(sell_list),
            })
        else:
            trader.percentile = pct

        pos = engine.positions.get(fund_code)
        current_shares = pos.shares if pos and pos.shares > 0 else 0.0

        for trigger_idx, share in trader.check_sells(current_nav, current_shares):
            ok, _ = sell_with_short_hold_penalty(
                engine, tracker, fund_code, date, share)
            if ok:
                trader.mark_sell(trigger_idx)
                trader.conversion_count += 1
                if trader.zone == ZONE_MID and trader.sell_triggers:
                    trader.reset_buys_below(
                        trader.sell_triggers[trigger_idx]['trigger_net_value'])
                # 允许同档循环
                trader.sell_used[trigger_idx] = False

        for trigger_idx, share in trader.check_buys(current_nav):
            if buy_with_lot_track(engine, tracker, fund_code, date, share):
                trader.mark_buy(trigger_idx)
                trader.conversion_count += 1

        engine.record_daily_value(date)

    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="净值分位自适应双区")
    trade_stats = calculate_trade_stats(engine.trades)
    trade_stats['转换次数'] = trader.conversion_count
    trade_stats['买入触发次数'] = trader.buy_count
    trade_stats['止盈触发次数'] = trader.sell_count
    trade_stats['当前区间'] = trader.zone
    trade_stats['当前分位'] = round(trader.percentile, 4)
    trade_stats['短期惩罚费(元)'] = round(tracker.total_penalty, 2)

    buy_hold_final = initial_cash / nav_data.loc[all_dates[0], 'close'] * nav_data.loc[all_dates[-1], 'close']
    buy_hold_return = (buy_hold_final / initial_cash - 1) * 100
    trade_stats['买入持有收益率(%)'] = round(buy_hold_return, 2)
    trade_stats['超额收益(%)'] = round(metrics['累计收益率(%)'] - buy_hold_return, 2)
    mf_final = initial_cash * (1 + money_fund_rate) ** (len(all_dates) / 365)
    trade_stats['纯货币基金收益率(%)'] = round((mf_final / initial_cash - 1) * 100, 2)

    if verbose:
        print_report(metrics, trade_stats, title="策略6-E：净值分位自适应双区")
        print(f"最新区间: {trader.zone} 分位={trader.percentile:.2%}")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    # 标注 zone 到参数
    for b in final_buy_list:
        b.setdefault('zone', trader.zone)
    for s in final_sell_list:
        s.setdefault('zone', trader.zone)
    _save_gf_params(final_buy_list, final_sell_list, output_dir, prefix='gf_dual_zone_')
    pd.DataFrame(enhance_save_rows(final_buy_list, final_sell_list)).to_csv(
        os.path.join(output_dir, 'gf_dual_zone_params.csv'),
        index=False, encoding='utf-8-sig')
    if zone_history:
        pd.DataFrame(zone_history).to_csv(
            os.path.join(output_dir, 'gf_dual_zone_history.csv'),
            index=False, encoding='utf-8-sig')

    engine.gf_buy_list = final_buy_list
    engine.gf_sell_list = final_sell_list
    engine.gf_zone = trader.zone
    engine.gf_percentile = trader.percentile
    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
