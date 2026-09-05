"""
策略6-F：广发约定净值转换 · 多标的资金轮动
============================================
中证红利 / 港股互联网 / 北证50 等低相关标的共用天天红B现金池：
  - 每周评估各基金净值分位 + 相对阶段高点跌幅
  - 可动用资金优先分配给分位最低、跌幅最大的标的
  - 高估区转回货币，再轮入低位标的
  - 轮动间隔 ≥7 个交易日，规避 C 类惩罚赎回费

本策略为组合层，不进入单基金 STRATEGIES 矩阵。
"""
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_otc_fund_nav
from common.metrics import calculate_metrics, calculate_trade_stats, print_report
from strategy6_gf_nav_conversion.gf_constraints import (
    HoldingLotTracker, buy_with_lot_track, sell_with_short_hold_penalty,
    nav_percentile, enforce_buy_tier_gap, apply_share_jitter
)


DEFAULT_POOL = [
    ('021400', '广发中证红利ETF发起式联接C'),
    ('021093', '广发中证港股通互联网ETF发起式联接C'),
    ('017513', '广发北证50成份指数C'),
]


def _score_fund(nav_series: pd.Series, window: int = 120) -> dict:
    s = nav_series.dropna().tail(max(window, 40))
    if len(s) < 20:
        return {'percentile': 0.5, 'drawdown': 0.0, 'nav': float(s.iloc[-1]) if len(s) else 1.0}
    cur = float(s.iloc[-1])
    high = float(s.max())
    pct = nav_percentile(s, cur)
    dd = (high - cur) / high if high > 0 else 0.0
    return {'percentile': pct, 'drawdown': dd, 'nav': cur, 'high': high}


def _pick_target(scores: Dict[str, dict],
                 under=0.35, over=0.80) -> Tuple[str, str]:
    """
    返回 (action, code)
    action: 'buy' | 'hold' | 'sell_over' | 'none'
    优先买入分位最低且未高估的标的。
    """
    # 高估列表
    overvalued = [c for c, sc in scores.items() if sc['percentile'] >= over]
    # 候选：分位越低、回撤越大越好
    candidates = [
        (c, sc) for c, sc in scores.items()
        if sc['percentile'] <= under
    ]
    if not candidates:
        # 放宽：取分位最低
        candidates = sorted(scores.items(), key=lambda x: (x[1]['percentile'], -x[1]['drawdown']))
        candidates = candidates[:1]
    else:
        candidates = sorted(candidates, key=lambda x: (x[1]['percentile'], -x[1]['drawdown']))

    best = candidates[0][0] if candidates else None
    return best, overvalued


def run_rotation(initial_cash: float = 1_000_000,
                 fund_pool: List[Tuple[str, str]] = None,
                 money_fund_rate: float = 0.02,
                 review_every: int = 5,  # 约每周
                 min_rotate_days: int = 7,
                 under_threshold: float = 0.35,
                 over_threshold: float = 0.80,
                 deploy_pct: float = 0.70,  # 可部署资金比例，留30%现金底仓
                 score_window: int = 120,
                 start_date: str = None,
                 end_date: str = None,
                 decimals: int = 2,
                 verbose: bool = True):
    """
    多标的共用现金池轮动回测。
    返回: metrics, trade_stats, equity_df, engine, allocation_df
    """
    fund_pool = fund_pool or DEFAULT_POOL
    price_dict = {}
    names = {}
    for code, name in fund_pool:
        nav = load_otc_fund_nav(code, name,
                                start_date='2019-01-01', end_date='2026-12-31',
                                verbose=False)
        if len(nav) < 60:
            if verbose:
                print(f"跳过 {code} 数据不足")
            continue
        price_dict[code] = nav
        names[code] = name

    if len(price_dict) < 2:
        raise RuntimeError("轮动池可用基金不足2只")

    # 公共交易日：取交集
    common_idx = None
    for df in price_dict.values():
        common_idx = df.index if common_idx is None else common_idx.intersection(df.index)
    common_idx = common_idx.sort_values()
    if start_date:
        common_idx = common_idx[common_idx >= pd.Timestamp(start_date)]
    if end_date:
        common_idx = common_idx[common_idx <= pd.Timestamp(end_date)]
    if len(common_idx) < 40:
        raise RuntimeError("公共交易日不足")

    # 预热后开始
    start_i = min(60, max(20, len(common_idx) // 10))
    all_dates = list(common_idx[start_i:])

    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0, slippage=0.0)
    engine.set_price_data(price_dict)
    trackers = {c: HoldingLotTracker() for c in price_dict}
    daily_interest = 1 + money_fund_rate / 365

    last_rotate_date = None
    current_target = None
    allocation_rows = []
    conversion_count = 0
    buy_count = 0
    sell_count = 0

    if verbose:
        print("╔" + "═" * 60 + "╗")
        print("║  广发约定净值转换 · 多标的资金轮动" + " " * 22 + "║")
        print("╚" + "═" * 60 + "╝")
        print(f"标的池: {', '.join(f'{c} {names[c][:8]}' for c in price_dict)}")
        print(f"区间: {all_dates[0].date()} ~ {all_dates[-1].date()} ({len(all_dates)}日)")
        print(f"现金底仓保留: {(1-deploy_pct):.0%}  复盘周期: {review_every}日  最小轮动间隔: {min_rotate_days}日\n")

    for i, date in enumerate(all_dates):
        engine.cash *= daily_interest

        # 非复盘日只记净值
        if i % review_every != 0:
            engine.record_daily_value(date)
            continue

        scores = {}
        for code, df in price_dict.items():
            hist = df['close'].loc[:date]
            scores[code] = _score_fund(hist, window=score_window)

        best, overvalued = _pick_target(scores, under=under_threshold, over=over_threshold)

        # 卖出高估持仓（且距上次轮动 ≥ min_rotate_days）
        can_rotate = (
            last_rotate_date is None
            or (date - last_rotate_date).days >= min_rotate_days
        )

        if can_rotate:
            for code in list(engine.positions.keys()):
                pos = engine.positions.get(code)
                if not pos or pos.shares <= 0:
                    continue
                sc = scores.get(code, {})
                # 高估 或 不是当前最优目标 → 转回货币
                should_exit = (
                    code in overvalued
                    or (best and code != best and sc.get('percentile', 0.5) > under_threshold)
                )
                if should_exit:
                    ok, _ = sell_with_short_hold_penalty(
                        engine, trackers[code], code, date, pos.shares)
                    if ok:
                        sell_count += 1
                        conversion_count += 1
                        last_rotate_date = date
                        if verbose and conversion_count <= 20:
                            print(f"  [{date.date()}] 卖出 {code} 分位={sc.get('percentile',0):.2f} "
                                  f"净值={sc.get('nav',0):.{decimals}f}")

        # 买入最优低估标的（高估不买）
        if can_rotate and best and scores[best]['percentile'] < over_threshold:
            pos = engine.positions.get(best)
            held = pos.shares if pos and pos.shares > 0 else 0.0
            total_val = engine.get_total_value()
            target_value = total_val * deploy_pct
            cur_nav = scores[best]['nav']
            # 已有持仓市值
            held_value = held * cur_nav
            need_value = max(0.0, target_value - held_value)
            # 保留现金底仓
            max_spend = max(0.0, engine.cash - total_val * (1 - deploy_pct))
            spend = min(need_value, max_spend)
            if spend > cur_nav * 100:  # 至少买一点
                shares = round(spend / cur_nav, 2)
                # 轻微 jitter
                shares = round(shares * (1 + (hash(str(date) + best) % 30 - 15) / 1000), 2)
                if buy_with_lot_track(engine, trackers[best], best, date, shares):
                    buy_count += 1
                    conversion_count += 1
                    current_target = best
                    last_rotate_date = date
                    if verbose and conversion_count <= 20:
                        print(f"  [{date.date()}] 买入 {best} 分位={scores[best]['percentile']:.2f} "
                              f"份额={shares} 净值={cur_nav:.{decimals}f}")

        # 记录配置
        row = {'date': date.strftime('%Y-%m-%d'), 'target': current_target or '',
               'cash': round(engine.cash, 2), 'total': round(engine.get_total_value(), 2)}
        for code, sc in scores.items():
            row[f'{code}_pct'] = round(sc['percentile'], 4)
            row[f'{code}_dd'] = round(sc['drawdown'], 4)
            pos = engine.positions.get(code)
            row[f'{code}_shares'] = round(pos.shares, 2) if pos else 0.0
        allocation_rows.append(row)

        engine.record_daily_value(date)

    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="多标的资金轮动")
    trade_stats = calculate_trade_stats(engine.trades)
    trade_stats['转换次数'] = conversion_count
    trade_stats['买入触发次数'] = buy_count
    trade_stats['止盈触发次数'] = sell_count
    trade_stats['短期惩罚费(元)'] = round(sum(t.total_penalty for t in trackers.values()), 2)

    # 等权买入持有基准（近似）
    bh_rets = []
    for code, df in price_dict.items():
        seg = df['close'].reindex(all_dates).dropna()
        if len(seg) > 1:
            bh_rets.append(seg.iloc[-1] / seg.iloc[0] - 1)
    bh_return = float(np.mean(bh_rets) * 100) if bh_rets else 0.0
    trade_stats['买入持有收益率(%)'] = round(bh_return, 2)
    trade_stats['超额收益(%)'] = round(metrics['累计收益率(%)'] - bh_return, 2)

    allocation_df = pd.DataFrame(allocation_rows)

    if verbose:
        print_report(metrics, trade_stats, title="策略6-F：多标的资金轮动")

    output_dir = os.path.dirname(os.path.abspath(__file__))
    alloc_path = os.path.join(output_dir, 'gf_rotation_allocation.csv')
    allocation_df.to_csv(alloc_path, index=False, encoding='utf-8-sig')
    equity_df.to_csv(os.path.join(output_dir, 'gf_rotation_equity.csv'))

    # 输出最新建议买入参数（当前目标的浅网格示意）
    if current_target and current_target in price_dict:
        hist = price_dict[current_target]['close'].loc[:all_dates[-1]].tail(score_window)
        cur = float(hist.iloc[-1])
        buy_list = enforce_buy_tier_gap([
            {'trigger_net_value': round(cur * 0.97, decimals), 'share': 80000, 'role': 'rotate', 'note': '轮动优先买入'},
            {'trigger_net_value': round(cur * 0.94, decimals), 'share': 100000, 'role': 'rotate', 'note': '加深加仓'},
            {'trigger_net_value': round(cur * 0.90, decimals), 'share': 120000, 'role': 'rotate', 'note': '深跌加仓'},
        ], decimals=decimals)
        buy_list = apply_share_jitter(buy_list, seed_base=707, enabled=True)
        sell_list = [{
            'trigger_net_value': round(float(hist.quantile(0.80)), decimals),
            'share': sum(b['share'] for b in buy_list),
            'role': 'rotate',
            'note': '高估区转回货币',
        }]
        engine.gf_buy_list = buy_list
        engine.gf_sell_list = sell_list
        engine.gf_rotation_target = current_target
    else:
        engine.gf_buy_list = []
        engine.gf_sell_list = []

    return metrics, trade_stats, equity_df, engine, allocation_df


if __name__ == '__main__':
    run_rotation()
