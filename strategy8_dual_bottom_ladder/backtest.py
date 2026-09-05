# -*- coding: utf-8 -*-
"""事件驱动阶梯回测"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from strategy8_dual_bottom_ladder import config as C
from strategy8_dual_bottom_ladder.ladder import LadderPlan, generate_ladder
from strategy8_dual_bottom_ladder.screening import compute_metrics, dual_bottom, risk_filter
from strategy8_dual_bottom_ladder.valuation import evaluate_valuation


@dataclass
class Trade:
    date: pd.Timestamp
    action: str
    price: float
    amount: float
    shares: float
    reason: str = ''


@dataclass
class BTResult:
    code: str
    name: str
    equity: pd.DataFrame
    trades: List[Trade] = field(default_factory=list)
    total_return: float = 0.0
    max_dd: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    stop_count: int = 0
    terminate_count: int = 0


def _eligible(hist: pd.Series) -> bool:
    m = compute_metrics(hist)
    if not m.get('ok'):
        return False
    val = evaluate_valuation(hist, is_index=False)
    risk = risk_filter(m, {'meta_ok': False})
    bottom = dual_bottom(m, val)
    return risk['pass'] and bottom['status'] in ('confirmed', 'watch')


def backtest_fund(code: str, name: str, nav: pd.Series,
                  start: str, capital: float = None,
                  sector: str = '其他') -> BTResult:
    capital = capital or C.CAPITAL
    s = nav.dropna().astype(float)
    if hasattr(s, 'columns'):
        s = s['close'] if 'close' in s.columns else s.iloc[:, 0]
    s = s.astype(float)
    start_ts = pd.Timestamp(start)
    idx = s.index[s.index >= start_ts]
    empty = BTResult(code, name, pd.DataFrame())
    if len(idx) < 60:
        return empty

    cash, shares, avg_cost = float(capital), 0.0, 0.0
    plan: Optional[LadderPlan] = None
    state = 'idle'
    age = days_rebal = cooldown = next_tier = 0
    tp_hit = [False, False, False]
    peak = 0.0
    trades: List[Trade] = []
    stop_n = term_n = 0
    rows = []

    def sell(price, ratio, reason, date):
        nonlocal cash, shares, avg_cost
        qty = shares * ratio
        if qty <= 0:
            return
        cash += qty * price
        shares -= qty
        trades.append(Trade(date, 'sell', price, qty * price, qty, reason))
        if shares <= 1e-9:
            shares = 0.0
            avg_cost = 0.0

    for date in idx:
        price = float(s.loc[date])
        stopped = False
        days_rebal += 1
        if cooldown > 0:
            cooldown -= 1

        if days_rebal >= C.REBAL_DAYS:
            days_rebal = 0
            hist = s[s.index <= date]
            if state == 'idle' and cooldown == 0 and _eligible(hist):
                m = compute_metrics(hist)
                plan = generate_ladder(code, name, price, m.get('vol', 0.2),
                                       sector=sector, planned_capital=capital)
                state, age, next_tier = 'active', 0, 0
                tp_hit = [False, False, False]
                peak = 0.0
            elif state == 'active' and shares == 0:
                m = compute_metrics(hist)
                if m.get('ok') and m['pct3'] > 0.30:
                    state, plan = 'idle', None

        if plan is not None and state == 'active':
            age += 1
            if age > plan.valid_days:
                state = 'expired'

        if plan is not None and state == 'active':
            while next_tier < len(plan.buy_tiers):
                tier = plan.buy_tiers[next_tier]
                if price <= tier.trigger_nav:
                    amt = capital * tier.weight
                    if amt > cash:
                        amt = cash
                    if amt > 1:
                        qty = amt / price
                        cash -= amt
                        total = avg_cost * shares + amt
                        shares += qty
                        avg_cost = total / shares if shares else 0
                        trades.append(Trade(date, 'buy', price, amt, qty,
                                            f'第{tier.idx+1}档'))
                        next_tier += 1
                    else:
                        break
                else:
                    break
            if price <= plan.termination_nav and state == 'active':
                state = 'terminated'
                term_n += 1
                cooldown = C.COOLDOWN
                if shares > 0 and price <= avg_cost * (1 - plan.hard_stop_pct):
                    sell(price, 1.0, '硬止损(终止)', date)
                    stop_n += 1
                    stopped = True

        if shares > 0 and plan is not None:
            peak = max(peak, price)
            for j, tp in enumerate(plan.tp_tiers):
                if j < len(tp_hit) and not tp_hit[j] and price >= avg_cost * (1 + tp.gain_pct):
                    sell(price, tp.sell_ratio, f'止盈{j+1}', date)
                    tp_hit[j] = True
            if tp_hit[0] and shares > 0 and price <= peak * (1 - plan.trail_pct):
                sell(price, 1.0, '移动止盈', date)
            if shares > 0 and price <= avg_cost * (1 - plan.hard_stop_pct):
                sell(price, 1.0, '硬止损', date)
                stop_n += 1
                stopped = True

        if shares == 0 and plan is not None and state in ('active', 'terminated', 'expired'):
            pending = next_tier < len(plan.buy_tiers) and state == 'active'
            if not pending:
                cooldown = max(cooldown, C.COOLDOWN if stopped else C.REBAL_DAYS)
                state, plan = 'idle', None

        total = cash + shares * price
        rows.append({'date': date, 'total_value': total, 'cash': cash,
                     'position_value': shares * price})

    eq = pd.DataFrame(rows).set_index('date') if rows else pd.DataFrame()
    if len(eq):
        final = float(eq['total_value'].iloc[-1])
        ret = final / capital - 1
        dd = float((eq['total_value'] / eq['total_value'].cummax() - 1).min())
    else:
        final, ret, dd = capital, 0.0, 0.0

    return BTResult(
        code=code, name=name, equity=eq, trades=trades,
        total_return=ret, max_dd=dd,
        n_buys=sum(1 for t in trades if t.action == 'buy'),
        n_sells=sum(1 for t in trades if t.action == 'sell'),
        stop_count=stop_n, terminate_count=term_n,
    )


def aggregate(results: List[BTResult]) -> pd.DataFrame:
    frames = []
    for r in results:
        if r.equity is None or not len(r.equity):
            continue
        frames.append(r.equity[['total_value']].rename(columns={'total_value': r.code}))
    if not frames:
        return pd.DataFrame()
    port = pd.concat(frames, axis=1).sort_index().ffill().bfill()
    port['total_value'] = port.sum(axis=1)
    port['nav'] = port['total_value'] / port['total_value'].iloc[0]
    return port
