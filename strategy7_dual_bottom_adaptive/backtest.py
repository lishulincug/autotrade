# -*- coding: utf-8 -*-
"""
策略7 · 条件单回测模块（事件驱动）
==================================

模拟实盘条件单工作流：
  1. 每 21 个交易日（月度）重跑筛选：双底确认 + 排雷通过 → 按当日净值生成
     自适应阶梯条件单（买单价位全部在现价下方）。
  2. 日度检查：
     - 净值 ≤ 下一档触发价 → 成交买入（跳空大跌可一天穿多档）；
     - 跌破终止线 → 暂停加仓（状态 terminated），触发硬止损则清仓；
     - 持仓中：分批止盈 / 首档止盈后移动止盈 / 硬止损；
     - 条件单 180 个交易日到期 → 未成交买单作废；
     - 月度复检若分位涨回 30% 以上且尚未建仓 → 条件单撤销（涨出低位）。
  3. 终止/清仓后设 63 个交易日冷却期，避免立刻重新接刀。

场外 C 类份额按净值成交，无申购费；回测不计佣金（与策略6口径一致）。
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np
import pandas as pd

from strategy7_dual_bottom_adaptive.screening import compute_metrics, four_dimension_risk, dual_bottom_confirm
from strategy7_dual_bottom_adaptive.ladder import generate_adaptive_ladder, LadderPlan

REBALANCE_DAYS = 21      # 月度重检
WARMUP_DAYS = 756        # 需要3年历史算分位
COOLDOWN_DAYS = 63       # 终止后冷却期（约3个月）


@dataclass
class BTTrade:
    date: pd.Timestamp
    action: str          # buy / sell
    price: float
    amount: float        # 金额
    shares: float
    reason: str = ''


@dataclass
class FundBacktestResult:
    code: str
    name: str
    equity: pd.DataFrame                 # index=date, total_value/cash/position_value/return
    trades: List[BTTrade] = field(default_factory=list)
    plans_activated: int = 0
    final_value: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    n_buys: int = 0
    n_sells: int = 0
    win_rounds: int = 0
    lose_rounds: int = 0
    avg_hold_days: float = 0.0
    stop_count: int = 0                  # 硬止损次数
    terminate_count: int = 0             # 触发终止线次数


def backtest_fund(code: str, name: str, nav: pd.Series,
                  start_date: str, capital: float = 100_000) -> FundBacktestResult:
    """
    单只基金条件单回测。
    :param nav: 完整净值序列（含 warmup 期）
    :param start_date: 回测开始日期（warmup 数据需在此之前 ≥3 年）
    """
    s = nav.dropna().astype(float)
    start = pd.Timestamp(start_date)
    bt_idx = s.index[s.index >= start]
    if len(bt_idx) < 60:
        return FundBacktestResult(code=code, name=name,
                                  equity=pd.DataFrame({'total_value': []}))

    # --- 状态 ---
    cash = capital
    shares = 0.0
    avg_cost = 0.0
    plan: Optional[LadderPlan] = None
    state = 'idle'          # idle / active / terminated / expired
    age = 0
    days_since_rebal = 0
    cooldown = 0
    next_tier = 0
    tp_hit = [False, False, False]
    peak = 0.0
    trades: List[BTTrade] = []
    plans_activated = 0
    stop_count = 0
    terminate_count = 0
    buy_dates: List[pd.Timestamp] = []

    equity_rows = []
    stopped_today = False

    def _sell(price, ratio, reason, date):
        nonlocal cash, shares, avg_cost
        sell_shares = shares * ratio
        if sell_shares <= 0:
            return
        proceeds = sell_shares * price
        cash += proceeds
        shares -= sell_shares
        trades.append(BTTrade(date, 'sell', price, proceeds, sell_shares, reason))
        if shares <= 1e-9:
            shares = 0.0
            avg_cost = 0.0

    for date in bt_idx:
        price = float(s.loc[date])
        stopped_today = False

        # ---------- 月度重检：新开/撤销条件单 ----------
        days_since_rebal += 1
        if cooldown > 0:
            cooldown -= 1
        if days_since_rebal >= REBALANCE_DAYS:
            days_since_rebal = 0
            hist = s[s.index <= date]
            m = compute_metrics(hist)
            if m.get('data_ok'):
                risk = four_dimension_risk(m)
                bottom = dual_bottom_confirm(m)
                # 新开仓
                if state == 'idle' and cooldown == 0 and risk['pass'] and bottom['status'] == 'confirmed':
                    plan = generate_adaptive_ladder(code, name, price, m['ann_vol'])
                    state = 'active'
                    age = 0
                    next_tier = 0
                    tp_hit = [False, False, False]
                    peak = 0.0
                    plans_activated += 1
                # 未建仓却涨出低位 → 撤销条件单
                elif state == 'active' and shares == 0.0 and m['pct_3y'] > 0.30:
                    state = 'idle'
                    plan = None

        # ---------- 条件单到期 ----------
        if plan is not None and state == 'active':
            age += 1
            if age > plan.valid_days:
                state = 'expired'   # 未成交买单作废，已持仓部分继续管理

        # ---------- 买入触发（可一天穿多档） ----------
        if plan is not None and state == 'active':
            while next_tier < len(plan.buy_tiers):
                tier = plan.buy_tiers[next_tier]
                if price <= tier.trigger_nav:
                    amount = capital * tier.weight
                    buy_shares = amount / price
                    cash -= amount
                    # 更新综合成本
                    total_cost = avg_cost * shares + amount
                    shares += buy_shares
                    avg_cost = total_cost / shares if shares > 0 else 0.0
                    buy_dates.append(date)
                    trades.append(BTTrade(date, 'buy', price, amount, buy_shares,
                                          f'第{tier.idx+1}档买入(跌{abs(tier.drop_pct):.0%})'))
                    next_tier += 1
                else:
                    break

            # 终止线：暂停加仓
            if price <= plan.termination_nav and state == 'active':
                state = 'terminated'
                terminate_count += 1
                cooldown = COOLDOWN_DAYS
                # 同时触发硬止损 → 清仓
                if shares > 0 and price <= avg_cost * (1 - plan.hard_stop_pct):
                    _sell(price, 1.0, '硬止损(基本面恶化)', date)
                    stop_count += 1
                    stopped_today = True

        # ---------- 持仓管理：止盈 / 移动止盈 / 硬止损 ----------
        if shares > 0 and plan is not None:
            peak = max(peak, price)
            # 分批止盈
            for j, tp in enumerate(plan.tp_tiers):
                if not tp_hit[j] and price >= avg_cost * (1 + tp.gain_pct):
                    _sell(price, tp.sell_ratio, f'第{j+1}档止盈(+{tp.gain_pct:.0%})', date)
                    tp_hit[j] = True
            # 移动止盈（首档止盈激活后）
            if tp_hit[0] and shares > 0 and price <= peak * (1 - plan.trail_pct):
                _sell(price, 1.0, f'移动止盈(峰值回撤{plan.trail_pct:.0%})', date)
            # 硬止损
            if shares > 0 and price <= avg_cost * (1 - plan.hard_stop_pct):
                _sell(price, 1.0, '硬止损(跌幅超限)', date)
                stop_count += 1
                stopped_today = True

        # ---------- 清仓后回收计划（仍有未成交买档时条件单继续挂单） ----------
        if shares == 0.0 and plan is not None and state in ('active', 'terminated', 'expired'):
            pending_buys = next_tier < len(plan.buy_tiers) and state == 'active'
            if not pending_buys:
                # 买档已全部成交/作废且仓位清空：止损冷却3个月，正常离场冷却1个月
                cooldown = max(cooldown, COOLDOWN_DAYS if stopped_today else REBALANCE_DAYS)
                state = 'idle'
                plan = None

        # ---------- 记录每日净值 ----------
        total = cash + shares * price
        equity_rows.append({'date': date, 'cash': cash,
                            'position_value': shares * price,
                            'total_value': total, 'price': price})

    eq = pd.DataFrame(equity_rows).set_index('date')
    eq['return'] = eq['total_value'].pct_change().fillna(0.0)

    # --- 统计 ---
    final_value = float(eq['total_value'].iloc[-1]) if len(eq) else capital
    total_return = final_value / capital - 1.0
    cummax = eq['total_value'].cummax()
    max_dd = float(((eq['total_value'] - cummax) / cummax).min()) if len(eq) else 0.0

    # 回合胜负（FIFO 简单匹配：每次清仓算一个回合）
    win_r, lose_r = 0, 0
    hold_days = []
    open_buys = []
    for t in trades:
        if t.action == 'buy':
            open_buys.append(t)
        elif t.action == 'sell' and open_buys:
            # 简化：清仓卖出（ratio=1）或部分卖出均按卖出价 vs 最早买入价判定
            pnl_pct = t.price / open_buys[0].price - 1.0
            if pnl_pct > 0:
                win_r += 1
            else:
                lose_r += 1
            hold_days.append((t.date - open_buys[0].date).days)
            open_buys = open_buys[1:] if t.shares < open_buys[0].shares else []

    n_buys = sum(1 for t in trades if t.action == 'buy')
    n_sells = sum(1 for t in trades if t.action == 'sell')

    return FundBacktestResult(
        code=code, name=name, equity=eq, trades=trades,
        plans_activated=plans_activated,
        final_value=final_value, total_return=total_return,
        max_drawdown=max_dd, n_buys=n_buys, n_sells=n_sells,
        win_rounds=win_r, lose_rounds=lose_r,
        avg_hold_days=float(np.mean(hold_days)) if hold_days else 0.0,
        stop_count=stop_count, terminate_count=terminate_count,
    )


def aggregate_portfolio(results: List[FundBacktestResult], capital_per_fund: float = 100_000):
    """
    等权汇总组合净值（每只基金独立资金，合并为一条权益曲线）。
    返回 DataFrame: total_value, return, nav（归一化=1）
    """
    frames = []
    for r in results:
        if r.equity is None or len(r.equity) == 0:
            continue
        frames.append(r.equity[['total_value']].rename(columns={'total_value': r.code}))
    if not frames:
        return pd.DataFrame()
    port = pd.concat(frames, axis=1).sort_index().ffill().bfill()
    port['total_value'] = port.sum(axis=1)
    port['return'] = port['total_value'].pct_change().fillna(0.0)
    port['nav'] = port['total_value'] / port['total_value'].iloc[0]
    return port
