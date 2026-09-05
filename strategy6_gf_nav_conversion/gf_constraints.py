"""
广发约定净值转换 · 共用约束与费用辅助
============================================
- 相邻买入档净值差 ≥3%（规避 7 天内连环转换）
- 非整数份额微调（避开整数档位拥堵）
- C 类持有＜7 天 1.5% 惩罚赎回费（FIFO 批次）
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


MIN_BUY_GAP_PCT = 0.03
SHORT_HOLD_DAYS = 7
SHORT_HOLD_PENALTY = 0.015


def enforce_buy_tier_gap(buy_list: List[dict],
                        min_gap_pct: float = MIN_BUY_GAP_PCT,
                        decimals: int = 2) -> List[dict]:
    """
    保证买入档按净值降序时，相邻档相对差距 ≥ min_gap_pct。
    从最高档向下推：下一档 ≤ 上一档 * (1 - min_gap_pct)。
    """
    if not buy_list:
        return buy_list
    ordered = sorted(buy_list, key=lambda x: x['trigger_net_value'], reverse=True)
    fixed = []
    prev = None
    for item in ordered:
        nv = float(item['trigger_net_value'])
        if prev is not None:
            max_allowed = prev * (1.0 - min_gap_pct)
            if nv > max_allowed:
                nv = max_allowed
        nv = round(nv, decimals)
        # 净值不能非正
        if nv <= 0:
            nv = round(max(prev * (1.0 - min_gap_pct), 10 ** (-decimals)), decimals) if prev else nv
        new_item = dict(item)
        new_item['trigger_net_value'] = nv
        fixed.append(new_item)
        prev = nv
    return fixed


def jitter_shares(share: float, seed: int = 0, enabled: bool = True) -> float:
    """轻微非整数化份额；enabled=False 时原样返回。"""
    if not enabled:
        return float(share)
    rng = np.random.default_rng(abs(int(seed)) % (2**31))
    # 在 ±1.5% 内微调，保留 2 位小数
    factor = 1.0 + float(rng.uniform(-0.015, 0.015))
    return round(float(share) * factor, 2)


def apply_share_jitter(items: List[dict], seed_base: int = 42, enabled: bool = True) -> List[dict]:
    out = []
    for i, item in enumerate(items):
        new_item = dict(item)
        new_item['share'] = jitter_shares(item['share'], seed=seed_base + i * 17, enabled=enabled)
        out.append(new_item)
    return out


@dataclass
class Lot:
    date: pd.Timestamp
    shares: float


class HoldingLotTracker:
    """按 FIFO 跟踪买入批次，卖出时计算＜7天惩罚费。"""

    def __init__(self, short_days: int = SHORT_HOLD_DAYS,
                 penalty_rate: float = SHORT_HOLD_PENALTY):
        self.lots: Deque[Lot] = deque()
        self.short_days = short_days
        self.penalty_rate = penalty_rate
        self.total_penalty = 0.0

    def on_buy(self, date: pd.Timestamp, shares: float):
        if shares > 0:
            self.lots.append(Lot(date=date, shares=float(shares)))

    def calc_penalty(self, date: pd.Timestamp, shares: float, price: float) -> float:
        """计算卖出份额对应的惩罚费（不修改批次）。"""
        remain = float(shares)
        penalty = 0.0
        for lot in self.lots:
            if remain <= 0:
                break
            take = min(lot.shares, remain)
            hold_days = (date - lot.date).days
            if hold_days < self.short_days:
                penalty += take * price * self.penalty_rate
            remain -= take
        return penalty

    def on_sell(self, date: pd.Timestamp, shares: float, price: float) -> float:
        """消耗批次并返回惩罚费金额。"""
        remain = float(shares)
        penalty = 0.0
        while remain > 1e-9 and self.lots:
            lot = self.lots[0]
            take = min(lot.shares, remain)
            hold_days = (date - lot.date).days
            if hold_days < self.short_days:
                penalty += take * price * self.penalty_rate
            lot.shares -= take
            remain -= take
            if lot.shares <= 1e-9:
                self.lots.popleft()
        self.total_penalty += penalty
        return penalty


def sell_with_short_hold_penalty(engine, tracker: HoldingLotTracker,
                                 symbol: str, date: pd.Timestamp,
                                 shares: float) -> Tuple[bool, float]:
    """
    卖出并扣除＜7天惩罚费。成功返回 (True, penalty)；失败 (False, 0)。
    惩罚费从 cash 扣除，并累加到最后一笔 Trade.commission。
    """
    price = engine._get_price(symbol, date)
    if price is None:
        return False, 0.0
    if not engine.sell(symbol, date, shares=shares):
        return False, 0.0
    # 实际成交份额以最后一笔交易为准
    last = engine.trades[-1]
    actual_shares = last.shares
    penalty = tracker.on_sell(date, actual_shares, last.price)
    if penalty > 0:
        engine.cash -= penalty
        last.commission = float(last.commission) + penalty
    return True, penalty


def buy_with_lot_track(engine, tracker: HoldingLotTracker,
                       symbol: str, date: pd.Timestamp,
                       shares: float) -> bool:
    """买入并记录批次。"""
    before = len(engine.trades)
    ok = engine.buy(symbol, date, shares=shares)
    if not ok:
        return False
    # 可能因资金不足缩量
    if len(engine.trades) > before:
        actual = engine.trades[-1].shares
        tracker.on_buy(date, actual)
    return True


def nav_percentile(nav_series: pd.Series, current_nav: float) -> float:
    """当前净值在序列中的经验分位 (0~1)。"""
    s = nav_series.dropna()
    if len(s) == 0:
        return 0.5
    return float((s < current_nav).sum() / len(s))


def enhance_save_rows(buy_list: List[dict], sell_list: List[dict]) -> List[dict]:
    """生成带可选 role/zone 字段的 CSV 行。"""
    rows = []
    for i, item in enumerate(buy_list, 1):
        row = {
            '方向': '买入(天天红B→基金)',
            '序号': i,
            '约定净值': item['trigger_net_value'],
            '转换份额': item['share'],
            '角色': item.get('role', ''),
            '区间': item.get('zone', ''),
            '备注': item.get('note', ''),
        }
        rows.append(row)
    for i, item in enumerate(sell_list, 1):
        row = {
            '方向': '止盈(基金→天天红B)',
            '序号': i,
            '约定净值': item['trigger_net_value'],
            '转换份额': item['share'],
            '角色': item.get('role', ''),
            '区间': item.get('zone', ''),
            '备注': item.get('note', ''),
        }
        rows.append(row)
    return rows
