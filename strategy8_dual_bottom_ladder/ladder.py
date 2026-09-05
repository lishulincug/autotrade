# -*- coding: utf-8 -*-
"""波动自适应三档阶梯：25/35/40，间距=0.5×vol"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from strategy8_dual_bottom_ladder import config as C


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class BuyTier:
    idx: int
    trigger_nav: float
    drop_pct: float
    weight: float
    cum_weight: float


@dataclass
class TpTier:
    idx: int
    gain_pct: float
    trigger_nav: float
    sell_ratio: float


@dataclass
class LadderPlan:
    code: str
    name: str
    nav0: float
    ann_vol: float
    spacing: float
    buy_tiers: List[BuyTier] = field(default_factory=list)
    tp_tiers: List[TpTier] = field(default_factory=list)
    blended_cost: float = 0.0
    termination_nav: float = 0.0
    hard_stop_pct: float = 0.0
    hard_stop_nav: float = 0.0
    trail_pct: float = 0.0
    valid_days: int = C.VALID_DAYS
    max_position_pct: float = C.POS_MAX
    sector: str = '其他'
    planned_capital: float = 0.0
    note: str = ''

    def order_rows(self) -> List[Dict]:
        rows = []
        for t in self.buy_tiers:
            rows.append({
                '基金代码': self.code, '基金简称': self.name, '行业': self.sector,
                '条件单类型': f'买入第{t.idx + 1}档',
                '触发净值': round(t.trigger_nav, 4),
                '较现价幅度': f'{t.drop_pct:+.1%}',
                '资金比例': f'{t.weight:.0%}',
                '累计投入': f'{t.cum_weight:.0%}',
                '计划金额': round(self.planned_capital * t.weight, 2),
                '说明': f'净值≤{t.trigger_nav:.4f}买入（跌{abs(t.drop_pct):.0%}）' + (f'；{self.note}' if self.note else ''),
                '有效期交易日': self.valid_days,
                '年化波动': round(self.ann_vol, 4), '档距': round(self.spacing, 4),
            })
        for t in self.tp_tiers:
            rows.append({
                '基金代码': self.code, '基金简称': self.name, '行业': self.sector,
                '条件单类型': f'止盈第{t.idx + 1}档',
                '触发净值': round(t.trigger_nav, 4),
                '较现价幅度': f'{(t.trigger_nav / self.nav0 - 1):+.1%}',
                '资金比例': f'卖出{t.sell_ratio:.0%}持仓',
                '累计投入': '-', '计划金额': '-',
                '说明': f'成本约{self.blended_cost:.4f}，涨{t.gain_pct:.0%}卖{t.sell_ratio:.0%}',
                '有效期交易日': self.valid_days,
                '年化波动': round(self.ann_vol, 4), '档距': round(self.spacing, 4),
            })
        rows.append({
            '基金代码': self.code, '基金简称': self.name, '行业': self.sector,
            '条件单类型': '⛔ 终止线', '触发净值': round(self.termination_nav, 4),
            '较现价幅度': f'{(self.termination_nav / self.nav0 - 1):+.1%}',
            '资金比例': '-', '累计投入': '-', '计划金额': '-',
            '说明': '跌破暂停加仓，转基本面复检',
            '有效期交易日': self.valid_days,
            '年化波动': round(self.ann_vol, 4), '档距': round(self.spacing, 4),
        })
        rows.append({
            '基金代码': self.code, '基金简称': self.name, '行业': self.sector,
            '条件单类型': '🛑 硬止损', '触发净值': round(self.hard_stop_nav, 4),
            '较现价幅度': f'{(self.hard_stop_nav / self.nav0 - 1):+.1%}',
            '资金比例': '清仓', '累计投入': '-', '计划金额': '-',
            '说明': f'相对成本-{self.hard_stop_pct:.0%}清仓',
            '有效期交易日': self.valid_days,
            '年化波动': round(self.ann_vol, 4), '档距': round(self.spacing, 4),
        })
        rows.append({
            '基金代码': self.code, '基金简称': self.name, '行业': self.sector,
            '条件单类型': '📉 移动止盈',
            '触发净值': f'峰值×{(1 - self.trail_pct):.2f}',
            '较现价幅度': '-', '资金比例': '清仓', '累计投入': '-', '计划金额': '-',
            '说明': f'首档止盈后，峰值回撤{self.trail_pct:.0%}离场',
            '有效期交易日': self.valid_days,
            '年化波动': round(self.ann_vol, 4), '档距': round(self.spacing, 4),
        })
        return rows


def generate_ladder(code: str, name: str, nav0: float, ann_vol: float,
                    sector: str = '其他', planned_capital: float = 100000) -> LadderPlan:
    vol = float(ann_vol) if ann_vol is not None and not np.isnan(ann_vol) else 0.20
    spacing = _clip(C.LADDER_K * vol, C.SPACING_MIN, C.SPACING_MAX)
    buys, cum = [], 0.0
    for i, w in enumerate(C.WEIGHTS):
        drop = (i + 1) * spacing
        cum += w
        buys.append(BuyTier(i, nav0 * (1 - drop), -drop, w, cum))
    blended = sum(t.trigger_nav * t.weight for t in buys)
    scale = spacing / 0.07
    tps = []
    for i, (g, ratio) in enumerate(zip(C.TP_GAINS, C.TP_RATIOS)):
        gain = g * max(0.6, min(1.8, scale))
        tps.append(TpTier(i, gain, blended * (1 + gain), ratio))
    deepest = buys[-1].trigger_nav
    stop_pct = _clip(C.STOP_K * spacing, C.STOP_MIN, C.STOP_MAX)
    trail = _clip(C.TRAIL_K * spacing, C.TRAIL_MIN, C.TRAIL_MAX)
    return LadderPlan(
        code=code, name=name, nav0=nav0, ann_vol=vol, spacing=spacing,
        buy_tiers=buys, tp_tiers=tps, blended_cost=blended,
        termination_nav=deepest * (1 - spacing),
        hard_stop_pct=stop_pct, hard_stop_nav=blended * (1 - stop_pct),
        trail_pct=trail, sector=sector, planned_capital=planned_capital,
    )


def plans_to_df(plans: List[LadderPlan]) -> pd.DataFrame:
    rows = []
    for p in plans:
        rows.extend(p.order_rows())
    return pd.DataFrame(rows)
