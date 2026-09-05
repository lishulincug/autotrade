# -*- coding: utf-8 -*-
"""行业仓位截断 ≤ SECTOR_CAP"""
from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd

from strategy8_dual_bottom_ladder import config as C
from strategy8_dual_bottom_ladder.ladder import LadderPlan, generate_ladder, plans_to_df


def apply_sector_cap(selected: pd.DataFrame,
                     capital: float = None,
                     sector_cap: float = None) -> Tuple[pd.DataFrame, List[LadderPlan], pd.DataFrame]:
    capital = capital or C.CAPITAL
    sector_cap = sector_cap or C.SECTOR_CAP
    if selected is None or not len(selected):
        return selected if selected is not None else pd.DataFrame(), [], pd.DataFrame()

    df = selected.copy()
    if 'sector' not in df.columns:
        df['sector'] = '其他'
    df['sector'] = df['sector'].fillna('其他')
    score_col = 'bottom_score' if 'bottom_score' in df.columns else None
    if score_col:
        df = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    total = max(len(df), 1) * capital
    limit = total * sector_cap
    used: Dict[str, float] = {}
    caps = []
    for _, row in df.iterrows():
        sec = row['sector']
        remain = max(0.0, limit - used.get(sec, 0.0))
        cap = min(capital, remain)
        if cap <= 0 and used.get(sec, 0) == 0:
            cap = min(capital * 0.5, limit)
        caps.append(cap)
        used[sec] = used.get(sec, 0.0) + cap
    df['planned_capital'] = caps
    df = df[df['planned_capital'] >= capital * 0.15].reset_index(drop=True)

    plans: List[LadderPlan] = []
    for _, row in df.iterrows():
        nav0 = row.get('nav_now')
        if nav0 is None or pd.isna(nav0):
            continue
        vol = row.get('ann_vol', 0.2)
        plan = generate_ladder(
            code=str(row['code']).zfill(6),
            name=str(row['name']),
            nav0=float(nav0),
            ann_vol=float(vol) if pd.notna(vol) else 0.2,
            sector=str(row.get('sector', '其他')),
            planned_capital=float(row['planned_capital']),
        )
        if row['planned_capital'] < capital * 0.6 and len(plan.buy_tiers) >= 3:
            plan.buy_tiers = plan.buy_tiers[:2]
            tw = sum(t.weight for t in plan.buy_tiers) or 1.0
            cum = 0.0
            for t in plan.buy_tiers:
                t.weight /= tw
                cum += t.weight
                t.cum_weight = cum
            plan.blended_cost = sum(t.trigger_nav * t.weight for t in plan.buy_tiers)
            plan.note = '行业仓位裁剪：仅保留前2档'
        plans.append(plan)

    return df, plans, plans_to_df(plans)
