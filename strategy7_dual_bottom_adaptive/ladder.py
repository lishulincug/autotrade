# -*- coding: utf-8 -*-
"""
策略7 · 波动率自适应阶梯抄底 + 动态终止/止盈
=============================================

升级点（相对固定 -10%/-20%/-30% 阶梯）：
  1. 档位间距 = 基准间距 × (基金年化波动 / 参考波动)，并夹在 [4%, 14%]：
     高波动品种档位更宽（防接飞刀），低波动品种档位更密（防踏空）。
  2. 高波动品种自动增加到 5 档，资金 deeper 更重。
  3. 动态终止：跌破最深买档再跌 1 个间距 → 暂停加仓（疑似基本面恶化，转人工复检）；
     相对持仓成本跌破硬止损 → 无条件清仓。
  4. 动态止盈：分批止盈（间距同样随波动自适应）+ 浮盈后移动止盈（峰值回撤 trail）。
  5. 条件单有效期 180 个交易日，到期未触发自动失效，需重新跑筛选。
  6. 单标的仓位上限 15%（行业指数防单一赛道深套）。
"""
from dataclasses import dataclass, field
from typing import List
import numpy as np

# ---------------- 阶梯参数（结构化常量） ----------------
LADDER_CFG = {
    'vol_ref': 0.20,          # 参考年化波动（20%）
    'spacing_base': 0.07,     # 基准档距 7%
    'spacing_min': 0.04,      # 低波动最密 4%
    'spacing_max': 0.14,      # 高波动最宽 14%
    'vol_high': 0.35,         # 年化波动 ≥35% 用 5 档
    'weights_4': [0.15, 0.20, 0.30, 0.35],          # 4档资金占比（越深越重）
    'weights_5': [0.10, 0.15, 0.25, 0.25, 0.25],    # 5档资金占比
    'tp_step_factor': 1.8,    # 止盈步长 = 1.8 × 档距
    'tp_step_min': 0.08,
    'tp_step_max': 0.30,
    'tp_gains': [1.0, 2.2, 3.5],    # 三档止盈相对成本的倍数
    'tp_sell_ratio': [1/3, 1/3, 1.0],  # 每档卖出持仓比例（最后一档清剩余）
    'hard_stop_factor': 2.5,  # 硬止损 = 2.5 × 档距
    'hard_stop_min': 0.12,
    'hard_stop_max': 0.35,
    'trail_factor': 1.2,      # 移动止盈回撤 = 1.2 × 档距
    'trail_min': 0.06,
    'trail_max': 0.18,
    'valid_days': 180,        # 条件单有效期（交易日）
    'max_position_pct': 0.15, # 单标的占权益资金上限
}


def volatility_regime(ann_vol: float) -> str:
    """波动档位：低波动 / 中波动 / 高波动"""
    if np.isnan(ann_vol):
        return '未知'
    if ann_vol < 0.15:
        return '低波动'
    if ann_vol < LADDER_CFG['vol_high']:
        return '中波动'
    return '高波动'


@dataclass
class BuyTier:
    """买入条件单档位"""
    idx: int
    trigger_nav: float       # 触发净值（低于现价）
    drop_pct: float          # 较现价跌幅
    weight: float            # 投入资金占该标的计划资金比例
    cum_weight: float        # 累计投入比例


@dataclass
class TpTier:
    """止盈条件单档位（相对持仓综合成本）"""
    idx: int
    gain_pct: float          # 相对成本涨幅
    trigger_nav: float       # 触发净值（基于全部买档成交后的综合成本测算）
    sell_ratio: float        # 卖出持仓比例


@dataclass
class LadderPlan:
    """一只基金的完整条件单计划"""
    code: str
    name: str
    nav0: float                       # 生成时当前净值
    ann_vol: float
    vol_regime: str
    spacing: float                    # 档距
    n_tiers: int
    buy_tiers: List[BuyTier] = field(default_factory=list)
    tp_tiers: List[TpTier] = field(default_factory=list)
    blended_cost: float = 0.0         # 全部买档成交后的综合成本
    termination_nav: float = 0.0      # 终止线（暂停加仓+复检）
    hard_stop_pct: float = 0.0        # 硬止损幅度（相对成本）
    hard_stop_nav: float = 0.0        # 硬止损净值（基于综合成本）
    trail_pct: float = 0.0            # 移动止盈回撤幅度
    valid_days: int = LADDER_CFG['valid_days']
    max_position_pct: float = LADDER_CFG['max_position_pct']

    def order_rows(self) -> List[dict]:
        """展开为条件单清单（CSV/表格用）"""
        rows = []
        for t in self.buy_tiers:
            rows.append({
                '条件单类型': f'买入第{t.idx+1}档',
                '触发净值': round(t.trigger_nav, 4),
                '较现价幅度': f"{t.drop_pct:+.1%}",
                '资金比例': f"{t.weight:.0%}",
                '累计投入': f"{t.cum_weight:.0%}",
                '说明': f"净值≤{t.trigger_nav:.4f} 买入（跌{abs(t.drop_pct):.0%}）",
            })
        for t in self.tp_tiers:
            rows.append({
                '条件单类型': f'止盈第{t.idx+1}档',
                '触发净值': round(t.trigger_nav, 4),
                '较现价幅度': f"{(t.trigger_nav/self.nav0 - 1):+.1%}",
                '资金比例': f"卖出{t.sell_ratio:.0%}持仓",
                '累计投入': '-',
                '说明': f"买入档成交后挂出：综合成本约{self.blended_cost:.4f}，"
                        f"成本×(1+{t.gain_pct:.0%})卖出{t.sell_ratio:.0%}持仓",
            })
        rows.append({
            '条件单类型': '⛔ 终止线',
            '触发净值': round(self.termination_nav, 4),
            '较现价幅度': f"{(self.termination_nav/self.nav0 - 1):+.1%}",
            '资金比例': '-', '累计投入': '-',
            '说明': '跌破则暂停一切加仓，转基本面复检（疑基本面恶化）',
        })
        rows.append({
            '条件单类型': '🛑 硬止损',
            '触发净值': round(self.hard_stop_nav, 4),
            '较现价幅度': f"{(self.hard_stop_nav/self.nav0 - 1):+.1%}",
            '资金比例': '清仓', '累计投入': '-',
            '说明': f"相对综合成本-{self.hard_stop_pct:.0%} 无条件清仓",
        })
        rows.append({
            '条件单类型': '📉 移动止盈',
            '触发净值': f"峰值×{(1-self.trail_pct):.2f}",
            '较现价幅度': '-', '资金比例': '清仓', '累计投入': '-',
            '说明': f"首档止盈激活后，自最高点回撤{self.trail_pct:.0%} 离场",
        })
        return rows


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


def generate_adaptive_ladder(code: str, name: str, nav0: float, ann_vol: float,
                             cfg: dict = LADDER_CFG) -> LadderPlan:
    """
    根据当前净值与年化波动率生成自适应条件单计划。
    :param nav0: 当前净值（条件单全部挂在现价下方）
    :param ann_vol: 近1年年化波动率
    """
    vol = ann_vol if not np.isnan(ann_vol) else cfg['vol_ref']

    # 1) 档距随波动自适应
    spacing = _clip(cfg['spacing_base'] * vol / cfg['vol_ref'],
                    cfg['spacing_min'], cfg['spacing_max'])

    # 2) 高波动 5 档，其余 4 档
    n_tiers = 5 if vol >= cfg['vol_high'] else 4
    weights = cfg['weights_5'] if n_tiers == 5 else cfg['weights_4']

    buy_tiers = []
    cum = 0.0
    for i, w in enumerate(weights):
        drop = (i + 1) * spacing
        trigger = nav0 * (1 - drop)
        cum += w
        buy_tiers.append(BuyTier(idx=i, trigger_nav=trigger, drop_pct=-drop,
                                 weight=w, cum_weight=cum))

    # 全部成交后的综合成本（资金加权）
    blended_cost = sum(t.trigger_nav * t.weight for t in buy_tiers)

    # 3) 止盈档（相对综合成本，步长随波动自适应）
    tp_step = _clip(cfg['tp_step_factor'] * spacing, cfg['tp_step_min'], cfg['tp_step_max'])
    tp_tiers = []
    for i, (g, ratio) in enumerate(zip(cfg['tp_gains'], cfg['tp_sell_ratio'])):
        gain = g * tp_step
        tp_tiers.append(TpTier(idx=i, gain_pct=gain,
                               trigger_nav=blended_cost * (1 + gain),
                               sell_ratio=ratio))

    # 4) 终止线：最深买档再跌 1 个档距
    deepest = buy_tiers[-1].trigger_nav
    termination_nav = deepest * (1 - spacing)

    # 5) 硬止损 / 移动止盈
    hard_stop_pct = _clip(cfg['hard_stop_factor'] * spacing,
                          cfg['hard_stop_min'], cfg['hard_stop_max'])
    trail_pct = _clip(cfg['trail_factor'] * spacing, cfg['trail_min'], cfg['trail_max'])

    return LadderPlan(
        code=code, name=name, nav0=nav0, ann_vol=vol,
        vol_regime=volatility_regime(vol),
        spacing=spacing, n_tiers=n_tiers,
        buy_tiers=buy_tiers, tp_tiers=tp_tiers,
        blended_cost=blended_cost,
        termination_nav=termination_nav,
        hard_stop_pct=hard_stop_pct,
        hard_stop_nav=blended_cost * (1 - hard_stop_pct),
        trail_pct=trail_pct,
    )
