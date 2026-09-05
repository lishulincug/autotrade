# -*- coding: utf-8 -*-
"""
策略7 · 筛选模块：双底确认 + 四维量化排雷
==========================================

一、双底确认（避免“越跌越贵”的价值陷阱）
  价格底：当前净值在近3年序列中的百分位 ≤ 15%（强）/ ≤ 30%（观察）
  估值底：开源净值接口不含 PE/PB，用三个可量化代理指标，满足 ≥2 项确认：
          ① 长周期（近5年/成立以来）净值分位 ≤ 30%
          ② 距近3年高点回撤 ≥ 25%（风险释放充分）
          ③ 净值低于250日年线 ≥ 10%
  （若接入理杏仁/中证指数 PE 数据，可把①替换为 PE 分位，接口位置见 _valuation_items）

二、四维排雷（全部为硬门槛，AND 关系；任一维度 FAIL 即剔除）
  D1 成立年限与数据质量：成立 ≥3 年；单日 |涨跌|>10% 的异常跳变 ≤3 次
  D2 清盘/僵尸风险：近2年年化 ≤ -20% 且近1年新低密度 >35%（持续缩水）
                   或 年化波动 <8% 却长期负收益（低波动阴跌僵尸）
  D3 价值陷阱：近252日创历史新低天数占比 >45%（跌了还有新低，无底）
  D4 波动/跟踪崩坏：年化波动 >50%、近60日波动突增 >2倍、或3年最大回撤 >80%

注意：硬门槛只做“剔除/保留”，样本量与形态的平衡通过综合评分（0~100）排序实现，
     不把核心条件放进 OR 逻辑里稀释。
"""
import numpy as np
import pandas as pd

# ---------------- 结构化阈值常量（调参只改这里） ----------------
SCREEN_CFG = {
    # 价格底
    'price_pct_strong': 0.15,   # 3年净值分位 ≤15% → 强价格底
    'price_pct_watch': 0.30,    # 15%~30% → 观察区
    # 估值底（3项代理，满足≥2）
    'long_pct_max': 0.30,       # 长周期分位 ≤30%
    'dd_from_peak_min': 0.25,   # 距高点回撤 ≥25%
    'ma250_dev_max': -0.10,     # 低于年线 ≥10%
    # 企稳信号
    'stabilize_days': 20,       # 近20个交易日
    'stabilize_ret_min': -0.03, # 近20日收益 > -3%
    # D1 数据质量（分层：硬门槛 <2年；2~3年放行但预警）
    'min_history_years': 3.0,
    'min_history_years_hard': 2.0,
    'jump_daily_threshold': 0.30,   # 单日|涨跌|>30% 才计为净值错印（真实极端行情如北证25%不属此列）
    'jump_count_max': 0,
    # D2 清盘/僵尸
    'zombie_trend_ann': -0.20,
    'zombie_newlow_density': 0.35,
    'zombie_vol_max': 0.08,
    'zombie_trend_soft': -0.05,
    # D3 价值陷阱
    'newlow_density_trap': 0.45,
    # D4 波动崩坏
    'vol_extreme': 0.50,
    'vol_surge_ratio': 2.0,
    'max_dd_blowup': 0.80,
    'max_dd_warn': 0.70,
}

TRADING_DAYS_1Y = 252
TRADING_DAYS_3Y = 756
TRADING_DAYS_5Y = 1260


def _pct_rank(window: pd.Series, value: float) -> float:
    """value 在 window 序列中的百分位（0~1）：低于 value 的样本占比"""
    s = window.dropna()
    if len(s) == 0:
        return np.nan
    return float((s < value).sum() / len(s))


def compute_metrics(nav: pd.Series, as_of=None) -> dict:
    """
    计算单只基金的全部筛选指标。
    :param nav: 净值序列（DatetimeIndex, close 列或纯 Series）
    :param as_of: 截止日期（回测按日重算用）；None 表示用全部数据
    """
    s = nav.dropna().astype(float)
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    n = len(s)
    if n < 60:
        return {'data_ok': False, 'n_days': n}

    now = float(s.iloc[-1])
    rets = s.pct_change().dropna()

    def _win(tdays):
        return s.iloc[-tdays:] if n >= tdays else s

    w1y = _win(TRADING_DAYS_1Y)
    w3y = _win(TRADING_DAYS_3Y)

    # --- 价格位置 ---
    pct_1y = _pct_rank(w1y, now)
    pct_3y = _pct_rank(w3y, now)
    # 长周期：满5年用5年，否则用成立以来全部
    w_long = _win(TRADING_DAYS_5Y) if n >= TRADING_DAYS_5Y else s
    pct_long = _pct_rank(w_long, now)

    # --- 回撤 ---
    peak_3y = float(w3y.max())
    dd_from_peak = now / peak_3y - 1.0
    cummax = w3y.cummax()
    max_dd_3y = float((w3y / cummax - 1.0).min())

    # --- 年线偏离 ---
    ma250 = float(s.iloc[-TRADING_DAYS_1Y:].mean())
    dev_ma250 = now / ma250 - 1.0
    # 年线斜率（近20日 vs 20日前）
    ma250_prev = float(s.iloc[-TRADING_DAYS_1Y - 20:-20].mean()) if n >= TRADING_DAYS_1Y + 20 else ma250
    ma250_slope = ma250 / ma250_prev - 1.0 if ma250_prev > 0 else 0.0

    # --- 波动率 ---
    ann_vol = float(rets.iloc[-TRADING_DAYS_1Y:].std() * np.sqrt(252)) if len(rets) >= 60 else np.nan
    vol_60 = float(rets.iloc[-60:].std() * np.sqrt(252)) if len(rets) >= 60 else ann_vol
    vol_surge = float(vol_60 / ann_vol) if ann_vol and ann_vol > 1e-6 else np.nan

    # --- 创新低密度（价值陷阱核心指标）---
    expanding_min = s.cummin()
    is_new_low = (s <= expanding_min)
    newlow_252 = float(is_new_low.iloc[-TRADING_DAYS_1Y:].mean()) if n >= TRADING_DAYS_1Y else float(is_new_low.mean())
    newlow_60 = float(is_new_low.iloc[-60:].mean()) if n >= 60 else np.nan

    # --- 长期趋势（近2年年化）---
    if n >= 505:
        trend_2y = (now / float(s.iloc[-505])) ** (252.0 / 504.0) - 1.0
    else:
        trend_2y = (now / float(s.iloc[0])) ** (252.0 / max(n - 1, 1)) - 1.0

    # --- 异常跳变 ---
    jump_count = int((rets.abs() > SCREEN_CFG['jump_daily_threshold']).sum())

    # --- 企稳信号 ---
    k = SCREEN_CFG['stabilize_days']
    last_k = s.iloc[-k:]
    low_1y = float(w1y.min())
    no_new_low = bool(last_k.min() > low_1y * 1.001)          # 近20日未创1年新低
    ret_k = (now / float(s.iloc[-(k + 1)]) - 1.0) if n > k else 0.0
    momentum_ok = bool(ret_k > SCREEN_CFG['stabilize_ret_min'])
    stabilized = no_new_low and momentum_ok

    return {
        'data_ok': True,
        'n_days': n,
        'years_history': n / 252.0,
        'nav_now': now,
        'pct_1y': pct_1y,
        'pct_3y': pct_3y,
        'pct_long': pct_long,
        'dd_from_peak': dd_from_peak,
        'max_dd_3y': max_dd_3y,
        'ma250': ma250,
        'dev_ma250': dev_ma250,
        'ma250_slope': ma250_slope,
        'ann_vol': ann_vol,
        'vol_60': vol_60,
        'vol_surge': vol_surge,
        'newlow_252': newlow_252,
        'newlow_60': newlow_60,
        'trend_2y': trend_2y,
        'jump_count': jump_count,
        'no_new_low_20': no_new_low,
        'ret_20': ret_k,
        'stabilized': stabilized,
    }


def four_dimension_risk(m: dict, cfg: dict = SCREEN_CFG) -> dict:
    """
    四维量化排雷。返回每个维度的 pass/reasons/score 与总体结论。
    硬门槛 AND 关系：任一维度 FAIL → 剔除。
    warnings 为非阻断预警（放行但扣分+展示）。
    """
    dims = {}
    warnings = []

    # D1 成立年限与数据质量（年限分层：<2年硬剔除，2~3年预警放行）
    reasons = []
    if m['years_history'] < cfg['min_history_years_hard']:
        reasons.append(f"成立不足2年（{m['years_history']:.1f}年），历史数据不足以定位底部")
    elif m['years_history'] < cfg['min_history_years']:
        warnings.append(f"⚠成立{m['years_history']:.1f}年(<3年)，3年分位基于有限历史，建议对照老份额/指数")
    if m['jump_count'] > cfg['jump_count_max']:
        reasons.append(f"净值异常跳变{m['jump_count']}次（单日|涨跌|>{cfg['jump_daily_threshold']:.0%}，疑似数据错印）")
    dims['D1'] = {'name': '成立年限与数据质量', 'pass': len(reasons) == 0, 'reasons': reasons}

    # D2 清盘/僵尸风险（规模数据开源接口拿不到，用“持续缩水/阴跌僵尸”代理）
    reasons = []
    if m['trend_2y'] <= cfg['zombie_trend_ann'] and m['newlow_252'] > cfg['zombie_newlow_density']:
        reasons.append(f"近2年年化{m['trend_2y']:.0%}且新低密度{m['newlow_252']:.0%}，疑似持续缩水")
    if (not np.isnan(m['ann_vol']) and m['ann_vol'] < cfg['zombie_vol_max']
            and m['trend_2y'] < cfg['zombie_trend_soft']):
        reasons.append("低波动却长期负收益，疑似阴跌僵尸基金")
    dims['D2'] = {'name': '清盘/僵尸风险', 'pass': len(reasons) == 0, 'reasons': reasons}

    # D3 价值陷阱（不断创新低，没有底）
    reasons = []
    if m['newlow_252'] > cfg['newlow_density_trap']:
        reasons.append(f"近1年{m['newlow_252']:.0%}交易日创历史新低，跌势未止")
    dims['D3'] = {'name': '价值陷阱(无底新低)', 'pass': len(reasons) == 0, 'reasons': reasons}

    # D4 波动/跟踪崩坏
    reasons = []
    if not np.isnan(m['ann_vol']) and m['ann_vol'] > cfg['vol_extreme']:
        reasons.append(f"年化波动{m['ann_vol']:.0%}极端偏高")
    if not np.isnan(m['vol_surge']) and m['vol_surge'] > cfg['vol_surge_ratio']:
        reasons.append(f"近60日波动突增{m['vol_surge']:.1f}倍，风险异常放大")
    if m['max_dd_3y'] < -cfg['max_dd_blowup']:
        reasons.append(f"3年最大回撤{abs(m['max_dd_3y']):.0%}，疑似跟踪崩坏/踩雷")
    dims['D4'] = {'name': '波动/跟踪崩坏', 'pass': len(reasons) == 0, 'reasons': reasons}

    fail_dims = [k for k, v in dims.items() if not v['pass']]
    risk_score = 25 * len(fail_dims)
    # 未到硬门槛但需扣分的预警项
    if m['years_history'] < cfg['min_history_years'] and m['years_history'] >= cfg['min_history_years_hard']:
        risk_score += 10
    if m['max_dd_3y'] < -cfg['max_dd_warn']:
        risk_score += 10
    if m['newlow_252'] > cfg['newlow_density_trap'] * 0.75:
        risk_score += 10
    risk_score = min(int(risk_score), 100)

    return {
        'pass': len(fail_dims) == 0,
        'dims': dims,
        'fail_dims': fail_dims,
        'warnings': warnings,
        'risk_score': risk_score,
        'all_reasons': [r for v in dims.values() for r in v['reasons']],
    }


def dual_bottom_confirm(m: dict, cfg: dict = SCREEN_CFG) -> dict:
    """
    双底确认：价格底 AND 估值底（3项代理满足≥2）。
    返回确认状态、各项明细与底部综合评分（0~100，越高越值得抄）。
    """
    # 价格底
    price_strong = m['pct_3y'] <= cfg['price_pct_strong']
    price_watch = m['pct_3y'] <= cfg['price_pct_watch']

    # 估值底（3项代理）
    v_long = m['pct_long'] <= cfg['long_pct_max']
    v_dd = m['dd_from_peak'] <= -cfg['dd_from_peak_min']
    v_ma = m['dev_ma250'] <= cfg['ma250_dev_max']
    val_items = [
        ('长周期分位≤30%', v_long, f"长周期分位{m['pct_long']:.0%}"),
        ('高点回撤≥25%', v_dd, f"距高点回撤{abs(m['dd_from_peak']):.0%}"),
        ('低于年线≥10%', v_ma, f"偏离年线{m['dev_ma250']:.0%}"),
    ]
    val_hits = sum(1 for _, ok, _ in val_items if ok)

    confirmed = price_strong and val_hits >= 2
    watch = (not confirmed) and price_watch and val_hits >= 1
    status = 'confirmed' if confirmed else ('watch' if watch else 'none')

    # --- 底部综合评分 ---
    # 价格深度 40 分
    score_price = 40.0 * np.clip(1.0 - m['pct_3y'] / cfg['price_pct_watch'], 0, 1)
    # 估值确认 30 分（每项10）
    score_val = 10.0 * val_hits
    # 企稳信号 15 分
    if m['stabilized']:
        score_stab = 15.0
    elif m['no_new_low_20'] or m['ret_20'] > cfg['stabilize_ret_min']:
        score_stab = 7.0
    else:
        score_stab = 0.0
    # 新低密度反向 15 分（新低越少越好）
    score_trend = 15.0 * (1.0 - np.clip(m['newlow_252'] / cfg['newlow_density_trap'], 0, 1))

    bottom_score = float(score_price + score_val + score_stab + score_trend)

    return {
        'status': status,
        'price_strong': price_strong,
        'price_watch': price_watch,
        'val_items': val_items,
        'val_hits': val_hits,
        'bottom_score': round(bottom_score, 1),
    }


def rating_of(risk_pass: bool, bottom: dict) -> str:
    """综合评级"""
    if not risk_pass:
        return '✗ 排雷剔除'
    if bottom['status'] == 'confirmed' and bottom['bottom_score'] >= 75:
        return '★★★ 重点抄底'
    if bottom['status'] == 'confirmed':
        return '★★ 可挂条件单'
    if bottom['status'] == 'watch':
        return '★ 观察名单'
    return '— 未到底部'


def screen_one(code: str, name: str, nav: pd.Series, data_source: str = 'real') -> dict:
    """对单只基金完成 指标→排雷→双底→评级 全流程"""
    m = compute_metrics(nav)
    if not m.get('data_ok'):
        return {
            'code': code, 'name': name, 'data_source': data_source,
            'rating': '— 数据不足', 'risk_pass': False,
            'bottom_status': 'none', 'bottom_score': 0, 'risk_score': 100,
            'risk_reasons': ['历史数据不足60条'],
        }
    risk = four_dimension_risk(m)
    bottom = dual_bottom_confirm(m)
    rating = rating_of(risk['pass'], bottom) if data_source == 'real' else '— 模拟数据'

    return {
        'code': code,
        'name': name,
        'data_source': data_source,
        'rating': rating,
        'risk_pass': risk['pass'],
        'risk_score': risk['risk_score'],
        'risk_fail_dims': '、'.join(risk['fail_dims']) if risk['fail_dims'] else ('预警' if risk['warnings'] else '-'),
        'risk_reasons': risk['all_reasons'] + risk['warnings'],
        'bottom_status': bottom['status'],
        'bottom_score': bottom['bottom_score'],
        'pct_3y': m['pct_3y'],
        'pct_long': m['pct_long'],
        'dd_from_peak': m['dd_from_peak'],
        'max_dd_3y': m['max_dd_3y'],
        'dev_ma250': m['dev_ma250'],
        'ann_vol': m['ann_vol'],
        'vol_surge': m['vol_surge'],
        'newlow_252': m['newlow_252'],
        'trend_2y': m['trend_2y'],
        'stabilized': m['stabilized'],
        'jump_count': m['jump_count'],
        'years_history': m['years_history'],
        'nav_now': m['nav_now'],
        'val_hits': bottom['val_hits'],
        'val_detail': '；'.join(f"{label}:{'✓' if ok else '✗'}({txt})"
                               for label, ok, txt in bottom['val_items']),
        'metrics': m,
    }


def screen_pool(funds: dict) -> pd.DataFrame:
    """
    批量筛选。
    :param funds: {code: {'name':..., 'nav': DataFrame/Series, 'source': 'real'/'sim'}}
    :return: 筛选结果 DataFrame（按底部评分降序）
    """
    rows = []
    for code, info in funds.items():
        nav = info['nav']
        if hasattr(nav, 'columns') and 'close' in nav.columns:
            nav = nav['close']
        res = screen_one(code, info['name'], nav, info.get('source', 'real'))
        rows.append(res)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['risk_pass', 'bottom_score'], ascending=[False, False]).reset_index(drop=True)
    return df
