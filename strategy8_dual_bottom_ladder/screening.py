# -*- coding: utf-8 -*-
"""双底确认 + 四维排雷"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from strategy8_dual_bottom_ladder import config as C
from strategy8_dual_bottom_ladder.valuation import evaluate_valuation


def _pct_rank(window: pd.Series, value: float) -> float:
    s = window.dropna()
    if not len(s):
        return float('nan')
    return float((s < value).sum() / len(s))


def compute_metrics(nav: pd.Series, as_of=None) -> Dict:
    s = nav.dropna().astype(float)
    if as_of is not None:
        s = s[s.index <= pd.Timestamp(as_of)]
    n = len(s)
    if n < 60:
        return {'ok': False, 'n': n}

    now = float(s.iloc[-1])
    rets = s.pct_change().dropna()

    def w(td):
        return s.iloc[-td:] if n >= td else s

    w1, w3 = w(252), w(756)
    wl = w(1260) if n >= 1260 else s
    pct3 = _pct_rank(w3, now)
    pctl = _pct_rank(wl, now)
    peak = float(w3.max())
    dd_peak = now / peak - 1.0
    mdd3 = float((w3 / w3.cummax() - 1).min())
    mdd1 = float((w1 / w1.cummax() - 1).min())
    ma = float(s.iloc[-252:].mean()) if n >= 60 else now
    dev = now / ma - 1.0 if ma else 0.0
    vol = float(rets.iloc[-252:].std() * np.sqrt(252)) if len(rets) >= 60 else float('nan')
    vol60 = float(rets.iloc[-60:].std() * np.sqrt(252)) if len(rets) >= 60 else vol
    surge = float(vol60 / vol) if vol and vol > 1e-8 else float('nan')
    is_nl = s <= s.cummin()
    nl252 = float(is_nl.iloc[-252:].mean()) if n >= 252 else float(is_nl.mean())
    if n >= 505:
        trend2 = (now / float(s.iloc[-505])) ** (252 / 504) - 1
    else:
        trend2 = (now / float(s.iloc[0])) ** (252 / max(n - 1, 1)) - 1
    jumps = int((rets.abs() > C.JUMP_TH).sum())
    last20 = s.iloc[-20:]
    no_nl = bool(last20.min() > float(w1.min()) * 1.001)
    ret20 = (now / float(s.iloc[-21]) - 1) if n > 20 else 0.0
    stable = no_nl and ret20 > -0.03

    return {
        'ok': True, 'n': n, 'years': n / 252.0, 'nav': now,
        'pct3': pct3, 'pctl': pctl, 'dd_peak': dd_peak,
        'mdd3': mdd3, 'mdd1': mdd1, 'dev_ma': dev,
        'vol': vol, 'surge': surge, 'nl252': nl252,
        'trend2': trend2, 'jumps': jumps,
        'stable': stable, 'ret20': ret20, 'no_nl20': no_nl,
    }


def risk_filter(m: Dict, meta: Optional[Dict] = None) -> Dict:
    meta = meta or {}
    dims, warns = {}, []

    r = []
    if m['years'] < C.MIN_YEARS_HARD:
        r.append(f"成立不足2年({m['years']:.1f})")
    elif m['years'] < C.MIN_YEARS:
        warns.append(f"⚠成立{m['years']:.1f}年(<3)")
    if m['jumps'] > C.JUMP_MAX:
        r.append(f"异常跳变{m['jumps']}次")
    dims['D1'] = {'name': '成立/数据', 'pass': not r, 'reasons': r}

    r = []
    scale = meta.get('scale_yi')
    is_idx = bool(meta.get('is_index'))
    if scale is not None:
        mx = C.SCALE_MAX_IDX if is_idx else C.SCALE_MAX
        if scale < C.SCALE_MIN:
            r.append(f"规模过小{scale:.2f}亿")
        elif scale > mx:
            r.append(f"规模过大{scale:.2f}亿")
    else:
        warns.append('⚠规模缺失未硬杀')
    my = meta.get('manager_years')
    if my is not None and my < C.MGR_MIN_YEARS:
        r.append(f"经理任职{my:.1f}年")
    elif not meta.get('meta_ok'):
        warns.append('⚠概况失败，规模/经理未硬杀')
    if m['trend2'] <= C.ZOMBIE_ANN and m['nl252'] > C.ZOMBIE_NL:
        r.append('疑似持续缩水僵尸')
    if not np.isnan(m['vol']) and m['vol'] < C.ZOMBIE_VOL and m['trend2'] < C.ZOMBIE_SOFT:
        r.append('低波阴跌僵尸')
    dims['D2'] = {'name': '规模经理/僵尸', 'pass': not r, 'reasons': r}

    r = []
    if m['nl252'] > C.NEWLOW_TRAP:
        r.append(f"新低密度{m['nl252']:.0%}")
    dims['D3'] = {'name': '价值陷阱', 'pass': not r, 'reasons': r}

    r = []
    if not np.isnan(m['vol']) and m['vol'] > C.VOL_EXTREME:
        r.append(f"波动极端{m['vol']:.0%}")
    if not np.isnan(m['surge']) and m['surge'] > C.VOL_SURGE:
        r.append(f"波动突增{m['surge']:.1f}x")
    if m['mdd3'] < -C.MDD_BLOWUP:
        r.append(f"回撤过大{abs(m['mdd3']):.0%}")
    dims['D4'] = {'name': '波动崩坏', 'pass': not r, 'reasons': r}

    fail = [k for k, v in dims.items() if not v['pass']]
    reasons = [x for v in dims.values() for x in v['reasons']] + warns
    return {
        'pass': not fail, 'fail': fail, 'dims': dims, 'warns': warns,
        'reasons': reasons, 'score': min(100, 25 * len(fail) + 10 * len(warns)),
    }


def dual_bottom(m: Dict, val: Dict) -> Dict:
    price_ok = (
        m['pct3'] <= C.PRICE_PCT_3Y
        and m['mdd3'] <= C.PRICE_MDD_3Y
        and m['mdd1'] <= C.PRICE_MDD_1Y
    )
    price_watch = m['pct3'] <= C.PRICE_PCT_WATCH
    val_ok = bool(val.get('pass'))
    confirmed = price_ok and val_ok
    watch = (not confirmed) and price_watch and val.get('hits', 0) >= 1
    status = 'confirmed' if confirmed else ('watch' if watch else 'none')

    sp = 40 * np.clip(1 - m['pct3'] / C.PRICE_PCT_WATCH, 0, 1)
    sv = 15 * min(val.get('hits', 0), 2) + (10 if val_ok else 0)
    ss = 15 if m['stable'] else (7 if m['no_nl20'] else 0)
    st = 15 * (1 - np.clip(m['nl252'] / C.NEWLOW_TRAP, 0, 1))
    score = float(sp + sv + ss + st)
    return {
        'status': status, 'price_ok': price_ok, 'price_watch': price_watch,
        'val_ok': val_ok, 'score': round(score, 1),
        'val_source': val.get('source', ''), 'val_hits': val.get('hits', 0),
        'val_items': val.get('items', []),
    }


def rating(risk_ok: bool, bottom: Dict, source: str = 'real') -> str:
    if source != 'real':
        return '— 模拟数据'
    if not risk_ok:
        return '✗ 排雷剔除'
    if bottom['status'] == 'confirmed' and bottom['score'] >= 75:
        return '★★★ 重点抄底'
    if bottom['status'] == 'confirmed':
        return '★★ 可挂条件单'
    if bottom['status'] == 'watch':
        return '★ 观察名单'
    return '— 未到底部'


def screen_one(code: str, name: str, nav: pd.Series,
               meta: Optional[Dict] = None, source: str = 'real') -> Dict:
    meta = meta or {}
    m = compute_metrics(nav)
    if not m.get('ok'):
        return {
            'code': code, 'name': name, 'source': source, 'rating': '— 数据不足',
            'risk_ok': False, 'bottom_status': 'none', 'bottom_score': 0,
            'risk_score': 100, 'risk_reasons': '数据不足', 'sector': meta.get('sector', '其他'),
            'ann_vol': np.nan, 'nav_now': np.nan,
        }
    is_idx = bool(meta.get('is_index')) or any(k in name for k in ('指数', '联接', 'ETF'))
    val = evaluate_valuation(nav, name=name, is_index=is_idx)
    risk = risk_filter(m, meta)
    bottom = dual_bottom(m, val)
    return {
        'code': code, 'name': name, 'source': source,
        'rating': rating(risk['pass'], bottom, source),
        'risk_ok': risk['pass'], 'risk_score': risk['score'],
        'risk_fail': '、'.join(risk['fail']) if risk['fail'] else ('预警' if risk['warns'] else '-'),
        'risk_reasons': '；'.join(risk['reasons']) if risk['reasons'] else '-',
        'bottom_status': bottom['status'], 'bottom_score': bottom['score'],
        'pct_3y': m['pct3'], 'pct_long': m['pctl'], 'dd_from_peak': m['dd_peak'],
        'max_dd_3y': m['mdd3'], 'max_dd_1y': m['mdd1'], 'dev_ma250': m['dev_ma'],
        'ann_vol': m['vol'], 'vol_surge': m['surge'], 'newlow_252': m['nl252'],
        'trend_2y': m['trend2'], 'stabilized': m['stable'], 'jump_count': m['jumps'],
        'years': m['years'], 'nav_now': m['nav'],
        'val_hits': bottom['val_hits'], 'val_source': bottom['val_source'],
        'val_detail': '；'.join(f"{a}:{'✓' if b else '✗'}({c})" for a, b, c in bottom['val_items']),
        'sector': meta.get('sector', '其他'), 'scale_yi': meta.get('scale_yi'),
        'is_index': is_idx,
    }


def screen_pool(candidates: pd.DataFrame, nav_map: Dict[str, pd.Series],
                meta_map: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    meta_map = meta_map or {}
    rows = []
    for _, r in candidates.iterrows():
        code = str(r['基金代码']).zfill(6)
        name = str(r.get('基金简称', code))
        nav = nav_map.get(code)
        if nav is None:
            rows.append({
                'code': code, 'name': name, 'rating': '— 无净值',
                'risk_ok': False, 'bottom_status': 'none', 'bottom_score': 0,
                'risk_reasons': '净值缺失', 'sector': '其他',
            })
            continue
        if hasattr(nav, 'columns') and 'close' in getattr(nav, 'columns', []):
            series = nav['close']
            src = getattr(nav, 'attrs', {}).get('nav_source', 'real')
        else:
            series = nav
            src = 'real'
        res = screen_one(code, name, series, meta_map.get(code, {}), src)
        for col in ('近1年', '近3年', '基金类型', '榜单排名', '单位净值'):
            if col in r.index:
                res[col] = r[col]
        rows.append(res)
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values(['risk_ok', 'bottom_score'], ascending=[False, False]).reset_index(drop=True)
    return df
