# -*- coding: utf-8 -*-
"""估值底：指数优先 PE 分位，否则 NAV 三代理"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd

from strategy8_dual_bottom_ladder import config as C

try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False


def _pct_rank(window: pd.Series, value: float) -> float:
    s = window.dropna()
    if not len(s):
        return float('nan')
    return float((s < value).sum() / len(s))


def nav_proxies(nav: pd.Series) -> Dict:
    s = nav.dropna().astype(float)
    n = len(s)
    now = float(s.iloc[-1])
    w_long = s.iloc[-1260:] if n >= 1260 else s
    pct_long = _pct_rank(w_long, now)
    peak = float(s.iloc[-756:].max()) if n >= 60 else float(s.max())
    dd = now / peak - 1.0
    ma = float(s.iloc[-252:].mean()) if n >= 60 else float(s.mean())
    dev = now / ma - 1.0 if ma > 0 else 0.0
    items = [
        ('长周期分位≤30%', pct_long <= C.VAL_LONG_PCT, f'{pct_long:.1%}'),
        ('距高点回撤≥25%', dd <= -C.VAL_DD_PEAK, f'{dd:.1%}'),
        ('低于年线≥10%', dev <= C.VAL_MA_DEV, f'{dev:.1%}'),
    ]
    hits = sum(1 for _, ok, _ in items if ok)
    return {
        'pass': hits >= C.VAL_MIN_HITS,
        'hits': hits,
        'items': items,
        'source': 'NAV代理',
        'pct_long': pct_long,
        'dd_from_peak': dd,
        'dev_ma250': dev,
    }


def _pe_pct(name: str) -> Tuple[Optional[float], str]:
    if not HAS_AK:
        return None, '无akshare'
    hint = name or ''
    if any(k in hint for k in ('纳指', '纳斯达克', '标普', '道琼斯')):
        return None, '海外跳过PE'
    try:
        if hasattr(ak, 'stock_index_pe_lg'):
            df = ak.stock_index_pe_lg()
            if df is not None and len(df) >= 30:
                pe_col = None
                for c in df.columns:
                    if '滚动市盈率' in str(c) or '市盈' in str(c):
                        pe_col = c
                        break
                if pe_col is not None:
                    series = pd.to_numeric(df[pe_col], errors='coerce').dropna()
                    if len(series) >= 30:
                        cur = float(series.iloc[-1])
                        pct = float((series < cur).sum() / len(series))
                        return pct, f'PE={cur:.2f}@{pct:.1%}'
    except Exception as e:
        return None, f'PE失败:{e}'
    return None, 'PE不可用'


def evaluate_valuation(nav: pd.Series, name: str = '', is_index: bool = False) -> Dict:
    proxy = nav_proxies(nav)
    if is_index:
        pe, detail = _pe_pct(name)
        if pe is not None:
            ok = pe <= C.VAL_PE_MAX
            return {
                'pass': ok, 'hits': 1 if ok else 0, 'source': 'PE分位',
                'items': [('PE历史分位≤30%', ok, detail)],
                'pe_pct': pe, 'detail': detail, 'proxy': proxy,
            }
        out = dict(proxy)
        out['source'] = 'NAV代理(PE回退)'
        out['detail'] = detail
        return out
    out = dict(proxy)
    out['detail'] = '主动/非指数'
    return out
