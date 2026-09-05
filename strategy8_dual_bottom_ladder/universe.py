# -*- coding: utf-8 -*-
"""东财开放式基金排行宇宙（天天基金同源）"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import pandas as pd

from strategy8_dual_bottom_ladder import config as C

try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False


def _cpath(symbol: str) -> str:
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    safe = re.sub(r'[^\w\u4e00-\u9fff]+', '_', symbol)
    return os.path.join(C.CACHE_DIR, f'rank_{safe}.pkl')


def _load_cache(symbol: str) -> Optional[pd.DataFrame]:
    p = _cpath(symbol)
    if not os.path.exists(p):
        return None
    if datetime.now() - datetime.fromtimestamp(os.path.getmtime(p)) > timedelta(hours=C.RANK_CACHE_HOURS):
        return None
    try:
        return pd.read_pickle(p)
    except Exception:
        return None


def _save_cache(symbol: str, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(_cpath(symbol))
    except Exception:
        pass


def fetch_rank(symbol: str, use_cache: bool = True) -> pd.DataFrame:
    if use_cache:
        hit = _load_cache(symbol)
        if hit is not None and len(hit):
            return hit
    if not HAS_AK:
        return pd.DataFrame()

    fns = []
    if hasattr(ak, 'fund_open_fund_rank_em'):
        fns.append(lambda: ak.fund_open_fund_rank_em(symbol=symbol))
    if hasattr(ak, 'fund_open_fund_rank_em'):
        fns.append(lambda: ak.fund_open_fund_rank_em(symbol=symbol))

    df = pd.DataFrame()
    for fn in fns:
        try:
            df = fn()
            if df is not None and len(df):
                break
        except Exception as e:
            print(f'  [universe] {symbol} 失败: {e}')
            df = pd.DataFrame()
        time.sleep(0.35)

    if df is None or not len(df):
        return pd.DataFrame()
    df = df.copy()
    df['基金类型'] = symbol
    _save_cache(symbol, df)
    return df


def _norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    m = {}
    for c in df.columns:
        s = str(c).strip()
        if s in ('基金代码', '代码'):
            m[c] = '基金代码'
        elif s in ('基金简称', '简称', '基金名称'):
            m[c] = '基金简称'
        elif '近1年' in s or s == '近一年':
            m[c] = '近1年'
        elif '近3年' in s:
            m[c] = '近3年'
        elif '近1周' in s:
            m[c] = '近1周'
        elif '近1月' in s:
            m[c] = '近1月'
        elif '日增长率' in s:
            m[c] = '日增长率'
        elif s == '单位净值':
            m[c] = '单位净值'
    out = df.rename(columns=m)
    if '基金代码' in out.columns:
        out['基金代码'] = out['基金代码'].astype(str).str.zfill(6)
    return out


def _base_name(name: str) -> str:
    n = str(name).strip()
    n = re.sub(r'[ABCDEFahcdef]\s*$', '', n)
    n = re.sub(r'[（(][ABCDEFahcdef][)）]\s*$', '', n)
    return n.strip()


def _class_score(name: str) -> int:
    n = str(name)
    pref = C.PREFER_CLASS.upper()
    if re.search(rf'[（(]?{pref}[)）]?\s*$', n, re.I):
        return 3
    if re.search(r'[（(]?C[)）]?\s*$', n, re.I):
        return 2
    if re.search(r'[（(]?A[)）]?\s*$', n, re.I):
        return 1
    return 0


def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not len(df):
        return df
    x = df.copy()
    x['_b'] = x['基金简称'].map(_base_name)
    x['_s'] = x['基金简称'].map(_class_score)
    x = x.sort_values(['_b', '_s'], ascending=[True, False])
    x = x.drop_duplicates('_b', keep='first')
    return x.drop(columns=['_b', '_s']).reset_index(drop=True)


def exclude_names(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or not len(df):
        return df
    mask = pd.Series(True, index=df.index)
    for kw in C.EXCLUDE_KEYWORDS:
        mask &= ~df['基金简称'].astype(str).str.contains(kw, na=False)
    return df.loc[mask].reset_index(drop=True)


def load_pool_file(path: Optional[str] = None) -> List[dict]:
    path = path or C.POOL_FILE
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '\t' in line:
                name, code = line.split('\t', 1)
            else:
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                name, code = parts
            rows.append({'基金代码': code.strip().zfill(6), '基金简称': name.strip(), '基金类型': '白名单'})
    return rows


def load_market_universe(
    sort_col: Optional[str] = None,
    ascending: Optional[bool] = None,
    top_n: Optional[int] = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sort_col = sort_col or C.SORT_COL
    ascending = C.SORT_ASCENDING if ascending is None else ascending
    top_n = C.DEEP_TOP_N if top_n is None else top_n

    frames = []
    for t in C.RANK_TYPES:
        if verbose:
            print(f'  [universe] 拉取: {t}')
        part = fetch_rank(t, use_cache=use_cache)
        if not len(part):
            continue
        frames.append(_norm_cols(part))
        time.sleep(0.2)

    if not frames:
        if verbose:
            print('  [universe] 排行失败，回退 fund_pool.txt')
        board = pd.DataFrame(load_pool_file())
        if len(board):
            board['榜单排名'] = range(1, len(board) + 1)
        return board, board.head(top_n).copy()

    board = pd.concat(frames, ignore_index=True)
    board = exclude_names(board)
    board = dedupe(board)
    if sort_col in board.columns:
        board[sort_col] = pd.to_numeric(board[sort_col], errors='coerce')
        board = board.sort_values(sort_col, ascending=ascending, na_position='last')
    board = board.reset_index(drop=True)
    board['榜单排名'] = range(1, len(board) + 1)
    deep = board.head(int(top_n)).copy()
    if verbose:
        print(f'  [universe] 榜单{len(board)}只 → 深度候选{len(deep)}只')
    return board, deep


def load_pool_universe() -> Tuple[pd.DataFrame, pd.DataFrame]:
    board = pd.DataFrame(load_pool_file())
    if len(board):
        board['榜单排名'] = range(1, len(board) + 1)
    return board, board.copy()
