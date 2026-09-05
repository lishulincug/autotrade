# -*- coding: utf-8 -*-
"""基金概况：规模/经理/行业（akshare，失败降级）"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Optional

from strategy8_dual_bottom_ladder import config as C

try:
    import akshare as ak
    HAS_AK = True
except ImportError:
    HAS_AK = False

SECTOR_KW = [
    ('红利', '红利价值'), ('高股息', '红利价值'), ('医药', '医药'), ('生物', '医药'),
    ('半导体', '科技成长'), ('芯片', '科技成长'), ('科技', '科技成长'), ('互联网', '科技成长'),
    ('新能源', '新能源'), ('光伏', '新能源'), ('军工', '军工'), ('消费', '消费'),
    ('白酒', '消费'), ('金融', '金融地产'), ('银行', '金融地产'), ('纳指', '海外'),
    ('标普', '海外'), ('恒生', '港股'), ('港股', '港股'), ('北证', '北交所'),
    ('创业板', '成长宽基'), ('科创', '成长宽基'), ('沪深300', '宽基'),
    ('中证500', '宽基'), ('中证1000', '宽基'),
]


def infer_sector(name: str, ftype: str = '') -> str:
    text = f'{name} {ftype}'
    for kw, sec in SECTOR_KW:
        if kw in text:
            return sec
    if '指数' in text:
        return '宽基'
    if '混合' in text:
        return '主动混合'
    if '股票' in text:
        return '主动股票'
    return '其他'


def is_index_fund(name: str, ftype: str = '') -> bool:
    t = f'{name}{ftype}'
    return any(k in t for k in ('指数', '联接', 'ETF', '增强'))


def _cache_path(code: str) -> str:
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    return os.path.join(C.CACHE_DIR, f'meta_{code}.json')


def _load(code: str) -> Optional[dict]:
    p = _cache_path(code)
    if not os.path.exists(p):
        return None
    if datetime.now() - datetime.fromtimestamp(os.path.getmtime(p)) > timedelta(hours=C.RANK_CACHE_HOURS):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return None


def _save(code: str, data: dict) -> None:
    try:
        with open(_cache_path(code), 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass


def _parse_yi(val) -> Optional[float]:
    if val is None:
        return None
    s = str(val).replace(',', '')
    m = re.search(r'([\d.]+)', s)
    if not m:
        return None
    num = float(m.group(1))
    return num / 10000.0 if '万' in s else num


def fetch_meta(code: str, name: str = '', ftype: str = '', use_cache: bool = True) -> Dict:
    code = str(code).zfill(6)
    if use_cache:
        c = _load(code)
        if c:
            return c

    out = {
        'code': code, 'name': name, 'fund_type': ftype, 'meta_ok': False,
        'scale_yi': None, 'establish_date': None, 'manager': None,
        'manager_years': None, 'sector': infer_sector(name, ftype),
        'is_index': is_index_fund(name, ftype), 'warnings': [],
    }
    if not HAS_AK:
        out['warnings'].append('akshare未安装')
        _save(code, out)
        return out

    try:
        if hasattr(ak, 'fund_individual_basic_info_xq'):
            basic = ak.fund_individual_basic_info_xq(symbol=code)
            if basic is not None and len(basic):
                kv = {}
                if 'item' in basic.columns and 'value' in basic.columns:
                    kv = dict(zip(basic['item'].astype(str), basic['value']))
                elif basic.shape[1] >= 2:
                    kv = dict(zip(basic.iloc[:, 0].astype(str), basic.iloc[:, 1]))
                out['name'] = out['name'] or str(kv.get('基金名称', name))
                out['fund_type'] = str(kv.get('基金类型', ftype) or ftype)
                out['establish_date'] = str(kv.get('成立日期', '') or '') or None
                out['manager'] = str(kv.get('基金经理', '') or '') or None
                if kv.get('上任日期'):
                    try:
                        import pandas as pd
                        start = pd.to_datetime(kv['上任日期']).to_pydatetime()
                        out['manager_years'] = (datetime.now() - start).days / 365.25
                    except Exception:
                        pass
                out['meta_ok'] = True
    except Exception as e:
        out['warnings'].append(f'基本信息:{e}')

    try:
        if hasattr(ak, 'fund_open_fund_info_em'):
            sdf = ak.fund_open_fund_info_em(symbol=code, indicator='基金规模')
            if sdf is not None and len(sdf):
                for col in sdf.columns:
                    if any(k in str(col) for k in ('净资产', '规模', 'value')):
                        yi = _parse_yi(sdf.iloc[-1][col])
                        if yi is not None:
                            out['scale_yi'] = yi
                            out['meta_ok'] = True
                            break
                if out['scale_yi'] is None:
                    yi = _parse_yi(sdf.iloc[-1, -1])
                    if yi is not None:
                        out['scale_yi'] = yi
                        out['meta_ok'] = True
    except Exception as e:
        out['warnings'].append(f'规模:{e}')

    out['sector'] = infer_sector(out.get('name') or name, out.get('fund_type') or ftype)
    out['is_index'] = is_index_fund(out.get('name') or name, out.get('fund_type') or ftype)
    _save(code, out)
    return out
