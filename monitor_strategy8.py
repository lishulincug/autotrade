# -*- coding: utf-8 -*-
"""
策略8 · 每日条件单监控
====================
用法:
  python monitor_strategy8.py
  python monitor_strategy8.py --orders strategy8_dual_bottom_ladder/conditional_orders.csv
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import datetime

import pandas as pd

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.data_loader import load_otc_fund_nav
from strategy8_dual_bottom_ladder import config as C


def latest_nav(code: str, name: str) -> float:
    df = load_otc_fund_nav(code, name, C.NAV_START, C.NAV_END, verbose=False)
    if df is None or not len(df):
        return float('nan')
    if 'close' in df.columns:
        return float(df['close'].iloc[-1])
    return float(df.iloc[-1, 0])


def check_orders(orders: pd.DataFrame) -> pd.DataFrame:
    alerts = []
    if not len(orders):
        return pd.DataFrame()

    codes = orders[['基金代码', '基金简称']].drop_duplicates()
    navs = {}
    for _, r in codes.iterrows():
        code = str(r['基金代码']).zfill(6)
        name = str(r['基金简称'])
        try:
            navs[code] = latest_nav(code, name)
            print(f'  {code} {name[:16]} 最新净值={navs[code]:.4f}')
        except Exception as e:
            print(f'  {code} 净值失败: {e}')
            navs[code] = float('nan')

    for _, row in orders.iterrows():
        code = str(row['基金代码']).zfill(6)
        nav = navs.get(code, float('nan'))
        if nav != nav:
            continue
        typ = str(row.get('条件单类型', ''))
        try:
            trigger = float(row.get('触发净值'))
        except (TypeError, ValueError):
            continue

        # 无持仓状态时只监控「向下触发」类（买入/终止/硬止损），
        # 避免未建仓时止盈价低于现价造成误报。
        hit, level = False, 'info'
        if '买入' in typ and nav <= trigger:
            hit, level = True, 'buy'
        elif ('终止' in typ or '⛔' in typ) and nav <= trigger:
            hit, level = True, 'terminate'
        elif ('止损' in typ or '🛑' in typ) and nav <= trigger:
            hit, level = True, 'stop'

        if hit:
            alerts.append({
                '时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                '基金代码': code,
                '基金简称': row.get('基金简称'),
                '条件单类型': typ,
                '触发净值': trigger,
                '最新净值': round(nav, 4),
                '偏离': f'{(nav / trigger - 1):+.2%}',
                '级别': level,
                '说明': row.get('说明', ''),
            })
    return pd.DataFrame(alerts)


def main():
    ap = argparse.ArgumentParser(description='策略8 条件单监控')
    ap.add_argument('--orders', default=C.OUT_ORDERS)
    ap.add_argument('--out', default=C.OUT_MON)
    args = ap.parse_args()

    print('=' * 50)
    print('策略8 每日条件单监控')
    print('=' * 50)

    if not os.path.exists(args.orders):
        print(f'未找到条件单: {args.orders}')
        print('请先运行: python run_strategy8_dual_bottom.py')
        return

    orders = pd.read_csv(args.orders, dtype={'基金代码': str})
    orders['基金代码'] = orders['基金代码'].astype(str).str.zfill(6)
    print(f'条件单 {len(orders)} 行，基金 {orders["基金代码"].nunique()} 只\n')

    alerts = check_orders(orders)
    if not len(alerts):
        print('\n今日无触发信号。')
        alerts = pd.DataFrame(columns=['时间', '基金代码', '基金简称', '条件单类型',
                                       '触发净值', '最新净值', '偏离', '级别', '说明'])
    else:
        print(f'\n触发 {len(alerts)} 条:')
        for _, a in alerts.iterrows():
            print(f"  [{a['级别']}] {a['基金代码']} {a['基金简称']} | {a['条件单类型']} | "
                  f"净值{a['最新净值']} vs {a['触发净值']}")

    alerts.to_csv(args.out, index=False, encoding='utf-8-sig')
    print(f'\n已写 {args.out}')


if __name__ == '__main__':
    main()
