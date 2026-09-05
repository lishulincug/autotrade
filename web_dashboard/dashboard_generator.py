"""
Web看板数据生成器：运行所有策略 -> 整理成ECharts可消费的JSON -> 生成HTML+静态资源
"""
import os
import sys
import json
import time
import traceback
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _strategy_defs():
    return [
        {
            'id': 's1',
            'short_name': 'RRG行业轮动',
            'full_name': '策略1 · RRG多因子行业轮动',
            'desc': '月度调仓 · 4因子打分 · RRG领先象限 · 止损8% · 适合结构化行情',
            'module': 'strategy1_rrg_rotation.strategy',
            'func': 'run_strategy',
            'color': '#FF6B6B',
            'primary_symbol': '512880',  # 展示K线用
        },
        {
            'id': 's2',
            'short_name': '双均线动量',
            'full_name': '策略2 · 双均线动量轮动+空仓风控',
            'desc': '日频判断 · MA28过滤 · 20日涨幅排名 · 空仓防守 · 适合趋势行情',
            'module': 'strategy2_dual_ma_momentum.strategy',
            'func': 'run_strategy',
            'color': '#4ECDC4',
            'primary_symbol': '510300',
        },
        {
            'id': 's3',
            'short_name': '网格交易',
            'full_name': '策略3 · 网格交易',
            'desc': '±20%区间 · 单格2.5% · 12份分仓 · 自动重建 · 适合震荡市',
            'module': 'strategy3_grid_trading.strategy',
            'func': 'run_strategy',
            'color': '#FFD93D',
            'primary_symbol': '510300',
        },
        {
            'id': 's4',
            'short_name': '宏观因子择时',
            'full_name': '策略4 · 宏观因子择时',
            'desc': '月度调仓 · 6大宏观因子 · 5档仓位 · 股债切换 · 穿越牛熊',
            'module': 'strategy4_macro_timing.strategy',
            'func': 'run_strategy',
            'color': '#6C5CE7',
            'primary_symbol': '510300',
        },
        {
            'id': 's5',
            'short_name': '指数增强选股',
            'full_name': '策略5 · 指数增强多因子选股',
            'desc': '季度调仓 · 价值/质量/成长/红利 · 估值仓位 · 赚取Alpha',
            'module': 'strategy5_index_enhancement.strategy',
            'func': 'run_strategy',
            'color': '#F8B500',
            'primary_symbol': '600519',
        },
        {
            'id': 's6',
            'short_name': '广发净值转换',
            'full_name': '策略6 · 广发约定净值转换',
            'desc': '天天红B计息 · 分位数触发净值 · 越跌份额越大 · 止盈回货币 · 适合震荡市',
            'module': 'strategy6_gf_nav_conversion.strategy',
            'func': 'run_strategy',
            'color': '#00D9A3',
            'primary_symbol': '510300',
        },
    ]


def run_strategies_and_build_dashboard(initial_cash=1_000_000):
    strats = _strategy_defs()
    results = []
    payload = {
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'initial_cash': initial_cash,
        'strategies': [],
    }

    for i, s in enumerate(strats):
        print()
        print("=" * 70)
        print(f"  [{i+1}/{len(strats)}] 正在运行 {s['full_name']}")
        print("=" * 70)

        t0 = time.time()
        metrics = trade_stats = equity_df = engine = None
        try:
            mod = __import__(s['module'], fromlist=[s['func']])
            run_func = getattr(mod, s['func'])
            metrics, trade_stats, equity_df, engine = run_func(initial_cash=initial_cash)
        except Exception as exc:
            print(f"  ❌ 运行失败: {exc}")
            traceback.print_exc()

        elapsed = time.time() - t0
        print(f"  ⏱  耗时: {elapsed:.2f} 秒")

        # 组装可视化数据
        strat_payload = _build_strategy_payload(s, metrics, trade_stats, equity_df, engine)
        payload['strategies'].append(strat_payload)
        results.append((s, metrics, trade_stats))

    return payload, results


def _build_strategy_payload(strat_def, metrics, trade_stats, equity_df, engine):
    p = {
        'id': strat_def['id'],
        'short_name': strat_def['short_name'],
        'full_name': strat_def['full_name'],
        'desc': strat_def['desc'],
        'color': strat_def['color'],
        'metrics': metrics or {},
        'trade_stats': trade_stats or {},
    }

    # 权益曲线与回撤
    if equity_df is not None and len(equity_df) > 0:
        dates = equity_df.index.strftime('%Y-%m-%d').tolist()
        p['equity_dates'] = dates
        p['equity_values'] = (equity_df['equity'].astype(float) * 100).round(3).tolist()
        p['drawdown_values'] = equity_df['drawdown'].astype(float).round(3).tolist()
        p['total_values'] = equity_df['total_value'].astype(float).round(2).tolist()
        # 日收益率
        ret = equity_df['return'].fillna(0).astype(float)
        p['daily_returns'] = (ret * 100).round(3).tolist()

    # 交易记录
    trades = []
    if engine is not None:
        for t in engine.trades:
            trades.append({
                'date': t.date.strftime('%Y-%m-%d'),
                'symbol': t.symbol,
                'action': '买入' if t.action == 'buy' else '卖出',
                'price': round(float(t.price), 4),
                'shares': int(t.shares),
                'amount': round(float(t.shares) * float(t.price), 2),
                'commission': round(float(t.commission), 2),
            })
    p['trades'] = trades

    # K线主图 + 买卖点标记（取主要标的）
    primary_sym = strat_def['primary_symbol']
    kline = _collect_kline_with_points(engine, equity_df, primary_sym)
    p['kline'] = kline

    # 多标的持仓变化（用于轮动策略）
    holding_heatmap = _collect_holding_heatmap(engine)
    p['holding_heatmap'] = holding_heatmap

    return p


def _collect_kline_with_points(engine, equity_df, primary_sym):
    """从回测引擎价格数据中提取K线，并叠加对应标的的买卖点"""
    result = {
        'symbol': primary_sym,
        'dates': [],
        'ohlc': [],   # [open, close, low, high]
        'volumes': [],
        'buy_points': [],  # [date_idx, price, symbol]
        'sell_points': [],
    }

    if engine is None or primary_sym not in engine.price_data:
        # 没有主标的就用所有交易中出现最多的symbol
        if engine is not None and engine.trades:
            from collections import Counter
            sym_counter = Counter(t.symbol for t in engine.trades)
            if sym_counter:
                primary_sym = sym_counter.most_common(1)[0][0]
                result['symbol'] = primary_sym
        if engine is None or primary_sym not in engine.price_data:
            return result

    df = engine.price_data[primary_sym]
    dates_list = df.index.strftime('%Y-%m-%d').tolist()
    date_to_idx = {d: i for i, d in enumerate(dates_list)}

    result['dates'] = dates_list
    result['ohlc'] = df[['open', 'close', 'low', 'high']].round(4).values.tolist()
    vols = df['volume'].fillna(0).astype(float).tolist()
    # 归一化成交量显示
    max_v = max(vols) if vols else 1
    result['volumes'] = [round(v / max_v * 100, 2) for v in vols]

    # 叠加该标的的买卖点
    if engine is not None:
        for t in engine.trades:
            if t.symbol != primary_sym:
                # 对于轮动策略，我们把非主标的也显示在价格线上，用symbol区分
                pass
            ds = t.date.strftime('%Y-%m-%d')
            if ds in date_to_idx:
                idx = date_to_idx[ds]
            else:
                # 找最近交易日
                closest = min(dates_list, key=lambda x: abs(pd.Timestamp(x).timestamp() - t.date.timestamp()))
                idx = date_to_idx[closest]
            item = [idx, round(float(t.price), 4), t.symbol, t.date.strftime('%m-%d')]
            if t.action == 'buy':
                result['buy_points'].append(item)
            else:
                result['sell_points'].append(item)

    return result


def _collect_holding_heatmap(engine):
    """收集每日持仓 => 持仓热力图（symbol × date）"""
    if engine is None or not engine.daily_values:
        return {'symbols': [], 'dates': [], 'values': []}

    all_symbols = set()
    for dv in engine.daily_values:
        for sym in dv.get('positions', {}).keys():
            all_symbols.add(sym)
    all_symbols = sorted(all_symbols)

    dates = [dv['date'].strftime('%Y-%m-%d') for dv in engine.daily_values]
    symbol_index = {s: i for i, s in enumerate(all_symbols)}

    # 每个交易日，对每个symbol，记录其占总净值百分比
    values = []
    for dv in engine.daily_values:
        total = max(dv['total_value'], 1)
        pos = dv.get('positions', {})
        for sym, info in pos.items():
            if sym not in symbol_index:
                continue
            pct = info.get('value', 0) / total * 100
            values.append([
                dv['date'].strftime('%Y-%m-%d'),
                sym,
                round(pct, 2)
            ])

    return {
        'symbols': all_symbols,
        'dates': dates,
        'values': values,
    }


def save_payload(payload, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, 'dashboard_data.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    return path
