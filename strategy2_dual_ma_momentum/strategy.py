"""
策略2：双均线动量轮动 + 空仓风控策略
=======================================
核心逻辑：基于"强者恒强"的价格动量效应，叠加均线趋势过滤，
跨大类资产轮动；弱势环境可完全空仓防守。

标的池（6-8只核心大类资产）：
  沪深300ETF、创业板ETF、中概互联ETF、纳指100ETF、黄金ETF、十年国债ETF

买入条件（需同时满足，每日收盘执行）：
  ① 20日涨幅全池排名第一
  ② 收盘价站稳 28 日均线

卖出条件（触发任一即可）：
  ① 20日涨幅掉出第一
  ② 收盘价跌破 28 日均线

终极风控：
  所有标的均不满足买入条件时，100% 空仓持币
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_asset_etfs
from common.metrics import calculate_metrics, calculate_trade_stats, print_report


def run_strategy(initial_cash: float = 1_000_000,
                 ma_period: int = 28,  # 28日均线
                 momentum_period: int = 20,  # 20日涨幅动量
                 start_date: str = '2020-06-01',
                 end_date: str = '2026-07-31'):
    """
    运行双均线动量轮动策略
    """
    print("正在加载大类资产ETF数据...")
    all_data, etf_info = load_asset_etfs()
    print(f"已加载 {len(all_data)} 只大类资产ETF")
    for code, info in etf_info.items():
        print(f"  {code}: {info['name']}")

    # 构建价格数据字典
    price_dict = {code: df for code, df in all_data.items()}
    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0003, slippage=0.001)
    engine.set_price_data(price_dict)

    # 获取回测日期范围
    all_dates = sorted(set().union(*[df.index for df in all_data.values()]))
    all_dates = [d for d in all_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date]

    print(f"\n回测区间: {start_date} ~ {end_date}")
    print(f"均线周期: MA{ma_period}")
    print(f"动量周期: {momentum_period}日涨幅排名")
    print(f"调仓频率: 每日收盘后判断")
    print()

    current_holding = None  # 当前持有的标的（单只满仓）

    for idx, date in enumerate(all_dates):
        if idx < max(ma_period, momentum_period) + 5:
            engine.record_daily_value(date)
            continue

        # 1. 计算每个标的的信号
        signals = {}
        for code, df in all_data.items():
            hist = df.loc[:date, 'close']
            if len(hist) < max(ma_period, momentum_period):
                continue

            # 20日涨幅
            ret_20d = hist.iloc[-1] / hist.iloc[-momentum_period] - 1

            # 28日均线
            ma = hist.rolling(ma_period).mean()
            if len(ma) < 1 or pd.isna(ma.iloc[-1]):
                continue
            ma_value = ma.iloc[-1]
            current_price = hist.iloc[-1]

            # 是否站稳均线
            above_ma = current_price >= ma_value

            signals[code] = {
                'ret_20d': ret_20d,
                'ma': ma_value,
                'price': current_price,
                'above_ma': above_ma,
            }

        if not signals:
            engine.record_daily_value(date)
            continue

        # 2. 按20日涨幅排序
        sorted_signals = sorted(signals.items(), key=lambda x: x[1]['ret_20d'], reverse=True)
        ranked_codes = [code for code, _ in sorted_signals]

        # 3. 寻找满足条件的标的
        target_code = None
        for code in ranked_codes:
            sig = signals[code]
            if sig['above_ma']:  # 站稳均线
                target_code = code
                break

        # 4. 执行交易
        if current_holding is not None:
            hold_sig = signals.get(current_holding, {})
            # 判断是否需要卖出当前持仓
            need_sell = False
            if not hold_sig.get('above_ma', False):
                # 跌破28日均线
                need_sell = True
            elif target_code != current_holding:
                # 20日涨幅不再排名第一且被其他站稳均线的标的超过
                if len(ranked_codes) > 0:
                    current_rank = ranked_codes.index(current_holding) if current_holding in ranked_codes else 999
                    if current_rank >= 1:
                        need_sell = True

            if need_sell:
                engine.sell(current_holding, date)
                current_holding = None

        # 5. 买入新标的
        if current_holding is None and target_code is not None:
            if engine.buy(target_code, date, amount=engine.cash * 0.99):
                current_holding = target_code

        engine.record_daily_value(date)

    # 获取结果
    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="双均线动量轮动+空仓风控策略")
    trade_stats = calculate_trade_stats(engine.trades)

    # 统计空仓天数
    empty_days = 0
    for dv in engine.daily_values:
        if not dv['positions']:
            empty_days += 1
    total_days = len(engine.daily_values)
    trade_stats['空仓天数'] = empty_days
    trade_stats['空仓占比(%)'] = round(empty_days / total_days * 100, 2) if total_days > 0 else 0

    print_report(metrics, trade_stats, title="策略2：双均线动量轮动 + 空仓风控策略")

    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    equity_df.to_csv(os.path.join(output_dir, 'dual_ma_equity_curve.csv'))

    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
