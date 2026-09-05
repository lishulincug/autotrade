"""
策略5：指数增强多因子选股策略
=================================
核心逻辑：在指数成分股内部，通过价值、质量、红利、成长等
多因子量化选股，构建增强组合，赚取 Alpha 超额收益。

因子体系：
  价值因子：低 PE、低 PB、高股息率
  质量因子：高 ROE、盈利稳定、低资产负债率
  成长因子：营收/净利润同比增速

买卖规则（季度调仓）：
  - 买入：指数估值分位低于 30% 时分批建仓，选因子得分前20%个股
  - 持有：中长期持有，按季度/半年度自动调仓
  - 卖出：指数估值分位高于 80% 止盈，或连续 2 期因子得分靠后剔除
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_index_components, generate_etf_data
from common.metrics import calculate_metrics, calculate_trade_stats, print_report


def calculate_stock_factors(stock_data: dict, current_date: pd.Timestamp) -> pd.DataFrame:
    """
    计算个股多因子得分
    """
    results = []

    for code, info in stock_data.items():
        df = info['data']
        hist = df.loc[:current_date]
        if len(hist) < 60:
            continue

        close = hist['close']
        volume = hist['volume']
        returns = close.pct_change().dropna()

        # === 价值因子（模拟 PE/PB 倒数，高股息率）===
        # 用价格/长期均值模拟估值水平（越低越便宜）
        price_to_avg = close.iloc[-1] / close.rolling(252).mean().iloc[-1] if len(close) >= 252 else 1.0
        value_pe = 1 / max(price_to_avg, 0.1) * 10
        # 股息率模拟：稳定低波动股给高分
        vol_1y = returns.rolling(252).std().iloc[-1] if len(returns) >= 252 else 0.02
        dividend_score = max(0, (0.03 - min(vol_1y, 0.03))) / 0.03 * 10
        value_score = value_pe * 0.6 + dividend_score * 0.4

        # === 质量因子（ROE模拟、盈利稳定性、低负债率）===
        # 用净值曲线的平滑度模拟盈利稳定性
        cumulative = (1 + returns).cumprod()
        residuals = np.log(cumulative) - np.polyval(np.polyfit(range(len(cumulative)),
                                                                np.log(cumulative), 1),
                                                    range(len(cumulative)))
        stability = 1 / (1 + residuals.std() * 10) * 10
        # 低波动=高质量
        quality_vol = max(0, (0.04 - min(returns.std() * np.sqrt(252), 0.04))) / 0.04 * 10
        quality_score = stability * 0.5 + quality_vol * 0.5

        # === 成长因子（盈利增速模拟）===
        # 长期动量 + 短期加速
        ret_6m = close.iloc[-1] / close.iloc[-126] - 1 if len(close) >= 126 else 0
        ret_3m = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0
        growth_score = (ret_6m * 0.4 + ret_3m * 0.6) * 30 + 5
        growth_score = max(0, growth_score)

        # === 红利因子（模拟）===
        # 长期上涨稳定股视为分红能力强
        long_term_trend = close.iloc[-1] / close.iloc[0] - 1 if len(close) > 0 else 0
        dividend_score_full = max(0, min(long_term_trend * 2, 1)) * 10

        results.append({
            'code': code,
            'name': info['name'],
            'value': value_score,
            'quality': quality_score,
            'growth': growth_score,
            'dividend': dividend_score_full,
            'price': close.iloc[-1],
        })

    df_score = pd.DataFrame(results)
    if len(df_score) == 0:
        return df_score

    # 标准化各因子到 0-100
    for col in ['value', 'quality', 'growth', 'dividend']:
        min_v = df_score[col].min()
        max_v = df_score[col].max()
        if max_v > min_v:
            df_score[col] = (df_score[col] - min_v) / (max_v - min_v) * 100
        else:
            df_score[col] = 50

    # 综合得分：价值30% + 质量30% + 成长25% + 红利15%
    df_score['total_score'] = (
        df_score['value'] * 0.30 +
        df_score['quality'] * 0.30 +
        df_score['growth'] * 0.25 +
        df_score['dividend'] * 0.15
    )

    return df_score.sort_values('total_score', ascending=False)


def get_index_valuation(benchmark_data: pd.DataFrame,
                        current_date: pd.Timestamp) -> float:
    """
    计算指数估值分位数（0~100%）
    """
    hist = benchmark_data.loc[:current_date, 'close']
    if len(hist) < 252:
        return 50.0

    one_year = hist.iloc[-252:]
    current = hist.iloc[-1]
    rank = (one_year < current).sum() / len(one_year) * 100
    return rank


def run_strategy(initial_cash: float = 1_000_000,
                 top_pct: float = 0.20,  # 选前20%的股票
                 max_holdings: int = 10,  # 最多持有10只
                 rebalance_freq: str = 'Q',  # 季度调仓
                 start_date: str = '2020-06-01',
                 end_date: str = '2026-07-31'):
    """
    运行指数增强多因子选股策略
    """
    print("正在加载指数成分股数据...")
    stock_data = load_index_components()
    print(f"已加载 {len(stock_data)} 只成分股")

    # 同时生成基准指数对比
    benchmark_data = generate_etf_data(
        '基准指数', base_price=1.0, annual_return=0.08,
        volatility=0.25, seed=999
    )
    benchmark_code = 'BENCHMARK'

    # 构建价格数据字典
    price_dict = {}
    for code, info in stock_data.items():
        price_dict[code] = info['data']
    price_dict[benchmark_code] = benchmark_data

    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0005, slippage=0.002)
    engine.set_price_data(price_dict)

    # 获取回测日期
    all_dates = sorted(set().union(*[df.index for df in price_dict.values()]))
    all_dates = [d for d in all_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date]

    # 找出季度调仓日
    rebalance_dates = []
    current_quarter = None
    for d in all_dates:
        q = (d.year, (d.month - 1) // 3 + 1)
        if q != current_quarter:
            rebalance_dates.append(d)
            current_quarter = q

    print(f"\n回测区间: {start_date} ~ {end_date}")
    print(f"调仓频率: {len(rebalance_dates)} 次（每季度）")
    print(f"选股范围: 前{int(top_pct*100)}%因子得分股，最多持仓 {max_holdings} 只")
    print()

    current_holdings = set()
    rebalance_count = 0

    for date in all_dates:
        # 季度调仓
        if date in rebalance_dates:
            rebalance_count += 1
            val_pct = get_index_valuation(benchmark_data, date)
            scores_df = calculate_stock_factors(stock_data, date)

            if len(scores_df) == 0:
                engine.record_daily_value(date)
                continue

            # 决定仓位比例（基于估值分位）
            if val_pct < 30:
                # 低估：90%~100%仓位
                position_ratio = 0.90 + (30 - val_pct) / 100
                build_msg = f"低估区(估值{val_pct:.0f}%)，高仓位"
            elif val_pct < 50:
                position_ratio = 0.70
                build_msg = f"合理偏低(估值{val_pct:.0f}%)，正常仓位"
            elif val_pct < 80:
                position_ratio = 0.50
                build_msg = f"合理偏高(估值{val_pct:.0f}%)，中等仓位"
            else:
                position_ratio = 0.20
                build_msg = f"高估区(估值{val_pct:.0f}%)，低仓位止盈"

            if rebalance_count <= 4 or rebalance_count % 4 == 0:
                print(f"[{date.strftime('%Y-%m-%d')}] Q调仓 #{rebalance_count} | {build_msg}")
                print(f"  因子得分Top5: {', '.join([f'{r.code}({r.total_score:.1f})' for _, r in scores_df.head(5).iterrows()])}")

            # 选前 N 只
            n_pick = min(max_holdings, max(3, int(len(scores_df) * top_pct)))
            target_holdings = set(scores_df.head(n_pick)['code'].tolist())

            # 卖出不在目标的
            for code in list(current_holdings):
                if code not in target_holdings:
                    engine.sell(code, date)
                    current_holdings.discard(code)

            # 计算目标资金量
            current_value = engine.get_total_value()
            target_cash_total = current_value * position_ratio
            cash_per_stock = target_cash_total / len(target_holdings) if target_holdings else 0

            # 买入/调平目标持仓
            for code in target_holdings:
                if code not in current_holdings:
                    if engine.buy(code, date, amount=cash_per_stock):
                        current_holdings.add(code)
                else:
                    # 再平衡：检查仓位偏差超过30%就调整
                    if code in engine.positions:
                        pos = engine.positions[code]
                        if abs(pos.market_value - cash_per_stock) / max(cash_per_stock, 1) > 0.3:
                            engine.sell(code, date)
                            current_holdings.discard(code)
                            if engine.buy(code, date, amount=cash_per_stock):
                                current_holdings.add(code)

        engine.record_daily_value(date)

    # 获取结果
    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="指数增强多因子选股策略")
    trade_stats = calculate_trade_stats(engine.trades)

    # 计算基准收益对比
    bench_start = benchmark_data.loc[benchmark_data.index >= all_dates[0], 'close'].iloc[0]
    bench_end = benchmark_data.loc[benchmark_data.index <= all_dates[-1], 'close'].iloc[-1]
    bench_return = (bench_end / bench_start - 1) * 100
    trade_stats['基准指数收益率(%)'] = round(bench_return, 2)
    trade_stats['超额Alpha(%)'] = round(metrics['累计收益率(%)'] - bench_return, 2)

    trade_stats['平均持股数量'] = round(len(current_holdings), 1)
    trade_stats['调仓次数'] = rebalance_count

    print_report(metrics, trade_stats, title="策略5：指数增强多因子选股策略")

    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    equity_df.to_csv(os.path.join(output_dir, 'enhance_equity_curve.csv'))

    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
