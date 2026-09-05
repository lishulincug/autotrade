"""
策略1：RRG 多因子行业轮动策略
=====================================
核心逻辑：融合相对强弱旋转图（RRG）与景气度、资金流多因子打分，
在行业ETF中捕捉"领涨且动量延续"的标的，赚取结构性行情超额收益。

因子体系：
  1. 景气度因子：盈利增速、分析师一致预期上调（模拟）
  2. 量价动量：20日涨跌幅、相对强弱RS值
  3. 资金流：北向资金流入、ETF份额净增长（模拟）
  4. 估值安全：PE/PB历史分位（模拟）

买卖规则：
  - 买入：月末调仓日入选 Top 3-5，且处于 RRG "领先象限"
  - 卖出：调出 Top 榜单，或跌入"落后象限"，或单只回撤超 8% 强制止损
  - 调仓频率：每月 1 次
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_industry_etfs
from common.metrics import calculate_metrics, calculate_trade_stats, print_report


def calculate_rrg(all_data: dict, current_date: pd.Timestamp,
                  benchmark_code: str = '510300', lookback: int = 60):
    """
    计算 RRG（相对强弱旋转图）指标
    返回每个ETF的 (RS-Ratio, RS-Momentum)
    """
    rrg = {}

    # 构造基准（用所有ETF等权作为基准，模拟宽基）
    all_closes = pd.DataFrame()
    for code, df in all_data.items():
        all_closes[code] = df['close']

    if benchmark_code not in all_closes.columns:
        benchmark = all_closes.mean(axis=1)
    else:
        benchmark = all_closes[benchmark_code]

    for code in all_data.keys():
        if code not in all_closes.columns:
            continue
        # 相对价格
        rel_price = all_closes[code] / benchmark
        rel_price = rel_price.loc[:current_date]
        if len(rel_price) < lookback:
            rrg[code] = (50, 50)
            continue

        # RS-Ratio: 相对强弱比（长期趋势）
        rel_ma_long = rel_price.rolling(window=lookback).mean()
        rel_ma_short = rel_price.rolling(window=10).mean()
        rs_ratio = 100 + (rel_ma_short.iloc[-1] / rel_ma_long.iloc[-1] - 1) * 500 if rel_ma_long.iloc[-1] > 0 else 50

        # RS-Momentum: 相对动量（短期变化）
        rel_mom = rel_price.pct_change(periods=20) * 100
        rs_momentum = 100 + rel_mom.iloc[-1] * 3 if not pd.isna(rel_mom.iloc[-1]) else 50

        rrg[code] = (rs_ratio, rs_momentum)

    return rrg


def calculate_multi_factor_scores(all_data: dict, etf_info: dict,
                                   current_date: pd.Timestamp) -> pd.DataFrame:
    """
    计算多因子综合得分
    """
    scores = []

    for code, df in all_data.items():
        hist = df.loc[:current_date]
        if len(hist) < 60:
            continue

        close = hist['close']
        volume = hist['volume']

        # === 因子1：量价动量（权重30%）===
        ret_20d = close.iloc[-1] / close.iloc[-20] - 1 if len(close) >= 20 else 0
        ret_60d = close.iloc[-1] / close.iloc[-60] - 1 if len(close) >= 60 else 0
        # 相对强弱：涨幅排名
        momentum_score = (ret_20d * 0.6 + ret_60d * 0.4)

        # === 因子2：资金流（权重25%）===
        vol_20d = volume.rolling(20).mean()
        vol_5d = volume.rolling(5).mean()
        vol_ratio = vol_5d.iloc[-1] / vol_20d.iloc[-1] if vol_20d.iloc[-1] > 0 else 1
        # 量价配合：价涨量增为正
        price_trend = 1 if ret_20d > 0 else -1
        flow_score = (vol_ratio - 1) * price_trend * 10

        # === 因子3：景气度模拟（权重25%）===
        # 用近3个月收益和波动率作为景气代理指标
        ret_3m = close.iloc[-1] / close.iloc[-63] - 1 if len(close) >= 63 else 0
        vol_3m = close.pct_change().rolling(63).std().iloc[-1] * np.sqrt(252)
        prosperity_score = ret_3m - vol_3m * 0.3  # 风险调整收益

        # === 因子4：估值安全（权重20%）===
        # 用价格历史分位数模拟估值分位（价格越低分位越安全）
        price_quantile = close.iloc[-1] / close.rolling(252).max().iloc[-1] if len(close) >= 252 else 1.0
        valuation_score = (1 - price_quantile) * 2  # 越低估分数越高

        scores.append({
            'code': code,
            'name': etf_info[code]['name'] if code in etf_info else code,
            'momentum': momentum_score,
            'flow': flow_score,
            'prosperity': prosperity_score,
            'valuation': valuation_score,
        })

    df_score = pd.DataFrame(scores)

    # 标准化每个因子到 0-100 分
    for col in ['momentum', 'flow', 'prosperity', 'valuation']:
        if len(df_score) > 1:
            min_v = df_score[col].min()
            max_v = df_score[col].max()
            if max_v > min_v:
                df_score[col] = (df_score[col] - min_v) / (max_v - min_v) * 100
            else:
                df_score[col] = 50

    # 加权综合得分
    df_score['total_score'] = (
        df_score['momentum'] * 0.30 +
        df_score['flow'] * 0.25 +
        df_score['prosperity'] * 0.25 +
        df_score['valuation'] * 0.20
    )

    return df_score.sort_values('total_score', ascending=False)


def run_strategy(initial_cash: float = 1_000_000, top_n: int = 4,
                 stop_loss_pct: float = -8.0,
                 start_date: str = '2020-06-01',
                 end_date: str = '2026-07-31'):
    """
    运行 RRG 多因子行业轮动策略
    """
    print("正在加载行业ETF数据...")
    all_data, etf_info = load_industry_etfs()
    print(f"已加载 {len(all_data)} 只行业ETF")

    # 构建价格数据字典
    price_dict = {code: df for code, df in all_data.items()}
    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0003, slippage=0.001)
    engine.set_price_data(price_dict)

    # 获取回测日期范围（每月最后一个交易日调仓）
    all_dates = sorted(set().union(*[df.index for df in all_data.values()]))
    all_dates = [d for d in all_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date]

    # 找出每月调仓日（月末）
    rebalance_dates = []
    current_month = None
    for i in range(len(all_dates) - 1, -1, -1):
        d = all_dates[i]
        month = (d.year, d.month)
        if month != current_month:
            rebalance_dates.append(d)
            current_month = month
    rebalance_dates = sorted(rebalance_dates)

    print(f"回测区间: {start_date} ~ {end_date}")
    print(f"调仓次数: {len(rebalance_dates)} 次（每月月末）")
    print(f"持仓数量: Top {top_n}")
    print()

    current_holdings = set()
    buy_costs = {}  # 记录买入成本用于止损

    for idx, date in enumerate(all_dates):
        # 检查是否是调仓日
        if date in rebalance_dates:
            # 1. 计算因子得分
            scores_df = calculate_multi_factor_scores(all_data, etf_info, date)
            if len(scores_df) == 0:
                engine.record_daily_value(date)
                continue

            # 2. 计算 RRG 位置
            rrg = calculate_rrg(all_data, date)

            # 3. 筛选候选：Top得分 + RRG领先/改善象限
            top_candidates = scores_df.head(top_n * 2)['code'].tolist()

            buy_list = []
            for code in top_candidates:
                rs_ratio, rs_mom = rrg.get(code, (50, 50))
                # 领先象限：RS-Ratio > 100 且 RS-Momentum > 100
                # 改善象限：RS-Ratio < 100 且 RS-Momentum > 100（动量反转向上）
                if (rs_ratio >= 100 and rs_mom >= 100) or (rs_mom >= 105 and rs_ratio >= 95):
                    buy_list.append(code)
                if len(buy_list) >= top_n:
                    break

            # 如果领先象限不够，直接用得分最高的
            if len(buy_list) < top_n:
                for code in top_candidates:
                    if code not in buy_list:
                        buy_list.append(code)
                    if len(buy_list) >= top_n:
                        break

            # 4. 执行调仓：先卖出不在buy_list的
            sell_list = [c for c in current_holdings if c not in buy_list]
            for code in sell_list:
                rs_ratio, rs_mom = rrg.get(code, (50, 50))
                # 落后象限：RS-Ratio < 100 且 RS-Momentum < 100 额外惩罚
                engine.sell(code, date)
                buy_costs.pop(code, None)
                current_holdings.discard(code)

            # 5. 买入新标的
            if buy_list:
                # 等权分配
                current_value = engine.get_total_value()
                cash_per_etf = current_value / len(buy_list)

                for code in buy_list:
                    if code not in current_holdings:
                        if engine.buy(code, date, amount=cash_per_etf):
                            price = price_dict[code].loc[:date, 'close'].iloc[-1]
                            buy_costs[code] = price
                            current_holdings.add(code)
                    else:
                        # 已持有，检查是否需要再平衡
                        pass

        # 每日检查止损
        for code in list(current_holdings):
            if code in price_dict:
                close_hist = price_dict[code].loc[:date, 'close']
                if len(close_hist) > 0 and code in buy_costs:
                    current_price = close_hist.iloc[-1]
                    ret = (current_price / buy_costs[code] - 1) * 100
                    if ret <= stop_loss_pct:
                        print(f"  [止损] {date.strftime('%Y-%m-%d')} {code} 回撤{ret:.2f}% 触发止损")
                        engine.sell(code, date)
                        buy_costs.pop(code, None)
                        current_holdings.discard(code)

        engine.record_daily_value(date)

    # 获取结果
    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="RRG多因子行业轮动策略")
    trade_stats = calculate_trade_stats(engine.trades)

    print_report(metrics, trade_stats, title="策略1：RRG 多因子行业轮动策略")

    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    equity_df.to_csv(os.path.join(output_dir, 'rrg_equity_curve.csv'))

    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
