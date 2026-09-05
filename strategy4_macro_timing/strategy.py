"""
策略4：宏观因子择时策略
==========================
核心逻辑：通过利率、通胀、流动性等宏观变量判断市场整体风险，
动态调整股票 ETF 仓位比例，熊市降仓防守、牛市加仓进攻。

宏观信号体系（月度/季度调仓）：
  加仓信号（股票仓位 80%~100%）：
    - 降息周期开启（利率趋势下行）
    - 信用利差收窄
    - PMI 回升（>50 扩张区间）

  减仓信号（股票仓位 0%~20%）：
    - 通胀上行（CPI 持续走高）
    - 流动性收紧（M2 增速下降）
    - 市场波动率飙升（VIX 类指标）

  空仓信号：
    - 宏观衰退预警叠加技术面破位 → 全仓切换防守资产（国债/黄金）

标的配置：
  - 进攻端：沪深 300、中证 500 宽基 ETF
  - 防守端：十年国债 ETF、黄金 ETF
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_asset_etfs, generate_macro_factors
from common.metrics import calculate_metrics, calculate_trade_stats, print_report


def calculate_macro_score(factors_df: pd.DataFrame, current_date: pd.Timestamp) -> dict:
    """
    计算宏观综合得分与信号
    返回: {'score': -100~100, 'position_pct': 0~1, 'defensive_pct': 0~1, ...}
    """
    # 取当前及过去的数据
    if current_date not in factors_df.index:
        idx = factors_df.index.get_indexer([current_date], method='pad')[0]
        if idx < 0:
            return {'score': 0, 'position_pct': 0.5, 'defensive_pct': 0.0}
        current_date = factors_df.index[idx]

    # 过去12个月
    hist = factors_df.loc[:current_date]
    if len(hist) < 6:
        return {'score': 50, 'position_pct': 0.5, 'defensive_pct': 0.0}

    recent = hist.iloc[-6:]
    older = hist.iloc[-12:-6] if len(hist) >= 12 else hist.iloc[:len(hist) // 2]

    # === 1. 利率因子（降息为正）===
    rate_now = recent['interest_rate'].mean()
    rate_pre = older['interest_rate'].mean() if len(older) > 0 else rate_now
    rate_delta = (rate_pre - rate_now) / max(rate_pre, 0.01) * 100
    rate_score = np.clip(rate_delta * 10, -20, 20)  # ±20分

    # === 2. 信用利差（收窄为正）===
    spread_now = recent['credit_spread'].mean()
    spread_pre = older['credit_spread'].mean() if len(older) > 0 else spread_now
    spread_delta = (spread_pre - spread_now) / max(spread_pre, 0.01) * 100
    spread_score = np.clip(spread_delta * 5, -15, 15)  # ±15分

    # === 3. PMI（景气扩张为正）===
    pmi_now = recent['pmi'].mean()
    pmi_delta = pmi_now - 50  # 50为荣枯线
    pmi_score = np.clip(pmi_delta * 3, -20, 20)  # ±20分

    # === 4. 通胀因子（温和为正，过高为负）===
    cpi_now = recent['cpi'].mean()
    if 1 <= cpi_now <= 3:
        cpi_score = 10  # 温和通胀
    elif cpi_now > 4:
        cpi_score = -15  # 高通胀
    elif cpi_now < 0:
        cpi_score = -10  # 通缩
    else:
        cpi_score = 5

    # === 5. 流动性因子（M2增速高为正）===
    m2_now = recent['m2'].mean()
    m2_score = np.clip((m2_now - 8) * 1.5, -15, 15)  # M2>8%为宽松

    # === 6. 波动率因子（低波动为正）===
    vix_now = recent['vix'].mean()
    vix_score = np.clip((30 - vix_now), -15, 15)  # VIX<30为正常

    total_score = (rate_score + spread_score + pmi_score +
                   cpi_score + m2_score + vix_score)

    # 映射到仓位比例
    if total_score >= 30:
        # 强进攻：80%~100%股票
        position_pct = 0.80 + min(total_score - 30, 40) / 200
        defensive_pct = 1 - position_pct
    elif total_score >= 10:
        # 温和进攻：50%~80%
        position_pct = 0.50 + (total_score - 10) / 66.7
        defensive_pct = 1 - position_pct
    elif total_score >= -10:
        # 中性：30%~50%
        position_pct = 0.30 + (total_score + 10) / 100
        defensive_pct = 1 - position_pct
    elif total_score >= -30:
        # 防守：10%~30%
        position_pct = 0.10 + (total_score + 30) / 100
        defensive_pct = 1 - position_pct
    else:
        # 空仓防守：0%~10%股票，其余全防守资产
        position_pct = max(0, total_score + 40) / 100
        defensive_pct = 1 - position_pct

    return {
        'score': total_score,
        'position_pct': position_pct,
        'defensive_pct': defensive_pct,
        'rate_score': rate_score,
        'spread_score': spread_score,
        'pmi_score': pmi_score,
        'cpi_score': cpi_score,
        'm2_score': m2_score,
        'vix_score': vix_score,
    }


def run_strategy(initial_cash: float = 1_000_000,
                 start_date: str = '2020-06-01',
                 end_date: str = '2026-07-31'):
    """
    运行宏观因子择时策略
    """
    print("正在加载大类资产ETF及宏观因子数据...")
    all_data, etf_info = load_asset_etfs()
    macro_factors = generate_macro_factors()
    print(f"已加载 {len(all_data)} 只ETF，宏观因子 {len(macro_factors)} 个月")

    # 划分进攻/防守标的
    offensive_codes = ['510300', '159915']  # 沪深300、创业板
    defensive_codes = ['511260', '518880']  # 国债、黄金

    # 构建价格数据字典
    price_dict = {code: df for code, df in all_data.items()}
    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0003, slippage=0.001)
    engine.set_price_data(price_dict)

    # 获取回测日期范围
    all_dates = sorted(set().union(*[df.index for df in all_data.values()]))
    all_dates = [d for d in all_dates if start_date <= d.strftime('%Y-%m-%d') <= end_date]

    # 找出每月调仓日（月初）
    rebalance_dates = []
    current_month = None
    for d in all_dates:
        month = (d.year, d.month)
        if month != current_month:
            rebalance_dates.append(d)
            current_month = month

    print(f"\n回测区间: {start_date} ~ {end_date}")
    print(f"调仓频率: {len(rebalance_dates)} 次（每月月初）")
    print(f"进攻标的: {[etf_info[c]['name'] for c in offensive_codes if c in etf_info]}")
    print(f"防守标的: {[etf_info[c]['name'] for c in defensive_codes if c in etf_info]}")
    print()

    signal_history = []

    for date in all_dates:
        # 月度调仓
        if date in rebalance_dates:
            signal = calculate_macro_score(macro_factors, date)
            signal['date'] = date
            signal_history.append(signal)

            pos_pct = signal['position_pct']
            def_pct = signal['defensive_pct']

            score = signal['score']
            if score >= 30:
                signal_name = "强进攻"
            elif score >= 10:
                signal_name = "温和进攻"
            elif score >= -10:
                signal_name = "中性观望"
            elif score >= -30:
                signal_name = "谨慎防守"
            else:
                signal_name = "空仓避险"

            print(f"[{date.strftime('%Y-%m-%d')}] 宏观得分:{score:+.1f} → {signal_name} | "
                  f"股票仓位:{pos_pct*100:.0f}%  防守仓位:{def_pct*100:.0f}%")

            current_value = engine.get_total_value()

            # 先清仓
            engine.sell_all(date)

            # 买入进攻标的
            if pos_pct > 0.01:
                cash_per_off = current_value * pos_pct / len(offensive_codes)
                for code in offensive_codes:
                    if code in all_data:
                        engine.buy(code, date, amount=cash_per_off)

            # 买入防守标的
            if def_pct > 0.01:
                cash_per_def = current_value * def_pct / len(defensive_codes)
                for code in defensive_codes:
                    if code in all_data:
                        engine.buy(code, date, amount=cash_per_def)

        engine.record_daily_value(date)

    # 获取结果
    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="宏观因子择时策略")
    trade_stats = calculate_trade_stats(engine.trades)

    # 仓位统计
    offensive_days = 0
    defensive_days = 0
    for dv in engine.daily_values:
        has_off = any(c in dv['positions'] for c in offensive_codes)
        has_def = any(c in dv['positions'] for c in defensive_codes)
        if has_off:
            offensive_days += 1
        if has_def:
            defensive_days += 1
    total_days = len(engine.daily_values)
    trade_stats['持有进攻资产天数'] = offensive_days
    trade_stats['持有防守资产天数'] = defensive_days
    trade_stats['股票仓位平均(%)'] = round(offensive_days / total_days * 100, 1) if total_days > 0 else 0

    print_report(metrics, trade_stats, title="策略4：宏观因子择时策略")

    # 保存结果
    output_dir = os.path.dirname(os.path.abspath(__file__))
    equity_df.to_csv(os.path.join(output_dir, 'macro_equity_curve.csv'))
    if signal_history:
        pd.DataFrame(signal_history).to_csv(os.path.join(output_dir, 'macro_signals.csv'), index=False)

    return metrics, trade_stats, equity_df, engine


if __name__ == '__main__':
    run_strategy()
