"""
策略绩效统计分析模块
"""
import numpy as np
import pandas as pd


def calculate_metrics(equity_df: pd.DataFrame, initial_cash: float = 1_000_000,
                      risk_free_rate: float = 0.03,
                      name: str = "策略") -> dict:
    """
    计算策略回测绩效指标
    """
    if equity_df is None or len(equity_df) < 2:
        return {
            '策略名称': name,
            '回测天数': 0,
            '年化收益率': 0,
            '累计收益率': 0,
            '最大回撤': 0,
            '最大回撤起始日': '-',
            '最大回撤结束日': '-',
            '夏普比率': 0,
            '卡玛比率': 0,
            '索提诺比率': 0,
            '胜率': 0,
            '盈亏比': 0,
            '波动率(年化)': 0,
            '总交易次数': 0,
            '期末总资产': 0,
        }

    df = equity_df.copy()
    total_days = len(df)
    years = total_days / 252

    # 收益率
    total_value = df['total_value'].values
    total_return = (total_value[-1] / total_value[0] - 1) * 100
    annual_return = ((total_value[-1] / total_value[0]) ** (1 / max(years, 0.01)) - 1) * 100

    # 最大回撤
    cumulative_max = df['total_value'].cummax()
    drawdown = (df['total_value'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100

    # 最大回撤起止日期
    max_dd_end_idx = drawdown.argmin()
    max_dd_start_idx = cumulative_max.iloc[:max_dd_end_idx + 1].argmax() if max_dd_end_idx > 0 else 0
    max_dd_start = df.index[max_dd_start_idx].strftime('%Y-%m-%d') if max_dd_start_idx < len(df) else '-'
    max_dd_end = df.index[max_dd_end_idx].strftime('%Y-%m-%d') if max_dd_end_idx < len(df) else '-'

    # 日收益率
    daily_returns = df['return'].dropna()

    # 年化波动率
    annual_vol = daily_returns.std() * np.sqrt(252) * 100

    # 夏普比率
    excess_return = daily_returns - risk_free_rate / 252
    if excess_return.std() > 0:
        sharpe = np.sqrt(252) * excess_return.mean() / excess_return.std()
    else:
        sharpe = 0

    # 卡玛比率
    if max_drawdown < 0:
        calmar = annual_return / abs(max_drawdown)
    else:
        calmar = 0

    # 索提诺比率（下行波动率）
    downside_returns = daily_returns[daily_returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() > 0:
        sortino = np.sqrt(252) * excess_return.mean() / (downside_returns.std())
    else:
        sortino = 0

    return {
        '策略名称': name,
        '回测天数': total_days,
        '年化收益率(%)': round(annual_return, 2),
        '累计收益率(%)': round(total_return, 2),
        '最大回撤(%)': round(max_drawdown, 2),
        '最大回撤起始日': max_dd_start,
        '最大回撤结束日': max_dd_end,
        '夏普比率': round(sharpe, 3),
        '卡玛比率': round(calmar, 3),
        '索提诺比率': round(sortino, 3),
        '波动率(年化%)': round(annual_vol, 2),
        '期末总资产(元)': round(total_value[-1], 2),
    }


def calculate_trade_stats(trades: list, positions_daily: list = None) -> dict:
    """
    计算交易统计信息
    """
    if not trades:
        return {
            '总交易次数': 0,
            '买入次数': 0,
            '卖出次数': 0,
            '盈利次数': 0,
            '亏损次数': 0,
            '胜率(%)': 0,
            '平均盈利(%)': 0,
            '平均亏损(%)': 0,
            '盈亏比': 0,
            '总手续费(元)': 0,
        }

    buy_trades = [t for t in trades if t.action == 'buy']
    sell_trades = [t for t in trades if t.action == 'sell']
    total_commission = sum(t.commission for t in trades)

    # 估算胜率和盈亏比（简化：基于买卖对）
    wins = 0
    losses = 0
    win_pcts = []
    loss_pcts = []

    # 简单的逐笔匹配（先进先出）
    buy_queue = []
    for t in trades:
        if t.action == 'buy':
            buy_queue.append({'price': t.price, 'shares': t.shares})
        elif t.action == 'sell' and buy_queue:
            sell_price = t.price
            remaining_shares = t.shares
            while remaining_shares > 0 and buy_queue:
                buy = buy_queue[0]
                matched_shares = min(remaining_shares, buy['shares'])
                pct = (sell_price - buy['price']) / buy['price'] * 100
                if pct > 0:
                    wins += 1
                    win_pcts.append(pct)
                else:
                    losses += 1
                    loss_pcts.append(pct)
                buy['shares'] -= matched_shares
                remaining_shares -= matched_shares
                if buy['shares'] == 0:
                    buy_queue.pop(0)

    total_round = wins + losses
    win_rate = (wins / total_round * 100) if total_round > 0 else 0
    avg_win = (sum(win_pcts) / len(win_pcts)) if win_pcts else 0
    avg_loss = (sum(loss_pcts) / len(loss_pcts)) if loss_pcts else 0
    profit_loss_ratio = (abs(avg_win / avg_loss)) if avg_loss != 0 else 0

    return {
        '总交易次数': len(trades),
        '买入次数': len(buy_trades),
        '卖出次数': len(sell_trades),
        '盈利次数': wins,
        '亏损次数': losses,
        '胜率(%)': round(win_rate, 2),
        '平均盈利(%)': round(avg_win, 2),
        '平均亏损(%)': round(avg_loss, 2),
        '盈亏比': round(profit_loss_ratio, 2),
        '总手续费(元)': round(total_commission, 2),
    }


def print_report(metrics: dict, trade_stats: dict, title: str = "策略回测报告"):
    """打印回测报告"""
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

    print("\n【绩效指标】")
    print("-" * 70)
    for k, v in metrics.items():
        print(f"  {k:<18s}: {v}")

    print("\n【交易统计】")
    print("-" * 70)
    for k, v in trade_stats.items():
        print(f"  {k:<18s}: {v}")

    print("\n" + "=" * 70)


def compare_strategies(strategy_results: list) -> pd.DataFrame:
    """
    对比多个策略的绩效
    strategy_results: [(策略名, metrics_dict, trade_stats_dict), ...]
    """
    rows = []
    for name, metrics, trade_stats in strategy_results:
        row = {'策略': name}
        row.update(metrics)
        row.update(trade_stats)
        rows.append(row)

    df = pd.DataFrame(rows)
    return df
