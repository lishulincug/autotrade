"""
策略6-A：广发约定净值转换 · 动态滚动网格
============================================
核心逻辑：基于历史净值分位数设定触发净值，在天天红B（广发货币基金）与目标基金
之间进行约定转换。越跌份额越大，达到止盈净值转回货币基金。

机制说明：
  1. 资金存放在天天红B（广发货币基金，年化约2%），每日计息
  2. 当基金净值 ≤ 约定净值 → 天天红B → 目标基金（买入，越跌份额越大）
  3. 当基金净值 ≥ 约定净值 → 目标基金 → 天天红B（止盈）
  4. 止盈后，低于止盈净值的买入档自动重置，形成"买低→卖高"循环

参数生成：基于历史净值的分位数（10%分位=低位，90%分位=高位）划定区间
标的基金：广发中证红利ETF发起式联接C（021400）
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.backtest_engine import BacktestEngine
from common.data_loader import load_otc_fund_nav
from common.metrics import calculate_metrics, calculate_trade_stats, print_report


# 目标基金（默认，可被参数覆盖）
FUND_CODE = '021400'
FUND_NAME = '广发中证红利ETF发起式联接C'


def resolve_backtest_dates(nav_data, start_date=None, end_date=None, warmup_days=60):
    """
    根据可用净值数据自适应确定回测日期区间与warmup净值序列
    - start_date=None: 自动跳过前 warmup_days 条作为预热
    - start_date给定: warmup=该日期之前的数据
    返回: (all_dates, warmup_navs, end_date)
    """
    if end_date is None:
        end_date = nav_data.index[-1].strftime('%Y-%m-%d')

    if start_date is None:
        # 自动：前 warmup_days 条预热，之后回测
        idx_dates = list(nav_data.index)
        if len(idx_dates) <= warmup_days + 30:
            # 数据太少，至少留30条回测
            split = max(10, len(idx_dates) // 4)
            warmup_navs = nav_data['close'].iloc[:split]
            all_dates = idx_dates[split:]
        else:
            warmup_navs = nav_data['close'].iloc[:warmup_days]
            all_dates = idx_dates[warmup_days:]
        end_date = nav_data.index[-1].strftime('%Y-%m-%d')
        return all_dates, warmup_navs, end_date

    all_dates = [d for d in nav_data.index
                 if start_date <= d.strftime('%Y-%m-%d') <= end_date]
    warmup_data = nav_data.loc[nav_data.index < all_dates[0]] if all_dates else nav_data.iloc[:0]
    if len(warmup_data) >= 20:
        warmup_navs = warmup_data['close']
    else:
        warmup_navs = nav_data['close'].iloc[:max(20, len(nav_data) // 4)]
    return all_dates, warmup_navs, end_date


def generate_grid_params(net_value_series,
                         buy_low_quantile=0.10,
                         sell_high_quantile=0.90,
                         buy_tier_count=4,
                         sell_tier_count=2,
                         base_share=5000,
                         share_increment=2000,
                         buy_high_quantile=0.75,
                         sell_low_quantile=0.80,
                         sell_share=5000,
                         decimals=2):
    """
    根据历史净值序列生成网格参数（广发约定净值转换用）

    参数：
        net_value_series: pd.Series 或 list，基金历史净值
        buy_low_quantile: float，买入区间下限分位数（越小越深，如0.10=10%分位）
        buy_high_quantile: float，买入区间上限分位数（最高买触发点，如0.75=75%分位）
        sell_high_quantile: float，止盈区间上限分位数（越大越高）
        sell_low_quantile: float，止盈区间下限分位数（最低止盈触发点）
        buy_tier_count: int，买入档数
        sell_tier_count: int，止盈档数
        base_share: int，首档买入份额（最高触发点，最浅买入）
        share_increment: int，每档买入份额递增量（越跌份额越大）
        sell_share: int，每档止盈份额
        decimals: 约定净值保留小数位数（广发基金净值只保留2位）

    返回：
        buy_list: [{trigger_net_value, share}, ...] 降序（高净值→低净值，份额递增）
        sell_list: [{trigger_net_value, share}, ...] 升序（低净值→高净值）
    """
    s = pd.Series(net_value_series) if not isinstance(net_value_series, pd.Series) else net_value_series

    buy_high_nv = round(float(s.quantile(buy_high_quantile)), decimals)
    buy_low_nv = round(float(s.quantile(buy_low_quantile)), decimals)
    sell_low_nv = round(float(s.quantile(sell_low_quantile)), decimals)
    sell_high_nv = round(float(s.quantile(sell_high_quantile)), decimals)

    buy_list = []
    sell_list = []

    # 生成买入档：从 buy_high_nv（高）到 buy_low_nv（低），越跌份额越大
    buy_nvs = sorted(np.linspace(buy_high_nv, buy_low_nv, buy_tier_count), reverse=True)
    for i, nv in enumerate(buy_nvs):
        share = base_share + i * share_increment  # i=0（最高触发）→ 最小份额
        buy_list.append({"trigger_net_value": round(float(nv), decimals), "share": share})

    # 生成止盈档：从 sell_low_nv 到 sell_high_nv
    sell_nvs = sorted(np.linspace(sell_low_nv, sell_high_nv, sell_tier_count))
    for i, nv in enumerate(sell_nvs):
        sell_list.append({"trigger_net_value": round(float(nv), decimals), "share": sell_share})

    # buy_list 已降序；sell_list 已升序
    return buy_list, sell_list


class GFNavConversionTrader:
    """
    广发约定净值转换交易器
    管理买入/止盈触发净值的状态机
    """

    def __init__(self, buy_list, sell_list):
        """
        buy_list: 降序排列（高净值先触发）
        sell_list: 升序排列（低净值先触发）
        """
        self.buy_triggers = buy_list
        self.sell_triggers = sell_list
        self.buy_used = [False] * len(buy_list)
        self.conversion_count = 0
        self.buy_count = 0
        self.sell_count = 0

    def check_buys(self, current_nav):
        """
        检查买入触发（天天红B → 目标基金）
        返回: [(trigger_idx, share), ...] 需要执行的买入列表（不修改状态）
        """
        triggers = []
        for i, trigger in enumerate(self.buy_triggers):
            if not self.buy_used[i] and current_nav <= trigger['trigger_net_value']:
                triggers.append((i, trigger['share']))
        return triggers

    def mark_buy_used(self, idx):
        """标记买入档已触发"""
        self.buy_used[idx] = True
        self.buy_count += 1

    def check_sells(self, current_nav, available_shares):
        """
        检查止盈触发（目标基金 → 天天红B）
        返回: [(trigger_idx, share), ...] 需要执行的卖出列表（不修改状态）
        """
        triggers = []
        remaining = available_shares
        for i, trigger in enumerate(self.sell_triggers):
            if remaining <= 0:
                break
            if current_nav >= trigger['trigger_net_value']:
                sell_share = min(trigger['share'], remaining)
                triggers.append((i, sell_share))
                remaining -= sell_share
        return triggers

    def reset_buys_below(self, sell_nav):
        """卖出止盈后，重置该止盈净值以下的所有买入档"""
        for i, trigger in enumerate(self.buy_triggers):
            if trigger['trigger_net_value'] < sell_nav:
                self.buy_used[i] = False

    def get_available_buy_count(self):
        return sum(1 for used in self.buy_used if not used)


def run_strategy(initial_cash: float = 1_000_000,
                 buy_low_quantile: float = 0.15,
                 sell_high_quantile: float = 0.90,
                 buy_tier_count: int = 5,
                 sell_tier_count: int = 3,
                 base_share: int = 50000,
                 share_increment: int = 20000,
                 buy_high_quantile: float = 0.70,
                 sell_low_quantile: float = 0.75,
                 sell_share: int = 50000,
                 money_fund_rate: float = 0.02,
                 start_date: str = None,
                 end_date: str = None,
                 nav_data: pd.DataFrame = None,
                 fund_code: str = FUND_CODE,
                 fund_name: str = FUND_NAME,
                 decimals: int = 2,
                 verbose: bool = True):
    """
    运行广发约定净值转换策略（策略A：动态滚动网格）

    参数：
        initial_cash: 初始资金（存入天天红B货币基金）
        ...
        start_date: 回测起始日（None=自动，跳过60日预热）
        end_date: 回测结束日（None=最新）
        nav_data: 预加载的净值数据（None则自动加载）
        fund_code/fund_name: 目标基金代码/名称
        verbose: 是否打印详情
    """
    # 加载场外基金净值
    if nav_data is None:
        if verbose:
            print("正在加载基金净值数据...")
        nav_data = load_otc_fund_nav(fund_code, fund_name,
                                     start_date='2019-01-01', end_date='2026-12-31',
                                     verbose=verbose)

    if verbose:
        print(f"标的: {fund_code} ({fund_name})")

    # 构建价格数据
    price_dict = {fund_code: nav_data}
    # 基金转换：0手续费、0滑点（广发约定转换免手续费）
    engine = BacktestEngine(initial_cash=initial_cash, commission_rate=0.0, slippage=0.0)
    engine.set_price_data(price_dict)

    # 回测日期 + warmup（自适应）
    all_dates, warmup_navs, end_date = resolve_backtest_dates(
        nav_data, start_date, end_date, warmup_days=60)
    if len(all_dates) < 30:
        if verbose:
            print("警告：回测期数据不足")

    # 生成网格参数（基于warmup分位数，静态）
    buy_list, sell_list = generate_grid_params(
        warmup_navs,
        buy_low_quantile=buy_low_quantile,
        sell_high_quantile=sell_high_quantile,
        buy_tier_count=buy_tier_count,
        sell_tier_count=sell_tier_count,
        base_share=base_share,
        share_increment=share_increment,
        buy_high_quantile=buy_high_quantile,
        sell_low_quantile=sell_low_quantile,
        sell_share=sell_share,
        decimals=decimals
    )

    trader = GFNavConversionTrader(buy_list, sell_list)

    if verbose:
        print(f"\n回测区间: {all_dates[0].strftime('%Y-%m-%d')} ~ {all_dates[-1].strftime('%Y-%m-%d')} ({len(all_dates)}日)")
        print(f"天天红B年化: {money_fund_rate * 100:.1f}%")
        print(f"初始资金: {initial_cash:,.0f} 元")
        print(f"\n====买入策略(天天红B→基金，约定净值≤xxx)====")
        for item in buy_list:
            print(f"  净值≤{item['trigger_net_value']}, 份额: {item['share']}")
        print(f"====止盈策略(基金→天天红B，约定净值≥xxx)====")
        for item in sell_list:
            print(f"  净值≥{item['trigger_net_value']}, 份额: {item['share']}")
        print()

    # 每日利息因子（天天红B货币基金日利息）
    daily_interest = 1 + money_fund_rate / 365

    # 回测主循环
    for idx, date in enumerate(all_dates):
        # 1. 货币基金计息（对现金余额=天天红B持有额）
        engine.cash *= daily_interest

        # 2. 获取当日净值
        current_nav = nav_data.loc[date, 'close']

        # 3. 先检查止盈（净值上涨时先止盈）
        pos = engine.positions.get(fund_code)
        current_shares = pos.shares if pos and pos.shares > 0 else 0
        sell_triggers = trader.check_sells(current_nav, current_shares)
        for trigger_idx, share in sell_triggers:
            if engine.sell(fund_code, date, shares=share):
                trader.sell_count += 1
                trader.conversion_count += 1
                trader.reset_buys_below(sell_list[trigger_idx]['trigger_net_value'])
                if verbose and trader.conversion_count <= 10:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 止盈触发: 净值={current_nav:.4f} ≥ "
                          f"{sell_list[trigger_idx]['trigger_net_value']}, 份额={share}")

        # 4. 再检查买入（净值下跌时买入）
        buy_triggers = trader.check_buys(current_nav)
        for trigger_idx, share in buy_triggers:
            if engine.buy(fund_code, date, shares=share):
                trader.mark_buy_used(trigger_idx)
                trader.conversion_count += 1
                if verbose and trader.conversion_count <= 10:
                    print(f"  [{date.strftime('%Y-%m-%d')}] 买入触发: 净值={current_nav:.4f} ≤ "
                          f"{buy_list[trigger_idx]['trigger_net_value']}, 份额={share}")

        # 5. 记录每日净值
        engine.record_daily_value(date)

    # 结果
    equity_df = engine.get_equity_curve()
    metrics = calculate_metrics(equity_df, initial_cash, name="广发约定净值转换")
    trade_stats = calculate_trade_stats(engine.trades)
    trade_stats['转换次数'] = trader.conversion_count
    trade_stats['买入触发次数'] = trader.buy_count
    trade_stats['止盈触发次数'] = trader.sell_count

    # 买入持有对比
    buy_hold_final = initial_cash / nav_data.loc[all_dates[0], 'close'] * nav_data.loc[all_dates[-1], 'close']
    buy_hold_return = (buy_hold_final / initial_cash - 1) * 100
    trade_stats['买入持有收益率(%)'] = round(buy_hold_return, 2)
    trade_stats['超额收益(%)'] = round(metrics['累计收益率(%)'] - buy_hold_return, 2)

    # 纯货币基金（不转换）对比
    mf_final = initial_cash * (1 + money_fund_rate) ** (len(all_dates) / 365)
    mf_return = (mf_final / initial_cash - 1) * 100
    trade_stats['纯货币基金收益率(%)'] = round(mf_return, 2)

    if verbose:
        print_report(metrics, trade_stats, title="策略6：广发约定净值转换")

    # 保存权益曲线
    output_dir = os.path.dirname(os.path.abspath(__file__))
    equity_df.to_csv(os.path.join(output_dir, 'gf_nav_equity_curve.csv'))

    # 保存广发参数表
    _save_gf_params(buy_list, sell_list, output_dir)

    # 将参数附加到engine上供优化器使用
    engine.gf_buy_list = buy_list
    engine.gf_sell_list = sell_list

    return metrics, trade_stats, equity_df, engine


def _save_gf_params(buy_list, sell_list, output_dir, prefix='gf_nav_'):
    """保存广发约定净值转换参数表（可直接录入广发系统）"""
    lines = []
    lines.append("=" * 50)
    lines.append("       广发约定净值转换 · 参数表")
    lines.append("=" * 50)
    lines.append("")
    lines.append("【买入策略】天天红B → 目标基金（约定净值≤触发值时转换）")
    lines.append(f"{'序号':<6}{'约定净值':<14}{'转换份额':<10}")
    lines.append("-" * 32)
    for i, item in enumerate(buy_list, 1):
        lines.append(f"{i:<6}{item['trigger_net_value']:<14}{item['share']:<10}")
    lines.append("")
    lines.append("【止盈策略】目标基金 → 天天红B（约定净值≥触发值时转换）")
    lines.append(f"{'序号':<6}{'约定净值':<14}{'转换份额':<10}")
    lines.append("-" * 32)
    for i, item in enumerate(sell_list, 1):
        lines.append(f"{i:<6}{item['trigger_net_value']:<14}{item['share']:<10}")
    lines.append("")
    lines.append("=" * 50)

    # 保存文本版
    txt_path = os.path.join(output_dir, prefix + 'params.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    # 保存CSV版（可直接复制到广发系统录入）
    csv_path = os.path.join(output_dir, prefix + 'params.csv')
    rows = []
    for i, item in enumerate(buy_list, 1):
        rows.append({'方向': '买入(天天红B→基金)', '序号': i,
                     '约定净值': item['trigger_net_value'], '转换份额': item['share']})
    for i, item in enumerate(sell_list, 1):
        rows.append({'方向': '止盈(基金→天天红B)', '序号': i,
                     '约定净值': item['trigger_net_value'], '转换份额': item['share']})
    pd.DataFrame(rows).to_csv(csv_path, index=False, encoding='utf-8-sig')

    return lines


if __name__ == '__main__':
    run_strategy()
