"""
广发约定净值转换 · 三策略对比回测
====================================
对比三种网格策略在同一基金（广发中证红利ETF联接C 021400）上的表现：
  A. 动态滚动网格     — 全历史分位数划定固定网格，止盈后滚动重置
  B. 估值分位网格     — 滚动窗口分位数，自适应网格
  C. 高点回撤止盈网格 — 高点回撤买入，成本止盈卖出

用法：
    python run_gf_three_strategies.py

输出：
    1. 终端对比表
    2. gf_three_strategies_comparison.csv — 三策略绩效对比
    3. 各策略参数表（gf_nav_params / gf_valuation_params / gf_drawdown_params）
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.data_loader import load_otc_fund_nav
from strategy6_gf_nav_conversion.strategy import run_strategy as run_a
from strategy6_gf_nav_conversion.strategy_b_valuation import run_strategy as run_b
from strategy6_gf_nav_conversion.strategy_c_drawdown import run_strategy as run_c


FUND_CODE = '021400'
FUND_NAME = '广发中证红利ETF发起式联接C'


def run_three_strategies(initial_cash: float = 1_000_000, end_date: str = '2026-09-04'):
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 10 + "广发约定净值转换 · 三策略对比回测" + " " * 24 + "║")
    print("╚" + "═" * 70 + "╝\n")

    # 预加载净值数据（三策略共用）
    print(f"预加载基金净值数据: {FUND_CODE} {FUND_NAME}")
    nav_data = load_otc_fund_nav(FUND_CODE, FUND_NAME,
                                 start_date='2024-09-01', end_date=end_date,
                                 verbose=False)
    print(f"  数据: {len(nav_data)} 条, {nav_data.index[0].date()} ~ {nav_data.index[-1].date()}")
    print(f"  净值范围: {nav_data['close'].min():.4f} ~ {nav_data['close'].max():.4f}\n")

    results = {}

    # ---- 策略A：动态滚动网格 ----
    print("─" * 70)
    print("  策略A：动态滚动网格（全历史分位数）")
    print("─" * 70)
    metrics_a, stats_a, equity_a, engine_a = run_a(
        initial_cash=initial_cash, nav_data=nav_data, verbose=False)
    results['A_动态滚动网格'] = {'metrics': metrics_a, 'stats': stats_a,
                                 'equity': equity_a, 'engine': engine_a}
    print(f"  累计收益: {metrics_a['累计收益率(%)']:.2f}%  "
          f"年化: {metrics_a['年化收益率(%)']:.2f}%  "
          f"最大回撤: {metrics_a['最大回撤(%)']:.2f}%  "
          f"夏普: {metrics_a.get('夏普比率', 0):.3f}  "
          f"转换: {stats_a['转换次数']}次  "
          f"超额: {stats_a['超额收益(%)']:.2f}%")

    # ---- 策略B：估值分位网格 ----
    print("\n" + "─" * 70)
    print("  策略B：估值分位网格（滚动窗口百分位）")
    print("─" * 70)
    metrics_b, stats_b, equity_b, engine_b = run_b(
        initial_cash=initial_cash, nav_data=nav_data, verbose=False)
    results['B_估值分位网格'] = {'metrics': metrics_b, 'stats': stats_b,
                                 'equity': equity_b, 'engine': engine_b}
    print(f"  累计收益: {metrics_b['累计收益率(%)']:.2f}%  "
          f"年化: {metrics_b['年化收益率(%)']:.2f}%  "
          f"最大回撤: {metrics_b['最大回撤(%)']:.2f}%  "
          f"夏普: {metrics_b.get('夏普比率', 0):.3f}  "
          f"转换: {stats_b['转换次数']}次  "
          f"超额: {stats_b['超额收益(%)']:.2f}%")

    # ---- 策略C：高点回撤止盈网格 ----
    print("\n" + "─" * 70)
    print("  策略C：高点回撤止盈网格（高点回撤买入+成本止盈）")
    print("─" * 70)
    metrics_c, stats_c, equity_c, engine_c = run_c(
        initial_cash=initial_cash, nav_data=nav_data, verbose=False)
    results['C_高点回撤止盈'] = {'metrics': metrics_c, 'stats': stats_c,
                                 'equity': equity_c, 'engine': engine_c}
    print(f"  累计收益: {metrics_c['累计收益率(%)']:.2f}%  "
          f"年化: {metrics_c['年化收益率(%)']:.2f}%  "
          f"最大回撤: {metrics_c['最大回撤(%)']:.2f}%  "
          f"夏普: {metrics_c.get('夏普比率', 0):.3f}  "
          f"转换: {stats_c['转换次数']}次  "
          f"超额: {stats_c['超额收益(%)']:.2f}%")

    # ---- 汇总对比表 ----
    print("\n\n" + "=" * 70)
    print("  三策略绩效对比汇总")
    print("=" * 70)
    header = f"{'策略':<16}{'累计收益%':>10}{'年化%':>8}{'最大回撤%':>10}{'夏普':>8}{'转换次数':>10}{'超额%':>10}{'胜率%':>8}"
    print(header)
    print("-" * 70)
    for name, r in results.items():
        m = r['metrics']
        s = r['stats']
        print(f"{name:<16}{m['累计收益率(%)']:>10.2f}{m['年化收益率(%)']:>8.2f}"
              f"{m['最大回撤(%)']:>10.2f}{m.get('夏普比率', 0):>8.3f}"
              f"{s['转换次数']:>10}{s['超额收益(%)']:>10.2f}"
              f"{s.get('胜率(%)', 0):>8.1f}")
    print("=" * 70)

    # ---- 输出CSV对比表 ----
    comp_rows = []
    for name, r in results.items():
        m = r['metrics']
        s = r['stats']
        comp_rows.append({
            '策略': name,
            '累计收益率(%)': m['累计收益率(%)'],
            '年化收益率(%)': m['年化收益率(%)'],
            '最大回撤(%)': m['最大回撤(%)'],
            '夏普比率': m.get('夏普比率', 0),
            '卡玛比率': m.get('卡玛比率', 0),
            '波动率(年化%)': m.get('波动率(年化%)', 0),
            '转换次数': s['转换次数'],
            '买入次数': s['买入触发次数'],
            '止盈次数': s['止盈触发次数'],
            '胜率(%)': s.get('胜率(%)', 0),
            '超额收益(%)': s['超额收益(%)'],
            '买入持有收益率(%)': s['买入持有收益率(%)'],
            '纯货币基金收益率(%)': s['纯货币基金收益率(%)'],
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'gf_three_strategies_comparison.csv')
    comp_df.to_csv(comp_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 三策略对比表: {comp_path}")

    # ---- 输出各策略参数表摘要 ----
    print("\n" + "=" * 70)
    print("  各策略最新参数表（可录入广发系统）")
    print("=" * 70)
    for name, r in results.items():
        engine = r['engine']
        buy_list = getattr(engine, 'gf_buy_list', None)
        sell_list = getattr(engine, 'gf_sell_list', None)
        if not buy_list or not sell_list:
            continue
        print(f"\n【{name}】")
        print("  买入（天天红B→基金，净值≤触发值）:")
        for i, item in enumerate(buy_list, 1):
            print(f"    {i}. 净值≤{item['trigger_net_value']:<8} 份额={item['share']}")
        print("  止盈（基金→天天红B，净值≥触发值）:")
        for i, item in enumerate(sell_list, 1):
            print(f"    {i}. 净值≥{item['trigger_net_value']:<8} 份额={item['share']}")

    print("\n" + "─" * 70)
    print("  ⚠  以上为历史回测结果，不构成投资建议。")
    print("     实际录入广发系统前请结合当前市场环境判断。")
    print("─" * 70)

    return results


if __name__ == '__main__':
    run_three_strategies()
