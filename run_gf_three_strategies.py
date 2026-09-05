"""
广发约定净值转换 · 多策略对比回测
====================================
对比约定净值网格策略在同一基金（广发中证红利ETF联接C 021400）上的表现：
  A. 动态滚动网格
  B. 估值分位网格
  C. 高点回撤止盈网格
  D. 底仓锁利+浮动网格
  E. 净值分位自适应双区
  G. 阶梯止盈+移动止损映射
（F 多标的轮动请运行 run_gf_rotation.py）

用法：
    python run_gf_three_strategies.py

输出：
    1. 终端对比表
    2. gf_three_strategies_comparison.csv
    3. 各策略参数表
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.data_loader import load_otc_fund_nav
from strategy6_gf_nav_conversion.strategy import run_strategy as run_a
from strategy6_gf_nav_conversion.strategy_b_valuation import run_strategy as run_b
from strategy6_gf_nav_conversion.strategy_c_drawdown import run_strategy as run_c
from strategy6_gf_nav_conversion.strategy_d_core_float import run_strategy as run_d
from strategy6_gf_nav_conversion.strategy_e_dual_zone import run_strategy as run_e
from strategy6_gf_nav_conversion.strategy_g_ladder_trail import run_strategy as run_g


FUND_CODE = '021400'
FUND_NAME = '广发中证红利ETF发起式联接C'

STRATEGIES = [
    ('A_动态滚动网格', run_a),
    ('B_估值分位网格', run_b),
    ('C_高点回撤止盈', run_c),
    ('D_底仓锁利+浮动', run_d),
    ('E_分位自适应双区', run_e),
    ('G_阶梯止盈映射', run_g),
]


def run_three_strategies(initial_cash: float = 1_000_000, end_date: str = '2026-09-04'):
    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 10 + "广发约定净值转换 · 多策略对比回测" + " " * 24 + "║")
    print("╚" + "═" * 70 + "╝\n")

    print(f"预加载基金净值数据: {FUND_CODE} {FUND_NAME}")
    nav_data = load_otc_fund_nav(FUND_CODE, FUND_NAME,
                                 start_date='2024-09-01', end_date=end_date,
                                 verbose=False)
    print(f"  数据: {len(nav_data)} 条, {nav_data.index[0].date()} ~ {nav_data.index[-1].date()}")
    print(f"  净值范围: {nav_data['close'].min():.4f} ~ {nav_data['close'].max():.4f}\n")

    results = {}
    for name, fn in STRATEGIES:
        print("─" * 70)
        print(f"  {name}")
        print("─" * 70)
        metrics, stats, equity, engine = fn(
            initial_cash=initial_cash, nav_data=nav_data, verbose=False)
        results[name] = {'metrics': metrics, 'stats': stats,
                         'equity': equity, 'engine': engine}
        print(f"  累计收益: {metrics['累计收益率(%)']:.2f}%  "
              f"年化: {metrics['年化收益率(%)']:.2f}%  "
              f"最大回撤: {metrics['最大回撤(%)']:.2f}%  "
              f"夏普: {metrics.get('夏普比率', 0):.3f}  "
              f"转换: {stats['转换次数']}次  "
              f"超额: {stats['超额收益(%)']:.2f}%")

    print("\n\n" + "=" * 70)
    print("  多策略绩效对比汇总")
    print("=" * 70)
    header = f"{'策略':<18}{'累计收益%':>10}{'年化%':>8}{'最大回撤%':>10}{'夏普':>8}{'转换次数':>10}{'超额%':>10}{'胜率%':>8}"
    print(header)
    print("-" * 70)
    for name, r in results.items():
        m = r['metrics']
        s = r['stats']
        print(f"{name:<18}{m['累计收益率(%)']:>10.2f}{m['年化收益率(%)']:>8.2f}"
              f"{m['最大回撤(%)']:>10.2f}{m.get('夏普比率', 0):>8.3f}"
              f"{s['转换次数']:>10}{s['超额收益(%)']:>10.2f}"
              f"{s.get('胜率(%)', 0):>8.1f}")
    print("=" * 70)

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
            '纯货币基金收益率(%)': s.get('纯货币基金收益率(%)', 0),
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'gf_three_strategies_comparison.csv')
    comp_df.to_csv(comp_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 多策略对比表: {comp_path}")

    print("\n" + "=" * 70)
    print("  各策略最新参数表（可录入广发系统）")
    print("=" * 70)
    for name, r in results.items():
        engine = r['engine']
        buy_list = getattr(engine, 'gf_buy_list', None) or []
        sell_list = getattr(engine, 'gf_sell_list', None) or []
        if not buy_list and not sell_list:
            continue
        print(f"\n【{name}】")
        if buy_list:
            print("  买入（天天红B→基金，净值≤触发值）:")
            for i, item in enumerate(buy_list, 1):
                role = f" [{item.get('role')}]" if item.get('role') else ""
                print(f"    {i}. 净值≤{item['trigger_net_value']:<8} 份额={item['share']}{role}")
        if sell_list:
            print("  止盈（基金→天天红B，净值≥触发值）:")
            for i, item in enumerate(sell_list, 1):
                role = f" [{item.get('role')}]" if item.get('role') else ""
                print(f"    {i}. 净值≥{item['trigger_net_value']:<8} 份额={item['share']}{role}")

    print("\n" + "─" * 70)
    print("  ⚠  以上为历史回测结果，不构成投资建议。")
    print("     实际录入广发系统前请结合当前市场环境判断。")
    print("─" * 70)

    return results


if __name__ == '__main__':
    run_three_strategies()
