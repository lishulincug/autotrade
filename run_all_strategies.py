"""
命令行版：运行所有5个策略，仅输出回测报告（不启动Web服务）。
如需网页版K线+买卖点+对比看板，请运行：

    python run_and_serve.py
"""
import os
import sys
import time
import traceback
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.metrics import compare_strategies


def run_all_strategies(initial_cash: float = 1_000_000):
    strategies = [
        ('策略1：RRG多因子行业轮动',       'strategy1_rrg_rotation.strategy'),
        ('策略2：双均线动量轮动+空仓风控', 'strategy2_dual_ma_momentum.strategy'),
        ('策略3：网格交易',               'strategy3_grid_trading.strategy'),
        ('策略4：宏观因子择时',           'strategy4_macro_timing.strategy'),
        ('策略5：指数增强多因子选股',     'strategy5_index_enhancement.strategy'),
        ('策略6：广发约定净值转换',       'strategy6_gf_nav_conversion.strategy'),
    ]

    results = []
    all_equities = {}

    for i, (name, module_path) in enumerate(strategies):
        print("\n" + "=" * 70)
        print(f"  正在运行 {name}  ({i + 1}/{len(strategies)})")
        print("=" * 70)
        t0 = time.time()
        try:
            mod = __import__(module_path, fromlist=['run_strategy'])
            metrics, trade_stats, equity_df, engine = mod.run_strategy(initial_cash=initial_cash)
            results.append((name, metrics, trade_stats))
            if equity_df is not None and len(equity_df) > 0:
                all_equities[name] = equity_df['equity']
        except Exception as e:
            print(f"  ❌ 失败: {e}")
            traceback.print_exc()
        print(f"  ⏱  耗时: {time.time() - t0:.2f} 秒")

    # 对比报告
    print("\n" + "╔" + "═" * 80 + "╗")
    print("║" + " " * 20 + "综合回测绩效对比报告" + " " * 34 + "║")
    print("╚" + "═" * 80 + "╝\n")

    compare_df = compare_strategies(results)
    if not compare_df.empty:
        key_cols = ['策略','年化收益率(%)','累计收益率(%)','最大回撤(%)',
                    '夏普比率','卡玛比率','胜率(%)','盈亏比','波动率(年化%)',
                    '总交易次数','期末总资产(元)']
        cols = [c for c in key_cols if c in compare_df.columns]
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        pd.set_option('display.float_format', lambda v: f"{v:>12.2f}")
        print(compare_df[cols].to_string(index=False))

        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_comparison.csv')
        compare_df.to_csv(out, index=False, encoding='utf-8-sig')
        print(f"\n💾 详细报告已保存: {out}")

    if all_equities:
        ec = pd.DataFrame(all_equities).ffill().fillna(1.0)
        ep = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'equity_curves_comparison.csv')
        ec.to_csv(ep, encoding='utf-8-sig')
        print(f"💾 权益曲线数据: {ep}")

    print("\n─" * 80)
    print("  📊 2026年8月市场适配组合：")
    print("     🟢 核心仓位60%  → RRG 多因子行业轮动")
    print("     🟡 底仓增强30%  → 宽基 ETF 网格交易")
    print("     🔴 防守仓位10%  → 宏观因子择时（国债/黄金）")
    print("\n  ⚠  以上为策略原理演示，不构成投资建议。")
    print("     🌐 如需交互式K线+买卖点看板，请运行: python run_and_serve.py")
    print("─" * 80)

    return results


if __name__ == '__main__':
    print("╔" + "═" * 80 + "╗")
    print("║" + " " * 25 + "ETF 量化交易策略 · 命令行回测版" + " " * 25 + "║")
    print("║" + " " * 12 + "6大策略：RRG / 双均线 / 网格 / 宏观择时 / 指数增强 / 广发净值转换" + " " * 4 + "║")
    print("╚" + "═" * 80 + "╝\n")
    run_all_strategies()
