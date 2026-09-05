"""
广发约定净值转换 · 参数遍历优化器
====================================
遍历多组参数组合 → 回测对比 → 输出可直接录入广发的完整参数表

用法：
    python run_gf_param_optimization.py

输出：
    1. gf_param_comparison.csv — 所有参数组合的回测绩效对比
    2. gf_nav_best_params.txt  — 最优参数表（广发系统可直接录入）
    3. gf_nav_best_params.csv  — 最优参数表（CSV版）
    4. gf_nav_best_equity.csv  — 最优组合权益曲线
"""
import os
import sys
import itertools
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.metrics import compare_strategies
from common.data_loader import load_otc_fund_nav
from strategy6_gf_nav_conversion.strategy import run_strategy, _save_gf_params


# ============================================================
# 参数遍历网格（可根据需要调整）
# ============================================================
PARAM_GRID = {
    'buy_low_quantile':   [0.05, 0.15],                # 买入下限分位数
    'buy_high_quantile':  [0.70, 0.80],                # 买入上限分位数
    'sell_low_quantile':  [0.75, 0.85],                # 止盈下限分位数
    'sell_high_quantile': [0.90, 0.95],                # 止盈上限分位数
    'buy_tier_count':     [3, 4, 5],                    # 买入档数
    'sell_tier_count':    [2, 3],                       # 止盈档数
    'base_share':         [30000, 50000, 80000],        # 首档买入份额（适配净值~1.1）
    'share_increment':    [10000, 20000],               # 每档递增份额
}

# 优化指标：累计收益率（网格策略最直观的收益指标）
OPTIMIZE_METRIC = '累计收益率(%)'


def _combo_name(params):
    """生成参数组合的可读名称"""
    return (f"买Q{params['buy_low_quantile']:.2f}-{params['buy_high_quantile']:.2f}"
            f"_卖Q{params['sell_low_quantile']:.2f}-{params['sell_high_quantile']:.2f}"
            f"_{params['buy_tier_count']}买{params['sell_tier_count']}卖"
            f"_基{params['base_share']}_增{params['share_increment']}")


def run_param_optimization(initial_cash: float = 1_000_000,
                           optimize_metric: str = OPTIMIZE_METRIC,
                           param_grid: dict = None,
                           top_n: int = 10):
    """
    参数遍历优化主函数

    流程：
        1. 预加载净值数据（所有组合共用）
        2. 遍历所有参数组合，逐个回测
        3. 按优化指标排序，输出对比表
        4. 取最优组合，输出可直接录入广发的参数表
    """
    grid = param_grid or PARAM_GRID
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))

    print("╔" + "═" * 70 + "╗")
    print("║" + " " * 12 + "广发约定净值转换 · 参数遍历优化器" + " " * 22 + "║")
    print("╚" + "═" * 70 + "╝\n")
    print(f"参数网格: {len(combos)} 组组合")
    print(f"优化指标: {optimize_metric}")
    print(f"初始资金: {initial_cash:,.0f} 元\n")

    # 预加载净值数据
    print("预加载净值数据...")
    nav_data = load_otc_fund_nav('021400', '广发中证红利ETF发起式联接C',
                                 start_date='2024-09-01', end_date='2026-09-04',
                                 verbose=False)
    print(f"数据: {len(nav_data)} 条, {nav_data.index[0].date()} ~ {nav_data.index[-1].date()}\n")

    results = []
    best_result = None
    best_metric_val = -float('inf')
    skipped = 0

    t0 = time.time()
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))

        # 跳过无效组合：买入上限必须 > 买入下限，止盈上限必须 > 止盈下限
        if params.get('buy_high_quantile', 1) <= params.get('buy_low_quantile', 0):
            skipped += 1
            continue
        if params.get('sell_high_quantile', 1) <= params.get('sell_low_quantile', 0):
            skipped += 1
            continue

        name = _combo_name(params)

        try:
            metrics, trade_stats, equity_df, engine = run_strategy(
                initial_cash=initial_cash,
                verbose=False,
                nav_data=nav_data,
                **params
            )
            results.append((name, metrics, trade_stats))

            # 追踪最优
            metric_val = metrics.get(optimize_metric, 0) or 0
            if metric_val > best_metric_val:
                best_metric_val = metric_val
                best_result = {
                    'name': name,
                    'params': params,
                    'metrics': metrics,
                    'trade_stats': trade_stats,
                    'equity_df': equity_df,
                    'engine': engine,
                }

        except Exception:
            skipped += 1

        # 进度播报
        if (i + 1) % 50 == 0 or (i + 1) == len(combos):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(combos) - i - 1)
            print(f"  进度: {i + 1}/{len(combos)}  耗时: {elapsed:.1f}s  剩余: ~{eta:.0f}s")

    elapsed_total = time.time() - t0
    print(f"\n遍历完成: {len(results)}/{len(combos)} 组成功 (跳过{skipped}) 耗时 {elapsed_total:.1f}s")

    if not results:
        print("❌ 无有效结果")
        return

    # ---- 对比报告 ----
    print("\n" + "─" * 80)
    print(f"  Top {min(top_n, len(results))} 参数组合（按{optimize_metric}排序）:")
    print("─" * 80)

    compare_df = compare_strategies(results)
    # 按优化指标排序
    if optimize_metric in compare_df.columns:
        compare_df = compare_df.sort_values(optimize_metric, ascending=False).reset_index(drop=True)

    # 显示Top N
    display_cols = ['策略', '年化收益率(%)', '累计收益率(%)', '最大回撤(%)',
                    optimize_metric, '夏普比率', '转换次数', '超额收益(%)']
    cols = [c for c in display_cols if c in compare_df.columns]
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.float_format', lambda v: f"{v:>10.2f}")
    print(compare_df[cols].head(top_n).to_string(index=False))

    # 保存完整对比表
    out_dir = os.path.dirname(os.path.abspath(__file__))
    comparison_path = os.path.join(out_dir, 'gf_param_comparison.csv')
    compare_df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 参数对比表: {comparison_path}")

    # ---- 最优参数表 ----
    if best_result:
        print("\n" + "╔" + "═" * 70 + "╗")
        print("║" + " " * 16 + "最优参数组合" + " " * 38 + "║")
        print("╚" + "═" * 70 + "╝\n")

        p = best_result['params']
        print(f"  参数: {p}")
        m = best_result['metrics']
        ts = best_result['trade_stats']
        print(f"  年化收益率: {m.get('年化收益率(%)', 0):.2f}%")
        print(f"  最大回撤:   {m.get('最大回撤(%)', 0):.2f}%")
        print(f"  {optimize_metric}: {m.get(optimize_metric, 0)}")
        print(f"  夏普比率:   {m.get('夏普比率', 0)}")
        print(f"  转换次数:   {ts.get('转换次数', 0)}")
        print(f"  超额收益:   {ts.get('超额收益(%)', 0):.2f}%")
        print(f"  纯货币收益: {ts.get('纯货币基金收益率(%)', 0):.2f}%")

        # 输出广发参数表
        engine = best_result['engine']
        buy_list = engine.gf_buy_list
        sell_list = engine.gf_sell_list

        lines = _save_gf_params(buy_list, sell_list, out_dir)

        # 重命名为 best
        best_txt = os.path.join(out_dir, 'gf_nav_best_params.txt')
        best_csv = os.path.join(out_dir, 'gf_nav_best_params.csv')
        # _save_gf_params 已写入 gf_nav_params.txt/csv，复制到 best 版本
        import shutil
        shutil.copy(os.path.join(out_dir, 'gf_nav_params.txt'), best_txt)
        shutil.copy(os.path.join(out_dir, 'gf_nav_params.csv'), best_csv)

        # 在文本版前追加最优参数信息
        with open(best_txt, 'r', encoding='utf-8') as f:
            param_text = f.read()
        header = (
            f"最优参数组合信息\n"
            f"{'─' * 50}\n"
            f"参数: {p}\n"
            f"年化收益率: {m.get('年化收益率(%)', 0):.2f}%\n"
            f"最大回撤:   {m.get('最大回撤(%)', 0):.2f}%\n"
            f"卡玛比率:   {m.get('卡玛比率', 0)}\n"
            f"夏普比率:   {m.get('夏普比率', 0)}\n"
            f"转换次数:   {ts.get('转换次数', 0)}\n"
            f"超额收益:   {ts.get('超额收益(%)', 0):.2f}%\n"
            f"{'─' * 50}\n\n"
        )
        with open(best_txt, 'w', encoding='utf-8') as f:
            f.write(header + param_text)

        print(f"\n💾 最优参数表(TXT): {best_txt}")
        print(f"💾 最优参数表(CSV): {best_csv}")

        # 保存最优权益曲线
        best_equity_path = os.path.join(out_dir, 'gf_nav_best_equity.csv')
        best_result['equity_df'].to_csv(best_equity_path, encoding='utf-8-sig')
        print(f"💾 最优权益曲线: {best_equity_path}")

        # 打印参数表
        print("\n" + "\n".join(lines))

    print("\n─" * 80)
    print("  ⚠  以上为历史回测结果，不构成投资建议。")
    print("     实际录入广发系统前请结合当前市场环境判断。")
    print("─" * 80)

    return best_result


if __name__ == '__main__':
    run_param_optimization()
