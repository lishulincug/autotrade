# -*- coding: utf-8 -*-
"""
策略7 · 双底确认 + 自适应阶梯抄底 —— 批量筛选 / 条件单生成 / 回测 / 可视化
=========================================================================

流程：
  1. 读取 strategy7_dual_bottom_adaptive/fund_pool.txt 基金池
  2. 拉取场外基金历史净值（东方财富公开接口，本地缓存，失败回退模拟并标记）
  3. 双底确认（价格底+估值底）+ 四维量化排雷 → 筛选结果表
  4. 对入选/观察基金生成波动率自适应阶梯条件单（买/止盈/终止/止损/移动止盈）
  5. 近3年事件驱动回测（月度重检、日度触发条件单）
  6. 输出交互式 HTML 仪表盘 + CSV

用法：
    python run_dual_bottom.py

输出：
    dual_bottom_dashboard.html                       — 可视化仪表盘
    strategy7_dual_bottom_adaptive/screening_results.csv   — 筛选结果
    strategy7_dual_bottom_adaptive/conditional_orders.csv  — 条件单清单
    strategy7_dual_bottom_adaptive/ladder_backtest.csv     — 回测明细
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
from datetime import date

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.data_loader import load_otc_fund_nav, fetch_real_etf_kline
from strategy7_dual_bottom_adaptive.screening import screen_pool
from strategy7_dual_bottom_adaptive.ladder import generate_adaptive_ladder
from strategy7_dual_bottom_adaptive.backtest import backtest_fund, aggregate_portfolio

import plotly.graph_objects as go
import plotly.io as pio

# ---------------- 配置 ----------------
STRATEGY_DIR = os.path.join(ROOT, 'strategy7_dual_bottom_adaptive')
FUND_POOL_FILE = os.path.join(STRATEGY_DIR, 'fund_pool.txt')
OUT_HTML = os.path.join(ROOT, 'dual_bottom_dashboard.html')
OUT_SCREEN_CSV = os.path.join(STRATEGY_DIR, 'screening_results.csv')
OUT_ORDERS_CSV = os.path.join(STRATEGY_DIR, 'conditional_orders.csv')
OUT_BT_CSV = os.path.join(STRATEGY_DIR, 'ladder_backtest.csv')
OUT_EQUITY_CSV = os.path.join(STRATEGY_DIR, 'ladder_backtest_equity.csv')

DATA_START = '2019-01-01'
DATA_END = '2026-12-31'
CAPITAL_PER_FUND = 100_000
BT_YEARS = 3

RATING_COLOR = {
    '★★★ 重点抄底': '#1a9850',
    '★★ 可挂条件单': '#66bd63',
    '★ 观察名单': '#fdae61',
    '✗ 排雷剔除': '#d73027',
    '— 未到底部': '#999999',
    '— 数据不足': '#bbbbbb',
    '— 模拟数据': '#bbbbbb',
}


def compute_start_date(years):
    today = date.today()
    try:
        return today.replace(year=today.year - years).isoformat()
    except ValueError:
        return today.replace(year=today.year - years, day=28).isoformat()


def parse_fund_list(path):
    """解析基金清单（每行：名称\\t代码）"""
    funds = []
    if not os.path.exists(path):
        return funds
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if '\t' in line:
                name, code = line.split('\t', 1)
            else:
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                name, code = parts
            name = name.strip().replace('‑', '-').replace('（', '(').replace('）', ')')
            code = code.strip()
            funds.append((code, name))
    return funds


def load_pool_data(fund_list, verbose=True):
    """加载基金池净值数据"""
    funds = {}
    for i, (code, name) in enumerate(fund_list):
        df = load_otc_fund_nav(code, name, DATA_START, DATA_END, verbose=False)
        source = df.attrs.get('nav_source', 'real')
        if len(df) > 60:
            funds[code] = {'name': name, 'nav': df, 'source': source}
            if verbose:
                tag = '真实' if source == 'real' else '模拟(未取到真实数据)'
                print(f'  [{tag}] {code} {name}: {len(df)}条 '
                      f'{df.index[0].date()}~{df.index[-1].date()}')
        else:
            print(f'  [跳过] {code} {name}: 数据不足')
    return funds


# =====================================================================
# 可视化
# =====================================================================

def fig_scatter(screen_df):
    """分位 × 回撤 气泡图（气泡大小=底部评分，颜色=评级）"""
    fig = go.Figure()
    for rating, color in RATING_COLOR.items():
        sub = screen_df[screen_df['rating'] == rating]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(
            x=sub['pct_3y'] * 100, y=sub['dd_from_peak'] * 100,
            mode='markers+text',
            text=sub['name'].str.slice(0, 8),
            textposition='top center', textfont_size=9,
            marker=dict(size=sub['bottom_score'].clip(lower=8) * 0.55 + 8,
                        color=color, line=dict(width=1, color='#333'), opacity=0.85),
            name=rating,
            customdata=np.stack([sub['code'], sub['bottom_score'],
                                 sub['ann_vol'] * 100, sub['risk_reasons'].apply(
                                     lambda x: '；'.join(x) if x else '-')], axis=-1),
            hovertemplate='<b>%{text}</b> (%{customdata[0]})<br>'
                          '3年分位: %{x:.1f}%<br>高点回撤: %{y:.1f}%<br>'
                          '底部评分: %{customdata[1]}<br>年化波动: %{customdata[2]:.1f}%<br>'
                          '排雷: %{customdata[3]}<extra></extra>',
        ))
    fig.add_vline(x=15, line_dash='dot', line_color='#1a9850',
                  annotation_text='价格底15%', annotation_position='top')
    fig.add_vline(x=30, line_dash='dot', line_color='#fdae61',
                  annotation_text='观察线30%', annotation_position='top')
    fig.update_layout(
        title='双底定位图：3年净值分位（越低越便宜）× 高点回撤（越深风险释放越充分）',
        xaxis_title='3年净值分位 (%)', yaxis_title='距3年高点回撤 (%)',
        height=460, template='plotly_white', hovermode='closest',
        legend=dict(orientation='h', y=-0.18),
    )
    return fig


def fig_percentile_bar(screen_df):
    """各基金3年分位条形图"""
    df = screen_df.sort_values('pct_3y')
    colors = [RATING_COLOR.get(r, '#999') for r in df['rating']]
    fig = go.Figure(go.Bar(
        x=df['pct_3y'] * 100, y=df['name'].str.slice(0, 14),
        orientation='h', marker_color=colors,
        customdata=np.stack([df['rating'], df['pct_long'] * 100,
                             df['bottom_score']], axis=-1),
        hovertemplate='<b>%{y}</b><br>3年分位: %{x:.1f}%<br>'
                      '评级: %{customdata[0]}<br>长周期分位: %{customdata[1]:.1f}%<br>'
                      '底部评分: %{customdata[2]}<extra></extra>',
    ))
    fig.add_vline(x=15, line_dash='dot', line_color='#1a9850')
    fig.add_vline(x=30, line_dash='dot', line_color='#fdae61')
    fig.update_layout(
        title='基金池3年净值分位排名（越靠左越接近历史底部）',
        xaxis_title='3年净值分位 (%)', height=480, template='plotly_white',
        margin=dict(l=180),
    )
    return fig


def fig_ladder_chart(code, name, nav_series, plan, bt_result):
    """单只基金：净值走势 + 条件单档位线 + 回测买卖点"""
    s = nav_series.dropna().astype(float)
    s = s[s.index >= pd.Timestamp(compute_start_date(BT_YEARS))]
    if len(s) < 30:
        s = nav_series.dropna().astype(float).iloc[-500:]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=s.index, y=s.values, mode='lines',
                             name='单位净值', line=dict(color='#2c6fbb', width=1.6),
                             hovertemplate='%{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>'))

    if plan is not None:
        # 买入档（绿色系，越深越浓）
        greens = ['#a6d96a', '#66bd63', '#1a9850', '#006d2c', '#00441b']
        for t in plan.buy_tiers:
            fig.add_hline(y=t.trigger_nav, line=dict(color=greens[min(t.idx, 4)], width=1.4, dash='solid'),
                          annotation_text=f'买{t.idx+1} {t.trigger_nav:.3f} ({abs(t.drop_pct):.0%})',
                          annotation_position='right', annotation_font_size=9,
                          annotation_font_color=greens[min(t.idx, 4)])
        # 止盈档（红色虚线）
        reds = ['#f4a582', '#ef6548', '#d7301f']
        for t in plan.tp_tiers:
            fig.add_hline(y=t.trigger_nav, line=dict(color=reds[min(t.idx, 2)], width=1.2, dash='dash'),
                          annotation_text=f'盈{t.idx+1} {t.trigger_nav:.3f} (+{t.gain_pct:.0%})',
                          annotation_position='left', annotation_font_size=9,
                          annotation_font_color=reds[min(t.idx, 2)])
        # 终止线 / 硬止损
        fig.add_hline(y=plan.termination_nav, line=dict(color='#fdae61', width=1.5, dash='dot'),
                      annotation_text=f'终止线 {plan.termination_nav:.3f}',
                      annotation_position='right', annotation_font_size=9, annotation_font_color='#e08214')
        fig.add_hline(y=plan.hard_stop_nav, line=dict(color='#7f0000', width=1.5, dash='dot'),
                      annotation_text=f'硬止损 {plan.hard_stop_nav:.3f}',
                      annotation_position='left', annotation_font_size=9, annotation_font_color='#7f0000')
        # 当前净值
        fig.add_trace(go.Scatter(x=[s.index[-1]], y=[s.iloc[-1]], mode='markers',
                                 marker=dict(size=11, color='#2c6fbb', symbol='diamond',
                                             line=dict(width=1.5, color='white')),
                                 name='当前净值', hovertemplate='当前净值: %{y:.4f}<extra></extra>'))

    # 回测买卖点
    if bt_result is not None and bt_result.trades:
        bt_dates = [t.date for t in bt_result.trades]
        bt_prices = [t.price for t in bt_result.trades]
        bt_buy = [t.action == 'buy' for t in bt_result.trades]
        fig.add_trace(go.Scatter(
            x=[d for d, b in zip(bt_dates, bt_buy) if b],
            y=[p for p, b in zip(bt_prices, bt_buy) if b],
            mode='markers', name='回测买入',
            marker=dict(size=9, color='#1a9850', symbol='triangle-up',
                        line=dict(width=1, color='white')),
            hovertemplate='买入 %{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>'))
        fig.add_trace(go.Scatter(
            x=[d for d, b in zip(bt_dates, bt_buy) if not b],
            y=[p for p, b in zip(bt_prices, bt_buy) if not b],
            mode='markers', name='回测卖出',
            marker=dict(size=9, color='#d73027', symbol='triangle-down',
                        line=dict(width=1, color='white')),
            hovertemplate='卖出 %{x|%Y-%m-%d}<br>净值: %{y:.4f}<extra></extra>'))

    title = f'{name}（{code}）'
    if plan is not None:
        title += f' — {plan.vol_regime} 档距{plan.spacing:.0%} / {plan.n_tiers}档'
    fig.update_layout(title=title, height=360, template='plotly_white',
                      margin=dict(l=60, r=60, t=40, b=30),
                      showlegend=True, legend=dict(orientation='h', y=-0.12),
                      hovermode='x unified')
    return fig


def fig_equity(portfolio, benchmark):
    """组合权益曲线 vs 沪深300"""
    fig = go.Figure()
    if portfolio is not None and len(portfolio):
        fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio['nav'],
                                 mode='lines', name='双底抄底等权组合',
                                 line=dict(color='#1a9850', width=2.2),
                                 hovertemplate='%{x|%Y-%m-%d}<br>组合净值: %{y:.3f}<extra></extra>'))
    if benchmark is not None and len(benchmark):
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark.values,
                                 mode='lines', name='沪深300ETF(510300)',
                                 line=dict(color='#999999', width=1.5, dash='dash'),
                                 hovertemplate='%{x|%Y-%m-%d}<br>基准净值: %{y:.3f}<extra></extra>'))
    fig.add_hline(y=1.0, line=dict(color='#ccc', width=1))
    fig.update_layout(title='近3年回测：双底确认+自适应阶梯组合 vs 沪深300',
                      xaxis_title='', yaxis_title='归一化净值',
                      height=420, template='plotly_white', hovermode='x unified',
                      legend=dict(orientation='h', y=-0.15))
    return fig


def _table(headers, rows, table_id, sortable=True):
    """生成可排序 HTML 表格"""
    th = ''.join(
        f'<th onclick="sortTable(\'{table_id}\', {i})" style="cursor:pointer">{h}▾</th>'
        if sortable else f'<th>{h}</th>'
        for i, h in enumerate(headers))
    trs = []
    for row in rows:
        tds = ''.join(f'<td>{c}</td>' for c in row)
        trs.append(f'<tr>{tds}</tr>')
    return (f'<table id="{table_id}" class="data-table"><thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(trs)}</tbody></table>')


def build_dashboard(screen_df, plans, orders_df, bt_results, portfolio,
                    benchmark, out_path):
    """生成单文件 HTML 仪表盘"""
    n_total = len(screen_df)
    n_pass = int(screen_df['risk_pass'].sum())
    n_confirmed = int(screen_df['bottom_status'].eq('confirmed').sum())
    n_watch = int(screen_df['bottom_status'].eq('watch').sum())
    n_risk_fail = int((~screen_df['risk_pass']).sum())
    n_real = int(screen_df['data_source'].eq('real').sum())
    avg_pct = screen_df['pct_3y'].mean() * 100

    # ---- 筛选结果表 ----
    rows = []
    for _, r in screen_df.iterrows():
        color = RATING_COLOR.get(r['rating'], '#666')
        reasons = '<br>'.join(r['risk_reasons']) if r['risk_reasons'] else '-'
        rows.append([
            f'<b>{r["name"]}</b><br><span class="code">{r["code"]}</span>',
            f'<span class="badge" style="background:{color}">{r["rating"]}</span>',
            f'{r["bottom_score"]:.0f}',
            f'{r["pct_3y"]*100:.1f}%',
            f'{r["pct_long"]*100:.1f}%',
            f'{r["dd_from_peak"]*100:.1f}%',
            f'{r["max_dd_3y"]*100:.1f}%',
            f'{r["dev_ma250"]*100:.1f}%',
            f'{r["ann_vol"]*100:.1f}%',
            f'{r["newlow_252"]*100:.0f}%',
            f'{r["trend_2y"]*100:.1f}%',
            '✓' if r['stabilized'] else '✗',
            f'{r["risk_score"]}',
            f'{r["risk_fail_dims"]}<br><span class="reason">{reasons}</span>',
            r['data_source'],
        ])
    screen_table = _table(
        ['基金', '评级', '底部<br>评分', '3年<br>分位', '长周期<br>分位', '高点<br>回撤',
         '3年最大<br>回撤', '年线<br>偏离', '年化<br>波动', '近1年<br>新低密度',
         '近2年<br>年化', '企稳', '风险<br>分', '排雷明细', '数据'],
        rows, 'screenTable')

    # ---- 条件单表 ----
    order_rows = []
    for _, r in orders_df.iterrows():
        order_rows.append([
            f'{r["基金名称"]}<br><span class="code">{r["基金代码"]}</span>',
            r['波动档位'], f'{r["档距"]*100:.0f}%', r['条件单类型'],
            f'<b>{r["触发净值"]}</b>', r['较现价幅度'], r['资金比例'],
            r['累计投入'], r['说明'],
        ])
    orders_table = _table(
        ['基金', '波动档', '档距', '条件单', '触发净值', '较现价', '仓位', '累计', '说明'],
        order_rows, 'orderTable')

    # ---- 回测统计表 ----
    bt_rows = []
    for r in bt_results:
        if r.equity is None or len(r.equity) == 0:
            continue
        win_rate = (r.win_rounds / (r.win_rounds + r.lose_rounds) * 100) \
            if (r.win_rounds + r.lose_rounds) else 0
        color = '#1a9850' if r.total_return > 0 else '#d73027'
        bt_rows.append([
            f'{r.name}<br><span class="code">{r.code}</span>',
            r.plans_activated, r.n_buys, r.n_sells,
            f'<span style="color:{color};font-weight:600">{r.total_return*100:.1f}%</span>',
            f'{r.max_drawdown*100:.1f}%',
            f'{win_rate:.0f}%',
            f'{r.avg_hold_days:.0f}天',
            r.terminate_count, r.stop_count,
        ])
    bt_table = _table(
        ['基金', '开仓<br>次数', '买入<br>笔数', '卖出<br>笔数', '区间收益',
         '最大回撤', '回合<br>胜率', '平均<br>持有', '终止<br>触发', '止损<br>次数'],
        bt_rows, 'btTable')

    # ---- 图表 ----
    divs = []
    plotly_js = True
    for fig in [fig_scatter(screen_df), fig_percentile_bar(screen_df)]:
        divs.append(pio.to_html(fig, full_html=False,
                                include_plotlyjs='cdn' if plotly_js else False))
        plotly_js = False

    # 条件单图（入选+观察）
    ladder_divs = []
    plan_map = {p.code: p for p in plans}
    bt_map = {r.code: r for r in bt_results}
    cand = screen_df[screen_df['bottom_status'].isin(['confirmed', 'watch'])
                     & (screen_df['risk_pass'] == True)]  # noqa: E712
    for _, r in cand.iterrows():
        code = r['code']
        nav = funds_cache[code]['nav']
        nav_s = nav['close'] if hasattr(nav, 'columns') else nav
        fig = fig_ladder_chart(code, r['name'], nav_s, plan_map.get(code),
                               bt_map.get(code))
        ladder_divs.append(pio.to_html(fig, full_html=False, include_plotlyjs=False))

    # 回测曲线
    divs.append(pio.to_html(fig_equity(portfolio, benchmark),
                            full_html=False, include_plotlyjs=False))

    port_ret = (portfolio['total_value'].iloc[-1] / portfolio['total_value'].iloc[0] - 1) \
        if portfolio is not None and len(portfolio) else 0
    bench_ret = (benchmark.iloc[-1] / benchmark.iloc[0] - 1) \
        if benchmark is not None and len(benchmark) else 0

    html = rf"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>双底确认+自适应阶梯抄底 · 筛选与条件单仪表盘</title>
<style>
  body {{ font-family: -apple-system, 'Microsoft YaHei', sans-serif; margin: 0;
         background: #f4f6f9; color: #222; }}
  .header {{ background: linear-gradient(135deg, #1a4d2e, #2c6fbb); color: #fff;
             padding: 26px 36px; }}
  .header h1 {{ margin: 0 0 6px; font-size: 24px; }}
  .header p {{ margin: 0; opacity: .85; font-size: 13px; }}
  .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 14px; margin-bottom: 24px; }}
  .card {{ background: #fff; border-radius: 10px; padding: 16px; text-align: center;
           box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .card .num {{ font-size: 28px; font-weight: 700; }}
  .card .label {{ font-size: 12px; color: #777; margin-top: 4px; }}
  .section {{ background: #fff; border-radius: 10px; padding: 20px 24px; margin-bottom: 22px;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
  .section h2 {{ font-size: 17px; margin: 0 0 14px; border-left: 4px solid #2c6fbb;
                 padding-left: 10px; }}
  .section h3 {{ font-size: 14px; color: #555; margin: 16px 0 8px; }}
  table.data-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  table.data-table th {{ background: #f0f3f7; padding: 7px 6px; text-align: center;
                         border-bottom: 2px solid #d0d7de; white-space: nowrap;
                         position: sticky; top: 0; }}
  table.data-table td {{ padding: 6px; border-bottom: 1px solid #eef0f3;
                         text-align: center; vertical-align: middle; }}
  table.data-table tbody tr:hover {{ background: #f6faf6; }}
  .badge {{ color: #fff; padding: 3px 9px; border-radius: 10px; font-size: 11px;
            white-space: nowrap; display: inline-block; }}
  .code {{ color: #999; font-size: 11px; }}
  .reason {{ color: #d73027; font-size: 11px; }}
  .note {{ font-size: 12px; color: #777; line-height: 1.7; background: #f8f9fb;
           border-radius: 8px; padding: 12px 16px; margin-top: 12px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 1000px) {{ .grid2 {{ grid-template-columns: 1fr; }} }}
  @media (max-width: 768px) {{
    body {{ margin: 0 !important; padding: 8px !important; }}
    .header {{ padding: 16px 14px !important; }}
    .header h1, h1 {{ font-size: 18px !important; }}
    .container {{ padding: 10px 8px !important; }}
    .card, .section {{ padding: 12px 10px !important; }}
    .card .num {{ font-size: 22px !important; }}
    table.data-table {{ font-size: 11px !important; }}
    .card, .section, .container {{ overflow-x: auto; -webkit-overflow-scrolling: touch; }}
    .js-plotly-plot, .plotly-graph-div {{ max-width: 100% !important; }}
  }}
</style>
</head><body>
<div class="header">
  <h1>双底确认 + 波动率自适应阶梯抄底体系</h1>
  <p>价格底(3年净值分位)+估值底(长周期分位/高点回撤/年线偏离) 双确认 ｜ 四维量化排雷 ｜
     高波动宽档、低波动密档 ｜ 动态终止/分批止盈/移动止盈 ｜ 基金池{n_total}只（真实{n_real}）· 生成于 {date.today()}</p>
</div>
<div class="container">

  <div class="cards">
    <div class="card"><div class="num">{n_total}</div><div class="label">基金池总数</div></div>
    <div class="card"><div class="num" style="color:#1a9850">{n_confirmed}</div><div class="label">双底确认(可挂单)</div></div>
    <div class="card"><div class="num" style="color:#fdae61">{n_watch}</div><div class="label">观察名单</div></div>
    <div class="card"><div class="num" style="color:#d73027">{n_risk_fail}</div><div class="label">排雷剔除</div></div>
    <div class="card"><div class="num">{avg_pct:.0f}%</div><div class="label">池均3年分位</div></div>
    <div class="card"><div class="num" style="color:{'#1a9850' if port_ret>0 else '#d73027'}">{port_ret*100:.1f}%</div><div class="label">组合回测收益(近3年)</div></div>
    <div class="card"><div class="num" style="color:#666">{bench_ret*100:.1f}%</div><div class="label">沪深300同期</div></div>
  </div>

  <div class="section">
    <h2>① 筛选结果总览（点击表头排序）</h2>
    <div style="max-height:560px;overflow:auto">{screen_table}</div>
    <div class="note">
      <b>评级规则</b>：✗排雷剔除 = 四维任一硬门槛不通过（D1年限/数据质量 · D2清盘僵尸 · D3价值陷阱新低 · D4波动崩坏）；
      ★★★/★★ = 价格底(3年分位≤15%)且估值底3项代理满足≥2项；★ = 分位≤30%且估值满足≥1项。<br>
      <b>估值底代理</b>：开源净值接口无PE/PB，用 长周期分位≤30% / 高点回撤≥25% / 低于年线≥10% 三项替代；接入理杏仁PE数据可替换第①项。
    </div>
  </div>

  <div class="section">
    <h2>② 底部定位分布</h2>
    <div class="grid2">{divs[0]}{divs[1]}</div>
  </div>

  <div class="section">
    <h2>③ 条件单阶梯图（绿=买入档 · 红虚线=止盈档 · 橙点线=终止线 · 深红=硬止损 · 三角=回测成交点）</h2>
    <div class="grid2">{''.join(ladder_divs)}</div>
  </div>

  <div class="section">
    <h2>④ 条件单清单（可直接录入券商/基金平台条件单）</h2>
    <div style="max-height:520px;overflow:auto">{orders_table}</div>
    <div class="note">
      <b>执行纪律</b>：条件单全部挂在<b>现价下方</b>，跌到才买，不现价梭哈；单只基金不超过权益资金15%；
      条件单有效期180个交易日，到期/涨回30%分位以上自动撤销，每月重跑筛选；
      跌破终止线立即暂停加仓并复检基本面，触发硬止损无条件离场。
    </div>
  </div>

  <div class="section">
    <h2>⑤ 近3年回测验证（月度重检 + 日度条件单触发）</h2>
    {divs[2]}
    <h3>分基金回测明细</h3>
    <div style="max-height:480px;overflow:auto">{bt_table}</div>
  </div>

</div>
<script>
function sortTable(tableId, col) {{
  var table = document.getElementById(tableId);
  var tbody = table.querySelector('tbody');
  var rows = Array.from(tbody.querySelectorAll('tr'));
  var dir = table.getAttribute('data-sort-' + col) === 'asc' ? 'desc' : 'asc';
  rows.sort(function(a, b) {{
    var x = a.cells[col].innerText.replace(/[^\d.-]/g, '');
    var y = b.cells[col].innerText.replace(/[^\d.-]/g, '');
    var nx = parseFloat(x), ny = parseFloat(y);
    if (!isNaN(nx) && !isNaN(ny) && x !== '' && y !== '') return dir === 'asc' ? nx - ny : ny - nx;
    return dir === 'asc'
      ? a.cells[col].innerText.localeCompare(b.cells[col].innerText, 'zh')
      : b.cells[col].innerText.localeCompare(a.cells[col].innerText, 'zh');
  }});
  rows.forEach(function(r) {{ tbody.appendChild(r); }});
  table.setAttribute('data-sort-' + col, dir);
}}
</script>
<script>
(function(){{
  function isMobile(){{return Math.min(screen.width, screen.height) <= 768;}}
  function adaptPlot(gd){{
    if(!window.Plotly||!gd||!gd.layout)return;
    if(!gd._deskLayout){{
      var m=gd.layout.margin||{{}};
      gd._deskLayout={{l:m.l,r:m.r,t:m.t,b:m.b}};
    }}
    var d=gd._deskLayout;
    if(isMobile()){{
      var left=d.l||80;
      var barHeavy=left>100;
      Plotly.relayout(gd,{{
        'margin.l':barHeavy?Math.min(left,80):20,
        'margin.r':8,
        'margin.t':Math.min(d.t||60,44),
        'margin.b':Math.min(d.b||40,28)
      }});
      gd._mobAdapted=true;
    }}else if(gd._mobAdapted){{
      var patch={{}};
      if(d.l!=null)patch['margin.l']=d.l;
      if(d.r!=null)patch['margin.r']=d.r;
      if(d.t!=null)patch['margin.t']=d.t;
      if(d.b!=null)patch['margin.b']=d.b;
      Plotly.relayout(gd,patch);
      gd._mobAdapted=false;
    }}
  }}
  function run(){{document.querySelectorAll('.js-plotly-plot').forEach(adaptPlot);}}
  var timer; function schedule(){{clearTimeout(timer);timer=setTimeout(run,200);}}
  if(document.readyState==='complete')schedule();
  else window.addEventListener('load',schedule);
  window.addEventListener('resize',schedule);
  setTimeout(schedule,800); setTimeout(schedule,1800);
}})();
</script>
</body></html>"""

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\n[输出] 仪表盘: {out_path}')


# 全局缓存供绘图复用
funds_cache = {}


def main():
    global funds_cache
    print('=' * 70)
    print('  策略7：双底确认 + 波动率自适应阶梯抄底')
    print('=' * 70)

    fund_list = parse_fund_list(FUND_POOL_FILE)
    print(f'\n[1/5] 加载基金池净值（{len(fund_list)}只）...')
    funds = load_pool_data(fund_list)
    funds_cache = funds
    if not funds:
        print('无可用数据，退出。')
        return

    print(f'\n[2/5] 双底确认 + 四维排雷筛选...')
    screen_df = screen_pool(funds)
    for _, r in screen_df.iterrows():
        flag = r['rating']
        print(f"  {flag:<12s} {r['code']} {r['name'][:16]:<16s} "
              f"评分{r['bottom_score']:>5.1f} 3年分位{r['pct_3y']*100:>5.1f}% "
              f"回撤{r['dd_from_peak']*100:>6.1f}% 波动{r['ann_vol']*100:>5.1f}%"
              + (f"  排雷:{r['risk_fail_dims']}" if not r['risk_pass'] else ''))

    # 筛选结果 CSV
    csv_df = screen_df.drop(columns=['metrics', 'risk_reasons', 'val_detail'], errors='ignore').copy()
    csv_df['risk_reasons'] = screen_df['risk_reasons'].apply(lambda x: '；'.join(x))
    csv_df.to_csv(OUT_SCREEN_CSV, index=False, encoding='utf-8-sig')
    print(f'\n[输出] 筛选结果: {OUT_SCREEN_CSV}')

    print(f'\n[3/5] 生成波动率自适应阶梯条件单...')
    plans = []
    order_rows = []
    # 纪律：排雷通过(risk_pass) 才允许挂条件单；✗剔除基金不出条件单
    candidates = screen_df[screen_df['bottom_status'].isin(['confirmed', 'watch'])
                           & (screen_df['data_source'] == 'real')
                           & (screen_df['risk_pass'] == True)]  # noqa: E712
    for _, r in candidates.iterrows():
        nav = funds[r['code']]['nav']
        nav_s = nav['close'] if hasattr(nav, 'columns') else nav
        m = r.get('metrics')
        ann_vol = m['ann_vol'] if m else nav_s.pct_change().iloc[-252:].std() * np.sqrt(252)
        plan = generate_adaptive_ladder(r['code'], r['name'], r['nav_now'], ann_vol)
        plans.append(plan)
        print(f"  {r['code']} {r['name'][:14]:<14s} {plan.vol_regime} "
              f"档距{plan.spacing:.0%} {plan.n_tiers}档 "
              f"买档{r['nav_now']*(1-plan.spacing):.4f}~{plan.buy_tiers[-1].trigger_nav:.4f} "
              f"止损{plan.hard_stop_pct:.0%}")
        for row in plan.order_rows():
            order_rows.append({
                '基金代码': r['code'], '基金名称': r['name'],
                '波动档位': plan.vol_regime, '档距': plan.spacing,
                '当前净值': round(plan.nav0, 4), **row,
            })
    orders_df = pd.DataFrame(order_rows)
    if len(orders_df):
        orders_df.to_csv(OUT_ORDERS_CSV, index=False, encoding='utf-8-sig')
        print(f'[输出] 条件单清单: {OUT_ORDERS_CSV}')

    print(f'\n[4/5] 近{BT_YEARS}年条件单回测...')
    bt_start = compute_start_date(BT_YEARS)
    bt_results = []
    bt_rows = []
    for code, info in funds.items():
        nav = info['nav']
        nav_s = nav['close'] if hasattr(nav, 'columns') else nav
        res = backtest_fund(code, info['name'], nav_s, bt_start, CAPITAL_PER_FUND)
        bt_results.append(res)
        if len(res.equity) and res.plans_activated > 0:
            print(f"  {code} {info['name'][:14]:<14s} 开仓{res.plans_activated}次 "
                  f"收益{res.total_return*100:>6.1f}% 回撤{res.max_drawdown*100:>6.1f}% "
                  f"买{res.n_buys} 卖{res.n_sells} 止损{res.stop_count}")
            bt_rows.append({
                '基金代码': code, '基金名称': info['name'],
                '开仓次数': res.plans_activated, '买入笔数': res.n_buys,
                '卖出笔数': res.n_sells, '区间收益%': round(res.total_return * 100, 2),
                '最大回撤%': round(res.max_drawdown * 100, 2),
                '胜回合': res.win_rounds, '负回合': res.lose_rounds,
                '平均持有天数': round(res.avg_hold_days, 0),
                '终止触发': res.terminate_count, '止损次数': res.stop_count,
                '期末资产': round(res.final_value, 0),
            })
    pd.DataFrame(bt_rows).to_csv(OUT_BT_CSV, index=False, encoding='utf-8-sig')

    portfolio = aggregate_portfolio(bt_results, CAPITAL_PER_FUND)
    if len(portfolio):
        portfolio.to_csv(OUT_EQUITY_CSV, encoding='utf-8-sig')
        print(f"  组合(等权{len(funds)}只) 收益{(portfolio['nav'].iloc[-1]-1)*100:.1f}%")

    # 基准：沪深300ETF
    benchmark = None
    try:
        bm = fetch_real_etf_kline('510300', '沪深300ETF', '2020-01-01', DATA_END)
        if len(bm):
            bm_s = bm['close']
            bm_s = bm_s[bm_s.index >= pd.Timestamp(bt_start)]
            benchmark = bm_s / bm_s.iloc[0]
    except Exception:
        pass

    print(f'\n[5/5] 生成可视化仪表盘...')
    build_dashboard(screen_df, plans, orders_df, bt_results, portfolio, benchmark, OUT_HTML)
    print('完成。')


if __name__ == '__main__':
    main()
