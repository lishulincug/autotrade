# -*- coding: utf-8 -*-
"""
策略8 · 全市场榜单 + 双底确认 + 自适应阶梯
==========================================
用法:
  python run_strategy8_dual_bottom.py
  python run_strategy8_dual_bottom.py --universe pool
  python run_strategy8_dual_bottom.py --universe market --sort 近1年 --top 200 --skip-bt
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from common.data_loader import load_otc_fund_nav
from strategy8_dual_bottom_ladder import config as C
from strategy8_dual_bottom_ladder.universe import load_market_universe, load_pool_universe
from strategy8_dual_bottom_ladder.data_meta import fetch_meta
from strategy8_dual_bottom_ladder.screening import screen_pool
from strategy8_dual_bottom_ladder.portfolio import apply_sector_cap
from strategy8_dual_bottom_ladder.backtest import backtest_fund, aggregate

try:
    import plotly.graph_objects as go
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def years_ago(n: int = 3) -> str:
    t = date.today()
    try:
        return t.replace(year=t.year - n).isoformat()
    except ValueError:
        return t.replace(year=t.year - n, day=28).isoformat()


def fetch_nav_map(deep: pd.DataFrame, verbose: bool = True) -> dict:
    out = {}
    total = len(deep)
    for i, (_, row) in enumerate(deep.iterrows(), 1):
        code = str(row['基金代码']).zfill(6)
        name = str(row.get('基金简称', code))
        if verbose and i % 25 == 0:
            print(f'  [nav] {i}/{total}')
        try:
            df = load_otc_fund_nav(code, name, C.NAV_START, C.NAV_END, verbose=False)
            if df is not None and len(df) > 60:
                out[code] = df
        except Exception as e:
            if verbose:
                print(f'  [nav] {code} 失败: {e}')
    if verbose:
        print(f'  [nav] 成功 {len(out)}/{total}')
    return out


def fetch_meta_map(deep: pd.DataFrame, verbose: bool = True) -> dict:
    out = {}
    for _, row in deep.iterrows():
        code = str(row['基金代码']).zfill(6)
        name = str(row.get('基金简称', code))
        ftype = str(row.get('基金类型', ''))
        m = fetch_meta(code, name, ftype)
        out[code] = {
            'meta_ok': m.get('meta_ok', False),
            'scale_yi': m.get('scale_yi'),
            'manager_years': m.get('manager_years'),
            'sector': m.get('sector', '其他'),
            'is_index': m.get('is_index', False),
            'warnings': m.get('warnings', []),
        }
    if verbose:
        ok = sum(1 for v in out.values() if v.get('meta_ok'))
        print(f'  [meta] 成功 {ok}/{len(out)}')
    return out


def build_dashboard(board, screen_df, orders, equity, bt_summary, out_html):
    if not HAS_PLOTLY:
        print('  [dash] 无 plotly，跳过 HTML')
        return

    figs = []
    show_cols = [c for c in ['榜单排名', '基金代码', '基金简称', '基金类型',
                             '近1周', '近1月', '近1年', '近3年', '日增长率', '单位净值']
                 if c in board.columns]
    board_show = board[show_cols].head(300) if show_cols else board.head(300)
    fig1 = go.Figure(data=[go.Table(
        header=dict(values=list(board_show.columns), fill_color='#1f4e79',
                    font=dict(color='white'), align='left'),
        cells=dict(values=[board_show[c] for c in board_show.columns],
                   fill_color='#f7f9fc', align='left'),
    )])
    fig1.update_layout(title=f'全市场开放式基金榜单预览（共{len(board)}只，展示前300）', height=520)
    figs.append(fig1)

    if len(screen_df) and 'pct_3y' in screen_df.columns:
        fig2 = go.Figure()
        for status, color in [('confirmed', '#1a9850'), ('watch', '#fdae61'), ('none', '#999')]:
            sub = screen_df[screen_df['bottom_status'] == status]
            if not len(sub):
                continue
            ycol = 'dd_from_peak' if 'dd_from_peak' in sub.columns else 'max_dd_3y'
            fig2.add_trace(go.Scatter(
                x=sub['pct_3y'] * 100,
                y=sub[ycol] * 100,
                mode='markers+text',
                text=sub['name'].astype(str).str.slice(0, 8),
                textposition='top center',
                marker=dict(size=np.clip(sub['bottom_score'].fillna(10) * 0.35, 8, 26),
                            color=color, opacity=0.85),
                name=status,
            ))
        fig2.add_vline(x=15, line_dash='dot', line_color='#1a9850')
        fig2.update_layout(title='双底定位：3年分位 × 距高点回撤',
                           xaxis_title='3年分位(%)', yaxis_title='回撤(%)',
                           height=460, template='plotly_white')
        figs.append(fig2)

    if len(screen_df):
        top = screen_df.head(30).iloc[::-1]
        colors = ['#1a9850' if s == 'confirmed' else '#fdae61' if s == 'watch' else '#aaa'
                  for s in top['bottom_status']]
        fig3 = go.Figure(go.Bar(
            x=top['bottom_score'], y=top['name'].astype(str).str.slice(0, 14),
            orientation='h', marker_color=colors, text=top['rating'], textposition='auto',
        ))
        fig3.update_layout(title='深度筛评分 Top30', height=560, template='plotly_white',
                           margin=dict(l=160))
        figs.append(fig3)

    if equity is not None and len(equity) and 'nav' in equity.columns:
        fig4 = go.Figure(go.Scatter(
            x=equity.index, y=equity['nav'], name='等权组合',
            line=dict(color='#2c6fbb', width=2)))
        fig4.update_layout(title='入选标的等权回测净值', height=400, template='plotly_white')
        figs.append(fig4)

    n_conf = int((screen_df['bottom_status'] == 'confirmed').sum()) if len(screen_df) else 0
    parts = [
        '<html><head><meta charset="utf-8">'
        '<title>策略8 双底阶梯仪表盘</title>',
        '<style>body{font-family:Segoe UI,Microsoft YaHei,sans-serif;margin:24px;background:#f4f6f8}'
        'h1{color:#1f4e79}.card{background:#fff;padding:16px;border-radius:8px;margin-bottom:16px;'
        'box-shadow:0 1px 3px rgba(0,0,0,.08)} table{border-collapse:collapse;width:100%}'
        'th,td{border:1px solid #ddd;padding:6px 8px;font-size:12px} th{background:#e8eef5}'
        '@media (max-width:768px){body{margin:0!important;padding:8px!important}'
        'h1{font-size:18px!important}.card{padding:12px 10px!important}'
        'table{font-size:11px!important}.card{overflow-x:auto;-webkit-overflow-scrolling:touch}'
        '.js-plotly-plot,.plotly-graph-div{max-width:100%!important}}</style>',
        '</head><body>',
        '<h1>策略8 · 全市场榜单 + 双底确认 + 自适应阶梯</h1>',
        f'<div class="card"><b>榜单</b> {len(board)} | <b>深度筛</b> {len(screen_df)} | '
        f'<b>双底确认</b> {n_conf} | <b>条件单</b> {0 if orders is None else len(orders)}</div>',
    ]
    for i, fig in enumerate(figs):
        parts.append('<div class="card">' + fig.to_html(
            full_html=False, include_plotlyjs='cdn' if i == 0 else False) + '</div>')
    if orders is not None and len(orders):
        parts.append('<div class="card"><h2>条件单预览</h2>')
        parts.append(orders.head(100).to_html(index=False, border=0))
        parts.append('</div>')
    if bt_summary is not None and len(bt_summary):
        parts.append('<div class="card"><h2>回测摘要</h2>')
        parts.append(bt_summary.to_html(index=False, border=0))
        parts.append('</div>')
    parts.append("""<script>
(function(){
  function isMobile(){return Math.min(screen.width, screen.height) <= 768;}
  function adaptPlot(gd){
    if(!window.Plotly||!gd||!gd.layout)return;
    if(!gd._deskLayout){
      var m=gd.layout.margin||{};
      gd._deskLayout={l:m.l,r:m.r,t:m.t,b:m.b};
    }
    var d=gd._deskLayout;
    if(isMobile()){
      var left=d.l||80;
      var barHeavy=left>100;
      Plotly.relayout(gd,{
        'margin.l':barHeavy?Math.min(left,80):20,
        'margin.r':8,
        'margin.t':Math.min(d.t||60,44),
        'margin.b':Math.min(d.b||40,28)
      });
      gd._mobAdapted=true;
    }else if(gd._mobAdapted){
      var patch={};
      if(d.l!=null)patch['margin.l']=d.l;
      if(d.r!=null)patch['margin.r']=d.r;
      if(d.t!=null)patch['margin.t']=d.t;
      if(d.b!=null)patch['margin.b']=d.b;
      Plotly.relayout(gd,patch);
      gd._mobAdapted=false;
    }
  }
  function run(){document.querySelectorAll('.js-plotly-plot').forEach(adaptPlot);}
  var timer; function schedule(){clearTimeout(timer);timer=setTimeout(run,200);}
  if(document.readyState==='complete')schedule();
  else window.addEventListener('load',schedule);
  window.addEventListener('resize',schedule);
  setTimeout(schedule,800); setTimeout(schedule,1800);
})();
</script>""")
    parts.append('</body></html>')
    with open(out_html, 'w', encoding='utf-8') as f:
        f.write('\n'.join(parts))
    print(f'  [dash] {out_html}')


def main():
    ap = argparse.ArgumentParser(description='策略8 双底阶梯')
    ap.add_argument('--universe', choices=['market', 'pool'], default='market')
    ap.add_argument('--sort', default=None)
    ap.add_argument('--desc', action='store_true')
    ap.add_argument('--top', type=int, default=None)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--skip-bt', action='store_true')
    args = ap.parse_args()

    print('=' * 60)
    print('策略8：全市场榜单 + 双底 + 自适应阶梯')
    print('=' * 60)

    print('\n[1/6] 加载基金宇宙...')
    if args.universe == 'pool':
        board, deep = load_pool_universe()
    else:
        board, deep = load_market_universe(
            sort_col=args.sort,
            ascending=not args.desc,
            top_n=args.top,
            use_cache=not args.no_cache,
        )
    if not len(deep):
        print('候选为空，退出')
        return
    board.to_csv(C.OUT_RANK, index=False, encoding='utf-8-sig')
    print(f'  榜单 {len(board)} → 深度候选 {len(deep)} → {C.OUT_RANK}')

    print('\n[2/6] 拉取净值与概况...')
    nav_map = fetch_nav_map(deep)
    meta_map = fetch_meta_map(deep)
    if not nav_map:
        print('无可用净值，退出')
        return

    print('\n[3/6] 双底确认 + 四维排雷...')
    screen_df = screen_pool(deep, nav_map, meta_map)
    screen_df.to_csv(C.OUT_SCREEN, index=False, encoding='utf-8-sig')
    watch = screen_df
    if len(screen_df) and 'risk_ok' in screen_df.columns and 'bottom_status' in screen_df.columns:
        watch = screen_df[screen_df['risk_ok'] & screen_df['bottom_status'].isin(['confirmed', 'watch'])]
    n_conf = int((screen_df['bottom_status'] == 'confirmed').sum()) if len(screen_df) else 0
    print(f'  筛完 {len(screen_df)} | 确认 {n_conf} | 可挂单 {len(watch)} → {C.OUT_SCREEN}')

    print('\n[4/6] 生成阶梯条件单（行业≤30%）...')
    selected = watch.head(C.BT_MAX).copy() if len(watch) else screen_df.head(0)
    _, plans, orders = apply_sector_cap(selected)
    orders.to_csv(C.OUT_ORDERS, index=False, encoding='utf-8-sig')
    print(f'  计划 {len(plans)} 只，条件单 {len(orders)} 行 → {C.OUT_ORDERS}')

    equity = pd.DataFrame()
    bt_summary = pd.DataFrame()
    if (not args.skip_bt) and plans:
        print('\n[5/6] 事件驱动回测...')
        start = years_ago(3)
        results, rows = [], []
        for plan in plans:
            nav = nav_map.get(plan.code)
            if nav is None:
                continue
            series = nav['close'] if hasattr(nav, 'columns') and 'close' in nav.columns else nav
            r = backtest_fund(plan.code, plan.name, series, start=start,
                              capital=plan.planned_capital, sector=plan.sector)
            results.append(r)
            rows.append({
                '基金代码': r.code, '基金简称': r.name,
                '总收益': round(r.total_return, 4), '最大回撤': round(r.max_dd, 4),
                '买入次数': r.n_buys, '卖出次数': r.n_sells,
                '止损次数': r.stop_count, '终止次数': r.terminate_count,
            })
            print(f'  {r.code} {r.name[:12]} 收益{r.total_return:.1%} 回撤{r.max_dd:.1%} '
                  f'买{r.n_buys}/卖{r.n_sells}')
        bt_summary = pd.DataFrame(rows)
        if len(bt_summary):
            bt_summary.to_csv(C.OUT_BT, index=False, encoding='utf-8-sig')
        equity = aggregate(results)
        if len(equity):
            equity.to_csv(C.OUT_EQ, encoding='utf-8-sig')
            col = 'nav' if 'nav' in equity.columns else equity.columns[-1]
            print(f'  组合最终净值 {float(equity[col].iloc[-1]):.3f}')
    else:
        print('\n[5/6] 跳过回测')

    print('\n[6/6] 生成仪表盘...')
    build_dashboard(board, screen_df, orders, equity, bt_summary, C.OUT_HTML)
    print('\n完成。')


if __name__ == '__main__':
    main()
