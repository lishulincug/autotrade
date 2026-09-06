# -*- coding: utf-8 -*-
"""
广发约定净值转换 · 多基金多策略可视化对比
============================================
读取 fund.txt 中的基金清单，对每只基金运行网格策略：
  A 动态滚动网格 / B 估值分位网格 / C 高点回撤止盈
  D 底仓锁利+浮动网格 / E 净值分位自适应双区 / G 阶梯止盈映射
（F 多标的轮动为组合层，见 run_gf_rotation.py）

用法：
    python run_gf_visualization.py

输出：
    gf_strategy_dashboard.html  — 交互式可视化报告
    gf_multi_fund_comparison.csv — 多基金×多策略指标表
"""
import os
import sys
import warnings
import pandas as pd
from datetime import date

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.data_loader import load_otc_fund_nav
from strategy6_gf_nav_conversion.strategy import run_strategy as run_a
from strategy6_gf_nav_conversion.strategy_b_valuation import run_strategy as run_b
from strategy6_gf_nav_conversion.strategy_c_drawdown import run_strategy as run_c
from strategy6_gf_nav_conversion.strategy_d_core_float import run_strategy as run_d
from strategy6_gf_nav_conversion.strategy_e_dual_zone import run_strategy as run_e
from strategy6_gf_nav_conversion.strategy_g_ladder_trail import run_strategy as run_g

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
from plotly.offline.offline import get_plotlyjs_version

# ---------- 配置 ----------
FUND_LIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'strategy6_gf_nav_conversion', 'fund.txt')
INITIAL_CASH = 1_000_000
MIN_DATA_LEN = 60  # 最少数据条数，不足则跳过
PARAM_DECIMALS = 2  # 策略最新参数（约定净值）保留小数位数，默认2位（广发基金约定设置转换净值只保留2位）

STRATEGIES = [
    ('A', '动态滚动网格', run_a, '#1f77b4'),
    ('B', '估值分位网格', run_b, '#ff7f0e'),
    ('C', '高点回撤止盈', run_c, '#2ca02c'),
    ('D', '底仓锁利+浮动', run_d, '#d62728'),
    ('E', '分位自适应双区', run_e, '#9467bd'),
    ('G', '阶梯止盈映射', run_g, '#8c564b'),
]

# 弹窗策略简介与关键参数说明（静态文案；动态值由 build_params_map 追加）
STRATEGY_HELP = {
    'A': {
        'intro': '用 warmup 期历史净值分位数划定固定买卖网格：净值≤档买入、≥档止盈；止盈后重置更低买入档，形成买低卖高循环。',
        'keys': [
            '网格：warmup 全历史分位数静态划定（非滚动重算）',
            '默认买档约 Q15–Q70，卖档约 Q75–Q90',
            '适合波动适中、长期有中枢的标的',
        ],
    },
    'B': {
        'intro': '用滚动窗口计算当前净值百分位：分位偏低买入、偏高止盈，网格随市场自适应。',
        'keys': [
            '默认滚动窗口约 120 日',
            '买入阈值约 30%/20%/10% 分位，止盈约 70%/80%/90%',
            '比 A 更跟趋势，但震荡市可能更频繁转换',
        ],
    },
    'C': {
        'intro': '跟踪滚动高点：从高点回撤 X% 分档买入；相对持仓成本上涨 Y% 分档止盈。',
        'keys': [
            '默认高点窗口约 60 日',
            '买入回撤约 3%/6%/9%/12%/15%，止盈约 3%/5%/8%',
            '适合“跌出来的机会、涨出来的利润”；相对规则需映射为固定净值才能录入广发',
        ],
    },
    'D': {
        'intro': '持仓拆成底仓与浮动仓：底仓低位买入后长期持有（不设止盈），浮动仓做高抛低吸，避免纯网格在牛市卖飞。',
        'keys': [
            '默认仓位：底仓 40% / 浮动 60%（以目标份额近似）',
            '买入最低档 + Q20 加仓计入底仓；卖出总份额=浮动仓，底仓永不卖',
            '净值≥历史 Q90 时仍只卖浮动仓；买入档间距≥3%',
            '回测启用持有＜7天 1.5% 惩罚赎回费',
        ],
    },
    'E': {
        'intro': '用基金自身近 3 年净值分位划分交易区：低估只买、震荡网格、高估只卖，每月或跨区切换条件单。',
        'keys': [
            '低估区 ≤20% 分位：只买不卖，买入份额放大',
            '震荡区 20%–80%：标准动态网格双向触发',
            '高估区 ≥80% 分位：只卖不买',
            '买入档间距≥3%；回测启用＜7天 1.5% 惩罚费',
        ],
    },
    'G': {
        'intro': '用多档固定净值模拟移动止损：净值每上台阶激活对应止损档，更高台阶作废更低止损。仅适合明确主升浪。',
        'keys': [
            '上涨台阶约 +5%/+10%/+15%/+20% 激活对应回撤止损',
            '实盘需手动删除低档止损单、挂上新档',
            '震荡市易反复触发磨损收益，不建议长期启用',
            '买入档间距≥3%；回测启用＜7天 1.5% 惩罚费',
        ],
    },
}

# ---------- 回测区间配置 ----------
# 下拉选项：近1年 / 近2年 / 近3年 / 全部，默认近3年
# 数据不足 N 年的基金自动用其全部可用数据（回测区间列如实显示实际跨度）
RANGES = [
    ('3y', '近3年', 3),
    ('2y', '近2年', 2),
    ('1y', '近1年', 1),
    ('all', '全部', None),
]
DEFAULT_RANGE = '3y'


def compute_start_date(years):
    """计算 N 年前的日期字符串（YYYY-MM-DD），处理闰年2月29日"""
    today = date.today()
    try:
        return today.replace(year=today.year - years).isoformat()
    except ValueError:
        # 2月29日 → 2月28日
        return today.replace(year=today.year - years, day=28).isoformat()


def parse_fund_list(path):
    """解析基金清单（每行：名称\\t代码）"""
    funds = []
    if not os.path.exists(path):
        return funds
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                # 兼容空格分隔
                parts = line.rsplit(None, 1)
                if len(parts) == 2:
                    name, code = parts
                else:
                    continue
            else:
                name, code = line.split('\t', 1)
            name = name.strip().replace('‑', '-').replace('（', '(').replace('）', ')')
            code = code.strip()
            if code and code.isdigit() and len(code) == 6:
                funds.append((code, name))
    return funds


def run_all_for_fund(code, name, start_date=None, nav_data=None):
    """对单只基金运行全部已注册策略，返回结果字典

    nav_data: 可选，预加载的净值数据（多区间复用，避免重复IO）
    start_date: 回测起始日期（None=自动warmup；给定日期则从该日起回测）
    """
    if nav_data is None:
        nav_data = load_otc_fund_nav(code, name,
                                     start_date='2019-01-01', end_date='2026-12-31',
                                     verbose=False)
    result = {'code': code, 'name': name, 'nav_data': nav_data, 'strategies': {}}

    if len(nav_data) < MIN_DATA_LEN:
        result['error'] = f'数据不足({len(nav_data)}条)'
        return result

    result['data_start'] = nav_data.index[0].strftime('%Y-%m-%d')
    result['data_end'] = nav_data.index[-1].strftime('%Y-%m-%d')
    result['data_len'] = len(nav_data)

    for sid, sname, fn, color in STRATEGIES:
        try:
            metrics, stats, equity, engine = fn(
                initial_cash=INITIAL_CASH,
                nav_data=nav_data,
                fund_code=code, fund_name=name,
                start_date=start_date,
                decimals=PARAM_DECIMALS,
                verbose=False
            )
            result['strategies'][sid] = {
                'name': sname, 'color': color,
                'metrics': metrics, 'stats': stats,
                'equity': equity, 'engine': engine,
            }
        except Exception as e:
            result['strategies'][sid] = {'error': str(e), 'color': color}

    # 实际回测区间：从首个可用权益曲线取，反映 start_date 过滤后的真实跨度
    # 数据不足N年的基金，策略会自然用其全部可用数据，bt_start 即为基金首日
    for sid, sname, fn, color in STRATEGIES:
        sres = result['strategies'].get(sid, {})
        eq = sres.get('equity')
        if eq is not None and len(eq) > 0:
            result['bt_start'] = eq.index[0].strftime('%Y-%m-%d')
            result['bt_end'] = eq.index[-1].strftime('%Y-%m-%d')
            break
    if 'bt_start' not in result:
        result['bt_start'] = result['data_start']
        result['bt_end'] = result['data_end']

    return result


def _fmt_nv(v):
    """格式化约定净值，按 PARAM_DECIMALS 保留小数"""
    return f"{float(v):.{PARAM_DECIMALS}f}"


def format_strategy_params(sres, sid=''):
    """格式化单个策略的最新买入/止盈参数为HTML字符串"""
    if 'engine' not in sres:
        return f'<div class="param-box"><span class="param-sid" style="color:#999">策略{sid}</span> 无参数</div>'
    eng = sres['engine']
    buy_list = getattr(eng, 'gf_buy_list', None) or []
    sell_list = getattr(eng, 'gf_sell_list', None) or []
    color = sres.get('color', '#888')
    sname = sres.get('name', '')
    zone = getattr(eng, 'gf_zone', None)
    pct = getattr(eng, 'gf_percentile', None)

    parts = [f'<div class="param-box"><span class="param-sid" style="color:{color}">●</span> '
             f'<b>策略{sid} {sname}</b>']
    if zone is not None:
        zone_label = {'under': '低估只买', 'mid': '震荡网格', 'over': '高估只卖'}.get(zone, zone)
        extra = f'区间={zone_label}'
        if pct is not None:
            extra += f' 分位={float(pct)*100:.1f}%'
        parts.append(f'<div class="param-row"><span class="param-label">状态:</span> {extra}</div>')
    if buy_list:
        buys = '/'.join(
            f'<span class="param-nv">{_fmt_nv(b["trigger_net_value"])}</span>'
            + (f'<sup>{b.get("role","")}</sup>' if b.get('role') else '')
            for b in buy_list)
        shares = '/'.join(str(b['share']) for b in buy_list)
        parts.append(f'<div class="param-row buy"><span class="param-label">买入净值:</span> {buys} '
                     f'<span class="param-share">份额: {shares}</span></div>')
    if sell_list:
        sells = '/'.join(
            f'<span class="param-nv">{_fmt_nv(s["trigger_net_value"])}</span>'
            + (f'<sup>{s.get("role","")}</sup>' if s.get('role') else '')
            for s in sell_list)
        shares = '/'.join(str(s['share']) for s in sell_list)
        parts.append(f'<div class="param-row sell"><span class="param-label">止盈净值:</span> {sells} '
                     f'<span class="param-share">份额: {shares}</span></div>')
    parts.append('</div>')
    return ''.join(parts)


def build_fund_params_block(res):
    """构建单只基金多策略最新参数展示块"""
    html = ['<div class="params-block">']
    for sid, sname, fn, color in STRATEGIES:
        sres = res['strategies'].get(sid, {})
        html.append(format_strategy_params(sres, sid))
    html.append('</div>')
    return '\n'.join(html)


def build_fund_figure(res):
    """构建单只基金的图表：净值+买卖点（上）、权益曲线（下）"""
    nav = res['nav_data']
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.5, 0.5],
        vertical_spacing=0.08,
        subplot_titles=(
            f"{res['name']}（{res['code']}）净值与买卖点",
            f"多策略权益曲线 vs 买入持有"
        )
    )

    # ---- 上图：基金净值 ----
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav['close'],
        mode='lines', name='基金净值',
        line=dict(color='#999999', width=1.5),
        legendgroup='nav', showlegend=True
    ), row=1, col=1)

    # 买卖点（按策略分色）
    for sid, sname, fn, color in STRATEGIES:
        sres = res['strategies'].get(sid, {})
        if 'engine' not in sres:
            continue
        trades = sres['engine'].trades
        buys = [(t.date, t.price) for t in trades if t.action == 'buy']
        sells = [(t.date, t.price) for t in trades if t.action == 'sell']
        if buys:
            bx, by = zip(*buys)
            fig.add_trace(go.Scatter(
                x=bx, y=by, mode='markers',
                name=f'{sid}买入', legendgroup=f'{sid}b',
                marker=dict(symbol='triangle-up', size=9, color=color,
                            line=dict(width=0.5, color='white')),
                hovertemplate=f'{sid}买入<br>%{{x}}<br>净值%{{y:.4f}}<extra></extra>'
            ), row=1, col=1)
        if sells:
            sx, sy = zip(*sells)
            fig.add_trace(go.Scatter(
                x=sx, y=sy, mode='markers',
                name=f'{sid}止盈', legendgroup=f'{sid}s',
                marker=dict(symbol='triangle-down', size=9, color=color,
                            line=dict(width=0.5, color='white'), opacity=0.65),
                marker_symbol='triangle-down',
                hovertemplate=f'{sid}止盈<br>%{{x}}<br>净值%{{y:.4f}}<extra></extra>'
            ), row=1, col=1)

    # ---- 下图：权益曲线 ----
    # 买入持有基准（从首条回测日归一）
    first_equity = None
    for sid, sname, fn, color in STRATEGIES:
        sres = res['strategies'].get(sid, {})
        if 'equity' not in sres or sres['equity'] is None or len(sres['equity']) == 0:
            continue
        eq = sres['equity']
        if first_equity is None:
            first_equity = eq.index[0]
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq['total_value'] / INITIAL_CASH,
            mode='lines', name=f'{sid} {sname}',
            line=dict(color=color, width=2),
            legendgroup=f'{sid}eq'
        ), row=2, col=1)

    # 买入持有（用全段净值，归一到起始资金=1）
    if first_equity is not None:
        bh = nav['close'].loc[nav.index >= first_equity]
        if len(bh) > 1:
            bh_norm = bh / bh.iloc[0]
            fig.add_trace(go.Scatter(
                x=bh_norm.index, y=bh_norm,
                mode='lines', name='买入持有',
                line=dict(color='#bbbbbb', width=1.5, dash='dash'),
            ), row=2, col=1)

    fig.add_hline(y=1.0, line=dict(color='#cccccc', width=1, dash='dot'), row=2, col=1)
    fig.update_yaxes(title_text='单位净值', row=1, col=1)
    fig.update_yaxes(title_text='权益(起始=1)', row=2, col=1)
    fig.update_layout(
        height=520, margin=dict(l=60, r=30, t=70, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02,
                    xanchor='left', x=0, font=dict(size=10)),
        hovermode='x unified',
        dragmode='pan',
        font=dict(family='Microsoft YaHei, Arial', size=12),
        title_text=None
    )
    return fig


def _clean_metric(v, n_conv, lo=-10, hi=10):
    """清洗近零波动导致的除零极端值"""
    if n_conv <= 0:
        return None
    try:
        v = float(v)
        if abs(v) > hi or (abs(v) < 1e-6 and v != 0):
            return None
        return v
    except Exception:
        return None


def build_summary_table(all_results):
    """构建汇总表 DataFrame（含原始指标，用于后续评分排序）"""
    rows = []
    for res in all_results:
        if 'error' in res:
            continue
        for sid, sname, fn, color in STRATEGIES:
            sres = res['strategies'].get(sid, {})
            if 'metrics' not in sres:
                continue
            m = sres['metrics']
            s = sres['stats']
            n_conv = s.get('转换次数', 0)
            rows.append({
                '基金代码': res['code'],
                '基金名称': res['name'],
                '策略': f'{sid} {sname}',
                '策略ID': sid,
                '累计收益率(%)': round(m['累计收益率(%)'], 2),
                '年化收益率(%)': round(m['年化收益率(%)'], 2),
                '最大回撤(%)': round(m['最大回撤(%)'], 2),
                '夏普比率': round(_clean_metric(m.get('夏普比率', 0), n_conv) or 0, 3),
                '卡玛比率': round(_clean_metric(m.get('卡玛比率', 0), n_conv, lo=-50, hi=50) or 0, 3),
                '转换次数': n_conv,
                '买入次数': s.get('买入触发次数', 0),
                '止盈次数': s.get('止盈触发次数', 0),
                '胜率(%)': round(s.get('胜率(%)', 0) or 0, 1),
                '超额收益(%)': round(s.get('超额收益(%)', 0) or 0, 2),
                '买入持有(%)': round(s.get('买入持有收益率(%)', 0) or 0, 2),
                '回测天数': m.get('回测天数', 0),
                '回测区间': f"{res.get('bt_start', res['data_start'])}~{res.get('bt_end', res['data_end'])}",
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # ---------- 综合评分 ----------
    # 对每个指标做百分位排名(0~1)，越高越好；回撤取绝对值后"越小越好"→ 1 - rank
    df['_年化_rank'] = df['年化收益率(%)'].rank(pct=True)
    df['_回撤_rank'] = (1 - df['最大回撤(%)'].rank(pct=True))  # 回撤越接近0越好
    df['_夏普_rank'] = df['夏普比率'].rank(pct=True)
    df['_卡玛_rank'] = df['卡玛比率'].rank(pct=True)
    df['_超额_rank'] = df['超额收益(%)'].rank(pct=True)
    df['_胜率_rank'] = df['胜率(%)'].rank(pct=True)
    # 转换次数：适度好（0太差也太差）
    df['_转换_rank'] = df['转换次数'].rank(pct=True)

    # 权重：收益25 / 回撤20 / 夏普15 / 卡玛15 / 超额15 / 胜率5 / 转换5
    weights = {'_年化_rank': 0.25, '_回撤_rank': 0.20,
               '_夏普_rank': 0.15, '_卡玛_rank': 0.15,
               '_超额_rank': 0.15, '_胜率_rank': 0.05,
               '_转换_rank': 0.05}
    df['综合评分'] = (df[list(weights.keys())] * list(weights.values())).sum(axis=1) * 100
    df['综合评分'] = df['综合评分'].round(1)
    df['排名'] = df['综合评分'].rank(ascending=False, method='min').astype(int)

    # 评级：前25%优秀，25-50%良好，50-75%一般，后25%不佳
    p25 = df['综合评分'].quantile(0.75)
    p50 = df['综合评分'].quantile(0.50)
    p75 = df['综合评分'].quantile(0.25)

    def _rate(score):
        if score >= p25:
            return '★ 优秀'
        elif score >= p50:
            return '良好'
        elif score >= p75:
            return '一般'
        else:
            return '✗ 不佳'

    df['评级'] = df['综合评分'].apply(_rate)
    # 清理内部rank列
    df = df.drop(columns=[c for c in df.columns if c.startswith('_')])
    return df


def build_evaluation_section(df):
    """生成统计评价HTML块：好/不好组合、按基金最优策略、按策略统计"""
    if df.empty:
        return '<div class="card">无有效数据</div>'

    # 下拉选项（与顶部汇总表共用 switchRange）
    opts = ''.join(f'<option value="{rk}"{" selected" if rk == DEFAULT_RANGE else ""}>{rlabel}</option>'
                   for rk, rlabel, _ in RANGES)
    html = ['<div class="card">']
    html.append('<div class="fund-title">🏆 统计评价与基金策略筛选'
                f'<select class="range-selector" onchange="switchRange(this.value)">{opts}</select></div>')
    html.append('<div class="legend-note">综合评分 = 年化收益25% + 回撤控制20% + 夏普15% + 卡玛15% + 超额收益15% + 胜率5% + 转换频率5%（百分位加权）</div>')

    # ---- 按基金：最优策略 ----
    html.append('<h3 class="sub-title">① 每只基金最优策略推荐</h3>')
    best_per_fund = df.loc[df.groupby('基金代码')['综合评分'].idxmax()].sort_values(
        '综合评分', ascending=False)
    html.append('<table><thead><tr>'
                '<th>基金代码</th><th>基金名称</th><th>推荐策略</th>'
                '<th>累计%</th><th>年化%</th><th>最大回撤%</th><th>夏普</th><th>超额%</th>'
                '<th>评分</th><th>评级</th></tr></thead><tbody>')
    for _, r in best_per_fund.iterrows():
        html.append(f'<tr class="eval-row" data-fund="{r["基金代码"]}" data-sid="{r["策略ID"]}">'
                    f'<td>{r["基金代码"]}</td>'
                    f'<td>{r["基金名称"][:16]}</td>'
                    f'<td>{r["策略"]}</td>'
                    f'<td class="pos">{r["累计收益率(%)"]}</td>'
                    f'<td class="pos">{r["年化收益率(%)"]}</td>'
                    f'<td class="neg">{r["最大回撤(%)"]}</td>'
                    f'<td>{r["夏普比率"]}</td>'
                    f'<td class="pos">{r["超额收益(%)"]}</td>'
                    f'<td><b>{r["综合评分"]}</b></td>'
                    f'<td>{r["评级"]}</td>'
                    '</tr>')
    html.append('</tbody></table>')

    # ---- TOP 10 推荐组合 ----
    html.append('<h3 class="sub-title">② 综合评分 TOP 10 基金策略组合（优先关注）</h3>')
    top10 = df.sort_values('综合评分', ascending=False).head(10)
    html.append('<table><thead><tr>'
                '<th>排名</th><th>基金代码</th><th>基金名称</th><th>策略</th>'
                '<th>累计%</th><th>年化%</th><th>最大回撤%</th><th>夏普</th><th>胜率%</th>'
                '<th>超额%</th><th>评分</th></tr></thead><tbody>')
    for i, (_, r) in enumerate(top10.iterrows(), 1):
        html.append(f'<tr class="eval-row" data-fund="{r["基金代码"]}" data-sid="{r["策略ID"]}">'
                    f'<td>{i}</td>'
                    f'<td>{r["基金代码"]}</td>'
                    f'<td>{r["基金名称"][:16]}</td>'
                    f'<td>{r["策略"]}</td>'
                    f'<td class="pos">{r["累计收益率(%)"]}</td>'
                    f'<td class="pos">{r["年化收益率(%)"]}</td>'
                    f'<td class="neg">{r["最大回撤(%)"]}</td>'
                    f'<td>{r["夏普比率"]}</td>'
                    f'<td>{r["胜率(%)"]}</td>'
                    f'<td class="pos">{r["超额收益(%)"]}</td>'
                    f'<td><b>{r["综合评分"]}</b></td>'
                    '</tr>')
    html.append('</tbody></table>')

    # ---- 需谨慎（不佳）组合 ----
    html.append('<h3 class="sub-title">③ 综合评分后 10 组合（需谨慎，避免配置）</h3>')
    bot10 = df.sort_values('综合评分', ascending=True).head(10)
    html.append('<table><thead><tr>'
                '<th>排名</th><th>基金代码</th><th>基金名称</th><th>策略</th>'
                '<th>累计%</th><th>年化%</th><th>最大回撤%</th><th>夏普</th><th>胜率%</th>'
                '<th>超额%</th><th>评分</th></tr></thead><tbody>')
    for i, (_, r) in enumerate(bot10.iterrows(), 1):
        html.append(f'<tr class="eval-row" data-fund="{r["基金代码"]}" data-sid="{r["策略ID"]}">'
                    f'<td>{i}</td>'
                    f'<td>{r["基金代码"]}</td>'
                    f'<td>{r["基金名称"][:16]}</td>'
                    f'<td>{r["策略"]}</td>'
                    f'<td class="neg">{r["累计收益率(%)"]}</td>'
                    f'<td class="neg">{r["年化收益率(%)"]}</td>'
                    f'<td class="neg">{r["最大回撤(%)"]}</td>'
                    f'<td>{r["夏普比率"]}</td>'
                    f'<td>{r["胜率(%)"]}</td>'
                    f'<td class="neg">{r["超额收益(%)"]}</td>'
                    f'<td>{r["综合评分"]}</td>'
                    '</tr>')
    html.append('</tbody></table>')

    # ---- 多策略横向统计 ----
    html.append('<h3 class="sub-title">④ 各策略横向统计（跨基金）</h3>')
    html.append('<table><thead><tr>'
                '<th>策略</th><th>基金数</th>'
                '<th>平均累计%</th><th>平均年化%</th><th>平均回撤%</th>'
                '<th>平均夏普</th><th>平均超额%</th>'
                '<th>正收益占比</th><th>超额正占比</th>'
                '</tr></thead><tbody>')
    for sid, sname, _, _ in STRATEGIES:
        sub = df[df['策略ID'] == sid]
        if sub.empty:
            continue
        html.append('<tr>'
                    f'<td>{sid} {sname}</td>'
                    f'<td>{len(sub)}</td>'
                    f'<td>{sub["累计收益率(%)"].mean():.2f}</td>'
                    f'<td>{sub["年化收益率(%)"].mean():.2f}</td>'
                    f'<td class="neg">{sub["最大回撤(%)"].mean():.2f}</td>'
                    f'<td>{sub["夏普比率"].mean():.3f}</td>'
                    f'<td>{sub["超额收益(%)"].mean():.2f}</td>'
                    f'<td>{(sub["累计收益率(%)"] > 0).mean() * 100:.0f}%</td>'
                    f'<td>{(sub["超额收益(%)"] > 0).mean() * 100:.0f}%</td>'
                    '</tr>')
    html.append('</tbody></table>')

    # ---- 评级分布 ----
    html.append('<h3 class="sub-title">⑤ 评级分布</h3>')
    dist = df['评级'].value_counts()
    for tag in ['★ 优秀', '良好', '一般', '✗ 不佳']:
        cnt = int(dist.get(tag, 0))
        pct = cnt / len(df) * 100 if len(df) else 0
        html.append(f'<div class="dist"><span class="tag">{tag}</span> '
                    f'{cnt} 个组合 ({pct:.0f}%)</div>')

    html.append('</div>')
    return '\n'.join(html)


def build_sortable_table_html(df, table_id='summary-table'):
    """生成可点击表头排序的汇总表"""
    if df.empty:
        return '<div class="card">无有效数据</div>'

    cols = ['基金代码', '基金名称', '策略', '累计收益率(%)', '年化收益率(%)',
            '最大回撤(%)', '夏普比率', '卡玛比率', '转换次数', '买入次数',
            '止盈次数', '胜率(%)', '超额收益(%)', '买入持有(%)',
            '综合评分', '评级', '回测区间']
    cols = [c for c in cols if c in df.columns]

    html = [f'<table id="{table_id}" class="sortable">',
            '<thead><tr>']
    for c in cols:
        html.append(f'<th data-sortable="true">{c}</th>')
    html.append('</tr></thead><tbody>')

    for _, r in df.iterrows():
        fcode = r['基金代码']
        sid = r['策略ID']
        html.append(f'<tr data-fund="{fcode}" data-sid="{sid}">')
        for c in cols:
            v = r[c]
            cls = ''
            if c in ('累计收益率(%)', '年化收益率(%)', '超额收益(%)', '买入持有(%)'):
                try:
                    cls = 'pos' if float(v) > 0 else ('neg' if float(v) < 0 else '')
                except Exception:
                    pass
            elif c == '最大回撤(%)':
                cls = 'neg'
            elif c == '评级':
                if '优秀' in str(v):
                    cls = 'rate-excellent'
                elif '良好' in str(v):
                    cls = 'rate-good'
                elif '一般' in str(v):
                    cls = 'rate-normal'
                elif '不佳' in str(v):
                    cls = 'rate-bad'
            elif c == '综合评分':
                cls = 'score'
            html.append(f'<td class="{cls}" data-val="{v}">{v}</td>')
        html.append('</tr>')
    html.append('</tbody></table>')
    return '\n'.join(html)


def _role_label(role):
    """条件单角色中文标签"""
    mapping = {
        'core': '底仓',
        'float': '浮动',
        'buy': '买入',
        'sell': '止盈',
        'trail_stop': '止损',
        'rotate': '轮动',
    }
    return mapping.get(role, role or '')


def _build_strategy_help_html(sid, eng, st):
    """拼策略简介 + 关键参数（静态说明 + 运行时动态值）"""
    help_info = STRATEGY_HELP.get(sid, {})
    intro = help_info.get('intro', '')
    keys = list(help_info.get('keys', []))
    meta = getattr(eng, 'gf_meta', None) or {}

    # 动态补充
    if sid == 'D':
        cr = meta.get('core_ratio', 0.40)
        fr = meta.get('float_ratio', 0.60)
        keys.insert(0, f'本次回测仓位：底仓 {cr:.0%} / 浮动 {fr:.0%}')
        if meta.get('nv_q20') is not None:
            keys.append(f'极端参考：Q20={_fmt_nv(meta["nv_q20"])}（加仓底仓） / Q90={_fmt_nv(meta["nv_q90"])}（只卖浮动）')
        if meta.get('core_share_total') is not None:
            keys.append(f'目标份额：底仓合计 {meta.get("core_share_total")} / 浮动合计 {meta.get("float_share_total")}')
        if st.get('底仓份额') is not None:
            keys.append(
                f'回测成交：底仓买入 {st.get("底仓份额")} · 浮动买入 {st.get("浮动买入份额")} · '
                f'浮动卖出 {st.get("浮动卖出份额")}'
            )
        if st.get('短期惩罚费(元)'):
            keys.append(f'短期惩罚费合计 {st.get("短期惩罚费(元)")} 元')
    elif sid == 'E':
        zone = getattr(eng, 'gf_zone', None) or st.get('当前区间')
        pct = getattr(eng, 'gf_percentile', None)
        if pct is None and st.get('当前分位') is not None:
            pct = st.get('当前分位')
        zone_label = {'under': '低估·只买', 'mid': '震荡·网格', 'over': '高估·只卖'}.get(zone, zone or '—')
        keys.insert(0, f'当前区间：{zone_label}' + (f'（分位 {float(pct)*100:.1f}%）' if pct is not None else ''))
        if st.get('短期惩罚费(元)'):
            keys.append(f'短期惩罚费合计 {st.get("短期惩罚费(元)")} 元')
    elif sid == 'G':
        level = getattr(eng, 'gf_active_level', None)
        if level is None and st.get('当前止损档') is not None:
            level = int(st.get('当前止损档', 0)) - 1
        ladder = getattr(eng, 'gf_ladder', None) or []
        if level is not None and level >= 0:
            keys.insert(0, f'当前有效止损档：第 {level + 1} 档（共 {len(ladder) or "?"} 档台阶）')
        else:
            keys.insert(0, f'当前尚未激活止损档（台阶数 {len(ladder) or "—"}）')
        if st.get('短期惩罚费(元)'):
            keys.append(f'短期惩罚费合计 {st.get("短期惩罚费(元)")} 元')

    parts = ['<div class="strategy-help">']
    if intro:
        parts.append(f'<div class="help-intro">{intro}</div>')
    if keys:
        parts.append('<ul class="param-meta">')
        for k in keys:
            parts.append(f'<li>{k}</li>')
        parts.append('</ul>')
    parts.append('</div>')
    return '\n'.join(parts)


def build_params_map(valid_results):
    """构建参数查找映射：code_sid -> 参数HTML（供点击表格行弹出模态框）"""
    params_map = {}
    for res in valid_results:
        for sid, sname, fn, color in STRATEGIES:
            sres = res['strategies'].get(sid, {})
            key = f"{res['code']}_{sid}"
            eng = sres.get('engine')
            buy_list = getattr(eng, 'gf_buy_list', None) or [] if eng else []
            sell_list = getattr(eng, 'gf_sell_list', None) or [] if eng else []
            m = sres.get('metrics', {})
            st = sres.get('stats', {})

            box = [f'<h3 style="margin:0 0 12px;color:{color}">{res["name"]}（{res["code"]}）'
                   f' · 策略{sid} {sname}</h3>']
            box.append(_build_strategy_help_html(sid, eng, st))

            box.append('<table class="param-detail-table" style="font-size:13px;margin-bottom:12px">'
                       '<thead><tr><th>方向</th><th>序号</th><th>约定净值</th>'
                       '<th>转换份额</th><th>角色</th><th>备注</th></tr></thead><tbody>')
            for i, b in enumerate(buy_list, 1):
                role = _role_label(b.get('role', ''))
                note = b.get('note', '') or ''
                box.append(
                    f'<tr><td style="color:#1f77b4">买入</td><td>{i}</td>'
                    f'<td>{_fmt_nv(b["trigger_net_value"])}</td><td>{b["share"]}</td>'
                    f'<td>{role}</td><td class="note-cell">{note}</td></tr>'
                )
            for i, s in enumerate(sell_list, 1):
                role = _role_label(s.get('role', ''))
                note = s.get('note', '') or ''
                box.append(
                    f'<tr><td style="color:#d6372f">止盈</td><td>{i}</td>'
                    f'<td>{_fmt_nv(s["trigger_net_value"])}</td><td>{s["share"]}</td>'
                    f'<td>{role}</td><td class="note-cell">{note}</td></tr>'
                )
            if not buy_list and not sell_list:
                box.append('<tr><td colspan="6" style="color:#999">暂无约定净值参数</td></tr>')
            box.append('</tbody></table>')

            box.append(f'<div style="font-size:12.5px;color:#555">'
                       f'累计收益率 <b style="color:#d6372f">{m.get("累计收益率(%)",0):.2f}%</b> · '
                       f'年化 {m.get("年化收益率(%)",0):.2f}% · '
                       f'最大回撤 <b style="color:#2e8b57">{m.get("最大回撤(%)",0):.2f}%</b> · '
                       f'夏普 {m.get("夏普比率",0):.3f} · '
                       f'转换 {st.get("转换次数",0)} 次</div>')
            params_map[key] = '\n'.join(box)
    return params_map


# 排序脚本（点击表头升/降序，适用于所有 .sortable 表）
SORT_SCRIPT = """
<script>
(function(){
  document.querySelectorAll('table.sortable').forEach(function(table){
    var headers = table.querySelectorAll('th[data-sortable]');
    headers.forEach(function(th, idx){
      th.style.cursor='pointer';
      th.addEventListener('click', function(){
        var tbody = table.querySelector('tbody');
        var rows = Array.from(tbody.querySelectorAll('tr'));
        var asc = th.dataset.sortDir !== 'asc';
        headers.forEach(function(h){ h.dataset.sortDir=''; h.textContent = h.textContent.replace(' ▲','').replace(' ▼',''); });
        th.dataset.sortDir = asc ? 'asc' : 'desc';
        rows.sort(function(a,b){
          var va = a.cells[idx].getAttribute('data-val');
          var vb = b.cells[idx].getAttribute('data-val');
          var na = parseFloat(va), nb = parseFloat(vb);
          var aNum = !isNaN(na), bNum = !isNaN(nb);
          var cmp;
          if(aNum && bNum){ cmp = na - nb; }
          else { cmp = String(va).localeCompare(String(vb),'zh'); }
          return asc ? cmp : -cmp;
        });
        rows.forEach(function(r){ tbody.appendChild(r); });
        th.textContent = th.textContent.replace(' ▲','').replace(' ▼','') + (asc ? ' ▲' : ' ▼');
      });
    });
  });
})();
</script>
"""

ROW_CLICK_SCRIPT = """
<script>
(function(){
  var CLICK_DELAY = 280; // 单击延迟，若此时间内有双击则取消单击
  function bindRow(tr){
    var clickTimer = null;
    tr.addEventListener('click', function(e){
      if(!tr.dataset || !tr.dataset.fund) return;
      if(clickTimer) return;
      clickTimer = setTimeout(function(){
        clickTimer = null;
        var key = tr.dataset.fund + '_' + tr.dataset.sid;
        if(window.openParamModal){ window.openParamModal(key); }
      }, CLICK_DELAY);
    });
    tr.addEventListener('dblclick', function(e){
      if(!tr.dataset || !tr.dataset.fund) return;
      if(clickTimer){ clearTimeout(clickTimer); clickTimer = null; }
      var target = document.getElementById('fund_' + tr.dataset.fund);
      if(target){
        target.scrollIntoView({behavior:'smooth', block:'start'});
        target.style.transition = 'background .6s';
        target.style.background = '#fff7e6';
        setTimeout(function(){ target.style.background = ''; }, 1500);
      }
    });
  }
  // 绑定所有区间表 + 评价表行
  document.querySelectorAll('.range-table tbody tr').forEach(bindRow);
  document.querySelectorAll('.eval-row').forEach(bindRow);
})();
</script>
"""

# 温和滚轮缩放：默认 pan，滚轮每格约 ±8%（替代 Plotly 默认偏大步长）
ZOOM_SCRIPT = """
<script>
(function(){
  var ZOOM_FACTOR = 1.08;
  function toMs(v){
    if(v == null) return null;
    if(typeof v === 'number') return v;
    var t = Date.parse(v);
    return isNaN(t) ? null : t;
  }
  function axisRange(gd, axName){
    var ax = gd._fullLayout && gd._fullLayout[axName];
    if(!ax || !ax.range) return null;
    var r0 = toMs(ax.range[0]), r1 = toMs(ax.range[1]);
    if(r0 == null || r1 == null) return null;
    return [r0, r1];
  }
  function softWheelZoom(gd, e){
    if(!window.Plotly || !gd || !gd._fullLayout) return;
    e.preventDefault();
    e.stopImmediatePropagation();
    var factor = e.deltaY < 0 ? 1 / ZOOM_FACTOR : ZOOM_FACTOR;
    var xNames = Object.keys(gd._fullLayout).filter(function(k){
      return /^xaxis\\d*$/.test(k) && gd._fullLayout[k] && gd._fullLayout[k].range;
    });
    if(!xNames.length) return;
    var update = {};
    xNames.forEach(function(name){
      var r = axisRange(gd, name);
      if(!r) return;
      var mid = (r[0] + r[1]) / 2;
      var half = (r[1] - r[0]) / 2 * factor;
      update[name + '.range'] = [new Date(mid - half), new Date(mid + half)];
    });
    if(Object.keys(update).length) Plotly.relayout(gd, update);
  }
  function bindPlot(gd){
    if(!gd || gd._softZoomBound) return;
    gd._softZoomBound = true;
    if(window.Plotly) Plotly.relayout(gd, {dragmode: 'pan'});
    gd.addEventListener('wheel', function(e){ softWheelZoom(gd, e); }, {passive: false, capture: true});
  }
  function bindAll(){
    document.querySelectorAll('.js-plotly-plot').forEach(bindPlot);
  }
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', bindAll);
  } else {
    bindAll();
  }
  setTimeout(bindAll, 500);
  setTimeout(bindAll, 1500);
})();
</script>
"""

# 手机端：收紧 Plotly 边距；折线高度保持桌面原值（无 viewport 时用 screen 检测）
MOBILE_ADAPT_SCRIPT = """
<script>
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
  function run(){
    document.querySelectorAll('.js-plotly-plot').forEach(adaptPlot);
  }
  var timer;
  function schedule(){clearTimeout(timer);timer=setTimeout(run,200);}
  if(document.readyState==='complete')schedule();
  else window.addEventListener('load',schedule);
  window.addEventListener('resize',schedule);
  setTimeout(schedule,800);
  setTimeout(schedule,1800);
})();
</script>
"""


def main():
    print("╔" + "═" * 66 + "╗")
    print("║" + " " * 8 + "广发约定净值转换 · 多基金多策略可视化" + " " * 20 + "║")
    print("╚" + "═" * 66 + "╝\n")

    funds = parse_fund_list(FUND_LIST_FILE)
    range_label_map = {rk: rlabel for rk, rlabel, _ in RANGES}
    print(f"读取基金清单: {len(funds)} 只")
    print(f"回测区间: {', '.join(r[1] for r in RANGES)}（默认{range_label_map[DEFAULT_RANGE]}）\n")

    # 多区间结果: {range_key: [results]} 与 {code: {range_key: result}}
    all_results_by_range = {rk: [] for rk, _, _ in RANGES}
    fund_results_map = {}  # code -> {range_key: result}
    n_real, n_sim = 0, 0

    for i, (code, name) in enumerate(funds, 1):
        print(f"  [{i}/{len(funds)}] {code} {name[:20]}", end=' ', flush=True)
        try:
            nav_data = load_otc_fund_nav(code, name,
                                         start_date='2019-01-01', end_date='2026-12-31',
                                         verbose=False)
            src = nav_data.attrs.get('nav_source', 'unknown')
            if src == 'real':
                n_real += 1
                print('[真实]', end=' ', flush=True)
            elif src == 'sim':
                n_sim += 1
                print('[模拟]', end=' ', flush=True)
            fund_results_map[code] = {}
            if len(nav_data) < MIN_DATA_LEN:
                print(f"数据不足({len(nav_data)}条)")
                for rk, _, _ in RANGES:
                    res = {'code': code, 'name': name, 'error': f'数据不足({len(nav_data)}条)',
                           'nav_data': nav_data, 'strategies': {}}
                    all_results_by_range[rk].append(res)
                    fund_results_map[code][rk] = res
                continue

            parts = []
            for rk, rlabel, years in RANGES:
                sd = compute_start_date(years) if years else None
                res = run_all_for_fund(code, name, start_date=sd, nav_data=nav_data)
                all_results_by_range[rk].append(res)
                fund_results_map[code][rk] = res
                if 'error' in res:
                    parts.append(f"{rlabel}:err")
                else:
                    bt = res.get('bt_start', '?')
                    parts.append(f"{rlabel}:{bt[2:]}")  # YY-MM-DD
            print(' '.join(parts))
        except Exception as e:
            print(f"异常: {e}")
            if code not in fund_results_map:
                fund_results_map[code] = {}
            for rk, _, _ in RANGES:
                res = {'code': code, 'name': name, 'error': str(e),
                       'nav_data': pd.DataFrame(), 'strategies': {}}
                all_results_by_range[rk].append(res)
                fund_results_map[code][rk] = res

    print(f"\n净值来源统计: 真实 {n_real} / 模拟 {n_sim}")
    if n_real == 0 and (n_real + n_sim) > 0:
        print("错误: 全部基金使用模拟净值，中止生成以免污染 gh-pages 部署")
        sys.exit(1)

    # ---- 各区间汇总表 ----
    summary_dfs = {rk: build_summary_table(results)
                   for rk, results in all_results_by_range.items()}

    # CSV：导出默认区间数据
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'gf_multi_fund_comparison.csv')
    summary_dfs[DEFAULT_RANGE].to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 汇总表({range_label_map[DEFAULT_RANGE]}): {csv_path}")

    # ---- HTML 报告 ----
    print("生成 HTML 可视化报告...")
    # CDN 版本必须与本机 plotly.py 捆绑的 plotly.js 一致（否则 bdata 曲线无法解码）
    plotly_cdn = (
        f'<script charset="utf-8" '
        f'src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js">'
        f'</script>'
    )

    html_parts = []
    html_parts.append("""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>广发约定净值转换 · 多基金多策略对比</title>
""" + plotly_cdn + """
<style>
body{font-family:'Microsoft YaHei',Arial,sans-serif;margin:0;background:#f5f6f8;color:#222}
.container{max-width:1180px;margin:0 auto;padding:24px}
h1{font-size:24px;margin:0 0 6px}
.sub{color:#666;font-size:13px;margin-bottom:20px}
.sub-title{font-size:15px;font-weight:600;margin:18px 0 8px;color:#333}
.card{background:#fff;border-radius:10px;padding:18px 22px;margin-bottom:22px;
 box-shadow:0 1px 4px rgba(0,0,0,.06)}
.fund-title{font-size:17px;font-weight:600;margin:0 0 4px;border-left:4px solid #1f77b4;padding-left:10px}
.tag{display:inline-block;background:#eef3fb;color:#1f5fa8;border-radius:4px;
 padding:2px 8px;font-size:12px;margin-left:8px}
.range-selector{float:right;font-size:13px;padding:3px 8px;border:1px solid #ccc;
 border-radius:4px;cursor:pointer;margin-top:1px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border:1px solid #e3e6ea;padding:6px 8px;text-align:center}
th{background:#f0f3f7;font-weight:600;user-select:none}
table.sortable th[data-sortable]{cursor:pointer}
table.sortable th[data-sortable]:hover{background:#e0e8f3}
tr:nth-child(even){background:#fafbfc}
.pos{color:#d6372f;font-weight:600}.neg{color:#2e8b57}
.score{background:#fff7e6;font-weight:700;color:#d48806}
.rate-excellent{background:#f6ffed;color:#389e0d;font-weight:700}
.rate-good{background:#e6f7ff;color:#096dd9;font-weight:600}
.rate-normal{background:#fffbe6;color:#d48806}
.rate-bad{background:#fff1f0;color:#cf1322;font-weight:600}
.dist{margin:4px 0}
.legend-note{color:#888;font-size:12px;margin-top:8px}
.params-block{display:flex;gap:8px;margin:4px 0 2px;flex-wrap:wrap}
.param-box{flex:1;min-width:230px;background:#f8f9fb;border:1px solid #e3e6ea;
 border-radius:5px;padding:5px 10px;font-size:11.5px;line-height:1.5}
.param-sid{font-size:12px;margin-right:3px}
.param-row{margin-top:1px}
.param-label{color:#666;margin-right:3px}
.param-nv{color:#1f77b4;font-weight:600}
.param-row.sell .param-nv{color:#d6372f}
.param-share{color:#888;margin-left:5px}
.range-table tbody tr{cursor:pointer}
.range-table tbody tr:hover{background:#fff7e6}
.eval-row{cursor:pointer}
.eval-row:hover{background:#fff7e6}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.5);
 z-index:9999;align-items:center;justify-content:center}
.modal-overlay.active{display:flex}
.modal-box{background:#fff;border-radius:10px;padding:24px;max-width:780px;
 width:90%;max-height:80vh;overflow:auto;position:relative}
.modal-close{position:absolute;top:10px;right:14px;cursor:pointer;font-size:22px;color:#999}
.strategy-help{background:#f5f7fa;border:1px solid #e4e8ee;border-radius:8px;
 padding:10px 12px;margin:0 0 14px;font-size:12.5px;line-height:1.55;color:#444}
.strategy-help .help-intro{margin:0 0 6px;color:#333}
.strategy-help .param-meta{margin:0;padding-left:18px}
.strategy-help .param-meta li{margin:2px 0}
.param-detail-table th,.param-detail-table td{padding:4px 8px;text-align:left}
.param-detail-table .note-cell{color:#666;font-size:12px;max-width:220px}
@media (max-width:768px){
body{margin:0!important;padding:8px!important}
.container{padding:10px 8px!important}
.header{padding:16px 14px!important}
.header h1,h1{font-size:18px!important}
.card,.section{padding:12px 10px!important}
.param-box{min-width:0!important}
.range-selector{float:none!important;display:block;margin:8px 0 0!important;width:100%;box-sizing:border-box}
.cards{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))!important;gap:10px!important}
.card .num{font-size:22px!important}
table{font-size:11px!important}
.card,.section,.container{overflow-x:auto;-webkit-overflow-scrolling:touch}
.js-plotly-plot,.plotly-graph-div{max-width:100%!important}
}
</style></head><body><div class="container">
<h1>广发约定净值转换 · 多基金多策略可视化对比</h1>
<div class="sub">A 动态滚动 / B 估值分位 / C 高点回撤 / D 底仓+浮动 / E 分位双区 / G 阶梯止盈 · 初始资金100万 · 天天红B年化2% · 点击表格行查看策略说明与参数</div>
<div class="card"><div class="fund-title">📊 多基金 × 多策略 汇总对比（点击表头可升/降序排序）
<select class="range-selector" id="rangeSelector" onchange="switchRange(this.value)">""")

    # 下拉选项（默认近3年）
    for rk, rlabel, _ in RANGES:
        sel = ' selected' if rk == DEFAULT_RANGE else ''
        html_parts.append(f'<option value="{rk}"{sel}>{rlabel}</option>')
    html_parts.append('</select></div>')

    # 各区间汇总表（默认区间可见，其余隐藏）
    for rk, rlabel, _ in RANGES:
        style = '' if rk == DEFAULT_RANGE else ' style="display:none"'
        html_parts.append(f'<div id="table-{rk}" class="range-table"{style}>')
        html_parts.append(build_sortable_table_html(summary_dfs[rk], table_id=f'tbl-{rk}'))
        html_parts.append('</div>')

    html_parts.append('<div class="legend-note">▲ 买入（彩色实角） / ▼ 止盈（彩色半透明） · 三种策略颜色：A蓝 B橙 C绿 · 切换区间仅刷新表格与参数，曲线为全部区间</div>')
    html_parts.append('</div>')

    # ---- 统计评价区块（4区间，下拉切换，默认近3年）----
    for rk, rlabel, _ in RANGES:
        style = '' if rk == DEFAULT_RANGE else ' style="display:none"'
        html_parts.append(f'<div class="range-eval" data-range="{rk}"{style}>')
        html_parts.append(build_evaluation_section(summary_dfs[rk]))
        html_parts.append('</div>')

    # ---- 各基金图表（曲线使用全部区间）----
    valid_all = [r for r in all_results_by_range['all'] if 'error' not in r]
    html_parts.append(f'<div class="card"><div class="fund-title">📈 各基金净值曲线与买卖点（{len(valid_all)}只有数据 · 曲线为全部区间）</div>')

    # 各区间参数映射（点击表格行弹出模态框，根据当前区间显示对应参数）
    params_maps = {}
    for rk, _, _ in RANGES:
        valid_r = [r for r in all_results_by_range[rk] if 'error' not in r]
        params_maps[rk] = build_params_map(valid_r)

    import json
    params_json = json.dumps(params_maps, ensure_ascii=False)

    for res in valid_all:
        code = res['code']
        fig = build_fund_figure(res)
        div_id = f"fund_{code}"
        html_parts.append(f'<div class="card" id="{div_id}">')
        html_parts.append(f'<div class="fund-title">{res["name"]}'
                          f'<span class="tag">{res["code"]}</span>'
                          f'<span class="tag">{res["data_len"]}条 {res["data_start"]}~{res["data_end"]}</span></div>')
        # 各区间参数块（默认区间可见，切换区间时前端显隐）
        for rk, rlabel, _ in RANGES:
            res_r = fund_results_map.get(code, {}).get(rk, res)
            style = '' if rk == DEFAULT_RANGE else ' style="display:none"'
            html_parts.append(f'<div class="range-params" data-range="{rk}"{style}>')
            html_parts.append(f'<div class="legend-note" style="margin:0 0 2px">参数区间：{rlabel}'
                              f'（{res_r.get("bt_start","?")}~{res_r.get("bt_end","?")}）</div>')
            html_parts.append(build_fund_params_block(res_r))
            html_parts.append('</div>')
        html_parts.append(pio.to_html(fig, include_plotlyjs=False, full_html=False,
                                      config={'displaylogo': False, 'scrollZoom': True}))
        html_parts.append('</div>')
    html_parts.append('</div>')

    # ---- 模态框 + 区间切换脚本 ----
    range_opts_js = ','.join(f"'{rk}':'{rlabel}'" for rk, rlabel, _ in RANGES)
    html_parts.append("""<div class="modal-overlay" id="paramModal">
<div class="modal-box"><span class="modal-close" onclick="closeParamModal()">×</span>
<div id="paramModalContent">点击表格行查看参数</div></div></div>
<script>
window.PARAMS_MAP = """ + params_json + """;
window.CURRENT_RANGE = '""" + DEFAULT_RANGE + """';
var RANGE_LABELS = {""" + range_opts_js + """};
function openParamModal(key){
  var map = window.PARAMS_MAP[window.CURRENT_RANGE] || {};
  var h = map[key];
  var c = document.getElementById('paramModalContent');
  if(h){ c.innerHTML = '<div style="font-size:12px;color:#888;margin-bottom:8px">参数区间：'
    + (RANGE_LABELS[window.CURRENT_RANGE]||window.CURRENT_RANGE) + '</div>' + h; }
  else { c.innerHTML = '暂无参数'; }
  document.getElementById('paramModal').classList.add('active');
}
function closeParamModal(){document.getElementById('paramModal').classList.remove('active');}
function switchRange(rk){
  window.CURRENT_RANGE = rk;
  // 切换汇总表显隐
  document.querySelectorAll('.range-table').forEach(function(t){ t.style.display='none'; });
  var tt = document.getElementById('table-' + rk);
  if(tt) tt.style.display = '';
  // 切换统计评价区块显隐
  document.querySelectorAll('.range-eval').forEach(function(e){ e.style.display='none'; });
  document.querySelectorAll('.range-eval[data-range="'+rk+'"]').forEach(function(e){ e.style.display=''; });
  // 切换各基金参数块显隐
  document.querySelectorAll('.range-params').forEach(function(p){ p.style.display='none'; });
  document.querySelectorAll('.range-params[data-range="'+rk+'"]').forEach(function(p){ p.style.display=''; });
  // 同步所有下拉
  document.querySelectorAll('.range-selector').forEach(function(s){ s.value = rk; });
}
document.getElementById('paramModal').addEventListener('click',function(e){
 if(e.target.id==='paramModal') closeParamModal();});
document.addEventListener('keydown',function(e){if(e.key==='Escape')closeParamModal();});
</script>
""")

    html_parts.append("""<div class="card" style="color:#999;font-size:12px">
    ⚠ 以上为历史回测结果，不构成投资建议。实际录入广发系统前请结合当前市场环境判断。</div>
</div>""" + SORT_SCRIPT + ROW_CLICK_SCRIPT + ZOOM_SCRIPT + MOBILE_ADAPT_SCRIPT + """</body></html>""")

    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'gf_strategy_dashboard.html')
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))
    print(f"💾 HTML报告: {html_path}")

    # ---- 终端摘要（基于全部区间）----
    print("\n" + "=" * 70)
    print("  完成摘要（全部区间）")
    print("=" * 70)
    all_df = summary_dfs['all']
    if len(all_df):
        for sid, sname, _, _ in STRATEGIES:
            sub = all_df[all_df['策略ID'] == sid]
            if len(sub):
                print(f"  策略{sid} {sname}: 平均累计{sub['累计收益率(%)'].mean():.2f}%  "
                      f"平均回撤{sub['最大回撤(%)'].mean():.2f}%  "
                      f"最佳{sub.loc[sub['累计收益率(%)'].idxmax(),'基金名称']}"
                      f"({sub['累计收益率(%)'].max():.2f}%)")
    print("=" * 70)


if __name__ == '__main__':
    main()
