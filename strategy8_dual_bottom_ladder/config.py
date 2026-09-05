# -*- coding: utf-8 -*-
"""策略8 参数（全文件唯一真相源）"""
import os

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PKG)
CACHE_DIR = os.path.join(ROOT, '.data_cache', 'strategy8')

RANK_TYPES = ['股票型', '混合型', '指数型']
SORT_COL = '近1年'
SORT_ASCENDING = True
DEEP_TOP_N = 500
RANK_CACHE_HOURS = 24
EXCLUDE_KEYWORDS = ['债', '货币', '理财', '短债', '中短债', '纯债', '固收']
PREFER_CLASS = 'C'

PRICE_PCT_3Y = 0.15
PRICE_MDD_3Y = -0.40
PRICE_MDD_1Y = -0.30
PRICE_PCT_WATCH = 0.30

VAL_PE_MAX = 0.30
VAL_LONG_PCT = 0.30
VAL_DD_PEAK = 0.25
VAL_MA_DEV = -0.10
VAL_MIN_HITS = 2

MIN_YEARS = 3.0
MIN_YEARS_HARD = 2.0
JUMP_TH = 0.10
JUMP_MAX = 3
SCALE_MIN = 2.0
SCALE_MAX = 150.0
SCALE_MAX_IDX = 500.0
MGR_MIN_YEARS = 2.0
NEWLOW_TRAP = 0.45
VOL_EXTREME = 0.50
VOL_SURGE = 2.0
MDD_BLOWUP = 0.80
ZOMBIE_ANN = -0.20
ZOMBIE_NL = 0.35
ZOMBIE_VOL = 0.08
ZOMBIE_SOFT = -0.05

LADDER_K = 0.5
SPACING_MIN = 0.04
SPACING_MAX = 0.20
WEIGHTS = [0.25, 0.35, 0.40]
TP_GAINS = [0.08, 0.16, 0.28]
TP_RATIOS = [1 / 3, 1 / 3, 1.0]
STOP_K = 2.5
STOP_MIN = 0.12
STOP_MAX = 0.35
TRAIL_K = 1.2
TRAIL_MIN = 0.06
TRAIL_MAX = 0.18
VALID_DAYS = 180
POS_MAX = 0.15
SECTOR_CAP = 0.30

NAV_START = '2019-01-01'
NAV_END = '2026-12-31'
REBAL_DAYS = 21
WARMUP = 756
COOLDOWN = 63
CAPITAL = 100_000
BT_MAX = 20

POOL_FILE = os.path.join(PKG, 'fund_pool.txt')
OUT_RANK = os.path.join(PKG, 'fund_rank_board.csv')
OUT_SCREEN = os.path.join(PKG, 'screening_results.csv')
OUT_ORDERS = os.path.join(PKG, 'conditional_orders.csv')
OUT_BT = os.path.join(PKG, 'ladder_backtest.csv')
OUT_EQ = os.path.join(PKG, 'ladder_backtest_equity.csv')
OUT_MON = os.path.join(PKG, 'monitor_alerts.csv')
OUT_HTML = os.path.join(ROOT, 'dual_bottom_ladder_dashboard.html')
