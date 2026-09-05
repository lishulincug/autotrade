"""
广发约定净值转换 · 多标的资金轮动回测入口
============================================
默认标的：中证红利C / 港股互联网C / 北证50C
共用天天红B现金池，按净值分位周度轮动。

用法：
    python run_gf_rotation.py

输出：
    strategy6_gf_nav_conversion/gf_rotation_allocation.csv
    strategy6_gf_nav_conversion/gf_rotation_equity.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy6_gf_nav_conversion.strategy_f_rotation import run_rotation


if __name__ == '__main__':
    run_rotation(verbose=True)
