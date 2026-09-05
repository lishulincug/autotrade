"""
数据加载模块 - 支持真实数据（东方财富公开API）与模拟数据（fallback）
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json
import time

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# 数据缓存目录
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.data_cache')
USE_REAL_DATA = True  # 默认尝试拉取真实数据，失败则回退模拟


def _get_secid(code: str) -> str:
    """根据ETF代码推导东财secid：沪(51开头/510/518等)=1.xxx；深(159/15开头)=0.xxx"""
    code = str(code).split('.')[0]
    if code.startswith(('5', '6', '9')):
        return f'1.{code}'
    else:
        return f'0.{code}'


def fetch_real_etf_kline(code: str, name: str = '',
                         start_date: str = '2020-01-01',
                         end_date: str = '2026-07-31',
                         use_cache: bool = True,
                         timeout: int = 8) -> pd.DataFrame:
    """
    从东方财富公开接口拉取ETF/指数的日K线（OHLCV）
    失败则返回空DataFrame
    """
    cache_path = os.path.join(CACHE_DIR, f'etf_{code}_{start_date}_{end_date}.csv')
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 30:
                return df
        except Exception:
            pass

    if not HAS_REQUESTS:
        return pd.DataFrame()

    os.makedirs(CACHE_DIR, exist_ok=True)
    secid = _get_secid(code)

    # 格式 YYYYMMDD
    s = start_date.replace('-', '')
    e = end_date.replace('-', '')

    url = (
        'http://push2his.eastmoney.com/api/qt/stock/kline/get'
        '?secid={secid}'
        '&fields1=f1,f2,f3,f4,f5,f6'
        '&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61'
        '&klt=101&fqt=1'  # klt=101 日K，fqt=1 前复权
        '&beg={s}&end={e}'
    ).format(secid=secid, s=s, e=e)

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://quote.eastmoney.com/',
    }

    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        data = r.json()
        if not data.get('data') or not data['data'].get('klines'):
            return pd.DataFrame()

        rows = []
        for line in data['data']['klines']:
            # f51=日期, f52=开, f53=收, f54=高, f55=低, f56=量, f57=额, f58=振幅, f59=涨跌幅, f60=涨跌额, f61=换手率
            parts = line.split(',')
            if len(parts) < 6:
                continue
            d = parts[0]
            o, c, h, l = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            v = float(parts[5])
            rows.append({
                'open': o, 'high': h, 'low': l, 'close': c,
                'volume': int(v),
                'date': pd.Timestamp(d)
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).set_index('date').sort_index()
        df['etf_name'] = name or code

        if use_cache:
            df.to_csv(cache_path)
        return df

    except Exception as exc:
        # print(f'  [拉取失败] {code}: {exc}')
        return pd.DataFrame()


def generate_etf_data(etf_name, start_date='2020-01-01', end_date='2026-07-31',
                      base_price=1.0, annual_return=0.1, volatility=0.25, seed=None):
    """
    生成单只ETF的模拟历史OHLCV数据（真实数据失败时回退用）
    """
    if seed is not None:
        np.random.seed(seed)

    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    daily_return = annual_return / 252
    daily_vol = volatility / np.sqrt(252)

    returns = np.random.normal(daily_return, daily_vol, n_days)
    for i in range(1, n_days):
        returns[i] += 0.05 * (daily_return - returns[i-1])

    prices = base_price * np.cumprod(1 + returns)

    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, n_days)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        'close': prices,
        'volume': np.random.randint(10_000_000, 500_000_000, n_days)
    }, index=dates)

    data['high'] = data[['high', 'open', 'close']].max(axis=1)
    data['low'] = data[['low', 'open', 'close']].min(axis=1)
    data['etf_name'] = etf_name
    return data


def _get_etf_smart(code: str, info: dict, start_date='2020-01-01',
                   end_date='2026-07-31', verbose=False) -> pd.DataFrame:
    """智能取数：优先真实（含重试）-> 失败用模拟"""
    name = info.get('name', code)
    if USE_REAL_DATA:
        # 最多重试2次，每次间隔1秒
        for attempt in range(3):
            df_real = fetch_real_etf_kline(code, name, start_date, end_date)
            if len(df_real) > 60:
                if verbose:
                    print(f'  [真实] {code} {name}: {len(df_real)}条K线 {df_real.index[0].date()}~{df_real.index[-1].date()}')
                return df_real
            time.sleep(0.8)  # 限速等待后重试

    df_sim = generate_etf_data(
        name, start_date=start_date, end_date=end_date,
        base_price=info.get('base', 1.0),
        annual_return=info.get('ret', 0.1),
        volatility=info.get('vol', 0.25),
        seed=info.get('seed', None)
    )
    if verbose:
        print(f'  [模拟] {code} {name}')
    return df_sim


def load_industry_etfs(start_date='2020-01-01', end_date='2026-07-31', verbose=True):
    """加载行业ETF池（RRG轮动策略用）"""
    etfs = {
        '512880': {'name': '证券ETF', 'base': 1.0, 'ret': 0.12, 'vol': 0.35, 'seed': 1},
        '512690': {'name': '酒ETF', 'base': 1.5, 'ret': 0.15, 'vol': 0.30, 'seed': 2},
        '512660': {'name': '军工ETF', 'base': 0.9, 'ret': 0.08, 'vol': 0.32, 'seed': 3},
        '512170': {'name': '医疗ETF', 'base': 1.2, 'ret': 0.06, 'vol': 0.28, 'seed': 4},
        '159995': {'name': '芯片ETF', 'base': 1.1, 'ret': 0.18, 'vol': 0.38, 'seed': 5},
        '515030': {'name': '新能源车ETF', 'base': 1.3, 'ret': 0.20, 'vol': 0.36, 'seed': 6},
        '512400': {'name': '有色金属ETF', 'base': 1.0, 'ret': 0.10, 'vol': 0.33, 'seed': 7},
        '512010': {'name': '医药ETF', 'base': 1.4, 'ret': 0.05, 'vol': 0.25, 'seed': 8},
        '512580': {'name': '环保ETF', 'base': 0.8, 'ret': 0.03, 'vol': 0.27, 'seed': 9},
        '512670': {'name': '国防ETF', 'base': 0.95, 'ret': 0.09, 'vol': 0.31, 'seed': 10},
        '159928': {'name': '消费ETF', 'base': 1.6, 'ret': 0.11, 'vol': 0.26, 'seed': 11},
        '515050': {'name': '5GETF', 'base': 1.05, 'ret': 0.07, 'vol': 0.34, 'seed': 12},
    }

    if verbose:
        print('拉取行业ETF数据（优先真实→失败回退模拟）...')
    all_data = {}
    for i, (code, info) in enumerate(etfs.items()):
        all_data[code] = _get_etf_smart(code, info, start_date, end_date, verbose=verbose)
        if verbose and (i + 1) % 3 == 0:
            time.sleep(0.3)  # 礼貌限速
    return all_data, etfs


def load_asset_etfs(start_date='2020-01-01', end_date='2026-07-31', verbose=True):
    """加载大类资产ETF池（双均线动量轮动用）"""
    etfs = {
        '510300': {'name': '沪深300ETF', 'base': 4.0, 'ret': 0.08, 'vol': 0.22, 'seed': 101},
        '159915': {'name': '创业板ETF', 'base': 2.5, 'ret': 0.12, 'vol': 0.30, 'seed': 102},
        '513050': {'name': '中概互联ETF', 'base': 1.2, 'ret': -0.02, 'vol': 0.40, 'seed': 103},
        '513100': {'name': '纳指100ETF', 'base': 2.0, 'ret': 0.15, 'vol': 0.25, 'seed': 104},
        '518880': {'name': '黄金ETF', 'base': 4.5, 'ret': 0.06, 'vol': 0.15, 'seed': 105},
        '511260': {'name': '十年国债ETF', 'base': 100.0, 'ret': 0.03, 'vol': 0.05, 'seed': 106},
    }

    if verbose:
        print('拉取大类资产ETF数据...')
    all_data = {}
    for i, (code, info) in enumerate(etfs.items()):
        all_data[code] = _get_etf_smart(code, info, start_date, end_date, verbose=verbose)
        time.sleep(0.3)  # 礼貌限速
    return all_data, etfs


def load_broad_etf(start_date='2020-01-01', end_date='2026-07-31', verbose=True):
    """加载单只宽基ETF（网格策略用）"""
    info = {'name': '沪深300ETF', 'base': 4.0, 'ret': 0.08, 'vol': 0.22, 'seed': 201}
    code = '510300'
    return _get_etf_smart(code, info, start_date, end_date, verbose=verbose)


def fetch_otc_fund_nav(fund_code: str, start_date='2019-01-01',
                       end_date='2026-12-31', use_cache: bool = True,
                       timeout: int = 10) -> pd.DataFrame:
    """
    从东方财富拉取场外基金历史净值（单位净值DWJZ）
    接口: http://api.fund.eastmoney.com/f10/lsjz
    返回 DataFrame: index=date, 含 close(=单位净值) 列
    """
    cache_path = os.path.join(CACHE_DIR, f'fund_{fund_code}_{start_date}_{end_date}.csv')
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 30:
                return df
        except Exception:
            pass

    if not HAS_REQUESTS:
        return pd.DataFrame()

    os.makedirs(CACHE_DIR, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'http://fundf10.eastmoney.com/',
    }

    all_rows = []
    page = 1
    page_size = 20  # 东方财富该接口单页最多20条
    while True:
        url = (
            'http://api.fund.eastmoney.com/f10/lsjz'
            f'?fundCode={fund_code}&pageIndex={page}&pageSize={page_size}'
        )
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            data = r.json()
            lsjz = (data.get('Data') or {}).get('LSJZList') or []
            if not lsjz:
                break
            for item in lsjz:
                fsrq = item.get('FSRQ')
                dwjz = item.get('DWJZ')
                if fsrq and dwjz:
                    all_rows.append({'date': pd.Timestamp(fsrq), 'close': float(dwjz)})
            if len(lsjz) < page_size:
                break
            page += 1
            time.sleep(0.15)
        except Exception:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows).set_index('date').sort_index()
    # 过滤日期范围
    s = pd.Timestamp(start_date)
    e = pd.Timestamp(end_date)
    df = df[(df.index >= s) & (df.index <= e)]

    if use_cache and len(df) > 30:
        df.to_csv(cache_path)
    return df


def load_otc_fund_nav(fund_code='021400', fund_name='广发中证红利ETF发起式联接C',
                      start_date='2019-01-01', end_date='2026-12-31',
                      verbose=True) -> pd.DataFrame:
    """
    加载场外基金历史净值（广发约定净值转换用）
    优先真实数据 -> 失败回退模拟
    返回 DataFrame: index=date, 含 close 列；attrs['nav_source'] 为 'real' 或 'sim'
    """
    if USE_REAL_DATA:
        for attempt in range(3):
            df_real = fetch_otc_fund_nav(fund_code, start_date, end_date)
            if len(df_real) > 60:
                df_real = df_real.copy()
                df_real['etf_name'] = fund_name
                df_real.attrs['nav_source'] = 'real'
                if verbose:
                    print(f'  [真实净值] {fund_code} {fund_name}: '
                          f'{len(df_real)}条 {df_real.index[0].date()}~{df_real.index[-1].date()}')
                return df_real
            time.sleep(0.8)

    # 回退：模拟净值数据（按基金代码派生种子，避免全基金同一序列）
    if verbose:
        print(f'  [模拟净值] {fund_code} {fund_name}')
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    try:
        seed = int(str(fund_code).split('.')[0]) % (2**31 - 1)
    except (TypeError, ValueError):
        seed = 2024
    np.random.seed(seed)
    daily_ret = 0.04 / 252
    daily_vol = 0.18 / np.sqrt(252)
    rets = np.random.normal(daily_ret, daily_vol, len(dates))
    navs = 1.0 * np.cumprod(1 + rets)
    df = pd.DataFrame({'close': navs, 'etf_name': fund_name}, index=dates)
    df.attrs['nav_source'] = 'sim'
    return df


def load_index_components(start_date='2020-01-01', end_date='2026-07-31', verbose=True):
    """加载指数成分股（沪深300真实权重股，优先真实数据）"""
    # 20只沪深300真实权重股
    real_like = [
        ('600519', {'name': '贵州茅台', 'base': 1600, 'ret': 0.06, 'vol': 0.25, 'seed': 301}),
        ('000858', {'name': '五粮液', 'base': 150, 'ret': 0.02, 'vol': 0.30, 'seed': 302}),
        ('601318', {'name': '中国平安', 'base': 45, 'ret': -0.02, 'vol': 0.28, 'seed': 303}),
        ('000333', {'name': '美的集团', 'base': 60, 'ret': 0.04, 'vol': 0.24, 'seed': 304}),
        ('600036', {'name': '招商银行', 'base': 35, 'ret': 0.02, 'vol': 0.22, 'seed': 305}),
        ('000001', {'name': '平安银行', 'base': 12, 'ret': 0.01, 'vol': 0.25, 'seed': 306}),
        ('601012', {'name': '隆基绿能', 'base': 20, 'ret': 0.05, 'vol': 0.40, 'seed': 307}),
        ('300750', {'name': '宁德时代', 'base': 200, 'ret': 0.08, 'vol': 0.45, 'seed': 308}),
        ('600900', {'name': '长江电力', 'base': 20, 'ret': 0.05, 'vol': 0.18, 'seed': 309}),
        ('601888', {'name': '中国中免', 'base': 80, 'ret': 0.03, 'vol': 0.35, 'seed': 310}),
        ('000725', {'name': '京东方A', 'base': 4.5, 'ret': 0.02, 'vol': 0.38, 'seed': 311}),
        ('600030', {'name': '中信证券', 'base': 25, 'ret': 0.04, 'vol': 0.32, 'seed': 312}),
        ('601166', {'name': '兴业银行', 'base': 18, 'ret': 0.01, 'vol': 0.25, 'seed': 313}),
        ('002594', {'name': '比亚迪', 'base': 250, 'ret': 0.10, 'vol': 0.42, 'seed': 314}),
        ('600276', {'name': '恒瑞医药', 'base': 45, 'ret': 0.02, 'vol': 0.28, 'seed': 315}),
        ('000568', {'name': '泸州老窖', 'base': 200, 'ret': 0.05, 'vol': 0.30, 'seed': 316}),
        ('600031', {'name': '三一重工', 'base': 18, 'ret': 0.03, 'vol': 0.35, 'seed': 317}),
        ('601628', {'name': '中国人寿', 'base': 35, 'ret': 0.01, 'vol': 0.26, 'seed': 318}),
        ('002475', {'name': '立讯精密', 'base': 35, 'ret': 0.08, 'vol': 0.38, 'seed': 319}),
        ('600887', {'name': '伊利股份', 'base': 30, 'ret': 0.03, 'vol': 0.24, 'seed': 320}),
    ]

    stocks = {}
    if verbose:
        print('加载指数成分股（沪深300真实权重股，优先真实数据）...')
    for i, (code, info) in enumerate(real_like):
        stocks[code] = {
            'name': info['name'],
            'data': _get_etf_smart(code, info, start_date, end_date, verbose=verbose)
        }
        if (i + 1) % 5 == 0:
            time.sleep(0.15)  # 礼貌限速

    # 补齐到50只（模拟）
    np.random.seed(350)
    n_fill = 50 - len(stocks)
    for i in range(n_fill):
        code = f'SIM{i+1:06d}'
        name = f'模拟股{i+1:02d}'
        base = 10 + np.random.uniform(-5, 20)
        ret = np.random.uniform(-0.05, 0.20)
        vol = np.random.uniform(0.20, 0.50)
        info = {'name': name, 'base': base, 'ret': ret, 'vol': vol, 'seed': 350+i}
        stocks[code] = {
            'name': name,
            'data': generate_etf_data(name, start_date=start_date, end_date=end_date,
                                      base_price=base, annual_return=ret,
                                      volatility=vol, seed=info['seed'])
        }
    return stocks


def fetch_real_macro_data(start_date='2020-01-01', end_date='2026-07-31') -> pd.DataFrame:
    """
    从东方财富数据中心抓取真实宏观数据（CPI/PMI/M2/利率等）
    失败返回空DataFrame
    """
    if not HAS_REQUESTS:
        return pd.DataFrame()

    cache_path = os.path.join(CACHE_DIR, 'macro_real.csv')
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 12:
                return df
        except Exception:
            pass

    os.makedirs(CACHE_DIR, exist_ok=True)
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Referer': 'https://data.eastmoney.com/',
    }

    months = pd.date_range(start=start_date, end=end_date, freq='ME')
    result = pd.DataFrame(index=months)
    # 预初始化所有列，避免KeyError
    for col in ['cpi', 'pmi', 'm2', 'interest_rate', 'vix', 'credit_spread']:
        result[col] = np.nan

    # 1) CPI - 东方财富CPI数据接口 (NATIONAL_SAME=同比)
    try:
        url_cpi = ('https://datacenter-web.eastmoney.com/api/data/v1/get'
                   '?reportName=RPT_ECONOMY_CPI'
                   '&columns=ALL&pageNumber=1&pageSize=120'
                   '&sortColumns=REPORT_DATE&sortTypes=-1')
        r = requests.get(url_cpi, headers=headers, timeout=10)
        rows = r.json().get('result', {}).get('data', [])
        for row in rows:
            d = pd.Timestamp(row.get('REPORT_DATE', '')).to_period('M').to_timestamp('M')
            if d in result.index:
                result.loc[d, 'cpi'] = float(row.get('NATIONAL_SAME', 0))
    except Exception:
        pass

    # 2) PMI - 制造业PMI (MAKE_INDEX=制造业指数)
    try:
        url_pmi = ('https://datacenter-web.eastmoney.com/api/data/v1/get'
                   '?reportName=RPT_ECONOMY_PMI'
                   '&columns=ALL&pageNumber=1&pageSize=120'
                   '&sortColumns=REPORT_DATE&sortTypes=-1')
        r = requests.get(url_pmi, headers=headers, timeout=10)
        rows = r.json().get('result', {}).get('data', [])
        for row in rows:
            d = pd.Timestamp(row.get('REPORT_DATE', '')).to_period('M').to_timestamp('M')
            if d in result.index:
                result.loc[d, 'pmi'] = float(row.get('MAKE_INDEX', 0))
    except Exception:
        pass

    # 3) M2 - 货币供应量同比（备选：东财富宏观数据）
    try:
        url_m2 = ('https://datacenter-web.eastmoney.com/api/data/v1/get'
                  '?reportName=RPT_ECONOMY_MONEY_SUPPLY'
                  '&columns=ALL&pageNumber=1&pageSize=120'
                  '&sortColumns=REPORT_DATE&sortTypes=-1')
        r = requests.get(url_m2, headers=headers, timeout=10)
        rows = (r.json().get('result') or {}).get('data', [])
        for row in rows:
            d = pd.Timestamp(row.get('REPORT_DATE', '')).to_period('M').to_timestamp('M')
            if d in result.index:
                # 尝试多个可能字段名
                for k in ['M2_YOY', 'M2_SAME', 'BROARD_MONEY_SAME']:
                    if k in row and row[k]:
                        result.loc[d, 'm2'] = float(row[k])
                        break
    except Exception:
        pass

    # 4) 利率 - 10年期国债到期收益率
    try:
        url_rate = ('https://datacenter-web.eastmoney.com/api/data/v1/get'
                    '?reportName=RPT_BOND_CN_TENYEALDR'
                    '&columns=ALL&pageNumber=1&pageSize=120'
                    '&sortColumns=REPORT_DATE&sortTypes=-1')
        r = requests.get(url_rate, headers=headers, timeout=10)
        rows = (r.json().get('result') or {}).get('data', [])
        for row in rows:
            d = pd.Timestamp(row.get('REPORT_DATE', '')).to_period('M').to_timestamp('M')
            if d in result.index:
                for k in ['AVG_YIELD', 'YIELD_AVG', 'CN_10Y']:
                    if k in row and row[k]:
                        result.loc[d, 'interest_rate'] = float(row[k])
                        break
    except Exception:
        pass

    # 5) VIX近似 - 默认值
    result['vix'] = 20.0
    # 6) 信用利差 - 默认值
    result['credit_spread'] = 1.5

    # 检查是否有足够真实数据（至少CPI和PMI有数据）
    real_count = sum(1 for c in ['cpi', 'pmi', 'm2', 'interest_rate'] if result[c].notna().sum() > 12)
    if real_count < 2:
        return pd.DataFrame()

    # 前向填充，然后用合理默认值填补仍为NaN的列
    result = result.astype(float).ffill().bfill()
    result['interest_rate'] = result['interest_rate'].fillna(3.0)   # 默认3%
    result['m2'] = result['m2'].fillna(10.0)                         # 默认10%
    result['cpi'] = result['cpi'].fillna(2.0)                        # 默认2%
    result['pmi'] = result['pmi'].fillna(50.0)                       # 默认50
    result.to_csv(cache_path)
    return result


def generate_macro_factors(start_date='2020-01-01', end_date='2026-07-31'):
    """宏观因子：优先真实数据，失败回退模拟"""
    # 尝试真实宏观数据
    if USE_REAL_DATA:
        df_real = fetch_real_macro_data(start_date, end_date)
        if len(df_real) > 12:
            real_cnt = sum(1 for c in ['cpi','pmi','m2'] if c in df_real.columns and df_real[c].notna().sum() > 12)
            if real_cnt >= 2:
                print('  [真实] 宏观因子数据: CPI/PMI/M2/利率')
                return df_real

    # 回退模拟
    print('  [模拟] 宏观因子数据（真实拉取失败）')
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    np.random.seed(500)
    n = len(dates)

    factors = pd.DataFrame(index=dates)
    factors['interest_rate'] = 3.0 + np.cumsum(np.random.normal(-0.02, 0.1, n))
    factors['interest_rate'] = factors['interest_rate'].clip(1.0, 5.0)
    factors['credit_spread'] = 1.5 + np.abs(np.random.normal(0, 0.3, n))
    factors['pmi'] = 50 + np.cumsum(np.random.normal(0, 0.5, n))
    factors['pmi'] = factors['pmi'].clip(45, 55)
    factors['cpi'] = 2.0 + np.random.normal(0, 0.8, n)
    factors['m2'] = 10 + np.random.normal(0, 2, n)
    factors['vix'] = 20 + np.abs(np.random.normal(0, 5, n))
    return factors


def save_data_to_csv(data_dict, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for key, df in data_dict.items():
        if isinstance(df, pd.DataFrame):
            safe_key = str(key).replace('/', '_').replace('\\', '_')
            df.to_csv(os.path.join(output_dir, f'{safe_key}.csv'))
