"""Incrementally update NAV caches; exit nonzero if any fund fails.

Run: python update_fund_nav.py
Defaults: 180 seconds overall, 20 seconds per fund, 3 attempts per page.
Timeouts leave the affected fund's cache untouched and produce a failure exit.
Cache filenames remain compatible with the strategy readers.
"""
import argparse
import math
import os
from pathlib import Path
import tempfile
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / '.data_cache'
START = pd.Timestamp('2019-01-01')
POOL_FILES = (
    'strategy6_gf_nav_conversion/fund.txt',
    'strategy7_dual_bottom_adaptive/fund_pool.txt',
    'strategy8_dual_bottom_ladder/fund_pool.txt',
)
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'Referer': 'https://fundf10.eastmoney.com/',
}


def check_deadline(deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError('NAV update time budget exhausted')
    return remaining


def fetch_page(session, code, page, deadline):
    """Retry transport, JSON and API errors; never report them as no updates."""
    for attempt in range(3):
        remaining = check_deadline(deadline)
        try:
            response = session.get(
                'https://api.fund.eastmoney.com/f10/lsjz',
                params={'fundCode': code, 'pageIndex': page, 'pageSize': 100},
                headers=HEADERS, timeout=(min(3, remaining / 2), min(8, remaining / 2)),
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get('ErrCode') != 0:
                raise ValueError(f'API error: {payload.get("ErrCode")}')
            rows = payload['Data']['LSJZList']
            total = int(payload['TotalCount'])
            if not isinstance(rows, list) or total <= 0 or not rows:
                raise ValueError('Empty or invalid NAV response')
            return rows, total
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            if attempt == 2:
                raise RuntimeError(f'{code} page {page} failed: {exc}') from exc
            time.sleep(min(2 ** attempt, check_deadline(deadline)))


def incremental_update(code, name='', *, session=None, cache_dir=None, deadline=None):
    if deadline is None:
        deadline = time.monotonic() + 20
    if session is None:
        with requests.Session() as owned_session:
            return incremental_update(code, name, session=owned_session, cache_dir=cache_dir, deadline=deadline)
    cache_path = Path(cache_dir if cache_dir is not None else CACHE) / f'fund_{code}_2019-01-01_2026-12-31.csv'
    old = pd.DataFrame(columns=['close'], index=pd.DatetimeIndex([], name='date'))
    if cache_path.exists():
        old = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if 'close' not in old or old.index.hasnans:
            raise ValueError(f'Invalid cache: {cache_path}')
        old['close'] = pd.to_numeric(old['close'], errors='raise')
        if not old['close'].map(lambda v: math.isfinite(v) and v > 0).all():
            raise ValueError(f'Invalid cached NAV: {cache_path}')
        old = old[~old.index.duplicated(keep='last')].sort_index()
        old = old.loc[old.index >= START]
    last_date = old.index.max() if len(old) else START - pd.Timedelta(days=1)
    new_rows = []
    previous_date = None
    received = 0
    for page in range(1, 51):
        rows, total = fetch_page(session, code, page, deadline)
        check_deadline(deadline)
        hit_old = False
        for item in rows:
            date = pd.Timestamp(item['FSRQ'])
            if pd.isna(date) or (previous_date is not None and date >= previous_date):
                raise ValueError(f'{code}: NAV dates are not strictly descending')
            previous_date = date
            if date <= last_date:
                hit_old = True
                continue
            nav = float(item['DWJZ'])
            if not math.isfinite(nav) or nav <= 0:
                raise ValueError(f'{code}: invalid NAV on {date}')
            new_rows.append({'date': date, 'close': nav})
        received += len(rows)
        if hit_old or received >= total:
            break
        time.sleep(0.15)
    else:
        raise RuntimeError(f'{code}: pagination limit reached; cache not written')

    if not new_rows:
        print(f'{code} {name}: no increment; local latest {last_date.date()}')
        return 0
    new = pd.DataFrame(new_rows).set_index('date')
    combined = pd.concat([old, new]) if len(old) else new
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    combined.index.name = 'date'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Replace only after every required page was fetched and validated.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=cache_path.parent, suffix='.tmp', delete=False) as tmp:
            temp_path = Path(tmp.name)
        combined.to_csv(temp_path)
        os.replace(temp_path, cache_path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
    added = len(combined) - len(old)
    print(f'{code} {name}: +{added}; latest {combined.index.max().date()}')
    return added


def load_funds():
    funds = {}
    for relative in POOL_FILES:
        for line in (ROOT / relative).read_text(encoding='utf-8-sig').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split()
            code = parts[-1]
            if len(code) != 6 or not code.isdigit():
                raise ValueError(f'Invalid fund code in {relative}: {code}')
            funds[code] = ' '.join(parts[:-1])
    if not funds:
        raise ValueError('Fund pools are empty')
    return funds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-seconds', type=float, default=180, help='Total time budget (default: 180)')
    parser.add_argument('--fund-timeout', type=float, default=20, help='Per-fund time budget (default: 20)')
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) and v > 0 for v in (args.max_seconds, args.fund_timeout)):
        parser.error('Time budgets must be finite and positive')
    deadline = time.monotonic() + args.max_seconds
    funds = load_funds()
    failed = []
    added = 0
    with requests.Session() as session:
        for code, name in funds.items():
            try:
                check_deadline(deadline)
                added += incremental_update(code, name, session=session,
                                            deadline=min(deadline, time.monotonic() + args.fund_timeout))
            except Exception as exc:
                failed.append(code)
                print(f'ERROR {code} {name}: {exc}', flush=True)
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(0.2, remaining))
    print(f'Funds: {len(funds)}; succeeded: {len(funds) - len(failed)}; added rows: {added}; failed: {failed}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
