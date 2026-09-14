"""Offline regression tests: no network access or real cache writes."""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

import update_fund_nav as nav


def item(date, value='1.2'):
    return {'FSRQ': date, 'DWJZ': value}


class NavUpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'fund_021400_2019-01-01_2026-12-31.csv'
        self.session = Mock()
        self.sleep = patch.object(nav.time, 'sleep').start()
        self.addCleanup(patch.stopall)

    def update(self):
        return nav.incremental_update('021400', session=self.session, cache_dir=self.tmp.name)

    def seed(self):
        self.path.write_text('date,close\n2026-09-11,1.1\n', encoding='utf-8')

    def test_increment_and_second_run_does_not_rewrite(self):
        self.seed()
        with patch.object(nav, 'fetch_page', return_value=([item('2026-09-14'), item('2026-09-11')], 500)) as fetch:
            self.assertEqual(self.update(), 1)
            before = self.path.read_bytes()
            self.assertEqual(self.update(), 0)
            self.assertEqual(self.path.read_bytes(), before)
            self.assertEqual(fetch.call_count, 2)

    def test_missing_cache_fetches_more_than_old_100_row_limit(self):
        dates = pd.date_range('2025-01-01', periods=230)[::-1]
        rows = [item(str(d.date())) for d in dates]
        pages = [(rows[:100], 230), (rows[100:200], 230), (rows[200:], 230)]
        with patch.object(nav, 'fetch_page', side_effect=pages):
            self.assertEqual(self.update(), 230)
        self.assertEqual(len(pd.read_csv(self.path)), 230)

    def test_later_page_failure_preserves_original_cache(self):
        self.seed()
        before = self.path.read_bytes()
        with patch.object(nav, 'fetch_page', side_effect=[([item('2026-09-14')], 500), RuntimeError('network')]):
            with self.assertRaises(RuntimeError):
                self.update()
        self.assertEqual(self.path.read_bytes(), before)

    def test_invalid_nav_preserves_original_cache(self):
        self.seed()
        before = self.path.read_bytes()
        with patch.object(nav, 'fetch_page', return_value=([item('2026-09-14', 'nan')], 1)):
            with self.assertRaises(ValueError):
                self.update()
        self.assertEqual(self.path.read_bytes(), before)

    def test_unsorted_cache_uses_maximum_date(self):
        self.path.write_text('date,close\n2026-09-14,1.2\n2026-09-11,1.1\n', encoding='utf-8')
        with patch.object(nav, 'fetch_page', return_value=([item('2026-09-14'), item('2026-09-11')], 2)):
            self.assertEqual(self.update(), 0)

    def test_retries_transport_failure(self):
        response = Mock()
        response.json.return_value = {'ErrCode': 0, 'Data': {'LSJZList': [item('2026-09-14')]}, 'TotalCount': 1}
        self.session.get.side_effect = [requests.Timeout(), response]
        self.assertEqual(nav.fetch_page(self.session, '021400', 1, time.monotonic() + 20)[1], 1)
        self.assertEqual(self.session.get.call_count, 2)

    def test_empty_api_response_is_failure(self):
        self.session.get.return_value.json.return_value = {'ErrCode': 0, 'Data': {'LSJZList': []}, 'TotalCount': 0}
        with self.assertRaises(RuntimeError):
            nav.fetch_page(self.session, '021400', 1, time.monotonic() + 20)
        self.assertEqual(self.session.get.call_count, 3)

    def test_expired_deadline_makes_no_request(self):
        with self.assertRaises(TimeoutError):
            nav.incremental_update('021400', session=self.session, cache_dir=self.tmp.name, deadline=time.monotonic() - 1)
        self.session.get.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_main_reports_partial_failure(self):
        with patch.object(nav, 'load_funds', return_value={'021400': 'a', '021093': 'b'}), patch.object(nav, 'incremental_update', side_effect=[TimeoutError('budget'), 1]) as update:
            self.assertEqual(nav.main([]), 1)
            self.assertEqual(update.call_count, 2)

    def test_total_budget_skips_remaining_funds(self):
        with patch.object(nav, 'load_funds', return_value={'021400': 'a', '021093': 'b'}), patch.object(nav, 'incremental_update') as update, patch.object(nav.time, 'monotonic', side_effect=[0, 181, 181, 181, 181]):
            self.assertEqual(nav.main(['--max-seconds', '180']), 1)
            update.assert_not_called()

    def test_timeout_after_response_does_not_write(self):
        self.seed()
        before = self.path.read_bytes()
        with patch.object(nav, 'fetch_page', return_value=([item('2026-09-14')], 1)), patch.object(nav, 'check_deadline', side_effect=TimeoutError('budget')):
            with self.assertRaises(TimeoutError):
                self.update()
        self.assertEqual(self.path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
