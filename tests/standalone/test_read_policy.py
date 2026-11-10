"""Policy tests use the real stdlib-only module, without device imports."""
import importlib.util
import ast
import os
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / 'lmcache/v1/multiprocess/read_policy.py'


class ReadPolicyTest(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location('read_policy_under_test', SOURCE)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def test_default_legacy(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.module.get_read_policy(), 'legacy')

    def test_invalid_and_mixed_recheck_rejected(self):
        for values in ({'KS_LMCACHE_MP_READ_POLICY': 'bad'},
                       {'KS_LMCACHE_MP_READ_POLICY': 'transfer_borrow',
                        'KS_LMCACHE_L1_RECHECK_ON_STATUS': '1'}):
            with patch.dict(os.environ, values, clear=True):
                with self.assertRaises(ValueError):
                    self.module.get_read_policy()

    def test_new_mode_requires_recompute_and_pp(self):
        validate = self.module.validate_connector_policy
        for policy, stage_local, failure in (
            ('transfer_borrow', True, 'fail'),
            ('transfer_borrow', False, 'recompute'),
        ):
            with self.assertRaises(ValueError):
                validate(policy, stage_local=stage_local, failure_policy=failure)
        validate('transfer_borrow', stage_local=True, failure_policy='recompute')
        validate('legacy', stage_local=False, failure_policy='fail')

    def test_only_scheduled_restore_advances_confirmed_stored_prefix(self):
        source = SOURCE.parents[2] / 'integration/vllm/lmcache_mp_metadata.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                   and n.name == 'LMCacheMPRequestTracker')
        method = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                      and n.name == 'confirm_borrowed_restore')
        ns = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), str(source), 'exec'), ns)
        tracker = SimpleNamespace(borrowed_restore_submitted=False, num_stored_tokens=0,
                                  num_lmcache_hit_tokens=8192)
        ns['confirm_borrowed_restore'](tracker, 8192, 512)
        self.assertEqual(tracker.num_stored_tokens, 0)  # APC-only hint
        tracker.borrowed_restore_submitted = True
        ns['confirm_borrowed_restore'](tracker, 8191, 512)
        self.assertEqual(tracker.num_stored_tokens, 7680)
        self.assertFalse(tracker.borrowed_restore_submitted)
        tracker.borrowed_restore_submitted = True
        tracker.num_lmcache_hit_tokens = tracker.num_stored_tokens = 0
        ns['confirm_borrowed_restore'](tracker, 0, 512)
        self.assertEqual(tracker.num_stored_tokens, 0)  # failed load/recompute

        # A failed load can retain a valid local APC prefix. Its old external
        # hint must not make GetStoreMetadata count uncomputed KV as available.
        tracker.borrowed_restore_submitted = True
        tracker.num_lmcache_hit_tokens = 8192
        tracker.num_vllm_hit_tokens = 1024
        ns['confirm_borrowed_restore'](tracker, 1024, 512)
        self.assertEqual(tracker.num_lmcache_hit_tokens, 0)
        self.assertEqual(tracker.num_vllm_hit_tokens, 1024)
        self.assertEqual(tracker.num_stored_tokens, 0)


if __name__ == '__main__':
    unittest.main()
