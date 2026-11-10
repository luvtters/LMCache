"""Actual registry/adapter/connector source, with CPU-only control replies."""
from concurrent.futures import Future
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace as NS
import time
import unittest

from test_lookup_deadline_contract import methods


ROOT = Path(__file__).resolve().parents[2]
PATH = ROOT / 'lmcache/v1/multiprocess/lookup_control_cleanup.py'
spec = importlib.util.spec_from_file_location('preemption_registry', PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Reply(Future):
    def query(self):
        return self.done()


def adapter(policy='transfer_borrow'):
    cls = methods('lmcache/integration/vllm/vllm_multi_process_adapter.py',
                  'LMCacheMPSchedulerAdapter', {
                      'cleanup_lookup_result', '_retain_lookup_controls',
                      '_check_lookup_reuse', 'maybe_submit_lookup_request', 'end_session',
                      'check_lookup_result',
                  }, {'get_read_policy': lambda: policy, 'time': time,
                      'LookupControlRegistry': module.LookupControlRegistry,
                      '_LookupAck': NS, 'logger': logging.getLogger('preemption-contract')})
    obj = cls()
    obj.is_healthy = True
    obj._server_urls = ['stage0', 'stage1']
    obj._pending_lookups = {'request'}
    obj._unacked_lookups = {}
    obj._lookup_status = {}
    obj._lookup_deadlines = {}
    obj._finished_lookup_results = {'request': 512}
    obj._per_server_hits = {'request': dict.fromkeys(obj._server_urls, 1)}
    obj._lookup_params = {'request': ([], '', None)}
    obj.sent = []
    obj.ends = {}

    def lookup(url, key, readers):
        obj.sent.append(url)
        return Reply()

    def end(url, rid):
        obj.ends[url] = Reply()
        return obj.ends[url]

    obj.req_clients = {
        url: NS(lookup=lambda key, readers, url=url: lookup(url, key, readers),
                end_session=lambda rid, url=url: end(url, rid))
        for url in obj._server_urls}
    obj._ensure_heartbeat_started = lambda: None
    obj.lmcache_tokens_per_chunk = 512
    obj._mq_timeout = 10
    obj._nonblocking_lookup_status = True
    obj.tp_size = 16
    obj._create_key = lambda *a, **kw: NS(no_worker_id_version=lambda: 'key')
    return obj


def connector(obj):
    states = NS(READY='READY', PREFETCHING='PREFETCHING', BYPASS_LMCACHE='BYPASS')
    cls = methods('lmcache/integration/vllm/lmcache_mp_connector.py',
                  'LMCacheMPConnector', {'_get_or_create_request_tracker',
                                       'get_num_new_matched_tokens'}, {
                      'RequestStatus': NS(PREEMPTED='PREEMPTED'),
                      'LMCacheMPRequestState': states, 'time': time,
                      'logger': logging.getLogger('preemption-contract'),
                      'LMCacheMPRequestTracker': lambda req: NS(
                          state='PREFETCHING', lookup_started_at=None,
                          get_token_ids=lambda: [], cache_salt='', request_configs=None),
                  })
    con = cls()
    con.request_trackers = {'request': NS(state='READY')}
    con.lazy_offload = False
    con._read_policy = 'transfer_borrow'
    con._reserve_last_token_for_lookup = False
    con._hit_alignment_tokens = 512
    con._connector_stats = NS(record_lookup=lambda duration: None)
    con.scheduler_adapter = obj
    return con


class PreemptedLookupTest(unittest.TestCase):
    def test_legacy_preemption_keeps_existing_lookup_path(self):
        obj = adapter(policy='legacy')
        obj.cleanup_lookup_result('request')
        con = connector(obj)
        con._read_policy = 'legacy'
        request = NS(request_id='request', status='PREEMPTED', num_computed_tokens=0)
        self.assertEqual(con.get_num_new_matched_tokens(request, 0), (None, True))
        self.assertEqual(obj.sent, ['stage0', 'stage1'])
        self.assertFalse(hasattr(obj, '_lookup_control_registry'))

    def test_normal_admission_then_preemption_submits_new_lookup(self):
        obj = adapter()
        obj.cleanup_lookup_result('request')
        con = connector(obj)
        old = con.request_trackers['request']
        request = NS(request_id='request', status='PREEMPTED', num_computed_tokens=0)
        self.assertEqual(con.get_num_new_matched_tokens(request, 0), (None, True))
        self.assertIsNot(old, con.request_trackers['request'])
        self.assertEqual(obj.sent, ['stage0', 'stage1'])
        self.assertEqual(obj.ends, {})
        # Repeated scheduler polls must retain the same two new RPCs.
        self.assertEqual(con.get_num_new_matched_tokens(request, 0), (None, True))
        self.assertEqual(obj.sent, ['stage0', 'stage1'])

    def test_ordinary_id_reuse_still_requires_end_ack(self):
        obj = adapter()
        obj.cleanup_lookup_result('request')
        with self.assertRaises(RuntimeError):
            obj.maybe_submit_lookup_request('request', [])
        self.assertEqual(obj.sent, [])

    def test_pending_failed_or_ending_controls_recompute_without_new_lookup(self):
        for kind in ('pending', 'failed', 'ending'):
            with self.subTest(kind=kind):
                obj = adapter()
                old = Reply()
                if kind == 'failed':
                    old.set_exception(ConnectionError('delivery outcome unknown'))
                elif kind == 'ending':
                    old.set_result(None)
                obj._unacked_lookups['request'] = NS(futures={'stage0': old})
                obj.cleanup_lookup_result('request')
                if kind == 'ending':
                    obj.end_session('request')
                # Even an old cached hit must not become the new tracker's hit
                # when admission of its replacement LOOKUP was refused.
                obj._finished_lookup_results['request'] = 8192
                obj._pending_lookups.add('request')
                con = connector(obj)
                request = NS(request_id='request', status='PREEMPTED', num_computed_tokens=0)
                self.assertEqual(con.get_num_new_matched_tokens(request, 0), (0, False))
                self.assertEqual(obj.sent, [])
                self.assertIn('request', obj._lookup_control_registry._states)

    def test_completed_retained_controls_allow_same_session_preemption(self):
        obj = adapter()
        old = Reply()
        obj._unacked_lookups['request'] = NS(futures={'stage0': old})
        obj.cleanup_lookup_result('request')
        old.set_result(None)
        obj.maybe_submit_lookup_request('request', [], resume_existing_session=True)
        self.assertEqual(obj.sent, ['stage0', 'stage1'])
        self.assertEqual(obj.ends, {})

    def test_recompute_keeps_late_controls_before_end_and_id_reuse(self):
        obj = adapter()
        old = Reply()
        obj._unacked_lookups['request'] = NS(futures={'stage0': old})
        obj.cleanup_lookup_result('request')
        obj.maybe_submit_lookup_request('request', [], resume_existing_session=True)
        self.assertEqual(obj.check_lookup_result('request'), 0)
        obj.end_session('request')
        self.assertEqual(set(obj.ends), {'stage1'})
        old.set_result(None)
        self.assertEqual(set(obj.ends), {'stage0', 'stage1'})
        with self.assertRaises(RuntimeError):
            obj._check_lookup_reuse('request')
        for future in obj.ends.values():
            future.set_result(None)
        self.assertTrue(obj._check_lookup_reuse('request'))


if __name__ == '__main__':
    unittest.main()
