# SPDX-License-Identifier: Apache-2.0
"""CPU contracts executing the real scheduler methods without device imports."""

import unittest
import time
import ast
import enum
from pathlib import Path
from types import SimpleNamespace

from test_lookup_deadline_contract import Reply, methods, scheduler
from test_preempted_lookup_contract import Reply as FutureReply, adapter, connector


def stages(enabled=True):
    obj, clock, _ = scheduler()
    obj._pp_query_fastpath = enabled
    obj._server_urls = ["early", "late"]
    obj.sent = []
    obj.replies = {"early": Reply(0), "late": Reply(2, ready=False)}

    def query(url, request_id):
        obj.sent.append((url, request_id))
        return obj.replies[url]

    obj.req_clients = {
        url: SimpleNamespace(query_prefetch_status=lambda rid, url=url: query(url, rid))
        for url in obj._server_urls
    }
    return obj, clock


class EarlyMissTest(unittest.TestCase):
    def test_invalid_config_does_not_create_transport_clients(self):
        source = "lmcache/integration/vllm/vllm_multi_process_adapter.py"
        tree = ast.parse((Path(__file__).resolve().parents[2] / source).read_text(encoding="utf-8"))
        defaults = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExtraConfigDefault")
        namespace = {"enum": enum, "Any": object}
        exec(compile(ast.Module(body=[defaults], type_ignores=[]), source, "exec"), namespace)
        config = namespace["ExtraConfigDefault"]
        created = []
        strategy = SimpleNamespace(pp_stage_local=False, pp_size=1)
        cls = methods(source, "LMCacheMPSchedulerAdapter", {"__init__"}, {
            "ExtraConfigDefault": config,
            "DEFAULT_MQ_TIMEOUT": 300.0,
            "DEFAULT_HEARTBEAT_INTERVAL": 10.0,
            "_normalize_adapter_init_args": lambda *a: (16, strategy, 10, False),
            "_resolve_extra_config": lambda extra: {
                item.name: (True if item.name == "pp_query_fastpath" else item.default)
                for item in config
            },
            "RequestClientFactory": SimpleNamespace(create=lambda *a, **k: created.append(a)),
            "get_read_policy": lambda: "transfer_borrow",
        })
        with self.assertRaises(ValueError):
            cls(["tcp://invalid:1"], object(), "model", 16, strategy, extra_config={})
        self.assertEqual(created, [])

    def test_pending_ack_does_not_renew_deadline(self):
        obj, clock = stages()
        pending = Reply(ready=False)
        obj.replies["early"] = Reply(3)
        obj._unacked_lookups["r"] = SimpleNamespace(
            futures={"early": Reply(None), "late": pending}, submitted_at=0.0
        )
        self.assertIsNone(obj.check_lookup_result("r"))
        clock.now = 5.1
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(obj.timed_out, ["late"])
        self.assertIs(obj._unacked_lookups["r"].futures["late"], pending)
        self.assertEqual(obj._lookup_deadlines["r"], 5.0)
        self.assertNotIn("r", obj._finished_lookup_results)

    def test_early_miss_admission_cleanup_then_end_waits_for_own_ack(self):
        obj = adapter()
        obj._pp_query_fastpath = True
        obj._finished_lookup_results.clear()
        obj._per_server_hits.clear()
        ready, pending, zero = FutureReply(), FutureReply(), FutureReply()
        ready.set_result(None)
        zero.set_result(0)
        obj._unacked_lookups["request"] = SimpleNamespace(
            futures={"stage0": ready, "stage1": pending},
            submitted_at=time.monotonic(),
        )
        obj.req_clients["stage0"].query_prefetch_status = lambda rid: zero
        con = connector(obj)
        con.request_trackers["request"] = SimpleNamespace(
            state="PREFETCHING", lookup_started_at=time.monotonic(),
            get_token_ids=lambda: [], cache_salt="", request_configs=None,
            num_allocated_blocks=lambda: {}, num_lmcache_hit_tokens=0,
        )
        request = SimpleNamespace(request_id="request", status="RUNNING")
        self.assertEqual(con.get_num_new_matched_tokens(request, 0), (0, False))
        alloc = methods(
            "lmcache/integration/vllm/lmcache_mp_connector.py",
            "LMCacheMPConnector", {"update_state_after_alloc"}, {
                "LMCacheMPRequestState": SimpleNamespace(
                    READY="READY", PREFETCHING="PREFETCHING", BYPASS_LMCACHE="BYPASS"
                )
            },
        )
        con._get_request_tracker = lambda rid: con.request_trackers[rid]
        alloc.update_state_after_alloc(
            con, request, SimpleNamespace(get_block_ids=lambda: ()), 0
        )
        self.assertEqual(con.request_trackers["request"].state, "READY")
        self.assertEqual(obj.ends, {})
        self.assertEqual(obj.sent, [])
        self.assertFalse(obj._check_lookup_reuse(
            "request", resume_existing_session=True
        ))
        obj.end_session("request")
        self.assertEqual(set(obj.ends), {"stage0"})
        pending.set_result(None)
        self.assertEqual(set(obj.ends), {"stage0", "stage1"})
        with self.assertRaises(RuntimeError):
            obj._check_lookup_reuse("request")
        for future in obj.ends.values():
            future.set_result(None)
        self.assertTrue(obj._check_lookup_reuse("request"))

    def test_ack_exception_is_not_cached_as_final_miss(self):
        obj, _ = stages()
        broken = FutureReply()
        broken.set_exception(ConnectionError("unknown delivery"))
        obj._unacked_lookups["r"] = SimpleNamespace(
            futures={"early": broken}, submitted_at=0.0
        )
        with self.assertRaises(ConnectionError):
            obj.check_lookup_result("r")
        self.assertNotIn("r", obj._finished_lookup_results)
        self.assertIs(obj._unacked_lookups["r"].futures["early"], broken)
        self.assertEqual(obj.sent, [])

    def test_disabled_preserves_all_ack_barrier(self):
        obj, _ = stages(enabled=False)
        obj._unacked_lookups["r"] = SimpleNamespace(
            futures={"early": Reply(None), "late": Reply(ready=False)},
            submitted_at=0.0,
        )
        self.assertIsNone(obj.check_lookup_result("r"))
        self.assertEqual(obj.sent, [])

    def test_own_ack_allows_status_while_peer_ack_is_pending(self):
        obj, _ = stages()
        pending = Reply(ready=False)
        obj._unacked_lookups["r"] = SimpleNamespace(
            futures={"early": Reply(None), "late": pending}, submitted_at=1.25
        )
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(obj.sent, [("early", "r")])
        self.assertIs(obj._unacked_lookups["r"].futures["late"], pending)
        self.assertEqual(obj._unacked_lookups["r"].submitted_at, 1.25)
        self.assertEqual(obj._lookup_deadlines["r"], 5.0)

    def test_empty_ack_is_not_a_miss_and_status_is_not_duplicated(self):
        obj, _ = stages()
        pending = Reply(ready=False)
        obj.replies["early"] = Reply(3, ready=False)
        obj._unacked_lookups["r"] = SimpleNamespace(
            futures={"early": Reply(None), "late": pending}, submitted_at=0.0
        )
        self.assertIsNone(obj.check_lookup_result("r"))
        self.assertIsNone(obj.check_lookup_result("r"))
        self.assertEqual(obj.sent, [("early", "r")])
        pending.ready = True
        obj.replies["late"].ready = True
        self.assertIsNone(obj.check_lookup_result("r"))
        obj.replies["early"].ready = True
        self.assertEqual(obj.check_lookup_result("r"), 8)
        self.assertEqual(obj.sent, [("early", "r"), ("late", "r")])

    def test_final_zero_does_not_wait_for_peer(self):
        obj, _ = stages()
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(obj._finished_lookup_results["r"], 0)
        self.assertIs(obj._lookup_status["r"]["late"][0], obj.replies["late"])
        obj.replies["late"].ready = True
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(len(obj.sent), 2)

    def test_disabled_preserves_full_status_barrier(self):
        obj, _ = stages(enabled=False)
        self.assertIsNone(obj.check_lookup_result("r"))

    def test_none_is_pending_and_positive_waits_for_all(self):
        obj, _ = stages()
        obj.replies["early"] = Reply(None)
        self.assertIsNone(obj.check_lookup_result("r"))
        obj.replies["early"] = Reply(3)
        self.assertIsNone(obj.check_lookup_result("r"))
        obj.replies["late"].ready = True
        self.assertEqual(obj.check_lookup_result("r"), 8)
        self.assertEqual(len(obj.sent), 3)


if __name__ == "__main__":
    unittest.main()
