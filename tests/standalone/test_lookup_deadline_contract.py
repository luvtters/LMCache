"""Actual source methods: bounded lookup polling and cancelled result cleanup.

Run with KS_TEST_SOURCE_REV=HEAD to check a committed baseline without editing
working sources. Network replies, the clock and result payloads are substitutes.
"""
import ast
from collections import deque
import logging
import os
from pathlib import Path
import subprocess
import threading
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
REV = os.environ.get("KS_TEST_SOURCE_REV")


def methods(relative, owner_name, names, namespace):
    path = ROOT / relative
    source = (subprocess.check_output(["git", "show", f"{REV}:{relative}"], cwd=ROOT)
              .decode("utf-8") if REV else path.read_text(encoding="utf-8"))
    tree = ast.parse(source)
    owner = next(n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == owner_name)
    nodes = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise AssertionError(f"Missing real methods in {REV or 'working'} {relative}")
    for n in nodes:
        n.decorator_list = []
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    code = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    return type(owner_name + "ActualMethods", (), {n.name: namespace[n.name] for n in nodes})


class Reply:
    def __init__(self, value=None, ready=True):
        self.value, self.ready = value, ready
        self.timeouts = []

    def query(self):
        return self.ready

    def result(self, timeout=None):
        self.timeouts.append(timeout)
        if not self.ready:
            raise TimeoutError("reply not ready")
        return self.value


def scheduler(nonblocking=True):
    clock = SimpleNamespace(now=0.0)
    cls = methods("lmcache/integration/vllm/vllm_multi_process_adapter.py",
        "LMCacheMPSchedulerAdapter", {"check_lookup_result"}, {
            "time": SimpleNamespace(monotonic=lambda: clock.now),
            "logger": logging.getLogger("lookup-contract"),
        })
    obj = cls()
    obj._pending_lookups = {"r"}
    obj._finished_lookup_results = {}
    obj._lookup_deadlines = {"r": 5.0}
    obj._unacked_lookups = {}
    obj._lookup_status = {}
    obj._per_server_hits = {}
    obj._server_urls = ["server"]
    obj.is_healthy = True
    obj._mq_timeout = 5.0
    obj._nonblocking_lookup_status = nonblocking
    obj.lmcache_tokens_per_chunk = 4
    obj.timed_out = []
    obj._mark_lookup_timed_out = lambda url: obj.timed_out.append(url)
    obj._health_events = {"server": threading.Event()}
    replies = deque()
    obj.requests = []

    def query(request_id):
        obj.requests.append(request_id)
        return replies.popleft()

    obj.req_clients = {"server": SimpleNamespace(query_prefetch_status=query)}
    obj._free_inconsistent_lookup_locks = lambda *args: None
    return obj, clock, replies


class LookupDeadlineTest(unittest.TestCase):
    def test_continuous_none_does_not_renew_total_deadline(self):
        obj, clock, replies = scheduler()
        for now in (0.0, 2.0, 4.9):
            clock.now = now
            replies.append(Reply(None))
            self.assertIsNone(obj.check_lookup_result("r"))
        clock.now = 5.1
        replies.append(Reply(None))
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(len(obj.requests), 3)
        self.assertEqual(obj._lookup_deadlines["r"], 5.0)

    def test_already_completed_late_positive_is_consumed(self):
        obj, clock, _ = scheduler()
        reply = Reply(3)
        obj._lookup_status = {"r": {"server": (reply, 1.0)}}
        clock.now = 8.0
        self.assertEqual(obj.check_lookup_result("r"), 12)
        self.assertEqual(obj._finished_lookup_results["r"], 12)
        self.assertEqual(obj.timed_out, [])

    def test_ack_timeout_uses_total_budget_and_retains_pending_dependency(self):
        obj, clock, _ = scheduler()
        reply = Reply(ready=False)
        obj._unacked_lookups["r"] = SimpleNamespace(
            futures={"server": reply}, submitted_at=4.0
        )
        clock.now = 5.1
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertEqual(obj.timed_out, ["server"])
        self.assertIs(obj._unacked_lookups["r"].futures["server"], reply)
        self.assertEqual(obj.requests, [])

    def test_blocking_status_uses_only_remaining_budget(self):
        obj, clock, replies = scheduler(nonblocking=False)
        clock.now = 4.0
        reply = Reply(2)
        replies.append(reply)
        self.assertEqual(obj.check_lookup_result("r"), 8)
        self.assertEqual(reply.timeouts, [1.0])

    def test_timed_out_status_is_retained_for_end_ordering(self):
        obj, clock, replies = scheduler(nonblocking=False)
        clock.now = 4.0
        reply = Reply(ready=False)
        replies.append(reply)
        self.assertEqual(obj.check_lookup_result("r"), 0)
        self.assertIs(obj._lookup_status["r"]["server"][0], reply)

    def test_legacy_without_total_deadline_preserves_per_poll_budget(self):
        obj, clock, replies = scheduler(nonblocking=False)
        obj._lookup_deadlines = {}
        clock.now = 100.0
        reply = Reply(2)
        replies.append(reply)
        self.assertEqual(obj.check_lookup_result("r"), 8)
        self.assertEqual(reply.timeouts, [5.0])


class EmptyCells:
    """Empty bitmap payload substitute; publication code itself is real."""
    def __len__(self):
        return 1

    def to_list(self):
        return [[False]]

    def __and__(self, other):
        return self

    def __sub__(self, other):
        return self

    def zeros_like(self):
        return self

    def popcount(self):
        return 0


def controller():
    def build(request_id, spec):
        return SimpleNamespace(
            request_id=request_id,
            key_states=SimpleNamespace(l1_locked_keys=SimpleNamespace(
                merge=EmptyCells, items=lambda: ())),
            key_groups=[SimpleNamespace(keys=["key"], sliding_window_size=-1)],
            fetching_policy="prefix", lock_mode="no_lock", l2_loaded_cells=[],
        )

    cls = methods("lmcache/v1/distributed/storage_controllers/prefetch_controller.py",
        "PrefetchController", {"query_prefetch_result", "discard_prefetch_result",
                               "_publish_result", "_finish_request",
                               "submit_prefetch_request"}, {
            "_build_request": build,
            "Bitmap2D": lambda rows: EmptyCells(),
            "fold_unfold_grouped": lambda rows, windows: (0, [[False]]),
            "PrefetchLockMode": SimpleNamespace(NO_LOCK="no_lock", LOCK="lock"),
            "PrefetchResult": lambda **kwargs: SimpleNamespace(**kwargs),
            "logger": logging.getLogger("lookup-contract"),
        })
    obj = cls()
    obj._prefetch_results_lock = threading.Lock()
    obj._prefetch_results_cv = threading.Condition(obj._prefetch_results_lock)
    obj._completed_results = {}
    obj._submission_lock = threading.Lock()
    obj._next_request_id = 7
    obj._l2_adapters = []
    obj._submission_queue = []
    obj._lock_l1_keys = lambda request: None
    obj._plan_load = lambda request, adapters: None
    return obj


class CancelledResultTest(unittest.TestCase):
    def test_no_l2_submit_publishes_synchronously_before_return(self):
        obj = controller()
        request_id = obj.submit_prefetch_request(SimpleNamespace())
        self.assertEqual(request_id, 7)
        self.assertEqual(obj._submission_queue, [])
        self.assertIn(request_id, obj._completed_results)
        obj.discard_prefetch_result(request_id)
        self.assertEqual(obj._completed_results, {})
        self.assertEqual(getattr(obj, "_discarded_results", set()), set())

    def test_discard_after_publication_removes_all_result_state(self):
        obj = controller()
        obj._publish_result(SimpleNamespace(request_id=7), object())
        obj.discard_prefetch_result(7)
        self.assertEqual(obj._completed_results, {})
        self.assertEqual(getattr(obj, "_discarded_results", set()), set())

    def test_late_discard_of_consumed_result_does_not_leak_marker(self):
        obj = controller()
        result = object()
        obj._publish_result(SimpleNamespace(request_id=7), result)
        self.assertIs(obj.query_prefetch_result(7), result)
        obj.discard_prefetch_result(7)
        obj.discard_prefetch_result(7)
        self.assertEqual(obj._completed_results, {})
        self.assertEqual(getattr(obj, "_discarded_results", set()), set())

    def test_end_then_inflight_consume_does_not_recreate_session(self):
        ns = {
            "TRANSFER_BORROW": "transfer_borrow",
            "Event": lambda **kwargs: SimpleNamespace(**kwargs),
            "EventType": SimpleNamespace(MP_VLLM_END_SESSION=1, MP_REQUEST_END=2,
                                         MP_LOOKUP_PREFETCH_END=3),
            "logger": logging.getLogger("lookup-contract"),
        }
        cls = methods("lmcache/v1/multiprocess/modules/lookup.py", "LookupModule",
                      {"end_session", "_consume_prefetch_result"}, ns)
        obj = cls()
        job = SimpleNamespace(request_id="r", handle=7, row_windows=[],
            attn_desc=SimpleNamespace(num_object_groups=1), requested_tokens=8,
            early_exit_reason=None, model_name="model", cache_salt="")
        obj._prefetch_jobs = {"r": job}
        obj._prefetch_job_lock = threading.Lock()
        sessions = {"r": SimpleNamespace(lookup_ipc_key=None)}
        recreations = []
        discarded = []
        result = SimpleNamespace(l1_owners={}, l1_hit_count=0, l2_hit_count=0)
        obj._ctx = SimpleNamespace(
            read_policy="transfer_borrow", chunk_size=4,
            event_bus=SimpleNamespace(publish=lambda event: None),
            session_manager=SimpleNamespace(
                remove=lambda rid: sessions.pop(rid, None),
                get_or_create=lambda rid: recreations.append(rid),
            ),
            storage_manager=SimpleNamespace(
                discard_prefetch_result=lambda handle: discarded.append(handle),
                query_prefetch_status=lambda handle: result,
            ),
        )
        obj.end_session("r")
        self.assertEqual(obj._consume_prefetch_result(job), 0)
        self.assertEqual(sessions, {})
        self.assertEqual(recreations, [])
        self.assertEqual(discarded, [7])
        self.assertEqual(obj._prefetch_jobs, {})


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
