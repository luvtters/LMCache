# SPDX-License-Identifier: Apache-2.0
"""STORE observation must retain the existing wait and release contracts."""
from __future__ import annotations

import logging
import importlib.util
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS
import unittest

from test_lookup_deadline_contract import methods

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "actual_pp_wait_observer", ROOT / "lmcache/v1/multiprocess/pp_wait_observer.py"
)
observer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observer_module)


class RecordingObserver:
    def __init__(self) -> None:
        self.records: list[tuple[str, int]] = []

    def record(self, metric: str, value: int = 1) -> None:
        self.records.append((metric, value))


class StoreWaitObserveTest(unittest.TestCase):
    def test_tracked_count_does_not_query_or_drain_store_handles(self):
        cls = methods(
            "lmcache/v1/multiprocess/transfer_context/worker_transfer.py",
            "LMCacheDrivenTransferContext", {"tracked_store_count"}, {},
        )
        obj = cls()
        handles = [object(), object()]
        obj._inflight_stores = handles
        obj._inflight_lock = threading.Lock()
        self.assertEqual(obj.tracked_store_count(), 2)
        self.assertIs(obj._inflight_stores, handles)
        self.assertEqual(len(handles), 2)

    def test_repeated_store_rpc_callbacks_keep_their_own_timestamps(self):
        observer = observer_module.PPWaitObserver("store", "pp0")
        first = observer.store_submitted("r")
        second = observer.store_submitted("r")
        observer.store_rpc_ready(first)
        observer.store_rpc_ready(second)
        self.assertEqual(observer.snapshot()["store_submit_to_rpc_ns"][0], 2)
        observer.store_device_seen("r")
        observer.store_device_seen("r")
        observer.store_reported("r")
        observer.store_reported("r")
        self.assertEqual(observer.snapshot()["store_submit_to_device_seen_ns"][0], 1)
        self.assertEqual(observer.snapshot()["store_submit_to_get_finished_ns"][0], 1)

    def test_observer_opt_in_is_explicit_and_pp_scoped(self):
        previous = os.environ.pop("LMCACHE_PP_WAIT_OBSERVE", None)
        try:
            self.assertIsNone(observer_module.create_pp_wait_observer(True, "store", "pp0"))
            os.environ["LMCACHE_PP_WAIT_OBSERVE"] = "1"
            self.assertIsNone(observer_module.create_pp_wait_observer(False, "store", "pp0"))
            self.assertIsNotNone(observer_module.create_pp_wait_observer(True, "store", "pp0"))
        finally:
            if previous is None:
                os.environ.pop("LMCACHE_PP_WAIT_OBSERVE", None)
            else:
                os.environ["LMCACHE_PP_WAIT_OBSERVE"] = previous

    def test_device_future_pending_and_completion_report_remain_exactly_once(self):
        from test_pp_completion_contract import adapter, Future
        obj = adapter()
        observer = observer_module.PPWaitObserver("store", "pp0")
        obj._pp_wait_observer = observer
        observer.store_submitted("request")
        future = Future(ready=False)
        obj.store_futures["request"] = future
        self.assertFalse(obj.get_finished({"request"})[0])
        self.assertNotIn("store_submit_to_device_seen_ns", observer.snapshot())
        future.ready = True
        self.assertEqual(obj.get_finished(set())[0], {"request"})
        before = observer.snapshot()
        self.assertEqual(before["store_submit_to_device_seen_ns"][0], 1)
        self.assertEqual(before["store_submit_to_get_finished_ns"][0], 1)
        self.assertFalse(obj.get_finished(set())[0])
        self.assertEqual(observer.snapshot(), before)

    def test_host_block_hold_notice_does_not_claim_native_release(self):
        observer = observer_module.PPWaitObserver("blocks", "scheduler")
        observer.delay_free("r", 3)
        observer.delay_notice("r")
        observer.delay_notice("r")
        counts = observer.snapshot()
        self.assertEqual(counts["delay_free_blocks"], (1, 3, 3))
        self.assertEqual(counts["delay_free_to_worker_notice_ns"][0], 1)
        self.assertNotIn("native_block_release", counts)

    def worker(self, fail: str | None = None):
        clock = iter([100, 130, 150, 190])
        calls = []

        def flush():
            calls.append("flush")
            if fail == "flush":
                raise ValueError("flush failed")

        def sync():
            calls.append("sync")
            if fail == "sync":
                raise ValueError("sync failed")

        cls = methods(
            "lmcache/integration/vllm/vllm_multi_process_adapter.py",
            "LMCacheMPWorkerAdapter", {"handle_preemptions"},
            {"torch_dev": NS(synchronize=sync),
             "time": NS(monotonic_ns=lambda: next(clock)),
             "logger": logging.getLogger("observe-test")},
        )
        obj = cls()
        obj.is_healthy = True
        obj.transfer_ctx = NS(flush_inflight_stores=flush, tracked_store_count=2,
                              last_flush_pending_count=1)
        obj._pp_wait_observer = RecordingObserver()
        return obj, calls

    def test_flush_and_sync_have_separate_nonoverlapping_intervals(self):
        obj, calls = self.worker()
        self.assertIsNone(obj.handle_preemptions(True))
        self.assertEqual(calls, ["flush", "sync"])
        self.assertEqual(dict(obj._pp_wait_observer.records), {
            "flush_pending_tracked": 2, "flush_pending": 1,
            "flush_ns": 30, "sync_ns": 40,
        })

    def test_no_hint_never_flushes_syncs_or_records(self):
        obj, calls = self.worker()
        obj.handle_preemptions(False)
        self.assertEqual(calls, [])
        self.assertEqual(obj._pp_wait_observer.records, [])

    def test_disabled_observer_preserves_calls_without_reading_clock(self):
        obj, calls = self.worker()
        obj._pp_wait_observer = None
        obj.handle_preemptions(True)
        self.assertEqual(calls, ["flush", "sync"])

    def test_flush_exception_is_preserved_and_sync_is_skipped(self):
        obj, calls = self.worker("flush")
        with self.assertRaisesRegex(ValueError, "flush failed"):
            obj.handle_preemptions(True)
        self.assertEqual(calls, ["flush"])
        self.assertEqual(dict(obj._pp_wait_observer.records).get("flush_ns"), 30)

    def test_sync_exception_is_preserved(self):
        obj, calls = self.worker("sync")
        with self.assertRaisesRegex(ValueError, "sync failed"):
            obj.handle_preemptions(True)
        self.assertEqual(calls, ["flush", "sync"])
        self.assertEqual(dict(obj._pp_wait_observer.records).get("sync_ns"), 40)


if __name__ == "__main__":
    unittest.main()
