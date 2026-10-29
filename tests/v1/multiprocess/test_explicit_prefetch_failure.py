# SPDX-License-Identifier: Apache-2.0
"""CPU failure contract with real ZMQ request/reply; no accelerator use.

Run directly with the project CPU-capable interpreter. Native Bitmap and the
normal ZMQ transport are used; BM/device work is not initialized.
"""
import threading
from types import SimpleNamespace
import unittest
import uuid

import zmq

from lmcache.integration.vllm.mp_lookup_state import LookupState
from lmcache.v1.distributed.storage_controllers.prefetch_controller import (
    PrefetchController, PrefetchKeyState,
)
from lmcache.v1.distributed.storage_controllers.utils import Bitmap2D
from lmcache.v1.multiprocess.futures import MessagingFuture, RemoteRequestError
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    MessageQueueClient, MessageQueueServer,
)


def failed_result():
    controller = PrefetchController.__new__(PrefetchController)
    lock = threading.Lock()
    controller._prefetch_results_lock = lock
    controller._prefetch_results_cv = threading.Condition(lock)
    controller._completed_results = {}
    states = PrefetchKeyState()
    cells = Bitmap2D.zeros(1, 1).ones_like()
    states.l2_locked_keys[0] = cells
    states.l1_reserved_keys[0] = cells
    states.l1_locked_keys[0] = cells
    request = SimpleNamespace(request_id=71, key_groups=[SimpleNamespace(keys=[1])],
                              key_states=states, num_kv_readers=2)
    controller._in_flight_requests = {71: request}
    released = []
    controller._l2_adapters = {0: SimpleNamespace(
        submit_unlock=lambda keys: released.append(("l2_unlock", keys)))}
    controller._l1_managers = {0: SimpleNamespace(
        finish_write_and_delete=lambda keys, **kw: released.append(("l1_delete", keys)),
        finish_read=lambda keys, **kw: released.append(
            ("l1_unlock", keys, kw["read_locks"])))}
    waiting, notified = threading.Event(), threading.Event()

    def waiter():
        with controller._prefetch_results_cv:
            waiting.set()
            if controller._prefetch_results_cv.wait_for(
                lambda: 71 in controller._completed_results, timeout=2
            ):
                notified.set()

    thread = threading.Thread(target=waiter)
    thread.start()
    assert waiting.wait(2)
    controller._abort_request(request, RuntimeError("BM peer rank 1 left"))
    thread.join(2)
    assert not thread.is_alive() and notified.is_set()
    assert released == [("l2_unlock", [1]), ("l1_delete", [1]), ("l1_unlock", [1], 2)]
    assert controller._in_flight_requests == {}
    assert list(request.key_states.l2_locked_keys.items()) == []
    return controller._completed_results[71]


class FakeClient:
    def __init__(self):
        self.lookups = []
        self.queries = []

    def lookup(self, key, tp_size):
        future = MessagingFuture()
        self.lookups.append(future)
        return future

    def query_prefetch_status(self, request_id):
        future = MessagingFuture()
        self.queries.append(future)
        return future


def lookup_state(client):
    return LookupState(
        SimpleNamespace(request_id="failed-request"), {"cpu-server": client},
        tp_size=1, tokens_per_chunk=128, timeout=.5, blocking_lookup=False,
    )


class ExplicitPrefetchFailureChecks(unittest.TestCase):
    def test_abort_releases_publishes_failure_wakes_waiter_and_retires(self):
        result = failed_result()
        self.assertIn("BM peer rank 1 left", result.error)
        self.assertEqual(sum(row.popcount() for row in result.hit_cells), 0)

    def test_lookup_consumes_failure_job_before_raising(self):
        module = LookupModule.__new__(LookupModule)
        job = SimpleNamespace(request_id="failed-request", handle=71)
        module._ctx = SimpleNamespace(storage_manager=SimpleNamespace(
            query_prefetch_status=lambda handle: failed_result()
        ))
        module._prefetch_job_lock = threading.Lock()
        module._prefetch_jobs = {job.request_id: job}
        with self.assertRaisesRegex(RuntimeError, "BM peer rank 1 left"):
            module._consume_prefetch_result(job)
        self.assertEqual(module._prefetch_jobs, {})

    def test_remote_prefetch_failure_is_terminal_without_requery_or_miss(self):
        client = FakeClient()
        state = lookup_state(client)
        state.submit()
        client.lookups[0].set_result(None)
        deadline = state._servers["cpu-server"].deadline
        client.queries[0].set_exception(RemoteRequestError("BM peer rank 1 left"))
        for _ in range(5):
            self.assertEqual(state.on_scheduler_check(), [])
            with self.assertRaisesRegex(RemoteRequestError, "BM peer rank 1 left"):
                state.get_result()
        self.assertEqual(len(client.queries), 1)
        self.assertEqual(state._servers["cpu-server"].deadline, deadline)
        state.close()
        with self.assertRaises(RemoteRequestError):
            state.get_result()

    def test_remote_lookup_failure_never_starts_prefetch_poll(self):
        client = FakeClient()
        state = lookup_state(client)
        state.submit()
        client.lookups[0].set_exception(RemoteRequestError("peer left"))
        self.assertEqual(state.on_scheduler_check(), [])
        self.assertEqual(client.queries, [])
        with self.assertRaisesRegex(RemoteRequestError, "peer left"):
            state.get_result()

    def test_normal_pending_and_zero_miss_are_unchanged(self):
        client = FakeClient()
        state = lookup_state(client)
        state.submit()
        client.lookups[0].set_result(None)
        client.queries[0].set_result(None)
        self.assertIsNone(state.get_result())
        state.on_scheduler_check()
        self.assertEqual(len(client.queries), 2)
        client.queries[1].set_result(0)
        self.assertEqual(state.get_result(), 0)

    def test_real_zmq_failure_reply_clears_pending_and_preserves_success(self):
        context = zmq.Context()
        server = MessageQueueServer("inproc://failure-" + uuid.uuid4().hex, context)

        def query(request_id: str) -> int | None:
            if request_id == "failure":
                module = LookupModule.__new__(LookupModule)
                module._ctx = SimpleNamespace(storage_manager=SimpleNamespace(
                    query_prefetch_status=lambda handle: failed_result()))
                module._prefetch_job_lock = threading.Lock()
                job = SimpleNamespace(request_id="failed-request", handle=71)
                module._prefetch_jobs = {job.request_id: job}
                try:
                    return module._consume_prefetch_result(job)
                finally:
                    assert module._prefetch_jobs == {}
            return {"pending": None, "miss": 0, "hit": 3}[request_id]

        def wait(request_id: str, timeout: float) -> int | None:
            raise RuntimeError("BM load failed")

        server.add_handler("query_prefetch_status", HandlerType.BLOCKING, query)
        server.add_normal_thread_pool(["query_prefetch_status"], max_workers=1)
        server.add_handler("wait_prefetch_status", HandlerType.SERIALIZED, wait)
        server.start()
        client = MessageQueueClient(
            server.socket.getsockopt_string(zmq.LAST_ENDPOINT), context)
        try:
            for operation, args, message in [
                ("query_prefetch_status", ["failure"], "BM peer rank 1 left"),
                ("wait_prefetch_status", ["failure", .5], "BM load failed"),
            ]:
                future = client.submit_request(operation, args)
                self.assertTrue(future.wait(timeout=2), "RPC failure reply was lost")
                with self.assertRaisesRegex(RemoteRequestError, message):
                    future.result(timeout=0)
                self.assertEqual(client.pending_futures, {})
            for request_id, expected in [("pending", None), ("miss", 0), ("hit", 3)]:
                future = client.submit_request("query_prefetch_status", [request_id])
                self.assertTrue(future.wait(timeout=2))
                self.assertEqual(future.result(timeout=0), expected)
                self.assertEqual(client.pending_futures, {})
        finally:
            client.close()
            server.close()
            context.term()


if __name__ == "__main__":
    unittest.main(verbosity=2)
