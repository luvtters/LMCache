# SPDX-License-Identifier: Apache-2.0
"""CPU queue timing must retain handler returns and exception propagation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
import time
from types import SimpleNamespace as NS
import unittest

from test_lookup_deadline_contract import methods


class ControlObserveTest(unittest.TestCase):
    def test_handler_exception_is_returned_by_same_future(self):
        cls = methods(
            "lmcache/v1/multiprocess/transport/zmq_impl/mq.py",
            "BlockingRequestHandler", {"__call__"}, {
                "unwrap_request_payloads": lambda values, classes: values,
                "AffinityThreadPool": type("UnusedAffinityPool", (), {}), "time": time,
            },
        )
        for observed in (False, True):
            seen = []
            obj = cls()
            obj.observer = NS(record=lambda name, value=1: seen.append((name, value))) if observed else None
            obj.operation, obj.payload_clss = "lookup", []
            def failing():
                raise ValueError("handler failure")
            obj.handler = failing
            with ThreadPoolExecutor(max_workers=1) as pool:
                obj.executor = pool
                with self.assertRaisesRegex(ValueError, "handler failure"):
                    obj([]).result(1)
            self.assertEqual(len(seen), 2 if observed else 0)

    def test_queue_wait_and_handler_time_are_recorded(self):
        cls = methods(
            "lmcache/v1/multiprocess/transport/zmq_impl/mq.py",
            "BlockingRequestHandler", {"__call__"}, {
                "unwrap_request_payloads": lambda values, classes: values,
                "AffinityThreadPool": type("UnusedAffinityPool", (), {}),
                "time": time,
            },
        )
        obj = cls()
        seen = []
        obj.observer = NS(record=lambda name, value=1: seen.append((name, value)))
        obj.operation = "ping"
        obj.payload_clss = [int]
        obj.handler = lambda value: value + 1
        gate, started = threading.Event(), threading.Event()
        with ThreadPoolExecutor(max_workers=1) as pool:
            def occupied():
                started.set()
                gate.wait(2)
            blocked = pool.submit(occupied)
            self.assertTrue(started.wait(1))
            obj.executor = pool
            reply = obj([2])
            self.assertFalse(reply.done())
            gate.set()
            blocked.result(1)
            self.assertEqual(reply.result(1), 3)
        self.assertEqual([name for name, _ in seen], ["ping.queue_ns", "ping.handler_ns"])
        self.assertTrue(all(value >= 0 for _, value in seen))


if __name__ == "__main__":
    unittest.main()
