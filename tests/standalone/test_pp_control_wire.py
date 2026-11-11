# SPDX-License-Identifier: Apache-2.0
"""Real same-host TCP/ZMQ control gates; no devices or storage service.

Production transport, notifier, plain futures, codec and scheduler methods are
executed from source. Device-only imports/types are excluded by the AST loader.
LOOKUP keys use str instead of IPCCacheServerKey; IPC/device payloads, actual
storage, cross-host networking and performance are outside this gate.
"""
from __future__ import annotations

import ast
import json
import logging
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace as NS
import unittest

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else None
SOURCES = globals().get("KS_SOURCES", {})
RESULTS = []


def source(path):
    return SOURCES[path] if path in SOURCES else (ROOT / path).read_text(encoding="utf8")


def load(path, namespace=None, only=None, exclude=()):
    tree = ast.parse(source(path))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            if any(name.startswith("lmcache") for name in names):
                continue
        elif only is not None and getattr(node, "name", None) not in only:
            if not (isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in only for t in node.targets
            )):
                continue
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in exclude for t in node.targets
        ):
            continue
        if isinstance(node, ast.ClassDef) and node.name in exclude:
            continue
        nodes.append(node)
    module = ModuleType("wire_gate_" + str(len(sys.modules)))
    sys.modules[module.__name__] = module
    module.__dict__.update(namespace or {})
    code = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes], type_ignores=[])
    exec(compile(ast.fix_missing_locations(code), path, "exec"), module.__dict__)
    return module


def transport():
    base = "lmcache/v1/multiprocess/"
    common = {"init_logger": logging.getLogger, "LMCacheTimeoutError": TimeoutError,
              "lmcache_deprecate": lambda *a: lambda fn: fn,
              "create_pp_wait_observer": lambda *a: None, "PPWaitObserver": object,
              "time": time}
    futures = load(base + "futures.py", common, only={"MessagingFuture", "MessagingStream", "T", "logger"})
    common.update(MessagingFuture=futures.MessagingFuture, MessagingStream=futures.MessagingStream)
    rpc = load(base + "rpc.py", common)
    common.update(RpcOperation=str, RpcSpec=rpc.RpcSpec)
    handler = load(base + "request_handler.py", common, only={"HandlerType"})
    notifier = load("lmcache/v1/platform/event_notifier.py")
    wire = load(base + "transport/zmq_impl/wire.py", common)
    # Build actual RPC specs from actual RequestClient signatures, substituting
    # only the LOOKUP device key type with the documented CPU string payload.
    contracts = load(base + "transport/base.py", {
        **common, "rpc_method": rpc.rpc_method, "IPCCacheServerKey": str,
    }, only={"RequestClient"})
    specs = {name: rpc._build_rpc_spec(name, getattr(contracts.RequestClient, name))
             for name in ("lookup", "query_prefetch_status", "end_session", "ping")}
    common.update(HandlerType=handler.HandlerType, EventNotifier=notifier.EventNotifier,
                  create_event_notifier=notifier.create_event_notifier,
                  encode_operation=wire.encode_operation, decode_operation=wire.decode_operation,
                  get_rpc_spec=specs.__getitem__, RequestServer=object,
                  AffinityThreadPool=type("ExcludedAffinityPool", (), {}),
                  _SPECIAL_ENCODER_DECODERS={})
    mq = load(base + "transport/zmq_impl/mq.py", common, exclude={"_SPECIAL_ENCODER_DECODERS"})
    return mq, specs


def until(condition, timeout=3):
    end = time.monotonic() + timeout
    while not condition():
        if time.monotonic() >= end:
            raise AssertionError("bounded CPU gate timed out")
        time.sleep(0.005)


class Harness:
    def __init__(self, budget=32):
        self.mq, specs = transport()
        self.boundaries = []
        self.pause_outbound = False
        self.outbound_blocked = threading.Event()
        self.outbound_release = threading.Event()
        self.ping_gate = threading.Event()
        self.ping_gate.set()
        original_outbound = self.mq.MessageQueueClient.process_outbound_task
        original_inbound = self.mq.MessageQueueClient.process_inbound
        original_submit = self.mq.MessageQueueClient.submit_request

        def outbound(client, **kwargs):
            queued = client.input_queue.qsize()
            if self.pause_outbound and queued and client is self.clients[0]:
                self.outbound_blocked.set()
                if not self.outbound_release.wait(3):
                    raise TimeoutError("test outbound gate not released")
                queued = client.input_queue.qsize()
            started = time.monotonic()
            result = original_outbound(client, **kwargs)
            self.boundaries.append(("send_batch", queued, client.input_queue.qsize(), started, time.monotonic()))
            return result

        def inbound(client):
            started = time.monotonic()
            result = original_inbound(client)
            self.boundaries.append(("receive", sum(c.input_queue.qsize() for c in self.clients), started, time.monotonic()))
            return result

        def submit(client, operation, payloads):
            started = time.monotonic()
            future = original_submit(client, operation, payloads)
            future.add_done_callback(lambda done: self.boundaries.append(
                ("future_complete", operation, started, time.monotonic())
            ))
            return future

        self.mq.MessageQueueClient.process_outbound_task = outbound
        self.mq.MessageQueueClient.process_inbound = inbound
        self.mq.MessageQueueClient.submit_request = submit
        self.context = self.mq.zmq.Context()
        # Disconnected-peer tests must not leave unsent test frames lingering
        # at context teardown. Production reconnect/close policy is not changed.
        self.context.setsockopt(self.mq.zmq.LINGER, 0)
        self.servers, self.clients, self.events = [], [], []
        self.ack_gates = [threading.Event(), threading.Event()]
        self.ack_gates[0].set()
        self.hit = [0, 2]
        for stage in range(2):
            server = self.mq.MessageQueueServer("tcp://127.0.0.1:*", self.context)
            url = server.socket.getsockopt_string(self.mq.zmq.LAST_ENDPOINT)

            def lookup(rid, readers, stage=stage):
                self.events.append((stage, "lookup_enter", rid, time.monotonic()))
                if not self.ack_gates[stage].wait(3):
                    raise TimeoutError("test ACK gate not released")
                self.events.append((stage, "lookup_return", rid, time.monotonic()))

            def status(rid, stage=stage):
                self.events.append((stage, "status", rid, time.monotonic()))
                return self.hit[stage]

            def end(rid, stage=stage):
                self.events.append((stage, "end", rid, time.monotonic()))

            def ping(instance, stage=stage):
                self.events.append((stage, "ping_enter", instance, time.monotonic()))
                if not self.ping_gate.wait(3):
                    raise TimeoutError("test ping gate not released")
                self.events.append((stage, "ping_return", instance, time.monotonic()))
                return True

            server.add_blocking_handler(specs["lookup"], lookup)
            server.add_sync_handler(specs["query_prefetch_status"], status)
            server.add_sync_handler(specs["end_session"], end)
            server.add_blocking_handler(specs["ping"], ping)
            server.add_normal_thread_pool(["lookup", "ping"], max_workers=1)
            server.start()
            client = self.mq.MessageQueueClient(url, self.context, outbound_batch_limit=budget)
            self.servers.append(server)
            self.clients.append(client)

    def scheduler(self, rid="r"):
        registry = load("lmcache/v1/multiprocess/lookup_control_cleanup.py")
        names = {"check_lookup_result", "cleanup_lookup_result", "_retain_lookup_controls",
                 "_check_lookup_reuse", "end_session"}
        tree = ast.parse(source("lmcache/integration/vllm/vllm_multi_process_adapter.py"))
        owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "LMCacheMPSchedulerAdapter")
        methods = [n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name in names]
        for method in methods:
            method.decorator_list = []
        namespace = {"time": time, "get_read_policy": lambda: "transfer_borrow",
                     "LookupControlRegistry": registry.LookupControlRegistry,
                     "logger": logging.getLogger("wire-scheduler")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *methods], type_ignores=[])), "scheduler-source", "exec"), namespace)
        obj = type("ActualScheduler", (), {n.name: namespace[n.name] for n in methods})()
        obj._pp_query_fastpath = True
        obj._pending_lookups = {rid}
        obj._finished_lookup_results = {}
        obj._per_server_hits = {}
        obj._lookup_status = {}
        obj._lookup_params = {rid: ([], "", None)}
        obj._lookup_deadlines = {rid: time.monotonic() + 3}
        obj._server_urls = ["s0", "s1"]
        obj._nonblocking_lookup_status = True
        obj._mq_timeout = 3
        obj.is_healthy = True
        obj.lmcache_tokens_per_chunk = 4
        obj._mark_lookup_timed_out = lambda url: None
        obj._free_inconsistent_lookup_locks = lambda *args: None
        obj.req_clients = {url: NS(
            query_prefetch_status=lambda rid, c=c: c.submit_request("query_prefetch_status", [rid]),
            end_session=lambda rid, c=c: c.submit_request("end_session", [rid]),
        ) for url, c in zip(obj._server_urls, self.clients)}
        obj._unacked_lookups = {rid: NS(
            futures={url: c.submit_request("lookup", [rid, 1]) for url, c in zip(obj._server_urls, self.clients)},
            submitted_at=time.monotonic(),
        )}
        return obj

    def close(self):
        self.outbound_release.set()
        self.ping_gate.set()
        for gate in self.ack_gates:
            gate.set()
        time.sleep(0.02)
        for client in self.clients:
            client.close()
        for server in self.servers:
            server.close()
        self.context.term()


class WireGate(unittest.TestCase):
    def test_cancel_after_tcp_disconnect_keeps_unknown_generation(self):
        h = Harness()
        try:
            h.hit[0] = 3
            h.ack_gates[1].set()
            obj = h.scheduler()
            until(lambda: all(f.query() for f in obj._unacked_lookups["r"].futures.values()))
            # Both LOOKUP replies arrived; then the peer disappears before STATUS.
            stopped = h.servers.pop()
            stopped.close()
            self.assertIsNone(obj.check_lookup_result("r"))
            until(lambda: "s0" in obj._per_server_hits["r"] or obj.check_lookup_result("r") is not None)
            obj.cleanup_lookup_result("r")
            obj.end_session("r")
            until(lambda: any(e[0] == 0 and e[1] == "end" for e in h.events))
            self.assertFalse(obj._check_lookup_reuse("r", resume_existing_session=True))
            with self.assertRaises(RuntimeError):
                obj._check_lookup_reuse("r")
            self.assertFalse(any(e[0] == 1 and e[1] == "end" for e in h.events))
            RESULTS.append({"gate": "cancel_disconnect_quarantine", "events": h.events})
        finally:
            h.close()

    def test_early_miss_delayed_ack_cleanup_end_reuse(self):
        h = Harness()
        try:
            obj = h.scheduler()
            until(lambda: obj.check_lookup_result("r") == 0)
            self.assertFalse(any(e[0] == 1 and e[1] == "status" for e in h.events))
            self.assertEqual(obj._finished_lookup_results["r"], 0)
            self.assertIn("s1", obj._unacked_lookups["r"].futures)
            obj.cleanup_lookup_result("r")
            self.assertFalse(any(e[1] == "end" for e in h.events))
            self.assertFalse(obj._check_lookup_reuse("r", resume_existing_session=True))
            obj.end_session("r")
            until(lambda: any(e[0] == 0 and e[1] == "end" for e in h.events))
            self.assertFalse(any(e[0] == 1 and e[1] == "end" for e in h.events))
            h.ack_gates[1].set()
            until(lambda: any(e[0] == 1 and e[1] == "end" for e in h.events))
            def reusable():
                try:
                    return obj._check_lookup_reuse("r")
                except RuntimeError:
                    return False
            until(reusable)
            self.assertEqual(sum(e[1] == "end" for e in h.events), 2)
            RESULTS.append({"gate": "early_miss_ack_end_reuse", "events": h.events})
        finally:
            h.close()

    def test_positive_min_waits_for_both_and_none_is_pending(self):
        h = Harness()
        try:
            h.hit = [None, 2]
            obj = h.scheduler()
            def first_status():
                obj.check_lookup_result("r")
                return any(e[0] == 0 and e[1] == "status" for e in h.events)
            until(first_status)
            self.assertIsNone(obj.check_lookup_result("r"))
            h.hit[0] = 3
            self.assertIsNone(obj.check_lookup_result("r"))
            h.ack_gates[1].set()
            until(lambda: obj.check_lookup_result("r") == 8)
            self.assertEqual(obj._finished_lookup_results["r"], 8)
            obj.end_session("r")
            RESULTS.append({"gate": "positive_min_none", "events": h.events})
        finally:
            h.close()

    def test_fixed_load_budget_zero_and_32_complete(self):
        for budget in (0, 32):
            h = Harness(budget)
            try:
                start = time.monotonic()
                h.ping_gate.clear()
                peer = h.clients[1].submit_request("ping", [999])
                until(lambda: any(e[0] == 1 and e[1] == "ping_enter" for e in h.events))
                h.pause_outbound = True
                futures = [h.clients[0].submit_request("ping", [i]) for i in range(256)]
                self.assertTrue(h.outbound_blocked.wait(3))
                h.ping_gate.set()
                until(lambda: any(e[0] == 1 and e[1] == "ping_return" for e in h.events))
                time.sleep(0.02)
                h.outbound_release.set()
                self.assertTrue(peer.result(timeout=3))
                self.assertTrue(all(f.result(timeout=3) for f in futures))
                if budget:
                    batches = [r for r in h.boundaries if r[0] == "send_batch"]
                    self.assertTrue(all(r[1] - r[2] <= budget for r in batches))
                    self.assertTrue(any(r[0] == "receive" and r[1] > 0 for r in h.boundaries))
                RESULTS.append({"gate": "fixed_load", "budget": budget,
                                "requests": len(futures), "elapsed_seconds": time.monotonic() - start,
                                "boundaries": h.boundaries})
            finally:
                h.close()


if __name__ == "__main__":
    print(json.dumps({"run_marker": "ks-pp-wire-g3", "pid": os.getpid()}), flush=True)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(WireGate)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    print(json.dumps({"success": result.wasSuccessful(), "tests": result.testsRun,
                      "results": RESULTS, "device_imports": [n for n in sys.modules if n.split(".")[0] in {"torch", "torch_npu", "acl", "cupy"}]}))
    if not result.wasSuccessful():
        raise SystemExit(1)
