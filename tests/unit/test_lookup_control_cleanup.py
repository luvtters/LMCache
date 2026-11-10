"""Actual scheduler source methods and cleanup helper; stdlib control futures."""
import ast
from concurrent.futures import Future
import importlib.util
from pathlib import Path
import threading
import weakref
from types import SimpleNamespace

import pytest


class ControlFuture(Future):
    def query(self):
        return self.done()

    def result(self, timeout=None):
        assert timeout == 0, "scheduler cleanup must never wait for control RPCs"
        return super().result(timeout=timeout)


@pytest.fixture
def adapter():
    root = Path(__file__).resolve().parents[2]
    path = root / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
    owner = next(node for node in ast.parse(path.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.ClassDef)
                 and node.name == "LMCacheMPSchedulerAdapter")
    names = {"cleanup_lookup_result", "end_session", "_retain_lookup_controls",
             "_check_lookup_reuse"}
    methods = [node for node in owner.body
               if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in methods:
        node.decorator_list = []
    namespace = {"get_read_policy": lambda: "transfer_borrow"}
    helper = root / "lmcache/v1/multiprocess/lookup_control_cleanup.py"
    if helper.exists():
        spec = importlib.util.spec_from_file_location("cleanup_test_helper", helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        namespace["LookupControlRegistry"] = module.LookupControlRegistry
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[]))
    exec(compile(tree, str(path), "exec"), namespace)
    cls = type("ActualSchedulerMethods", (), {node.name: namespace[node.name]
                                             for node in methods})
    obj = cls()
    obj.is_healthy = False
    obj._server_urls = ["slow", "fast"]
    sent = []
    endings = {url: ControlFuture() for url in obj._server_urls}

    def submit(url, request_id):
        sent.append((url, request_id))
        return endings[url]

    obj.req_clients = {url: SimpleNamespace(end_session=lambda rid, url=url: submit(url, rid))
                       for url in obj._server_urls}
    lookups = {url: ControlFuture() for url in obj._server_urls}
    status = {url: ControlFuture() for url in obj._server_urls}
    obj._pending_lookups = {"request"}
    obj._unacked_lookups = {"request": SimpleNamespace(futures=lookups, submitted_at=0)}
    obj._lookup_status = {"request": {url: (future, 0) for url, future in status.items()}}
    obj._finished_lookup_results = {"request": 16}
    obj._per_server_hits = {"request": {"slow": 1, "fast": 1}}
    obj._lookup_params = {"request": ([], "", None)}
    return obj, lookups, status, endings, sent


def test_admission_cleanup_retains_controls_without_ending_session(adapter):
    obj, lookups, _, _, sent = adapter
    obj.cleanup_lookup_result("request")
    assert not sent
    lookups["fast"].set_result(None)
    with pytest.raises(RuntimeError, match="request|quiescent"):
        obj._check_lookup_reuse("request")


def test_each_endpoint_ends_after_its_own_controls_without_global_health(adapter):
    obj, lookups, status, endings, sent = adapter
    obj.cleanup_lookup_result("request")
    obj.end_session("request")
    assert not sent
    lookups["fast"].set_result(None)
    assert not sent
    status["fast"].set_result(None)
    assert sent == [("fast", "request")]
    endings["fast"].set_result(None)
    with pytest.raises(RuntimeError):
        obj._check_lookup_reuse("request")
    status["slow"].set_result(None)
    lookups["slow"].set_result(None)
    assert sent == [("fast", "request"), ("slow", "request")]
    endings["slow"].set_result(None)
    obj._check_lookup_reuse("request")


def test_end_session_without_prior_cleanup_retains_and_orders_controls(adapter):
    obj, lookups, status, _, sent = adapter
    obj.end_session("request")
    assert not sent
    lookups["fast"].set_exception(RuntimeError("lookup failed"))
    status["fast"].set_result(None)
    assert sent == [("fast", "request")]


def test_failed_end_ack_keeps_request_id_quarantined(adapter):
    obj, lookups, status, endings, _ = adapter
    for future in [*lookups.values(), *status.values()]:
        future.set_result(None)
    obj.end_session("request")
    endings["fast"].set_exception(RuntimeError("end failed"))
    endings["slow"].set_result(None)
    with pytest.raises(RuntimeError):
        obj._check_lookup_reuse("request")


def test_duplicate_end_does_not_duplicate_rpcs(adapter):
    obj, lookups, status, _, sent = adapter
    obj.end_session("request")
    obj.end_session("request")
    for future in [*lookups.values(), *status.values()]:
        future.set_result(None)
    assert sorted(sent) == [("fast", "request"), ("slow", "request")]


def test_old_generation_callback_cannot_retire_reused_id(adapter):
    obj, lookups, status, endings, _ = adapter
    obj.end_session("request")
    old = obj._lookup_control_registry._states["request"]
    for future in [*lookups.values(), *status.values(), *endings.values()]:
        future.set_result(None)
    obj._check_lookup_reuse("request")
    obj.cleanup_lookup_result("request")
    old._end_complete("slow", endings["slow"])
    with pytest.raises(RuntimeError):
        obj._check_lookup_reuse("request")


def test_already_completed_controls_and_end_acks_are_reentrant(adapter):
    obj, lookups, status, endings, sent = adapter
    for future in [*lookups.values(), *status.values(), *endings.values()]:
        future.set_result(None)
    obj.end_session("request")
    assert sorted(sent) == [("fast", "request"), ("slow", "request")]
    obj._check_lookup_reuse("request")


def test_concurrent_control_callbacks_submit_each_end_once(adapter):
    obj, lookups, status, _, sent = adapter
    obj.cleanup_lookup_result("request")
    obj.end_session("request")
    barrier = threading.Barrier(5)

    def complete(future):
        barrier.wait(timeout=5)
        future.set_result(None)

    threads = [threading.Thread(target=complete, args=(future,), daemon=True)
               for future in [*lookups.values(), *status.values()]]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert sorted(sent) == [("fast", "request"), ("slow", "request")]


def test_cleanup_keeps_strong_control_references_until_end(adapter):
    obj, lookups, status, _, sent = adapter
    references = [weakref.ref(future) for future in [*lookups.values(), *status.values()]]
    obj.cleanup_lookup_result("request")
    lookups.clear()
    status.clear()
    assert all(reference() is not None for reference in references)
    obj.end_session("request")
    for reference in references:
        reference().set_result(None)
    assert sorted(sent) == [("fast", "request"), ("slow", "request")]


def test_cleanup_removes_lookup_deadline_without_ending_admitted_session(adapter):
    obj, _, _, _, sent = adapter
    obj._lookup_deadlines = {"request": 123.0}
    obj.cleanup_lookup_result("request")
    assert not obj._lookup_deadlines
    assert not sent
