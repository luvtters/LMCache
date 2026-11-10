"""Run actual worker source methods with simulated device futures and clock.

No framework/device imports: these check ownership and reporting only, not DMA.
"""
# Standard
import ast
import logging
from pathlib import Path
from types import SimpleNamespace

# Third Party
import pytest


@pytest.fixture
def worker():
    source = (
        Path(__file__).resolve().parents[2]
        / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
    )
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                 and node.name == "LMCacheMPWorkerAdapter")
    names = {"get_finished", "get_finished_with_lazy_offload",
             "_process_finished_stores", "_update_and_get_finished_store",
             "_check_unhealthy_transfers"}
    nodes = [node for node in owner.body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    for node in nodes:
        node.decorator_list = []
    clock = SimpleNamespace(now=0.0)
    namespace = {
        "time": SimpleNamespace(monotonic=lambda: clock.now),
        "logger": logging.getLogger(__name__),
        "get_read_policy": lambda: "transfer_borrow",
    }
    future_import = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future_import, *nodes], type_ignores=[])
    )
    exec(compile(module, str(source), "exec"), namespace)
    cls = type("ActualWorkerMethods", (), {node.name: namespace[node.name]
                                         for node in nodes})
    obj = cls()
    obj.dispatcher = None
    obj.is_healthy = False
    obj.lazy_offload = False
    obj._mq_timeout = 5.0
    obj.store_futures = {"store": DeviceFuture()}
    obj.retrieve_futures = {"retrieve": (DeviceFuture(), [7, 8])}
    obj.store_events = {"store": object()}
    obj.retrieve_events = {"retrieve": object()}
    obj._pending_store_kv_events = {"store": object()}
    obj.error_block_ids = set()
    obj._failed_request_ids = set()
    obj._failed_store_requests = set()
    obj._completed_store_requests = {}
    obj._dropped_retrieves = set()
    obj.finished_stores = set()
    obj.previously_finished = set()
    obj._returned_finished = set()
    obj.request_telemetry = SimpleNamespace(on_request_store_finished=lambda **kw: None)
    obj.model_name, obj.world_size, obj.worker_id = "test", 1, 0
    obj._publish_store_kv_events = lambda rid: obj._pending_store_kv_events.pop(rid)
    return obj, clock, namespace


class DeviceFuture:
    def __init__(self):
        self.ready = False
        self.value = True
        self.result_calls = 0

    def query(self):
        return self.ready

    def result(self, timeout=None):
        assert self.ready, "pending device work must not be waited on here"
        self.result_calls += 1
        return self.value


def poll(obj, lazy):
    obj.lazy_offload = lazy
    if lazy:
        return obj.get_finished_with_lazy_offload()
    return obj.get_finished({"store", "retrieve"})


@pytest.mark.parametrize("lazy", [False, True])
def test_unhealthy_keeps_pending_device_futures_and_events(worker, lazy):
    obj, _, _ = worker
    stores, retrieves = poll(obj, lazy)
    assert stores is None if lazy else stores == set()
    assert retrieves == set()
    assert "store" in obj.store_futures and "store" in obj.store_events
    assert "retrieve" in obj.retrieve_futures and "retrieve" in obj.retrieve_events
    assert "store" in obj._pending_store_kv_events
    assert not obj.error_block_ids
    assert not obj._failed_request_ids
    assert not obj._completed_store_requests


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("success", [False, True])
def test_unhealthy_consumes_only_device_confirmed_results(worker, lazy, success):
    obj, _, _ = worker
    store = obj.store_futures["store"]
    retrieve = obj.retrieve_futures["retrieve"][0]
    store.ready = retrieve.ready = True
    store.value = retrieve.value = success
    stores, retrieves = poll(obj, lazy)
    assert stores is None if lazy else stores == {"store"}
    assert retrieves == {"retrieve"}
    assert store.result_calls == retrieve.result_calls == 1
    assert not obj.store_futures and not obj.retrieve_futures
    assert not obj.store_events and not obj.retrieve_events
    assert obj.error_block_ids == (set() if success else {7, 8})
    assert obj._failed_request_ids == (set() if success else {"retrieve"})
    if lazy:
        assert obj._completed_store_requests == {"store": 1}
        assert obj._failed_store_requests == (set() if success else {"store"})


@pytest.mark.parametrize("lazy", [False, True])
def test_unknown_device_work_is_bounded_without_releasing_it(worker, lazy):
    obj, clock, _ = worker
    poll(obj, lazy)
    clock.now = 4.9
    poll(obj, lazy)
    clock.now = 5.1
    with pytest.raises(RuntimeError, match="device|transfer"):
        poll(obj, lazy)
    assert "store" in obj.store_futures and "store" in obj.store_events
    assert "retrieve" in obj.retrieve_futures and "retrieve" in obj.retrieve_events
    assert not obj.error_block_ids
    assert not obj._completed_store_requests


@pytest.mark.parametrize("lazy", [False, True])
def test_legacy_keeps_existing_unhealthy_drain(worker, lazy):
    obj, _, namespace = worker
    namespace["get_read_policy"] = lambda: "legacy"
    _, retrieves = poll(obj, lazy)
    assert retrieves == {"retrieve"}
    assert not obj.store_futures and not obj.retrieve_futures
    assert obj.error_block_ids == {7, 8}


@pytest.mark.parametrize("lazy", [False, True])
def test_healthy_pending_device_work_also_has_a_deadline(worker, lazy):
    obj, clock, _ = worker
    obj.is_healthy = True
    stores, retrieves = poll(obj, lazy)
    assert stores is None if lazy else stores == set()
    assert not retrieves
    clock.now = 100.0
    with pytest.raises(RuntimeError, match="device|transfer"):
        poll(obj, lazy)
    assert "store" in obj.store_futures
    assert "retrieve" in obj.retrieve_futures


@pytest.mark.parametrize("lazy", [False, True])
def test_health_recovery_does_not_restart_unknown_device_deadline(worker, lazy):
    obj, clock, _ = worker
    poll(obj, lazy)
    clock.now = 4.0
    obj.is_healthy = True
    poll(obj, lazy)
    clock.now = 5.1
    with pytest.raises(RuntimeError, match="device|transfer"):
        poll(obj, lazy)
    assert "store" in obj.store_futures and "retrieve" in obj.retrieve_futures


@pytest.mark.parametrize("lazy", [False, True])
def test_new_future_identity_gets_its_own_deadline(worker, lazy):
    obj, clock, _ = worker
    obj.retrieve_futures.clear()
    poll(obj, lazy)
    clock.now = 4.0
    obj.store_futures["store"] = DeviceFuture()
    poll(obj, lazy)
    clock.now = 6.0
    poll(obj, lazy)
    assert "store" in obj.store_futures
    clock.now = 9.1
    with pytest.raises(RuntimeError, match="device|transfer"):
        poll(obj, lazy)


@pytest.mark.parametrize("lazy", [False, True])
def test_device_ready_is_checked_before_expired_deadline(worker, lazy):
    obj, clock, _ = worker
    poll(obj, lazy)
    clock.now = 6.0
    obj.store_futures["store"].ready = True
    obj.retrieve_futures["retrieve"][0].ready = True
    _, retrieves = poll(obj, lazy)
    assert retrieves == {"retrieve"}
    assert not obj._transfer_pending_since


@pytest.mark.parametrize("lazy", [False, True])
def test_removed_futures_do_not_leave_deadlines(worker, lazy):
    obj, clock, _ = worker
    poll(obj, lazy)
    obj.store_futures.clear()
    obj.retrieve_futures.clear()
    clock.now = 6.0
    poll(obj, lazy)
    assert not obj._transfer_pending_since


@pytest.mark.parametrize("lazy", [False, True])
def test_completion_between_deadline_check_and_collection_clears_timer(worker, lazy):
    obj, _, _ = worker
    futures = [obj.store_futures["store"], obj.retrieve_futures["retrieve"][0]]
    for future in futures:
        answers = iter([False, True])
        future.ready = True
        future.query = lambda answers=answers: next(answers)
    poll(obj, lazy)
    assert not obj._transfer_pending_since
