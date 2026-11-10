# SPDX-License-Identifier: Apache-2.0
# Standard
import gc
import logging
import threading
from types import SimpleNamespace
import weakref

# Third Party
import pytest

# First Party
from lmcache.v1.multiprocess.futures import DeviceMessagingFuture, MessagingFuture
from lmcache.v1.multiprocess.lookup_control_cleanup import LookupControlRegistry


@pytest.mark.parametrize("completed_first", [False, True])
@pytest.mark.parametrize("failed", [False, True])
def test_callbacks_observe_published_terminal_state(completed_first, failed):
    future = MessagingFuture()
    error = RuntimeError("rpc failed")
    seen = []

    def callback(done):
        assert done is future
        assert done.query() and done.wait(0)
        if failed:
            with pytest.raises(RuntimeError) as raised:
                done.result(0)
            assert raised.value is error
        else:
            assert done.result(0) == 42
        seen.append(done)

    def finish():
        if failed:
            future.set_exception(error)
        else:
            future.set_result(42)

    if completed_first:
        finish()
    future.add_done_callback(callback)
    if not completed_first:
        assert not seen
        finish()
    assert seen == [future]


@pytest.mark.parametrize("failed", [False, True])
def test_registration_completion_race_runs_each_registration_once(failed):
    for _ in range(25):
        future = MessagingFuture()
        barrier = threading.Barrier(9)
        seen = []
        lock = threading.Lock()

        def callback(done):
            assert done.query()
            with lock:
                seen.append(done)

        def register():
            barrier.wait(timeout=5)
            future.add_done_callback(callback)

        threads = [threading.Thread(target=register, daemon=True) for _ in range(8)]
        for thread in threads:
            thread.start()
        barrier.wait(timeout=5)
        if failed:
            future.set_exception(RuntimeError("failed"))
        else:
            future.set_result(42)
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
        assert seen == [future] * 8


def test_callbacks_allow_reentrant_registration_and_completion():
    future = MessagingFuture()
    seen = []

    def callback(done):
        seen.append(done.result(0))
        done.add_done_callback(lambda completed: seen.append(completed.result(0)))
        done.set_result("late")
        done.set_exception(RuntimeError("late"))

    future.add_done_callback(callback)
    thread = threading.Thread(target=future.set_result, args=(42,), daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "callback ran under the internal lock"
    assert seen == [42, 42]
    assert future.result(0) == 42


@pytest.mark.parametrize("completed_first", [False, True])
@pytest.mark.parametrize("error_type", [ValueError, SystemExit])
def test_callback_failure_is_logged_and_does_not_stop_other_callbacks(
    completed_first, error_type, caplog, monkeypatch
):
    logger = logging.getLogger("lmcache.v1.multiprocess.futures")
    monkeypatch.setattr(logger, "handlers", [*logger.handlers, caplog.handler])
    future = MessagingFuture()
    seen = []

    def broken_callback(done):
        raise error_type("callback failure sentinel")

    if completed_first:
        future.set_result(42)
    future.add_done_callback(broken_callback)
    future.add_done_callback(lambda done: seen.append(done.result(0)))
    if not completed_first:
        future.set_result(42)
    assert seen == [42]
    assert "callback failure sentinel" in caplog.text


@pytest.mark.parametrize("first_failed", [False, True])
def test_duplicate_completion_preserves_first_terminal_value(first_failed):
    future = MessagingFuture()
    seen = []
    future.add_done_callback(lambda done: seen.append(done))
    error = RuntimeError("first failure")
    if first_failed:
        future.set_exception(error)
    else:
        future.set_result(42)
    future.set_result("late")
    future.set_exception(RuntimeError("late failure"))
    future.add_done_callback(lambda done: seen.append(done))
    assert seen == [future, future]
    if first_failed:
        with pytest.raises(RuntimeError) as raised:
            future.result(0)
        assert raised.value is error
    else:
        assert future.result(0) == 42


def test_completion_does_not_release_retained_resources():
    class Resource:
        pass

    resource = Resource()
    reference = weakref.ref(resource)
    future = MessagingFuture()
    future.retain_reference(resource)
    del resource
    future.add_done_callback(lambda done: None)
    future.set_result(42)
    gc.collect()
    assert reference() is not None
    future.release_references()
    gc.collect()
    assert reference() is None


def test_success_exception_race_has_one_terminal_value_and_callback():
    future = MessagingFuture()
    barrier = threading.Barrier(3)
    seen = []
    error = RuntimeError("competing failure")
    future.add_done_callback(lambda done: seen.append(done))

    def complete(failed):
        barrier.wait(timeout=5)
        if failed:
            future.set_exception(error)
        else:
            future.set_result(42)

    threads = [
        threading.Thread(target=complete, args=(failed,), daemon=True)
        for failed in (False, True)
    ]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert seen == [future]
    if future.exception_ is None:
        assert future.result(0) == 42
    else:
        with pytest.raises(RuntimeError) as raised:
            future.result(0)
        assert raised.value is error
        assert future.result_ is None


def test_device_future_rejects_callbacks_before_and_after_raw_rpc_completion():
    class PendingBackend:
        def import_event(self, handle, device):
            return object()

        def query_event(self, event):
            return False

    raw = MessagingFuture()
    device = DeviceMessagingFuture(raw, device=0, event_backend=PendingBackend())
    seen = []
    with pytest.raises(NotImplementedError, match="device"):
        device.add_done_callback(lambda done: seen.append(done))
    raw.set_result((b"pending-device-event", 42))
    assert raw.query()
    assert not device.query()
    with pytest.raises(NotImplementedError, match="device"):
        device.add_done_callback(lambda done: seen.append(done))
    assert not seen


@pytest.mark.parametrize("completed_first", [False, True])
def test_actual_control_callbacks_order_endpoint_cleanup(completed_first):
    registry = LookupControlRegistry()
    controls = {url: MessagingFuture() for url in ("slow", "fast")}
    endings = {url: MessagingFuture() for url in controls}
    sent = []

    def end(url, request_id):
        sent.append((url, request_id))
        return endings[url]

    clients = {url: SimpleNamespace(end_session=lambda rid, url=url: end(url, rid))
               for url in controls}
    if completed_first:
        controls["fast"].set_result(None)
    registry.retain("request", clients, {url: [future] for url, future in controls.items()})
    registry.end("request")
    if not completed_first:
        controls["fast"].set_result(None)
    assert sent == [("fast", "request")]
    endings["fast"].set_result(None)
    with pytest.raises(RuntimeError):
        registry.check_reuse("request")
    controls["slow"].set_result(None)
    assert sent == [("fast", "request"), ("slow", "request")]
    endings["slow"].set_result(None)
    registry.check_reuse("request")
