# SPDX-License-Identifier: Apache-2.0
"""Regression tests for L1 runtime cleanup during MP server startup."""

from types import SimpleNamespace

import pytest

from lmcache.v1.multiprocess import server as server_module
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.engine_module import DiscoverableModule
from lmcache.v1.mp_observability.config import ObservabilityConfig


def test_module_factory_failure_closes_previously_built_modules(monkeypatch):
    close_calls = []
    factory_error = RuntimeError("second module factory failed")

    class FirstModule(DiscoverableModule):
        module_name = "first"

        @classmethod
        def create(cls, build_ctx):
            return SimpleNamespace(close=lambda: close_calls.append("first"))

    class FailingModule(DiscoverableModule):
        module_name = "failing"

        @classmethod
        def create(cls, build_ctx):
            raise factory_error

    monkeypatch.setattr(
        server_module,
        "load_server_module_classes",
        lambda _: [(FirstModule, {}), (FailingModule, {})],
    )
    monkeypatch.setattr(server_module, "discover_modules", lambda: [])
    monkeypatch.setattr(server_module, "order_modules", lambda modules, _: modules)
    monkeypatch.setattr(server_module, "build_server_module_router", lambda *_: None)

    with pytest.raises(RuntimeError) as raised:
        server_module._build_server_components(
            object(), SimpleNamespace(server_modules=[]), object()
        )

    assert raised.value is factory_error
    assert close_calls == ["first"]


def test_request_server_start_failure_closes_server_then_engine_once(monkeypatch):
    cleanup_calls = []
    start_error = RuntimeError("request server start failed")
    runtime = object()

    class FakeContext:
        def __init__(self, **kwargs):
            assert kwargs["l1_runtime"] is runtime
            self.chunk_size = kwargs["chunk_size"]
            self.storage_manager = object()

        def add_chunk_size_bind_listener(self, _listener):
            pass

        def close(self):
            cleanup_calls.append("context.close")

    class FakeEngine:
        def __init__(self, ctx, modules):
            self.ctx = ctx
            assert modules == []

        def close(self):
            cleanup_calls.append("engine.close")
            self.ctx.close()

    class FakeRequestServer:
        def start(self):
            raise start_error

        def close(self):
            cleanup_calls.append("server.close")

    monkeypatch.setattr(server_module, "set_isolated_ipc", lambda _: None)
    monkeypatch.setattr(server_module, "init_observability", lambda *_, **__: object())
    monkeypatch.setattr(server_module, "init_gc_monitor", lambda _: None)
    monkeypatch.setattr(
        server_module, "maybe_initialize_trace_recorder", lambda *_, **__: None
    )
    monkeypatch.setattr(server_module, "shutdown_gc_monitor", lambda: None)
    monkeypatch.setattr(server_module, "MPCacheServerContext", FakeContext)
    monkeypatch.setattr(server_module, "MPCacheServer", FakeEngine)
    monkeypatch.setattr(
        server_module,
        "_build_server_components",
        lambda *_: server_module.ServerBuildComponents(modules=[]),
    )
    monkeypatch.setattr(server_module, "InitializeMPUsageContext", lambda *_: None)
    monkeypatch.setattr(server_module, "InitializeMPContinuousUsage", lambda *_: None)
    monkeypatch.setattr(server_module, "InitializeL2ConnectorUsage", lambda *_: None)
    monkeypatch.setattr(server_module, "InitializeL1Usage", lambda *_: None)
    monkeypatch.setattr(
        server_module, "create_request_server", lambda *_: FakeRequestServer()
    )
    monkeypatch.setattr(server_module, "torch_dev", SimpleNamespace(init=lambda: None))
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.l1_runtime.create_l1_runtime",
        lambda *_: runtime,
    )
    monkeypatch.setattr(
        "lmcache.v1.multiprocess.engine_context.MPCacheServerContext",
        FakeContext,
    )

    mp_config = MPServerConfig()
    storage_config = SimpleNamespace(l1_manager_configs=[])
    obs_config = ObservabilityConfig()
    coordinator_config = SimpleNamespace(url="")

    with pytest.raises(RuntimeError) as raised:
        server_module.run_cache_server(
            mp_config,
            storage_config,
            obs_config,
            return_engine=True,
            coordinator_config=coordinator_config,
        )

    assert raised.value is start_error
    assert cleanup_calls == ["server.close", "engine.close", "context.close"]
