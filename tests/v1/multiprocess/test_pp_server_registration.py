# SPDX-License-Identifier: Apache-2.0
"""CPU handler/lifecycle tests; runnable directly without native extensions.

Run: python tests/v1/multiprocess/test_pp_server_registration.py
The production class and registry are compiled from their original AST, with
only imports removed. Device contexts/events/dispatcher are explicit fixtures;
no device transfer, transport serialization or performance is tested here.
"""

# Standard
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, get_args
from unittest.mock import MagicMock, patch
import ast
import importlib.util
import json
import threading
import time
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[3]


def load_metadata() -> Any:
    """Load stdlib DTOs directly, avoiding package device initialization."""
    import sys

    spec = importlib.util.spec_from_file_location(
        "pp_server_test_metadata", ROOT / "lmcache/v1/multiprocess/pp_layout.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_production(path: str, names: set[str], namespace: dict[str, Any]) -> None:
    """Execute unmodified production definitions with explicit CPU dependencies."""
    source = ROOT / path
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(source), "exec", flags=0x1000000), namespace)


class PPServerRegistrationTest(unittest.TestCase):
    """Exercise real handlers, transactional registry refs and liveness removal."""

    def setUp(self) -> None:
        """Use a two-layer CPU layout and the actual reference-counted registry."""
        self.dto = load_metadata()
        self.contexts: list[Any] = []
        self.dtype = "float16"
        self.capacity = 50
        self.gate = SimpleNamespace(SYNC=0, BLOCKING=1, NON_BLOCKING=2)
        self.namespace = dict(
            dataclass=dataclass,
            replace=replace,
            threading=threading,
            time=time,
            uuid=uuid,
            json=json,
            InstanceLivenessTarget=object,
            request_handler=lambda *args, **kwargs: lambda function: function,
            _lmcache_nvtx_annotate=lambda function: function,
            HandlerType=self.gate,
            logger=MagicMock(),
            DeviceHostFuncDispatcher=MagicMock,
            torch_dev=MagicMock(),
            get_event_ipc_backend=lambda device: MagicMock(),
            create_cache_context=self.make_context,
            **{
                name: getattr(self.dto, name)
                for name in (
                    "PPRegistrationAck",
                    "PPRegistrationInfo",
                    "PPStageLocator",
                    "PPStageProfile",
                    "PPStageStatus",
                )
            },
        )
        self.namespace.update(
            GroupKind=Literal["attention", "recurrent", "aux"], get_args=get_args
        )
        load_production(
            "lmcache/v1/distributed/api.py", {"AttnWindowDesc"}, self.namespace
        )
        self.namespace["DEFAULT_ATTN_WINDOW_DESC"] = self.namespace["AttnWindowDesc"](
            num_chunks_in_sw=[-1]
        )
        load_production(
            "lmcache/v1/multiprocess/engine_context.py",
            {"_LayoutDescEntry", "LayoutDescRegistry"},
            self.namespace,
        )
        self.ctx = SimpleNamespace(
            chunk_size=16,
            separate_object_groups=False,
            full_sw_kv=False,
            layout_desc_registry=self.namespace["LayoutDescRegistry"](),
            storage_manager=MagicMock(),
        )
        load_production(
            "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py",
            {"ContextEntry", "get_pp_layout_json", "LMCacheDrivenTransferModule"},
            self.namespace,
        )
        self.namespace["ObjectKey"] = object
        self.namespace["L1WriteCompletion"] = object
        self.namespace["get_layout_desc"] = lambda context, *args, **kwargs: (
            context.layout
        )
        self.module = self.namespace["LMCacheDrivenTransferModule"](self.ctx)
        self.locator = self.dto.PPStageLocator("model", 2, 2, 0, ((0, 2), (2, 4)))

    def tearDown(self) -> None:
        """Close registered resources via the real lifecycle path."""
        self.module.close()

    def make_context(self, *args: Any, **kwargs: Any) -> Any:
        """Device fixture; capacity intentionally differs from chunk geometry."""
        group = SimpleNamespace(
            layer_indices=[0, 1],
            engine_kv_format="TEST_FORMAT",
            engine_group_idx=0,
            tokens_per_block=8,
            shape_desc=SimpleNamespace(bs=8, nb=self.capacity),
            sw_size_tokens=-1,
            recurrent_state=False,
        )
        object_group = SimpleNamespace(
            kernel_group_indices=[0], sw_size_chunks=-1, aux=False, recurrent=False
        )
        window = self.namespace["AttnWindowDesc"](num_chunks_in_sw=[-1])
        manager = SimpleNamespace(
            kernel_groups=[group],
            object_groups=[object_group],
            num_object_groups=1,
            get_attn_desc=lambda: window,
        )
        dtype = self.dtype
        context = SimpleNamespace(
            num_layers=2,
            device="cpu",
            kv_layer_groups_manager=manager,
            get_kernel_group_shape_dtype=lambda *args: ([2, 2, 16, 4], dtype),
            layout=object(),
            close=MagicMock(),
        )
        self.contexts.append(context)
        return context

    def info(self, rank: int = 0, token: str = "binding") -> Any:
        """Construct one worker's valid DTO."""
        return self.dto.PPRegistrationInfo(
            self.locator, rank, rank, ("layer0", "layer1"), token, True
        )

    def register(self, iid: int = 10, rank: int = 0, token: str = "binding") -> Any:
        """Invoke the production public PP handler."""
        return self.module.register_pp_kv_cache(
            iid, [], "model", 1, "engine", {}, [], self.info(rank, token)
        )

    def test_ack_does_not_wait_for_peers_and_status_requires_all_tp(self) -> None:
        """A one-piece MLA profile still requires both reader workers."""
        self.assertFalse(self.module.get_pp_stage_status(self.locator).ready)
        ack = self.register()
        self.assertEqual(ack.profile.kv_world_size, 1)
        self.assertFalse(self.module.get_pp_stage_status(self.locator).ready)
        self.register(11, 1)
        status = self.module.get_pp_stage_status(self.locator)
        self.assertTrue(status.ready)
        self.assertEqual(status.registered_tp_ranks, (0, 1))
        self.assertEqual(status.server_instance_id, ack.server_instance_id)

    def test_retry_is_idempotent_and_excludes_pool_capacity(self) -> None:
        """Reimport is closed and adds no registry ref; pool size is irrelevant."""
        first = self.register()
        self.capacity = 200
        self.assertEqual(self.register(), first)
        self.contexts[-1].close.assert_called_once()
        self.module.unregister_kv_cache(10)
        self.assertIsNone(self.ctx.layout_desc_registry.find(first.cache_namespace, 1))

    def test_conflicts_preserve_old_context_and_close_new_import(self) -> None:
        """Changed token, occupied slot and actual dtype cannot overwrite state."""
        ack = self.register()
        for iid, rank, token, dtype in (
            (10, 0, "new-token", "float16"),
            (20, 0, "binding", "float16"),
            (11, 1, "binding", "bfloat16"),
        ):
            self.dtype = dtype
            with self.assertRaises(ValueError):
                self.register(iid, rank, token)
            self.contexts[-1].close.assert_called_once()
            self.assertEqual(self.module.tracked_instance_count(), 1)
            self.assertEqual(
                self.module.get_pp_stage_status(self.locator).profile, ack.profile
            )
        self.contexts[0].close.assert_not_called()

    def test_unregister_and_reap_remove_exact_layout_references(self) -> None:
        """Lifecycle shares one registry and removes all PP status with entries."""
        ack = self.register()
        self.register(11, 1)
        self.module.unregister_kv_cache(10)
        self.assertIsNotNone(self.ctx.layout_desc_registry.find(ack.cache_namespace, 1))
        self.module.touch_instance(11)
        self.assertEqual(self.module.reap_stale_instances(-1, -1), [11])
        self.assertIsNone(self.ctx.layout_desc_registry.find(ack.cache_namespace, 1))
        self.assertIsNone(self.module.get_pp_stage_status(self.locator).profile)
        self.dtype = "bfloat16"
        self.assertNotEqual(self.register(12).cache_namespace, ack.cache_namespace)

    def test_validation_and_legacy_cannot_bypass_pp_identity(self) -> None:
        """Wrong geometry rejects before import; legacy refresh cannot hide PP."""
        with self.assertRaises(ValueError):
            self.module.register_pp_kv_cache(
                10, [], "wrong-model", 1, "engine", {}, [], self.info()
            )
        with self.assertRaises(ValueError):
            self.module.register_pp_kv_cache(
                10, [], "model", 2, "engine", {}, [], self.info()
            )
        self.assertEqual(len(self.contexts), 0)
        self.register()
        with self.assertRaises(ValueError):
            self.module.register_kv_cache(10, [], "model", 1, "engine", {}, [])

    def test_registry_failure_closes_context_without_publishing(self) -> None:
        """A failed layout registration never creates a TP status slot."""
        with patch.object(
            self.ctx.layout_desc_registry,
            "register",
            side_effect=RuntimeError("registry failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "registry failed"):
                self.register()
        self.contexts[-1].close.assert_called_once()
        self.assertEqual(self.module.tracked_instance_count(), 0)
        self.assertFalse(self.module.get_pp_stage_status(self.locator).ready)

    def test_simultaneous_same_slot_publish_accepts_exactly_one(self) -> None:
        """Context creation can overlap, while slot publication stays atomic."""
        from concurrent.futures import ThreadPoolExecutor

        barrier = threading.Barrier(2)
        original_factory = self.namespace["create_cache_context"]

        def concurrent_factory(*args: Any, **kwargs: Any) -> Any:
            context = original_factory(*args, **kwargs)
            barrier.wait(timeout=3)
            return context

        self.namespace["create_cache_context"] = concurrent_factory
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = [pool.submit(self.register, iid) for iid in (10, 20)]
            accepted, rejected = [], []
            for future in pending:
                try:
                    accepted.append(future.result())
                except ValueError as error:
                    rejected.append(error)
        self.assertEqual((len(accepted), len(rejected)), (1, 1))
        self.assertEqual(self.module.tracked_instance_count(), 1)
        self.assertEqual(sum(context.close.call_count for context in self.contexts), 1)


if __name__ == "__main__":
    unittest.main()
