# SPDX-License-Identifier: Apache-2.0
"""Execute the extension-hook methods without importing vLLM/native libraries.

These source-level CPU checks exercise the production method bodies with
network/engine boundaries replaced. They do not validate full module imports.
"""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock
import __future__
import ast
import unittest


ROOT = Path(__file__).resolve().parents[2]
ADAPTER = "lmcache/integration/vllm/vllm_multi_process_adapter.py"
CONNECTOR = "lmcache/integration/vllm/lmcache_mp_connector.py"


def load_methods(
    path: str, class_name: str, names: set[str], namespace: dict[str, Any]
) -> type:
    """Load actual method bodies with isolated dependencies, preserving dispatch."""
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    source = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    methods = [
        node for node in source.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {method.name for method in methods} != names:
        raise ValueError("Requested extension methods are missing")
    for method in methods:
        method.decorator_list = []
    isolated = ast.Module(body=[ast.ClassDef(
        name=class_name, bases=[], keywords=[], body=methods, decorator_list=[]
    )], type_ignores=[])
    ast.fix_missing_locations(isolated)
    exec(compile(isolated, path, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[class_name]


class ExtensionHookTests(unittest.TestCase):
    def scheduler(self, custom_lookup: bool = False) -> tuple[Any, MagicMock, Any]:
        """Set up only the dependencies needed by public lookup submission."""
        operation = SimpleNamespace(submit=MagicMock())
        factory = MagicMock(return_value=operation)
        scheduler_cls = load_methods(ADAPTER, "LMCacheMPSchedulerAdapter", {
            "maybe_submit_lookup_request", "_create_lookup_state",
        }, {"LookupState": factory, "num_packed_tokens": lambda data: len(data) // 4})
        if custom_lookup:
            class CustomScheduler(scheduler_cls):
                def _create_lookup_state(self, key: Any) -> Any:
                    return operation
            scheduler_cls = CustomScheduler
        adapter = scheduler_cls()
        adapter._ensure_heartbeat_started = lambda: None
        adapter.is_healthy = True
        adapter._lookups = {}
        adapter.lmcache_tokens_per_chunk = 128
        adapter.req_clients = {"pool": object()}
        adapter.tp_size = 1
        adapter._mq_timeout = 2.0
        adapter._nonblocking_lookup_status = True
        key = SimpleNamespace(no_worker_id_version=lambda: "aligned-key")
        adapter._create_key = MagicMock(return_value=key)
        return adapter, factory, operation

    def test_default_lookup_keeps_existing_protocol_arguments(self) -> None:
        adapter, factory, operation = self.scheduler()
        self.assertTrue(adapter.maybe_submit_lookup_request("r", bytes(1024)))
        factory.assert_called_once_with(
            "aligned-key", adapter.req_clients, tp_size=1,
            tokens_per_chunk=128, timeout=2.0, blocking_lookup=False,
        )
        operation.submit.assert_called_once_with()
        self.assertFalse(adapter.maybe_submit_lookup_request("r", bytes(1024)))

    def test_custom_lookup_is_used_by_public_submission(self) -> None:
        adapter, factory, operation = self.scheduler(custom_lookup=True)
        self.assertTrue(adapter.maybe_submit_lookup_request("r", bytes(1024)))
        factory.assert_not_called()
        operation.submit.assert_called_once_with()

    def test_connector_default_factory_and_constructor_dispatch(self) -> None:
        factory = MagicMock(return_value="stock-scheduler")
        connector_cls = load_methods(CONNECTOR, "LMCacheMPConnector", {
            "create_lmcache_mp_scheduler_adapter",
        }, {"LMCacheMPSchedulerAdapter": factory})
        self.assertEqual(
            connector_cls().create_lmcache_mp_scheduler_adapter(model_name="qwen3"),
            "stock-scheduler",
        )
        factory.assert_called_once_with(model_name="qwen3")
        tree = ast.parse((ROOT / CONNECTOR).read_text(encoding="utf-8"))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertTrue(any(
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "create_lmcache_mp_scheduler_adapter"
            for call in calls
        ))


if __name__ == "__main__":
    unittest.main()
