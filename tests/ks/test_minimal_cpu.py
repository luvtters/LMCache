# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the historical 17-line adaptation and index pinning.

Execute actual source methods without importing vLLM, torch or a device runtime.
The tensor fixture models values/ownership only; it proves no device semantics.
"""

import array
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
CONNECTOR = ROOT / "lmcache/integration/vllm/lmcache_mp_connector.py"
CONTEXT = ROOT / "lmcache/v1/platform/base/cache_context.py"


def method(path: Path, class_name: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def function(node: ast.FunctionDef, namespace: dict) -> object:
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "<actual-source>", "exec"), namespace)
    return namespace[node.name]


class HostTensor:
    def __init__(self, values: object, pinned: bool = False) -> None:
        self.values = list(values)
        self.pinned = pinned

    def pin_memory(self) -> "HostTensor":
        return HostTensor(self.values, pinned=True)


class DeviceBuffer:
    shape = (100,)

    def __init__(self) -> None:
        self.source = None

    def __getitem__(self, index: slice) -> "DeviceBuffer":
        return self

    def copy_(self, source: HostTensor, non_blocking: bool) -> None:
        self.source = source
        self.non_blocking = non_blocking


class MinimalTests(unittest.TestCase):
    def test_local_cache_accepts_mooncake_pp_handshake(self) -> None:
        names = [n.name for n in next(n for n in ast.parse(CONNECTOR.read_text(encoding="utf-8")).body
                 if isinstance(n, ast.ClassDef) and n.name == "LMCacheMPConnector").body
                 if isinstance(n, ast.FunctionDef)]
        self.assertIn("set_xfer_handshake_metadata_pp_aware", names)
        call = function(method(CONNECTOR, "LMCacheMPConnector", names[names.index("set_xfer_handshake_metadata_pp_aware")]), {"Any": object})
        self.assertIsNone(call(object(), {(0, 0): "front", (1, 0): "rear"}))

    def apply_new_request(self, existing: list, allocated: tuple) -> SimpleNamespace:
        tree = ast.parse(CONNECTOR.read_text(encoding="utf-8"))
        loops = [n for n in ast.walk(tree) if isinstance(n, ast.For)
                 and ast.unparse(n.iter) == "scheduler_output.scheduled_new_reqs"
                 and n.body and isinstance(n.body[0], ast.Assign)
                 and "_get_request_tracker" in ast.unparse(n.body[0])]
        loop = loops[0]
        tracker = SimpleNamespace(allocated_block_ids=existing, appended=[])

        def append(blocks: tuple, relocation: int) -> None:
            tracker.allocated_block_ids = list(blocks)
            tracker.appended.append((blocks, relocation))

        tracker.append_block_ids = append
        owner = SimpleNamespace(_get_request_tracker=lambda _: tracker, _mamba_relocation_window=3)
        # Execute the real new-request prefix, before token accounting.
        body = [loop.body[0]]
        if len(loop.body) > 1 and isinstance(loop.body[1], ast.If):
            body.append(loop.body[1])
        node = ast.Module(body=body, type_ignores=[])
        exec(compile(ast.fix_missing_locations(node), "<actual-store-prefix>", "exec"),
             {"self": owner, "new_request": SimpleNamespace(req_id="r", block_ids=allocated)})
        return tracker

    def test_nonselected_writer_recovers_initial_blocks(self) -> None:
        result = self.apply_new_request([], ([4, 5],))
        self.assertEqual(result.appended, [(([4, 5],), 3)])

    def test_existing_loader_allocation_is_preserved(self) -> None:
        result = self.apply_new_request([[1, 2]], ([4, 5],))
        self.assertEqual(result.allocated_block_ids, [[1, 2]])
        self.assertEqual(result.appended, [])

    def test_no_blocks_are_fabricated(self) -> None:
        self.assertEqual(self.apply_new_request([], ()).allocated_block_ids, [])

    def test_index_source_is_pinned_without_changing_values(self) -> None:
        torch = SimpleNamespace(Tensor=HostTensor, long="int64",
                                frombuffer=lambda values, dtype: HostTensor(values))
        call = function(method(CONTEXT, "BaseCacheContext", "stage_block_ids"),
                        {"torch": torch, "array": array})
        buffer = DeviceBuffer()
        views = call(SimpleNamespace(block_ids_buffer_=buffer), [[12, 35], [], [47]])
        self.assertTrue(buffer.source.pinned)
        self.assertEqual(buffer.source.values, [12, 35, 47])
        self.assertTrue(buffer.non_blocking)
        self.assertEqual(len(views), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
