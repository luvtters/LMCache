"""Run real metadata methods without importing vLLM or a device runtime.

KS_TEST_SOURCE_REV=9a155a5^ reproduces the partial-APC failure against the
pre-fix source. Unset it to exercise the working source. Payload constructors
and block slicing are substitutes; confirmation and STORE bounds are real AST.
"""
import ast
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
RELATIVE = "lmcache/integration/vllm/lmcache_mp_metadata.py"


def load_methods():
    revision = os.environ.get("KS_TEST_SOURCE_REV")
    source = (
        subprocess.check_output(
            ["git", "show", f"{revision}:{RELATIVE}"], cwd=ROOT
        ).decode("utf-8")
        if revision else (ROOT / RELATIVE).read_text(encoding="utf-8")
    )
    tree = ast.parse(source)
    tracker_names = {
        "confirm_borrowed_restore", "num_allocated_blocks", "get_token_ids",
        "increase_num_stored_tokens", "increase_num_scheduled_tokens",
    }
    nodes = []
    for owner, names in (
        ("LMCacheMPRequestTracker", tracker_names),
        ("LMCacheMPRequestMetadata", {"GetStoreMetadata"}),
    ):
        cls = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == owner)
        selected = [n for n in cls.body
                    if isinstance(n, ast.FunctionDef) and n.name in names]
        if {n.name for n in selected} != names:
            raise AssertionError(f"Missing real methods in {revision or 'working'}")
        for node in selected:
            node.decorator_list = []
        nodes.extend(selected)
    namespace = {
        # Only the result payload is substituted; neither stub calculates bounds.
        "LoadStoreOp": SimpleNamespace,
        "LMCacheMPRequestMetadata": SimpleNamespace,
        "slice_block_ids_per_group": lambda blocks, groups, start, end: blocks,
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, *nodes], type_ignores=[])
    )
    exec(compile(module, str(ROOT / RELATIVE), "exec"), namespace)
    tracker_class = type(
        "ActualTrackerMethods", (), {name: namespace[name] for name in tracker_names}
    )
    return tracker_class, namespace["GetStoreMetadata"]


class PartialAPCStoreBoundTest(unittest.TestCase):
    def setUp(self):
        self.tracker_class, self.store_metadata = load_methods()

    def tracker(self, *, submitted=True, apc=1024, prompt=12288, blocks=16):
        tracker = self.tracker_class()
        tracker.request_id = "partial-apc"
        tracker.borrowed_restore_submitted = submitted
        tracker.num_stored_tokens = tracker.num_scheduled_tokens = 0
        tracker.num_lmcache_hit_tokens = 8192
        tracker.num_vllm_hit_tokens = apc
        tracker.all_token_ids = list(range(prompt))
        tracker.num_prompt_tokens = prompt
        tracker.allocated_block_ids = {0: list(range(blocks))}
        tracker.mm_adjusted_prompt_ids = []
        tracker.max_offload_tokens = None
        tracker.cache_salt = ""
        tracker.request_configs = None
        return tracker

    def test_partial_failure_never_stores_uncomputed_retained_allocations(self):
        tracker = self.tracker()
        # The actual scheduler keeps all allocated blocks when a failed load
        # retains nonzero APC, then schedules local recomputation from 1024.
        tracker.confirm_borrowed_restore(1024, 512)
        tracker.increase_num_scheduled_tokens(512)
        metadata = self.store_metadata(tracker, 512, [512])
        self.assertIsNotNone(metadata)
        self.assertLessEqual(metadata.op.end, 1024 + 512)
        self.assertEqual((metadata.op.start, metadata.op.end), (0, 1536))
        self.assertEqual(tracker.num_stored_tokens, 1536)
        self.assertFalse(tracker.borrowed_restore_submitted)
        self.assertIsNone(self.store_metadata(tracker, 512, [512]))

    def test_successful_full_hit_recomputes_only_last_token(self):
        tracker = self.tracker(prompt=8192)
        # The actual scheduler subtracts one on a successful full prompt hit
        # so the last token is computed for sampling. This is not load failure.
        tracker.confirm_borrowed_restore(8191, 512)
        self.assertEqual(tracker.num_lmcache_hit_tokens, 8192)
        self.assertEqual(tracker.num_stored_tokens, 7680)
        tracker.increase_num_scheduled_tokens(1)
        metadata = self.store_metadata(tracker, 512, [512])
        self.assertEqual((metadata.op.start, metadata.op.end), (7680, 8192))
        self.assertFalse(tracker.borrowed_restore_submitted)

    def test_apc_only_does_not_confirm_an_unsubmitted_restore(self):
        tracker = self.tracker(submitted=False, apc=8192, blocks=17)
        tracker.confirm_borrowed_restore(8192, 512)
        self.assertEqual(tracker.num_stored_tokens, 0)
        tracker.increase_num_scheduled_tokens(512)
        metadata = self.store_metadata(tracker, 512, [512])
        # APC-only KV still needs its normal STORE; no restored prefix is skipped.
        self.assertEqual((metadata.op.start, metadata.op.end), (0, 8704))


if __name__ == "__main__":
    unittest.main()
