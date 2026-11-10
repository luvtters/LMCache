"""Execute the actual borrow RETRIEVE method with controlled device completions.

The method is loaded from the source AST, not copied into this test. Device
streams/events/transfers and storage are substitutes: this checks ownership
ordering and failure handling, not DMA correctness or performance.
"""

import ast
from contextlib import nullcontext
from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"


@dataclass(frozen=True)
class Token:
    server_instance_id: str
    sequence: int


def load_method(namespace, source, class_name, method_name):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    owner = next(
        n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    method = next(
        n for n in owner.body
        if isinstance(n, ast.FunctionDef) and n.name == method_name
    )
    method.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[future, method], type_ignores=[])
    )
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[method_name]


def load_retrieve(namespace):
    return load_method(
        namespace, SOURCE, "LMCacheDrivenTransferModule", "_retrieve_borrowed"
    )


class Harness:
    def __init__(self, *, miss=False, fail=None):
        self.trace = []
        self.fail = fail
        self.pending = []
        self.active = False
        self.token = Token("server", 1)
        self.objects = {
            name: SimpleNamespace(name=name, get_size=lambda: 16)
            for name in ("a0", "a1", "b0", "b1", "aux0", "aux1")
        }
        self.groups = [["a0", "a1"], ["b0", "b1"], ["aux0", "aux1"]]
        self.storage = SimpleNamespace(
            borrow_read=lambda keys: self.borrow(keys, miss),
            release_borrow=self.release,
            touch_keys=lambda keys: self.trace.append(("touch", tuple(keys))),
        )
        self.stream = SimpleNamespace(synchronize=self.synchronize)
        self.backend = SimpleNamespace(
            create_event=lambda device: self.mark("create_event", object()),
            import_event=lambda handle, device: self.mark("import_event", object()),
            wait_event=lambda event, stream: self.mark("wait_event"),
            record_event=self.record,
            export_event=self.export,
        )
        groups = SimpleNamespace(
            num_object_groups=3,
            num_kernel_groups=3,
            get_attn_desc=lambda: SimpleNamespace(
                num_chunks_in_sw=[-1, 1, -1],
                group_kinds=("attention", "attention", "aux"),
            ),
        )
        self.cache = SimpleNamespace(
            device="cpu-test",
            stream=self.stream,
            cupy_stream=self.stream,
            kv_layer_groups_manager=groups,
            calculate_num_blocks=lambda chunk_size, group: 1,
            max_batch_size=8,
            hold_imported_event=lambda event: self.mark("hold_import", 7),
            release_imported_event=lambda token: self.mark("release_import"),
        )
        self.entry = SimpleNamespace(
            cache_context=self.cache,
            model_name="test-model",
            event_backend=self.backend,
        )
        bus = SimpleNamespace(
            publish=lambda event: self.mark("publish"),
            publish_on_stream=lambda stream, event: self.mark("publish_stream"),
        )
        self.ctx = SimpleNamespace(
            read_policy="transfer_borrow",
            chunk_size=4,
            storage_manager=self.storage,
            event_bus=bus,
            resolve_obj_keys=lambda key, groups: self.groups,
        )
        self.instance = SimpleNamespace(
            _ctx=self.ctx,
            get_and_touch_context_entry=lambda instance_id: self.entry,
        )
        self.key = SimpleNamespace(
            request_id="request", cache_salt="", worker_id=0,
            world_size=1, start=0, end=8,
        )
        self.namespace = {
            "logger": logging.getLogger("borrowed-retrieve-test"),
            "torch_dev": SimpleNamespace(
                device=lambda device: nullcontext(),
                stream=lambda stream: nullcontext(),
                synchronize=self.synchronize,
            ),
            "Event": lambda **kwargs: SimpleNamespace(**kwargs),
            "EventType": SimpleNamespace(
                MP_RETRIEVE_SUBMITTED="submitted",
                MP_RETRIEVE_START="start",
                MP_RETRIEVE_END="end",
            ),
            "ReadBorrowToken": Token,
            "next_transfer_key": lambda request_id: "transfer",
            "downsample_and_stage_block_ids": self.stage,
            "transfer_kv_per_object_group": self.transfer,
            "submit_callback_to_stream": self.enqueue,
            "lmcache_native": SimpleNamespace(
                TransferDirection=SimpleNamespace(H2D="H2D")
            ),
        }
        self.method = load_retrieve(self.namespace)

    def mark(self, name, result=None):
        self.trace.append((name,))
        return result

    def borrow(self, keys, miss):
        self.trace.append(("borrow", tuple(keys)))
        if miss:
            return None
        self.active = True
        return SimpleNamespace(
            token=self.token, objects={key: self.objects[key] for key in keys}
        )

    def release(self, token):
        self.trace.append(("release", token))
        assert token == self.token
        existed = self.active
        self.active = False
        return existed

    def stage(self, cache, blocks):
        assert self.active, "block staging started before all-group borrow"
        self.trace.append(("stage",))
        if self.fail == "stage":
            raise RuntimeError("injected block staging failure")
        return blocks

    def transfer(self, cache, blocks, objects, **kwargs):
        assert self.active, "H2D started without source ownership"
        group = kwargs["object_group_id"]
        self.trace.append((
            "h2d", group,
            tuple(obj.name if obj is not None else None for obj in objects),
            kwargs["skip_first_n_tokens"],
        ))
        if group == 1 and self.fail in ("partial", "partial_sync"):
            raise RuntimeError("injected failure after first group submitted")

    def enqueue(self, stream, kind, payload):
        self.trace.append(("enqueue", kind))
        if kind == "release_read_borrow" and self.fail in ("enqueue", "enqueue_sync"):
            raise RuntimeError("injected completion enqueue failure")
        self.pending.append((kind, payload))
        if kind == "release_read_borrow" and self.fail == "enqueue_after_accept":
            raise RuntimeError("injected failure after callback accepted")

    def complete_callbacks(self):
        for kind, payload in self.pending:
            self.trace.append(("callback", kind))
            if kind == "release_read_borrow":
                self.release(Token(*payload))
        self.pending.clear()

    def synchronize(self, *args):
        self.mark("sync_start")
        if self.fail in ("partial_sync", "enqueue_sync"):
            self.mark("sync_failed")
            raise RuntimeError("injected synchronization failure")
        self.mark("sync_complete")
        self.complete_callbacks()

    def record(self, event, stream):
        self.mark("record")
        if self.fail == "record":
            raise RuntimeError("injected event record failure")

    def export(self, event, device):
        self.mark("export")
        if self.fail == "export":
            raise RuntimeError("injected event export failure")
        return b"completion"

    def run(self):
        return self.method(
            self.instance, self.key, 1, [[0, 1], [2, 3], [4, 5]], b"producer", 2
        )

    def events(self):
        return [item[0] for item in self.trace]


class BorrowedRetrieveContractTest(unittest.TestCase):
    def test_all_group_miss_submits_no_staging_or_h2d(self):
        h = Harness(miss=True)
        self.assertEqual(h.run(), (b"", False))
        self.assertEqual(h.trace[0], ("borrow", ("a0", "a1", "b1")))
        self.assertNotIn("stage", h.events())
        self.assertNotIn("h2d", h.events())
        self.assertNotIn("release", h.events())

    def test_one_complete_borrow_precedes_every_group_and_callback_owns_release(self):
        h = Harness()
        self.assertEqual(h.run(), (b"completion", True))
        self.assertEqual(
            [item for item in h.trace if item[0] == "borrow"],
            [("borrow", ("a0", "a1", "b1"))],
        )
        self.assertLess(h.events().index("borrow"), h.events().index("stage"))
        self.assertEqual(
            [item for item in h.trace if item[0] == "h2d"],
            [("h2d", 0, ("a0", "a1"), 2), ("h2d", 1, (None, "b1"), 2)],
        )
        self.assertTrue(h.active)
        self.assertNotIn("release", h.events())
        h.complete_callbacks()
        self.assertFalse(h.active)
        self.assertEqual(h.events().count("release"), 1)

    def test_partial_transfer_error_waits_for_completion_before_releasing(self):
        h = Harness(fail="partial")
        _, ok = h.run()
        self.assertFalse(ok)
        self.assertIn("h2d", h.events())
        if h.active:
            self.assertNotIn("release", h.events())
            h.complete_callbacks()
        self.assertFalse(h.active)
        release_at = h.events().index("release")
        preceding = h.trace[:release_at]
        self.assertTrue(
            ("sync_complete",) in preceding
            or ("callback", "release_read_borrow") in preceding
        )

    def test_completion_failures_sync_before_release(self):
        for fail in ("enqueue", "enqueue_after_accept", "export", "record", "stage"):
            with self.subTest(fail=fail):
                h = Harness(fail=fail)
                _, ok = h.run()
                self.assertFalse(ok)
                self.assertFalse(h.active)
                self.assertLess(
                    h.events().index("sync_complete"), h.events().index("release")
                )

    def test_sync_failure_retains_source_token(self):
        h = Harness(fail="enqueue_sync")
        with self.assertRaisesRegex(RuntimeError, "Cannot prove RETRIEVE"):
            h.run()
        self.assertIn("sync_failed", h.events())
        self.assertTrue(h.active)
        self.assertNotIn("release", h.events())

    def test_invalid_block_ranges_and_unregistered_context_never_borrow(self):
        for blocks in ([[0]], [[0], [2, 3], [4, 5]]):
            with self.subTest(blocks=blocks):
                h = Harness()
                self.assertEqual(h.method(h.instance, h.key, 1, blocks, b"in", 0),
                                 (b"", False))
                self.assertNotIn("borrow", h.events())
                self.assertNotIn("stage", h.events())
        h = Harness()
        h.entry = None
        self.assertEqual(h.run(), (b"", False))
        self.assertEqual(h.trace, [])


class PPBorrowPolicyContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Real stdlib-only protocol dataclasses, loaded without framework imports.
        spec = importlib.util.spec_from_file_location(
            "_borrow_test_pp_layout", ROOT / "lmcache/v1/multiprocess/pp_layout.py"
        )
        cls.protocol = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.protocol
        spec.loader.exec_module(cls.protocol)

    def stages(self, policies):
        p = self.protocol
        locators = [p.PPStageLocator("model", 2, 2, rank, ((0, 1), (1, 2)))
                    for rank in range(2)]
        profiles = [p.PPStageProfile(loc, (f"layer{loc.pp_rank}",), False, "{}")
                    for loc in locators]
        statuses = [p.PPStageStatus(loc, f"server{rank}", profiles[rank], (0, 1),
                                   read_policy=policies[rank])
                    for rank, loc in enumerate(locators)]
        return locators, profiles, statuses

    def test_registration_rejects_mixed_policy_before_device_context_creation(self):
        calls = []
        namespace = {"create_cache_context": lambda *a, **k: calls.append("create")}
        register = load_method(namespace, SOURCE, "LMCacheDrivenTransferModule",
                               "register_pp_kv_cache")
        for server, worker in (("legacy", "transfer_borrow"),
                               ("transfer_borrow", "legacy")):
            with self.subTest(server=server, worker=worker):
                owner = SimpleNamespace(_ctx=SimpleNamespace(read_policy=server))
                info = SimpleNamespace(read_policy=worker)
                with self.assertRaisesRegex(ValueError, "read policies differ"):
                    register(owner, 1, [], "model", 2, "engine", {}, [], info)
        self.assertEqual(calls, [])

    def test_scheduler_requires_same_policy_and_complete_ranks_on_both_stages(self):
        path = ROOT / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
        namespace = {
            "PPStageStatus": self.protocol.PPStageStatus,
            "get_read_policy": lambda: "transfer_borrow",
            "time": time,
            "logger": logging.getLogger("borrow-policy-test"),
        }
        initialize = load_method(namespace, path, "LMCacheMPSchedulerAdapter",
                                 "_initialize_pp_stage_profiles")
        for kind in ("uniform", "mixed", "incomplete"):
            with self.subTest(kind=kind):
                policies = ("transfer_borrow",
                            "legacy" if kind == "mixed" else "transfer_borrow")
                locators, profiles, statuses = self.stages(policies)
                if kind == "incomplete":
                    statuses[1] = self.protocol.PPStageStatus(
                        locators[1], "server1", profiles[1], (0,),
                        read_policy="transfer_borrow",
                    )
                clients = {
                    url: SimpleNamespace(get_pp_stage_status=lambda loc, value=status:
                        SimpleNamespace(result=lambda timeout: value))
                    for url, status in zip(("first", "second"), statuses)
                }
                owner = SimpleNamespace(
                    _server_urls=["first", "second"], _mq_timeout=3,
                    req_clients=clients, world_size=2,
                    parallel_strategy=SimpleNamespace(mla_only=False, num_kv_readers=1),
                    _server_model_names={"old": "unchanged"},
                )
                if kind == "uniform":
                    initialize(owner, locators)
                    self.assertEqual(owner._server_model_names,
                        {"first": profiles[0].cache_namespace,
                         "second": profiles[1].cache_namespace})
                else:
                    with self.assertRaises(ValueError):
                        initialize(owner, locators)
                    self.assertEqual(owner._server_model_names, {"old": "unchanged"})

    def test_stage_status_reports_server_policy_snapshot(self):
        locators, profiles, _ = self.stages(("transfer_borrow", "transfer_borrow"))
        namespace = {"PPStageStatus": self.protocol.PPStageStatus}
        get_status = load_method(namespace, SOURCE, "LMCacheDrivenTransferModule",
                                 "get_pp_stage_status")
        entries = {
            rank: SimpleNamespace(
                registration_info=SimpleNamespace(locator=locators[0], tp_rank=rank),
                pp_profile=profiles[0],
            ) for rank in range(2)
        }
        owner = SimpleNamespace(
            _registration_lock=nullcontext(),
            _ctx=SimpleNamespace(read_policy="transfer_borrow"),
            _server_instance_id="server0",
            context_entries_snapshot=lambda: entries,
        )
        status = get_status(owner, locators[0])
        self.assertTrue(status.ready)
        self.assertEqual(status.read_policy, "transfer_borrow")


if __name__ == "__main__":
    logging.disable(logging.CRITICAL)
    unittest.main()
