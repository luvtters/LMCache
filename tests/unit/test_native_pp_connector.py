# SPDX-License-Identifier: Apache-2.0
"""Native PP client control paths with stdlib-only boundary fixtures.

AST extraction executes production methods unchanged. Only unavailable vLLM,
device, transport and base classes are replaced; no network/device work runs.
These checks do not validate tensor layout, SHM transfers or model correctness.
Run directly with Python, without importing the repository pytest conftest.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, replace
import importlib.util
import math
import os
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[2]
CONNECTOR = ROOT / "lmcache/integration/vllm/lmcache_mp_connector.py"
ADAPTER = ROOT / "lmcache/integration/vllm/vllm_multi_process_adapter.py"


def extract(path, namespace, *, functions=(), class_name=None, methods=None):
    """Compile actual source nodes with deferred unavailable type annotations."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            nodes.append(node)
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            if methods is not None:
                node.bases = [ast.Name(id="FixtureBase", ctx=ast.Load())]
                node.decorator_list = []
                node.body = [
                    n
                    for n in node.body
                    if isinstance(n, ast.FunctionDef) and n.name in methods
                ]
                for method in node.body:
                    method.decorator_list = [
                        d
                        for d in method.decorator_list
                        if isinstance(d, ast.Name) and d.id == "property"
                    ]
            nodes.append(node)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace.get(class_name) if class_name else namespace


spec = importlib.util.spec_from_file_location(
    "native_pp_layout_client_test", ROOT / "lmcache/v1/multiprocess/pp_layout.py"
)
layout = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = layout
spec.loader.exec_module(layout)
ParallelStrategy = extract(
    ADAPTER, {"dataclass": dataclass, "math": math}, class_name="ParallelStrategy"
)
LOG = NS(info=Mock(), warning=Mock(), exception=Mock())
URLS = ["tcp://front:5565", "tcp://rear:5565"]


def config(rank=0, tp=16, pp=2, mla=True, urls=None):
    extra = {"lmcache.mp.server_urls": URLS if urls is None else urls}
    return NS(
        parallel_config=NS(
            world_size=tp * pp,
            rank=rank,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        model_config=NS(mla_only=mla, get_total_num_hidden_layers=lambda: 78),
        cache_config=NS(mamba_cache_mode="none"),
        kv_transfer_config=NS(
            get_from_extra_config=lambda name, default: extra.get(name, default),
            is_kv_producer=True,
            kv_connector_extra_config=extra,
        ),
    )


class FixtureBase:
    def __init__(self, cfg, role, cache):
        self._vllm_config, self._role, self._kv_cache_config = cfg, role, cache


def connector_class():
    ns = dict(
        FixtureBase=FixtureBase,
        ParallelStrategy=ParallelStrategy,
        mla_only=lambda model: model.mla_only,
        urlsplit=urlsplit,
        urlunsplit=urlunsplit,
        PPStageLocator=layout.PPStageLocator,
        math=math,
        sys=sys,
        KVConnectorRole=NS(SCHEDULER="scheduler", WORKER="worker"),
        KVConnectorTransferResults=None,
        LMCacheMPConnectorStats=object,
        validate_mamba_step_alignment=lambda *_: None,
        validate_kv_cache_groups=lambda *_: None,
        validate_dcp_support=lambda *_: None,
        get_group_tokens_per_block=lambda *_: [128],
        get_vllm_scheduler_block_size=lambda *_: 128,
        get_dcp_decorated_model_name=lambda *_: "glm-5",
        is_mp_server_autostart_enabled=lambda *_: False,
        zmq=NS(Context=NS(instance=lambda: object())),
        logger=LOG,
        print_banner_once=lambda *_: None,
        LMCacheMPSchedulerAdapter=lambda **kw: NS(**kw, lmcache_tokens_per_chunk=512),
        LMCacheMPWorkerAdapter=lambda **kw: NS(**kw),
    )
    extract(
        CONNECTOR,
        ns,
        functions={
            "build_parallel_strategy_from_vllm_config",
            "_ensure_transport_scheme",
            "normalize_pp_server_urls",
            "build_pp_stage_locators",
        },
    )
    cls = extract(
        CONNECTOR,
        ns,
        class_name="LMCacheMPConnector",
        methods={
            "__init__",
            "build_parallel_strategy",
            "role",
            "transfer_intermediate_tensors",
            "set_xfer_handshake_metadata_pp_aware",
        },
    )
    return cls, ns


def partition_module():
    module = ModuleType("vllm.distributed.utils")
    # This boundary stands in for vLLM's resolved, possibly unequal partition.
    module.get_pp_indices = Mock(
        side_effect=lambda total, rank, count: ((0, 41), (41, total))[rank]
    )
    return module


class ConnectorTests(unittest.TestCase):
    def test_actual_constructor_routes_32_workers_and_scheduler(self):
        cls, _ = connector_class()
        module = partition_module()
        with patch.dict(sys.modules, {"vllm.distributed.utils": module}):
            for rank in range(32):
                obj = cls(config(rank), "worker")
                self.assertEqual(obj.worker_adapter.server_url, URLS[rank // 16])
                self.assertEqual(
                    obj.worker_adapter.pp_stage_locator.pp_rank, rank // 16
                )
                self.assertEqual(
                    obj.worker_adapter.pp_stage_locator.layer_ranges,
                    ((0, 41), (41, 78)),
                )
            obj = cls(config(), "scheduler")
        self.assertEqual(obj.scheduler_adapter.server_urls, URLS)
        self.assertEqual(len(obj.scheduler_adapter.pp_stage_locators), 2)
        self.assertEqual(module.get_pp_indices.call_count, 66)

    def test_generic_geometry_mla_and_sharded_kv(self):
        cls, _ = connector_class()
        obj = cls.__new__(cls)
        for tp, pp in ((1, 3), (4, 3), (16, 2)):
            for mla in (True, False):
                for rank in range(tp * pp):
                    strategy = obj.build_parallel_strategy(
                        config(rank, tp, pp, mla), pp
                    )
                    self.assertTrue(strategy.pp_stage_local)
                    self.assertEqual(strategy.kv_world_size, 1 if mla else tp)
                    self.assertEqual(strategy.kv_worker_id, 0 if mla else rank % tp)
                    self.assertEqual(strategy.num_kv_readers, tp if mla else 1)
                    self.assertEqual(strategy.is_kv_writer, not mla or rank % tp == 0)

    def test_reject_unsupported_parallel_geometry(self):
        cls, _ = connector_class()
        obj = cls.__new__(cls)
        for field, value in (
            ("data_parallel_size", 2),
            ("decode_context_parallel_size", 2),
            ("prefill_context_parallel_size", 2),
            ("world_size", 16),
            ("rank", 32),
            ("rank", -1),
        ):
            cfg = config()
            setattr(cfg.parallel_config, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                obj.build_parallel_strategy(cfg, 2)
        with self.assertRaisesRegex(ValueError, "one server"):
            obj.build_parallel_strategy(config(), 4)

    def test_legacy_single_server_and_pp1_strategy_unchanged(self):
        cls, ns = connector_class()
        obj = cls.__new__(cls)
        for cfg, servers in ((config(), 1), (config(tp=16, pp=1), 2)):
            expected = ns["build_parallel_strategy_from_vllm_config"](cfg, servers)
            actual = obj.build_parallel_strategy(cfg, servers)
            self.assertEqual(actual, expected)
            self.assertFalse(actual.pp_stage_local)

    def test_actual_legacy_constructors_keep_routing_without_pp_status(self):
        cls, _ = connector_class()
        for cfg in (config(urls=[URLS[0]]), config(rank=15, tp=16, pp=1)):
            obj = cls(cfg, "worker")
            expected_url = (
                URLS[0] if cfg.parallel_config.pipeline_parallel_size > 1 else URLS[1]
            )
            self.assertEqual(obj.worker_adapter.server_url, expected_url)
            self.assertIsNone(obj.worker_adapter.pp_stage_locator)
            obj = cls(cfg, "scheduler")
            self.assertEqual(obj.scheduler_adapter.pp_stage_locators, ())

    def test_endpoint_normalization_rejects_duplicates_before_adapter(self):
        cls, ns = connector_class()
        self.assertEqual(
            ns["normalize_pp_server_urls"](["TCP://Front:5565/", "tcp://front:5566"]),
            ["tcp://front:5565", "tcp://front:5566"],
        )
        with self.assertRaisesRegex(ValueError, "distinct"):
            cls(config(urls=["TCP://Front:5565/", "front:5565"]), "worker")

    def test_native_handshake_preserves_mooncake_metadata(self):
        cls, _ = connector_class()
        metadata = {0: {0: object()}, 1: {0: object()}}
        snapshot = {k: dict(v) for k, v in metadata.items()}
        self.assertIsNone(
            cls.__new__(cls).set_xfer_handshake_metadata_pp_aware(metadata)
        )
        self.assertEqual(metadata, snapshot)


class Key(NS):
    def no_worker_id_version(self):
        return self


def scheduler():
    ns = dict(
        FixtureBase=object,
        time=time,
        os=os,
        logger=LOG,
        IPCCacheServerKey=Key,
        _LookupAck=lambda **kw: NS(**kw),
        PPStageStatus=layout.PPStageStatus,
    )
    cls = extract(
        ADAPTER,
        ns,
        class_name="LMCacheMPSchedulerAdapter",
        methods={
            "_initialize_pp_stage_profiles",
            "world_size",
            "tp_size",
            "_create_key",
            "maybe_submit_lookup_request",
            "free_lookup_locks",
            "_free_inconsistent_lookup_locks",
            "report_block_allocations",
        },
    )
    obj = cls()
    obj.parallel_strategy = ParallelStrategy(True, 32, 0, 16, 2, 2, 1, True)
    obj._server_urls = URLS
    obj.req_clients = {
        url: NS(
            lookup=Mock(),
            free_lookup_locks=Mock(),
            report_block_allocation=Mock(),
            get_pp_stage_status=Mock(),
        )
        for url in URLS
    }
    obj._server_model_names = dict(zip(URLS, ("stage0", "stage1"), strict=True))
    obj.model_name = "undecorated-model"
    obj._mq_timeout = 10
    obj.lmcache_tokens_per_chunk = 4
    obj._ensure_heartbeat_started = Mock()
    obj.is_healthy = True
    obj._pending_lookups, obj._unacked_lookups, obj._lookup_params = set(), {}, {}
    return obj, ns


def locators():
    return tuple(
        layout.PPStageLocator("glm", 16, 2, stage, ((0, 41), (41, 78)))
        for stage in range(2)
    )


def profile(locator, **kwargs):
    return layout.PPStageProfile(
        locator,
        (f"model.layers.{locator.layer_ranges[locator.pp_rank][0]}",),
        True,
        kwargs.get("layout_json", '{"shape":[2,4]}'),
    )


def statuses(obj, *, alias=False, incomplete=False):
    expected = locators()
    for index, (url, locator) in enumerate(zip(URLS, expected, strict=True)):
        status = layout.PPStageStatus(
            locator,
            "same" if alias else str(index),
            profile(locator),
            tuple(range(15 if incomplete and index == 1 else 16)),
        )
        obj.req_clients[url].get_pp_stage_status.return_value = NS(
            result=Mock(return_value=status)
        )
    return expected


class SchedulerTests(unittest.TestCase):
    def test_lookup_free_tail_and_reports_use_each_stage_namespace(self):
        obj, _ = scheduler()
        obj.maybe_submit_lookup_request("r", list(range(12)), cache_salt="salt")
        obj.free_lookup_locks(list(range(12)), 0, 4, "r", cache_salt="salt")
        obj._free_inconsistent_lookup_locks("r", {URLS[0]: 1, URLS[1]: 3}, 1)
        records = [object()]
        obj.report_block_allocations(records)
        for index, url in enumerate(URLS):
            client = obj.req_clients[url]
            lookup = client.lookup.call_args.args[0]
            self.assertEqual(
                (lookup.model_name, lookup.cache_salt, lookup.end),
                (f"stage{index}", "salt", 12),
            )
            self.assertEqual(
                client.free_lookup_locks.call_args_list[0].args[0].model_name,
                f"stage{index}",
            )
            self.assertEqual(
                client.report_block_allocation.call_args.args[1:],
                (f"stage{index}", records),
            )
        self.assertEqual(obj.req_clients[URLS[0]].free_lookup_locks.call_count, 1)
        tail = obj.req_clients[URLS[1]].free_lookup_locks.call_args.args[0]
        self.assertEqual((tail.model_name, tail.start, tail.end), ("stage1", 4, 12))

    def test_complete_status_installs_distinct_content_namespaces(self):
        obj, _ = scheduler()
        expected = statuses(obj)
        obj._initialize_pp_stage_profiles(expected)
        self.assertEqual(
            obj._server_model_names,
            {
                url: profile(locator).cache_namespace
                for url, locator in zip(URLS, expected, strict=True)
            },
        )
        self.assertNotEqual(*obj._server_model_names.values())

    def test_status_missing_tp_alias_or_wrong_layout_rejected_atomically(self):
        for failure in ("missing", "alias", "layout"):
            obj, _ = scheduler()
            before = obj._server_model_names.copy()
            expected = statuses(
                obj, alias=failure == "alias", incomplete=failure == "missing"
            )
            if failure == "layout":
                invalid = layout.PPStageStatus(
                    expected[0], "rear", profile(expected[0]), tuple(range(16))
                )
                obj.req_clients[
                    URLS[1]
                ].get_pp_stage_status.return_value.result.return_value = invalid
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                obj._initialize_pp_stage_profiles(expected)
            self.assertEqual(obj._server_model_names, before)

    def test_stage_status_waits_share_one_timeout_budget(self):
        obj, ns = scheduler()
        expected = statuses(obj)
        ns["time"] = NS(monotonic=Mock(side_effect=[100, 102, 109]))
        obj._initialize_pp_stage_profiles(expected)
        self.assertEqual(
            obj.req_clients[
                URLS[0]
            ].get_pp_stage_status.return_value.result.call_args.kwargs["timeout"],
            8,
        )
        self.assertEqual(
            obj.req_clients[
                URLS[1]
            ].get_pp_stage_status.return_value.result.call_args.kwargs["timeout"],
            1,
        )


class TransferFixture:
    def __init__(self, ack):
        self.register_pp = Mock(return_value=ack)


def worker():
    ns = dict(
        FixtureBase=object,
        logger=LOG,
        PPRegistrationInfo=layout.PPRegistrationInfo,
        PPRegistrationAck=layout.PPRegistrationAck,
        LMCacheDrivenTransferContext=TransferFixture,
        EngineType=NS(VLLM="vllm"),
        vllm_layout_hints=lambda: "hints",
    )
    cls = extract(
        ADAPTER,
        ns,
        class_name="LMCacheMPWorkerAdapter",
        methods={
            "register_kv_caches",
            "_send_register_kv_caches_request",
            "world_size",
            "_reregister_kv_caches_callback",
            "_heartbeat_stop_requested",
        },
    )
    obj = cls()
    obj.parallel_strategy = ParallelStrategy(True, 32, 0, 16, 2, 2, 1, True)
    obj._pp_stage_locator = locators()[0]
    obj._pp_registration_info, obj._pp_profile = None, None
    obj._pp_registration_token = "stable-worker-token"
    obj.instance_id, obj.req_client = 42, object()
    obj._pp_base_model_name, obj.model_name = "glm", "glm"
    obj._mq_timeout, obj.lmcache_tokens_per_chunk = 10, 512
    obj._mp_transfer_mode, obj._heartbeat, obj.dispatcher = None, None, None
    ack = layout.PPRegistrationAck(profile(obj._pp_stage_locator), "server-id", 42, 0)
    contexts = []

    def create(*args, **kwargs):
        context = TransferFixture(ack)
        contexts.append(context)
        return context

    ns["create_transfer_context"] = create
    return obj, ns, contexts, ack


class WorkerTests(unittest.TestCase):
    def test_registration_and_recovery_reuse_token_and_namespace(self):
        obj, _, contexts, ack = worker()
        caches = {"model.layers.0": object()}
        obj.register_kv_caches(caches)
        self.assertEqual(obj.model_name, ack.cache_namespace)
        self.assertTrue(obj._reregister_kv_caches_callback())
        self.assertEqual(len(contexts), 2)
        infos = [ctx.register_pp.call_args.args[4] for ctx in contexts]
        self.assertIs(infos[0], infos[1])
        self.assertEqual(infos[0].registration_token, "stable-worker-token")
        self.assertEqual(obj._pp_registration_token, "stable-worker-token")

    def test_wrong_registration_ack_rejected(self):
        for field, value in (
            ("instance_id", 43),
            ("tp_rank", 1),
            ("profile", profile(locators()[1])),
        ):
            obj, ns, _, ack = worker()
            bad = replace(ack, **{field: value})
            ns["create_transfer_context"] = lambda *a, _ack=bad, **kw: TransferFixture(
                _ack
            )
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "ACK"):
                obj.register_kv_caches({"model.layers.0": object()})
            self.assertIsNone(obj._pp_profile)
            self.assertEqual(obj.model_name, "glm")

    def test_changed_recovery_profile_cannot_become_healthy(self):
        obj, ns, _, ack = worker()
        obj.register_kv_caches({"model.layers.0": object()})
        changed = replace(
            ack, profile=replace(ack.profile, layout_json='{"shape":[2,8]}')
        )
        ns["create_transfer_context"] = lambda *a, **kw: TransferFixture(changed)
        self.assertFalse(obj._reregister_kv_caches_callback())
        self.assertEqual(obj._pp_profile, ack.profile)
        self.assertEqual(obj.model_name, ack.cache_namespace)

    def test_engine_driven_rejected_and_public_rebinding_rejected(self):
        obj, ns, _, _ = worker()
        ns["create_transfer_context"] = lambda *a, **kw: object()
        with self.assertRaisesRegex(ValueError, "lmcache_driven"):
            obj.register_kv_caches({"model.layers.0": object()})
        obj, _, _, _ = worker()
        obj.register_kv_caches({"model.layers.0": object()})
        with self.assertRaisesRegex(ValueError, "rebinding"):
            obj.register_kv_caches({"model.layers.0": object()})


if __name__ == "__main__":
    unittest.main()
