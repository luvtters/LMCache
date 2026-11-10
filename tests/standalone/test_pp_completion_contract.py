"""CPU source-method checks for delayed send ownership; no framework imports.

Transfer futures, telemetry and STORE metadata construction are substitutes.
The connector, tracker, metadata and worker-adapter methods are actual source.
This does not validate device transfers, PP communication or service performance.
"""
import argparse
import ast
from collections import defaultdict
import copy
from dataclasses import dataclass, field
import enum
import logging
from pathlib import Path
import sys
import types
import unittest

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
options, remaining = parser.parse_known_args()
ROOT = options.source_root.resolve()
logging.disable(logging.CRITICAL)


def source_nodes(path, class_name, method_names=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    if method_names is None:
        return [copy.deepcopy(owner)]
    result = [copy.deepcopy(n) for n in owner.body
              if isinstance(n, ast.FunctionDef) and n.name in method_names]
    assert {n.name for n in result} == set(method_names)
    for node in result:
        node.decorator_list = []
    return result


def execute(nodes, namespace, path):
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


module = types.ModuleType("_completion_contract_source")
sys.modules[module.__name__] = module
ns = module.__dict__
ns.update(dataclass=dataclass, field=field, enum=enum,
          logger=logging.getLogger("completion-contract"), ConstantList=list,
          get_read_policy=lambda: "legacy",
          KVConnectorMetadata=object,
          extract_request_configs_from_request=lambda request: None,
          extract_mm_features=lambda request: ([], []),
          _has_preemption_reqs=lambda output: False)
metadata_file = ROOT / "lmcache/integration/vllm/lmcache_mp_metadata.py"
for name in ("LMCacheMPRequestState", "LMCacheMPRequestTracker", "LMCacheMPConnectorMetadata"):
    execute(source_nodes(metadata_file, name), ns, metadata_file)
State, Tracker, Metadata = (ns[n] for n in (
    "LMCacheMPRequestState", "LMCacheMPRequestTracker", "LMCacheMPConnectorMetadata"))


class RequestStatus(enum.Enum):
    WAITING = 1
    PREEMPTED = 2


class RequestMetadata:
    @staticmethod
    def GetStoreMetadata(tracker, *args, **kwargs):
        return types.SimpleNamespace(request_id=tracker.request_id, direction="STORE")


ns.update(RequestStatus=RequestStatus, LMCacheMPRequestMetadata=RequestMetadata)
connector_file = ROOT / "lmcache/integration/vllm/lmcache_mp_connector.py"
connector_names = {"get_finished", "start_load_kv", "build_connector_meta",
                   "request_finished", "_cleanup_request_tracker",
                   "_get_or_create_request_tracker", "_process_retrieve_requests",
                   "_process_new_requests", "_process_cached_requests"}
execute(source_nodes(connector_file, "LMCacheMPConnector", connector_names), ns, connector_file)
Connector = type("ActualConnectorMethods", (), {n: ns[n] for n in connector_names})
adapter_file = ROOT / "lmcache/integration/vllm/vllm_multi_process_adapter.py"
adapter_names = {"get_finished", "_process_finished_stores", "_update_and_get_finished_store"}
execute(source_nodes(adapter_file, "LMCacheMPWorkerAdapter", adapter_names), ns, adapter_file)
Adapter = type("ActualAdapterMethods", (), {n: ns[n] for n in adapter_names})


class Future:
    def __init__(self, value=True, ready=True):
        self.value, self.ready = value, ready

    def query(self):
        return self.ready

    def result(self, timeout=None):
        return self.value


def request():
    return types.SimpleNamespace(request_id="request", cache_salt="", all_token_ids=list(range(16)),
        num_prompt_tokens=16, prompt_token_ids=list(range(16)), kv_transfer_params=None,
        status=RequestStatus.WAITING)


def schedule(new=False, cached=False):
    return types.SimpleNamespace(
        scheduled_new_reqs=[types.SimpleNamespace(req_id="request", block_ids=([1],))] if new else [],
        scheduled_cached_reqs=types.SimpleNamespace(req_ids=["request"] if cached else [],
            new_block_ids=[([2],)] if cached else [], resumed_req_ids=set()),
        num_scheduled_tokens={"request": 4} if new or cached else {})


def scheduler(can_store=True, lazy=False):
    obj = Connector()
    obj._can_store, obj.lazy_offload = can_store, lazy
    obj._delay_free_req_ids = set()
    obj.request_trackers = {"request": Tracker(request())}
    obj.scheduler_adapter = types.SimpleNamespace(lmcache_tokens_per_chunk=4,
        cleanup_lookup_result=lambda rid: None, end_session=lambda rid: None)
    obj._group_tokens_per_block, obj._mamba_relocation_window = [4], 0
    obj._save_decode_cache = False
    obj._get_request_tracker = lambda rid: obj.request_trackers[rid]
    obj._report_block_allocation_deltas = lambda output: None
    obj._lazy_offload_manager = types.SimpleNamespace(
        on_request_finished=lambda rid: types.SimpleNamespace(sessions_to_end=[]))
    return obj


def adapter(healthy=True):
    obj = Adapter()
    obj.dispatcher, obj.is_healthy = None, healthy
    for name in ("store_futures", "store_events", "retrieve_futures", "retrieve_events",
                 "_pending_store_kv_events"):
        setattr(obj, name, {})
    for name in ("error_block_ids", "_failed_request_ids", "_dropped_retrieves",
                 "finished_stores", "previously_finished", "_returned_finished"):
        setattr(obj, name, set())
    obj.model_name, obj.world_size, obj.worker_id = "cpu-contract", 16, 0
    obj.request_telemetry = types.SimpleNamespace(on_request_store_finished=lambda **kwargs: None)
    obj._publish_store_kv_events = lambda rid: None
    return obj


def worker(a=None, can_store=True):
    obj = Connector()
    obj.worker_adapter = a if a is not None else adapter()
    obj.lazy_offload, obj._can_store, obj.dispatcher = False, can_store, None
    obj._delay_free_req_ids = set()
    obj.metadata = Metadata()
    obj._get_connector_metadata = lambda: obj.metadata
    return obj


def handoff(s, w):
    w.metadata = s.build_connector_meta(schedule())
    w.start_load_kv(None)


class CompletionContract(unittest.TestCase):
    def test_failed_receive_then_engine_finish_never_sends(self):
        s, w = scheduler(), worker()
        w.worker_adapter.retrieve_futures["request"] = (Future(False), [1])
        self.assertEqual(w.get_finished(set()), (set(), {"request"}))
        self.assertFalse(s.request_finished(request(), [1])[0])
        handoff(s, w)
        self.assertEqual(w.get_finished({"request"}), (set(), set()))

    def test_same_step_failed_receive_and_finish_never_sends(self):
        s, w = scheduler(), worker()
        self.assertFalse(s.request_finished(request(), [1])[0])
        handoff(s, w)
        w.worker_adapter.retrieve_futures["request"] = (Future(False), [1])
        self.assertEqual(w.get_finished({"request"}), (set(), {"request"}))

    def test_load_only_completion_is_filtered_without_failure(self):
        w = worker()
        self.assertEqual(w.get_finished({"request"}), (set(), set()))

    def test_normal_store_waits_for_future_then_reports_once(self):
        s, w = scheduler(), worker()
        s.build_connector_meta(schedule(new=True))
        self.assertTrue(s.request_finished(request(), [1])[0])
        handoff(s, w)
        future = Future(True, ready=False)
        w.worker_adapter.store_futures["request"] = future
        self.assertEqual(w.get_finished({"request"}), (set(), set()))
        future.ready = True
        self.assertEqual(w.get_finished(set()), ({"request"}, set()))
        self.assertEqual(w.get_finished({"request"}), (set(), set()))

    def test_cached_store_also_delays_free(self):
        s = scheduler()
        s.build_connector_meta(schedule(cached=True))
        self.assertTrue(getattr(s.request_trackers["request"], "store_scheduled", False))
        self.assertTrue(s.request_finished(request(), [2])[0])

    def test_non_writer_still_acknowledges_authorized_store(self):
        s, w = scheduler(), worker()
        s.build_connector_meta(schedule(new=True))
        self.assertTrue(s.request_finished(request(), [1])[0])
        handoff(s, w)
        self.assertEqual(w.get_finished({"request"}), ({"request"}, set()))

    def test_unhealthy_worker_still_acknowledges_authorized_store(self):
        s, w = scheduler(), worker(adapter(healthy=False))
        s.build_connector_meta(schedule(new=True))
        s.request_finished(request(), [1])
        handoff(s, w)
        self.assertEqual(w.get_finished({"request"}), ({"request"}, set()))

    def test_consumer_receives_but_never_sends(self):
        s, w = scheduler(can_store=False), worker(can_store=False)
        self.assertFalse(s.request_finished(request(), [1])[0])
        handoff(s, w)
        w.worker_adapter.retrieve_futures["request"] = (Future(True), [1])
        self.assertEqual(w.get_finished({"request"}), (set(), {"request"}))

    def test_lazy_offload_keeps_existing_false_defer_contract(self):
        s = scheduler(lazy=True)
        self.assertFalse(s.request_finished(request(), [1])[0])
        self.assertEqual(s._delay_free_req_ids, set())

    def test_preemption_recreates_tracker_without_old_store_permission(self):
        s = scheduler()
        s.build_connector_meta(schedule(new=True))
        s.request_trackers["request"].state = State.READY
        req = request()
        req.status = RequestStatus.PREEMPTED
        tracker = s._get_or_create_request_tracker(req)
        self.assertFalse(getattr(tracker, "store_scheduled", False))
        self.assertFalse(s.request_finished(req, [2])[0])

    def test_two_pp_by_sixteen_tp_control_instances_complete_once(self):
        s = scheduler()
        s.build_connector_meta(schedule(new=True))
        s.request_finished(request(), [1])
        metadata = s.build_connector_meta(schedule())
        counts = defaultdict(int)
        for pp in range(2):
            for tp in range(16):
                w = worker(adapter(healthy=tp != 15))
                w.metadata = metadata
                w.start_load_kv(None)
                sends, receives = w.get_finished({"request"})
                self.assertFalse(receives)
                for rid in sends:
                    counts[rid] += 1
                self.assertEqual(w.get_finished({"request"}), (set(), set()))
        self.assertEqual(counts["request"], 32)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0], *remaining])
