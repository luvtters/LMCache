# SPDX-License-Identifier: Apache-2.0
"""Validate PP options before constructing any transport clients."""
from __future__ import annotations

import ast
import enum
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

from test_lookup_deadline_contract import methods

SOURCE = "lmcache/integration/vllm/vllm_multi_process_adapter.py"


class CreatedClient(Exception):
    """Stop a successful construction at the real factory boundary."""


class AdapterConfigTest(unittest.TestCase):
    def construct(self, owner, *, stage_local=True, pp_size=2, budget=32, fast=False,
                  nonblocking=True, policy="transfer_borrow", urls=None):
        tree = ast.parse((Path(__file__).resolve().parents[2] / SOURCE).read_text(encoding="utf-8"))
        defaults = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExtraConfigDefault")
        ns = {"enum": enum, "Any": object}
        exec(compile(ast.Module(body=[defaults], type_ignores=[]), SOURCE, "exec"), ns)
        defaults = ns["ExtraConfigDefault"]
        cfg = {item.name: item.default for item in defaults}
        cfg.update(pp_control_outbound_budget=budget, pp_query_fastpath=fast,
                   nonblocking_lookup_status=nonblocking)
        created = []
        def factory(url, **kwargs):
            created.append((url, kwargs))
            raise CreatedClient()
        strategy = NS(pp_stage_local=stage_local, pp_size=pp_size)
        cls = methods(SOURCE, owner, {"__init__"}, {
            "ExtraConfigDefault": defaults, "DEFAULT_MQ_TIMEOUT": 300,
            "DEFAULT_HEARTBEAT_INTERVAL": 10, "_EXTRA_CONFIG_KEY_PREFIX": "lmcache.mp.",
            "_normalize_adapter_init_args": lambda *a: (16, strategy, 300, False),
            "_resolve_extra_config": lambda _: cfg,
            "get_read_policy": lambda: policy,
            "RequestClientFactory": NS(create=factory),
            "set_ipc_policy": lambda **k: None,
        })
        urls = urls or ["tcp://pp0:1", "tcp://pp1:1"]
        try:
            if owner == "LMCacheMPSchedulerAdapter":
                cls(urls, object(), "model", 16, strategy, extra_config={})
            else:
                cls(urls[0], object(), "model", 16, strategy,
                    extra_config={}, pp_stage_locator=object())
        except (CreatedClient, ValueError) as exc:
            return exc, created
        self.fail("Constructor should stop at factory or config rejection")

    def test_budget_reaches_scheduler_and_worker_factory(self):
        for owner in ("LMCacheMPSchedulerAdapter", "LMCacheMPWorkerAdapter"):
            error, created = self.construct(owner)
            self.assertIsInstance(error, CreatedClient)
            self.assertEqual(created[0][1]["outbound_batch_limit"], 32)

    def test_non_pp_and_negative_budget_reject_before_client_creation(self):
        for owner in ("LMCacheMPSchedulerAdapter", "LMCacheMPWorkerAdapter"):
            for kwargs in ({"stage_local": False}, {"pp_size": 1}, {"budget": -1}):
                error, created = self.construct(owner, **kwargs)
                self.assertIsInstance(error, ValueError)
                self.assertEqual(created, [])

    def test_default_zero_preserves_non_pp_constructor_path(self):
        for owner in ("LMCacheMPSchedulerAdapter", "LMCacheMPWorkerAdapter"):
            error, created = self.construct(owner, stage_local=False, pp_size=1, budget=0)
            self.assertIsInstance(error, CreatedClient)
            self.assertEqual(created[0][1]["outbound_batch_limit"], 0)

    def test_fastpath_rejects_legacy_blocking_and_duplicate_urls(self):
        for kwargs in ({"policy": "legacy"}, {"nonblocking": False},
                       {"urls": ["tcp://pp0:1", "tcp://pp0:1"]}):
            error, created = self.construct("LMCacheMPSchedulerAdapter", fast=True, **kwargs)
            self.assertIsInstance(error, ValueError)
            self.assertEqual(created, [])

    def test_mixed_grpc_budget_rejects_before_first_zmq_client(self):
        error, created = self.construct("LMCacheMPSchedulerAdapter", urls=["tcp://pp0:1", "grpc://pp1:1"])
        self.assertIsInstance(error, ValueError)
        self.assertEqual(created, [])


if __name__ == "__main__":
    unittest.main()
