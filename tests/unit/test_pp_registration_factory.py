"""Execute factory ASTs without importing accelerator runtimes."""

import ast
from functools import partial
from pathlib import Path
from types import SimpleNamespace
import unittest
import sys
from types import ModuleType
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load_node(path, name, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if getattr(n, "name", None) == name)
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(
                    body=[
                        ast.ImportFrom(
                            module="__future__",
                            names=[ast.alias(name="annotations")],
                            level=0,
                        ),
                        node,
                    ],
                    type_ignores=[],
                )
            ),
            str(path),
            "exec",
        ),
        namespace,
    )
    return namespace[name]


class RegistrationFactoryTests(unittest.TestCase):
    def test_default_hook_strips_metadata(self):
        cls = load_node(
            ROOT / "lmcache/v1/platform/base/device_spec.py", "DeviceSpec", {}
        )
        calls = []
        spec = cls()
        spec.create_cache_context = lambda *a, **k: calls.append((a, k)) or "legacy"
        self.assertEqual(
            spec.create_cache_context_for_registration(
                1, registration_info=object(), x=2
            ),
            "legacy",
        )
        self.assertEqual(calls, [((1,), {"x": 2})])

    def test_factory_legacy_and_registration_dispatch(self):
        calls = []
        spec = SimpleNamespace(
            create_cache_context=lambda *a: calls.append(("legacy", a)),
            create_cache_context_for_registration=lambda *a, **k: calls.append(
                ("pp", a, k)
            ),
        )
        factory = load_node(
            ROOT / "lmcache/v1/platform/cache_context.py",
            "create_cache_context",
            {
                "EngineType": SimpleNamespace(VLLM=1),
                "partial": partial,
                "_detect_device_type": lambda _: "cpu",
                "get_device_spec": lambda _: spec,
            },
        )
        factory([object()])
        metadata = object()
        factory([object()], registration_info=metadata)
        self.assertEqual(calls[0][0], "legacy")
        self.assertEqual(calls[1][2], {"registration_info": metadata})
        self.assertEqual(len(calls[0][1]), 7)

    def test_plugin_lazy_idempotent_and_legacy(self):
        plugin = (
            ROOT.parent
            / "lmcache-ascend-native-pp-luv"
            / "lmcache_ascend/v1/pp_stage_local_activation.py"
        )
        namespace = {"__name__": "activation_under_test"}
        npu = ModuleType("lmcache.v1.platform.devices.npu")

        class Spec:
            def create_cache_context(self, *args, **kwargs):
                return ("legacy", args, kwargs)

        npu.NpuDeviceSpec = Spec
        pinned_name = "lmcache_ascend.v1.pp_stage_pinning"
        with patch.dict(sys.modules, {npu.__name__: npu}):
            exec(
                compile(plugin.read_text(encoding="utf-8"), str(plugin), "exec"),
                namespace,
            )
            self.assertNotIn(pinned_name, sys.modules)
            original = Spec.create_cache_context
            namespace["install_npu_registration_context"]()
            hook = Spec.create_cache_context_for_registration
            namespace["install_npu_registration_context"]()
            self.assertIs(Spec.create_cache_context_for_registration, hook)
            self.assertIs(Spec.create_cache_context, original)
            self.assertNotIn(pinned_name, sys.modules)
            spec = Spec()
            self.assertEqual(
                spec.create_cache_context_for_registration(1, registration_info=None),
                ("legacy", (1,), {}),
            )
            pinned = ModuleType(pinned_name)
            pinned.PinnedNpuCacheContext = lambda *a, **k: ("pinned", a, k)
            with patch.dict(sys.modules, {pinned_name: pinned}):
                self.assertEqual(
                    spec.create_cache_context_for_registration(
                        2, registration_info=object(), x=3
                    ),
                    ("pinned", (2,), {"x": 3}),
                )
                namespace["install_npu_stage_pinning"]()
                self.assertEqual(spec.create_cache_context(4), ("pinned", (4,), {}))

    def test_bootstrap_installs_new_hook_before_legacy_flag(self):
        path = ROOT.parent / "lmcache-ascend-native-pp-luv/lmcache_ascend/__init__.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bootstrap = next(
            n
            for n in tree.body
            if isinstance(n, ast.If)
            and ast.unparse(n.test) == "not LMCACHE_ASCEND_PATCHED"
        )
        calls = [ast.unparse(n) for n in bootstrap.body]
        new_index = calls.index("install_npu_registration_context()")
        legacy = next(
            (i, n)
            for i, n in enumerate(bootstrap.body)
            if isinstance(n, ast.If)
            and "LMCACHE_ASCEND_PP_STAGE_LOCAL" in ast.unparse(n.test)
        )
        self.assertLess(new_index, legacy[0])
        self.assertIn("install_npu_stage_pinning()", ast.unparse(legacy[1]))


if __name__ == "__main__":
    unittest.main()
