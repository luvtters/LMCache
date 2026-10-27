# SPDX-License-Identifier: Apache-2.0
"""Source-only import contract; optional implementation loading is a stand-in.

This executes the real package/API/factory modules with isolated parent-package
and annotation dependencies. It does not import a native backend or test a
complete RequestClient; the real process-exit check runs in the device image.
"""

# Standard
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch
import importlib
import importlib.util
import logging
import sys
import unittest


IMPL = "lmcache.v1.distributed.transfer_channel.impl"
FACTORY = "lmcache.v1.distributed.transfer_channel.factory"


class _ImplementationLoader(MetaPathFinder, Loader):
    def __init__(self) -> None:
        self.loads = 0
        self.creations: list[dict[str, Any]] = []

    def find_spec(
        self, fullname: str, path: Any, target: Any = None
    ) -> ModuleSpec | None:
        if fullname == IMPL:
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec: ModuleSpec) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        self.loads += 1

        def create(**kwargs: Any) -> dict[str, Any]:
            self.creations.append(kwargs)
            return kwargs

        factory = importlib.import_module(FACTORY)
        factory.register_transfer_channel_factory("lazy-test", create)


class LazyImportTests(unittest.TestCase):
    def test_api_import_defers_implementations_until_factory_use(self) -> None:
        root = Path(__file__).parents[4]
        isolated: dict[str, ModuleType] = {}
        for name in ("lmcache", "lmcache.v1", "lmcache.v1.distributed"):
            module = ModuleType(name)
            module.__path__ = [str(root.joinpath(*name.split(".")))]
            isolated[name] = module
        logger = ModuleType("lmcache.logging")
        logger.init_logger = logging.getLogger
        isolated[logger.__name__] = logger
        annotations = ModuleType("lmcache.v1.distributed.internal_api")
        annotations.L1MemoryDesc = object
        isolated[annotations.__name__] = annotations
        loader = _ImplementationLoader()
        sys.meta_path.insert(0, loader)
        try:
            with patch.dict(sys.modules, isolated):
                for name in list(sys.modules):
                    if name.startswith("lmcache.v1.distributed.transfer_channel"):
                        del sys.modules[name]
                api = importlib.import_module(
                    "lmcache.v1.distributed.transfer_channel.api"
                )
                factory = importlib.import_module(FACTORY)
                self.assertEqual(api.TransferChannelAddress(0, 32).size, 32)
                self.assertNotIn(IMPL, sys.modules)
                self.assertEqual(loader.loads, 0)
                for index in range(2):
                    result = factory.create_transfer_channel_context(
                        "lazy-test", object(), "listen", "advertise", index=index
                    )
                    self.assertEqual(result["index"], index)
                    self.assertEqual(loader.loads, 1)
                self.assertEqual(len(loader.creations), 2)
                with self.assertRaisesRegex(ValueError, "Unsupported"):
                    factory.create_transfer_channel_context(
                        "unknown", object(), "listen", "advertise"
                    )
                self.assertEqual(loader.loads, 1)
        finally:
            sys.meta_path.remove(loader)


if __name__ == "__main__":
    unittest.main()
