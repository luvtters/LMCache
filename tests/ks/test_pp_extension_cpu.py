# SPDX-License-Identifier: Apache-2.0
"""The default connector must retain its PP guard behind a virtual hook."""

from types import SimpleNamespace
import unittest

from test_minimal_cpu import CONNECTOR, function, method


class ExtensionTests(unittest.TestCase):
    def test_default_connector_still_rejects_multiple_servers_with_pp(self) -> None:
        try:
            node = method(CONNECTOR, "LMCacheMPConnector", "build_parallel_strategy")
        except StopIteration:
            self.fail("No overridable parallel-strategy hook exists")
        call = function(node, {"ParallelStrategy": object,
                              "build_parallel_strategy_from_vllm_config": lambda *_: "unchanged"})
        config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
        with self.assertRaisesRegex(ValueError, "pipeline parallelism"):
            call(object(), config, 2)

    def test_default_single_server_geometry_is_unchanged(self) -> None:
        try:
            node = method(CONNECTOR, "LMCacheMPConnector", "build_parallel_strategy")
        except StopIteration:
            self.fail("No overridable parallel-strategy hook exists")
        marker = object()
        call = function(node, {"ParallelStrategy": object,
                              "build_parallel_strategy_from_vllm_config": lambda *_: marker})
        config = SimpleNamespace(parallel_config=SimpleNamespace(pipeline_parallel_size=2))
        self.assertIs(call(object(), config, 1), marker)


if __name__ == "__main__":
    unittest.main(verbosity=2)
