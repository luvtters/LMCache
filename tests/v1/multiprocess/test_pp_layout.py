# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests loading production PP metadata without package imports."""

# Standard
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import sys
import unittest

_PATH = Path(__file__).resolve().parents[3] / "lmcache/v1/multiprocess/pp_layout.py"
_SPEC = importlib.util.spec_from_file_location("pp_layout_under_test", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
pp = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pp
_SPEC.loader.exec_module(pp)


class PPLayoutTests(unittest.TestCase):
    """Exercise public metadata validation and stable content identities."""

    def setUp(self) -> None:
        self.locator = pp.PPStageLocator("model", 2, 2, 0, ((0, 2), (2, 5)))
        self.profile = pp.PPStageProfile(
            self.locator,
            ("layer.0", "layer.1"),
            False,
            '{"dtype":"float16","shape":[2,16],"groups":["a","b"]}',
        )

    def test_topology_validation(self) -> None:
        for changes in (
            {"tp_size": 0},
            {"tp_size": True},
            {"pp_size": 0},
            {"pp_rank": 2},
            {"schema_version": 2},
            {"base_model_name": " "},
            {"layer_ranges": ((1, 2), (2, 5))},
            {"layer_ranges": ((0, 2), (3, 5))},
            {"layer_ranges": ((0, 3), (2, 5))},
            {"layer_ranges": ((0, 2), (2, 2))},
            {"layer_ranges": ((0, 5),)},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.locator, **changes)

    def test_frozen_and_normalized(self) -> None:
        locator = replace(self.locator, layer_ranges=[[0, 2], [2, 5]])
        self.assertEqual(locator, self.locator)
        with self.assertRaises(FrozenInstanceError):
            locator.tp_size = 4

    def test_worker_identity(self) -> None:
        info = pp.PPRegistrationInfo(
            self.locator, 1, 1, self.profile.layer_names, "token", False
        )
        for changes in (
            {"global_rank": 2},
            {"tp_rank": 2},
            {"registration_token": ""},
            {"layer_names": ()},
            {"layer_names": ("layer.0", "layer.0")},
            {"layer_names": ("", "layer.1")},
            {"mla_only": 1},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(info, **changes)
        other_stage = replace(self.locator, pp_rank=1)
        pp.PPRegistrationInfo(other_stage, 3, 1, ("a", "b", "c"), "token", True)

    def test_auxiliary_registered_groups_are_preserved(self) -> None:
        names = ("layer.0", "layer.1", "mtp.0", "auxiliary.cache")
        info = pp.PPRegistrationInfo(self.locator, 0, 0, names, "token", False)
        profile = replace(self.profile, layer_names=names)
        self.assertEqual(info.layer_names, names)
        self.assertEqual(profile.layer_names, names)
        self.assertNotEqual(profile.cache_namespace, self.profile.cache_namespace)

    def test_namespace_canonicalization(self) -> None:
        reordered = replace(
            self.profile,
            layout_json=' { "groups": ["a", "b"], "shape": [2,16], '
            '"dtype": "float16" } ',
        )
        self.assertEqual(self.profile, reordered)
        self.assertEqual(self.profile.cache_namespace, reordered.cache_namespace)
        self.assertTrue(
            self.profile.cache_namespace.startswith("model::lmcache-pp-v1::")
        )
        self.assertEqual(len(self.profile.cache_namespace.rsplit("::", 1)[1]), 64)

    def test_namespace_separates_content(self) -> None:
        variants = (
            replace(self.profile, layer_names=("layer.1", "layer.0")),
            replace(self.profile, mla_only=True),
            replace(self.profile, layout_json='{"dtype":"bfloat16"}'),
            replace(self.profile, layout_json='{"groups":["b","a"]}'),
            replace(self.profile, locator=replace(self.locator, tp_size=4)),
            replace(
                self.profile,
                locator=replace(self.locator, layer_ranges=((0, 2), (2, 6))),
            ),
        )
        for profile in variants:
            self.assertNotEqual(self.profile.cache_namespace, profile.cache_namespace)

    def test_layout_rejects_nonprimitive_json(self) -> None:
        for layout in ('{"x":NaN}', '{"x":Infinity}', '{"x":1,"x":2}', "bad"):
            with self.subTest(layout=layout), self.assertRaises(ValueError):
                replace(self.profile, layout_json=layout)

    def test_control_identity_does_not_change_namespace(self) -> None:
        first = pp.PPRegistrationAck(self.profile, "server-a", 10, 0)
        second = pp.PPRegistrationAck(self.profile, "server-b", 11, 1)
        self.assertEqual(first.cache_namespace, second.cache_namespace)
        for token in ("old-token", "new-token"):
            info = pp.PPRegistrationInfo(
                self.locator, 0, 0, self.profile.layer_names, token, False
            )
            profile = pp.PPStageProfile(
                info.locator, info.layer_names, info.mla_only, self.profile.layout_json
            )
            self.assertEqual(first.cache_namespace, profile.cache_namespace)

    def test_reader_geometry(self) -> None:
        self.assertEqual(
            (self.profile.kv_world_size, self.profile.num_kv_readers), (2, 1)
        )
        mla = replace(self.profile, mla_only=True)
        self.assertEqual((mla.kv_world_size, mla.num_kv_readers), (1, 2))

    def test_status_exact_complete_set(self) -> None:
        status = pp.PPStageStatus(self.locator, "server", self.profile, (1, 0))
        self.assertTrue(status.ready)
        for changes in (
            {"profile": None},
            {"registered_tp_ranks": (0,)},
            {"registered_tp_ranks": (0, 0)},
            {"registered_tp_ranks": (0, 1, 1)},
            {
                "profile": replace(
                    self.profile, locator=replace(self.locator, tp_size=4)
                )
            },
        ):
            self.assertFalse(replace(status, **changes).ready)
        with self.assertRaises(ValueError):
            replace(status, registered_tp_ranks=(0, 2))


if __name__ == "__main__":
    unittest.main()
