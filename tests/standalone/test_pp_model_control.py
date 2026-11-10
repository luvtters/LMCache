# SPDX-License-Identifier: Apache-2.0
"""CPU ownership/signal guard checks; never terminate a real model or process."""
from __future__ import annotations

# Standard
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
from unittest.mock import call

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pp_model", ROOT / "patches/native-pp-recovery-20261110/pp_model.py")
control = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(control)


def process(pid: int, ppid: int, args: tuple[str, ...], output: str = "", state: str = "S") -> dict[str, object]:
    """Construct process metadata as supplied by the Linux proc reader."""
    return {"pid": pid, "ppid": ppid, "args": args, "outputs": [output],
            "state": state, "start": str(pid * 10), "cwd": str(control.ROOT)}


class ModelControlChecks(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        self.root_patch = patch.object(control, "ROOT", self.root)
        self.root_patch.start()
        self.log = str(self.root / "logs/pp-current.log")

    def tearDown(self) -> None:
        self.root_patch.stop()
        self.temporary.cleanup()

    def test_ownership_keeps_cache_observer_benchmark_and_shell(self) -> None:
        rows = {
            1: process(1, 0, ("init",)),
            2: process(2, 1, ("bash",), self.log),
            100: process(100, 2, ("python3", "pp_model.py", "stop")),
            10: process(10, 2, ("bash", str(self.root / "startpp.sh"))),
            11: process(11, 10, ("VLLM::EngineCore",)),
            12: process(12, 11, ("VLLM::Worker",)),
            13: process(13, 1, ("VLLM::Worker",), self.log),
            20: process(20, 2, ("lmcache", "server", "--port", "5565"), self.log),
            21: process(21, 2, ("python3", "watch_pp.py", "_run"), self.log),
            22: process(22, 2, ("vllm", "bench", "serve"), self.log),
            23: process(23, 2, ("VLLM::Worker",), self.log, state="Z"),
            24: process(24, 2, ("VLLM::EngineCore",), str(self.root / "other/model.log")),
        }
        self.assertEqual(control.select_models(rows, 100), {10, 11, 12, 13})
        self.assertEqual(control.unknown_models(rows, {10, 11, 12, 13}), [24])

    def test_pid_reuse_and_zombie_are_not_signaled(self) -> None:
        original = process(10, 1, ("VLLM::Worker",), self.log)
        changed = {**original, "start": "999"}
        with patch.object(control, "read_process", return_value=changed), patch.object(control.os, "kill") as kill:
            self.assertFalse(control.signal_identity(original, signal.SIGTERM))
            kill.assert_not_called()
        with patch.object(control, "read_process", return_value={**original, "state": "Z"}), patch.object(control.os, "kill") as kill:
            self.assertFalse(control.signal_identity(original, signal.SIGTERM))
            kill.assert_not_called()

    def test_stop_terms_owned_root_only_and_preserves_other_services(self) -> None:
        root = process(10, 1, ("vllm", "serve"), self.log)
        worker = process(11, 10, ("VLLM::Worker",), self.log)
        cache = process(20, 1, ("lmcache", "server", "--port", "5565"))
        observer = process(21, 1, ("python3", "watch_pp.py", "_run"))
        with patch.object(control, "snapshot", side_effect=[{10: root, 11: worker, 20: cache, 21: observer}, {20: cache, 21: observer}]), patch.object(control, "read_process", return_value=root), patch.object(control.os, "kill") as kill:
            control.stop_models(.1, False)
            kill.assert_called_once_with(10, signal.SIGTERM)

    def test_duplicate_start_is_rejected(self) -> None:
        rows = {10: process(10, 1, ("VLLM::Worker",), self.log)}
        with patch.object(control, "snapshot", return_value=rows), patch.object(control.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "MODEL_ALREADY_RUNNING"):
                control.start_model()
            launch.assert_not_called()

    def test_stop_signals_model_entry_as_well_as_bash_wrapper(self) -> None:
        wrapper = process(10, 1, ("bash", str(self.root / "startpp.sh")))
        entry = process(11, 10, ("VLLM::HeadlessEngine",), self.log)
        worker = process(12, 11, ("VLLM::Worker",), self.log)
        rows = {10: wrapper, 11: entry, 12: worker}
        with patch.object(control, "snapshot", side_effect=[rows, {}]), patch.object(control, "read_process", side_effect=lambda pid: rows[pid]), patch.object(control.os, "kill") as kill:
            control.stop_models(.1, False)
            self.assertEqual(kill.call_args_list, [call(10, signal.SIGTERM), call(11, signal.SIGTERM)])

    def test_launch_keeps_original_script_and_working_directory(self) -> None:
        (self.root / "startpp.sh").write_bytes(b"#!/bin/bash\nexit 0\n")
        child = types.SimpleNamespace(pid=501, poll=lambda: None, returncode=None)
        with patch.object(control, "snapshot", return_value={}), patch.object(control, "check_cache"), patch.object(control.time, "sleep"), patch.object(control, "read_process", return_value={}), patch.object(control.subprocess, "Popen", return_value=child) as launch:
            control.start_model()
            self.assertEqual(launch.call_args.args[0], ["bash", str(self.root / "startpp.sh")])
            self.assertEqual(launch.call_args.kwargs["cwd"], Path.cwd())
            self.assertTrue(launch.call_args.kwargs["start_new_session"])
            receipt = json.loads((self.root / "logs/model-process.json").read_text())
            self.assertEqual(receipt["state"], "submitted")

    def test_stale_cache_registration_blocks_new_model(self) -> None:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = json.dumps({
            "is_healthy": True, "registered_gpu_ids": [10, 11]}).encode()
        opener = MagicMock()
        opener.open.return_value = response
        with patch.object(control.urllib.request, "build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "CACHE_OLD_REGISTRATIONS_REMAIN"):
                control.check_cache()


if __name__ == "__main__":
    unittest.main()
