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
        self.kill_patch = patch.object(signal, "SIGKILL", 9, create=True)
        self.kill_patch.start()
        self.log = str(self.root / "logs/pp-current.log")

    def tearDown(self) -> None:
        self.kill_patch.stop()
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

    def test_stop_does_not_lose_a_reparented_worker(self) -> None:
        wrapper = process(10, 1, ("bash", str(self.root / "startpp.sh")))
        worker = process(11, 10, ("VLLM::Worker",))
        orphan = {**worker, "ppid": 1}
        with patch.object(control, "snapshot", side_effect=[{10: wrapper, 11: worker}, {11: orphan}, {}]), patch.object(control, "read_process", side_effect=lambda pid: {10: wrapper, 11: orphan}.get(pid, {})), patch.object(control.os, "kill") as kill, patch.object(control.time, "monotonic", side_effect=[0, 1]), patch.object(control.time, "sleep"):
            control.stop_models(.1, True)
            self.assertIn(call(11, signal.SIGKILL), kill.call_args_list)

    def test_cache_stop_requires_model_shutdown_first(self) -> None:
        rows = {10: process(10, 1, ("bash", str(self.root / "startpp.sh")))}
        with patch.object(control, "snapshot", return_value=rows), patch.object(control.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "MODEL_STILL_RUNNING"):
                control.stop_cache(.1, False)
            kill.assert_not_called()

    def test_cache_stop_targets_only_the_5565_server(self) -> None:
        cache = process(20, 1, ("python3", "/usr/local/bin/lmcache", "server", "--port", "5565"))
        other = process(21, 1, ("lmcache", "server", "--port", "6000"))
        observer = process(22, 1, ("python3", "watch_pp.py", "_run"))
        with patch.object(control, "snapshot", side_effect=[{20: cache, 21: other, 22: observer}, {21: other, 22: observer}]), patch.object(control, "read_process", return_value=cache), patch.object(control.os, "kill") as kill:
            control.stop_cache(.1, False)
            kill.assert_called_once_with(20, signal.SIGTERM)

    def test_cache_budget_honors_cgroup_limit_and_reserve(self) -> None:
        gib = 1 << 30
        values = {
            "/proc/meminfo": f"MemAvailable: {2400 * 1024**2} kB\n",
            "/sys/fs/cgroup/memory.max": str(512 * gib),
            "/sys/fs/cgroup/memory.current": str(128 * gib),
        }
        with patch.object(control, "read_text_path", side_effect=lambda p: values.get(p, "")):
            with self.assertRaisesRegex(RuntimeError, "CACHE_MEMORY_BUDGET"):
                control.check_cache_budget(1024)
            control.check_cache_budget(256)
        values["/sys/fs/cgroup/memory.max"] = "max"
        with patch.object(control, "read_text_path", side_effect=lambda p: values.get(p, "")):
            control.check_cache_budget(1024)

    def test_cache_launch_preserves_flags_and_uses_requested_size(self) -> None:
        (self.root / "env.sh").write_text("true\n")
        child = types.SimpleNamespace(pid=502, poll=lambda: None, returncode=None)
        with patch.object(control, "snapshot", return_value={}), patch.object(control, "check_cache_budget"), patch.object(control, "check_cache_ports"), patch.object(control, "wait_cache_ready") as ready, patch.object(control, "read_process", return_value={}), patch.object(control.subprocess, "Popen", return_value=child) as launch:
            control.start_cache(1024, 600)
            command = launch.call_args.args[0]
            self.assertEqual(command[command.index("--l1-size-gb") + 1], "1024")
            for flag in ("--disable-metrics", "--enable-extra-logging", "--max-gpu-workers"):
                self.assertIn(flag, command)
            self.assertEqual((self.root / "cache-size-gib.txt").read_text(), "1024\n")
            ready.assert_called_once()

    def test_cache_launch_refuses_an_existing_cache(self) -> None:
        cache = process(20, 1, ("lmcache", "server", "--port", "5565"))
        with patch.object(control, "snapshot", return_value={20: cache}), patch.object(control.subprocess, "Popen") as launch:
            with self.assertRaisesRegex(RuntimeError, "CACHE_ALREADY_RUNNING"):
                control.start_cache(1024, 600)
            launch.assert_not_called()

    def test_cache_ready_requires_the_requested_pool_size(self) -> None:
        cache = process(20, 1, ("lmcache", "server", "--port", "5565"))
        states = [{"is_healthy": True, "storage_manager": {"l1_managers": {
            "_default": {"memory_configured_bytes": size * (1 << 30)}
        }}} for size in (128, 1024)]
        with patch.object(control, "read_process", return_value=cache), patch.object(control, "cache_status", side_effect=states) as status, patch.object(control.time, "sleep"):
            control.wait_cache_ready(20, 1024, 1, self.root / "cache.log")
            self.assertEqual(status.call_count, 2)

    def test_cache_ready_reports_a_dead_process_without_signaling(self) -> None:
        cache = process(20, 1, ("lmcache", "server", "--port", "5565"))
        with patch.object(control, "read_process", side_effect=[cache, {}]), patch.object(control.os, "kill") as kill:
            with self.assertRaisesRegex(RuntimeError, "CACHE_EXITED_DURING_LAUNCH"):
                control.wait_cache_ready(20, 1024, 1, self.root / "cache.log")
            kill.assert_not_called()


if __name__ == "__main__":
    unittest.main()
