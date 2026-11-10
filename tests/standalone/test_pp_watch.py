# SPDX-License-Identifier: Apache-2.0
"""Targeted CPU checks for the independent collector, using local fake HTTP."""
from __future__ import annotations

# Standard
import argparse
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "patches/native-pp-recovery-20261110"


def load_script(name: str) -> ModuleType:
    """Load a standalone helper without importing any framework package."""
    spec = importlib.util.spec_from_file_location(name, PACKAGE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


watch = load_script("watch_pp")
collector = load_script("collect_lookup_logs")


class FakeHTTP(BaseHTTPRequestHandler):
    count = 0

    def do_GET(self) -> None:
        if self.path == "/metrics":
            type(self).count += 1
            n = type(self).count
            body = (f'vllm:num_requests_running{{engine="0"}} 3\n'
                    f'vllm:num_requests_waiting{{engine="0"}} 12\n'
                    f'vllm:lmcache_mp_lookup_duration_seconds_count {n}\n'
                    f'vllm:lmcache_mp_lookup_duration_seconds_sum {n * .05}\n'
                    f'vllm:prompt_tokens_by_source_total{{source="external_kv_transfer"}} {n * 512}\n').encode()
        else:
            body = json.dumps({"is_healthy": True, "active_sessions": 3,
                "active_prefetch_jobs": 7, "registered_gpu_ids": list(range(16)),
                "storage_manager": {"l1_managers": {"default": {
                    "read_ttl_seconds": 300, "read_locked_count": 16,
                    "memory_usage_ratio": .7}}}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


class WatchChecks(unittest.TestCase):
    def test_prometheus_labels_and_sources(self) -> None:
        text = ('vllm:num_requests_waiting{engine="0"} 12\n'
                'vllm:num_requests_waiting{engine="1"} 3\n'
                'vllm:prompt_tokens_by_source_total{model_name="x",source="external_kv_transfer"} 512\n')
        values = watch.model_metrics(text)
        self.assertEqual(values["waiting"], 15)
        self.assertEqual(values["prompt_source.external_kv_transfer"], 512)

    def test_old_log_snapshot_stops_at_original_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, contextlib.redirect_stdout(io.StringIO()):
            path = Path(temporary) / "growing.log"
            path.write_bytes(b"old line\n" * 1000)
            iterator = collector.snapshot_lines(path, 1024)
            first = next(iterator)
            with path.open("ab") as stream:
                stream.write(b"NEW_AFTER_SNAPSHOT\n" * 100)
            lines = [first, *iterator]
            self.assertFalse(any("NEW_AFTER_SNAPSHOT" in line for _, line in lines))
            self.assertTrue(lines[0][0].startswith("tail+"))
            self.assertLess(len(lines), 120)

    def test_log_flood_is_bounded_and_keeps_head_and_tail_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            path = logs / "pp-fixture.log"
            path.touch()
            args = argparse.Namespace(node="85", log_dir=logs)
            capture = watch.Capture(args, root)
            try:
                path.write_bytes(b"Something went wrong with retrieve\n" + b"x" * 2300000
                                 + b"\nEngineCore encountered a fatal error\n")
                capture.tail()
                self.assertGreater(capture.data["log_bytes_skipped"], 0)
                self.assertEqual(capture.data["event_counts"]["retrieve_failed"], 1)
                self.assertEqual(capture.data["event_counts"]["engine_fatal"], 1)
            finally:
                capture.output.close()

    def test_detached_start_duplicate_summary_and_stop(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeHTTP)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "logs"
            logs.mkdir()
            model_log = logs / "pp-fixture.log"
            model_log.write_text("EngineCore encountered a fatal error OLD\n", encoding="utf-8")
            target = root / "capture"
            base = [sys.executable, str(PACKAGE / "watch_pp.py")]
            common = ["--root", str(target)]
            url = "http://127.0.0.1:" + str(server.server_port)
            try:
                result = subprocess.run([*base, "start", *common, "--node", "85",
                    "--model-url", url, "--cache-url", url, "--log-dir", str(logs),
                    "--interval", ".1", "--cache-interval", ".15", "--max-seconds", "30"],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("WATCH_READY", result.stdout)
                duplicate = subprocess.run([*base, "start", *common], capture_output=True,
                    text=True, timeout=5)
                self.assertIn("WATCH_ALREADY_RUNNING", duplicate.stdout)
                self.assertEqual(len(list(target.glob("*/ready.json"))), 1)
                with model_log.open("a", encoding="utf-8") as stream:
                    stream.write("Something went wrong with retrieve NEW\n")
                time.sleep(.7)
                stopped = subprocess.run([*base, "stop", *common], capture_output=True,
                    text=True, timeout=15)
                self.assertEqual(stopped.returncode, 0, stopped.stderr)
                self.assertIn("WATCH_STOPPED", stopped.stdout)
                self.assertLessEqual(len(stopped.stdout.splitlines()), 20)
                run = Path(json.loads((target / "latest.json").read_text())["run_dir"])
                state = json.loads((run / "summary.json").read_text())
                self.assertFalse(state["running"])
                self.assertGreaterEqual(state["sample_counts"]["model"], 2)
                self.assertEqual(state["maxima"]["waiting"], 12)
                self.assertGreater(state["prompt_token_deltas"]["external_kv_transfer"], 0)
                self.assertAlmostEqual(state["maxima"]["lookup_interval_avg_ms"], 50)
                self.assertEqual(state["event_counts"].get("retrieve_failed"), 1)
                self.assertNotIn("engine_fatal", state["event_counts"])
            finally:
                run = watch.latest_run(target)
                if run:
                    (run / "STOP").touch()
                    for _ in range(50):
                        if not watch.alive(watch.read_json(run / "ready.json")):
                            break
                        time.sleep(.1)
                server.shutdown()
                server.server_close()


if __name__ == "__main__":
    unittest.main()
