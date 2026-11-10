# SPDX-License-Identifier: Apache-2.0
"""Detached, read-only PP incident capture using only the Python standard library.

Records HTTP metrics/status, new log events and CPU process state. It does not
import device frameworks, send inference, change source/config, or stop services.
Aggregate observations cannot establish per-request lock age or pure RPC time.
"""
from __future__ import annotations

# Standard
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_ROOT = Path("/opt/ks-luv-pp/watch")
EVENTS = {
    "retrieve_failed": r"Some keys not found during retrieve|Something went wrong.*retrieve",
    "missing_read_key": r"finish read on non-existing key|finish read on non-read-locked key",
    "request_failed": r"Failing \d+ request.*KV load failure",
    "engine_fatal": r"EngineCore encountered a fatal error",
    "mq_timeout": r"QUERY_PREFETCH_STATUS.*timed out|LOOKUP.*timed out|TimeoutError",
    "disconnected": r"Response destination disconnected|ZMQError: Host unreachable",
    "lookup_log": r"LMCache MP lookup avg latency",
}
RULES = {name: re.compile(pattern) for name, pattern in EVENTS.items()}
SAMPLE = re.compile(r'^([\w:]+)(?:\{([^}]*)\})?\s+([^\s]+)(?:\s+\d+)?$')


def utc_now() -> str:
    """Return a UTC timestamp; retain source log clocks separately."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    """Atomically publish a small local receipt or summary at path."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, object]:
    """Read an object-shaped JSON receipt, or return an empty object."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def process_state(pid: int) -> dict[str, object]:
    """Read Linux PID identity, CPU ticks, RSS and wait channel without signals."""
    root = Path("/proc") / str(pid)
    try:
        fields = (root / "stat").read_text().rsplit(")", 1)[1].split()
        pages = (root / "statm").read_text().split()
        return {"pid": pid, "state": fields[0], "start_ticks": fields[19],
                "cpu_ticks": int(fields[11]) + int(fields[12]),
                "rss_mb": int(pages[1]) * os.sysconf("SC_PAGE_SIZE") / 1048576,
                "wchan": (root / "wchan").read_text().strip()}
    except (OSError, ValueError, IndexError):
        return {}


def alive(receipt: dict[str, object]) -> bool:
    """Check the capture process itself, with start-time matching on Linux."""
    pid = int(receipt.get("pid", 0))
    if pid <= 0:
        return False
    if os.name != "nt":
        state = process_state(pid)
        return bool(state and state.get("state") != "Z"
                    and state.get("start_ticks") == receipt.get("start_ticks"))
    # Windows is supported only for the local CPU launch/stop check.
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    try:
        code = wintypes.DWORD()
        return bool(kernel.GetExitCodeProcess(handle, ctypes.byref(code)) and code.value == 259)
    finally:
        kernel.CloseHandle(handle)


def http_get(url: str) -> tuple[int, str, float]:
    """Read up to 2 MiB with a socket timeout; report probe elapsed milliseconds."""
    begin = time.monotonic()
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=2) as response:
            body = response.read(2 * 1024 * 1024 + 1)
            if len(body) > 2 * 1024 * 1024:
                raise ValueError("response exceeds 2 MiB capture limit")
            code, text = response.status, body.decode("utf-8", errors="replace")
    except urllib.error.HTTPError as error:
        code, text = error.code, str(error)
    except Exception as error:
        code, text = 0, str(error)[:200]
    return code, text, (time.monotonic() - begin) * 1000


def model_metrics(text: str) -> dict[str, float]:
    """Extract existing gauges and counters; never infer internal phase timings."""
    names = {
        "vllm:num_requests_running": "running",
        "vllm:num_requests_waiting": "waiting",
        "vllm:num_requests_deferred": "deferred",
        "vllm:kv_cache_usage_perc": "gpu_kv_fraction",
        "vllm:gpu_cache_usage_perc": "gpu_kv_fraction",
        "vllm:lmcache_mp_lookup_duration_seconds_count": "lookup_count",
        "vllm:lmcache_mp_lookup_duration_seconds_sum": "lookup_seconds",
    }
    result: dict[str, float] = {}
    for line in text.splitlines():
        match = SAMPLE.match(line)
        if match is None:
            continue
        name, labels, raw = match.groups()
        key = names.get(name)
        if name == "vllm:prompt_tokens_by_source_total":
            source = re.search(r'(?:^|,)\s*source="([^"]+)"', labels or "")
            if source:
                key = "prompt_source." + source.group(1)
        if key is None:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            result[key] = result.get(key, 0) + value
    return result


def cache_metrics(text: str) -> dict[str, object]:
    """Keep cache counters and capacity/lock state, excluding payload/layout data."""
    data = json.loads(text)
    manager = data.get("storage_manager", {})
    fields = ("read_ttl_seconds", "read_locked_count", "write_locked_count",
              "memory_used_bytes", "memory_configured_bytes", "memory_usage_ratio",
              "staging_object_count", "total_object_count")
    return {"healthy": data.get("is_healthy"),
            "registered_workers": len(data.get("registered_gpu_ids", [])),
            "active_sessions": data.get("active_sessions"),
            "active_prefetch_jobs": data.get("active_prefetch_jobs"),
            "l1": {name: {key: values.get(key) for key in fields}
                   for name, values in manager.get("l1_managers", {}).items()}}


def process_snapshot() -> tuple[list[dict[str, object]], list[Path]]:
    """Find local model/cache processes and regular cache stdout files on Linux."""
    records: list[dict[str, object]] = []
    logs: list[Path] = []
    if os.name == "nt":
        return records, logs
    for entry in Path("/proc").iterdir():
        if not entry.name.isdecimal():
            continue
        try:
            args = (entry / "cmdline").read_bytes()[:4096].split(b"\0")
            command = b" ".join(args)
            role = ""
            if b"EngineCore" in command:
                role = "core"
            elif b"serve" in args and b"vllm" in command:
                role = "model"
            elif b"server" in args and b"lmcache" in command and b"5565" in command:
                role = "cache"
            if not role:
                continue
            record = process_state(int(entry.name))
            if role == "cache":
                target = Path(os.readlink(entry / "fd/1"))
                if record:
                    record["stdout"] = str(target)
                if target.is_file():
                    logs.append(target)
            if record:
                records.append({"role": role, **record})
        except (OSError, ValueError):
            continue
    return records, logs


class Capture:
    """Own local JSONL/summary output; serialize records from two probe threads."""

    def __init__(self, args: argparse.Namespace, run: Path) -> None:
        self.args, self.run = args, run
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.output = (run / "samples.jsonl").open("a", encoding="utf-8", buffering=1)
        self.data: dict[str, object] = {"node": args.node, "running": True,
            "started_utc": utc_now(), "output_dir": str(run), "sample_counts": {},
            "maxima": {}, "event_counts": {}, "first_errors": [], "last": {},
            "logs": [], "log_bytes_skipped": 0, "lookup_new_count": 0,
            "lookup_new_seconds": 0.0, "lookup_counter_resets": 0,
            "prompt_token_deltas": {},
            "limits": "read-only aggregate capture; no per-request lock age or pure RPC timing"}
        self.previous_lookup: tuple[float, float] | None = None
        self.previous_sources: dict[str, float] = {}
        self.previous_process: dict[tuple[int, str], tuple[float, int]] = {}
        self.offsets: dict[Path, int] = {}
        self.buffers: dict[Path, bytes] = {}
        self.cache_logs: list[Path] = []
        self.discover(initial=True)

    def discover(self, initial: bool = False) -> None:
        """Track newest model/cache logs and the current cache's stdout target."""
        selected = list(self.cache_logs)
        for pattern in ("pp-*.log", "*cache*.log"):
            paths: list[tuple[float, Path]] = []
            for path in self.args.log_dir.glob(pattern):
                try:
                    if path.is_file():
                        paths.append((path.stat().st_mtime, path))
                except OSError:
                    continue
            if paths:
                selected.append(max(paths, key=lambda item: item[0])[1])
        for path in set(selected):
            try:
                if path.is_file() and path not in self.offsets:
                    self.offsets[path] = path.stat().st_size if initial else 0
                    self.buffers[path] = b""
            except OSError:
                continue
        self.data["logs"] = [str(path) for path in self.offsets]

    def record(self, kind: str, values: dict[str, object]) -> None:
        """Append one observation and update only measured maxima/counter deltas."""
        entry = {"utc": utc_now(), "monotonic": time.monotonic(), "kind": kind, **values}
        with self.lock:
            self.output.write(json.dumps(entry, ensure_ascii=False) + "\n")
            counts = self.data["sample_counts"]
            counts[kind] = counts.get(kind, 0) + 1
            self.data["last"][kind] = entry
            maxima = self.data["maxima"]
            if kind == "model" and isinstance(values.get("metrics"), dict):
                metrics = values["metrics"]
                for key, value in metrics.items():
                    if key.startswith("prompt_source."):
                        previous = self.previous_sources.get(key)
                        if previous is not None and value >= previous:
                            name = key.removeprefix("prompt_source.")
                            deltas = self.data["prompt_token_deltas"]
                            deltas[name] = deltas.get(name, 0) + int(value - previous)
                        self.previous_sources[key] = value
                for name in ("running", "waiting", "deferred", "gpu_kv_fraction"):
                    if name in metrics:
                        maxima[name] = max(maxima.get(name, 0), metrics[name])
                if "lookup_count" in metrics and "lookup_seconds" in metrics:
                    current = (metrics["lookup_count"], metrics["lookup_seconds"])
                    if self.previous_lookup is not None:
                        dc, ds = (current[n] - self.previous_lookup[n] for n in (0, 1))
                        if dc < 0 or ds < 0:
                            self.data["lookup_counter_resets"] += 1
                        elif dc > 0:
                            self.data["lookup_new_count"] += int(dc)
                            self.data["lookup_new_seconds"] += ds
                            maxima["lookup_interval_avg_ms"] = max(
                                maxima.get("lookup_interval_avg_ms", 0), ds / dc * 1000)
                    self.previous_lookup = current
            if kind == "cache" and isinstance(values.get("metrics"), dict):
                metrics = values["metrics"]
                for name in ("active_sessions", "active_prefetch_jobs"):
                    value = metrics.get(name)
                    if isinstance(value, (int, float)):
                        maxima[name] = max(maxima.get(name, 0), value)
                for stats in metrics.get("l1", {}).values():
                    for name in ("memory_usage_ratio", "read_locked_count"):
                        value = stats.get(name)
                        if isinstance(value, (int, float)):
                            maxima[name] = max(maxima.get(name, 0), value)
            if kind == "event":
                name = str(values["event"])
                events = self.data["event_counts"]
                events[name] = events.get(name, 0) + 1
                errors = self.data["first_errors"]
                if name != "lookup_log" and len(errors) < 4:
                    errors.append(entry)
            self.data["updated_utc"] = utc_now()

    def probe(self, kind: str, url: str, interval: float) -> None:
        """Sample one endpoint independently so a slow cache cannot block metrics."""
        while not self.stop.is_set():
            code, body, elapsed = http_get(url)
            values: dict[str, object] = {"http": code, "probe_elapsed_ms": elapsed}
            try:
                if code == 200:
                    values["metrics"] = model_metrics(body) if kind == "model" else cache_metrics(body)
                else:
                    values["error"] = body[:200]
            except (ValueError, TypeError, AttributeError) as error:
                values["error"] = str(error)
            self.record(kind, values)
            # Status traverses cache bookkeeping. Back off if that probe is slow.
            delay = max(interval, 30) if kind == "cache" and elapsed > 1000 else interval
            self.stop.wait(delay)

    def tail(self) -> None:
        """Read bounded new bytes only, with explicit accounting for log floods."""
        for path in list(self.offsets):
            try:
                with path.open("rb") as stream:
                    end = os.fstat(stream.fileno()).st_size
                    offset = self.offsets[path]
                    if end < offset:
                        offset = 0
                        self.buffers[path] = b""
                    stream.seek(offset)
                    block = stream.read(min(65536, end - offset))
                    offset += len(block)
                    if end - offset > 2 * 1024 * 1024:
                        target = max(offset, end - 65536)
                        self.data["log_bytes_skipped"] += target - offset
                        stream.seek(target)
                        stream.readline(min(65536, end - target))
                        block += b"\n" + stream.read(max(0, end - stream.tell()))
                        offset = end
                        self.buffers[path] = b""
                    self.offsets[path] = offset
                rows = (self.buffers[path] + block).split(b"\n")
                self.buffers[path] = rows.pop()[:32768]
                for raw in rows:
                    line = raw[:4096].decode("utf-8", errors="replace")
                    for name, pattern in RULES.items():
                        if pattern.search(line):
                            self.record("event", {"event": name, "log": str(path), "line": line[:500]})
                            break
            except OSError as error:
                self.record("log_error", {"log": str(path), "error": str(error)})

    def save(self, running: bool = True) -> None:
        """Publish a consistent summary while endpoint threads may be sampling."""
        with self.lock:
            self.data["running"] = running
            self.data["updated_utc"] = utc_now()
            write_json(self.run / "summary.json", self.data)

    def run_capture(self) -> None:
        """Run the capture until its own STOP file or duration, never stopping models."""
        threads = [threading.Thread(target=self.probe, daemon=True,
            args=("cache", self.args.cache_url + "/status", self.args.cache_interval))]
        if self.args.node == "85":
            threads.append(threading.Thread(target=self.probe, daemon=True,
                args=("model", self.args.model_url + "/metrics", self.args.interval)))
        self.save()
        receipt = {"pid": os.getpid(), "node": self.args.node, "run_dir": str(self.run),
                   "start_ticks": process_state(os.getpid()).get("start_ticks"), "ready_utc": utc_now()}
        write_json(self.run / "ready.json", receipt)
        for thread in threads:
            thread.start()
        deadline = time.monotonic() + self.args.max_seconds
        next_process = next_save = 0.0
        try:
            while time.monotonic() < deadline and not (self.run / "STOP").exists():
                now = time.monotonic()
                if now >= next_process:
                    processes, logs = process_snapshot()
                    for process in processes:
                        key = (int(process["pid"]), str(process["start_ticks"]))
                        current = (now, int(process["cpu_ticks"]))
                        previous = self.previous_process.get(key)
                        if previous and now > previous[0]:
                            process["cpu_pct"] = ((current[1] - previous[1])
                                / os.sysconf("SC_CLK_TCK") / (now - previous[0]) * 100)
                        self.previous_process[key] = current
                    self.cache_logs = logs
                    # Newly discovered stdout targets may predate this run.
                    for path in logs:
                        if path not in self.offsets:
                            try:
                                self.offsets[path], self.buffers[path] = path.stat().st_size, b""
                            except OSError:
                                continue
                    self.discover()
                    self.record("processes", {"values": processes, "cache_stdout_logs": [str(p) for p in logs]})
                    next_process = now + self.args.cache_interval
                self.tail()
                if now >= next_save:
                    self.save()
                    next_save = now + self.args.interval
                self.stop.wait(min(0.5, self.args.interval))
        finally:
            self.stop.set()
            for thread in threads:
                thread.join(timeout=5)
            self.tail()
            self.save(running=False)
            self.output.close()


def latest_run(root: Path) -> Path | None:
    """Resolve the capture-owned latest pointer within the requested root."""
    value = read_json(root / "latest.json").get("run_dir")
    if not isinstance(value, str):
        return None
    path = Path(value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("capture pointer is outside capture root")
    return path


def print_summary(run: Path) -> None:
    """Print a short, pasteable report and retain full records locally."""
    data = read_json(run / "summary.json")
    receipt = read_json(run / "ready.json")
    print("WATCH_NODE", data.get("node"), "PROCESS_ALIVE", alive(receipt))
    print("WATCH_WINDOW_UTC", data.get("started_utc"), data.get("updated_utc"))
    print("SAMPLES", data.get("sample_counts", {}))
    print("PEAKS", data.get("maxima", {}))
    count = data.get("lookup_new_count", 0)
    duration = data.get("lookup_new_seconds", 0.0)
    print("LOOKUP_NEW_COUNT", count, "OBSERVED_MEAN_MS", duration / count * 1000 if count else "NO_NEW_COMPLETIONS")
    print("LOOKUP_COUNTER_RESETS", data.get("lookup_counter_resets", 0))
    print("PROMPT_TOKEN_DELTAS", data.get("prompt_token_deltas", {}), "(between usable samples)")
    print("EVENT_COUNTS", data.get("event_counts", {}), "LOG_BYTES_SKIPPED", data.get("log_bytes_skipped", 0))
    last = data.get("last", {})
    for name in ("model", "cache"):
        item = last.get(name, {})
        print("LAST_PROBE", name, "HTTP", item.get("http", "NOT_SAMPLED"), "ELAPSED_MS", round(item.get("probe_elapsed_ms", 0), 2), "ERROR", item.get("error"))
    cache = last.get("cache", {}).get("metrics", {})
    print("LAST_CACHE", "HEALTHY", cache.get("healthy"), "REGISTERED_WORKERS", cache.get("registered_workers"))
    processes = last.get("processes", {}).get("values", [])
    print("LAST_PROCESSES", [{key: p.get(key) for key in ("role", "pid", "state", "cpu_pct", "rss_mb", "wchan", "stdout")} for p in processes])
    print("LOGS", data.get("logs", []))
    for error in data.get("first_errors", []):
        print("FIRST_ERROR", error.get("utc"), error.get("event"), str(error.get("line", ""))[:240])
    print("OUTPUT_DIR", str(run))
    print("LIMIT", data.get("limits"))


def main() -> None:
    """Start, summarize or stop only this independent diagnostic capture process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "summary", "stop", "_run"))
    parser.add_argument("--node", choices=("85", "86"), default="85")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--log-dir", type=Path, default=Path("/opt/ks-luv-pp/logs"))
    parser.add_argument("--model-url", default="http://127.0.0.1:8000")
    parser.add_argument("--cache-url", default="http://127.0.0.1:15565")
    parser.add_argument("--interval", type=float, default=5)
    parser.add_argument("--cache-interval", type=float, default=15)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    if min(args.interval, args.cache_interval, args.max_seconds) <= 0:
        parser.error("capture intervals and duration must be positive")
    args.root = args.root.resolve()
    if args.action == "_run":
        if args.run_dir is None:
            parser.error("internal capture requires --run-dir")
        Capture(args, args.run_dir).run_capture()
        return
    run = latest_run(args.root)
    if args.action == "start":
        if run and alive(read_json(run / "ready.json")):
            print("WATCH_ALREADY_RUNNING")
            print_summary(run)
            return
        args.root.mkdir(parents=True, exist_ok=True)
        name = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + str(os.getpid())
        run = args.root / name
        run.mkdir()
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "_run",
            "--node", args.node, "--root", str(args.root), "--run-dir", str(run),
            "--log-dir", str(args.log_dir), "--model-url", args.model_url,
            "--cache-url", args.cache_url, "--interval", str(args.interval),
            "--cache-interval", str(args.cache_interval), "--max-seconds", str(args.max_seconds)]
        with (run / "watcher.log").open("ab") as output:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                stderr=subprocess.STDOUT, close_fds=True, start_new_session=os.name != "nt",
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        for _ in range(100):
            ready = read_json(run / "ready.json")
            if ready and alive(ready):
                write_json(args.root / "latest.json", {"run_dir": str(run)})
                print("WATCH_READY", "NODE=" + args.node, "PID=" + str(child.pid), "DIR=" + str(run))
                return
            if child.poll() is not None:
                break
            time.sleep(0.05)
        raise RuntimeError("capture did not become ready; inspect " + str(run / "watcher.log"))
    if run is None:
        raise RuntimeError("no capture receipt found")
    if args.action == "stop":
        (run / "STOP").touch()
        for _ in range(80):
            if not alive(read_json(run / "ready.json")):
                break
            time.sleep(0.1)
        print("WATCH_STOP_REQUESTED" if alive(read_json(run / "ready.json")) else "WATCH_STOPPED")
    print_summary(run)


if __name__ == "__main__":
    main()
