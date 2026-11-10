"""Print a bounded summary of existing incident logs; no inference or changes."""
import argparse
import ast
from collections import Counter, deque
import datetime
import json
import importlib.util
from pathlib import Path
import re
import socket
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-log", type=Path)
    parser.add_argument("--request-id")
    parser.add_argument("--start", default="03:40:00")
    parser.add_argument("--end", default="03:51:40")
    args = parser.parse_args()
    log_dir = Path("/opt/ks-luv-pp/logs")
    candidates = list(log_dir.glob("pp-*.log"))
    model_log = args.model_log
    if model_log is None and candidates:
        model_log = max(candidates, key=lambda p: p.stat().st_mtime)
    print("HOST", socket.gethostname())
    print("WINDOW", args.start, args.end, "(log clock; no timezone conversion)")
    clock = re.compile(r"11-10\s+(\d\d:\d\d:\d\d)")
    options = re.compile(r"['\"](lmcache\.mp\.(?:mq_timeout|nonblocking_lookup_status|eager_prefetch))['\"]\s*:\s*(True|False|true|false|\d+(?:\.\d+)?)")
    rules = {
        "LOOKUP_METRIC": re.compile(r"LMCache MP lookup (?:count|avg latency)"),
        "ENGINE_PRESSURE": re.compile(r"Avg prompt throughput:.*Running:"),
        "FAILURE": re.compile(r"Some keys not found during retrieve|Something went wrong.*retrieve|Failing \d+ request|EngineCore encountered a fatal error|TimeoutError|timed out"),
        "L1_STATE": re.compile(r"finish read on non-existing key|finish read on non-read-locked key|L1 read reserved|L1 read finished|L1 eviction|L1Manager: cleared"),
    }
    def summarize(path, model=False):
        counts = Counter()
        samples = {key: deque(maxlen=3) for key in rules}
        request_lines = deque(maxlen=5)
        first_request = None
        startup_options = {}
        with path.open(encoding="utf-8", errors="replace") as stream:
            for line_number, line in enumerate(stream, 1):
                if model:
                    startup_options.update(options.findall(line))
                match = clock.search(line)
                if match is None or not args.start <= match.group(1) <= args.end:
                    continue
                brief = f"{line_number}: {line.rstrip()[:420]}"
                for label, pattern in rules.items():
                    if pattern.search(line):
                        counts[label] += 1
                        samples[label].append(brief)
                if args.request_id and args.request_id in line:
                    if first_request is None:
                        first_request = brief
                    request_lines.append(brief)
        print("LOG", str(path), "MATCH_COUNTS", dict(counts))
        if model:
            print("RECORDED_STARTUP_OPTIONS", startup_options or "NOT_RECORDED")
        for label in rules:
            for line in samples[label]:
                print(label, line)
        if model and args.request_id:
            print("REQUEST_FIRST_RECORDED_LINE", first_request or "NOT_RECORDED_IN_WINDOW")
            for line in request_lines:
                if line != first_request:
                    print("REQUEST_LAST", line)
    if model_log and model_log.is_file():
        summarize(model_log, model=True)
    else:
        print("MODEL_LOG_NOT_FOUND")
    cache_logs = sorted(log_dir.glob("*cache*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:1]
    for path in cache_logs:
        summarize(path)
    if not cache_logs:
        print("CACHE_LOG_FILE_NOT_FOUND: foreground cache output may not have been saved")
    try:
        spec = importlib.util.find_spec("lmcache")
        source = Path(spec.origin).resolve().parent / "integration/vllm/vllm_multi_process_adapter.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ExtraConfigDefault")
        defaults = {}
        for node in config.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for name in node.targets:
                    if isinstance(name, ast.Name) and name.id in ("mq_timeout", "nonblocking_lookup_status"):
                        defaults[name.id] = node.value.value
        print("CURRENT_SOURCE_DEFAULTS", defaults, "(startup overrides take priority)")
    except Exception as error:
        print("SOURCE_DEFAULTS_UNAVAILABLE", str(error))
    try:
        http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with http.open("http://127.0.0.1:15565/status", timeout=5) as response:
            status = json.load(response)
        fields = ("read_ttl_seconds", "read_locked_count", "memory_used_bytes",
                  "memory_configured_bytes", "memory_usage_ratio")
        print("CURRENT_SNAPSHOT_UTC", datetime.datetime.now(datetime.timezone.utc).isoformat())
        print("CURRENT_CACHE_HEALTHY", status.get("is_healthy"))
        for name, value in status.get("storage_manager", {}).get("l1_managers", {}).items():
            print("CURRENT_L1", name, {key: value.get(key) for key in fields})
    except Exception as error:
        print("CURRENT_CACHE_STATUS_UNAVAILABLE", str(error))
    print("LIMIT: INFO logs without per-request lookup timestamps cannot split server work, RPC queue and scheduler polling waits")


if __name__ == "__main__":
    main()
