# SPDX-License-Identifier: Apache-2.0
"""Inspect/start/stop the private PP model and its port-5565 cache service.

Model stop preserves cache/observers. Explicit cache-stop/cache-start require
all model processes to exit first. Every stop checks PID start identities.
"""
from __future__ import annotations

# Standard
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path("/opt/ks-luv-pp")


def read_process(pid: int) -> dict[str, object]:
    """Read a PID identity, parent, arguments and output targets without imports."""
    directory = Path("/proc") / str(pid)
    try:
        fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
        args = tuple(part.decode(errors="replace") for part in
                     (directory / "cmdline").read_bytes().split(b"\0") if part)
        outputs = []
        for fd in ("1", "2"):
            try:
                outputs.append(os.readlink(directory / "fd" / fd))
            except OSError:
                pass
        return {"pid": pid, "ppid": int(fields[1]), "start": fields[19],
                "state": fields[0], "args": args, "outputs": outputs,
                "cwd": os.readlink(directory / "cwd")}
    except (OSError, IndexError, ValueError):
        return {}


def snapshot() -> dict[int, dict[str, object]]:
    """Read processes visible inside the current container."""
    result = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdecimal():
            process = read_process(int(entry.name))
            if process:
                result[int(entry.name)] = process
    return result


def excluded(process: dict[str, object]) -> bool:
    """Exclude cache, benchmark, observer and this control helper from model stops."""
    args = process["args"]
    names = {Path(arg).name for arg in args}
    return bool(names & {"watch_pp.py", "collect_lookup_logs.py", "patchctl.py", "pp_model.py"}
                or "bench" in args
                or ("server" in args and any("lmcache" in arg for arg in args)))


def select_models(processes: dict[int, dict[str, object]], caller: int) -> set[int]:
    """Select script/log-owned model processes and their descendants, not all vLLM."""
    protected = {caller, 1}
    parent = caller
    while parent in processes:
        parent = int(processes[parent]["ppid"])
        if parent in protected:
            break
        protected.add(parent)
    selected = set()
    for pid, process in processes.items():
        if pid in protected or process["state"] == "Z" or excluded(process):
            continue
        script_owned = any(
            Path(arg).name == "startpp.sh"
            and (Path(process["cwd"]) / arg).resolve() == ROOT / "startpp.sh"
            for arg in process["args"])
        log_owned = any(
            Path(output.removesuffix(" (deleted)")).parent == ROOT / "logs"
            and Path(output.removesuffix(" (deleted)")).name.startswith("pp-")
            and Path(output.removesuffix(" (deleted)")).suffix == ".log"
            for output in process["outputs"])
        if script_owned or log_owned:
            selected.add(pid)
    while True:
        descendants = {pid for pid, process in processes.items()
                       if pid not in protected and process["state"] != "Z"
                       and not excluded(process) and process["ppid"] in selected}
        if descendants <= selected:
            return selected
        selected |= descendants


def signal_identity(process: dict[str, object], signum: int) -> bool:
    """Send a signal only while the PID still has its captured start identity."""
    current = read_process(int(process["pid"]))
    if not current or current["state"] == "Z" or current["start"] != process["start"]:
        return False
    try:
        os.kill(int(process["pid"]), signum)
        return True
    except ProcessLookupError:
        return False


def unknown_models(processes: dict[int, dict[str, object]], owned: set[int]) -> list[int]:
    """List unowned live vLLM processes so an unidentified model is never duplicated."""
    return [pid for pid, p in processes.items() if pid not in owned and p["state"] != "Z"
            and not excluded(p) and (
                any(arg.startswith("VLLM::") for arg in p["args"])
                or ("serve" in p["args"] and any(Path(arg).name == "vllm" for arg in p["args"])))]


def check_cache() -> None:
    """Require a healthy cache with old worker registrations gone; keep cached KV."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:15565/status", timeout=5) as response:
        state = json.load(response)
    if state.get("is_healthy") is not True:
        raise RuntimeError("CACHE_NOT_HEALTHY: no model started")
    workers = state.get("registered_gpu_ids")
    if not isinstance(workers, list):
        raise RuntimeError("CACHE_REGISTRATION_STATE_UNKNOWN: no model started")
    if workers:
        raise RuntimeError(f"CACHE_OLD_REGISTRATIONS_REMAIN: count={len(workers)}; wait for old workers to unregister, then start again; cached data was preserved")


def stop_models(timeout: float, force: bool) -> None:
    """Stop the owned model roots gracefully; never stop cache/observer processes."""
    processes = snapshot()
    targets = select_models(processes, os.getpid())
    known = {pid: processes[pid] for pid in targets}

    def remaining_models() -> dict[int, dict[str, object]]:
        """Keep captured identities even after workers become orphaned."""
        current = snapshot()
        for pid in select_models(current, os.getpid()):
            known[pid] = current[pid]
        return {pid: p for pid, p in known.items()
                if pid in current and current[pid]["state"] != "Z"
                and current[pid]["start"] == p["start"]}
    if not targets and unknown_models(processes, targets):
        raise RuntimeError(f"MODEL_OWNERSHIP_UNKNOWN: {unknown_models(processes, targets)}; inspect these processes first")
    roots = {pid for pid in targets if processes[pid]["ppid"] not in targets}
    wrappers = {pid for pid in targets if any(
        Path(arg).name == "startpp.sh"
        and (Path(processes[pid]["cwd"]) / arg).resolve() == ROOT / "startpp.sh"
        for arg in processes[pid]["args"])}
    roots |= {pid for pid in targets if processes[pid]["ppid"] in wrappers}
    # Non-interactive Bash need not forward TERM to the foreground vLLM.
    # Signal its actual entry process too and let that entry drain workers.
    roots |= {pid for pid in targets if (
        "serve" in processes[pid]["args"]
        and any(Path(arg).name == "vllm" for arg in processes[pid]["args"]))
        or any(arg.startswith("VLLM::APIServer") for arg in processes[pid]["args"])}
    print("MODEL_TERM_ROOTS", sorted(roots), flush=True)
    for pid in sorted(roots):
        signal_identity(processes[pid], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        remaining = remaining_models()
        if not remaining:
            print("MODEL_STOPPED")
            return
        time.sleep(.25)
    remaining = remaining_models()
    if remaining and force:
        print("MODEL_FORCE_KILL", sorted(remaining), flush=True)
        for pid in remaining:
            signal_identity(remaining[pid], signal.SIGKILL)
        time.sleep(.5)
        remaining = remaining_models()
    if remaining:
        raise RuntimeError(f"MODEL_STOP_TIMEOUT: {sorted(remaining)}; no new model started. Inspect these PIDs; --force applies only to this model.")
    print("MODEL_STOPPED")


def cache_processes(processes: dict[int, dict[str, object]]) -> dict[int, dict[str, object]]:
    """Select only an lmcache CLI server explicitly listening on port 5565."""
    result = {}
    for pid, process in processes.items():
        args = process["args"]
        if process["state"] == "Z":
            continue
        for index, arg in enumerate(args[:-1]):
            if Path(arg).name != "lmcache" or args[index + 1] != "server":
                continue
            options = args[index + 2:]
            if "--port=5565" in options or any(
                options[n:n + 2] == ("--port", "5565")
                for n in range(len(options) - 1)
            ):
                result[pid] = process
    return result


def require_models_stopped(processes: dict[int, dict[str, object]]) -> None:
    """Refuse cache maintenance while a model or an unidentified worker is live."""
    models = select_models(processes, os.getpid())
    models |= set(unknown_models(processes, models))
    if models:
        raise RuntimeError(f"MODEL_STILL_RUNNING: {sorted(models)}; stop models on both hosts first")


def stop_cache(timeout: float, force: bool) -> None:
    """Stop the verified local cache after model shutdown, preserving observers."""
    processes = snapshot()
    require_models_stopped(processes)
    targets = cache_processes(processes)
    print("CACHE_TERM_PIDS", sorted(targets), flush=True)
    for process in targets.values():
        signal_identity(process, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while True:
        current = snapshot()
        remaining = {pid: process for pid, process in targets.items()
                     if pid in current and current[pid]["state"] != "Z"
                     and current[pid]["start"] == process["start"]}
        if not remaining:
            print("CACHE_STOPPED")
            return
        if time.monotonic() >= deadline:
            break
        time.sleep(.25)
    if force:
        print("CACHE_FORCE_KILL", sorted(remaining), flush=True)
        for process in remaining.values():
            signal_identity(process, signal.SIGKILL)
        time.sleep(.5)
        current = snapshot()
        remaining = {pid: process for pid, process in remaining.items()
                     if pid in current and current[pid]["state"] != "Z"
                     and current[pid]["start"] == process["start"]}
    if remaining:
        raise RuntimeError(f"CACHE_STOP_TIMEOUT: {sorted(remaining)}; no new cache started")
    print("CACHE_STOPPED")


def read_text_path(path: str) -> str:
    """Read a Linux status file, returning empty only when it is absent."""
    try:
        return Path(path).read_text().strip()
    except FileNotFoundError:
        return ""


def check_cache_budget(size_gib: int) -> None:
    """Require the requested DDR allocation plus 64 GiB in host/cgroup headroom."""
    info = dict(line.split(":", 1) for line in read_text_path("/proc/meminfo").splitlines())
    available = int(info["MemAvailable"].split()[0]) * 1024
    for limit_path, used_path in (
        ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
        ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes"),
    ):
        limit, used = read_text_path(limit_path), read_text_path(used_path)
        if limit and used and limit != "max":
            available = min(available, max(0, int(limit) - int(used)))
    print("CACHE_AVAILABLE_GIB", round(available / (1 << 30), 1),
          "REQUESTED_GIB", size_gib, "MIN_RESERVE_GIB", 64, flush=True)
    if available < (size_gib + 64) * (1 << 30):
        raise RuntimeError("CACHE_MEMORY_BUDGET: insufficient host/cgroup headroom; no cache started")


def check_cache_ports() -> None:
    """Refuse duplicate/foreign listeners before allocating a large DDR pool."""
    for host, port in (("0.0.0.0", 5565), ("127.0.0.1", 15565)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind((host, port))


def cache_status() -> dict[str, object]:
    """Read the local management endpoint with proxies disabled."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open("http://127.0.0.1:15565/status", timeout=2) as response:
        return json.load(response)


def wait_cache_ready(pid: int, size_gib: int, timeout: float, log: Path) -> None:
    """Wait for this cache's healthy configured pool; timeout never kills it."""
    identity = read_process(pid)
    deadline = time.monotonic() + timeout
    next_message = 0.0
    while time.monotonic() < deadline:
        current = read_process(pid)
        if not identity or not current or current["state"] == "Z" or current["start"] != identity["start"]:
            raise RuntimeError(f"CACHE_EXITED_DURING_LAUNCH: LOG={log}")
        try:
            state = cache_status()
            managers = state.get("storage_manager", {}).get("l1_managers", {})
            capacity = sum(value.get("memory_configured_bytes", 0) for value in managers.values())
            if state.get("is_healthy") is True and capacity == size_gib * (1 << 30):
                print(f"CACHE_READY SIZE_GIB={size_gib} PID={pid} LOG={log}", flush=True)
                return
        except (OSError, ValueError):
            pass
        if time.monotonic() >= next_message:
            print(f"CACHE_LOADING PID={pid} LOG={log}", flush=True)
            next_message = time.monotonic() + 10
        time.sleep(1)
    raise RuntimeError(f"CACHE_NOT_READY_YET: process was preserved; inspect LOG={log}")


def start_cache(size_gib: int, ready_timeout: float) -> None:
    """Start the original cache configuration with a new pool and INFO statistics."""
    processes = snapshot()
    require_models_stopped(processes)
    if cache_processes(processes):
        raise RuntimeError("CACHE_ALREADY_RUNNING: use cache-stop first; no duplicate launch")
    if size_gib <= 0:
        raise ValueError("cache size must be positive")
    check_cache_budget(size_gib)
    check_cache_ports()
    environment = ROOT / "env.sh"
    if not environment.is_file():
        raise FileNotFoundError(environment)
    command = ["bash", "-c",
        'set -e; source "$1"; unset LMCACHE_ASCEND_PP_STAGE_LOCAL; shift; exec lmcache server "$@"',
        "pp-cache", str(environment), "--host", "0.0.0.0", "--port", "5565",
        "--http-host", "127.0.0.1", "--http-port", "15565", "--chunk-size", "512",
        "--supported-transfer-mode", "lmcache_driven", "--l1-size-gb", str(size_gib),
        "--eviction-policy", "LRU", "--l1-align-bytes", "65536", "--disable-metrics",
        "--max-gpu-workers", "1", "--enable-extra-logging", "--extra-logging-interval", "10"]
    logs = ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = logs / f"lmcache-{stamp}-{os.getpid()}.log"
    with log.open("xb") as output:
        child = subprocess.Popen(command, cwd=Path.cwd(), stdin=subprocess.DEVNULL,
            stdout=output, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
    (ROOT / "cache-size-gib.txt").write_text(str(size_gib) + "\n", encoding="utf-8")
    (logs / "cache-process.json").write_text(json.dumps({
        "pid": child.pid, "log": str(log), "size_gib": size_gib, "state": "submitted"
    }) + "\n", encoding="utf-8")
    print(f"CACHE_SUBMITTED PID={child.pid} LOG={log}", flush=True)
    wait_cache_ready(child.pid, size_gib, ready_timeout, log)


def start_model() -> None:
    """Submit the unchanged startup script with a private log and detached session."""
    processes = snapshot()
    existing = select_models(processes, os.getpid())
    if existing:
        raise RuntimeError(f"MODEL_ALREADY_RUNNING: {sorted(existing)}; no duplicate launch")
    unknown = unknown_models(processes, existing)
    if unknown:
        raise RuntimeError(f"MODEL_OWNERSHIP_UNKNOWN: {unknown}; no duplicate launch")
    check_cache()
    script = ROOT / "startpp.sh"
    content = script.read_bytes()
    if b"\r" in content or content.startswith(b"\xef\xbb\xbf"):
        raise RuntimeError("START_SCRIPT_HAS_CR_OR_BOM: fix script format before launching")
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log = log_dir / f"pp-{stamp}-{os.getpid()}.log"
    cwd = Path.cwd()
    with log.open("xb") as output:
        child = subprocess.Popen(["bash", str(script)], cwd=cwd,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True, close_fds=True)
    time.sleep(.3)
    if child.poll() is not None:
        raise RuntimeError(f"MODEL_EXITED_DURING_LAUNCH code={child.returncode} LOG={log}")
    receipt = {"pid": child.pid, "process": read_process(child.pid),
               "log": str(log), "cwd": str(cwd), "state": "submitted"}
    (log_dir / "model-process.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"MODEL_SUBMITTED PID={child.pid} LOG={log}")
    print("85 and 86 must both start; HTTP 200 on 85 establishes service readiness")


def print_status() -> None:
    """Print model, cache and observer PID groups without dumping large argv."""
    processes = snapshot()
    models = select_models(processes, os.getpid())
    caches = [pid for pid, p in processes.items() if p["state"] != "Z"
              and "server" in p["args"] and any("lmcache" in a for a in p["args"])]
    watchers = [pid for pid, p in processes.items() if p["state"] != "Z"
                and any(Path(a).name == "watch_pp.py" for a in p["args"])
                and "_run" in p["args"]]
    print("MODEL_COUNT", len(models), "MODEL_PIDS", sorted(models))
    print("UNOWNED_VLLM_PIDS", unknown_models(processes, models))
    print("CACHE_PIDS", sorted(caches))
    print("WATCHER_PIDS", sorted(watchers))
    print("LAST_LAUNCH", read_receipt())
    try:
        state = cache_status()
        managers = state.get("storage_manager", {}).get("l1_managers", {})
        print("CACHE_HEALTHY", state.get("is_healthy"), "POOL_GIB",
              sum(p.get("memory_configured_bytes", 0) for p in managers.values()) / (1 << 30),
              "REGISTERED_WORKERS", len(state.get("registered_gpu_ids", [])))
    except (OSError, ValueError) as error:
        print("CACHE_STATUS_UNAVAILABLE", str(error))
    try:
        print("CACHE_LAST_LAUNCH", json.loads((ROOT / "logs/cache-process.json").read_text()))
    except (OSError, ValueError):
        pass


def read_receipt() -> dict[str, object]:
    """Read only the previous launch's log/state, not treat an old PID as alive."""
    try:
        data = json.loads((ROOT / "logs/model-process.json").read_text())
        return {key: data.get(key) for key in ("pid", "log", "cwd", "state")}
    except (OSError, ValueError):
        return {}


def main() -> None:
    """Run one explicit model/cache action; never control the observer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop", "start", "restart", "cache-stop", "cache-start"))
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--size-gib", type=int, default=1024)
    parser.add_argument("--ready-timeout", type=float, default=600)
    args = parser.parse_args()
    if os.name != "posix" or not Path("/proc/self/stat").exists():
        parser.error("run inside the customer Linux container")
    if min(args.timeout, args.ready_timeout, args.size_gib) <= 0:
        parser.error("timeouts and size must be positive")
    if args.action == "status":
        print_status()
        return
    # Prevent two terminals from concurrently stopping/starting this same model.
    import fcntl
    with (ROOT / "model-control.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.action in ("stop", "restart"):
            stop_models(args.timeout, args.force)
        if args.action in ("start", "restart"):
            start_model()
        if args.action == "cache-stop":
            stop_cache(args.timeout, args.force)
        if args.action == "cache-start":
            start_cache(args.size_gib, args.ready_timeout)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("PP_ACTION_FAILED:", str(error), file=sys.stderr)
        raise SystemExit(1)
