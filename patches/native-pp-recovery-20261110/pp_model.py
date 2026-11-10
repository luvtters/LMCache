# SPDX-License-Identifier: Apache-2.0
"""Inspect/start/stop only the PP model owned by /opt/ks-luv-pp/startpp.sh.

Linux container helper. Cache and independent observer processes are excluded.
Stop verifies PID start times, uses TERM first, and refuses a new start if old
model processes remain. --force is an explicit option for those model remnants.
"""
from __future__ import annotations

# Standard
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
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
        remaining = select_models(snapshot(), os.getpid())
        if not remaining:
            print("MODEL_STOPPED")
            return
        time.sleep(.25)
    remaining_processes = snapshot()
    remaining = select_models(remaining_processes, os.getpid())
    if remaining and force:
        print("MODEL_FORCE_KILL", sorted(remaining), flush=True)
        for pid in remaining:
            signal_identity(remaining_processes[pid], signal.SIGKILL)
        time.sleep(.5)
        remaining = select_models(snapshot(), os.getpid())
    if remaining:
        raise RuntimeError(f"MODEL_STOP_TIMEOUT: {sorted(remaining)}; no new model started. Inspect these PIDs; --force applies only to this model.")
    print("MODEL_STOPPED")


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


def read_receipt() -> dict[str, object]:
    """Read only the previous launch's log/state, not treat an old PID as alive."""
    try:
        data = json.loads((ROOT / "logs/model-process.json").read_text())
        return {key: data.get(key) for key in ("pid", "log", "cwd", "state")}
    except (OSError, ValueError):
        return {}


def main() -> None:
    """Run a container-local model action while preserving cache and capture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("status", "stop", "start", "restart"))
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if os.name != "posix" or not Path("/proc/self/stat").exists():
        parser.error("run inside the customer Linux container")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
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


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("MODEL_ACTION_FAILED:", str(error), file=sys.stderr)
        raise SystemExit(1)
