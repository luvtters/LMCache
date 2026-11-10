"""Apply/reverse only the completion patch against the installed LMCache source."""
import argparse
import ast
import importlib.util
from pathlib import Path
import subprocess
import sys

TARGETS = (
    "lmcache/integration/vllm/lmcache_mp_connector.py",
    "lmcache/integration/vllm/lmcache_mp_metadata.py",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("apply", "rollback"))
    args = parser.parse_args()
    spec = importlib.util.find_spec("lmcache")
    if spec is None or not spec.origin:
        raise RuntimeError("LMCACHE_PACKAGE_NOT_FOUND: source /opt/ks-luv-pp/env.sh first")
    root = Path(spec.origin).resolve().parent.parent
    patch = Path(__file__).resolve().with_name("0001-lmcache-fix-delayed-send-completion.patch")
    if not patch.is_file():
        raise RuntimeError("PATCH_FILE_MISSING")
    for name in TARGETS:
        ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
    print(f"LMCACHE_ROOT={root}", flush=True)

    def run(reverse, check):
        command = ["git", "-C", str(root), "apply"]
        if reverse:
            command.append("--reverse")
        if check:
            command.append("--check")
        command.append(str(patch))
        return subprocess.run(command, capture_output=True, text=True)

    reverse = args.action == "rollback"
    requested = run(reverse, True)
    if requested.returncode:
        if run(not reverse, True).returncode == 0:
            print("ALREADY_REVERTED" if reverse else "ALREADY_APPLIED")
            return
        print("SOURCE_MISMATCH: neither complete apply nor complete rollback matches; no files changed", file=sys.stderr)
        print(requested.stderr.strip()[:1200], file=sys.stderr)
        raise SystemExit(1)

    # git apply validates every hunk in both files before writing either;
    # never use --reject or copy whole backups over customer edits.
    result = run(reverse, False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    try:
        for name in TARGETS:
            ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
    except SyntaxError:
        undo = run(not reverse, False)
        if undo.returncode:
            raise RuntimeError("SYNTAX_CHECK_FAILED; automatic undo also failed: " + undo.stderr.strip())
        raise RuntimeError("SYNTAX_CHECK_FAILED; this operation was undone")
    print("PATCH_REVERTED" if reverse else "PATCH_APPLIED")
    print("MODEL_RESTART_REQUIRED: restart both PP model stages to load the changed Python code")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"PATCH_OPERATION_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
