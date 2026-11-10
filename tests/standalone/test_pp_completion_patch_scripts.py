"""Exercise the actual apply/rollback manager against isolated source fixtures."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "patches/native-pp-recovery-20261110"
TARGETS = ("lmcache/integration/vllm/lmcache_mp_connector.py",
           "lmcache/integration/vllm/lmcache_mp_metadata.py")


class PatchScripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lmcache-completion-patch-")
        self.root = Path(self.temp.name)
        self.env = dict(os.environ)
        self.env.update(PYTHONPATH=str(self.root), GIT_CONFIG_COUNT="1",
                        GIT_CONFIG_KEY_0="core.autocrlf", GIT_CONFIG_VALUE_0="false")
        self.original = {}
        for name in TARGETS:
            data = subprocess.check_output(["git", "-C", str(ROOT), "show",
                "89d93a2ea62030f6890b5bbe8f902c5b92945c8d:" + name])
            self.original[name] = data
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        # find_spec must locate this package without executing its __init__.
        (self.root / "lmcache/__init__.py").write_text(
            "raise RuntimeError('PACKAGE_MUST_NOT_BE_IMPORTED')\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def run_manager(self, action):
        return subprocess.run([sys.executable, str(PACKAGE / "patchctl.py"), action],
            env=self.env, cwd=self.root, capture_output=True, text=True)

    def snapshot(self):
        return {name: (self.root / name).read_bytes() for name in TARGETS}

    def test_apply_repeat_rollback_repeat_preserves_customer_edits(self):
        target = self.root / TARGETS[0]
        target.write_bytes(target.read_bytes() + b"\n# unrelated customer edit\n")
        before = self.snapshot()
        result = self.run_manager("apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PATCH_APPLIED", result.stdout)
        self.assertIn(str(self.root), result.stdout)
        self.assertNotIn("PACKAGE_MUST_NOT_BE_IMPORTED", result.stderr)
        self.assertNotEqual(self.snapshot(), before)
        result = self.run_manager("apply")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ALREADY_APPLIED", result.stdout)
        result = self.run_manager("rollback")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PATCH_REVERTED", result.stdout)
        self.assertEqual(self.snapshot(), before)
        result = self.run_manager("rollback")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("ALREADY_REVERTED", result.stdout)
        self.assertEqual(self.snapshot(), before)

    def test_unknown_source_stops_before_changing_either_file(self):
        target = self.root / TARGETS[0]
        data = target.read_bytes().replace(
            b"return self._can_store, (return_params or None)",
            b"return False, (return_params or None)")
        target.write_bytes(data)
        before = self.snapshot()
        result = self.run_manager("apply")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SOURCE_MISMATCH", result.stderr)
        self.assertEqual(self.snapshot(), before)

    def test_partial_patch_stops_before_changing_either_file(self):
        (self.root / TARGETS[0]).write_bytes((ROOT / TARGETS[0]).read_text(
            encoding="utf-8").encode("utf-8"))
        before = self.snapshot()
        for action in ("apply", "rollback"):
            result = self.run_manager(action)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SOURCE_MISMATCH", result.stderr)
            self.assertEqual(self.snapshot(), before)


if __name__ == "__main__":
    unittest.main()
