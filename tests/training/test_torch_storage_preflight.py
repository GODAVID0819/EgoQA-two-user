from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.torch_storage_preflight import validate_storage_environment


class TorchStoragePreflightTests(unittest.TestCase):
    def test_accepts_writable_paths_under_allowed_root_without_deleting_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scratch" / "user" / "runtime"
            home = root / "home"
            home.mkdir(parents=True)
            sentinel = home / "keep.txt"
            sentinel.write_text("keep", encoding="utf-8")

            result = validate_storage_environment(
                allowed_root=root,
                environ={"HOME": str(home), "TMPDIR": str(root / "tmp")},
                required_variables=("HOME", "TMPDIR"),
            )

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["failed_checks"], [])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")
            self.assertEqual(len(result["paths"]), 2)
            for entry in result["paths"]:
                self.assertTrue(entry["absolute"])
                self.assertTrue(entry["within_allowed_root"])
                self.assertTrue(entry["writable"])
                self.assertGreater(entry["filesystem_total_bytes"], 0)
                self.assertGreaterEqual(entry["filesystem_free_bytes"], 0)

    def test_rejects_missing_relative_and_outside_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "scratch" / "user" / "runtime"
            outside = base / "home" / "user" / ".cache"
            result = validate_storage_environment(
                allowed_root=root,
                environ={"HOME": str(outside), "TMPDIR": "relative/tmp"},
                required_variables=("HOME", "TMPDIR", "HF_HOME"),
            )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failed_checks"], ["HOME", "TMPDIR", "HF_HOME"])
            by_name = {entry["variable"]: entry for entry in result["paths"]}
            self.assertFalse(by_name["HOME"]["within_allowed_root"])
            self.assertFalse(by_name["TMPDIR"]["absolute"])
            self.assertEqual(by_name["HF_HOME"]["error"], "environment variable is unset")
            self.assertFalse(outside.exists())

    def test_reports_write_probe_failure_without_removing_blocking_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scratch" / "user" / "runtime"
            root.mkdir(parents=True)
            blocker = root / "blocked"
            blocker.write_text("do not delete", encoding="utf-8")

            result = validate_storage_environment(
                allowed_root=root,
                environ={"TMPDIR": str(blocker / "tmp")},
                required_variables=("TMPDIR",),
            )

            self.assertEqual(result["status"], "failed")
            self.assertEqual(result["failed_checks"], ["TMPDIR"])
            self.assertFalse(result["paths"][0]["writable"])
            self.assertIn("FileExistsError", result["paths"][0]["error"])
            self.assertEqual(blocker.read_text(encoding="utf-8"), "do not delete")


if __name__ == "__main__":
    unittest.main()
