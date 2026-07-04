from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from grpo_ready.replay import run_replay


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "outputs" / "qa_mcq.intermediate.jsonl"


class ReplayTests(unittest.TestCase):
    def test_replay_writes_complete_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            summary = run_replay(INPUT_PATH, output_dir)
            expected = {
                "reward_replay_results.jsonl",
                "reward_replay_results.csv",
                "reward_replay_summary.json",
                "reward_replay_summary.md",
                "run_manifest.json",
            }
            self.assertTrue(expected.issubset({path.name for path in output_dir.iterdir()}))
            jsonl_rows = [
                json.loads(line)
                for line in (output_dir / "reward_replay_results.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            with (output_dir / "reward_replay_results.csv").open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))

        self.assertEqual(len(jsonl_rows), 121)
        self.assertEqual(len(csv_rows), 121)
        self.assertEqual(summary["attempt_count"], 121)
        self.assertEqual(summary["accepted_count"], 27)
        self.assertEqual(summary["rejected_count"], 94)
        self.assertLessEqual(len(summary["contradiction_cases"]), 5)
        self.assertIn("complete_reward_coverage", summary)
        self.assertIn("accepted", summary["by_historical_label"])
        self.assertIn("rejected", summary["by_historical_label"])

    def test_replay_manifest_records_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            run_replay(INPUT_PATH, output_dir)
            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["reward_version"], "v0")
        self.assertEqual(len(manifest["input_sha256"]), 64)
        self.assertEqual(manifest["input_path"], str(INPUT_PATH.resolve()))
        self.assertIn("git_commit", manifest)


if __name__ == "__main__":
    unittest.main()
