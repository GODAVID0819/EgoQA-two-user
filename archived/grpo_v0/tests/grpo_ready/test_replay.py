from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from grpo_ready.replay import run_replay


def fixture_packet() -> dict:
    attempts = []
    for index, accepted in enumerate((False, True), start=1):
        raw = json.dumps(
            {
                "question": f"question-{index}",
                "options": ["A", "B", "C", "D"],
                "correct": "A",
                "required_users": ["P1", "P2"],
            }
        )
        attempts.append(
            {
                "attempt": index,
                "evidence_id": "E1",
                "question_type": "difference",
                "generation_mode": "baseline",
                "feedback_in": "" if index == 1 else "repair grounding",
                "media": {
                    "image_paths": ["generator.jpg"],
                    "video_paths": ["generator.mp4"],
                    "full_image_paths": ["evaluator.jpg"],
                    "full_video_paths": ["evaluator.mp4"],
                },
                "generation": {
                    "prompt": "fixed prompt",
                    "raw_output": raw,
                    "parsed_qa": {
                        "question": f"question-{index}",
                        "options": ["A", "B", "C", "D"],
                        "correct": "A",
                        "required_users": ["P1", "P2"],
                    },
                },
                "judge": {
                    "qa_formality": {"parsed": {"status": "PASS"}},
                    "evidence_groundedness": {"parsed": {"status": "PASS"}},
                },
                "answerability": {
                    "evaluations": [
                        {"condition_type": "combined_all_users", "users": ["P1", "P2"], "choice": "A"},
                        {"condition_type": "single_user", "users": ["P1"], "choice": "B"},
                        {"condition_type": "single_user", "users": ["P2"], "choice": "B"},
                    ]
                },
                "result": {"accepted": accepted, "reason": ""},
            }
        )
    return {
        "evidence_id": "E1",
        "status": "accepted",
        "question_type": "difference",
        "generation_mode": "baseline",
        "attempts": attempts,
    }


def write_fixture_jsonl(path: Path) -> None:
    path.write_text(json.dumps(fixture_packet(), ensure_ascii=False) + "\n", encoding="utf-8")


class ReplayTests(unittest.TestCase):
    def test_replay_writes_complete_output_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "attempts.jsonl"
            output_dir = root / "out"
            write_fixture_jsonl(input_path)
            summary = run_replay(input_path, output_dir)
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

        self.assertEqual(len(jsonl_rows), 2)
        self.assertEqual(len(csv_rows), 2)
        self.assertEqual(summary["attempt_count"], 2)
        self.assertEqual(summary["accepted_count"], 1)
        self.assertEqual(summary["rejected_count"], 1)
        self.assertLessEqual(len(summary["contradiction_cases"]), 5)
        self.assertIn("complete_reward_coverage", summary)
        self.assertIn("accepted", summary["by_historical_label"])
        self.assertIn("rejected", summary["by_historical_label"])

    def test_replay_manifest_records_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "attempts.jsonl"
            output_dir = root / "out"
            write_fixture_jsonl(input_path)
            run_replay(input_path, output_dir)
            manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["reward_version"], "v0")
        self.assertEqual(len(manifest["input_sha256"]), 64)
        self.assertEqual(manifest["input_path"], str(input_path.resolve()))
        self.assertIn("git_commit", manifest)


if __name__ == "__main__":
    unittest.main()
