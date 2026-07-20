from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_formality_replay import replay_file, replay_trace


def _row(call_index: int, candidate_index: int, margin: float) -> dict:
    return {
        "reward_call_index": call_index,
        "candidate_index": candidate_index,
        "record": {
            "judge_trace": {
                "qa_formality": {
                    "parsed": {
                        "checks": {
                            "qa_formality": {
                                "status": "PASS" if margin >= 0 else "FAIL"
                            }
                        },
                        "choice_logit_signal": {
                            "available": True,
                            "choice_logprobs": {
                                "PASS": margin,
                                "FAIL": 0.0,
                            },
                        },
                    }
                }
            }
        },
    }


class FormalityReplayTests(unittest.TestCase):
    def test_replays_complete_groups_and_reports_contract(self) -> None:
        rows = [
            _row(call_index, candidate_index, float(candidate_index + call_index))
            for call_index in range(2)
            for candidate_index in range(4)
        ]
        report = replay_trace(rows, input_sha256="abc123")

        self.assertEqual(report["schema_version"], "grpo_v3_formality_replay_v1")
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["complete_group_count"], 2)
        self.assertEqual(report["finite_reward_count"], 8)
        self.assertEqual(report["positive_std_group_count"], 2)
        self.assertEqual(report["positive_std_ratio"], 1.0)
        self.assertEqual(report["reward_components"], ["qa_formality_confidence"])
        self.assertEqual(report["input_sha256"], "abc123")

    def test_fails_when_complete_groups_have_zero_variance(self) -> None:
        rows = [_row(0, candidate_index, 8.0) for candidate_index in range(4)]
        report = replay_trace(rows, input_sha256="same")

        self.assertEqual(report["status"], "failed")
        self.assertIn("positive_std_ratio_at_least_0_8", report["failed_checks"])
        self.assertEqual(report["positive_std_group_count"], 0)

    def test_missing_logprob_is_counted_and_excludes_incomplete_group(self) -> None:
        rows = [_row(0, candidate_index, float(candidate_index)) for candidate_index in range(4)]
        del rows[-1]["record"]["judge_trace"]["qa_formality"]["parsed"][
            "choice_logit_signal"
        ]["choice_logprobs"]["FAIL"]
        report = replay_trace(rows, input_sha256="missing")

        self.assertEqual(report["missing_logprob_count"], 1)
        self.assertEqual(report["finite_reward_count"], 3)
        self.assertEqual(report["complete_group_count"], 0)
        self.assertEqual(report["status"], "failed")

    def test_replay_file_hashes_exact_input_and_writes_json(self) -> None:
        rows = [_row(0, candidate_index, float(candidate_index)) for candidate_index in range(4)]
        payload = "\n".join(json.dumps(row) for row in rows) + "\n"
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            output = Path(tmp) / "report.json"
            trace.write_bytes(payload.encode("utf-8"))
            report = replay_file(trace, output)
            written = json.loads(output.read_text(encoding="utf-8"))

        expected_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.assertEqual(report["input_sha256"], expected_hash)
        self.assertEqual(written, report)


if __name__ == "__main__":
    unittest.main()
