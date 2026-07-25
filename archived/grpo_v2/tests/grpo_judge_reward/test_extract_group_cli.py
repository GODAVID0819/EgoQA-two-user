import json
import tempfile
import unittest
from pathlib import Path

from grpo_judge_reward.cli import run_score
from grpo_judge_reward.extract import iter_intermediate_attempt_inputs
from grpo_judge_reward.group import compute_group_records


def attempt(choice="B", index=1):
    return {
        "evidence_id": "e1",
        "qa_id": f"qa{index}",
        "attempt": index,
        "generation": {
            "parsed_qa": {
                "qa_id": f"qa{index}",
                "correct": "B",
                "required_users": ["Alice", "Bob"],
            }
        },
        "judge": {
            "merged": {
                "checks": {
                    "qa_formality": {
                        "status": "PASS",
                        "semantic_subchecks": {
                            "other_person_activity_query": {"status": "PASS"}
                        },
                        "schema_branch": {"status": "PASS", "errors": []},
                    },
                    "evidence_groundedness": {"status": "PASS"},
                },
                "feedback_to_generator": "",
                "gate": {"passed": True},
            }
        },
        "answerability": {
            "evaluations": [
                {"condition_type": "single_user", "condition_id": "single_user::Alice", "users": ["Alice"], "choice": "insufficient"},
                {"condition_type": "combined_all_users", "condition_id": "combined_all_users::Alice+Bob", "users": ["Alice", "Bob"], "choice": choice},
            ],
            "gate": {"passed": choice == "B", "speaker_user": "Alice", "evidence_provider_user": "Bob"},
        },
        "result": {"accepted": choice == "B"},
    }


class ExtractGroupCliTests(unittest.TestCase):
    def test_extracts_attempt_inputs_from_intermediate_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "intermediate.jsonl"
            path.write_text(json.dumps({"evidence_id": "e1", "status": "accepted", "attempts": [attempt()]}) + "\n", encoding="utf-8")

            rows = list(iter_intermediate_attempt_inputs(path))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["qa"]["correct"], "B")
        self.assertEqual(rows[0]["group_id"], "e1")

    def test_missing_review_signals_are_masked_by_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "intermediate.jsonl"
            out = Path(tmp) / "out"
            source.write_text(json.dumps({"evidence_id": "e1", "attempts": [{"attempt": 1}]}) + "\n", encoding="utf-8")

            summary = run_score(source, "intermediate", out)
            records = [json.loads(line) for line in (out / "judge_reward_records.jsonl").read_text(encoding="utf-8").splitlines()]

        self.assertEqual(summary["masked_candidates"], 1)
        self.assertEqual(records[0]["mask_reason"], "missing_review_signals")

    def test_group_advantages_skip_too_few_and_zero_variance(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "intermediate.jsonl"
            out = Path(tmp) / "out"
            source.write_text(
                json.dumps({"evidence_id": "e1", "attempts": [attempt(index=1), attempt(index=2)]}) + "\n"
                + json.dumps({"evidence_id": "e2", "attempts": [attempt(choice="A", index=1)]}) + "\n",
                encoding="utf-8",
            )
            run_score(source, "intermediate", out)
            records = [json.loads(line) for line in (out / "judge_reward_records.jsonl").read_text(encoding="utf-8").splitlines()]

        groups = compute_group_records(records)
        by_group = {row.group_id: row for row in groups}
        self.assertFalse(by_group["e1"].trainer_eligible)
        self.assertEqual(by_group["e1"].skip_reason, "zero_reward_variance")
        self.assertFalse(by_group["e2"].trainer_eligible)
        self.assertEqual(by_group["e2"].skip_reason, "too_few_valid_candidates")

    def test_cli_writes_summary_and_case_studies(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "intermediate.jsonl"
            out = Path(tmp) / "out"
            source.write_text(json.dumps({"evidence_id": "e1", "attempts": [attempt(), attempt(choice="A", index=2)]}) + "\n", encoding="utf-8")

            summary = run_score(source, "intermediate", out)
            self.assertIn("raw_candidates", summary)
            self.assertTrue((out / "judge_reward_summary.json").is_file())
            self.assertTrue((out / "judge_case_studies.md").is_file())


if __name__ == "__main__":
    unittest.main()
