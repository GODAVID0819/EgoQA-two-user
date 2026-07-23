from __future__ import annotations

import unittest

from training.grpo_v3.evaluation.greedy_compare import compare_runs


def _row(evidence_id: str, question_type: str, reward: float, groundedness: str) -> dict:
    return {
        "evidence_id": evidence_id,
        "question_type": question_type,
        "raw_completion": f"QA {evidence_id}",
        "reward": reward,
        "record": {
            "groundedness_status": groundedness,
            "combined_correct": reward > 0,
            "reward_components": {
                "groundedness": 1.0 if groundedness == "PASS" else -1.2,
                "combined_answerability": 1.0 if reward > 0 else -1.2,
                "format": 0.0,
            },
            "format_validation": {"status": "raw_valid"},
        },
    }


class GreedyCompareTests(unittest.TestCase):
    def test_strictly_pairs_rows_and_reports_component_deltas(self) -> None:
        gate2 = [_row("E1", "commonality", 0.5, "PASS"), _row("E2", "difference", -1.0, "FAIL")]
        gate3 = [_row("E2", "difference", 0.0, "PASS"), _row("E1", "commonality", 1.5, "PASS")]
        result = compare_runs({"gate2": gate2, "gate3_old": gate3}, baseline_label="gate2")

        self.assertEqual(result["paired_count"], 2)
        comparison = result["comparisons"]["gate3_old_vs_gate2"]
        self.assertEqual(comparison["wins"], 2)
        self.assertEqual(comparison["losses"], 0)
        self.assertAlmostEqual(comparison["reward_delta_mean"], 1.0)
        self.assertAlmostEqual(comparison["component_delta_means"]["groundedness"], 1.1)

    def test_rejects_misaligned_or_duplicate_eval_sets(self) -> None:
        with self.assertRaisesRegex(ValueError, "集合不一致"):
            compare_runs(
                {
                    "gate2": [_row("E1", "commonality", 0.0, "PASS")],
                    "gate3": [_row("E2", "commonality", 0.0, "PASS")],
                },
                baseline_label="gate2",
            )
        duplicate = [_row("E1", "commonality", 0.0, "PASS")] * 2
        with self.assertRaisesRegex(ValueError, "重复"):
            compare_runs({"gate2": duplicate}, baseline_label="gate2")


if __name__ == "__main__":
    unittest.main()
