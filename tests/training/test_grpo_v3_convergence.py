from __future__ import annotations

import math
import unittest

from training.grpo_v3_convergence import analyze_convergence


def _group(call_index: int, phase: str, evidence_id: str, mean: float, spread: bool = True) -> list[dict]:
    offsets = [-0.3, -0.1, 0.1, 0.3] if spread else [0.0] * 4
    return [
        {
            "reward_call_index": call_index,
            "phase": phase,
            "evidence_id": evidence_id,
            "candidate_index": index,
            "reward": mean + offsets[index],
            "record": {
                "masked": False,
                "eligible_for_grpo": True,
                "format_validation": {"status": "raw_valid"},
                "reward_components": {
                    "groundedness": mean,
                    "combined_answerability": 0.5,
                    "qa_formality": 0.5,
                },
            },
        }
        for index in range(4)
    ]


class ConvergenceTests(unittest.TestCase):
    def test_gate3_passes_with_finite_improving_diverse_groups(self) -> None:
        rows = []
        for index in range(20):
            rows.extend(_group(index, "train", "E1", 0.1 + index * 0.05, spread=index < 17))
        result = analyze_convergence(rows, gate=3, trainer_state={"global_step": 20})
        self.assertEqual(result["status"], "passed")
        self.assertGreater(result["train"]["late_reward_mean"], result["train"]["early_reward_mean"])
        self.assertEqual(result["train"]["group_count"], 20)
        self.assertGreaterEqual(result["train"]["positive_std_ratio"], 0.8)

    def test_gate3_reports_group_series_and_early_late_format_windows(self) -> None:
        rows = []
        for index in range(20):
            group = _group(index, "train", "E1", 0.1 + index * 0.05)
            status = "raw_valid" if index < 5 else "repaired"
            penalty = 0.0 if status == "raw_valid" else -0.5
            for row in group:
                row["record"]["format_validation"]["status"] = status
                row["record"]["reward_components"]["format_penalty"] = penalty
            rows.extend(group)

        result = analyze_convergence(
            rows,
            gate=3,
            trainer_state={"global_step": 20, "log_history": [{"completions/mean_length": 640.0}]},
        )

        train = result["train"]
        self.assertEqual(len(train["group_reward_series"]), 20)
        self.assertEqual(train["group_reward_series"][0]["format_counts"], {"raw_valid": 4})
        self.assertEqual(train["early_format_counts"], {"raw_valid": 20})
        self.assertEqual(train["late_format_counts"], {"repaired": 20})
        self.assertEqual(train["early_format_rates"]["raw_valid"], 1.0)
        self.assertEqual(train["late_format_rates"]["repaired"], 1.0)
        self.assertAlmostEqual(train["component_delta"]["format_penalty"], -0.5)
        self.assertEqual(result["trainer_metrics"]["completions/mean_length"], [640.0])

    def test_gate3_fails_nonfinite_masked_bad_cardinality_or_no_improvement(self) -> None:
        base = []
        for index in range(20):
            base.extend(_group(index, "train", "E1", 1.0 - index * 0.01))
        base[0]["reward"] = math.inf
        base[1]["record"]["masked"] = True
        base.pop()
        result = analyze_convergence(base, gate=3, trainer_state={"global_step": 19})
        self.assertEqual(result["status"], "failed")
        self.assertIn("all_rewards_finite", result["failed_checks"])
        self.assertIn("masked_count_zero", result["failed_checks"])
        self.assertIn("all_groups_have_four_candidates", result["failed_checks"])
        self.assertIn("train_reward_improved", result["failed_checks"])

    def test_gate4_passes_with_stable_holdout_and_target_component_gain(self) -> None:
        rows = []
        call = 0
        eval_ids = [f"V{i}" for i in range(10)]
        for evidence_id in eval_ids:
            rows.extend(_group(call, "eval", evidence_id, 1.0)); call += 1
        for index in range(40):
            rows.extend(_group(call, "train", f"T{index}", 0.2 + index * 0.03)); call += 1
        for evidence_id in eval_ids:
            rows.extend(_group(call, "eval", evidence_id, 0.95)); call += 1
        result = analyze_convergence(
            rows,
            gate=4,
            trainer_state={"global_step": 40, "log_history": [{"grad_norm": 1.2, "clip_ratio": 0.1, "completion_length": 128}]},
            expected_eval_ids=eval_ids,
        )
        self.assertEqual(result["status"], "passed")
        self.assertAlmostEqual(result["eval"]["reward_delta"], -0.05)
        self.assertTrue(result["checks"]["target_component_improved"])
        self.assertEqual(result["trainer_metrics"]["grad_norm"], [1.2])
        self.assertEqual(result["trainer_metrics"]["completion_length"], [128.0])

    def test_gate4_rejects_eval_leakage_or_excessive_holdout_drop(self) -> None:
        rows = []
        call = 0
        for index in range(10):
            rows.extend(_group(call, "eval", f"V{index}", 1.0)); call += 1
        for index in range(40):
            rows.extend(_group(call, "train", f"T{index}", 0.2 + index * 0.01)); call += 1
        for index in range(10):
            rows.extend(_group(call, "eval", f"V{index}", 0.7)); call += 1
        result = analyze_convergence(
            rows,
            gate=4,
            trainer_state={"global_step": 40, "log_history": [{"grad_norm": math.inf}]},
            expected_eval_ids=[f"V{i}" for i in range(9)] + ["WRONG"],
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("eval_ids_match_manifest", result["failed_checks"])
        self.assertIn("holdout_drop_within_limit", result["failed_checks"])
        self.assertIn("trainer_logged_metrics_finite", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
