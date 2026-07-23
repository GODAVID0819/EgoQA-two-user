from __future__ import annotations

import unittest

from training.grpo_v3.experiments.archived.formality.convergence import analyze_formality_convergence


def _rows(*, steps: int = 40, improving: bool = True) -> list[dict]:
    rows: list[dict] = []
    for step in range(steps):
        direction = step if improving else -step
        center = -0.4 + direction * 0.01
        for candidate, offset in enumerate((-0.03, -0.01, 0.01, 0.03)):
            reward = center + offset
            rows.append(
                {
                    "reward_kind": "qa_formality_confidence",
                    "reward_call_index": step,
                    "phase": "train",
                    "candidate_index": candidate,
                    "reward": reward,
                    "record": {
                        "masked": False,
                        "judge_called": True,
                        "reward_source": "judge_pass_fail_logprob_margin",
                        "qa_formality_status": "PASS" if reward >= 0 else "FAIL",
                        "reward_components": {"qa_formality_confidence": reward},
                        "judge_trace": {"qa_formality": {"parsed": {}}},
                    },
                }
            )
    return rows


class FormalityConvergenceTests(unittest.TestCase):
    def test_passes_forty_improving_diverse_groups(self) -> None:
        result = analyze_formality_convergence(_rows(), expected_steps=40)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["group_count"], 40)
        self.assertEqual(result["finite_reward_count"], 160)
        self.assertGreater(result["reward_delta"], 0.0)
        self.assertGreater(result["reward_slope"], 0.0)
        self.assertGreaterEqual(result["positive_std_ratio"], 0.8)

    def test_rejects_nonimproving_reward(self) -> None:
        result = analyze_formality_convergence(
            _rows(improving=False),
            expected_steps=40,
        )
        self.assertIn("last_window_reward_improved", result["failed_checks"])
        self.assertIn("reward_slope_positive", result["failed_checks"])

    def test_rejects_component_or_judge_contamination(self) -> None:
        rows = _rows()
        rows[0]["record"]["reward_components"]["groundedness"] = 1.0
        rows[1]["record"]["judge_trace"]["evidence_groundedness"] = {}
        result = analyze_formality_convergence(rows, expected_steps=40)

        self.assertIn("only_formality_reward_component", result["failed_checks"])
        self.assertIn("only_formality_judge_called", result["failed_checks"])

    def test_rejects_too_many_zero_variance_groups(self) -> None:
        rows = _rows()
        for row in rows:
            if row["reward_call_index"] < 12:
                row["reward"] = 0.0
                row["record"]["reward_components"] = {"qa_formality_confidence": 0.0}
        result = analyze_formality_convergence(rows, expected_steps=40)
        self.assertIn("positive_std_ratio_at_least_0_8", result["failed_checks"])

    def test_rejects_rising_unjudgeable_rate(self) -> None:
        rows = _rows()
        for row in rows:
            if row["reward_call_index"] >= 30 and row["candidate_index"] == 0:
                row["reward"] = -1.0
                row["record"].update(
                    {
                        "judge_called": False,
                        "reward_source": "deterministic_unjudgeable_floor",
                        "qa_formality_status": "FAIL",
                        "reward_components": {"qa_formality_confidence": -1.0},
                        "judge_trace": {},
                    }
                )
        result = analyze_formality_convergence(rows, expected_steps=40)
        self.assertIn("unjudgeable_rate_not_increased", result["failed_checks"])

    def test_rejects_bad_cardinality_mask_or_nonfinite_reward(self) -> None:
        rows = _rows()
        rows.pop()
        rows[0]["record"]["masked"] = True
        rows[1]["reward"] = float("nan")
        result = analyze_formality_convergence(rows, expected_steps=40)

        self.assertIn("all_groups_have_four_candidates", result["failed_checks"])
        self.assertIn("all_rewards_finite", result["failed_checks"])
        self.assertIn("infrastructure_mask_count_zero", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
