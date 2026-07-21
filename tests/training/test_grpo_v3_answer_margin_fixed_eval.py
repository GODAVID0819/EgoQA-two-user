from __future__ import annotations

import math
import unittest

from training.grpo_v3_answer_margin_fixed_eval import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CHECKPOINT_STEPS,
    EVAL_SEEDS,
    TEMPERATURE,
    analyze_fixed_eval,
)


def _rows(*, delta: float = 0.2) -> list[dict]:
    rows = []
    for step in CHECKPOINT_STEPS:
        for index, seed in enumerate(EVAL_SEEDS):
            base = -0.4 + index * 0.01
            row = {
                "checkpoint_step": step,
                "seed": seed,
                "reward": base + (delta if step == 40 else 0.0),
                "temperature": 0.5,
                "top1_hit": index < (25 if step == 0 else 27),
                "core_qa_extracted": index < (31 if step == 0 else 30),
            }
            if step == 0:
                row.update(source_job="gate2_14119442", checkpoint="checkpoint-1")
            else:
                row.update(source_mode="probe40", adapter_dir="/scratch/probe40/checkpoint-40")
            rows.append(row)
    return rows


def _training(**overrides: object) -> dict:
    value = {
        "run_status": "passed",
        "mode": "probe40",
        "trace_count": 160,
        "finite_reward_count": 160,
        "masked_reward_count": 0,
        "positive_variance_group_count": 32,
    }
    value.update(overrides)
    return value


class AnswerMarginFixedEvalTests(unittest.TestCase):
    def test_frozen_constants_and_passing_result(self) -> None:
        self.assertEqual(len(EVAL_SEEDS), 32)
        self.assertEqual(CHECKPOINT_STEPS, (0, 40))
        self.assertEqual(TEMPERATURE, 0.5)
        self.assertEqual(BOOTSTRAP_SEED, 20260721)
        self.assertEqual(BOOTSTRAP_REPLICATES, 10_000)
        summary = analyze_fixed_eval(_rows(), _training(), {"status": "passed"})
        self.assertEqual(summary["row_count"], 64)
        self.assertEqual(summary["pair_count"], 32)
        self.assertEqual(summary["experiment_conclusion"], "passed")
        self.assertTrue(all(summary["checks"].values()))
        self.assertGreater(summary["paired_bootstrap_95_ci"][0], 0)

    def test_complete_numeric_failure_is_not_converged(self) -> None:
        summary = analyze_fixed_eval(
            _rows(delta=-0.1), _training(positive_variance_group_count=31), {"status": "passed"}
        )
        self.assertEqual(summary["run_status"], "passed")
        self.assertEqual(summary["experiment_conclusion"], "not_converged")
        self.assertIn("step40_mean_strictly_higher", summary["failed_checks"])
        self.assertIn(
            "training_positive_variance_groups_at_least_80_percent",
            summary["failed_checks"],
        )

    def test_integrity_failures_are_invalid(self) -> None:
        cases = []
        missing = _rows()[:-1]
        cases.append((missing, _training(), {"status": "passed"}))
        bad_parent = _rows()
        bad_parent[0]["source_job"] = "gate3_14169924"
        cases.append((bad_parent, _training(), {"status": "passed"}))
        bad_training = _training(masked_reward_count=1)
        cases.append((_rows(), bad_training, {"status": "passed"}))
        cases.append((_rows(), _training(), {"status": "failed"}))
        for rows, training, reload in cases:
            with self.subTest(rows=len(rows), training=training, reload=reload):
                self.assertEqual(
                    analyze_fixed_eval(rows, training, reload)["experiment_conclusion"],
                    "invalid",
                )

    def test_duplicate_key_and_nonfinite_reward_are_invalid(self) -> None:
        duplicate = _rows()
        duplicate[-1] = dict(duplicate[0])
        self.assertIn(
            "duplicate_step_seed_key",
            analyze_fixed_eval(duplicate, _training(), {"status": "passed"})[
                "failed_integrity_checks"
            ],
        )
        for value in (math.nan, math.inf):
            rows = _rows()
            rows[0]["reward"] = value
            result = analyze_fixed_eval(rows, _training(), {"status": "passed"})
            self.assertEqual(result["experiment_conclusion"], "invalid")
            self.assertIn("nonfinite_fixed_eval_reward", result["failed_integrity_checks"])


if __name__ == "__main__":
    unittest.main()
