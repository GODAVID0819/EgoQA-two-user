from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.text_only_a_density.fixed_rollout_probe import (
    analyze_fixed_rollout_summary,
    fixed_rollout_spec,
)


class FixedRolloutProbeTests(unittest.TestCase):
    def test_fixed_rollout_spec_is_deterministic_and_has_positive_negative_pairs(self) -> None:
        spec = fixed_rollout_spec()
        self.assertEqual(spec["schema_version"], "text_only_a_density_fixed_rollout_spec_v1")
        self.assertEqual(spec["prompt_count"], 1)
        self.assertEqual(spec["completion_texts"], ["A", "B", "A", "B"])
        self.assertEqual(spec["rewards"], [1.0, -1.0, 1.0, -1.0])
        self.assertEqual(spec["advantages"], [1.0, -1.0, 1.0, -1.0])
        self.assertFalse(spec["uses_generation"])
        self.assertFalse(spec["uses_video"])
        self.assertFalse(spec["uses_reviewer_or_judge"])
        self.assertFalse(spec["parses_json"])

    def test_analyzer_passes_only_when_margin_and_checkpoint_reload_improve(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "initial_margin": -0.2,
                        "final_margin": 0.45,
                        "reloaded_margin": 0.45,
                        "steps": 10,
                        "nonzero_trainable_delta": True,
                        "all_grad_norms_finite": True,
                    }
                ),
                encoding="utf-8",
            )
            result = analyze_fixed_rollout_summary(path)
        self.assertEqual(result["status"], "passed")
        self.assertGreaterEqual(result["final_minus_initial_margin"], 0.5)
        self.assertEqual(result["failed_checks"], [])

    def test_analyzer_fails_flat_or_unreloaded_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "initial_margin": 0.0,
                        "final_margin": 0.1,
                        "reloaded_margin": -0.2,
                        "steps": 10,
                        "nonzero_trainable_delta": False,
                        "all_grad_norms_finite": True,
                    }
                ),
                encoding="utf-8",
            )
            result = analyze_fixed_rollout_summary(path)
        self.assertEqual(result["status"], "not_converged")
        self.assertIn("margin_improved_by_at_least_0_5", result["failed_checks"])
        self.assertIn("reload_preserves_final_margin", result["failed_checks"])
        self.assertIn("nonzero_trainable_delta", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
