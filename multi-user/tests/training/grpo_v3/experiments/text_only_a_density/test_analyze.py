from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.text_only_a_density.analyze import analyze_trace


def write_trace(path: Path, means: list[float]) -> None:
    rows = []
    for group, mean in enumerate(means):
        values = [mean - 0.1, mean, mean + 0.1, mean]
        for candidate, reward in enumerate(values):
            rows.append(
                {
                    "reward_kind": "text_only_a_density",
                    "reward_revision": "text_only_a_density_v1",
                    "reward_call_index": group,
                    "candidate_index": candidate,
                    "trial_id": f"train-{group:02d}",
                    "reward": reward,
                    "completion": "AB",
                    "n_A": 1,
                    "n_B": 1,
                    "n_valid": 2,
                    "formal_result": False,
                }
            )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


class QuickAnalyzeTests(unittest.TestCase):
    def test_positive_trend_passes_lightweight_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward_trace.jsonl"
            write_trace(trace, [-0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5])
            result = analyze_trace(trace, expected_steps=10)
        self.assertEqual(result["status"], "passed")
        self.assertGreater(result["late_mean_minus_early_mean"], 0)

    def test_complete_but_flat_trace_is_not_converged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward_trace.jsonl"
            write_trace(trace, [0.0] * 10)
            result = analyze_trace(trace, expected_steps=10)
        self.assertEqual(result["status"], "not_converged")


if __name__ == "__main__":
    unittest.main()
