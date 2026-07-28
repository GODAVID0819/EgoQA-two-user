from __future__ import annotations

import unittest

from training.grpo_v3.experiments.qa_cross_view_relation.convergence import analyze_trace


def row(group: int, candidate: int, reward: float, *, unrecoverable: bool = False):
    return {
        "reward_kind": "qa_cross_view_relation",
        "reward_call_index": group,
        "candidate_index": candidate,
        "failure_stage": None,
        "reward": reward,
        "record": {
            "reward_revision": "qa_cross_view_relation_v2",
            "reward_components": {"qa_cross_view_relation": reward},
            "deterministic": {"format_status": "unrecoverable" if unrecoverable else "raw_valid"},
        },
    }


class CrossViewRelationConvergenceTests(unittest.TestCase):
    def test_detects_positive_online_probe(self):
        rows = []
        for group in range(40):
            base = 0.2 + group * 0.01
            rows.extend(row(group, idx, base + idx * 0.01) for idx in range(4))
        result = analyze_trace(rows)
        self.assertEqual(result["status"], "passed")

    def test_fails_when_reward_declines_or_unrecoverable_increases(self):
        rows = []
        for group in range(40):
            base = 0.8 - group * 0.01
            rows.extend(row(group, idx, base + idx * 0.01, unrecoverable=(group >= 30 and idx == 0)) for idx in range(4))
        result = analyze_trace(rows)
        self.assertEqual(result["status"], "failed")
        self.assertIn("reward_slope_positive", result["failed_checks"])
        self.assertIn("unrecoverable_rate_not_increased", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
