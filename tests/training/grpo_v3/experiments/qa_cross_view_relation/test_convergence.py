from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.qa_cross_view_relation.analyze_probe import (
    analyze,
    analyze_question_patterns,
)
from training.grpo_v3.experiments.qa_cross_view_relation.convergence import analyze_trace


def row(
    group: int,
    candidate: int,
    reward: float,
    *,
    unrecoverable: bool = False,
    reward_revision: str = "qa_cross_view_relation_v2",
):
    return {
        "reward_kind": "qa_cross_view_relation",
        "reward_call_index": group,
        "candidate_index": candidate,
        "failure_stage": None,
        "reward": reward,
        "record": {
            "reward_revision": reward_revision,
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

    def test_accepts_v3_trace_when_expected_revision_is_v3(self):
        rows = []
        for group in range(40):
            base = 0.2 + group * 0.01
            rows.extend(
                row(
                    group,
                    candidate,
                    base + candidate * 0.01,
                    reward_revision="qa_cross_view_relation_v3",
                )
                for candidate in range(4)
            )
        result = analyze_trace(
            rows,
            expected_reward_revision="qa_cross_view_relation_v3",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["reward_revision"], "qa_cross_view_relation_v3")

    def test_reports_template_concentration_and_per_evidence_coverage(self):
        rows = [
            {
                "evidence_id": "E1",
                "question_type": "commonality",
                "phase": "train",
                "reward": 0.95,
                "record": {
                    "deterministic": {
                        "qa": {"question": "After I left the counter, where did the mug end up?"}
                    }
                },
            },
            {
                "evidence_id": "E2",
                "question_type": "difference",
                "phase": "heldout",
                "reward": 0.30,
                "record": {
                    "deterministic": {
                        "qa": {"question": "After I left the table, where did the plate end up?"}
                    }
                },
            },
        ]
        result = analyze_question_patterns(rows)
        self.assertEqual(result["exact_question_unique_rate"], 1.0)
        self.assertEqual(result["normalized_template_unique_rate"], 0.5)
        self.assertEqual(result["top_template_fraction"], 1.0)
        self.assertEqual(set(result["per_evidence_metrics"]), {"E1", "E2"})
        self.assertIn("mug", result["object_word_distribution"])
        self.assertEqual(result["specified_template_fraction"], 0.0)
        self.assertEqual(result["high_reward"]["question_count"], 1)
        self.assertEqual(set(result["by_phase"]), {"train", "heldout"})

    def test_probe_summary_uses_dynamic_last_windows_for_probe120(self):
        rows = []
        for group in range(120):
            for candidate in range(4):
                reward = group / 120 + candidate * 0.001
                rows.append(
                    {
                        "reward_kind": "qa_cross_view_relation",
                        "reward_call_index": group,
                        "candidate_id": f"candidate_{candidate}",
                        "evidence_id": f"E{group % 20}",
                        "question_type": "commonality",
                        "reward": reward,
                        "record": {
                            "reward_source": "qa_cross_view_relation_v3",
                            "format_status": "raw_valid",
                            "deterministic": {
                                "blocking_errors": [],
                                "qa": {"question": f"Where is object {group}?"},
                            },
                            "judge_trace": {"candidate_scores": []},
                        },
                    }
                )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "reward_trace.jsonl").write_text(
                "\n".join(json.dumps(item) for item in rows) + "\n",
                encoding="utf-8",
            )
            result = analyze(root)
        self.assertEqual(result["judge_only_candidate_mean"]["last10_n"], 40)
        self.assertAlmostEqual(
            result["judge_only_candidate_mean"]["last10_mean"],
            sum(item["reward"] for item in rows if item["reward_call_index"] >= 110) / 40,
        )


if __name__ == "__main__":
    unittest.main()
