from __future__ import annotations

import unittest
import uuid
from pathlib import Path

from training.grpo_v3.evaluation.policy_comparison import (
    _blind_rows,
    _summarize,
    _write_blind_html,
    extract_inference_completion,
)


class PolicyComparisonTests(unittest.TestCase):
    def test_extract_inference_completion_supports_swift_result_shapes(self) -> None:
        self.assertEqual(extract_inference_completion({"response": "direct"}), "direct")
        self.assertEqual(
            extract_inference_completion({
                "messages": [
                    {"role": "user", "content": "prompt"},
                    {"role": "assistant", "content": "answer"},
                ]
            }),
            "answer",
        )
        self.assertEqual(
            extract_inference_completion({
                "response": {
                    "choices": [{"message": {"content": "nested"}}]
                }
            }),
            "nested",
        )

    def test_summary_uses_paired_baseline_deltas(self) -> None:
        policies = ["baseline", "lr_1e-6", "lr_1e-5", "lr_3e-5"]
        rewards = {
            "baseline": [0.2, 0.4],
            "lr_1e-6": [0.3, 0.5],
            "lr_1e-5": [0.1, 0.4],
            "lr_3e-5": [0.6, 0.7],
        }
        rows = []
        for policy in policies:
            for index, reward in enumerate(rewards[policy]):
                rows.append({
                    "policy": policy,
                    "evidence_id": f"e{index}",
                    "reward": reward,
                    "reward_source": "score_only_ordinal_reviewer",
                    "expected_scores": {
                        "evidence_quality": 2.0,
                        "answerability": 2.0,
                        "qa_formality": 2.0,
                    },
                })
        summary = _summarize(
            rows,
            policy_order=policies,
            baseline_policy="baseline",
            bootstrap_seed=42,
        )
        self.assertAlmostEqual(
            summary["paired_vs_baseline"]["lr_1e-6"]["mean_reward_delta"],
            0.1,
        )
        self.assertEqual(summary["paired_vs_baseline"]["lr_1e-6"]["wins"], 2)
        self.assertEqual(summary["ranking_by_mean_reviewer_reward"][0], "lr_3e-5")

    def test_blind_artifacts_hide_policy_and_reward(self) -> None:
        policies = ["baseline", "lr_1e-6", "lr_1e-5", "lr_3e-5"]
        scored = [{
            "evidence_id": "e1",
            "decoding": "greedy",
            "policy": policy,
            "completion": "completion",
            "qa": {
                "question": "question",
                "options": ["a", "b", "c", "d", "e"],
                "correct": "A",
                "answer": "a",
            },
            "reward": 0.5,
            "reward_source": "score_only_ordinal_reviewer",
            "expected_scores": {
                "evidence_quality": 2.0,
                "answerability": 2.0,
                "qa_formality": 2.0,
            },
        } for policy in policies]
        dataset = [{
            "evidence_id": "e1",
            "required_users": ["A1", "A2"],
            "generator_image_paths": ["frame.jpg"],
            "reviewer_video_paths": ["a.mp4", "b.mp4"],
        }]
        blind, key = _blind_rows(
            scored,
            dataset_rows=dataset,
            decoding_order=["greedy"],
            policy_order=policies,
            seed=42,
        )
        self.assertNotIn("policy", blind[0]["candidates"][0])
        self.assertNotIn("reward", blind[0]["candidates"][0])
        self.assertEqual(
            {row["policy"] for row in key["rows"][0]["mapping"]}, set(policies)
        )
        output = Path("tmp") / f"policy_comparison_{uuid.uuid4().hex}.html"
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_blind_html(output, blind)
            text = output.read_text(encoding="utf-8")
            self.assertNotIn("lr_1e-6", text)
            self.assertIn("Download labels JSON", text)
        finally:
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
