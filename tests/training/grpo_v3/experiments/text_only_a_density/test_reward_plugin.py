from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.text_only_a_density.reward_plugin import (
    TextOnlyADensityReward,
    orms,
)


class DensityRewardPluginTests(unittest.TestCase):
    def test_registers_new_orm_without_replacing_existing_orms(self) -> None:
        self.assertIs(orms["egoqa_text_only_a_density"], TextOnlyADensityReward)
        self.assertIn("egoqa_gate1_controlled", orms)
        self.assertIn("egoqa_repo_native_judge", orms)
        self.assertIn("egoqa_combined_video_answer_margin", orms)

    def test_scores_raw_completions_and_writes_exact_trace_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            reward = TextOnlyADensityReward(trace_path=path)
            values = reward(
                ["AAAA", "AABB", "BBBB", "lowercase ab"],
                trial_id=["train-00"] * 4,
                phase=["train"] * 4,
                candidate_index=[0, 1, 2, 3],
            )
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(values, [1.0, 0.0, -1.0, -1.0])
        self.assertEqual(len(rows), 4)
        required = {
            "reward_kind", "reward_revision", "phase", "reward_call_index",
            "candidate_index", "trial_id", "completion", "completion_sha256",
            "n_A", "n_B", "n_valid", "non_ab_character_count", "reward",
            "formal_result",
        }
        self.assertTrue(all(set(row) == required for row in rows))
        self.assertTrue(all(row["formal_result"] is False for row in rows))

    def test_rejects_group_or_metadata_misalignment(self) -> None:
        reward = TextOnlyADensityReward()
        with self.assertRaisesRegex(ValueError, "4"):
            reward(["A"], trial_id=["train-00"], phase=["train"], candidate_index=[0])
        with self.assertRaisesRegex(ValueError, "trial_id"):
            reward(
                ["A"] * 4,
                trial_id=["train-00", "train-01"],
                phase=["train"] * 4,
                candidate_index=[0, 1, 2, 3],
            )
        with self.assertRaisesRegex(ValueError, "candidate"):
            reward(
                ["A"] * 4,
                trial_id=["train-00"] * 4,
                phase=["train"] * 4,
                candidate_index=[0, 1, 1, 3],
            )


if __name__ == "__main__":
    unittest.main()
