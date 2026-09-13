from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from training.grpo_v3.experiments.archived.formality import reward_plugin as plugin
from training.grpo_v3.experiments.archived.formality.reward_plugin import (
    FormalityConfidenceReward,
)


class FormalityPluginTests(unittest.TestCase):
    def test_registers_new_orm_without_replacing_old_orms(self) -> None:
        self.assertIn("egoqa_gate1_controlled", plugin.orms)
        self.assertIn("egoqa_repo_native_judge", plugin.orms)
        self.assertIn("egoqa_qa_formality_confidence", plugin.orms)

    def test_aligns_metadata_returns_rewards_and_writes_trace(self) -> None:
        expected = [-0.5, 0.0, 0.5, 1.0]

        def score_fn(**kwargs: Any) -> dict[str, Any]:
            value = expected[int(kwargs["candidate_index"])]
            return {
                "reward": value,
                "record": {
                    "masked": False,
                    "eligible_for_grpo": True,
                    "reward_components": {"qa_formality_confidence": value},
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            reward = FormalityConfidenceReward(trace_path=trace, score_fn=score_fn)
            values = reward(
                ["a", "b", "c", "d"],
                packet_json=[json.dumps({"evidence_id": "E1"})] * 4,
                evidence_id=["E1"] * 4,
                question_type=["difference"] * 4,
                generation_mode=["default"] * 4,
            )
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(values, expected)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["reward_kind"] == "qa_formality_confidence" for row in rows))
        self.assertEqual({row["reward_call_index"] for row in rows}, {0})
        self.assertEqual([row["candidate_index"] for row in rows], [0, 1, 2, 3])

    def test_scorer_exception_is_traced_then_propagated(self) -> None:
        def score_fn(**kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("reviewer unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            reward = FormalityConfidenceReward(trace_path=trace, score_fn=score_fn)
            with self.assertRaisesRegex(RuntimeError, "reviewer unavailable"):
                reward(
                    ["a", "b", "c", "d"],
                    packet_json=[json.dumps({"evidence_id": "E1"})] * 4,
                    evidence_id=["E1"] * 4,
                    question_type=["difference"] * 4,
                    generation_mode=["default"] * 4,
                )
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        error = rows[0]["record"]["infrastructure_error"]
        self.assertEqual(error["type"], "RuntimeError")
        self.assertEqual(error["message"], "reviewer unavailable")

    def test_nonfinite_reward_is_traced_then_rejected(self) -> None:
        def score_fn(**kwargs: Any) -> dict[str, Any]:
            return {"reward": float("nan"), "record": {"masked": False}}

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            reward = FormalityConfidenceReward(trace_path=trace, score_fn=score_fn)
            with self.assertRaisesRegex(ValueError, "非有限"):
                reward(
                    ["a", "b", "c", "d"],
                    packet_json=[json.dumps({"evidence_id": "E1"})] * 4,
                    evidence_id=["E1"] * 4,
                    question_type=["difference"] * 4,
                    generation_mode=["default"] * 4,
                )
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(
            rows[0]["record"]["infrastructure_error"]["type"],
            "NonFiniteRewardError",
        )


if __name__ == "__main__":
    unittest.main()
