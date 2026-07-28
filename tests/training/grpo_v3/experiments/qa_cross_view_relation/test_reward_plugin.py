from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


def completion(question: str = "After I left the counter, where did the mug end up?") -> str:
    return json.dumps(
        {
            "question_type": "neutral",
            "question": question,
            "options": ["counter", "sink", "laptop table", "shelf", "being carried"],
            "correct": "C",
            "answer": "laptop table",
            "required_users": ["Jake", "Katrina"],
        }
    )


class FakeJudge:
    def __init__(self, *, fail: Exception | None = None) -> None:
        self.calls = []
        self.fail = fail

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        return {
            "candidate_scores": [
                {
                    "candidate_id": item.candidate_id,
                    "cross_view_relation_score": 2,
                    "semantic_naturalness_score": 2,
                    "internal_consistency_score": 2,
                    "anchor_tier": 2,
                    "pairwise_preferences": {
                        other.candidate_id: ("TIE" if other.candidate_id != item.candidate_id else "TIE")
                        for other in kwargs["candidates"]
                        if other.candidate_id != item.candidate_id
                    },
                    "reasons": {"summary": "fixture"},
                }
                for item in kwargs["candidates"]
            ]
        }


class CrossViewRelationPluginTests(unittest.TestCase):
    def kwargs(self):
        packet = json.dumps({"evidence_id": "E1", "required_users": ["Jake", "Katrina"]})
        return {
            "packet_json": [packet],
            "evidence_id": ["E1"],
            "question_type": ["neutral"],
            "generation_mode": ["baseline"],
            "global_step": [0],
        }

    def test_registers_group_orm_and_writes_four_trace_rows(self):
        from training.grpo_v3.runtime.reward_plugin import CrossViewRelationReward, orms

        fake = FakeJudge()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            reward = CrossViewRelationReward(trace_path=trace, judge_group_fn=fake)
            values = reward([completion(), completion(), completion(), completion()], **self.kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertIs(orms["egoqa_cross_view_relation_v2"], CrossViewRelationReward)
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(len(values), 4)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["reward_kind"] == "qa_cross_view_relation" for row in rows))
        self.assertTrue(all(row["record"]["reward_revision"] == "qa_cross_view_relation_v2" for row in rows))
        self.assertTrue(all(set(row["record"]["reward_components"]) == {"qa_cross_view_relation"} for row in rows))

    def test_reviewer_error_is_traced_then_reraised(self):
        from training.grpo_v3.runtime.reward_plugin import CrossViewRelationReward

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "trace.jsonl"
            reward = CrossViewRelationReward(trace_path=trace, judge_group_fn=FakeJudge(fail=TimeoutError("judge timeout")))
            with self.assertRaisesRegex(TimeoutError, "judge timeout"):
                reward([completion(), completion(), completion(), completion()], **self.kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["failure_stage"], "group_judge")
        self.assertEqual(rows[0]["record"]["infrastructure_error"]["type"], "TimeoutError")


if __name__ == "__main__":
    unittest.main()
