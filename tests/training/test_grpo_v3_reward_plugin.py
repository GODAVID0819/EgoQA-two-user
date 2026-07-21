from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.grpo_v3_reward_plugin import ControlledGateReward, RepoNativeJudgeReward


class StubAnswerMarginScoreFn:
    def __init__(self, *, failure: Exception | None = None) -> None:
        self.calls = []
        self.failure = failure

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return {"reward": 0.25, "record": {"masked": False, "eligible_for_grpo": True}}


class ControlledRewardTests(unittest.TestCase):
    def test_returns_four_finite_nonconstant_rewards_and_traces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward.jsonl"
            reward = ControlledGateReward(trace_path=trace)
            values = reward(
                ["a", "b", "c", "d"],
                evidence_id=["E1"] * 4,
            )
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(values), 4)
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertGreater(len(set(values)), 1)
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["reward_kind"] == "gate1_controlled" for row in rows))


class RepoNativeRewardTests(unittest.TestCase):
    def _kwargs(self) -> dict:
        packet = json.dumps({"evidence_id": "E1", "clips": []})
        return {
            "packet_json": [packet],
            "evidence_id": ["E1"],
            "question_type": ["commonality"],
            "generation_mode": ["baseline"],
        }

    def test_aligns_metadata_and_writes_auditable_trace(self) -> None:
        calls = []

        def scorer(**kwargs):
            calls.append(kwargs)
            return {"reward": float(kwargs["candidate_index"]), "record": {"eligible_for_grpo": True}}

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(trace_path=trace, score_fn=scorer)
            values = reward(["a", "b", "c", "d"], **self._kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(values, [0.0, 1.0, 2.0, 3.0])
        self.assertEqual([call["candidate_index"] for call in calls], [0, 1, 2, 3])
        self.assertTrue(all(call["evidence_id"] == "E1" for call in calls))
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["reward_kind"] == "repo_native_judge" for row in rows))
        self.assertEqual([row["completion_length_chars"] for row in rows], [1, 1, 1, 1])

    def test_records_monotonic_call_index_and_train_eval_phase(self) -> None:
        scorer = lambda **kwargs: {"reward": float(kwargs["candidate_index"]), "record": {"masked": False}}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"EGOQA_EVAL_EVIDENCE_IDS": "E2"}, clear=False
        ):
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(trace_path=trace, score_fn=scorer)
            reward(["a", "b", "c", "d"], **self._kwargs())
            eval_kwargs = self._kwargs()
            eval_kwargs["evidence_id"] = ["E2"]
            reward(["e", "f", "g", "h"], **eval_kwargs)
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["reward_call_index"] for row in rows], [0] * 4 + [1] * 4)
        self.assertEqual([row["phase"] for row in rows], ["train"] * 4 + ["eval"] * 4)

    def test_mixed_train_eval_call_is_traced_then_rejected(self) -> None:
        kwargs = self._kwargs()
        kwargs["evidence_id"] = ["E1", "E1", "E2", "E2"]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"EGOQA_EVAL_EVIDENCE_IDS": "E2"}, clear=False
        ):
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(
                trace_path=trace,
                score_fn=lambda **_kwargs: {"reward": 1.0, "record": {"masked": False}},
            )
            with self.assertRaisesRegex(ValueError, "train/eval"):
                reward(["a", "b", "c", "d"], **kwargs)
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertTrue(all(row["record"]["infrastructure_error"]["type"] == "MixedPhaseGroupError" for row in rows))

    def test_masked_completion_is_traced_then_aborts_whole_group(self) -> None:
        def scorer(**kwargs):
            if kwargs["candidate_index"] == 1:
                return {"reward": None, "record": {"masked": True, "mask_reason": "schema_fail"}}
            return {"reward": 1.0, "record": {"eligible_for_grpo": True}}

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(trace_path=trace, score_fn=scorer)
            with self.assertRaisesRegex(RuntimeError, "masked"):
                reward(["a", "b", "c", "d"], **self._kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual(rows[1]["record"]["mask_reason"], "schema_fail")

    def test_unrecoverable_format_reward_stays_finite_and_does_not_abort_group(self) -> None:
        def scorer(**kwargs):
            index = kwargs["candidate_index"]
            if index == 2:
                return {
                    "reward": -3.0,
                    "record": {
                        "masked": False,
                        "eligible_for_grpo": True,
                        "format_validation": {"status": "unrecoverable"},
                    },
                }
            return {"reward": [0.5, -1.9, 0.0, -1.9][index], "record": {"masked": False}}

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(trace_path=trace, score_fn=scorer)
            values = reward(["a", "b", "bad", "d"], **self._kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(values, [0.5, -1.9, -3.0, -1.9])
        self.assertTrue(all(math.isfinite(value) for value in values))
        self.assertFalse(rows[2]["record"]["masked"])
        self.assertEqual(rows[2]["record"]["format_validation"]["status"], "unrecoverable")

    def test_scorer_exception_is_traced_before_it_propagates(self) -> None:
        def scorer(**kwargs):
            if kwargs["candidate_index"] == 1:
                raise TimeoutError("reviewer timeout")
            return {"reward": 1.0, "record": {"masked": False}}

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(trace_path=trace, score_fn=scorer)
            with self.assertRaisesRegex(TimeoutError, "reviewer timeout"):
                reward(["a", "b", "c", "d"], **self._kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]["record"]["infrastructure_error"]["type"], "TimeoutError")
        self.assertEqual(rows[1]["record"]["infrastructure_error"]["message"], "reviewer timeout")

    def test_nonfinite_reward_is_traced_before_abort(self) -> None:
        def scorer(**kwargs):
            return {"reward": math.inf, "record": {"masked": False}}

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(trace_path=trace, score_fn=scorer)
            with self.assertRaisesRegex(ValueError, "非有限"):
                reward(["a", "b", "c", "d"], **self._kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["record"]["infrastructure_error"]["type"], "NonFiniteRewardError")

    def test_invalid_packet_json_is_traced_before_abort(self) -> None:
        kwargs = self._kwargs()
        kwargs["packet_json"] = ["{not-json"]

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "reward.jsonl"
            reward = RepoNativeJudgeReward(
                trace_path=trace,
                score_fn=lambda **kwargs: {"reward": 1.0, "record": {}},
            )
            with self.assertRaises(json.JSONDecodeError):
                reward(["a", "b", "c", "d"], **kwargs)
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["record"]["infrastructure_error"]["type"], "JSONDecodeError")


class AnswerMarginPluginTests(unittest.TestCase):
    def _kwargs(self) -> dict:
        return {
            "packet_json": [json.dumps({"evidence_id": "E1", "required_users": ["u1", "u2"], "clips": []})],
            "evidence_id": ["E1"],
            "question_type": ["commonality"],
            "generation_mode": ["baseline"],
        }

    def test_registers_exact_orm_name_and_expands_existing_fields(self):
        from training.grpo_v3_reward_plugin import AnswerMarginReward, orms

        score_fn = StubAnswerMarginScoreFn()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=score_fn)
            values = reward(["a", "b", "c", "d"], **self._kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertIs(orms["egoqa_combined_video_answer_margin"], AnswerMarginReward)
        self.assertEqual(values, [0.25] * 4)
        self.assertEqual([call["candidate_index"] for call in score_fn.calls], [0, 1, 2, 3])
        self.assertTrue(all(call["packet"]["evidence_id"] == "E1" for call in score_fn.calls))
        self.assertTrue(all(call["question_type"] == "commonality" for call in score_fn.calls))
        self.assertTrue(all(call["generation_mode"] == "baseline" for call in score_fn.calls))
        self.assertTrue(all(row["reward_kind"] == "combined_video_answer_margin" for row in rows))

    def test_infrastructure_error_is_masked_traced_then_reraised(self):
        from training.grpo_v3_reward_plugin import AnswerMarginReward

        error = TimeoutError("answer scorer timeout")
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn(failure=error))
            with self.assertRaisesRegex(TimeoutError, "answer scorer timeout"):
                reward(["a", "b", "c", "d"], **self._kwargs())
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["reward"])
        self.assertTrue(rows[0]["record"]["masked"])
        self.assertFalse(rows[0]["record"]["eligible_for_grpo"])
        self.assertEqual(rows[0]["record"]["infrastructure_error"]["type"], "TimeoutError")

    def test_client_configuration_requires_explicit_environment(self):
        from training.grpo_v3_reward_plugin import AnswerMarginReward

        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "EGOQA_ANSWER_SCORER_BASE_URL"):
            AnswerMarginReward(trace_path=Path("unused.jsonl"))


if __name__ == "__main__":
    unittest.main()
