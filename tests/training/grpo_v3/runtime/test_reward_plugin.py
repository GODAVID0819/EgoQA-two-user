from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.grpo_v3.runtime.reward_plugin import ControlledGateReward, RepoNativeJudgeReward


class StubAnswerMarginScoreFn:
    def __init__(self, *, failure: Exception | None = None, fail_at: int | None = None) -> None:
        self.calls = []
        self.failure = failure
        self.fail_at = fail_at

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if self.failure is not None and (self.fail_at is None or kwargs["candidate_index"] == self.fail_at):
            raise self.failure
        return {"reward": 0.25, "record": {"masked": False, "eligible_for_grpo": True}}


class CustomAuditValue:
    pass


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
    def _kwargs(self, root: Path) -> dict:
        clips = []
        for user in ("u1", "u2"):
            video = root / f"{user}.mp4"
            video.write_bytes(b"video")
            clips.append({"agent_name": user, "local_video": str(video)})
        return {
            "packet_json": [json.dumps({"evidence_id": "E1", "required_users": ["u1", "u2"], "clips": clips})],
            "evidence_id": ["E1"],
            "question_type": ["commonality"],
            "generation_mode": ["baseline"],
            "global_step": [3],
        }

    def test_registers_exact_orm_name_and_expands_existing_fields(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward, orms

        score_fn = StubAnswerMarginScoreFn()
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=score_fn)
            values = reward(["a", "b", "c", "d"], **self._kwargs(Path(tmp)))
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertIs(orms["egoqa_combined_video_answer_margin"], AnswerMarginReward)
        self.assertEqual(values, [0.25] * 4)
        self.assertEqual([call["candidate_index"] for call in score_fn.calls], [0, 1, 2, 3])
        self.assertTrue(all(call["packet"]["evidence_id"] == "E1" for call in score_fn.calls))
        self.assertTrue(all(call["question_type"] == "commonality" for call in score_fn.calls))
        self.assertTrue(all(call["generation_mode"] == "baseline" for call in score_fn.calls))
        self.assertTrue(all(row["reward_kind"] == "combined_video_answer_margin" for row in rows))
        required = {
            "schema_version", "reward_revision", "experiment_version",
            "experiment_condition_id", "phase", "global_step", "reward_call_index",
            "candidate_index", "evidence_id", "raw_completion", "packet_summary",
            "video_inputs", "permutation_key",
        }
        self.assertTrue(all(required <= set(row) for row in rows))
        self.assertTrue(all(row["experiment_condition_id"] == "t05" for row in rows))
        self.assertTrue(all(row["global_step"] == 3 for row in rows))
        self.assertTrue(all(len(row["video_inputs"]) == 2 for row in rows))

    def test_infrastructure_error_is_masked_traced_then_reraised(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        error = TimeoutError("answer scorer timeout")
        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn(failure=error))
            with self.assertRaisesRegex(TimeoutError, "answer scorer timeout"):
                reward(["a", "b", "c", "d"], **self._kwargs(Path(tmp)))
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0]["reward"])
        self.assertTrue(rows[0]["record"]["masked"])
        self.assertFalse(rows[0]["record"]["eligible_for_grpo"])
        self.assertEqual(rows[0]["record"]["infrastructure_error"]["type"], "TimeoutError")
        self.assertEqual(rows[0]["experiment_condition_id"], "t05")
        self.assertEqual(rows[0]["global_step"], 3)
        self.assertEqual(rows[0]["raw_completion"], "a")
        self.assertEqual(len(rows[0]["video_inputs"]), 2)
        self.assertIsNotNone(rows[0]["permutation_key"])
        self.assertEqual(rows[0]["failure_stage"], "scoring")

    def test_partial_group_failure_writes_each_processed_candidate_once(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(
                trace_path=trace,
                score_fn=StubAnswerMarginScoreFn(failure=TimeoutError("late"), fail_at=1),
            )
            with self.assertRaises(TimeoutError):
                reward(["a", "b", "c", "d"], **self._kwargs(Path(tmp)))
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["candidate_index"] for row in rows], [0, 1])

    def test_wrong_condition_is_masked_traced_then_aborts_before_scorer(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        score_fn = StubAnswerMarginScoreFn()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"EGOQA_ANSWER_MARGIN_CONDITION_ID": "temperature_0.5"}, clear=False
        ):
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=score_fn)
            with self.assertRaisesRegex(ValueError, "t05"):
                reward(["a", "b", "c", "d"], **self._kwargs(Path(tmp)))
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(score_fn.calls, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["failure_stage"], "configuration")
        self.assertEqual(rows[0]["global_step"], 3)
        self.assertIsNone(rows[0]["permutation_key"])
        self.assertEqual(len(rows[0]["video_inputs"]), 2)

    def test_missing_global_step_is_masked_not_silently_fabricated(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._kwargs(Path(tmp))
            del kwargs["global_step"]
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
            with self.assertRaisesRegex(ValueError, "global_step"):
                reward(["a", "b", "c", "d"], **kwargs)
            row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        self.assertIsNone(row["global_step"])
        self.assertEqual(row["failure_stage"], "metadata")
        self.assertEqual(len(row["video_inputs"]), 2)

    def test_global_step_alignment_error_is_traced_then_reraised(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._kwargs(Path(tmp))
            kwargs["global_step"] = [0, 1]
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
            with self.assertRaisesRegex(ValueError, "global_step"):
                reward(["a", "b", "c", "d"], **kwargs)
            rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["failure_stage"], "metadata")
        self.assertIsNone(rows[0]["candidate_index"])
        self.assertEqual(rows[0]["raw_completions"], ["a", "b", "c", "d"])
        self.assertEqual(rows[0]["available_metadata"]["global_step"], [0, 1])

    def test_missing_packet_json_is_traced_then_original_keyerror_is_reraised(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._kwargs(Path(tmp))
            del kwargs["packet_json"]
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
            with self.assertRaises(KeyError):
                reward(["a", "b", "c", "d"], **kwargs)
            row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["failure_stage"], "metadata")
        self.assertIsNone(row["packet_summary"])
        self.assertEqual(row["experiment_condition_id"], "t05")

    def test_evidence_alignment_error_does_not_fabricate_candidate_identity(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._kwargs(Path(tmp))
            kwargs["evidence_id"] = ["E1", "E2"]
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
            with self.assertRaisesRegex(ValueError, "evidence_id"):
                reward(["a", "b", "c", "d"], **kwargs)
            row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        self.assertIsNone(row["evidence_id"])
        self.assertIsNone(row["candidate_index"])
        self.assertEqual(row["available_metadata"]["evidence_id"], ["E1", "E2"])

    def test_requires_exactly_four_completions_and_traces_empty_or_short_group(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward, CompletionGroupSizeError

        for completions in ([], ["a", "b", "c"]):
            with self.subTest(count=len(completions)), tempfile.TemporaryDirectory() as tmp:
                trace = Path(tmp) / "answer_margin.jsonl"
                reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
                with self.assertRaises(CompletionGroupSizeError):
                    reward(completions, **self._kwargs(Path(tmp)))
                rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["candidate_index"])
            self.assertEqual(rows[0]["failure_stage"], "metadata")
            self.assertEqual(rows[0]["record"]["infrastructure_error"]["type"], "CompletionGroupSizeError")
            self.assertEqual(rows[0]["actual_completion_count"], len(completions))

    def test_path_metadata_snapshot_cannot_replace_original_alignment_error(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._kwargs(Path(tmp))
            kwargs["evidence_id"] = [Path("E1"), Path("E2")]
            kwargs["question_type"] = [{"values": {"b", "a"}, "blob": b"abc"}]
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
            with self.assertRaisesRegex(ValueError, "evidence_id"):
                reward(["a", "b", "c", "d"], **kwargs)
            self.assertGreater(trace.stat().st_size, 0)
            row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        evidence_snapshot = row["available_metadata"]["evidence_id"]
        self.assertEqual([item["type"] for item in evidence_snapshot], ["Path", "Path"])
        self.assertEqual(row["available_metadata"]["question_type"][0]["blob"]["type"], "bytes")
        self.assertEqual(row["available_metadata"]["question_type"][0]["values"]["type"], "set")

    def test_nonfinite_reward_with_unsafe_prior_record_is_traced_before_original_error(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        def nonfinite_score(**_kwargs):
            return {
                "reward": math.nan,
                "record": {
                    "raw_margin": math.nan,
                    "clipped_margin": math.inf,
                    "path": Path("audit/value"),
                    "custom": CustomAuditValue(),
                    "masked": False,
                    "eligible_for_grpo": True,
                },
            }

        with tempfile.TemporaryDirectory() as tmp:
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=nonfinite_score)
            with self.assertRaisesRegex(ValueError, "answer-margin reward 非有限"):
                reward(["a", "b", "c", "d"], **self._kwargs(Path(tmp)))
            self.assertGreater(trace.stat().st_size, 0)
            row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["failure_stage"], "response_validation")
        self.assertEqual(row["record"]["raw_margin"], "nan")
        self.assertEqual(row["record"]["clipped_margin"], "inf")
        self.assertEqual(row["record"]["path"], {"type": "Path", "value": "audit\\value"})
        self.assertTrue(row["record"]["custom"]["type"].endswith(".CustomAuditValue"))
        self.assertTrue(row["record"]["masked"])
        self.assertFalse(row["record"]["eligible_for_grpo"])
        self.assertEqual(row["record"]["infrastructure_error"]["type"], "NonFiniteRewardError")

    def test_rejects_mixed_group_identity_before_any_scorer_call(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward, GroupIdentityError

        def mutations(kwargs):
            packet_rows = kwargs["packet_json"] * 4
            changed_packet = json.loads(packet_rows[-1])
            changed_packet["extra_identity"] = "different"
            return {
                "evidence": {**kwargs, "evidence_id": ["E1", "E1", "E1", "E2"]},
                "step": {**kwargs, "global_step": [3, 3, 3, 4]},
                "packet": {**kwargs, "packet_json": [*packet_rows[:3], json.dumps(changed_packet)]},
                "question_type": {**kwargs, "question_type": ["commonality"] * 3 + ["difference"]},
                "generation_mode": {**kwargs, "generation_mode": ["baseline"] * 3 + ["other"]},
            }

        for name in ("evidence", "step", "packet", "question_type", "generation_mode"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                kwargs = self._kwargs(Path(tmp))
                scorer = StubAnswerMarginScoreFn()
                trace = Path(tmp) / "answer_margin.jsonl"
                reward = AnswerMarginReward(trace_path=trace, score_fn=scorer)
                with self.assertRaises(GroupIdentityError):
                    reward(["a", "b", "c", "d"], **mutations(kwargs)[name])
                rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(scorer.calls, [])
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]["candidate_index"])
            self.assertEqual(rows[0]["failure_stage"], "metadata")
            self.assertEqual(rows[0]["record"]["infrastructure_error"]["type"], "GroupIdentityError")
            self.assertEqual(len(rows[0]["group_identity"]["packet_sha256"]), 4)

    def test_equivalent_packet_json_key_orders_share_one_group_identity(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._kwargs(Path(tmp))
            packet_value = json.loads(kwargs["packet_json"][0])
            reversed_value = dict(reversed(list(packet_value.items())))
            kwargs["packet_json"] = [
                json.dumps(packet_value, ensure_ascii=False),
                json.dumps(reversed_value, ensure_ascii=False),
                json.dumps(packet_value, ensure_ascii=False, sort_keys=True),
                json.dumps(reversed_value, ensure_ascii=False, sort_keys=True),
            ]
            scorer = StubAnswerMarginScoreFn()
            reward = AnswerMarginReward(trace_path=Path(tmp) / "trace.jsonl", score_fn=scorer)
            values = reward(["a", "b", "c", "d"], **kwargs)
        self.assertEqual(values, [0.25] * 4)
        self.assertEqual(len(scorer.calls), 4)

    def test_cyclic_metadata_snapshot_preserves_original_alignment_error(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        cyclic_dict = {}
        cyclic_dict["self"] = cyclic_dict
        first_list = []
        second_list = [first_list]
        first_list.append(second_list)
        for name, cyclic_value, expected_type in (
            ("dict", cyclic_dict, "dict"),
            ("list", first_list, "list"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                kwargs = self._kwargs(Path(tmp))
                kwargs["question_type"] = [cyclic_value]
                kwargs["global_step"] = [0, 1]
                trace = Path(tmp) / "answer_margin.jsonl"
                reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
                with self.assertRaisesRegex(ValueError, "global_step"):
                    reward(["a", "b", "c", "d"], **kwargs)
                rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            serialized = json.dumps(rows[0]["available_metadata"]["question_type"])
            self.assertIn('"type": "cycle"', serialized)
            self.assertIn(f'"container_type": "{expected_type}"', serialized)

    def test_shared_noncyclic_metadata_is_not_marked_as_cycle(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        shared = {"value": 1}
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = self._kwargs(Path(tmp))
            kwargs["question_type"] = [{"left": shared, "right": shared}]
            kwargs["global_step"] = [0, 1]
            trace = Path(tmp) / "answer_margin.jsonl"
            reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
            with self.assertRaisesRegex(ValueError, "global_step"):
                reward(["a", "b", "c", "d"], **kwargs)
            row = json.loads(trace.read_text(encoding="utf-8").splitlines()[0])
        snapshot = row["available_metadata"]["question_type"][0]
        self.assertEqual(snapshot["left"], {"value": 1})
        self.assertEqual(snapshot["right"], {"value": 1})
        self.assertNotIn('"type": "cycle"', json.dumps(snapshot))

    def test_deep_noncyclic_metadata_is_bounded_without_covering_original_error(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        deep_list = []
        cursor = deep_list
        for _ in range(1200):
            child = []
            cursor.append(child)
            cursor = child
        deep_dict = {}
        dict_cursor = deep_dict
        for _ in range(1200):
            child = {}
            dict_cursor["child"] = child
            dict_cursor = child

        for name, deep_value, expected_type in (
            ("list", deep_list, "list"),
            ("dict", deep_dict, "dict"),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                kwargs = self._kwargs(Path(tmp))
                kwargs["question_type"] = [deep_value]
                kwargs["global_step"] = [0, 1]
                trace = Path(tmp) / "answer_margin.jsonl"
                reward = AnswerMarginReward(trace_path=trace, score_fn=StubAnswerMarginScoreFn())
                with self.assertRaisesRegex(ValueError, "global_step"):
                    reward(["a", "b", "c", "d"], **kwargs)
                rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 1)
            serialized = json.dumps(rows[0]["available_metadata"]["question_type"])
            self.assertIn('"type": "max_depth"', serialized)
            self.assertIn(f'"container_type": "{expected_type}"', serialized)

    def test_client_configuration_requires_explicit_environment(self):
        from training.grpo_v3.runtime.reward_plugin import AnswerMarginReward

        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(RuntimeError, "EGOQA_ANSWER_SCORER_BASE_URL"):
            AnswerMarginReward(trace_path=Path("unused.jsonl"))


if __name__ == "__main__":
    unittest.main()
