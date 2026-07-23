from __future__ import annotations

import math
import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace

from training.grpo_v3.baseline.repo_reward import (
    apply_content_reward_revision,
    make_repo_score_fn,
    validate_groundedness_audit_approval,
)


class _Record:
    def __init__(self, reward_total: float | None, *, masked: bool = False) -> None:
        self.reward_total = reward_total
        self._data = {
            "masked": masked,
            "eligible_for_grpo": not masked,
            "reward_total": reward_total,
            "reward_components": {"groundedness": 2.0},
        }

    def to_dict(self) -> dict:
        return dict(self._data)


class RepoRewardThreeTierTests(unittest.TestCase):
    def _modules(
        self,
        *,
        reward_total: float | None = 2.0,
        full_videos: list[str] | None = None,
        reviewer_error: Exception | None = None,
    ) -> tuple[dict, list[dict]]:
        reviewer_calls: list[dict] = []

        class Runner:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs

        def run_parallel_review_judges(**kwargs):
            reviewer_calls.append(kwargs)
            if reviewer_error is not None:
                raise reviewer_error
            return (
                {"gate": {"passed": True, "reason": "ok"}},
                {"gate": {"passed": True}},
                {"trace": "reviewed"},
            )

        modules = {
            "media_for_clips": lambda *args, **kwargs: ([], list(full_videos or ["u1.mp4", "u2.mp4"])),
            "complete_generator_metadata": lambda qa, **kwargs: qa,
            "video_evidence_for_packet": lambda packet: [{"user": "u1"}, {"user": "u2"}],
            "human_audit_packet": lambda packet: {"passed": True},
            "run_parallel_review_judges": run_parallel_review_judges,
            "build_review_from_gates": lambda **kwargs: {"status": "passed"},
            "validate_qa_item": lambda qa: [],
            "OpenAICompatibleLocalRunner": Runner,
            "compute_judge_reward": lambda data: _Record(
                reward_total,
                masked=reward_total is None,
            ),
        }
        return modules, reviewer_calls

    @staticmethod
    def _packet() -> dict:
        return {
            "evidence_id": "E1",
            "required_users": ["u1", "u2"],
            "clips": [{"user": "u1"}, {"user": "u2"}],
        }

    def _score(self, modules: dict, raw_completion: str, *, packet: dict | None = None) -> dict:
        scorer = make_repo_score_fn(
            review_model_id="reviewer",
            review_base_url="http://reviewer/v1",
            policy_model_id="policy",
            review_max_new_tokens=128,
            modules=modules,
        )
        return scorer(
            raw_completion=raw_completion,
            packet=packet or self._packet(),
            evidence_id="E1",
            question_type="commonality",
            generation_mode="baseline",
            candidate_index=2,
        )

    def test_raw_valid_runs_reviewer_with_zero_format_component(self) -> None:
        modules, reviewer_calls = self._modules()
        result = self._score(modules, '{"question":"q"}')

        self.assertEqual(result["reward"], 2.0)
        self.assertEqual(len(reviewer_calls), 1)
        self.assertEqual(result["record"]["format_validation"]["status"], "raw_valid")
        self.assertEqual(result["record"]["reward_components"]["groundedness"], 2.0)
        self.assertEqual(result["record"]["reward_components"]["format"], 0.0)
        self.assertEqual(result["record"]["reward_total"], 2.0)

    def test_repaired_runs_reviewer_on_repaired_object_and_subtracts_half(self) -> None:
        modules, reviewer_calls = self._modules()
        raw = '{"combined_answerability":"sufficient"\n"generator_rationale":"ok"}'
        result = self._score(modules, raw)

        self.assertEqual(result["reward"], 1.5)
        self.assertEqual(len(reviewer_calls), 1)
        self.assertEqual(reviewer_calls[0]["qa_item"]["generator_rationale"], "ok")
        validation = result["record"]["format_validation"]
        self.assertEqual(validation["status"], "repaired")
        self.assertEqual(validation["raw_completion"], raw)
        self.assertIn(',', validation["repaired_completion"])
        self.assertEqual(validation["repair_operations"][0]["operation"], "insert_missing_member_comma")
        self.assertEqual(result["record"]["reward_components"]["format"], -0.5)
        self.assertEqual(result["record"]["reward_total"], 1.5)

    def test_unrecoverable_skips_reviewer_and_returns_finite_unmasked_reward(self) -> None:
        modules, reviewer_calls = self._modules()
        raw = '{"question":"truncated"'
        result = self._score(modules, raw)

        self.assertEqual(result["reward"], -3.0)
        self.assertEqual(reviewer_calls, [])
        self.assertFalse(result["record"]["masked"])
        self.assertTrue(result["record"]["eligible_for_grpo"])
        self.assertEqual(result["record"]["reward_total"], -3.0)
        self.assertEqual(result["record"]["reward_components"], {"format": -3.0})
        self.assertEqual(result["record"]["format_validation"]["status"], "unrecoverable")
        self.assertEqual(result["record"]["format_validation"]["raw_completion"], raw)
        self.assertEqual(result["record"]["format_validation"]["parse_error"]["type"], "JSONDecodeError")

    def test_missing_one_of_two_reviewer_videos_remains_masked_even_for_bad_json(self) -> None:
        modules, reviewer_calls = self._modules(full_videos=["u1.mp4"])
        result = self._score(modules, '{"question":"truncated"')

        self.assertIsNone(result["reward"])
        self.assertTrue(result["record"]["masked"])
        self.assertEqual(result["record"]["mask_reason"], "reviewer_full_video_count_mismatch: expected=2 actual=1")
        self.assertEqual(reviewer_calls, [])

    def test_evidence_mismatch_remains_masked_instead_of_format_reward(self) -> None:
        modules, reviewer_calls = self._modules()
        packet = self._packet()
        packet["evidence_id"] = "E2"
        result = self._score(modules, '{"question":"truncated"', packet=packet)

        self.assertIsNone(result["reward"])
        self.assertTrue(result["record"]["masked"])
        self.assertIn("evidence_id_mismatch", result["record"]["mask_reason"])
        self.assertEqual(reviewer_calls, [])

    def test_reviewer_failure_propagates_instead_of_becoming_minus_three(self) -> None:
        modules, _ = self._modules(reviewer_error=TimeoutError("reviewer timeout"))

        with self.assertRaisesRegex(TimeoutError, "reviewer timeout"):
            self._score(modules, '{"question":"q"}')

    def test_none_content_reward_remains_masked(self) -> None:
        modules, reviewer_calls = self._modules(reward_total=None)
        result = self._score(modules, '{"question":"q"}')

        self.assertIsNone(result["reward"])
        self.assertEqual(len(reviewer_calls), 1)
        self.assertTrue(result["record"]["masked"])
        self.assertEqual(result["record"]["reward_components"]["format"], 0.0)

    def test_nonfinite_content_reward_is_not_converted_to_format_failure(self) -> None:
        modules, reviewer_calls = self._modules(reward_total=math.inf)
        result = self._score(modules, '{"question":"q"}')

        self.assertTrue(math.isinf(result["reward"]))
        self.assertEqual(len(reviewer_calls), 1)
        self.assertNotEqual(result["reward"], -3.0)

    def test_ground_answer_gap_revision_recomputes_scalar_and_preserves_caps(self) -> None:
        record = {
            "groundedness_status": "PASS",
            "combined_correct": True,
            "provider_only_correct": False,
            "speaker_only_correct": False,
            "reward_components": {
                "groundedness": 1.0,
                "combined_answerability": 1.0,
                "grounded_answerable_bonus": 0.5,
                "subset_leakage": 0.0,
                "qa_formality": 0.5,
                "shallow_activity_query": 0.0,
                "provider_only_cap": 0.0,
                "shallow_activity_cap": 0.0,
                "speaker_leakage_cap": 0.0,
            },
            "shallow_activity_status": "PASS",
        }
        revised = apply_content_reward_revision(record, "ground_answer_gap_v1")
        self.assertEqual(revised["reward_components"]["groundedness"], 1.5)
        self.assertEqual(revised["reward_components"]["combined_answerability"], 1.5)
        self.assertEqual(revised["reward_total"], 4.0)
        self.assertEqual(revised["content_reward_revision"], "ground_answer_gap_v1")

        capped = dict(record)
        capped["provider_only_correct"] = True
        capped_result = apply_content_reward_revision(capped, "ground_answer_gap_v1")
        self.assertEqual(capped_result["reward_total"], 2.0)
        self.assertEqual(capped_result["reward_components"]["provider_only_cap"], 2.0)

    def test_reward_revision_requires_explicit_human_audit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "grpo_v3_groundedness_audit_v1",
                        "completed_count": 24,
                        "reviewer_pass_completed": 12,
                        "reviewer_fail_completed": 12,
                        "approved_for_weight_change": False,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "未批准"):
                validate_groundedness_audit_approval(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["approved_for_weight_change"] = True
            path.write_text(json.dumps(data), encoding="utf-8")
            approved = validate_groundedness_audit_approval(path)
            self.assertEqual(approved["completed_count"], 24)

    def test_reward_revision_accepts_multisignal_v2_audit_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "grpo_v3_multisignal_audit_v2",
                        "completed_count": 24,
                        "reviewer_pass_completed": 12,
                        "reviewer_fail_completed": 12,
                        "approved_for_weight_change": True,
                        "signals": {},
                    }
                ),
                encoding="utf-8",
            )
            approved = validate_groundedness_audit_approval(path)
            self.assertEqual(approved["schema_version"], "grpo_v3_multisignal_audit_v2")


if __name__ == "__main__":
    unittest.main()
