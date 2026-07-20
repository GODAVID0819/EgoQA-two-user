from __future__ import annotations

import json
import math
import unittest
from typing import Any

from training.grpo_v3_formality_reward import (
    FORMALITY_COMPONENT,
    confidence_reward,
    make_formality_score_fn,
)


def _completion() -> str:
    return json.dumps(
        {
            "question": "Which mug was still on the counter after I left?",
            "options": ["red", "blue", "green", "white", "black"],
            "correct": "white",
        }
    )


class _Runner:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


def _modules(*, pass_logprob: float = -2.0, fail_logprob: float = -10.0) -> dict[str, Any]:
    calls: list[dict[str, Any]] = []

    def run_model_judge_branch(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {
            "review_passed": True,
            "checks": {
                "qa_formality": {
                    "status": "PASS",
                    "quality_score": 3,
                }
            },
            "blocking_failures": [],
            "choice_logit_signal": {
                "available": True,
                "choice_logprobs": {
                    "PASS": pass_logprob,
                    "FAIL": fail_logprob,
                },
            },
            "raw_output": "{}",
        }

    return {
        "OpenAICompatibleLocalRunner": _Runner,
        "build_qa_formality_judge_prompt": lambda qa, packet, schema_errors: "formality prompt",
        "run_model_judge_branch": run_model_judge_branch,
        "qa_for_judger_prompt": lambda qa: qa,
        "validate_qa_item": lambda qa: [],
        "complete_generator_metadata": lambda qa, packet, question_type: None,
        "calls": calls,
    }


def _score(modules: dict[str, Any]):
    return make_formality_score_fn(
        review_model_id="reviewer",
        review_base_url="http://127.0.0.1:8001/v1",
        policy_model_id="policy",
        review_max_new_tokens=256,
        modules=modules,
    )


class FormalityConfidenceMathTests(unittest.TestCase):
    def test_margin_is_clipped_and_scaled_to_unit_interval(self) -> None:
        self.assertEqual(confidence_reward(40.0, 0.0), 1.0)
        self.assertEqual(confidence_reward(-40.0, 0.0), -1.0)
        self.assertEqual(confidence_reward(-2.0, -10.0), 0.25)

    def test_nonfinite_logprob_is_infrastructure_error(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "非有限"):
                    confidence_reward(value, 0.0)


class FormalityScoreTests(unittest.TestCase):
    def test_valid_completion_uses_only_text_formality_judge(self) -> None:
        modules = _modules()
        result = _score(modules)(
            raw_completion=_completion(),
            packet={"required_users": ["A", "B"]},
            evidence_id="E1",
            question_type="difference",
            generation_mode="default",
            candidate_index=0,
        )

        self.assertEqual(result["reward"], 0.25)
        record = result["record"]
        self.assertEqual(record["reward_components"], {FORMALITY_COMPONENT: 0.25})
        self.assertEqual(record["reward_source"], "judge_pass_fail_logprob_margin")
        self.assertEqual(set(record["judge_trace"]), {"qa_formality"})
        self.assertEqual(record["qa_formality_status"], "PASS")
        self.assertTrue(record["judge_called"])
        self.assertFalse(record["masked"])
        self.assertEqual(len(modules["calls"]), 1)
        self.assertEqual(modules["calls"][0]["image_paths"], [])
        self.assertEqual(modules["calls"][0]["video_paths"], [])

    def test_unrecoverable_completion_is_formality_floor_without_judge(self) -> None:
        modules = _modules()
        result = _score(modules)(
            raw_completion='{"question": "truncated',
            packet={"required_users": ["A", "B"]},
            evidence_id="E1",
            question_type="difference",
            generation_mode="default",
            candidate_index=1,
        )

        self.assertEqual(result["reward"], -1.0)
        record = result["record"]
        self.assertEqual(record["reward_components"], {FORMALITY_COMPONENT: -1.0})
        self.assertEqual(record["reward_source"], "deterministic_unjudgeable_floor")
        self.assertEqual(record["qa_formality_status"], "FAIL")
        self.assertFalse(record["judge_called"])
        self.assertEqual(modules["calls"], [])

    def test_missing_choice_logprobs_is_infrastructure_error(self) -> None:
        modules = _modules()
        modules["run_model_judge_branch"] = lambda **kwargs: {
            "checks": {"qa_formality": {"status": "PASS"}},
            "choice_logit_signal": {"available": False},
        }
        with self.assertRaisesRegex(ValueError, "logprob"):
            _score(modules)(
                raw_completion=_completion(),
                packet={"required_users": ["A", "B"]},
                evidence_id="E1",
                question_type="difference",
                generation_mode="default",
                candidate_index=2,
            )

    def test_invalid_judge_status_is_infrastructure_error(self) -> None:
        modules = _modules()
        modules["run_model_judge_branch"] = lambda **kwargs: {
            "checks": {"qa_formality": {"status": "UNKNOWN"}},
            "choice_logit_signal": {
                "available": True,
                "choice_logprobs": {"PASS": -1.0, "FAIL": -2.0},
            },
        }
        with self.assertRaisesRegex(ValueError, "PASS/FAIL"):
            _score(modules)(
                raw_completion=_completion(),
                packet={"required_users": ["A", "B"]},
                evidence_id="E1",
                question_type="difference",
                generation_mode="default",
                candidate_index=3,
            )


if __name__ == "__main__":
    unittest.main()
