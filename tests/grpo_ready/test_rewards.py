from __future__ import annotations

import unittest

from grpo_ready.records import AttemptRecord
from grpo_ready.rewards import compute_reward


def make_attempt(
    *,
    parsed: bool = True,
    schema_pass: bool = True,
    judge: dict | None = None,
    speaker_choice: str = "insufficient",
    provider_choice: str = "A",
    combined_choice: str = "A",
) -> AttemptRecord:
    parsed_qa = {
        "correct": "A",
        "required_users": ["Jake", "Alice"],
    } if parsed else None
    if judge is None:
        judge = {
            "qa_formality": {
                "parsed": {"checks": {"qa_formality": {"status": "PASS"}}}
            },
            "evidence_groundedness": {
                "parsed": {"checks": {"evidence_groundedness": {"status": "PASS"}}}
            },
        }
    evaluations = [
        {"condition_type": "single_user", "users": ["Jake"], "choice": speaker_choice},
        {"condition_type": "single_user", "users": ["Alice"], "choice": provider_choice},
        {"condition_type": "combined_all_users", "users": ["Jake", "Alice"], "choice": combined_choice},
    ]
    return AttemptRecord(
        attempt_id="E1::attempt::1",
        evidence_id="E1",
        packet_status="accepted",
        question_type="difference",
        mode="baseline",
        attempt_index=1,
        feedback="",
        generator_prompt="prompt",
        generator_image_paths=(),
        generator_video_paths=(),
        evaluator_image_paths=(),
        evaluator_video_paths=(),
        raw_qa='{"correct":"A"}' if parsed else "not json",
        parsed_qa=parsed_qa,
        schema_errors=() if schema_pass else ("bad schema",),
        judge=judge,
        answerability={"evaluations": evaluations},
        accepted=True,
    )


class RewardTests(unittest.TestCase):
    def test_parse_failure_only_scores_parse_component(self) -> None:
        reward = compute_reward(make_attempt(parsed=False))

        self.assertEqual(reward.parse_reward, -2.0)
        self.assertIsNone(reward.schema_reward)
        self.assertIsNone(reward.groundedness_reward)
        self.assertEqual(reward.total, -2.0)
        self.assertIn("groundedness", reward.missing_components)
        self.assertFalse(reward.is_complete_reward)

    def test_full_pass_scores_expected_total(self) -> None:
        reward = compute_reward(make_attempt())

        self.assertEqual(reward.total, 5.0)
        self.assertEqual(reward.provider_alone_reward, 0.0)
        self.assertEqual(reward.speaker_leakage_reward, 0.0)
        self.assertTrue(reward.is_complete_reward)

    def test_schema_and_groundedness_failures_have_documented_scores(self) -> None:
        judge = {
            "qa_formality": {
                "parsed": {"checks": {"qa_formality": {"status": "PASS"}}}
            },
            "evidence_groundedness": {
                "parsed": {"checks": {"evidence_groundedness": {"status": "FAIL"}}}
            },
        }

        reward = compute_reward(make_attempt(schema_pass=False, judge=judge))

        self.assertEqual(reward.schema_reward, -0.5)
        self.assertEqual(reward.groundedness_reward, -2.0)

    def test_speaker_correct_is_penalized(self) -> None:
        reward = compute_reward(make_attempt(speaker_choice="A"))

        self.assertEqual(reward.speaker_leakage_reward, -2.0)
        self.assertEqual(reward.total, 3.0)

    def test_provider_correct_is_observed_but_not_penalized(self) -> None:
        reward = compute_reward(make_attempt(provider_choice="A"))

        self.assertTrue(reward.provider_alone_correct)
        self.assertEqual(reward.provider_alone_reward, 0.0)

    def test_missing_judge_is_not_treated_as_failure(self) -> None:
        attempt = make_attempt()
        attempt = AttemptRecord(**{**attempt.to_dict(), "judge": None})

        reward = compute_reward(attempt)

        self.assertIsNone(reward.formality_reward)
        self.assertIsNone(reward.groundedness_reward)
        self.assertIn("formality", reward.missing_components)
        self.assertIn("groundedness", reward.missing_components)
        self.assertFalse(reward.is_complete_reward)


if __name__ == "__main__":
    unittest.main()
