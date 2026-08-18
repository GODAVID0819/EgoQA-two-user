from __future__ import annotations

import json
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.video_qa_loop import (  # noqa: E402
    answerability_gate,
    build_answerability_conditions,
    run_answerability_eval,
)


SIX_USERS = ["speaker", "anchor_one", "anchor_two", "context_one", "context_two", "context_three"]


def six_user_qa(*, correct: str = "A") -> dict[str, object]:
    return {
        "qa_id": "six-user-qa",
        "required_users": list(SIX_USERS),
        "question": "Which item completed the shared setup?",
        "options": [f"Option {letter}" for letter in "ABCDE"],
        "correct": correct,
    }


def evaluation(condition: dict[str, object], choice: object) -> dict[str, object]:
    return {
        **condition,
        "choice": choice,
        "answer_text": "selected option",
        "evidence_used": "visible evidence",
    }


class SixUserAnswerabilityTests(unittest.TestCase):
    def test_six_users_build_exactly_two_conditions(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)

        self.assertEqual(
            conditions,
            [
                {
                    "condition_id": "speaker_only::speaker",
                    "condition_type": "speaker_only",
                    "users": ["speaker"],
                },
                {
                    "condition_id": "combined_all_six_users::" + "+".join(SIX_USERS),
                    "condition_type": "combined_all_six_users",
                    "users": SIX_USERS,
                },
            ],
        )

    def test_six_user_gate_passes_only_for_cross_view_gain(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            qa,
            [evaluation(conditions[0], "B"), evaluation(conditions[1], "A")],
        )

        self.assertTrue(gate["passed"])
        self.assertFalse(gate["speaker_only_correct"])
        self.assertTrue(gate["all_six_correct"])
        self.assertEqual(gate["speaker_only_choice"], "B")
        self.assertEqual(gate["all_six_choice"], "A")
        self.assertEqual(gate["cross_view_gain"], 1)
        self.assertEqual(gate["answerability_evaluated_condition_count"], 2)

    def test_speaker_only_correct_is_blocking(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            qa,
            [evaluation(conditions[0], "A"), evaluation(conditions[1], "A")],
        )

        self.assertFalse(gate["passed"])
        self.assertTrue(gate["speaker_only_correct"])
        self.assertIn("speaker-only", gate["reason"])

    def test_all_six_wrong_is_blocking_without_noise_claim(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            qa,
            [evaluation(conditions[0], "B"), evaluation(conditions[1], "C")],
        )

        self.assertFalse(gate["passed"])
        self.assertFalse(gate["all_six_correct"])
        self.assertEqual(gate["failure_label"], "all_six_wrong")
        self.assertNotIn("noise", gate["reason"].lower())

    def test_unparsed_speaker_or_all_six_choice_is_blocking(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)

        speaker_invalid = answerability_gate(
            qa,
            [evaluation(conditions[0], None), evaluation(conditions[1], "A")],
        )
        all_six_invalid = answerability_gate(
            qa,
            [evaluation(conditions[0], "B"), evaluation(conditions[1], None)],
        )

        self.assertFalse(speaker_invalid["passed"])
        self.assertEqual(speaker_invalid["failure_label"], "speaker_only_unparsed")
        self.assertFalse(all_six_invalid["passed"])
        self.assertEqual(all_six_invalid["failure_label"], "all_six_unparsed")

    def test_run_eval_calls_runner_exactly_twice_for_six_users(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                self.choices = iter(("B", "A"))

            def generate(self, prompt, *, image_paths, video_paths):
                choice = next(self.choices)
                self.calls.append(
                    {
                        "prompt": prompt,
                        "image_paths": list(image_paths),
                        "video_paths": list(video_paths),
                    }
                )
                return json.dumps(
                    {
                        "choice": choice,
                        "answer_text": f"Option {choice}",
                        "evidence_used": "visible evidence",
                    }
                )

        runner = Runner()
        result = run_answerability_eval(
            qa_item=six_user_qa(correct="A"),
            packet={"clips": []},
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=[],
        )

        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(len(result["evaluations"]), 2)
        self.assertTrue(result["gate"]["passed"])

    def test_two_user_condition_contract_is_unchanged(self) -> None:
        conditions = build_answerability_conditions(["speaker", "provider"])

        self.assertEqual(len(conditions), 3)
        self.assertEqual(
            [condition["condition_type"] for condition in conditions],
            ["single_user", "single_user", "combined_all_users"],
        )


if __name__ == "__main__":
    unittest.main()
