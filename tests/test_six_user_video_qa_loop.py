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
    complete_generator_metadata,
    human_audit_packet,
    run_answerability_eval,
)
from egolife_two_user_qa.schema import validate_qa_item  # noqa: E402


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


def six_user_packet() -> dict[str, object]:
    return {
        "evidence_id": "six-user-packet",
        "required_users": list(SIX_USERS),
        "input_users": list(SIX_USERS),
        "speaker_user": SIX_USERS[0],
        "anchor_provider_users": SIX_USERS[1:3],
        "additional_provider_users": SIX_USERS[3:],
        "evidence_provider_user": SIX_USERS[1],
        "evidence_provider_users": SIX_USERS[1:3],
        "media_roles": {
            SIX_USERS[0]: "speaker_pruned",
            SIX_USERS[1]: "anchor_provider_pruned",
            SIX_USERS[2]: "anchor_provider_pruned",
            SIX_USERS[3]: "additional_provider_full",
            SIX_USERS[4]: "additional_provider_full",
            SIX_USERS[5]: "additional_provider_full",
        },
        "clips": [
            {
                "agent_name": user,
                "local_video": f"{user}.mp4",
                "generator_video": f"generator-{user}.mp4",
                "media_role": role,
            }
            for user, role in (
                (SIX_USERS[0], "speaker_pruned"),
                (SIX_USERS[1], "anchor_provider_pruned"),
                (SIX_USERS[2], "anchor_provider_pruned"),
                (SIX_USERS[3], "additional_provider_full"),
                (SIX_USERS[4], "additional_provider_full"),
                (SIX_USERS[5], "additional_provider_full"),
            )
        ],
        "source_urls": {},
    }


def qa_for_metadata(*, supporting_user: str = "anchor_one") -> dict[str, object]:
    return {
        **six_user_qa(correct="A"),
        "answer": "Option A",
        "evidence": [],
        "single_user_answerability": {
            "speaker": "insufficient because the external detail is not visible",
        },
        "combined_answerability": "sufficient because the videos support one answer",
        "model_id": "test-model",
        "source_urls": {},
        "review": {},
        "per_user_evidence_claims": [
            {
                "user": supporting_user,
                "claim": "This provider view shows the answer-bearing item.",
            }
        ],
    }


class SixUserAnswerabilityTests(unittest.TestCase):
    def test_six_user_audit_and_metadata_expose_explicit_roles(self) -> None:
        packet = six_user_packet()
        audit = human_audit_packet(packet)
        qa = qa_for_metadata()

        complete_generator_metadata(qa, packet=packet, question_type="neutral")

        for output in (audit, qa):
            self.assertEqual(output["input_users"], SIX_USERS)
            self.assertEqual(output["speaker_user"], "speaker")
            self.assertEqual(output["anchor_provider_users"], ["anchor_one", "anchor_two"])
            self.assertEqual(
                output["additional_provider_users"],
                ["context_one", "context_two", "context_three"],
            )
            self.assertEqual(output["evidence_provider_user"], "anchor_one")
            self.assertEqual(
                output["evidence_provider_users"],
                ["anchor_one", "anchor_two"],
            )
            self.assertEqual(set(output["media_roles"]), set(SIX_USERS))

        self.assertEqual(
            qa["supporting_user_claims"],
            [
                {
                    "user": "anchor_one",
                    "claim": "This provider view shows the answer-bearing item.",
                }
            ],
        )
        self.assertNotIn("context_one", {row["user"] for row in qa["supporting_user_claims"]})
        self.assertEqual(validate_qa_item(qa), [])

    def test_six_user_schema_rejects_supporting_claim_outside_input_users(self) -> None:
        qa = qa_for_metadata(supporting_user="outsider")
        complete_generator_metadata(qa, packet=six_user_packet(), question_type="neutral")

        errors = validate_qa_item(qa)

        self.assertTrue(any("supporting_user_claims" in error for error in errors), errors)

    def test_six_user_packet_rejects_role_order_mismatch(self) -> None:
        packet = six_user_packet()
        packet["anchor_provider_users"] = ["anchor_two", "anchor_one"]

        with self.assertRaisesRegex(ValueError, "anchor_provider_users"):
            human_audit_packet(packet)

    def test_two_user_metadata_remains_valid_without_six_user_fields(self) -> None:
        users = ["speaker", "provider"]
        packet = {
            "required_users": users,
            "clips": [{"agent_name": user} for user in users],
            "source_urls": {},
        }
        qa = {
            "qa_id": "two-user-qa",
            "question": "What item was added after I left?",
            "options": [f"Option {letter}" for letter in "ABCDE"],
            "correct": "A",
            "answer": "Option A",
            "required_users": users,
            "evidence": [],
            "single_user_answerability": {
                "speaker": "insufficient because the item is not visible",
                "provider": "sufficient because the item is visible",
            },
            "combined_answerability": "sufficient because both views support the answer",
            "model_id": "test-model",
            "source_urls": {},
            "review": {},
            "per_user_evidence_claims": [
                {"user": "provider", "claim": "The item is visible."}
            ],
        }

        complete_generator_metadata(qa, packet=packet, question_type="neutral")

        self.assertNotIn("input_users", qa)
        self.assertNotIn("supporting_user_claims", qa)
        self.assertEqual(validate_qa_item(qa), [])

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
