from __future__ import annotations

import json
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.video_qa_loop import (  # noqa: E402
    answerability_gate,
    build_answerability_conditions,
    complete_generator_metadata,
    condition_media_for_clips,
    human_audit_packet,
    media_for_clips,
    merge_parallel_judges,
    run_parallel_review_judges,
    run_answerability_eval,
    video_evidence_for_packet,
)
from egolife_two_user_qa import video_qa_loop  # noqa: E402
from egolife_two_user_qa.schema import validate_qa_item  # noqa: E402


SIX_USERS = ["speaker", "provider_one", "provider_two", "provider_three", "provider_four", "provider_five"]


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
        "provider_users": SIX_USERS[1:],
        "evidence_provider_user": SIX_USERS[1],
        "evidence_provider_users": SIX_USERS[1:],
        "media_roles": {
            SIX_USERS[0]: "speaker_consensus_pruned",
            SIX_USERS[1]: "provider_consensus_pruned",
            SIX_USERS[2]: "provider_consensus_pruned",
            SIX_USERS[3]: "provider_consensus_pruned",
            SIX_USERS[4]: "provider_consensus_pruned",
            SIX_USERS[5]: "provider_consensus_pruned",
        },
        "clips": [
            {
                "agent_name": user,
                "local_video": f"{user}.mp4",
                "generator_video": f"generator-{user}.mp4",
                "media_role": role,
            }
            for user, role in (
                (SIX_USERS[0], "speaker_consensus_pruned"),
                (SIX_USERS[1], "provider_consensus_pruned"),
                (SIX_USERS[2], "provider_consensus_pruned"),
                (SIX_USERS[3], "provider_consensus_pruned"),
                (SIX_USERS[4], "provider_consensus_pruned"),
                (SIX_USERS[5], "provider_consensus_pruned"),
            )
        ],
        "source_urls": {},
    }


def qa_for_metadata(*, supporting_user: str = "provider_one") -> dict[str, object]:
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
    def setUp(self) -> None:
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        self.tmp_path = tmp_root / f"six_user_loop_{uuid.uuid4().hex}"
        self.tmp_path.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp_path, True)

    def media_packet(self) -> dict[str, object]:
        packet = six_user_packet()
        clips = []
        for index, clip in enumerate(packet["clips"]):
            generator_video = self.tmp_path / f"generator_{index}.mp4"
            full_video = self.tmp_path / f"full_{index}.mp4"
            generator_video.write_bytes(b"generator")
            full_video.write_bytes(b"full")
            clips.append(
                {
                    **clip,
                    "local_video": str(generator_video),
                    "generator_local_video": str(generator_video),
                    "full_local_video": str(full_video),
                    "original_local_video": str(full_video),
                    "duration_seconds": 10.0,
                    "is_pruned": index < 3,
                }
            )
        packet["clips"] = clips
        return packet

    def test_six_user_audit_and_metadata_expose_explicit_roles(self) -> None:
        packet = six_user_packet()
        audit = human_audit_packet(packet)
        qa = qa_for_metadata()

        complete_generator_metadata(qa, packet=packet, question_type="neutral")

        for output in (audit, qa):
            self.assertEqual(output["input_users"], SIX_USERS)
            self.assertEqual(output["speaker_user"], "speaker")
            self.assertEqual(output["provider_users"], SIX_USERS[1:])
            self.assertNotIn("anchor_provider_users", output)
            self.assertNotIn("additional_provider_users", output)
            self.assertEqual(output["evidence_provider_user"], "provider_one")
            self.assertEqual(
                output["evidence_provider_users"],
                SIX_USERS[1:],
            )
            self.assertEqual(set(output["media_roles"]), set(SIX_USERS))

        self.assertEqual(
            qa["supporting_user_claims"],
            [
                {
                    "user": "provider_one",
                    "claim": "This provider view shows the answer-bearing item.",
                }
            ],
        )
        self.assertNotIn("provider_two", {row["user"] for row in qa["supporting_user_claims"]})
        self.assertEqual(validate_qa_item(qa), [])

    def test_six_user_audit_accepts_provider_only_similarity_pruning_roles(self) -> None:
        packet = six_user_packet()
        packet["media_roles"] = {
            SIX_USERS[0]: "speaker_reference_unpruned",
            **{
                user: "provider_similarity_pruned"
                for user in SIX_USERS[1:]
            },
        }

        audit = human_audit_packet(packet)

        self.assertEqual(audit["media_roles"], packet["media_roles"])

    def test_six_user_schema_rejects_supporting_claim_outside_input_users(self) -> None:
        qa = qa_for_metadata(supporting_user="outsider")
        complete_generator_metadata(qa, packet=six_user_packet(), question_type="neutral")

        errors = validate_qa_item(qa)

        self.assertTrue(any("supporting_user_claims" in error for error in errors), errors)

    def test_six_user_packet_rejects_role_order_mismatch(self) -> None:
        packet = six_user_packet()
        packet["provider_users"] = list(reversed(SIX_USERS[1:]))

        with self.assertRaisesRegex(ValueError, "provider_users"):
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

    def test_six_users_build_speaker_and_all_six_conditions(self) -> None:
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

    def test_six_user_gate_passes_when_speaker_is_wrong_and_all_six_is_correct(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            qa,
            [evaluation(conditions[0], "B"), evaluation(conditions[1], "A")],
        )

        self.assertTrue(gate["passed"])
        self.assertFalse(gate["speaker_only_correct"])
        self.assertEqual(gate["speaker_only_choice"], "B")
        self.assertTrue(gate["all_six_correct"])
        self.assertEqual(gate["all_six_choice"], "A")
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

    def test_unparsed_speaker_choice_is_blocking(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)

        speaker_invalid = answerability_gate(
            qa,
            [evaluation(conditions[0], None), evaluation(conditions[1], "A")],
        )

        self.assertFalse(speaker_invalid["passed"])
        self.assertEqual(speaker_invalid["failure_label"], "speaker_only_unparsed")

    def test_all_six_wrong_missing_or_unparsed_is_blocking(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)

        wrong = answerability_gate(
            qa,
            [evaluation(conditions[0], "B"), evaluation(conditions[1], "C")],
        )
        missing = answerability_gate(qa, [evaluation(conditions[0], "B")])
        unparsed = answerability_gate(
            qa,
            [evaluation(conditions[0], "B"), evaluation(conditions[1], None)],
        )

        self.assertEqual(wrong["failure_label"], "all_six_wrong")
        self.assertEqual(missing["failure_label"], "all_six_missing")
        self.assertEqual(unparsed["failure_label"], "all_six_unparsed")

    def test_run_eval_calls_runner_twice_for_six_users(self) -> None:
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

    def test_six_user_media_routes_generator_and_judges_in_order(self) -> None:
        packet = self.media_packet()
        clips = packet["clips"]

        _images, generator_videos = media_for_clips(
            clips,
            backend="transformers-local",
            allow_openai_video_input=False,
            media_role="generator",
        )
        _images, full_videos = media_for_clips(
            clips,
            backend="transformers-local",
            allow_openai_video_input=False,
            media_role="full",
        )

        self.assertEqual(generator_videos, [clip["local_video"] for clip in clips])
        self.assertEqual(full_videos, [clip["full_local_video"] for clip in clips])
        self.assertEqual(
            [row["media_role"] for row in video_evidence_for_packet(packet)],
            [
                "speaker_consensus_pruned",
                "provider_consensus_pruned",
                "provider_consensus_pruned",
                "provider_consensus_pruned",
                "provider_consensus_pruned",
                "provider_consensus_pruned",
            ],
        )

        condition_media = condition_media_for_clips(
            condition={
                "condition_id": "combined_all_six_users::all",
                "condition_type": "combined_all_six_users",
                "users": SIX_USERS,
            },
            clips=clips,
            image_paths=[],
            video_paths=full_videos,
            media_role="full",
        )
        self.assertEqual(condition_media["total_duration_seconds"], 60.0)

    def test_answerability_uses_full_speaker_then_all_six_full_videos(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.calls = []

            def generate(self, prompt, *, image_paths, video_paths):
                self.calls.append(
                    {
                        "prompt": prompt,
                        "image_paths": list(image_paths),
                        "video_paths": list(video_paths),
                    }
                )
                choice = "B" if len(self.calls) == 1 else "A"
                return json.dumps(
                    {
                        "choice": choice,
                        "answer_text": f"Option {choice}",
                        "evidence_used": "visible evidence",
                    }
                )

        packet = self.media_packet()
        runner = Runner()
        prompt_rows = []
        result = run_answerability_eval(
            qa_item=six_user_qa(correct="A"),
            packet=packet,
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=prompt_rows,
        )

        full_videos = [clip["full_local_video"] for clip in packet["clips"]]
        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(runner.calls[0]["video_paths"], full_videos[:1])
        self.assertEqual(runner.calls[1]["video_paths"], full_videos)
        self.assertEqual(len(prompt_rows), 2)
        self.assertTrue(all("elapsed_seconds" in row for row in prompt_rows))
        self.assertTrue(
            all("elapsed_seconds" in evaluation for evaluation in result["evaluations"])
        )
        self.assertFalse(result["gate"]["speaker_only_correct"])
        self.assertTrue(result["gate"]["all_six_correct"])

    def test_qa_formality_failure_remains_blocking_for_six_users(self) -> None:
        merged = merge_parallel_judges(
            qa_formality_judge={
                "checks": {
                    "qa_formality": {
                        "status": "FAIL",
                        "reason": "synthetic formality failure",
                        "fix": "rewrite the question",
                    }
                }
            },
            evidence_groundedness_judge={
                "checks": {
                    "evidence_groundedness": {
                        "status": "PASS",
                        "reason": "grounded",
                        "fix": "",
                    }
                }
            },
            answerability={"gate": {"passed": True, "reason": "speaker chose wrong"}},
            schema_errors=[],
            qa_item=six_user_qa(),
        )

        self.assertFalse(merged["review_passed"])
        self.assertIn("qa_formality", merged["blocking_failures"])

    def test_groundedness_prompt_row_uses_all_six_full_videos(self) -> None:
        packet = self.media_packet()
        _images, full_videos = media_for_clips(
            packet["clips"],
            backend="transformers-local",
            allow_openai_video_input=False,
            media_role="full",
        )
        prompt_rows = []

        def fake_judge_branch(*, check_name, **kwargs):
            return {
                "checks": {
                    check_name: {"status": "PASS", "reason": "ok", "fix": ""}
                },
                "blocking_failures": [],
                "feedback_to_generator": "",
                "raw_output": "{}",
                "elapsed_seconds": 0.01,
            }

        with (
            mock.patch.object(
                video_qa_loop,
                "run_model_judge_branch",
                side_effect=fake_judge_branch,
            ),
            mock.patch.object(
                video_qa_loop,
                "run_answerability_eval",
                return_value={
                    "evaluations": [],
                    "gate": {"passed": True, "reason": "test"},
                },
            ),
        ):
            _judge, _answerability, trace = run_parallel_review_judges(
                qa_item=six_user_qa(correct="A"),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=prompt_rows,
                full_image_paths=[],
                full_video_paths=full_videos,
                attempt=1,
            )

        groundedness_rows = [
            row for row in prompt_rows if row["stage"] == "evidence_groundedness_judge"
        ]
        self.assertEqual(len(groundedness_rows), 1)
        self.assertEqual(groundedness_rows[0]["video_paths"], full_videos)
        self.assertEqual(trace["evidence_groundedness"]["elapsed_seconds"], 0.01)

    def test_two_user_condition_contract_is_unchanged(self) -> None:
        conditions = build_answerability_conditions(["speaker", "provider"])

        self.assertEqual(len(conditions), 3)
        self.assertEqual(
            [condition["condition_type"] for condition in conditions],
            ["single_user", "single_user", "combined_all_users"],
        )


if __name__ == "__main__":
    unittest.main()
