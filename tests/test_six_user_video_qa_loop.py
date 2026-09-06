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
    parsed_answerability_sufficiency,
    run_parallel_review_judges,
    run_answerability_eval,
    six_user_ten_minute_reasoning_profiles,
    video_evidence_for_packet,
)
from egolife_two_user_qa import video_qa_loop  # noqa: E402
from egolife_two_user_qa.prompts import (  # noqa: E402
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
)
from egolife_two_user_qa.schema import (  # noqa: E402
    VIDEO_FIRST_REQUIRED_FIELDS,
    validate_qa_item,
)


SIX_USERS = ["speaker", "provider_one", "provider_two", "provider_three", "provider_four", "provider_five"]


def test_six_user_ten_minute_reasoning_profiles_are_stage_specific() -> None:
    profiles = six_user_ten_minute_reasoning_profiles()

    assert profiles["generator"].max_new_tokens == 8192
    assert profiles["generator"].disable_thinking is False
    assert profiles["evidence_segment_observation"] is profiles["generator"]
    assert profiles["evidence_groundedness_aggregation"] is profiles["generator"]
    assert profiles["answerability"] is profiles["generator"]
    assert profiles["qa_formality"].max_new_tokens == 2048
    assert profiles["qa_formality"].disable_thinking is True
    assert profiles["json_repair"] is profiles["qa_formality"]


def test_six_user_ten_minute_reasoning_profiles_allow_formality_token_override() -> None:
    profiles = six_user_ten_minute_reasoning_profiles(formality_max_new_tokens=3072)

    assert profiles["generator"].max_new_tokens == 8192
    assert profiles["qa_formality"].max_new_tokens == 3072
    assert profiles["json_repair"] is profiles["qa_formality"]


def test_six_user_ten_minute_fast_profiles_disable_thinking_stage_contract() -> None:
    profiles = video_qa_loop.six_user_ten_minute_fast_profiles()

    assert (profiles["generator"].max_new_tokens, profiles["generator"].disable_thinking) == (
        8192,
        True,
    )
    assert (
        profiles["qa_formality"].max_new_tokens,
        profiles["qa_formality"].disable_thinking,
    ) == (1024, True)
    assert (
        profiles["speaker_only_answerability"].max_new_tokens,
        profiles["speaker_only_answerability"].disable_thinking,
    ) == (2048, True)
    assert (
        profiles["all_six_answerability"].max_new_tokens,
        profiles["all_six_answerability"].disable_thinking,
    ) == (4096, True)
    assert (
        profiles["evidence_groundedness"].max_new_tokens,
        profiles["evidence_groundedness"].disable_thinking,
    ) == (4096, True)
    assert (profiles["json_repair"].max_new_tokens, profiles["json_repair"].disable_thinking) == (
        1024,
        True,
    )


def test_six_user_one_pass_profiles_use_stage_specific_video_quality() -> None:
    profiles = video_qa_loop.six_user_one_pass_profiles()

    assert (profiles["generator"].max_new_tokens, profiles["generator"].video_fps, profiles["generator"].max_image_pixels) == (
        4096,
        0.5,
        65_536,
    )
    for name in (
        "qa_formality",
        "speaker_only_answerability",
        "all_six_answerability",
        "evidence_groundedness",
        "json_repair",
    ):
        assert (profiles[name].video_fps, profiles[name].max_image_pixels) == (
            0.25,
            65_536,
        )
        assert profiles[name].disable_thinking is True


def test_video_first_schema_does_not_require_why_two_users_needed() -> None:
    assert "why_two_users_needed" not in VIDEO_FIRST_REQUIRED_FIELDS


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


def status_evaluation(
    condition: dict[str, object],
    status: str,
    choice: object,
) -> dict[str, object]:
    return {
        **condition,
        "answerability_status": status,
        "choice": choice,
        "answer_text": "selected option" if choice else "",
        "evidence_used": "visible evidence",
    }


def sufficiency_evaluation(
    condition: dict[str, object],
    answerable: bool,
    *,
    confidence: str = "HIGH",
    available_evidence: list[str] | None = None,
    missing_evidence: list[str] | None = None,
) -> dict[str, object]:
    visibility = "VISIBLE" if answerable else "NOT_VISIBLE"
    return {
        **condition,
        "reason": "The supplied videos provide the required evidence.",
        "needed_facts": [
            {
                "fact_id": "F1",
                "fact": "the later destination",
                "why_needed": "the question asks where the object ended up",
                "visibility": visibility,
                "confidence": confidence,
                "source_user": condition.get("users", [None])[0] if answerable else None,
                "original_time_range": "00:01:10-00:01:20" if answerable else None,
                "visual_description": (
                    (available_evidence or ["The handoff is visible."])[0]
                    if answerable
                    else (missing_evidence or ["The later destination is not visible."])[0]
                ),
            }
        ],
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


def provider_only_similarity_packet() -> dict[str, object]:
    packet = six_user_packet()
    media_roles = {
        SIX_USERS[0]: "speaker_reference_unpruned",
        **{user: "provider_similarity_pruned" for user in SIX_USERS[1:]},
    }
    packet["media_roles"] = media_roles
    packet["generator_media_mode"] = "speaker_full_five_provider_pruned_videos"
    packet["clips"] = [
        {**clip, "media_role": media_roles[str(clip["agent_name"])]}
        for clip in packet["clips"]
    ]
    return packet


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

    def test_fail_fast_review_stops_after_formality_on_first_attempt(self) -> None:
        packet = self.media_packet()

        def failed_formality(*, check_name, **_kwargs):
            self.assertEqual(check_name, "qa_formality")
            return {
                "review_passed": False,
                "checks": {
                    "qa_formality": {
                        "status": "FAIL",
                        "reason": "not a natural first-person question",
                        "fix": "rewrite naturally",
                        "semantic_subchecks": {
                            name: {"status": "FAIL", "reason": "failed"}
                            for name in video_qa_loop.QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                        },
                    }
                },
                "blocking_failures": ["qa_formality"],
                "feedback_to_generator": "rewrite naturally",
                "raw_output": "{}",
                "elapsed_seconds": 0.1,
            }

        with (
            mock.patch.object(
                video_qa_loop,
                "run_model_judge_branch",
                side_effect=failed_formality,
            ) as model_judge,
            mock.patch.object(video_qa_loop, "run_answerability_condition_eval") as answerability,
            mock.patch.object(video_qa_loop, "run_evidence_groundedness_review") as groundedness,
        ):
            judge, answerability_result, trace = video_qa_loop.run_fail_fast_review_judges(
                qa_item=six_user_qa(),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=[],
                full_image_paths=[],
                full_video_paths=[clip["full_local_video"] for clip in packet["clips"]],
                attempt=1,
                max_attempts=3,
                stage_profiles=video_qa_loop.six_user_ten_minute_fast_profiles(),
            )

        self.assertFalse(judge["gate"]["passed"])
        self.assertTrue(answerability_result["gate"]["skipped"])
        self.assertEqual(
            trace["skipped_checks"],
            ["speaker_only_answerability", "all_six_answerability", "evidence_groundedness"],
        )
        model_judge.assert_called_once()
        answerability.assert_not_called()
        groundedness.assert_not_called()

    def test_fail_fast_review_third_attempt_runs_every_metric_after_failure(self) -> None:
        packet = self.media_packet()
        observed_conditions: list[str] = []

        def judge_branch(*, check_name, **_kwargs):
            status = "FAIL" if check_name == "qa_formality" else "PASS"
            check = {"status": status, "reason": check_name, "fix": ""}
            if check_name == "qa_formality":
                check["semantic_subchecks"] = {
                    name: {"status": status, "reason": status.lower()}
                    for name in video_qa_loop.QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                }
            return {
                "review_passed": status == "PASS",
                "checks": {check_name: check},
                "blocking_failures": [] if status == "PASS" else [check_name],
                "feedback_to_generator": "",
                "raw_output": "{}",
                "elapsed_seconds": 0.1,
            }

        def condition_eval(*, condition, **_kwargs):
            observed_conditions.append(condition["condition_type"])
            return sufficiency_evaluation(
                condition,
                condition["condition_type"] == "combined_all_six_users",
            )

        groundedness_result = {
            "review_passed": True,
            "checks": {
                "evidence_groundedness": {
                    "status": "PASS",
                    "reason": "supported",
                    "fix": "",
                }
            },
            "blocking_failures": [],
            "feedback_to_generator": "",
            "raw_output": "{}",
            "elapsed_seconds": 0.1,
        }
        with (
            mock.patch.object(
                video_qa_loop,
                "run_model_judge_branch",
                side_effect=judge_branch,
            ),
            mock.patch.object(
                video_qa_loop,
                "run_answerability_condition_eval",
                side_effect=condition_eval,
            ),
            mock.patch.object(
                video_qa_loop,
                "run_evidence_groundedness_review",
                return_value=groundedness_result,
            ) as groundedness,
        ):
            judge, _answerability, trace = video_qa_loop.run_fail_fast_review_judges(
                qa_item=six_user_qa(),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=[],
                full_image_paths=[],
                full_video_paths=[clip["full_local_video"] for clip in packet["clips"]],
                attempt=3,
                max_attempts=3,
                stage_profiles=video_qa_loop.six_user_ten_minute_fast_profiles(),
            )

        self.assertFalse(judge["gate"]["passed"])
        self.assertEqual(
            observed_conditions,
            ["speaker_only", "combined_all_six_users"],
        )
        groundedness.assert_called_once()
        self.assertTrue(trace["force_complete_review"])
        self.assertEqual(trace["skipped_checks"], [])

    def test_sufficiency_parser_derives_false_from_missing_fact(self) -> None:
        condition = build_answerability_conditions(SIX_USERS)[0]
        self.assertEqual(
            parsed_answerability_sufficiency(sufficiency_evaluation(condition, False)),
            (False, None),
        )

    def test_sufficiency_parser_rejects_answer_fields_and_invalid_visible_provenance(self) -> None:
        condition = build_answerability_conditions(SIX_USERS)[0]
        answer_field_result = parsed_answerability_sufficiency(
            {
                **sufficiency_evaluation(condition, True),
                "choice": "A",
            }
        )
        self.assertIsNone(answer_field_result[0])
        self.assertIn("forbidden answer fields", answer_field_result[1])

        inconsistent = sufficiency_evaluation(condition, True)
        inconsistent["needed_facts"][0]["original_time_range"] = None
        inconsistent_result = parsed_answerability_sufficiency(inconsistent)
        self.assertIsNone(inconsistent_result[0])
        self.assertIn("original_time_range", inconsistent_result[1])

    def test_visible_fact_requires_high_confidence_for_sufficiency(self) -> None:
        condition = build_answerability_conditions(SIX_USERS)[0]
        evaluation_row = sufficiency_evaluation(
            condition,
            True,
            confidence="MEDIUM",
        )

        answerable, error = parsed_answerability_sufficiency(evaluation_row)

        self.assertIsNone(error)
        self.assertFalse(answerable)

    def test_six_user_gate_requires_speaker_insufficient_and_all_six_sufficient(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)
        result = answerability_gate(
            six_user_qa(),
            [
                sufficiency_evaluation(conditions[0], False),
                sufficiency_evaluation(conditions[1], True),
            ],
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["answerability_mode"], "evidence_sufficiency_reasoning")

    def test_answerability_fact_contract_rejects_rewritten_all_six_fact(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)
        speaker = sufficiency_evaluation(conditions[0], False)
        all_six = sufficiency_evaluation(conditions[1], True)
        all_six["needed_facts"][0]["fact"] = "a rewritten fact"

        result = answerability_gate(six_user_qa(), [speaker, all_six])

        self.assertFalse(result["passed"])
        self.assertEqual(
            result["failure_label"],
            "answerability_fact_contract_mismatch",
        )

    def test_six_user_answerability_passes_speaker_facts_to_all_six(self) -> None:
        observed: list[tuple[str, object]] = []

        def fake_condition_eval(*, condition, canonical_facts=None, **_kwargs):
            observed.append((condition["condition_type"], canonical_facts))
            row = sufficiency_evaluation(
                condition,
                condition["condition_type"] == "combined_all_six_users",
            )
            if canonical_facts:
                row["needed_facts"][0].update(canonical_facts[0])
            return row

        with mock.patch.object(
            video_qa_loop,
            "run_answerability_condition_eval",
            side_effect=fake_condition_eval,
        ):
            result = run_answerability_eval(
                qa_item=six_user_qa(),
                packet=six_user_packet(),
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=[],
            )

        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(observed[0], ("speaker_only", None))
        self.assertEqual(observed[1][0], "combined_all_six_users")
        self.assertEqual(
            observed[1][1],
            [
                {
                    "fact_id": "F1",
                    "fact": "the later destination",
                    "why_needed": "the question asks where the object ended up",
                }
            ],
        )

    def test_six_user_gate_emits_minimum_union_of_fact_source_users(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)
        all_six = sufficiency_evaluation(conditions[1], True)
        all_six["needed_facts"] = [
            {
                "fact_id": "F1",
                "fact": "the destination",
                "why_needed": "the question asks where the object ended up",
                "visibility": "VISIBLE",
                "confidence": "HIGH",
                "source_user": "provider_two",
                "original_time_range": "00:04:10-00:04:20",
                "visual_description": "Provider two shows the destination.",
            },
            {
                "fact_id": "F2",
                "fact": "the final state",
                "why_needed": "the final state distinguishes the answer",
                "visibility": "VISIBLE",
                "confidence": "HIGH",
                "source_user": "provider_four",
                "original_time_range": "00:07:30-00:07:40",
                "visual_description": "Provider four shows the final state.",
            },
            {
                "fact_id": "F3",
                "fact": "the destination label",
                "why_needed": "the label disambiguates the destination",
                "visibility": "VISIBLE",
                "confidence": "HIGH",
                "source_user": "provider_two",
                "original_time_range": "00:04:18-00:04:24",
                "visual_description": "Provider two also shows the label.",
            },
        ]
        speaker = sufficiency_evaluation(conditions[0], False)
        speaker["needed_facts"] = [
            {
                "fact_id": fact["fact_id"],
                "fact": fact["fact"],
                "why_needed": fact["why_needed"],
                "visibility": "NOT_VISIBLE",
                "confidence": "LOW",
                "source_user": None,
                "original_time_range": None,
                "visual_description": "The speaker view does not show this fact.",
            }
            for fact in all_six["needed_facts"]
        ]

        result = answerability_gate(
            six_user_qa(),
            [
                speaker,
                all_six,
            ],
        )

        self.assertTrue(result["passed"])
        self.assertEqual(
            result["minimum_required_users"],
            ["provider_two", "provider_four"],
        )
        self.assertEqual(result["minimum_required_user_count"], 2)
        self.assertEqual(
            result["minimum_required_users_basis"],
            "visible_high_needed_fact_source_union",
        )

    def test_six_user_gate_does_not_consume_choice_or_gold_answer(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)
        result = answerability_gate(
            six_user_qa(correct="E"),
            [
                {
                    **sufficiency_evaluation(conditions[0], False),
                    "choice": "E",
                },
                sufficiency_evaluation(conditions[1], True),
            ],
        )
        self.assertFalse(result["passed"])
        self.assertIn("unparsed", result["failure_label"])

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
        packet = provider_only_similarity_packet()

        audit = human_audit_packet(packet)

        self.assertEqual(audit["media_roles"], packet["media_roles"])

    def test_six_user_schema_accepts_sampled_frame_media_roles(self) -> None:
        packet = six_user_packet()
        packet["media_roles"] = {
            SIX_USERS[0]: "speaker_all_clustering_frames",
            **{user: "provider_retained_cluster_frames" for user in SIX_USERS[1:]},
        }
        qa = qa_for_metadata()

        complete_generator_metadata(qa, packet=packet, question_type="neutral")

        self.assertEqual(validate_qa_item(qa), [])

    def test_provider_only_packet_crosses_zero_gpu_qa_contract_pipeline(self) -> None:
        packet = provider_only_similarity_packet()
        qa = qa_for_metadata()

        audit = human_audit_packet(packet)
        complete_generator_metadata(qa, packet=packet, question_type="neutral")
        schema_errors = validate_qa_item(qa)
        formality_prompt = build_qa_formality_judge_prompt(
            qa,
            packet,
            schema_errors=schema_errors,
        )
        groundedness_prompt = build_evidence_groundedness_judge_prompt(qa, packet)
        answerability_conditions = build_answerability_conditions(SIX_USERS)

        self.assertEqual(audit["media_roles"], packet["media_roles"])
        self.assertEqual(schema_errors, [])
        self.assertIn("six-user", formality_prompt)
        self.assertIn("six-user", groundedness_prompt)
        self.assertEqual(
            [condition["condition_type"] for condition in answerability_conditions],
            ["speaker_only", "combined_all_six_users"],
        )

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
            [condition["condition_type"] for condition in conditions],
            ["speaker_only", "combined_all_six_users"],
        )

    def test_prompt_identity_separates_reused_qa_id_across_slots(self) -> None:
        self.assertTrue(hasattr(video_qa_loop, "prompt_rows_by_generation_identity"))
        rows = [
            {
                "stage": "answerability",
                "generation_slot_id": slot_id,
                "qa_id": "reused-qa-id",
                "attempt": 1,
                "condition_type": condition_type,
            }
            for slot_id in ("slot-1", "slot-2")
            for condition_type in ("speaker_only", "combined_all_six_users")
        ]

        indexed = video_qa_loop.prompt_rows_by_generation_identity(
            rows,
            stage="answerability",
        )

        self.assertEqual(len(indexed), 2)
        self.assertEqual(len(indexed[("slot-1", "reused-qa-id", 1)]), 2)
        self.assertEqual(len(indexed[("slot-2", "reused-qa-id", 1)]), 2)

    def test_six_user_gate_blocks_when_all_six_is_missing(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            six_user_qa(),
            [sufficiency_evaluation(conditions[0], False)],
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failure_label"], "all_six_missing")

    def test_speaker_answerable_is_blocking(self) -> None:
        condition = build_answerability_conditions(SIX_USERS)[0]

        gate = answerability_gate(
            six_user_qa(),
            [sufficiency_evaluation(condition, True)],
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failure_label"], "speaker_only_answerable")

    def test_all_six_insufficient_is_blocking(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            six_user_qa(),
            [
                sufficiency_evaluation(conditions[0], False),
                sufficiency_evaluation(conditions[1], False),
            ],
        )

        self.assertFalse(gate["passed"])
        self.assertEqual(gate["failure_label"], "all_six_not_answerable")

    def test_invalid_speaker_sufficiency_is_blocking(self) -> None:
        conditions = build_answerability_conditions(SIX_USERS)

        speaker_invalid = answerability_gate(
            six_user_qa(),
            [{**conditions[0], "answerable": True, "reason": "model decided directly", "needed_facts": []}],
        )

        self.assertFalse(speaker_invalid["passed"])
        self.assertEqual(speaker_invalid["failure_label"], "speaker_only_unparsed")

    def test_program_derives_sufficiency_from_every_needed_fact(self) -> None:
        condition = build_answerability_conditions(SIX_USERS)[0]
        evaluation_row = sufficiency_evaluation(condition, True)
        evaluation_row["needed_facts"].append(
            {
                "fact_id": "F2",
                "fact": "the recipient identity",
                "why_needed": "the question asks who received the object",
                "visibility": "AMBIGUOUS",
                "confidence": "LOW",
                "source_user": None,
                "original_time_range": None,
                "visual_description": "Several people could be the recipient.",
            }
        )

        answerable, error = parsed_answerability_sufficiency(evaluation_row)

        self.assertIsNone(error)
        self.assertFalse(answerable)

    def test_visible_fact_requires_a_condition_user_and_time_range(self) -> None:
        condition = build_answerability_conditions(SIX_USERS)[0]
        evaluation_row = sufficiency_evaluation(condition, True)
        evaluation_row["needed_facts"][0]["source_user"] = "provider_one"

        answerable, error = parsed_answerability_sufficiency(evaluation_row)

        self.assertIsNone(answerable)
        self.assertIn("source_user", error)

    def test_run_eval_calls_runner_for_speaker_and_all_six_conditions(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []
                self.answers = iter((False, True))

            def generate(self, prompt, *, image_paths, video_paths):
                answerable = next(self.answers)
                self.calls.append(
                    {
                        "prompt": prompt,
                        "image_paths": list(image_paths),
                        "video_paths": list(video_paths),
                    }
                )
                return json.dumps(
                    sufficiency_evaluation(
                        {
                            "condition_type": (
                                "speaker_only" if len(self.calls) == 1 else "combined_all_six_users"
                            ),
                            "users": ["speaker"] if len(self.calls) == 1 else list(SIX_USERS),
                        },
                        answerable,
                    )
                )

        runner = Runner()
        result = run_answerability_eval(
            qa_item={**six_user_qa(correct="A"), "generation_slot_id": "slot-001"},
            packet={"clips": []},
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=[],
        )

        self.assertEqual(len(runner.calls), 2)
        self.assertEqual(len(result["evaluations"]), 2)
        self.assertTrue(all("generation_slot_id" in row for row in result["evaluations"]))
        self.assertTrue(all(row["generation_slot_id"] == "slot-001" for row in result["evaluations"]))
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(
            result["gate"]["answerability_mode"],
            "evidence_sufficiency_reasoning",
        )

    def test_run_eval_propagates_answerability_call_profile(self) -> None:
        class Runner:
            def __init__(self) -> None:
                self.calls: list[object] = []

            def generate(self, prompt, *, image_paths, video_paths, call_profile=None):
                self.calls.append(call_profile)
                condition = {
                    "condition_type": (
                        "speaker_only" if len(self.calls) == 1 else "combined_all_six_users"
                    ),
                    "users": ["speaker"] if len(self.calls) == 1 else list(SIX_USERS),
                }
                return json.dumps(
                    sufficiency_evaluation(condition, len(self.calls) == 2)
                )

        profile = six_user_ten_minute_reasoning_profiles()["answerability"]
        runner = Runner()
        prompt_rows: list[dict[str, object]] = []

        run_answerability_eval(
            qa_item=six_user_qa(correct="A"),
            packet={"clips": []},
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=prompt_rows,
            call_profile=profile,
        )

        self.assertEqual(runner.calls, [profile, profile])
        self.assertEqual(
            [(row["reasoning_enabled"], row["max_new_tokens"]) for row in prompt_rows],
            [(True, 8192), (True, 8192)],
        )

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

    def test_answerability_uses_full_speaker_then_all_six_videos(self) -> None:
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
                answerable = len(self.calls) == 2
                return json.dumps(
                    sufficiency_evaluation(
                        {
                            "condition_type": (
                                "speaker_only" if len(self.calls) == 1 else "combined_all_six_users"
                            ),
                            "users": ["speaker"] if len(self.calls) == 1 else list(SIX_USERS),
                        },
                        answerable,
                    )
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
        self.assertTrue(result["gate"]["passed"])
        self.assertFalse(result["gate"]["speaker_only_answerable"])
        self.assertTrue(result["gate"]["all_six_answerable"])

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
        qa_item = six_user_qa(correct="A")

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
                    "gate": {
                        "passed": True,
                        "reason": "test",
                        "minimum_required_users": ["provider_two", "provider_four"],
                        "minimum_required_user_count": 2,
                    },
                },
            ),
        ):
            _judge, _answerability, trace = run_parallel_review_judges(
                qa_item=qa_item,
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
        self.assertEqual(
            qa_item["minimum_required_users"],
            ["provider_two", "provider_four"],
        )
        self.assertEqual(qa_item["minimum_required_user_count"], 2)

    def test_production_evidence_uses_one_full_video_call_for_all_segment_counts(self) -> None:
        simple_result = {
            "review_passed": True,
            "checks": {
                "evidence_groundedness": {
                    "status": "PASS",
                    "reason": "supported",
                    "fix": "",
                }
            },
            "blocking_failures": [],
            "feedback_to_generator": "",
            "raw_output": "{}",
            "elapsed_seconds": 0.1,
        }
        for segment_count in (6, 20):
            packet = self.media_packet()
            for clip in packet["clips"]:
                clip["segments"] = [
                    {"time_token": f"segment-{index:02d}"}
                    for index in range(segment_count)
                ]
            prompt_rows: list[dict[str, object]] = []

            with (
                mock.patch.object(
                    video_qa_loop,
                    "run_model_judge_branch",
                    return_value=simple_result,
                ) as simple,
                mock.patch.object(
                    video_qa_loop,
                    "run_chunked_evidence_groundedness_eval",
                    return_value={"chunked_evidence_review": True},
                ) as chunked,
            ):
                result = video_qa_loop.run_evidence_groundedness_review(
                    qa_item=six_user_qa(),
                    packet=packet,
                    runner=object(),
                    prompt_rows=prompt_rows,
                    full_image_paths=[],
                    full_video_paths=[
                        str(clip["full_local_video"])
                        for clip in packet["clips"]
                    ],
                    attempt=1,
                    judge_media_role="full",
                    stage_profiles=video_qa_loop.six_user_ten_minute_fast_profiles(),
                )

            simple.assert_called_once()
            chunked.assert_not_called()
            self.assertEqual(
                result["checks"]["evidence_groundedness"]["status"],
                "PASS",
            )
            self.assertEqual(
                [row["stage"] for row in prompt_rows],
                ["evidence_groundedness_judge"],
            )

    def test_chunked_evidence_uses_six_user_calls_then_text_aggregation(self) -> None:
        self.assertTrue(hasattr(video_qa_loop, "evidence_segment_specs"))
        self.assertTrue(hasattr(video_qa_loop, "run_chunked_evidence_groundedness_eval"))
        packet = six_user_packet()
        for clip in packet["clips"]:
            clip["segments"] = [
                {
                    "time_token": token,
                    "clip_clock": f"20:0{6 + index // 2}:{30 * (index % 2):02d}.00",
                    "video_url": f"https://example.test/{clip['agent_name']}/{token}.mp4",
                }
                for index, token in enumerate((
                    "20060000",
                    "20063000",
                    "20070000",
                    "20073000",
                    "20080000",
                    "20083000",
                ))
            ]
        specs = video_qa_loop.evidence_segment_specs(
            packet,
            [f"/full/{user}.mp4" for user in SIX_USERS],
        )
        self.assertEqual(list(specs), SIX_USERS)
        self.assertTrue(all(len(rows) == 6 for rows in specs.values()))
        self.assertEqual(
            [row["start_seconds"] for row in specs["speaker"]],
            [0.0, 30.0, 60.0, 90.0, 120.0, 150.0],
        )
        self.assertEqual(
            [row["original_time_range"] for row in specs["speaker"]],
            [
                "20:06:00-20:06:30",
                "20:06:30-20:07:00",
                "20:07:00-20:07:30",
                "20:07:30-20:08:00",
                "20:08:00-20:08:30",
                "20:08:30-20:09:00",
            ],
        )

        class Runner:
            model_id = "chunk-test-runner"

            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            def generate(self, prompt, *, image_paths, video_paths):
                self.calls.append(
                    {
                        "prompt": prompt,
                        "image_paths": list(image_paths),
                        "video_paths": list(video_paths),
                    }
                )
                if video_paths:
                    user = SIX_USERS[len(self.calls) - 1]
                    return json.dumps(
                        {
                            "user": user,
                            "segments": [
                                {
                                    "segment_index": index,
                                    "time_token": token,
                                    "claims": [
                                        {
                                            "claim": "the relevant object is visible",
                                            "status": "SUPPORTED",
                                            "confidence": "HIGH",
                                            "visual_description": "A concrete object is visible.",
                                            "original_time_range": f"00:0{index}:00-00:0{index}:30",
                                        }
                                    ],
                                }
                                for index, token in enumerate(
                                    (
                                        "20060000",
                                        "20063000",
                                        "20070000",
                                        "20073000",
                                        "20080000",
                                        "20083000",
                                    )
                                )
                            ],
                            "user_vote": {
                                "visible": True,
                                "confidence": "HIGH",
                                "supported_option": "A",
                                "supporting_segment_indices": [0],
                                "reason": "The relevant object is directly visible.",
                            },
                        }
                    )
                return json.dumps(
                    {
                        "premises_supported": True,
                        "high_confidence_material_conflict": False,
                        "reason": "Every material premise is supported.",
                    }
                )

        runner = Runner()
        prompt_rows: list[dict[str, object]] = []
        segment_paths = {
            user: [f"/chunks/{user}/{index}.mp4" for index in range(6)]
            for user in SIX_USERS
        }

        result = video_qa_loop.run_chunked_evidence_groundedness_eval(
            qa_item={**six_user_qa(), "generation_slot_id": "slot-chunk"},
            packet=packet,
            runner=runner,
            full_video_paths=[f"/full/{user}.mp4" for user in SIX_USERS],
            prompt_rows=prompt_rows,
            attempt=1,
            segment_paths_by_user=segment_paths,
        )

        self.assertEqual(len(runner.calls), 7)
        self.assertEqual([len(call["video_paths"]) for call in runner.calls[:6]], [6] * 6)
        self.assertEqual(runner.calls[-1]["video_paths"], [])
        self.assertNotIn('"raw_output"', runner.calls[-1]["prompt"])
        self.assertEqual(result["checks"]["evidence_groundedness"]["status"], "PASS")
        self.assertTrue(result["vote_summary"]["passed"])
        self.assertEqual(result["vote_summary"]["option_support_counts"]["A"], 6)
        self.assertTrue(result["premise_audit"]["premises_supported"])
        self.assertEqual(len(result["chunk_observations"]), 6)
        self.assertTrue(all("generation_slot_id" in row for row in prompt_rows))
        self.assertTrue(all(row["generation_slot_id"] == "slot-chunk" for row in prompt_rows))
        self.assertEqual(
            [row["stage"] for row in prompt_rows],
            ["evidence_segment_observation"] * 6 + ["evidence_groundedness_aggregation"],
        )

    def test_parallel_review_uses_simple_evidence_for_segmented_packet(self) -> None:
        packet = self.media_packet()
        for clip in packet["clips"]:
            clip["segments"] = [
                {"time_token": f"segment-{index}", "video_url": f"https://example/{index}.mp4"}
                for index in range(6)
            ]
        observed_calls: list[dict[str, object]] = []

        def fake_branch(
            *,
            check_name,
            call_profile=None,
            repair_call_profile=None,
            video_paths,
            **_kwargs,
        ):
            observed_calls.append(
                {
                    "check_name": check_name,
                    "call_profile": call_profile,
                    "repair_call_profile": repair_call_profile,
                    "video_paths": list(video_paths),
                }
            )
            check = {"status": "PASS", "reason": "ok", "fix": ""}
            if check_name == "qa_formality":
                check["semantic_subchecks"] = {
                    name: {"status": "PASS", "reason": "ok"}
                    for name in video_qa_loop.QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                }
            return {
                "review_passed": True,
                "checks": {check_name: check},
                "blocking_failures": [],
                "feedback_to_generator": "",
                "raw_output": "{}",
                "elapsed_seconds": 0.1,
            }

        profiles = six_user_ten_minute_reasoning_profiles()
        with (
            mock.patch.object(video_qa_loop, "run_model_judge_branch", side_effect=fake_branch),
            mock.patch.object(
                video_qa_loop,
                "run_chunked_evidence_groundedness_eval",
            ) as chunked,
            mock.patch.object(
                video_qa_loop,
                "run_answerability_eval",
                return_value={"evaluations": [], "gate": {"passed": True, "reason": "test"}},
            ) as answerability,
        ):
            _judge, _answerability, trace = run_parallel_review_judges(
                qa_item=six_user_qa(),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=(prompt_rows := []),
                full_image_paths=[],
                full_video_paths=[clip["full_local_video"] for clip in packet["clips"]],
                attempt=1,
                stage_profiles=profiles,
        )

        chunked.assert_not_called()
        self.assertEqual(
            [row["check_name"] for row in observed_calls],
            ["qa_formality", "evidence_groundedness"],
        )
        self.assertIs(
            observed_calls[0]["call_profile"],
            profiles["qa_formality"],
        )
        self.assertIs(
            observed_calls[1]["call_profile"],
            profiles["evidence_groundedness_aggregation"],
        )
        self.assertIs(
            answerability.call_args.kwargs["call_profile"],
            profiles["answerability"],
        )
        self.assertFalse(trace["evidence_groundedness"]["chunked_evidence_review"])
        self.assertEqual(trace["evidence_groundedness"]["chunk_observations"], [])
        self.assertEqual(
            [row["stage"] for row in prompt_rows if row["stage"] == "evidence_groundedness_judge"],
            ["evidence_groundedness_judge"],
        )

    def test_evidence_segment_materialization_uses_ffmpeg_cacheable_mp4_temporaries(self) -> None:
        packet = six_user_packet()
        for clip in packet["clips"]:
            clip["segments"] = [
                {"time_token": token, "video_url": f"https://example/{index}.mp4"}
                for index, token in enumerate(
                    (
                        "20060000",
                        "20063000",
                        "20070000",
                        "20073000",
                        "20080000",
                        "20083000",
                    )
                )
            ]
        full_paths = []
        for index in range(6):
            path = self.tmp_path / f"full-segment-source-{index}.mp4"
            path.write_bytes(b"full")
            full_paths.append(str(path))
        specs = video_qa_loop.evidence_segment_specs(packet, full_paths)
        commands = []

        def fake_runner(command, **_kwargs):
            commands.append(list(command))
            Path(command[-1]).write_bytes(b"chunk")

        cache = self.tmp_path / "evidence-cache"
        first = video_qa_loop.materialize_evidence_segment_paths(
            packet,
            specs,
            cache_dir=cache,
            ffmpeg_binary="ffmpeg-test",
            command_runner=fake_runner,
        )
        second = video_qa_loop.materialize_evidence_segment_paths(
            packet,
            specs,
            cache_dir=cache,
            ffmpeg_binary="ffmpeg-test",
            command_runner=fake_runner,
        )

        self.assertEqual(len(commands), 36)
        self.assertTrue(all(command[-1].endswith(".part.mp4") for command in commands))
        self.assertEqual(first, second)
        self.assertTrue(all(Path(path).is_file() for paths in first.values() for path in paths))

    def test_two_user_condition_contract_is_unchanged(self) -> None:
        conditions = build_answerability_conditions(["speaker", "provider"])

        self.assertEqual(len(conditions), 3)
        self.assertEqual(
            [condition["condition_type"] for condition in conditions],
            ["single_user", "single_user", "combined_all_users"],
        )


if __name__ == "__main__":
    unittest.main()
