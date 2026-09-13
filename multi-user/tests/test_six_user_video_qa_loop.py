from __future__ import annotations

import json
import shutil
import sys
import threading
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

from egolife_two_user_qa.group_relative_clip_sampling import (  # noqa: E402
    build_candidate_packet,
)
from egolife_two_user_qa.prompts import (  # noqa: E402
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
)
from egolife_two_user_qa.video_qa_loop import (  # noqa: E402
    SIX_USER_JUDGE_MODE_LEGACY,
    SIX_USER_JUDGE_MODE_SEQUENTIAL,
    JudgeInfrastructureError,
    answerability_gate,
    build_answerability_conditions,
    compact_prompt_record,
    complete_generator_metadata,
    condition_media_for_clips,
    human_audit_packet,
    intermediate_checkpoint_row,
    media_for_clips,
    merge_parallel_judges,
    run_combined_direct_judge,
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


def evaluation(condition: dict[str, object], answerable: object) -> dict[str, object]:
    users = list(condition.get("users") or [])
    source_user = users[-1] if users else "speaker"
    if answerable is None:
        audits = []
    else:
        audits = [
            {
                "fact_id": "F1",
                "visibility": "VISIBLE",
                "source_users": [users[0]],
                "segment_references": ["segment_001"],
                "visual_description": "The question anchor is visible.",
            },
            {
                "fact_id": "F2",
                "visibility": "VISIBLE" if answerable else "NOT_VISIBLE",
                "source_users": [source_user] if answerable else [],
                "segment_references": ["segment_002"] if answerable else [],
                "visual_description": (
                    "The answer-bearing detail is visible."
                    if answerable
                    else "The answer-bearing detail is absent."
                ),
            },
        ]
    return {
        **condition,
        "reason": "the frozen visual facts are available or missing",
        "fact_audits": audits,
        "shared_fact_ids": ["F1", "F2"],
    }


def six_user_packet() -> dict[str, object]:
    roles = ["speaker_all_clustering_frames", *(["provider_retained_cluster_frames"] * 5)]
    return build_candidate_packet(
        {
            "day": "DAY1",
            "time_token": "12000000",
            "selected_clips": [
                {
                    "agent_name": user,
                    "generator_media_mode": (
                        "all_clustering_frames_only"
                        if index == 0
                        else "retained_cluster_frames_only"
                    ),
                    "force_frame_inputs": True,
                    "media_role": role,
                    "is_pruned": index != 0,
                }
                for index, (user, role) in enumerate(zip(SIX_USERS, roles))
            ],
            "selection": {
                "method": "six_user_speaker_consensus",
                "speaker_index": 0,
            },
            "speaker_consensus_pruning": {},
            "speaker_attempts": [],
        }
    )


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
            full_video = self.tmp_path / f"full_{index}.mp4"
            full_video.write_bytes(b"full")
            frames = []
            for frame_index in range(3 if index == 0 else 2):
                frame_path = self.tmp_path / f"generator_{index}_{frame_index}.jpg"
                frame_path.write_bytes(b"frame")
                frames.append(
                    {
                        "path": str(frame_path),
                        "timestamp_seconds": float(frame_index),
                    }
                )
            source_segments = []
            for segment_index in range(20):
                segment_path = self.tmp_path / f"source_{index}_{segment_index:03d}.mp4"
                segment_path.write_bytes(b"segment")
                source_segments.append(
                    {
                        "segment_index": segment_index,
                        "local_video": str(segment_path),
                        "time_token": f"SECRET_TIME_{segment_index}",
                        "window_start_seconds": float(segment_index * 30),
                        "window_end_seconds": float((segment_index + 1) * 30),
                    }
                )
            clips.append(
                {
                    **clip,
                    "full_local_video": str(full_video),
                    "original_local_video": str(full_video),
                    "frames": frames,
                    "source_segments": source_segments,
                    "segment_count": len(source_segments),
                    "duration_seconds": 10.0,
                    "is_pruned": index != 0,
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
            qa["media_roles"],
            {
                SIX_USERS[0]: "speaker_all_clustering_frames",
                **{
                    user: "provider_retained_cluster_frames"
                    for user in SIX_USERS[1:]
                },
            },
        )

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
        for provider in SIX_USERS[1:]:
            self.assertIn("not preclassified", qa["single_user_answerability"][provider])
        self.assertIn("one or more provider views", qa["generator_rationale"])
        self.assertIn("naturally want to ask", qa["review"]["generator_self_check"])
        self.assertEqual(validate_qa_item(qa), [])

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

    def test_six_user_gate_passes_when_speaker_is_insufficient_and_all_six_is_sufficient(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            qa,
            [evaluation(conditions[0], False), evaluation(conditions[1], True)],
        )

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["answerability_mode"], "shared_fact_visibility_audit")
        self.assertFalse(gate["speaker_only_answerable"])
        self.assertTrue(gate["all_six_answerable"])
        self.assertEqual(gate["answerability_evaluated_condition_count"], 2)
        self.assertEqual(gate["shared_fact_ids"], ["F1", "F2"])
        self.assertEqual(gate["provider_resolved_fact_ids"], ["F2"])
        self.assertNotIn("speaker_only_choice", gate)
        self.assertNotIn("all_six_choice", gate)

    def test_speaker_only_answerable_is_blocking(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)
        gate = answerability_gate(
            qa,
            [evaluation(conditions[0], True), evaluation(conditions[1], True)],
        )

        self.assertFalse(gate["passed"])
        self.assertTrue(gate["speaker_only_answerable"])
        self.assertEqual(gate["failure_label"], "speaker_only_answerable")
        self.assertIn("speaker-only", gate["reason"])

    def test_unparsed_speaker_sufficiency_is_blocking(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)

        speaker_invalid = answerability_gate(
            qa,
            [evaluation(conditions[0], None), evaluation(conditions[1], True)],
        )

        self.assertFalse(speaker_invalid["passed"])
        self.assertEqual(speaker_invalid["failure_label"], "speaker_only_unparsed")

    def test_all_six_insufficient_missing_or_unparsed_is_blocking(self) -> None:
        qa = six_user_qa(correct="A")
        conditions = build_answerability_conditions(SIX_USERS)

        insufficient = answerability_gate(
            qa,
            [evaluation(conditions[0], False), evaluation(conditions[1], False)],
        )
        missing = answerability_gate(qa, [evaluation(conditions[0], False)])
        unparsed = answerability_gate(
            qa,
            [evaluation(conditions[0], False), evaluation(conditions[1], None)],
        )

        self.assertEqual(insufficient["failure_label"], "all_six_not_answerable")
        self.assertEqual(missing["failure_label"], "all_six_missing")
        self.assertEqual(unparsed["failure_label"], "all_six_unparsed")

    def test_six_user_gate_does_not_read_the_gold_answer_or_accept_answer_fields(self) -> None:
        qa = six_user_qa(correct="A")
        qa.pop("correct")
        conditions = build_answerability_conditions(SIX_USERS)

        gate = answerability_gate(
            qa,
            [evaluation(conditions[0], False), evaluation(conditions[1], True)],
        )
        leaked = evaluation(conditions[0], False)
        leaked["choice"] = "B"
        leaked_gate = answerability_gate(
            qa,
            [leaked, evaluation(conditions[1], True)],
        )

        self.assertTrue(gate["passed"])
        self.assertFalse(leaked_gate["passed"])
        self.assertEqual(leaked_gate["failure_label"], "speaker_only_unparsed")
        self.assertIn("forbidden answer fields", leaked_gate["reason"])

    def test_run_eval_uses_shared_plan_six_user_maps_and_two_reductions(self) -> None:
        class Runner:
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
                if "Stage marker: answerability_fact_plan" in prompt:
                    return json.dumps(
                        {
                            "reason": "Two facts are required.",
                            "needed_facts": [
                                {"fact_id": "F1", "fact": "The setup is visible.", "why_needed": "It anchors the question."},
                                {"fact_id": "F2", "fact": "The completing item is visible.", "why_needed": "It resolves the detail."},
                            ],
                        }
                    )
                if "Stage marker: answerability_user_fact_audit" in prompt:
                    user = next(user for user in SIX_USERS if f'user "{user}"' in prompt)
                    second_visible = user == "provider_one"
                    return json.dumps(
                        {
                            "reason": "This user establishes some frozen facts.",
                            "fact_audits": [
                                {
                                    "fact_id": "F1",
                                    "visibility": "VISIBLE" if user == "speaker" else "NOT_VISIBLE",
                                    "source_users": [user] if user == "speaker" else [],
                                    "segment_references": ["segment_001"] if user == "speaker" else [],
                                    "visual_description": "The anchor is visible or absent.",
                                },
                                {
                                    "fact_id": "F2",
                                    "visibility": "VISIBLE" if second_visible else "NOT_VISIBLE",
                                    "source_users": [user] if second_visible else [],
                                    "segment_references": ["segment_020"] if second_visible else [],
                                    "visual_description": "The detail is visible or absent.",
                                },
                            ],
                        }
                    )
                all_six = '"condition_type": "combined_all_six_users"' in prompt
                return json.dumps(
                    {
                        "reason": "The included audits were combined.",
                        "fact_audits": [
                            {
                                "fact_id": "F1",
                                "visibility": "VISIBLE",
                                "source_users": ["speaker"],
                                "segment_references": ["segment_001"],
                                "visual_description": "The anchor is visible.",
                            },
                            {
                                "fact_id": "F2",
                                "visibility": "VISIBLE" if all_six else "NOT_VISIBLE",
                                "source_users": ["provider_one"] if all_six else [],
                                "segment_references": ["segment_020"] if all_six else [],
                                "visual_description": "The detail is visible only with a provider.",
                            },
                        ],
                    }
                )

        runner = Runner()
        result = run_answerability_eval(
            qa_item=six_user_qa(correct="A"),
            packet=self.media_packet(),
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=[],
        )

        self.assertEqual(len(runner.calls), 9)
        self.assertEqual(
            [len(call["video_paths"]) for call in runner.calls],
            [0, 20, 20, 20, 20, 20, 20, 0, 0],
        )
        self.assertEqual(len(result["evaluations"]), 2)
        self.assertTrue(result["gate"]["passed"])

    def test_sequential_fact_audit_runs_speaker_gate_before_other_users(self) -> None:
        class Runner:
            supports_concurrent_batching = True

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
                if "Stage marker: answerability_fact_plan" in prompt:
                    return json.dumps(
                        {
                            "reason": "Two facts are required.",
                            "needed_facts": [
                                {
                                    "fact_id": "F1",
                                    "fact": "The setup is visible.",
                                    "why_needed": "It anchors the question.",
                                },
                                {
                                    "fact_id": "F2",
                                    "fact": "The completing item is visible.",
                                    "why_needed": "It resolves the detail.",
                                },
                            ],
                        }
                    )
                if "Stage marker: answerability_user_fact_audit" in prompt:
                    user = next(user for user in SIX_USERS if f'user "{user}"' in prompt)
                    second_visible = user == "provider_one"
                    return json.dumps(
                        {
                            "reason": "This user establishes some frozen facts.",
                            "fact_audits": [
                                {
                                    "fact_id": "F1",
                                    "visibility": "VISIBLE" if user == "speaker" else "NOT_VISIBLE",
                                    "source_users": [user] if user == "speaker" else [],
                                    "segment_references": ["segment_001"] if user == "speaker" else [],
                                    "visual_description": "The anchor is visible or absent.",
                                },
                                {
                                    "fact_id": "F2",
                                    "visibility": "VISIBLE" if second_visible else "NOT_VISIBLE",
                                    "source_users": [user] if second_visible else [],
                                    "segment_references": ["segment_020"] if second_visible else [],
                                    "visual_description": "The detail is visible or absent.",
                                },
                            ],
                        }
                    )
                answerable_condition = any(
                    marker in prompt
                    for marker in (
                        '"condition_type": "combined_all_six_users"',
                        '"condition_type": "minimum_required_users"',
                    )
                )
                return json.dumps(
                    {
                        "reason": "The included audits were combined.",
                        "fact_audits": [
                            {
                                "fact_id": "F1",
                                "visibility": "VISIBLE",
                                "source_users": ["speaker"],
                                "segment_references": ["segment_001"],
                                "visual_description": "The anchor is visible.",
                            },
                            {
                                "fact_id": "F2",
                                "visibility": (
                                    "VISIBLE" if answerable_condition else "NOT_VISIBLE"
                                ),
                                "source_users": (
                                    ["provider_one"] if answerable_condition else []
                                ),
                                "segment_references": (
                                    ["segment_020"] if answerable_condition else []
                                ),
                                "visual_description": "The detail is visible only with a provider.",
                            },
                        ],
                    }
                )

        runner = Runner()
        result = run_answerability_eval(
            qa_item=six_user_qa(correct="A"),
            packet=self.media_packet(),
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=[],
            six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
        )

        self.assertEqual(
            [len(call["video_paths"]) for call in runner.calls],
            [0, 20, 0, 20, 20, 20, 20, 20, 0, 0],
        )
        self.assertEqual(len(result["evaluations"]), 3)
        self.assertIsNone(result["early_exit"])
        self.assertTrue(result["gate"]["passed"], result["gate"])
        self.assertEqual(
            result["trial_order"],
            ["combined_all_six_users", "minimum_required_users"],
        )
        self.assertEqual(
            result["minimum_required_users"],
            ["speaker", "provider_one"],
        )
        self.assertEqual(
            result["gate"]["minimum_required_users"],
            ["speaker", "provider_one"],
        )
        self.assertEqual(
            result["remaining_user_audit_execution"],
            "concurrent_vllm_batch",
        )
        self.assertEqual(
            result["condition_aggregation_execution"],
            "concurrent_vllm_batch",
        )

    def test_sequential_fact_audit_stops_when_speaker_is_sufficient(self) -> None:
        class Runner:
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
                if "Stage marker: answerability_fact_plan" in prompt:
                    return json.dumps(
                        {
                            "reason": "One fact is required.",
                            "needed_facts": [
                                {
                                    "fact_id": "F1",
                                    "fact": "The answer detail is visible.",
                                    "why_needed": "It resolves the question.",
                                }
                            ],
                        }
                    )
                return json.dumps(
                    {
                        "reason": "The speaker already sees the fact.",
                        "fact_audits": [
                            {
                                "fact_id": "F1",
                                "visibility": "VISIBLE",
                                "source_users": ["speaker"],
                                "segment_references": ["segment_001"],
                                "visual_description": "The answer detail is visible.",
                            }
                        ],
                    }
                )

        runner = Runner()
        result = run_answerability_eval(
            qa_item=six_user_qa(correct="A"),
            packet=self.media_packet(),
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=[],
            six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
        )

        self.assertEqual(
            [len(call["video_paths"]) for call in runner.calls],
            [0, 20, 0],
        )
        self.assertEqual(result["early_exit"], "speaker_only")
        self.assertFalse(result["gate"]["passed"])
        self.assertEqual(result["gate"]["failure_label"], "speaker_only_answerable")

    def test_legacy_zero_shot_answerability_uses_one_and_six_full_videos(self) -> None:
        class Runner:
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
                speaker_only = '"condition_type": "speaker_only"' in prompt
                return json.dumps(
                    {
                        "answerable": not speaker_only,
                        "reason": "The needed visible facts are absent or present.",
                        "available_evidence": [] if speaker_only else ["answer fact"],
                        "missing_evidence": ["answer fact"] if speaker_only else [],
                    }
                )

        runner = Runner()
        prompt_rows: list[dict[str, object]] = []
        result = run_answerability_eval(
            qa_item=six_user_qa(correct="A"),
            packet=self.media_packet(),
            runner=runner,
            media_backend="transformers-local",
            allow_openai_video_input=False,
            prompt_rows=prompt_rows,
            six_user_judge_mode=SIX_USER_JUDGE_MODE_LEGACY,
        )

        self.assertEqual(len(runner.calls), 2)
        self.assertEqual([len(call["video_paths"]) for call in runner.calls], [1, 6])
        self.assertEqual([row["stage"] for row in prompt_rows], ["answerability"] * 2)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(
            result["gate"]["answerability_mode"],
            "direct_video_sufficiency_zero_shot",
        )

    def test_legacy_zero_shot_parallel_review_uses_direct_six_video_groundedness(self) -> None:
        packet = self.media_packet()
        _images, full_videos = media_for_clips(
            packet["clips"],
            backend="transformers-local",
            allow_openai_video_input=False,
            media_role="full",
        )
        prompt_rows: list[dict[str, object]] = []

        def fake_judge_branch(*, check_name, video_paths, **kwargs):
            return {
                "checks": {
                    check_name: {"status": "PASS", "reason": "ok", "fix": ""}
                },
                "blocking_failures": [],
                "feedback_to_generator": "",
                "raw_output": "{}",
                "elapsed_seconds": 0.01,
                "seen_video_count": len(video_paths),
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
            ) as answerability_mock,
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
                six_user_judge_mode=SIX_USER_JUDGE_MODE_LEGACY,
            )

        stages = [row["stage"] for row in prompt_rows]
        self.assertEqual(stages.count("evidence_groundedness_judge"), 1)
        self.assertNotIn("evidence_segment_observation", stages)
        self.assertNotIn("evidence_groundedness_aggregation", stages)
        groundedness_row = next(
            row for row in prompt_rows if row["stage"] == "evidence_groundedness_judge"
        )
        self.assertEqual(groundedness_row["media_summary"]["video_count"], 6)
        self.assertEqual(trace["evidence_groundedness"]["mode"], "single_full_media_call")
        self.assertEqual(trace["six_user_judge_mode"], SIX_USER_JUDGE_MODE_LEGACY)
        self.assertEqual(
            answerability_mock.call_args.kwargs["six_user_judge_mode"],
            SIX_USER_JUDGE_MODE_LEGACY,
        )

    def test_sequential_separated_review_runs_strictly_in_gate_order(self) -> None:
        packet = self.media_packet()
        generator_images = ["speaker-frame.jpg", "provider-frame.jpg"]
        _images, full_videos = media_for_clips(
            packet["clips"],
            backend="transformers-local",
            allow_openai_video_input=False,
            media_role="full",
        )
        events: list[str] = []
        formality_judge = {
            "review_passed": True,
            "checks": {
                "qa_formality": {
                    "status": "PASS",
                    "reason": "form is valid",
                    "fix": "",
                    "semantic_subchecks": {
                        name: {"status": "PASS", "reason": "ok"}
                        for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                    },
                },
            },
            "blocking_failures": [],
            "why_generator_asked_this": "",
            "feedback_to_generator": "",
            "elapsed_seconds": 0.01,
        }
        evidence_judge = {
            "review_passed": True,
            "checks": {
                "evidence_groundedness": {
                    "status": "PASS",
                    "reason": "sampled evidence is grounded",
                    "fix": "",
                },
            },
            "blocking_failures": [],
            "why_generator_asked_this": "",
            "feedback_to_generator": "",
            "elapsed_seconds": 0.01,
        }

        def fake_judge_branch(*, check_name, image_paths, video_paths, **_kwargs):
            events.append(check_name)
            if check_name == "qa_formality":
                self.assertEqual(image_paths, [])
                self.assertEqual(video_paths, [])
                return formality_judge
            self.assertEqual(check_name, "evidence_groundedness")
            self.assertEqual(image_paths, generator_images)
            self.assertEqual(video_paths, [])
            return evidence_judge

        def fake_answerability(**kwargs):
            events.append("answerability")
            self.assertEqual(
                kwargs["six_user_judge_mode"],
                SIX_USER_JUDGE_MODE_SEQUENTIAL,
            )
            return {
                "evaluations": [],
                "gate": {"passed": True, "reason": "fact audit passed"},
            }

        prompt_rows: list[dict[str, object]] = []
        with (
            mock.patch.object(
                video_qa_loop,
                "run_model_judge_branch",
                side_effect=fake_judge_branch,
            ),
            mock.patch.object(
                video_qa_loop,
                "run_answerability_eval",
                side_effect=fake_answerability,
            ),
        ):
            judge, _answerability, trace = run_parallel_review_judges(
                qa_item=six_user_qa(correct="A"),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=prompt_rows,
                full_image_paths=[],
                full_video_paths=full_videos,
                generator_image_paths=generator_images,
                generator_video_paths=[],
                attempt=1,
                six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
            )

        self.assertEqual(
            events,
            ["qa_formality", "evidence_groundedness", "answerability"],
        )
        self.assertTrue(judge["gate"]["passed"])
        self.assertFalse(trace["parallel"])
        self.assertEqual(
            [row["stage"] for row in prompt_rows],
            ["qa_formality_judge", "evidence_groundedness_judge"],
        )
        self.assertEqual(prompt_rows[0]["media_summary"]["image_count"], 0)
        self.assertEqual(prompt_rows[0]["media_summary"]["video_count"], 0)
        self.assertEqual(prompt_rows[1]["media_summary"]["image_count"], len(generator_images))
        self.assertEqual(prompt_rows[1]["media_summary"]["video_count"], 0)
        self.assertEqual(
            trace["execution_order"][:3],
            [
                "deterministic_schema",
                "qa_formality_judge",
                "evidence_groundedness_judge",
            ],
        )

    def test_vllm_sequential_fact_mode_runs_all_three_review_branches_together(self) -> None:
        class BatchingRunner:
            supports_concurrent_batching = True
            model_id = "fake-vllm"

        barrier = threading.Barrier(3)
        entered: list[str] = []

        formality_judge = {
            "review_passed": True,
            "checks": {
                "qa_formality": {
                    "status": "PASS",
                    "reason": "form is valid",
                    "fix": "",
                    "semantic_subchecks": {
                        name: {"status": "PASS", "reason": "ok"}
                        for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                    },
                }
            },
            "blocking_failures": [],
            "feedback_to_generator": "",
            "elapsed_seconds": 0.01,
        }
        groundedness_judge = {
            "review_passed": False,
            "checks": {
                "evidence_groundedness": {
                    "status": "FAIL",
                    "reason": "the evidence is not grounded",
                    "fix": "regenerate",
                }
            },
            "blocking_failures": ["evidence_groundedness"],
            "feedback_to_generator": "regenerate",
            "elapsed_seconds": 0.01,
        }

        def fake_judge_branch(*, check_name, **_kwargs):
            entered.append(check_name)
            barrier.wait(timeout=2)
            return (
                formality_judge
                if check_name == "qa_formality"
                else groundedness_judge
            )

        def fake_answerability(**kwargs):
            entered.append("answerability")
            self.assertEqual(
                kwargs["six_user_judge_mode"],
                SIX_USER_JUDGE_MODE_SEQUENTIAL,
            )
            barrier.wait(timeout=2)
            return {
                "evaluations": [],
                "gate": {"passed": True, "reason": "fact audit passed"},
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
                side_effect=fake_answerability,
            ),
        ):
            judge, answerability, trace = run_parallel_review_judges(
                qa_item=six_user_qa(correct="A"),
                packet=self.media_packet(),
                schema_errors=[],
                runner=BatchingRunner(),
                media_backend="vllm-local",
                allow_openai_video_input=False,
                prompt_rows=[],
                full_image_paths=[],
                full_video_paths=[],
                generator_image_paths=["sampled.jpg"],
                generator_video_paths=[],
                attempt=1,
                six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
            )

        self.assertCountEqual(
            entered,
            ["qa_formality", "evidence_groundedness", "answerability"],
        )
        self.assertTrue(answerability["gate"]["passed"])
        self.assertFalse(judge["gate"]["passed"])
        self.assertTrue(trace["parallel"])
        self.assertTrue(trace["single_packet_batching"])
        self.assertEqual(
            trace["evidence_groundedness"]["mode"],
            "single_generator_media_call",
        )

    def test_combined_direct_gate_does_not_require_downstream_answerability(self) -> None:
        class Runner:
            def generate(self, _prompt, *, image_paths, video_paths):
                self.assert_media = (list(image_paths), list(video_paths))
                return json.dumps(
                    {
                        "review_passed": False,
                        "checks": {
                            "qa_formality": {
                                "status": "PASS",
                                "reason": "form is valid",
                                "fix": "",
                                "semantic_subchecks": {
                                    name: {"status": "PASS", "reason": "ok"}
                                    for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                                },
                            },
                            "evidence_groundedness": {
                                "status": "PASS",
                                "reason": "evidence is grounded",
                                "fix": "",
                            },
                        },
                        "blocking_failures": ["answerability"],
                        "why_generator_asked_this": "",
                        "feedback_to_generator": "",
                    }
                )

        runner = Runner()
        result = run_combined_direct_judge(
            prompt="review",
            runner=runner,
            image_paths=["sampled.jpg"],
            video_paths=[],
            evidence_id="evidence",
            qa_id="qa",
            attempt=1,
        )

        self.assertTrue(result["gate"]["passed"], result["gate"])
        self.assertEqual(result["gate"]["failed_checks"], [])
        self.assertIn("answerability", result["gate"]["model_blocking_failures"])
        self.assertNotIn("missing checks", result["gate"]["reason"])

    def test_sequential_separated_review_stops_after_grounding_failure(self) -> None:
        packet = self.media_packet()
        formality_judge = {
            "review_passed": True,
            "checks": {
                "qa_formality": {
                    "status": "PASS",
                    "reason": "form is valid",
                    "fix": "",
                    "semantic_subchecks": {
                        name: {"status": "PASS", "reason": "ok"}
                        for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                    },
                },
            },
            "blocking_failures": [],
            "feedback_to_generator": "",
        }
        evidence_judge = {
            "review_passed": False,
            "checks": {
                "evidence_groundedness": {
                    "status": "FAIL",
                    "reason": "answer claim is not visible",
                    "fix": "regenerate from visible evidence",
                },
            },
            "blocking_failures": ["evidence_groundedness"],
            "feedback_to_generator": "regenerate from visible evidence",
        }

        def fake_judge_branch(*, check_name, **_kwargs):
            return (
                formality_judge
                if check_name == "qa_formality"
                else evidence_judge
            )

        with (
            mock.patch.object(
                video_qa_loop,
                "run_model_judge_branch",
                side_effect=fake_judge_branch,
            ),
            mock.patch.object(
                video_qa_loop,
                "run_answerability_eval",
            ) as answerability_mock,
        ):
            judge, answerability, trace = run_parallel_review_judges(
                qa_item=six_user_qa(correct="A"),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=[],
                full_image_paths=[],
                full_video_paths=[],
                generator_image_paths=["sampled.jpg"],
                generator_video_paths=[],
                attempt=1,
                six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
            )

        answerability_mock.assert_not_called()
        self.assertFalse(judge["gate"]["passed"])
        self.assertEqual(
            answerability["gate"]["failure_label"],
            "upstream_evidence_groundedness_failed",
        )
        self.assertEqual(
            trace["execution_order"],
            [
                "deterministic_schema",
                "qa_formality_judge",
                "evidence_groundedness_judge",
            ],
        )

    def test_sequential_separated_review_does_not_start_grounding_after_formality_failure(self) -> None:
        packet = self.media_packet()
        formality_judge = {
            "review_passed": False,
            "checks": {
                "qa_formality": {
                    "status": "FAIL",
                    "reason": "question wording is ambiguous",
                    "fix": "rewrite the question",
                    "semantic_subchecks": {
                        name: {
                            "status": "FAIL" if name == "reference_clarity" else "PASS",
                            "reason": "ambiguous" if name == "reference_clarity" else "ok",
                        }
                        for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
                    },
                }
            },
            "blocking_failures": ["qa_formality"],
            "feedback_to_generator": "rewrite the question",
        }

        with (
            mock.patch.object(
                video_qa_loop,
                "run_model_judge_branch",
                return_value=formality_judge,
            ) as judge_branch_mock,
            mock.patch.object(
                video_qa_loop,
                "run_answerability_eval",
            ) as answerability_mock,
        ):
            judge, answerability, trace = run_parallel_review_judges(
                qa_item=six_user_qa(correct="A"),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=[],
                full_image_paths=[],
                full_video_paths=[],
                generator_image_paths=["sampled.jpg"],
                generator_video_paths=[],
                attempt=1,
                six_user_judge_mode=SIX_USER_JUDGE_MODE_SEQUENTIAL,
            )

        self.assertEqual(judge_branch_mock.call_count, 1)
        self.assertEqual(
            judge_branch_mock.call_args.kwargs["check_name"],
            "qa_formality",
        )
        answerability_mock.assert_not_called()
        self.assertFalse(judge["gate"]["passed"])
        self.assertEqual(
            answerability["gate"]["failure_label"],
            "upstream_qa_formality_failed",
        )
        self.assertEqual(
            trace["execution_order"],
            ["deterministic_schema", "qa_formality_judge"],
        )

    def test_cuda_oom_in_legacy_judge_is_infrastructure_failure(self) -> None:
        packet = self.media_packet()
        _images, full_videos = media_for_clips(
            packet["clips"],
            backend="transformers-local",
            allow_openai_video_input=False,
            media_role="full",
        )

        def fake_judge_branch(*, check_name, **kwargs):
            if check_name == "evidence_groundedness":
                raise RuntimeError("CUDA out of memory while allocating tensor")
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
                return_value={"evaluations": [], "gate": {"passed": True}},
            ),
            self.assertRaises(JudgeInfrastructureError) as raised,
        ):
            run_parallel_review_judges(
                qa_item=six_user_qa(correct="A"),
                packet=packet,
                schema_errors=[],
                runner=object(),
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=[],
                full_image_paths=[],
                full_video_paths=full_videos,
                attempt=1,
                six_user_judge_mode=SIX_USER_JUDGE_MODE_LEGACY,
            )

        self.assertEqual(raised.exception.stage, "evidence_groundedness_judge")

    def test_six_user_media_routes_generator_and_judges_in_order(self) -> None:
        packet = self.media_packet()
        clips = packet["clips"]

        generator_images, generator_videos = media_for_clips(
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

        self.assertEqual(
            generator_images,
            [frame["path"] for clip in clips for frame in clip["frames"]],
        )
        self.assertEqual(generator_videos, [])
        self.assertEqual(full_videos, [clip["full_local_video"] for clip in clips])
        self.assertEqual(
            [row["media_role"] for row in video_evidence_for_packet(packet)],
            [
                "speaker_all_clustering_frames",
                "provider_retained_cluster_frames",
                "provider_retained_cluster_frames",
                "provider_retained_cluster_frames",
                "provider_retained_cluster_frames",
                "provider_retained_cluster_frames",
            ],
        )
        self.assertFalse(clips[0]["is_pruned"])
        self.assertTrue(all(clip["is_pruned"] for clip in clips[1:]))
        self.assertTrue(all("local_video" not in clip for clip in clips))
        self.assertTrue(all("generator_local_video" not in clip for clip in clips))

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

    def test_intermediate_checkpoint_and_prompt_log_omit_frame_manifests(self) -> None:
        frame_paths = [
            f"/scratch/private/frame_mapping/frame_{index:05d}.png"
            for index in range(6000)
        ]
        condition_media = {
            "condition_id": "combined_all_six_users::all",
            "condition_type": "combined_all_six_users",
            "users": SIX_USERS,
            "media_role": "full",
            "image_paths": frame_paths,
            "video_paths": [f"/scratch/private/user_{index}.mp4" for index in range(6)],
            "total_duration_seconds": 3600.0,
            "video_evidence": [
                {
                    "sampled_frames": [
                        {"path": path, "timestamp_seconds": float(index)}
                        for index, path in enumerate(frame_paths)
                    ]
                }
            ],
        }
        answerability = {
            "evaluations": [
                {
                    "condition_id": condition_media["condition_id"],
                    "answerable": True,
                    "raw_output": '{"answerable": true}',
                    "condition_media": condition_media,
                }
            ],
            "gate": {"passed": True},
        }
        attempt = {
            "evidence_id": "large-packet",
            "qa_id": "qa-large",
            "question_type": "neutral",
            "generation_mode": "baseline",
            "attempt": 1,
            "media": {
                "image_paths": frame_paths,
                "video_paths": [],
                "judge_image_paths": [],
                "judge_video_paths": condition_media["video_paths"],
                "human_audit": {"video_evidence": condition_media["video_evidence"]},
            },
            "generation": {
                "prompt": "small prompt",
                "raw_output": '{"question": "What was moved?"}',
            },
            "judge": {
                "qa_formality": {
                    "prompt": "judge prompt",
                    "raw_output": "x" * 1_000_000,
                    "parsed": {"status": "PASS", "raw_output": "y" * 1_000_000},
                },
                "answerability": answerability,
                "merged": {
                    "gate": {"passed": True},
                    "branches": {"qa_formality": {"raw_output": "z" * 1_000_000}},
                },
            },
            "answerability": answerability,
            "result": {"accepted": True},
        }
        qa = {
            **six_user_qa(),
            "generation_trace": [attempt],
            "human_audit": {"video_evidence": condition_media["video_evidence"]},
            "video_evidence": condition_media["video_evidence"],
            "review": {"answerability": answerability},
        }

        checkpoint = intermediate_checkpoint_row(
            evidence_id="large-packet",
            qa_id="qa-large",
            question_type="neutral",
            generation_mode="baseline",
            status="accepted",
            attempts=[attempt],
            qa=qa,
        )
        prompt_record = compact_prompt_record(
            {
                "stage": "generation",
                "evidence_id": "large-packet",
                "prompt": "small prompt",
                "image_paths": frame_paths,
                "video_paths": [],
                "condition_media": condition_media,
            }
        )
        serialized_checkpoint = json.dumps(checkpoint)
        serialized_prompt_record = json.dumps(prompt_record)

        self.assertLess(len(serialized_checkpoint), 50_000)
        self.assertNotIn(frame_paths[0], serialized_checkpoint)
        self.assertNotIn(frame_paths[-1], serialized_checkpoint)
        self.assertNotIn("sampled_frames", serialized_checkpoint)
        self.assertNotIn('"raw_output"', serialized_checkpoint)
        self.assertNotIn("x" * 1000, serialized_checkpoint)
        self.assertNotIn("image_paths", prompt_record)
        self.assertNotIn("video_paths", prompt_record)
        self.assertNotIn("condition_media", prompt_record)
        self.assertNotIn(frame_paths[0], serialized_prompt_record)
        self.assertEqual(prompt_record["media_summary"]["image_count"], 6000)

    def test_answerability_reuses_cached_source_segments_without_media_mappings(self) -> None:
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
                if "Stage marker: answerability_fact_plan" in prompt:
                    return json.dumps(
                        {
                            "reason": "One external fact is needed.",
                            "needed_facts": [
                                {"fact_id": "F1", "fact": "The completing item is visible.", "why_needed": "It resolves the question."}
                            ],
                        }
                    )
                if "Stage marker: answerability_user_fact_audit" in prompt:
                    user = next(user for user in SIX_USERS if f'user "{user}"' in prompt)
                    visible = user == "provider_one"
                    return json.dumps(
                        {
                            "reason": "The fact is visible or absent.",
                            "fact_audits": [
                                {
                                    "fact_id": "F1",
                                    "visibility": "VISIBLE" if visible else "NOT_VISIBLE",
                                    "source_users": [user] if visible else [],
                                    "segment_references": ["segment_019"] if visible else [],
                                    "visual_description": "The item is visible or absent.",
                                }
                            ],
                        }
                    )
                all_six = '"condition_type": "combined_all_six_users"' in prompt
                return json.dumps(
                    {
                        "reason": "The included audits were reduced.",
                        "fact_audits": [
                            {
                                "fact_id": "F1",
                                "visibility": "VISIBLE" if all_six else "NOT_VISIBLE",
                                "source_users": ["provider_one"] if all_six else [],
                                "segment_references": ["segment_019"] if all_six else [],
                                "visual_description": "Only the combined condition contains the fact.",
                            }
                        ],
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

        expected_segment_paths = [
            [row["local_video"] for row in clip["source_segments"]]
            for clip in packet["clips"]
        ]
        self.assertEqual(len(runner.calls), 9)
        visual_calls = [call for call in runner.calls if call["video_paths"]]
        self.assertEqual(len(visual_calls), 6)
        self.assertEqual(
            [call["video_paths"] for call in visual_calls],
            expected_segment_paths,
        )
        self.assertEqual(len(prompt_rows), 9)
        self.assertTrue(all("elapsed_seconds" in row for row in prompt_rows))
        self.assertTrue(all("video_paths" not in row for row in prompt_rows))
        for row in prompt_rows:
            self.assertNotIn("SECRET_TIME_", row["prompt"])
            self.assertNotIn(str(self.tmp_path), row["prompt"])
        self.assertTrue(
            all("elapsed_seconds" in evaluation for evaluation in result["evaluations"])
        )
        self.assertFalse(result["gate"]["speaker_only_answerable"])
        self.assertTrue(result["gate"]["all_six_answerable"])

    def test_concurrent_activity_subcheck_remains_blocking_for_six_users(self) -> None:
        semantic_subchecks = {
            name: {"status": "PASS", "reason": "passed"}
            for name in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES
        }
        semantic_subchecks["other_person_activity_query"] = {
            "status": "FAIL",
            "reason": "the answer is another person's concurrent activity",
        }
        merged = merge_parallel_judges(
            qa_formality_judge={
                "checks": {
                    "qa_formality": {
                        "status": "PASS",
                        "reason": "model overall pass",
                        "fix": "",
                        "semantic_subchecks": semantic_subchecks,
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
        self.assertIn(
            "other_person_activity_query fail",
            merged["checks"]["qa_formality"]["reason"],
        )
        self.assertIn(
            "replace any concurrent-activity report",
            merged["checks"]["qa_formality"]["fix"],
        )

    def test_groundedness_maps_each_user_source_sequence_then_reduces_text_only(self) -> None:
        packet = self.media_packet()
        _images, full_videos = media_for_clips(
            packet["clips"],
            backend="transformers-local",
            allow_openai_video_input=False,
            media_role="full",
        )
        prompt_rows = []

        class Runner:
            model_id = "test-runner"

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
                user = next(user for user in SIX_USERS if f'user "{user}"' in prompt)
                return json.dumps(
                    {
                        "user": user,
                        "claims": [
                            {
                                "claim": "The material event is visible.",
                                "status": "SUPPORTED",
                                "segment_references": ["segment_010"],
                                "visual_description": "The event appears clearly.",
                            }
                        ],
                    }
                )

        runner = Runner()

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
                runner=runner,
                media_backend="transformers-local",
                allow_openai_video_input=False,
                prompt_rows=prompt_rows,
                full_image_paths=[],
                full_video_paths=full_videos,
                attempt=1,
            )

        map_rows = [
            row for row in prompt_rows if row["stage"] == "evidence_segment_observation"
        ]
        reduce_rows = [
            row for row in prompt_rows if row["stage"] == "evidence_groundedness_aggregation"
        ]
        self.assertEqual(len(map_rows), 6)
        self.assertEqual(len(reduce_rows), 1)
        self.assertEqual([row["media_summary"]["video_count"] for row in map_rows], [20] * 6)
        self.assertEqual(reduce_rows[0]["media_summary"]["video_count"], 0)
        self.assertEqual(len(runner.calls), 6)
        self.assertTrue(all(len(call["video_paths"]) == 20 for call in runner.calls))
        self.assertTrue(all("SECRET_TIME_" not in row["prompt"] for row in map_rows))
        self.assertFalse(any(row["stage"] == "evidence_groundedness_judge" for row in prompt_rows))
        self.assertEqual(trace["evidence_groundedness"]["elapsed_seconds"], 0.01)
        self.assertEqual(
            trace["evidence_groundedness"]["mode"],
            "per_user_source_segment_map_reduce",
        )

    def test_two_user_condition_contract_is_unchanged(self) -> None:
        conditions = build_answerability_conditions(["speaker", "provider"])

        self.assertEqual(len(conditions), 3)
        self.assertEqual(
            [condition["condition_type"] for condition in conditions],
            ["single_user", "single_user", "combined_all_users"],
        )


if __name__ == "__main__":
    unittest.main()
