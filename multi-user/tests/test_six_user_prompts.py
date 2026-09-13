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

from egolife_two_user_qa.prompts import (  # noqa: E402
    LONG_HORIZON_FORMALITY_GUIDANCE,
    LONG_HORIZON_GROUNDEDNESS_GUIDANCE,
    OPTIONAL_LONG_HORIZON_GUIDANCE,
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
    build_answerability_prompt,
    build_answerability_condition_aggregation_prompt,
    build_answerability_fact_plan_prompt,
    build_answerability_user_fact_audit_prompt,
    build_evidence_groundedness_judge_prompt,
    build_evidence_observation_aggregation_prompt,
    build_evidence_segment_observation_prompt,
    build_qa_formality_judge_prompt,
    build_video_generation_prompt,
    video_packet_brief,
)


USERS = ["speaker", "provider_one", "provider_two", "provider_three", "provider_four", "provider_five"]


def six_user_packet() -> dict[str, object]:
    roles = ["speaker_all_clustering_frames", *(["provider_retained_cluster_frames"] * 5)]
    return {
        "evidence_id": "six-user-example",
        "generator_media_mode": (
            "speaker_all_clustering_frames_five_provider_retained_cluster_frames"
        ),
        "required_users": list(USERS),
        "input_users": list(USERS),
        "speaker_user": USERS[0],
        "provider_users": USERS[1:],
        "clips": [
            {
                "agent_name": user,
                "generator_media_mode": (
                    "all_clustering_frames_only"
                    if index == 0
                    else "retained_cluster_frames_only"
                ),
                "force_frame_inputs": True,
                "frames": [
                    {
                        "path": f"{user}-{frame_index}.jpg",
                        "timestamp_seconds": float(frame_index),
                    }
                    for frame_index in range(3 if index == 0 else 2)
                ],
                "media_role": role,
                "is_pruned": index != 0,
            }
            for index, (user, role) in enumerate(zip(USERS, roles))
        ],
    }


def six_user_qa() -> dict[str, object]:
    return {
        "required_users": list(USERS),
        "question": "Which item completed the shared setup?",
        "options": ["First item", "Second item", "Third item", "Fourth item", "Fifth item"],
        "correct": "C",
    }


def two_user_packet() -> dict[str, object]:
    users = ["speaker", "provider"]
    return {
        "evidence_id": "two-user-regression",
        "required_users": users,
        "clips": [
            {"agent_name": user, "local_video": f"{user}.mp4"}
            for user in users
        ],
    }


class SixUserPromptTests(unittest.TestCase):
    def test_packet_brief_exposes_six_user_roles(self) -> None:
        brief = video_packet_brief(six_user_packet())

        self.assertIn('"speaker_user": "speaker"', brief)
        self.assertIn('"provider_users": [', brief)
        self.assertNotIn('"anchor_provider_users"', brief)
        self.assertNotIn('"additional_provider_users"', brief)
        self.assertIn("required_users[1] through required_users[5] are providers", brief)
        self.assertIn("speaker_all_clustering_frames", brief)
        self.assertIn("provider_retained_cluster_frames", brief)
        self.assertIn("speaker input contains every frame sampled for CLIP clustering", brief)
        self.assertIn("asker_all_clustering_frames_provider_retained_cluster_frames", brief)

    def test_generation_prompt_requires_cross_view_but_not_every_provider(self) -> None:
        prompt = build_video_generation_prompt(six_user_packet(), "neutral")

        self.assertIn("required_users[0] is the speaker", prompt)
        self.assertIn("required_users[1] through required_users[5] are providers", prompt)
        self.assertIn("speaker's sampled frames alone must remain insufficient", prompt)
        self.assertIn(
            "The combined six-user image input must directly support exactly one correct option",
            prompt,
        )
        self.assertIn("One or more provider views may supply the answer", prompt)
        self.assertIn("Do not require every provider to contribute", prompt)
        self.assertIn("speaker would naturally have and genuinely want to ask", prompt)
        self.assertIn("every CLIP-sampled frame from the speaker", prompt)
        self.assertIn("only sampled members of provider clusters that survived pruning", prompt)
        self.assertIn("It receives no MP4", prompt)
        self.assertIn("Concurrent-activity restriction", prompt)
        self.assertIn(
            "Do not generate a question whose answer is what one person was doing",
            prompt,
        )
        self.assertIn("Do not generate options that encode pairs of concurrent activities", prompt)
        self.assertNotIn("small red tape dispenser with a torn white label", prompt)
        self.assertNotIn("top drawer of the blue rolling cart beside the pantry door", prompt)
        self.assertNotIn("Only all three required users", prompt)
        self.assertNotIn("omitting either evidence provider", prompt)

    def test_ten_minute_prompt_injects_examples_and_sampled_frame_contract(self) -> None:
        packet = six_user_packet()
        packet["generator_context_budget"] = {
            "policy": "complete_surviving_sampled_frames",
            "analysis_sample_fps": 1.0,
            "aggregate_frame_budget": 3600,
            "model_input_frame_count": 12,
        }
        packet["speaker_consensus_pruning"] = {
            "method": "speaker_provider_time_aware_provider_only",
            "comparison_scope": (
                "every_asker_cluster_x_every_provider_cluster_"
                "within_plus_minus_time_window"
            ),
            "temporal_policy": "window_30s_all_pairs_contiguous_asker_preserved",
            "pruned_side": "providers_only",
            "asker_preserved": True,
            "max_pair_time_difference_seconds": 30.0,
            "mutual_nearest_only": False,
            "split_noncontiguous_clusters": True,
            "cluster_window_seconds": 30.0,
            "cluster_count_per_window": 12,
            "speaker_preserved": True,
        }
        for clip in packet["clips"]:
            clip["context_sampling"] = {
                "policy": "complete_surviving_sampled_frames",
                "analysis_sample_fps": 1.0,
                "source_frame_count": len(clip["frames"]),
                "model_input_frame_count": len(clip["frames"]),
                "aggregate_frame_budget": 3600,
            }
            clip["temporal_pruning"] = dict(packet["speaker_consensus_pruning"])

        prompt = build_video_generation_prompt(packet, "neutral")
        judge = build_evidence_groundedness_judge_prompt(six_user_qa(), packet)

        self.assertEqual(OPTIONAL_LONG_HORIZON_GUIDANCE.count("Example structure:"), 5)
        self.assertIn(OPTIONAL_LONG_HORIZON_GUIDANCE, prompt)
        self.assertIn("every sampled frame from the speaker", prompt)
        self.assertIn("sampled member of provider clusters that survived pruning", prompt)
        self.assertIn("no MP4 or pruned video", prompt)
        self.assertNotIn("window_30s_all_pairs_contiguous_asker_preserved", prompt)
        self.assertNotIn(
            "every_asker_cluster_x_every_provider_cluster_within_plus_minus_time_window",
            prompt,
        )
        self.assertNotIn('"speaker_consensus_pruning"', prompt)
        self.assertNotIn('"six_user_pruning_policy"', prompt)
        self.assertNotIn('"pruning_summary"', prompt)
        self.assertIn("samples at one frame per second", prompt)
        self.assertIn("clusters each 30-second block independently with K=12", prompt)
        self.assertIn("only full-video judge decoding is downsampled", prompt)
        self.assertIn(LONG_HORIZON_GROUNDEDNESS_GUIDANCE, judge)

    def test_prompt_packet_omits_exact_frame_mappings(self) -> None:
        packet = six_user_packet()
        packet["generator_context_budget"] = {
            "policy": "complete_surviving_sampled_frames",
            "model_input_frame_count": 13,
            "selected_frame_indices": [0, 2, 7],
        }
        for clip in packet["clips"]:
            clip["clip_clock"] = "12:34:56"
            clip["local_video"] = "C:/private/exact/source.mp4"
            clip["context_sampling"] = {
                "policy": "complete_surviving_sampled_frames",
                "model_input_frame_count": len(clip["frames"]),
                "selected_frame_indices": [0, 2],
                "frame_to_cluster": {"0": "cluster-17"},
            }
            clip["temporal_pruning"] = {
                "method": "speaker_provider_time_aware_provider_only",
                "kept_duration_seconds": 8.0,
                "keep_intervals": [
                    [float(index), float(index) + 0.5]
                    for index in range(500)
                ],
                "pruned_to_original_time_map": [
                    {
                        "pruned_start_seconds": float(index),
                        "pruned_end_seconds": float(index) + 0.5,
                        "original_start_seconds": float(index) + 1000.0,
                        "original_end_seconds": float(index) + 1000.5,
                    }
                    for index in range(500)
                ],
            }

        brief = json.loads(video_packet_brief(packet))
        serialized_brief = json.dumps(brief, sort_keys=True)
        prompts = [
            build_video_generation_prompt(packet, "neutral"),
            build_qa_formality_judge_prompt(six_user_qa(), packet),
            build_evidence_groundedness_judge_prompt(six_user_qa(), packet),
        ]

        for forbidden_key in (
            "clip_clock",
            "local_video",
            "path",
            "timestamp_seconds",
            "selected_frame_indices",
            "frame_to_cluster",
            "keep_intervals",
            "pruned_to_original_time_map",
            "pruning_summary",
            "six_user_pruning_policy",
        ):
            self.assertNotIn(f'"{forbidden_key}"', serialized_brief)
        for exact_value in (
            "speaker-0.jpg",
            "provider_one-0.jpg",
            "C:/private/exact/source.mp4",
            "12:34:56",
            "cluster-17",
        ):
            for prompt in prompts:
                self.assertNotIn(exact_value, prompt)
        for time_grid_key in (
            "pruned_start_seconds",
            "pruned_end_seconds",
            "original_start_seconds",
            "original_end_seconds",
            "temporal_alignment_contract",
        ):
            for prompt in prompts:
                self.assertNotIn(time_grid_key, prompt)
        self.assertIn("exact per-frame paths, indices, timestamps", prompts[0])

    def test_groundedness_prompt_allows_unused_providers(self) -> None:
        prompt = build_evidence_groundedness_judge_prompt(
            six_user_qa(),
            six_user_packet(),
        )

        self.assertIn("six-user", prompt)
        self.assertIn("at least one external provider view or provider combination", prompt)
        self.assertIn("Do not fail merely because an input provider is unused", prompt)
        self.assertIn("full original speaker view grounds", prompt)
        self.assertNotIn("distinct answer-bearing contribution from each", prompt)

    def test_answerability_prompts_share_one_answer_neutral_fact_plan(self) -> None:
        qa_with_secret_gold = six_user_qa()
        qa_with_secret_gold["correct"] = "DO_NOT_EXPOSE_GOLD"
        qa_with_secret_gold["answer"] = "DO_NOT_EXPOSE_ANSWER"
        fact_plan = {
            "reason": "Both facts are needed.",
            "needed_facts": [
                {"fact_id": "F1", "fact": "The referenced setup is visible.", "why_needed": "It anchors the question."},
                {"fact_id": "F2", "fact": "The completing item is visible.", "why_needed": "It resolves the missing detail."},
            ],
        }
        plan_prompt = build_answerability_fact_plan_prompt(qa_with_secret_gold)
        speaker_audit_prompt = build_answerability_user_fact_audit_prompt(
            qa_with_secret_gold,
            user="speaker",
            fact_plan=fact_plan,
            segment_count=20,
        )
        aggregation_prompt = build_answerability_condition_aggregation_prompt(
            qa_with_secret_gold,
            condition={
                "condition_id": "speaker_only::speaker",
                "condition_type": "speaker_only",
                "users": ["speaker"],
            },
            fact_plan=fact_plan,
            user_audits=[],
        )

        for prompt in (plan_prompt, speaker_audit_prompt, aggregation_prompt):
            self.assertNotIn("DO_NOT_EXPOSE_GOLD", prompt)
            self.assertNotIn("DO_NOT_EXPOSE_ANSWER", prompt)
            self.assertNotIn('"answerable"', prompt)
            self.assertNotIn('"choice"', prompt)
        self.assertIn("Stage marker: answerability_fact_plan", plan_prompt)
        self.assertIn("exact frozen list", plan_prompt)
        self.assertIn("20 ordered source-video segments", speaker_audit_prompt)
        self.assertIn("segment_001", speaker_audit_prompt)
        self.assertNotIn("segment_002", speaker_audit_prompt)
        self.assertIn('"fact_id": "F1"', speaker_audit_prompt)
        self.assertIn('"fact_id": "F1"', aggregation_prompt)
        self.assertIn("caller derives sufficiency", aggregation_prompt)

    def test_per_user_groundedness_prompts_are_generalized_and_path_free(self) -> None:
        qa = six_user_qa()
        qa["answer"] = "Third item"
        observation_prompt = build_evidence_segment_observation_prompt(
            qa,
            user="speaker",
            segment_count=20,
        )
        aggregation_prompt = build_evidence_observation_aggregation_prompt(
            qa,
            six_user_packet(),
            observations=[
                {
                    "user": "speaker",
                    "claims": [
                        {
                            "claim": "The setup is visible.",
                            "status": "SUPPORTED",
                            "segment_references": ["segment_011"],
                            "visual_description": "The setup appears clearly.",
                        }
                    ],
                }
            ],
        )

        self.assertIn("20 ordered source-video segments", observation_prompt)
        self.assertIn(LONG_HORIZON_GROUNDEDNESS_GUIDANCE, observation_prompt)
        self.assertIn(LONG_HORIZON_GROUNDEDNESS_GUIDANCE, aggregation_prompt)
        self.assertNotIn("segment_002", observation_prompt)
        for forbidden in ("local_video", "timestamp_seconds", "clip_clock", "C:/"):
            self.assertNotIn(forbidden, observation_prompt)
            self.assertNotIn(forbidden, aggregation_prompt)

    def test_formality_prompt_uses_six_user_scope(self) -> None:
        prompt = build_qa_formality_judge_prompt(six_user_qa(), six_user_packet())

        self.assertIn("qa_formality judge for a six-user multiple-choice question", prompt)
        self.assertIn("plausible information need the speaker would naturally have", prompt)
        self.assertIn("other_person_activity_query", QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES)
        self.assertIn("3. other_person_activity_query", prompt)
        self.assertIn("one provider's event used to query another provider's activity", prompt)
        self.assertIn("FAIL pair-matching questions", prompt)
        self.assertIn("resolvable in its local sentence", prompt)
        self.assertIn("not automatic failures when local context does resolve them", prompt)
        self.assertNotIn("three-user multiple-choice question", prompt)

    def test_legacy_judges_receive_all_five_structural_hints(self) -> None:
        qa_item = six_user_qa()
        packet = six_user_packet()
        prompts = {
            "qa_formality": build_qa_formality_judge_prompt(qa_item, packet),
            "evidence_groundedness": build_evidence_groundedness_judge_prompt(
                qa_item,
                packet,
            ),
            "answerability_speaker_only": build_answerability_prompt(
                qa_item,
                {
                    "condition_id": "speaker_only::speaker",
                    "condition_type": "speaker_only",
                    "users": ["speaker"],
                },
            ),
            "answerability_combined_all_six_users": build_answerability_prompt(
                qa_item,
                {
                    "condition_id": "combined_all_six_users",
                    "condition_type": "combined_all_six_users",
                    "users": list(USERS),
                },
            ),
        }

        self.assertIn(LONG_HORIZON_FORMALITY_GUIDANCE, prompts["qa_formality"])
        for prompt_name in (
            "evidence_groundedness",
            "answerability_speaker_only",
            "answerability_combined_all_six_users",
        ):
            self.assertIn(LONG_HORIZON_GROUNDEDNESS_GUIDANCE, prompts[prompt_name])

        structural_hints = (
            "Object trajectory",
            "Cross-user before/after state",
            "Same-user revisit with a cross-user intervention",
            "Last-seen or most-recent interaction",
            "Cross-user temporal ordering",
        )
        for prompt_name, prompt in prompts.items():
            with self.subTest(prompt=prompt_name):
                for hint in structural_hints:
                    self.assertIn(hint, prompt)

    def test_two_user_generation_prompt_keeps_legacy_dependency(self) -> None:
        prompt = build_video_generation_prompt(two_user_packet(), "neutral")

        self.assertIn("it must require additional evidence from required_users[1]", prompt)
        self.assertIn(
            "required_users[1] supplies additional evidence; report each user's individual answerability truthfully",
            prompt,
        )
        self.assertIn("Concurrent-activity restriction", prompt)
        self.assertNotIn("required_users[2]", prompt)
        self.assertNotIn("Six-user interaction-chain example", prompt)


if __name__ == "__main__":
    unittest.main()
