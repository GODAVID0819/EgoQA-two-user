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
    ANSWERABILITY_SUFFICIENCY_SCHEMA,
    QA_FORMALITY_CHECK_SCHEMA,
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
    SAMPLED_FRAME_GENERATOR_MEDIA_MODES,
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
    build_video_generation_prompt,
    video_packet_brief,
)
from egolife_two_user_qa import prompts as prompts_module  # noqa: E402


USERS = ["speaker", "provider_one", "provider_two", "provider_three", "provider_four", "provider_five"]


def six_user_packet() -> dict[str, object]:
    return {
        "evidence_id": "six-user-example",
        "required_users": list(USERS),
        "input_users": list(USERS),
        "speaker_user": USERS[0],
        "provider_users": USERS[1:],
        "clips": [
            {"agent_name": user, "local_video": f"{user}.mp4"}
            for user in USERS
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
        self.assertIn("full unpruned speaker video", brief)
        self.assertIn("An unused provider does not invalidate the item", brief)

    def test_packet_brief_supports_full_speaker_and_retained_provider_frames(self) -> None:
        packet = six_user_packet()
        packet["generator_media_mode"] = (
            "speaker_all_clustering_frames_five_provider_retained_cluster_frames"
        )
        for index, clip in enumerate(packet["clips"]):
            clip["generator_media_mode"] = (
                "all_clustering_frames_only"
                if index == 0
                else "retained_cluster_frames_only"
            )
            clip["media_role"] = "speaker" if index == 0 else "provider"
            clip["is_pruned"] = index != 0
            clip["frames"] = [{"timestamp_seconds": float(index)}]
            clip["temporal_pruning"] = {"enabled": True}

        brief = json.loads(video_packet_brief(packet))

        self.assertIn("all_clustering_frames_only", SAMPLED_FRAME_GENERATOR_MEDIA_MODES)
        self.assertIn(
            "speaker_all_clustering_frames_five_provider_retained_cluster_frames",
            SAMPLED_FRAME_GENERATOR_MEDIA_MODES,
        )
        self.assertEqual(
            brief["generator_media_contract"]["mode"],
            "asker_all_clustering_frames_provider_retained_cluster_frames",
        )
        self.assertEqual(brief["clips"][0]["media_role"], "speaker")
        self.assertFalse(brief["clips"][0]["is_pruned"])
        self.assertNotIn("pruning_summary", brief["clips"][0])

    def test_generation_prompt_requires_cross_view_but_not_every_provider(self) -> None:
        prompt = build_video_generation_prompt(six_user_packet(), "neutral")

        self.assertIn("required_users[0] is the speaker", prompt)
        self.assertIn("required_users[1] through required_users[5] are providers", prompt)
        self.assertIn("naturally have and genuinely want to ask", prompt)
        self.assertIn("full unpruned speaker video", prompt)
        self.assertIn("scan the full unpruned speaker video from beginning to end", prompt)
        self.assertIn("speaker video must naturally motivate the question but remain insufficient", prompt)
        self.assertIn(
            "The combined six-user video input must directly support exactly one correct option",
            prompt,
        )
        self.assertIn("One or more provider views may supply the answer", prompt)
        self.assertIn("Do not require every provider to contribute", prompt)
        self.assertIn("Provider videos may be pruned", prompt)
        self.assertIn("Concurrent-activity restriction", prompt)
        self.assertNotIn("Six-user interaction-chain example", prompt)
        self.assertNotIn("red tape dispenser with a torn white label", prompt)
        self.assertNotIn("Only all three required users", prompt)
        self.assertNotIn("omitting either evidence provider", prompt)

    def test_raw_six_user_prompt_audits_entire_speaker_video_before_claiming_missing_evidence(self) -> None:
        prompt = build_video_generation_prompt(six_user_packet(), "neutral")

        self.assertIn("full unpruned speaker video", prompt)
        self.assertIn("scan the full unpruned speaker video from beginning to end", prompt)
        self.assertIn("including the final minutes", prompt)
        self.assertIn("If any speaker frame directly shows the answer", prompt)
        self.assertNotIn("The generator receives images only", prompt)

    def test_groundedness_prompt_checks_speaker_motivation_and_provider_evidence(self) -> None:
        prompt = build_evidence_groundedness_judge_prompt(
            six_user_qa(),
            six_user_packet(),
        )

        self.assertIn("six-user", prompt)
        self.assertIn("full original speaker view grounds", prompt)
        self.assertIn("at least one external provider view or provider combination", prompt)
        self.assertIn("Do not fail merely because an input provider is unused", prompt)
        self.assertNotIn("answer-bearing evidence missing from the speaker view", prompt)
        self.assertNotIn("distinct answer-bearing contribution from each", prompt)

    def test_chunked_evidence_prompts_preserve_user_segment_provenance(self) -> None:
        self.assertTrue(hasattr(prompts_module, "build_evidence_segment_observation_prompt"))
        self.assertTrue(hasattr(prompts_module, "build_evidence_observation_aggregation_prompt"))
        segments = [
            {
                "segment_index": index,
                "time_token": token,
                "original_time_range": f"00:0{index}:00-00:0{index}:30",
            }
            for index, token in enumerate(
                ["20060000", "20063000", "20070000", "20073000", "20080000", "20083000"]
            )
        ]

        observation_prompt = prompts_module.build_evidence_segment_observation_prompt(
            six_user_qa(),
            user="speaker",
            segments=segments,
        )
        aggregation_prompt = prompts_module.build_evidence_observation_aggregation_prompt(
            six_user_qa(),
            six_user_packet(),
            observations=[
                {
                    "user": "speaker",
                    "segments": [
                        {
                            "segment_index": 0,
                            "time_token": "20060000",
                            "claims": [],
                        }
                    ],
                }
            ],
            vote_summary={
                "passed": True,
                "correct": "A",
                "visible_user_count": 3,
                "option_support_counts": {"A": 3, "B": 0, "C": 0, "D": 0, "E": 0},
                "threshold_options": ["A"],
            },
        )

        self.assertIn("6 separate 30-second videos", observation_prompt)
        self.assertIn("20083000", observation_prompt)
        self.assertIn("SUPPORTED", observation_prompt)
        self.assertIn("CONTRADICTED", observation_prompt)
        self.assertIn("HIGH", observation_prompt)
        self.assertIn("user_vote", observation_prompt)
        self.assertIn("When visibility or identity is uncertain", observation_prompt)
        self.assertIn("text-only evidence aggregator", aggregation_prompt)
        self.assertIn("20060000", aggregation_prompt)
        self.assertIn("authoritative deterministic vote summary", aggregation_prompt)
        self.assertIn("premises_supported", aggregation_prompt)
        self.assertIn("high_confidence_material_conflict", aggregation_prompt)
        self.assertIn("Do not recalculate option support", aggregation_prompt)

    def test_ten_minute_observation_prompt_uses_actual_segment_count(self) -> None:
        segments = [
            {
                "segment_index": index,
                "time_token": f"token-{index}",
                "original_time_range": f"range-{index}",
            }
            for index in range(20)
        ]

        prompt = prompts_module.build_evidence_segment_observation_prompt(
            six_user_qa(),
            user="speaker",
            segments=segments,
        )

        self.assertIn("20 separate 30-second videos", prompt)
        self.assertIn("include all 20 segment rows", prompt)

    def test_answerability_prompts_describe_only_two_conditions(self) -> None:
        speaker_prompt = build_answerability_prompt(
            six_user_qa(),
            {
                "condition_id": "speaker_only::speaker",
                "condition_type": "speaker_only",
                "users": ["speaker"],
            },
        )
        all_six_prompt = build_answerability_prompt(
            six_user_qa(),
            {
                "condition_id": "combined_all_six_users::" + "+".join(USERS),
                "condition_type": "combined_all_six_users",
                "users": USERS,
            },
        )

        self.assertIn("speaker-only condition", speaker_prompt)
        self.assertIn("six-video condition", all_six_prompt)
        self.assertIn("evidence-sufficiency judge", speaker_prompt)
        self.assertIn("Do not answer the question yourself", speaker_prompt)
        self.assertIn("Do not select an option", speaker_prompt)
        self.assertNotIn('"answerable"', speaker_prompt)
        self.assertIn("needed_facts", all_six_prompt)
        self.assertNotIn("You must choose exactly one answer", speaker_prompt)
        self.assertNotIn('"choice"', speaker_prompt)
        self.assertIn("full unpruned speaker video", speaker_prompt)
        self.assertIn("full original speaker video", all_six_prompt)
        self.assertIn("all five full original provider videos", all_six_prompt)
        self.assertNotIn("proper-subset", speaker_prompt)
        self.assertNotIn("every provider", all_six_prompt)

    def test_six_user_answerability_uses_evidence_sufficiency_schema(self) -> None:
        self.assertEqual(
            ANSWERABILITY_SUFFICIENCY_SCHEMA["required"],
            [
                "reason",
                "needed_facts",
            ],
        )
        self.assertNotIn("answerable", ANSWERABILITY_SUFFICIENCY_SCHEMA["properties"])
        fact_schema = ANSWERABILITY_SUFFICIENCY_SCHEMA["properties"]["needed_facts"]["items"]
        self.assertEqual(
            fact_schema["required"],
            [
                "fact",
                "why_needed",
                "visibility",
                "confidence",
                "source_user",
                "original_time_range",
                "visual_description",
            ],
        )
        prompt = build_answerability_prompt(
            six_user_qa(),
            {
                "condition_id": "speaker_only::speaker",
                "condition_type": "speaker_only",
                "users": ["speaker"],
            },
        )
        self.assertIn("Do not select an option", prompt)
        self.assertNotIn('"answerable"', prompt)
        self.assertIn("needed_facts", prompt)
        self.assertIn("NOT_VISIBLE", prompt)
        self.assertIn("AMBIGUOUS", prompt)
        self.assertIn("HIGH", prompt)
        self.assertIn("MEDIUM/LOW", prompt)
        self.assertIn("does not make the condition sufficient", prompt)
        self.assertIn("original_time_range", prompt)
        self.assertIn("The program computes sufficiency", prompt)

    def test_generation_prompt_records_round_diversity_focus(self) -> None:
        packet = six_user_packet()
        packet["generation_diversity_focus"] = {
            "round_index": 2,
            "temporal_band_seconds": [60, 90],
            "focal_provider": "provider_three",
            "relation_focus": "identity link",
        }
        packet["previous_questions_to_avoid"] = ["Where was the cup?"]

        prompt = build_video_generation_prompt(packet, "neutral")

        self.assertIn("speaker video must naturally motivate the question but remain insufficient", prompt)
        self.assertIn("Do not require every provider to contribute", prompt)
        self.assertNotIn("generation_diversity_focus", prompt)

    def test_formality_prompt_uses_six_user_scope(self) -> None:
        prompt = build_qa_formality_judge_prompt(six_user_qa(), six_user_packet())

        self.assertIn("qa_formality judge for a six-user multiple-choice question", prompt)
        self.assertIn("plausible information need the speaker would naturally have", prompt)
        self.assertIn("contrived third-party quiz", prompt)
        self.assertNotIn("three-user multiple-choice question", prompt)
        self.assertIn("other_person_activity_query", prompt)
        self.assertIn("concurrent activity report", prompt)
        self.assertIn("other_person_activity_query", QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES)
        self.assertIn("other_person_activity_query", QA_FORMALITY_CHECK_SCHEMA["semantic_subchecks"])

    def test_formality_reference_clarity_uses_local_resolvability(self) -> None:
        prompt = build_qa_formality_judge_prompt(six_user_qa(), six_user_packet())

        self.assertIn("multiple equally plausible referents", prompt)
        self.assertIn("would change the meaning or answer", prompt)
        self.assertIn("does not need to be globally unique", prompt)
        self.assertIn("the wooden chair in the middle of the room", prompt)
        self.assertNotIn('Examples that FAIL include "the other room", "the other person", "the cup"', prompt)

    def test_two_user_generation_prompt_keeps_legacy_dependency(self) -> None:
        prompt = build_video_generation_prompt(two_user_packet(), "neutral")

        self.assertIn("it must require additional evidence from required_users[1]", prompt)
        self.assertIn(
            "required_users[1] supplies additional evidence; report each user's individual answerability truthfully",
            prompt,
        )
        self.assertNotIn("required_users[2]", prompt)
        self.assertNotIn("Six-user interaction-chain example", prompt)


if __name__ == "__main__":
    unittest.main()
