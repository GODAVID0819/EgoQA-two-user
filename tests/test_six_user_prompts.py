from __future__ import annotations

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
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_qa_formality_judge_prompt,
    build_video_generation_prompt,
    video_packet_brief,
)


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

    def test_generation_prompt_requires_cross_view_but_not_every_provider(self) -> None:
        prompt = build_video_generation_prompt(six_user_packet(), "neutral")

        self.assertIn("required_users[0] is the speaker", prompt)
        self.assertIn("required_users[1] through required_users[5] are providers", prompt)
        self.assertIn("speaker's video alone must remain insufficient", prompt)
        self.assertIn("the six-video input must support exactly one correct option", prompt)
        self.assertIn("One or more provider views may supply the answer", prompt)
        self.assertIn("Do not require every provider to contribute", prompt)
        self.assertNotIn("Only all three required users", prompt)
        self.assertNotIn("omitting either evidence provider", prompt)

    def test_groundedness_prompt_allows_unused_providers(self) -> None:
        prompt = build_evidence_groundedness_judge_prompt(
            six_user_qa(),
            six_user_packet(),
        )

        self.assertIn("six-user", prompt)
        self.assertIn("at least one external provider view or provider combination", prompt)
        self.assertIn("Do not fail merely because an input provider is unused", prompt)
        self.assertNotIn("distinct answer-bearing contribution from each", prompt)

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
        self.assertIn("make the best forced choice", speaker_prompt)
        self.assertNotIn("proper-subset", speaker_prompt)
        self.assertNotIn("every provider", all_six_prompt)

    def test_formality_prompt_uses_six_user_scope(self) -> None:
        prompt = build_qa_formality_judge_prompt(six_user_qa(), six_user_packet())

        self.assertIn("qa_formality judge for a six-user multiple-choice question", prompt)
        self.assertNotIn("three-user multiple-choice question", prompt)

    def test_two_user_generation_prompt_keeps_legacy_dependency(self) -> None:
        prompt = build_video_generation_prompt(two_user_packet(), "neutral")

        self.assertIn("it must require additional evidence from required_users[1]", prompt)
        self.assertIn("required_users[1] supplies additional evidence and may be sufficient alone", prompt)
        self.assertNotIn("required_users[2]", prompt)


if __name__ == "__main__":
    unittest.main()
