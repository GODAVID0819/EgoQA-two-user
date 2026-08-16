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
)


def three_user_packet() -> dict[str, object]:
    users = ["speaker", "provider_one", "provider_two"]
    return {
        "evidence_id": "three-user-example",
        "required_users": users,
        "clips": [
            {"agent_name": user, "local_video": f"{user}.mp4"}
            for user in users
        ],
    }


def three_user_qa() -> dict[str, object]:
    return {
        "required_users": ["speaker", "provider_one", "provider_two"],
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


class ThreeUserPromptTests(unittest.TestCase):
    def test_formality_prompt_uses_three_user_scope(self) -> None:
        prompt = build_qa_formality_judge_prompt(
            three_user_qa(),
            three_user_packet(),
        )

        self.assertIn(
            "qa_formality judge for a three-user multiple-choice question",
            prompt,
        )
        self.assertNotIn(
            "qa_formality judge for a two-user multiple-choice question",
            prompt,
        )

    def test_generation_prompt_requires_both_providers_for_one_qa(self) -> None:
        prompt = build_video_generation_prompt(three_user_packet(), "neutral")

        self.assertIn(
            "required_users[1] and required_users[2] as two distinct evidence providers",
            prompt,
        )
        self.assertIn(
            "required_users[0] together with only required_users[1] must remain insufficient",
            prompt,
        )
        self.assertIn(
            "required_users[0] together with only required_users[2] must remain insufficient",
            prompt,
        )
        self.assertIn(
            "Only all three required users together may determine the unique correct option",
            prompt,
        )
        self.assertNotIn("required_users[1] may be sufficient alone", prompt)

    def test_groundedness_prompt_checks_both_provider_contributions(self) -> None:
        prompt = build_evidence_groundedness_judge_prompt(
            three_user_qa(),
            three_user_packet(),
        )

        self.assertIn(
            "verify a distinct answer-bearing contribution from each of required_users[1] and required_users[2]",
            prompt,
        )
        self.assertIn(
            "FAIL if the declared answer remains fully supported after omitting either evidence provider",
            prompt,
        )

    def test_answerability_prompt_preserves_strong_three_user_dependency(self) -> None:
        prompt = build_answerability_prompt(
            three_user_qa(),
            {
                "condition_id": "proper_subset::speaker+provider_one",
                "condition_type": "proper_subset",
                "users": ["speaker", "provider_one"],
            },
        )

        self.assertIn(
            "The accepted question requires the speaker and both evidence providers",
            prompt,
        )
        self.assertIn(
            "Any single-user or proper-subset condition is intentionally incomplete",
            prompt,
        )
        self.assertNotIn("evidence-provider-only condition", prompt)

    def test_two_user_generation_prompt_keeps_legacy_dependency(self) -> None:
        prompt = build_video_generation_prompt(two_user_packet(), "neutral")

        self.assertIn(
            "it must require additional evidence from required_users[1]",
            prompt,
        )
        self.assertIn(
            "required_users[1] supplies additional evidence and may be sufficient alone",
            prompt,
        )
        self.assertNotIn("required_users[2]", prompt)


if __name__ == "__main__":
    unittest.main()
