from __future__ import annotations

import json
import unittest

from training.grpo_v3.experiments.human_preference_reviewer.v1.data import CandidateRecord
from training.grpo_v3.experiments.human_preference_reviewer.v1.prompting import build_messages


class PromptingTests(unittest.TestCase):
    def test_prompt_preserves_speaker_provider_order_and_excludes_labels(self) -> None:
        candidate = CandidateRecord(
            candidate_id="e1::candidate_01", evidence_id="e1", display_order=1,
            question="What happened?", options=("a", "b", "c", "d", "e"),
            correct="B", answer="b", evidence_quality=3, answerability=2,
            qa_formality=1, overall_rank=4,
        )

        messages = build_messages(
            candidate,
            video_a_path="speaker.mp4",
            video_b_path="provider.mp4",
            video_a_user="Jake",
            video_b_user="Shure",
        )

        content = messages[0]["content"]
        self.assertEqual(content[0], {"type": "video", "video": "speaker.mp4"})
        self.assertEqual(content[1], {"type": "video", "video": "provider.mp4"})
        text = json.dumps(content[2], ensure_ascii=False)
        self.assertIn("Video A (speaker): Jake", text)
        self.assertIn("Video B (provider): Shure", text)
        self.assertIn("What happened?", text)
        for forbidden in (
            "candidate_01", "evidence_quality", "answerability_score", "overall_rank", '"rank"'
        ):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
