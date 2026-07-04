from __future__ import annotations

import json
import unittest
from pathlib import Path

from grpo_ready.extract_attempts import extract_packet_attempts, iter_attempt_records


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "outputs" / "qa_mcq.intermediate.jsonl"


def fixture_packet() -> dict:
    attempts = []
    for index, accepted in enumerate((False, True), start=1):
        raw = json.dumps({"question": f"question-{index}"})
        attempts.append(
            {
                "attempt": index,
                "evidence_id": "E1",
                "question_type": "difference",
                "generation_mode": "baseline",
                "feedback_in": "" if index == 1 else "repair grounding",
                "media": {
                    "image_paths": ["generator.jpg"],
                    "video_paths": ["generator.mp4"],
                    "full_image_paths": ["evaluator.jpg"],
                    "full_video_paths": ["evaluator.mp4"],
                },
                "generation": {
                    "prompt": "fixed prompt",
                    "raw_output": raw,
                    "parsed_qa": {"question": f"question-{index}"},
                },
                "judge": {"qa_formality": {"parsed": {"review_passed": True}}},
                "answerability": {"evaluations": []},
                "result": {"accepted": accepted, "reason": ""},
            }
        )
    return {
        "evidence_id": "E1",
        "status": "accepted",
        "question_type": "difference",
        "generation_mode": "baseline",
        "attempts": attempts,
    }


class ExtractAttemptsTests(unittest.TestCase):
    def test_attempt_acceptance_does_not_inherit_packet_status(self) -> None:
        rows = extract_packet_attempts(fixture_packet())

        self.assertEqual([row.accepted for row in rows], [False, True])
        self.assertEqual(rows[0].attempt_id, "E1::attempt::1")
        self.assertEqual(rows[0].raw_qa, '{"question": "question-1"}')
        self.assertEqual(rows[0].generator_video_paths, ("generator.mp4",))
        self.assertEqual(rows[0].evaluator_video_paths, ("evaluator.mp4",))
        self.assertEqual(rows[1].feedback, "repair grounding")

    def test_rejected_compact_attempts_are_recovered_from_generation_trace(self) -> None:
        packet = fixture_packet()
        packet["status"] = "rejected"
        packet["generation_trace"] = packet["attempts"]
        packet["attempts"] = [
            {"attempt": 1, "qa": None, "reason": "judge failed"},
            {"attempt": 2, "qa": None, "reason": "judge failed"},
        ]

        rows = extract_packet_attempts(packet)

        self.assertEqual([row.raw_qa for row in rows], [
            '{"question": "question-1"}',
            '{"question": "question-2"}',
        ])
        self.assertEqual([row.accepted for row in rows], [False, True])

    def test_real_intermediate_contains_all_attempts(self) -> None:
        rows = list(iter_attempt_records(INPUT_PATH))

        self.assertEqual(len(rows), 121)
        self.assertEqual(sum(row.accepted for row in rows), 27)
        self.assertTrue(all(row.raw_qa.strip() for row in rows))
        self.assertEqual(len({row.attempt_id for row in rows}), 121)


if __name__ == "__main__":
    unittest.main()
