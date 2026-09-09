from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from manual_ablation_review import SIX_PARTICIPANTS, build_review_data


def _audit() -> dict:
    day = "DAY3"
    token = "10153000"
    clips = [
        {
            "user": user,
            "agent_dir": agent_dir,
            "day": day,
            "time_token": token,
            "clip_clock": "10:15:30.00",
            "video_url": (
                "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/"
                f"{agent_dir}/{day}/{day}_{agent_dir}_{token}.mp4"
            ),
        }
        for agent_dir, user in SIX_PARTICIPANTS[:2]
    ]
    return {
        "required_users": ["Jake", "Alice"],
        "source_urls": {"videos": [clip["video_url"] for clip in clips]},
        "video_evidence": clips,
    }


def _parsed_qa() -> dict:
    return {
        "qa_id": "QA_TEST_001",
        "question_type": "neutral",
        "question": "What was on the table after I looked away?",
        "options": ["Cup", "Plate", "Book", "Phone", "Bag"],
        "correct": "A",
        "answer": "Cup",
        "required_users": ["Jake", "Alice"],
        "generator_rationale": "The second view supplies the missing detail.",
        "why_two_users_needed": "The asker did not see the table.",
        "per_user_evidence_claims": [
            {"user": "Alice", "claim": "A cup was on the table."}
        ],
    }


class ManualReviewIntermediateTest(unittest.TestCase):
    @patch.object(Path, "is_file", return_value=True)
    @patch("manual_ablation_review._read_jsonl")
    def test_six_user_mode_selects_only_accepted_intermediate_rows(
        self,
        read_jsonl,
        _is_file,
    ) -> None:
        accepted_attempt = {
            "attempt": 2,
            "question_type": "neutral",
            "generation": {
                "parsed_qa": _parsed_qa(),
                "normalized_qa": {
                    "single_user_answerability": {"Jake": "insufficient"},
                    "combined_answerability": "sufficient",
                },
            },
            "media": {"human_audit": _audit(), "judge_media_role": "full"},
            "result": {"accepted": True},
        }
        source_rows = [
            {"evidence_id": "EVIDENCE_001", "status": "rejected", "attempts": []},
            {
                "evidence_id": "EVIDENCE_001",
                "qa_id": "QA_TEST_001",
                "status": "accepted",
                "attempts": [accepted_attempt],
            },
        ]
        read_jsonl.return_value = (b"intermediate fixture", source_rows)

        data, analysis, review_rows = build_review_data(
            [("experiment", Path("fixture.jsonl"))],
            six_user=True,
        )

        self.assertEqual(data["review_mode"], "six_user")
        self.assertEqual(data["summary"]["qa_count"], 1)
        self.assertEqual(data["summary"]["video_count"], 6)
        self.assertEqual(
            [video["agent_dir"] for video in data["evidence"][0]["videos"]],
            [agent_dir for agent_dir, _ in SIX_PARTICIPANTS],
        )
        self.assertEqual(review_rows[0]["video_count"], 6)
        self.assertIn("A6_SHURE", review_rows[0]["video_6_url"])
        self.assertEqual(
            analysis["run_summaries"][0]["skipped_nonaccepted_row_count"],
            1,
        )

    @patch.object(Path, "is_file", return_value=True)
    @patch("manual_ablation_review._read_jsonl")
    def test_cyclic_intermediate_uses_nested_accepted_qa(
        self,
        read_jsonl,
        _is_file,
    ) -> None:
        qa = {
            **_parsed_qa(),
            "evidence_id": "EVIDENCE_002",
            "source_urls": _audit()["source_urls"],
            "video_evidence": _audit()["video_evidence"],
        }
        source_rows = [
            {
                "evidence_id": "EVIDENCE_002",
                "status": "accepted",
                "attempt": 1,
                "attempts": [{"attempt": 1, "accepted": True, "qa": qa}],
            }
        ]
        read_jsonl.return_value = (b"cyclic fixture", source_rows)

        data, _, _ = build_review_data(
            [("cyclic", Path("fixture.jsonl"))],
            six_user=False,
        )

        self.assertEqual(data["summary"]["qa_count"], 1)
        self.assertEqual(data["summary"]["video_count"], 2)
        self.assertEqual(data["evidence"][0]["qas"][0]["question"], qa["question"])


if __name__ == "__main__":
    unittest.main()
