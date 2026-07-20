from __future__ import annotations

import unittest

from training.grpo_v3_groundedness_audit import (
    build_review_rows,
    select_audit_cases,
    summarize_reviews,
)


def _trace(index: int, status: str) -> dict:
    question_type = "commonality" if index % 2 == 0 else "difference"
    format_status = "raw_valid" if index % 3 == 0 else "repaired"
    return {
        "reward_call_index": index,
        "phase": "train",
        "evidence_id": f"E{index:03d}",
        "candidate_index": index % 4,
        "reward": float(index % 7 - 3),
        "record": {
            "group_id": f"E{index:03d}",
            "groundedness_status": status,
            "qa_formality_status": "PASS",
            "shallow_activity_status": "PASS",
            "combined_correct": True,
            "speaker_only_correct": True,
            "provider_only_correct": False,
            "reward_components": {
                "groundedness": 1.0,
                "combined_answerability": 1.0,
                "speaker_leakage_cap": 0.5,
            },
            "raw_qa": "raw",
            "qa": {
                "question_type": question_type,
                "question": f"问题 {index}",
                "options": ["A", "B", "C", "D", "E"],
                "correct": "A",
                "answer": "A",
                "required_users": ["u1", "u2"],
                "evidence": [
                    {"user": "u1", "needed_fact": "事实一", "timeframe": "0:02-0:05"},
                    {"user": "u2", "needed_fact": "事实二", "timeframe": "0:07-0:09"},
                ],
                "referred_timestamps": [
                    {"user": "u1", "timestamp_seconds": 3.0, "moment": "时刻一"},
                    {"user": "u2", "timestamp_seconds": 8.0, "moment": "时刻二"},
                ],
            },
            "review": {
                "judger": {
                    "checks": {
                        "qa_formality": {
                            "status": "PASS",
                            "reason": f"形式理由 {index}",
                            "fix": "",
                        },
                        "evidence_groundedness": {
                            "status": status,
                            "reason": f"理由 {index}",
                            "fix": f"修复 {index}",
                        }
                    }
                },
            },
            "answerability": {
                "evaluations": [
                    {
                        "condition_id": "single_user::u1",
                        "condition_type": "single_user",
                        "users": ["u1"],
                        "choice": "A",
                        "evidence_used": "提问者证据",
                        "insufficient_reason": "",
                        "choice_uncertainty": {"normalized_entropy": 0.1},
                    },
                    {
                        "condition_id": "single_user::u2",
                        "condition_type": "single_user",
                        "users": ["u2"],
                        "choice": "B",
                        "evidence_used": "提供者证据",
                        "insufficient_reason": "",
                        "choice_uncertainty": {"normalized_entropy": 0.2},
                    },
                    {
                        "condition_id": "combined_all_users::u1+u2",
                        "condition_type": "combined_all_users",
                        "users": ["u1", "u2"],
                        "choice": "A",
                        "evidence_used": "合并证据",
                        "insufficient_reason": "",
                        "choice_uncertainty": {"normalized_entropy": 0.05},
                    },
                ],
                "gate": {
                    "passed": False,
                    "reason": "asker condition answered correctly",
                    "speaker_user": "u1",
                    "evidence_provider_user": "u2",
                },
            },
            "format_validation": {"status": format_status},
            "judge_prompts": [
                {
                    "stage": "evidence_groundedness_judge",
                    "video_paths": [f"/videos/E{index:03d}_u1.mp4", f"/videos/E{index:03d}_u2.mp4"],
                    "media_role": "full",
                }
            ],
        },
    }


class GroundednessAuditTests(unittest.TestCase):
    def test_selects_balanced_deterministic_cases_and_preserves_judgement_evidence(self) -> None:
        traces = [_trace(i, "PASS" if i < 20 else "FAIL") for i in range(40)]
        first = select_audit_cases(traces, pass_count=12, fail_count=12)
        second = select_audit_cases(list(reversed(traces)), pass_count=12, fail_count=12)

        self.assertEqual([row["case_id"] for row in first], [row["case_id"] for row in second])
        self.assertEqual(sum(row["reviewer_groundedness"] == "PASS" for row in first), 12)
        self.assertEqual(sum(row["reviewer_groundedness"] == "FAIL" for row in first), 12)
        self.assertTrue(all(len(row["video_paths"]) == 2 for row in first))
        self.assertTrue(all(len(row["video_windows"]) == 2 for row in first))
        self.assertTrue(all(row["video_windows"][0]["start_seconds"] == 0.0 for row in first))
        self.assertTrue(all(row["evidence_claims"] for row in first))
        self.assertTrue(all(row["referred_timestamps"] for row in first))

        case = first[0]
        self.assertEqual(case["reviewer_combined_answerability"], "PASS")
        self.assertEqual(case["reviewer_speaker_leakage"], "LEAK")
        self.assertEqual(case["reviewer_provider_answerability"], "NOT_ANSWERABLE")
        self.assertEqual(case["reviewer_qa_formality"], "PASS")
        self.assertEqual(case["reviewer_shallow_activity"], "NO_SHALLOW")
        self.assertEqual(case["speaker_user"], "u1")
        self.assertEqual(case["provider_user"], "u2")
        self.assertFalse(case["reviewer_answerability_gate_passed"])
        self.assertEqual(case["reviewer_answerability_gate_reason"], "asker condition answered correctly")
        self.assertEqual(len(case["answerability_evaluations"]), 3)
        self.assertEqual(case["answerability_evaluations"][0]["normalized_entropy"], 0.1)
        self.assertEqual(case["reward_components"]["groundedness"], 1.0)

    def test_missing_machine_signal_is_preserved_as_unknown(self) -> None:
        traces = [_trace(i, "PASS" if i < 12 else "FAIL") for i in range(24)]
        traces[12]["record"].pop("combined_correct")
        cases = select_audit_cases(traces, pass_count=12, fail_count=12)
        case = next(row for row in cases if row["evidence_id"] == "E012")
        self.assertEqual(case["reviewer_combined_answerability"], "")

    def test_fails_when_a_stratum_is_too_small(self) -> None:
        with self.assertRaisesRegex(ValueError, "FAIL.*12"):
            select_audit_cases([_trace(i, "PASS") for i in range(30)], pass_count=12, fail_count=12)

    def test_absolute_clock_timeframes_are_normalized_to_clip_relative_windows(self) -> None:
        traces = [_trace(i, "PASS" if i < 12 else "FAIL") for i in range(24)]
        for trace in traces:
            raw = trace["record"]["qa"]
            raw["evidence"][0]["timeframe"] = "11:35:00-11:35:04"
            raw["referred_timestamps"][0]["timestamp_seconds"] = 41700.0
        cases = select_audit_cases(traces, pass_count=12, fail_count=12)
        self.assertTrue(all(case["video_windows"][0]["end_seconds"] <= 10.0 for case in cases))

    def test_review_template_and_summary_require_twenty_completed_rows(self) -> None:
        cases = select_audit_cases(
            [_trace(i, "PASS" if i < 20 else "FAIL") for i in range(40)],
            pass_count=12,
            fail_count=12,
        )
        reviews = build_review_rows(cases)
        for index, row in enumerate(reviews[:20]):
            row["human_groundedness"] = row["reviewer_groundedness"] if index < 17 else "UNCERTAIN"
            row["claim_visible"] = "yes"
            row["answer_supported"] = "yes"
            row["notes"] = "已观看两段视频"

        summary = summarize_reviews(reviews, approved_for_weight_change=True)
        self.assertEqual(summary["completed_count"], 20)
        self.assertEqual(summary["uncertain_count"], 3)
        self.assertEqual(summary["agreement_count"], 17)
        self.assertTrue(summary["approved_for_weight_change"])

        with self.assertRaisesRegex(ValueError, "至少完成 20"):
            summarize_reviews(reviews[:19], approved_for_weight_change=True)


if __name__ == "__main__":
    unittest.main()
