from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.evaluation.groundedness_audit import (
    _markdown,
    build_review_rows,
    export_audit,
    merge_existing_reviews,
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
        self.assertIn("reviewer_combined_answerability", reviews[0])
        self.assertIn("human_combined_answerability", reviews[0])
        self.assertIn("reviewer_speaker_leakage", reviews[0])
        self.assertIn("human_speaker_leakage", reviews[0])
        self.assertIn("reviewer_provider_answerability", reviews[0])
        self.assertIn("human_provider_answerability", reviews[0])
        self.assertIn("reviewer_qa_formality", reviews[0])
        self.assertIn("human_qa_formality", reviews[0])
        self.assertIn("reviewer_shallow_activity", reviews[0])
        self.assertIn("human_shallow_activity", reviews[0])
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

    def test_merge_existing_reviews_preserves_human_and_unknown_columns(self) -> None:
        cases = select_audit_cases(
            [_trace(i, "PASS" if i < 12 else "FAIL") for i in range(24)],
            pass_count=12,
            fail_count=12,
        )
        new_rows = build_review_rows(cases)
        old_rows = [dict(row) for row in new_rows]
        old_rows[0]["human_groundedness"] = "PASS"
        old_rows[0]["notes"] = "已经看过视频"
        old_rows[0]["custom_note"] = "保留"

        merged = merge_existing_reviews(new_rows, old_rows)

        self.assertEqual(merged[0]["human_groundedness"], "PASS")
        self.assertEqual(merged[0]["notes"], "已经看过视频")
        self.assertEqual(merged[0]["human_speaker_leakage"], "")
        self.assertEqual(merged[0]["custom_note"], "保留")

    def test_merge_existing_reviews_rejects_case_id_mismatch(self) -> None:
        cases = select_audit_cases(
            [_trace(i, "PASS" if i < 12 else "FAIL") for i in range(24)],
            pass_count=12,
            fail_count=12,
        )
        new_rows = build_review_rows(cases)
        old_rows = [dict(row) for row in new_rows[:-1]]
        with self.assertRaisesRegex(ValueError, "case_id 集合不一致"):
            merge_existing_reviews(new_rows, old_rows)

    def test_export_backs_up_and_preserves_existing_review_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace_path = root / "reward_trace.jsonl"
            trace_path.write_text(
                "".join(
                    json.dumps(_trace(index, "PASS" if index == 0 else "FAIL"), ensure_ascii=False) + "\n"
                    for index in range(2)
                ),
                encoding="utf-8",
            )
            output_dir = root / "audit"
            export_audit(trace_path, output_dir, pass_count=1, fail_count=1)
            csv_path = output_dir / "groundedness_audit_review.csv"
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["human_groundedness"] = "PASS"
            rows[0]["notes"] = "保留填写内容"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            export_audit(trace_path, output_dir, pass_count=1, fail_count=1)

            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                preserved = list(csv.DictReader(handle))
            self.assertEqual(preserved[0]["human_groundedness"], "PASS")
            self.assertEqual(preserved[0]["notes"], "保留填写内容")
            self.assertEqual(len(list(output_dir.glob("groundedness_audit_review.backup_*.csv"))), 1)

    def test_markdown_explains_and_renders_all_audit_signals(self) -> None:
        cases = select_audit_cases(
            [_trace(0, "PASS"), _trace(1, "FAIL")],
            pass_count=1,
            fail_count=1,
        )
        guide = _markdown(cases)

        self.assertIn("先独立判断，再阅读 reviewer", guide)
        self.assertIn("human_combined_answerability", guide)
        self.assertIn("human_speaker_leakage", guide)
        self.assertIn("human_provider_answerability", guide)
        self.assertIn("human_qa_formality", guide)
        self.assertIn("human_shallow_activity", guide)
        self.assertIn("Speaker leakage", guide)
        self.assertIn("Combined answerability", guide)
        self.assertIn("提问者证据", guide)
        self.assertIn("Reward components", guide)

    def test_summary_reports_multisignal_agreement_and_human_answerability_gate(self) -> None:
        cases = select_audit_cases(
            [_trace(i, "PASS" if i < 20 else "FAIL") for i in range(40)],
            pass_count=12,
            fail_count=12,
        )
        reviews = build_review_rows(cases)
        for row in reviews[:20]:
            row["human_groundedness"] = row["reviewer_groundedness"]
            row["human_combined_answerability"] = "PASS"
            row["human_speaker_leakage"] = "NO_LEAK"
            row["human_provider_answerability"] = "ANSWERABLE"
            row["human_qa_formality"] = "PASS"
            row["human_shallow_activity"] = "PASS"
        reviews[0]["human_speaker_leakage"] = "LEAK"
        reviews[1]["human_speaker_leakage"] = "UNCERTAIN"
        reviews[2]["human_combined_answerability"] = "FAIL"
        reviews[0]["human_qa_formality"] = "MAYBE"

        summary = summarize_reviews(reviews, approved_for_weight_change=False)

        self.assertEqual(summary["schema_version"], "grpo_v3_multisignal_audit_v2")
        self.assertEqual(summary["signals"]["speaker_leakage"]["completed"], 20)
        self.assertEqual(summary["signals"]["speaker_leakage"]["counts"]["LEAK"], 1)
        self.assertEqual(summary["signals"]["speaker_leakage"]["counts"]["NO_LEAK"], 18)
        self.assertEqual(summary["signals"]["qa_formality"]["completed"], 19)
        self.assertEqual(summary["signals"]["shallow_activity"]["agreement_count"], 20)
        self.assertEqual(summary["human_answerability_gate"]["derivable_count"], 20)
        self.assertEqual(summary["human_answerability_gate"]["passed"], 17)
        self.assertEqual(summary["human_answerability_gate"]["failed"], 2)
        self.assertEqual(summary["human_answerability_gate"]["uncertain"], 1)
        self.assertTrue(any(item["field"] == "human_qa_formality" for item in summary["invalid_values"]))
        self.assertFalse(summary["approved_for_weight_change"])


if __name__ == "__main__":
    unittest.main()
