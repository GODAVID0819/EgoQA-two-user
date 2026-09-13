from __future__ import annotations

import json
import unittest
from pathlib import Path

from training.grpo_v3.experiments.qa_cross_view_relation.dataset_audit import audit_dataset_rows
from training.grpo_v3.experiments.qa_cross_view_relation.domain import TEXT_CHECK_NAMES
from training.grpo_v3.experiments.qa_cross_view_relation.reviewer_audit import run_audit


HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "reviewer_audit_cases.json"


class ReviewerFixtureTests(unittest.TestCase):
    def test_manual_review_fixture_preserves_31_cases_and_9_text_negatives(self):
        data = json.loads(FIXTURE.read_text(encoding="utf-8-sig"))
        cases = data["cases"]
        self.assertEqual(len(cases), 31)
        self.assertEqual(sum(case["expected_text_issue"] for case in cases), 9)
        self.assertTrue(all(len(case["options"]) == 5 for case in cases))
        self.assertTrue(
            all(bool(case["comment"]) == case["expected_text_issue"] for case in cases)
        )

    def test_reviewer_audit_batches_four_candidates_and_preserves_both_raw_passes(self):
        calls = []

        def runner(prompt):
            payload = json.loads(prompt)
            calls.append(payload)
            ids = [item["candidate_id"] for item in payload["candidates"]]
            return json.dumps(
                {
                    "candidate_scores": [
                        {
                            "candidate_id": candidate_id,
                            "cross_view_relation_score": 2,
                            "semantic_naturalness_score": 2,
                            "internal_consistency_score": 2,
                            "anchor_tier": 2,
                            "pairwise_preferences": {
                                other: "TIE" for other in ids if other != candidate_id
                            },
                            "checks": {
                                name: {"status": "PASS", "reason": "fixture"}
                                for name in TEXT_CHECK_NAMES
                            },
                            "reasons": {"summary": "fixture"},
                        }
                        for candidate_id in ids
                    ]
                }
            )

        fixture = {
            "source_job": "fixture",
            "cases": [
                {
                    "case_id": f"case_{index}",
                    "question": f"Where was object {index}?",
                    "options": ["one", "two", "three", "four", "five"],
                    "answer": "one",
                    "expected_text_issue": False,
                }
                for index in range(5)
            ],
        }
        result = run_audit(fixture, runner=runner)
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(result["groups"]), 2)
        self.assertTrue(all(len(group["candidate_ids"]) == 4 for group in result["groups"]))
        self.assertEqual(result["case_count"], 5)
        self.assertEqual(len(result["groups"][0]["raw_outputs"]), 2)
        self.assertEqual(len(result["groups"][0]["item_orders"]), 2)
        self.assertEqual(result["order_instability_rate"], 0.0)


class DatasetAuditTests(unittest.TestCase):
    @staticmethod
    def rows(count=8):
        return [
            {
                "packet_json": json.dumps({"evidence_id": f"E{index}"}),
                "evidence_id": f"E{index}",
                "question_type": "commonality" if index % 2 == 0 else "difference",
                "generation_mode": "native_video",
            }
            for index in range(count)
        ]

    def test_rejects_insufficient_evidence_or_train_heldout_overlap(self):
        with self.assertRaisesRegex(ValueError, "at least 8 distinct evidence_id"):
            audit_dataset_rows(self.rows(1), heldout_evidence_ids={"H1", "H2"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_dataset_rows(self.rows(), heldout_evidence_ids={"E1", "H2"})
        fake_heldout = [
            {
                "packet_json": json.dumps({"evidence_id": "REAL_H1"}),
                "evidence_id": "REAL_H1",
                "question_type": "commonality",
                "generation_mode": "native_video",
            },
            {
                "packet_json": json.dumps({"evidence_id": "REAL_H2"}),
                "evidence_id": "REAL_H2",
                "question_type": "difference",
                "generation_mode": "native_video",
            },
        ]
        with self.assertRaisesRegex(ValueError, "heldout dataset IDs"):
            audit_dataset_rows(
                self.rows(),
                heldout_evidence_ids={"DOES_NOT_EXIST_1", "DOES_NOT_EXIST_2"},
                heldout_rows=fake_heldout,
            )

    def test_accepts_multi_evidence_with_both_question_types(self):
        result = audit_dataset_rows(
            self.rows(),
            heldout_evidence_ids={"H1", "H2"},
        )
        self.assertEqual(result["distinct_evidence_id_count"], 8)
        self.assertEqual(result["question_type_counts"], {"commonality": 4, "difference": 4})
        self.assertEqual(result["per_evidence_group_counts"]["E0"], 1)


if __name__ == "__main__":
    unittest.main()
