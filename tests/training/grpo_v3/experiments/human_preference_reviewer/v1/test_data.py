from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.data import (
    build_split_manifest,
    load_annotation_csv,
    validate_split_manifest,
)


FIELDS = (
    "schema_version", "dataset_fingerprint", "assignment_id", "reviewer_id",
    "annotation_status", "packet_order", "evidence_id", "display_order",
    "candidate_id", "formality_score", "evidence_grounding_score",
    "answerability_score", "fea_total_score", "aggregate_rank", "packet_skipped",
    "skip_reason", "notes", "question", "options", "correct", "answer",
    "video_1_user", "video_1_source", "video_2_user", "video_2_source",
)


def rows_for(evidence_id: str, *, status: str = "completed", scored: bool = True) -> list[dict[str, str]]:
    rows = []
    for index in range(1, 7):
        formality = 1 + (index - 1) % 3
        evidence = 1 + (index - 1) % 3
        answerability = 1 + (index - 1) % 3
        options = [f"option-{letter}" for letter in "ABCDE"]
        rows.append({
            "schema_version": "egolife_rlhf_packet_ranking_v4",
            "dataset_fingerprint": "fingerprint",
            "assignment_id": "assignment",
            "reviewer_id": "HM",
            "annotation_status": status,
            "packet_order": "1",
            "evidence_id": evidence_id,
            "display_order": str(index),
            "candidate_id": f"{evidence_id}::candidate_{index:02d}",
            "formality_score": str(formality) if scored else "",
            "evidence_grounding_score": str(evidence) if scored else "",
            "answerability_score": str(answerability) if scored else "",
            "fea_total_score": str(formality + evidence + answerability) if scored else "",
            "aggregate_rank": "1" if index <= 2 and scored else (str(index) if scored else ""),
            "packet_skipped": "false",
            "skip_reason": "",
            "notes": "private note",
            "question": f"Question {index}?",
            "options": json.dumps(options),
            "correct": "A",
            "answer": options[0],
            "video_1_user": "speaker",
            "video_1_source": f"https://example.test/{evidence_id}-a.mp4",
            "video_2_user": "provider",
            "video_2_source": f"https://example.test/{evidence_id}-b.mp4",
        })
    return rows


class AnnotationDataTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, str]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "annotations.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_completed_pending_and_unscored_groups_are_separated(self) -> None:
        path = self._write(
            rows_for("eligible")
            + rows_for("pending-scored", status="pending")
            + rows_for("pending-empty", status="pending", scored=False)
        )

        audit = load_annotation_csv(path)

        self.assertEqual(tuple(row.evidence_id for row in audit.eligible_evidence), ("eligible",))
        self.assertEqual(audit.quarantined_scored_evidence_ids, ("pending-scored",))
        self.assertEqual(audit.unscored_evidence_ids, ("pending-empty",))
        self.assertEqual(audit.label_distribution["evidence_quality"], {1: 2, 2: 2, 3: 2})
        self.assertEqual(audit.eligible_evidence[0].candidates[0].overall_rank, 1)

    def test_rejects_non_six_candidate_group_and_derived_score_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            load_annotation_csv(self._write(rows_for("short")[:5]))

        rows = rows_for("bad-total")
        rows[0]["fea_total_score"] = "9"
        with self.assertRaisesRegex(ValueError, "fea_total_score"):
            load_annotation_csv(self._write(rows))

    def test_completed_scores_do_not_require_redundant_total_column(self) -> None:
        rows = rows_for("without-total")
        for row in rows:
            row["fea_total_score"] = ""

        audit = load_annotation_csv(self._write(rows))

        self.assertEqual(len(audit.eligible_evidence), 1)
        self.assertEqual(audit.label_distribution["evidence_quality"], {1: 2, 2: 2, 3: 2})

    def test_training_features_exclude_annotation_metadata(self) -> None:
        candidate = load_annotation_csv(self._write(rows_for("e1"))).eligible_evidence[0].candidates[0]

        features = candidate.model_features()

        self.assertEqual(set(features), {"question", "options", "correct", "answer"})
        serialized = json.dumps(features)
        for forbidden in ("private note", "reviewer_id", "aggregate_rank", "fea_total_score"):
            self.assertNotIn(forbidden, serialized)

    def test_split_is_deterministic_exact_and_evidence_disjoint(self) -> None:
        records = [load_annotation_csv(self._write(rows_for(f"e-{index:02d}"))).eligible_evidence[0] for index in range(6)]

        first = build_split_manifest(records, train_count=2, validation_count=2, locked_test_count=2, seed=7)
        second = build_split_manifest(records, train_count=2, validation_count=2, locked_test_count=2, seed=7)

        self.assertEqual(first, second)
        sets = [set(first[f"{name}_evidence_ids"]) for name in ("train", "validation", "locked_test")]
        self.assertEqual([len(value) for value in sets], [2, 2, 2])
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_split_rejects_insufficient_evidence(self) -> None:
        records = [load_annotation_csv(self._write(rows_for(f"e-{index}"))).eligible_evidence[0] for index in range(5)]
        with self.assertRaisesRegex(ValueError, "need at least 6"):
            build_split_manifest(records, train_count=2, validation_count=2, locked_test_count=2)

    def test_split_keeps_excess_completed_evidence_as_reserve(self) -> None:
        records = [load_annotation_csv(self._write(rows_for(f"e-{index}"))).eligible_evidence[0] for index in range(7)]

        manifest = build_split_manifest(records, train_count=2, validation_count=2, locked_test_count=2, seed=9)

        selected = sum((manifest[f"{name}_evidence_ids"] for name in ("train", "validation", "locked_test")), [])
        self.assertEqual(len(selected), 6)
        self.assertEqual(len(manifest["reserve_evidence_ids"]), 1)
        self.assertFalse(set(selected) & set(manifest["reserve_evidence_ids"]))

    def test_split_supports_sixty_ten_without_locked_test(self) -> None:
        records = [
            load_annotation_csv(self._write(rows_for(f"e-{index:02d}"))).eligible_evidence[0]
            for index in range(70)
        ]

        manifest = build_split_manifest(
            records,
            train_count=60,
            validation_count=10,
            locked_test_count=0,
            seed=42,
        )

        self.assertEqual(len(manifest["train_evidence_ids"]), 60)
        self.assertEqual(len(manifest["validation_evidence_ids"]), 10)
        self.assertEqual(manifest["locked_test_evidence_ids"], [])
        self.assertEqual(manifest["reserve_evidence_ids"], [])
        self.assertNotIn("locked_test", manifest["label_support"])
        validate_split_manifest(manifest, expected_counts=(60, 10, 0))

    def test_split_rejects_negative_locked_test_count(self) -> None:
        records = [load_annotation_csv(self._write(rows_for(f"e-{index}"))).eligible_evidence[0] for index in range(2)]

        with self.assertRaisesRegex(ValueError, "locked_test_count"):
            build_split_manifest(records, train_count=1, validation_count=1, locked_test_count=-1)

    def test_manifest_validation_rejects_cross_split_overlap(self) -> None:
        manifest = {
            "train_evidence_ids": ["e1", "e2"],
            "validation_evidence_ids": ["e2", "e3"],
            "locked_test_evidence_ids": ["e4"],
            "reserve_evidence_ids": ["e5"],
        }

        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_split_manifest(manifest)

    def test_manifest_validation_enforces_contract_and_exact_counts(self) -> None:
        manifest = {
            "contract_version": "wrong",
            "split_unit": "candidate_id",
            "csv_sha256": "HASH",
            "train_evidence_ids": ["e1"],
            "validation_evidence_ids": ["e2"],
            "locked_test_evidence_ids": ["e3"],
            "reserve_evidence_ids": [],
        }
        with self.assertRaisesRegex(ValueError, "contract_version"):
            validate_split_manifest(manifest, expected_counts=(1, 1, 1), require_contract=True)
        manifest.update(contract_version="human_preference_reviewer_absolute_v1", split_unit="evidence_id")
        validate_split_manifest(manifest, expected_counts=(1, 1, 1), require_contract=True)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            validate_split_manifest(manifest, expected_counts=(2, 1, 1), require_contract=True)

    def test_split_retries_deterministically_until_all_splits_have_support(self) -> None:
        records = []
        for index in range(9):
            rows = rows_for(f"e-{index}")
            grade = 3 if index < 3 else (2 if index < 6 else 1)
            for row in rows:
                row["answerability_score"] = str(grade)
                row["fea_total_score"] = str(
                    int(row["formality_score"]) + int(row["evidence_grounding_score"]) + grade
                )
            records.append(load_annotation_csv(self._write(rows)).eligible_evidence[0])

        first = build_split_manifest(records, train_count=3, validation_count=3, locked_test_count=3, seed=4)
        second = build_split_manifest(records, train_count=3, validation_count=3, locked_test_count=3, seed=4)

        self.assertEqual(first, second)
        self.assertGreaterEqual(first["selection_attempt"], 0)
        for split in ("train", "validation", "locked_test"):
            self.assertGreater(first["label_support"][split]["answerability"]["3"], 0)

    def test_formal_split_rejects_missing_grade_support(self) -> None:
        records = []
        for index in range(6):
            rows = rows_for(f"e-{index}")
            for row in rows:
                row["answerability_score"] = "1"
                row["fea_total_score"] = str(int(row["formality_score"]) + int(row["evidence_grounding_score"]) + 1)
            records.append(load_annotation_csv(self._write(rows)).eligible_evidence[0])
        with self.assertRaisesRegex(ValueError, "class support"):
            build_split_manifest(records, train_count=2, validation_count=2, locked_test_count=2)


if __name__ == "__main__":
    unittest.main()
