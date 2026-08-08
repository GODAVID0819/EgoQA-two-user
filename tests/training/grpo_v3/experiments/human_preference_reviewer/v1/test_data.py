from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.data import (
    build_split_manifest,
    load_annotation_csv,
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

    def test_training_features_exclude_annotation_metadata(self) -> None:
        candidate = load_annotation_csv(self._write(rows_for("e1"))).eligible_evidence[0].candidates[0]

        features = candidate.model_features()

        self.assertEqual(set(features), {"candidate_id", "question", "options", "correct", "answer"})
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
        with self.assertRaisesRegex(ValueError, "need exactly 6"):
            build_split_manifest(records, train_count=2, validation_count=2, locked_test_count=2)


if __name__ == "__main__":
    unittest.main()
