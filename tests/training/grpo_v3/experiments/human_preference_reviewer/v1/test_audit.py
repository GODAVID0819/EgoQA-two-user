from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.audit import annotation_audit_report, build_media_map
from tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_data import FIELDS, rows_for


class AuditTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, str]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "annotations.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path
    def test_report_exposes_formal_split_blocker_without_discarding_audit(self) -> None:
        path = self._write(rows_for("e1"))

        report = annotation_audit_report(path, train_count=40, validation_count=10, locked_test_count=10)

        self.assertEqual(report["status"], "insufficient_data")
        self.assertEqual(report["formal_split_gate"]["required_evidence_count"], 60)
        self.assertEqual(report["formal_split_gate"]["eligible_evidence_count"], 1)
        self.assertEqual(report["formal_split_gate"]["missing_evidence_count"], 59)

    def test_cli_defaults_use_sixty_ten_without_locked_test(self) -> None:
        from training.grpo_v3.experiments.human_preference_reviewer.v1.audit import _parser

        args = _parser().parse_args(["annotation-csv", "--csv", "annotations.csv"])

        self.assertEqual(args.train_evidence_count, 60)
        self.assertEqual(args.validation_evidence_count, 10)
        self.assertEqual(args.locked_test_evidence_count, 0)

    def test_media_map_resolves_huggingface_relative_paths(self) -> None:
        rows = rows_for("e1")
        rows[0]["video_1_source"] = "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A1/DAY5/a.mp4"
        for row in rows:
            row["video_1_source"] = rows[0]["video_1_source"]
            row["video_2_source"] = "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/A2/DAY5/b.mp4"
        csv_path = self._write(rows)
        root = csv_path.parent / "EgoLife"
        (root / "A1/DAY5").mkdir(parents=True)
        (root / "A2/DAY5").mkdir(parents=True)
        (root / "A1/DAY5/a.mp4").write_bytes(b"a")
        (root / "A2/DAY5/b.mp4").write_bytes(b"b")

        mapping = build_media_map(csv_path, root)

        self.assertEqual(len(mapping), 2)
        self.assertEqual(Path(mapping[rows[0]["video_1_source"]]).parts[-3:], ("A1", "DAY5", "a.mp4"))


if __name__ == "__main__":
    unittest.main()
