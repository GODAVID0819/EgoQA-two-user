from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.train import (
    _parser,
    load_media_map,
    select_evidence,
)
from tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_data import FIELDS, rows_for
from training.grpo_v3.experiments.human_preference_reviewer.v1.data import load_annotation_csv


class TrainContractTests(unittest.TestCase):
    def _write(self, rows: list[dict[str, str]]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "annotations.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return path
    def test_select_evidence_uses_manifest_ids_only(self) -> None:
        records = tuple(
            load_annotation_csv(self._write(rows_for(f"e{index}"))).eligible_evidence[0]
            for index in range(3)
        )
        manifest = {
            "train_evidence_ids": ["e0"],
            "validation_evidence_ids": ["e1"],
            "locked_test_evidence_ids": ["e2"],
        }
        self.assertEqual([row.evidence_id for row in select_evidence(records, manifest, "train")], ["e0"])
        self.assertEqual([row.evidence_id for row in select_evidence(records, manifest, "locked_test")], ["e2"])

    def test_media_map_requires_existing_nonempty_local_files(self) -> None:
        directory = Path(tempfile.mkdtemp())
        video = directory / "clip.mp4"
        video.write_bytes(b"video")
        mapping = directory / "media.json"
        mapping.write_text(json.dumps({"https://example.test/a.mp4": str(video)}), encoding="utf-8")

        loaded = load_media_map(mapping)

        self.assertEqual(loaded["https://example.test/a.mp4"], str(video.resolve()))
        mapping.write_text(json.dumps({"https://example.test/a.mp4": str(directory / "missing.mp4")}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or empty"):
            load_media_map(mapping)

    def test_cli_accepts_explicit_stage0_mode(self) -> None:
        args = _parser().parse_args([
            "smoke",
            "--csv", "annotations.csv",
            "--media-map", "media.json",
            "--model", "Qwen3-VL-8B-Instruct",
            "--output-dir", "out",
            "--stage", "stage0",
        ])

        self.assertEqual(args.stage, "stage0")


if __name__ == "__main__":
    unittest.main()
