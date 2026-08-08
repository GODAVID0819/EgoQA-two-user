from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.train import (
    _candidate_observation,
    _controlled_overfit_gate,
    _parser,
    load_media_map,
    select_evidence,
)
from tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_data import FIELDS, rows_for
from training.grpo_v3.experiments.human_preference_reviewer.v1.data import load_annotation_csv


class TrainContractTests(unittest.TestCase):
    @staticmethod
    def _probe_metrics(
        *,
        loss: float,
        accuracy: float,
        predictions: list[int],
        losses: list[float],
        candidate_ids: list[str] | None = None,
    ) -> dict[str, object]:
        labels = [1, 2, 3, 3]
        ids = candidate_ids or [f"candidate_{index}" for index in range(4)]
        return {
            "loss": loss,
            "accuracy": accuracy,
            "candidate_results": [
                {
                    "evidence_id": f"evidence_{index // 2}",
                    "candidate_id": candidate_id,
                    "label": labels[index],
                    "prediction": predictions[index],
                    "loss": losses[index],
                    "probabilities": [0.2, 0.3, 0.5],
                }
                for index, candidate_id in enumerate(ids)
            ],
        }

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

    def test_controlled_overfit_gate_accepts_uniform_pre_post_improvement(self) -> None:
        pre = self._probe_metrics(
            loss=1.0,
            accuracy=0.5,
            predictions=[3, 3, 3, 3],
            losses=[1.2, 1.1, 0.9, 0.8],
        )
        post = self._probe_metrics(
            loss=0.2,
            accuracy=1.0,
            predictions=[1, 2, 3, 3],
            losses=[0.2, 0.3, 0.1, 0.2],
        )

        gate = _controlled_overfit_gate(pre, post)

        self.assertTrue(gate["passed"])
        self.assertEqual(gate["measurements"]["improved_candidate_count"], 4)
        self.assertAlmostEqual(gate["measurements"]["loss_reduction_ratio"], 0.8)
        self.assertEqual(gate["failures"], [])

    def test_controlled_overfit_gate_rejects_single_class_collapse(self) -> None:
        pre = self._probe_metrics(
            loss=1.0,
            accuracy=0.0,
            predictions=[2, 1, 1, 1],
            losses=[1.0, 1.0, 1.0, 1.0],
        )
        post = self._probe_metrics(
            loss=0.5,
            accuracy=0.5,
            predictions=[3, 3, 3, 3],
            losses=[0.5, 0.5, 0.5, 0.5],
        )

        gate = _controlled_overfit_gate(pre, post)

        self.assertFalse(gate["passed"])
        self.assertIn("prediction_class_count_below_minimum", gate["failures"])

    def test_controlled_overfit_gate_requires_identical_candidate_set(self) -> None:
        pre = self._probe_metrics(
            loss=1.0,
            accuracy=0.0,
            predictions=[2, 1, 1, 1],
            losses=[1.0, 1.0, 1.0, 1.0],
        )
        post = self._probe_metrics(
            loss=0.2,
            accuracy=1.0,
            predictions=[1, 2, 3, 3],
            losses=[0.2, 0.2, 0.2, 0.2],
            candidate_ids=["candidate_0", "candidate_1", "candidate_2", "different"],
        )

        with self.assertRaisesRegex(ValueError, "same candidate identities"):
            _controlled_overfit_gate(pre, post)

    def test_candidate_observation_records_identity_prediction_and_loss(self) -> None:
        result = _candidate_observation(
            evidence_id="evidence_1",
            candidate_id="candidate_4",
            label=2,
            probabilities=[0.1, 0.7, 0.2],
            loss=0.25,
        )

        self.assertEqual(result["evidence_id"], "evidence_1")
        self.assertEqual(result["candidate_id"], "candidate_4")
        self.assertEqual(result["label"], 2)
        self.assertEqual(result["prediction"], 2)
        self.assertEqual(result["probabilities"], [0.1, 0.7, 0.2])
        self.assertEqual(result["loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
