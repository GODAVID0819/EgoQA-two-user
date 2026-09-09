from __future__ import annotations

import importlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


try:
    from egolife_two_user_qa import cross_user_temporal_gate_grid_sidecar as gate_grid
    from egolife_two_user_qa import temporal_kmeans_grid_sidecar as temporal
except ImportError:
    gate_grid = importlib.import_module(
        "multi-user.cross_user_temporal_gate_grid_sidecar"
    )
    temporal = importlib.import_module("multi-user.temporal_kmeans_grid_sidecar")


def _frames(*timestamps: float) -> list[dict[str, object]]:
    return [
        {"timestamp_seconds": value, "path": f"frame_{index}.jpg"}
        for index, value in enumerate(timestamps)
    ]


class CrossUserTemporalGateGridTests(unittest.TestCase):
    def test_center_and_interval_gaps_have_distinct_semantics(self) -> None:
        left = {
            "timestamp_seconds": 0.0,
            "temporal_center_seconds": 5.0,
            "member_timestamps": [0.0, 10.0],
        }
        right = {
            "timestamp_seconds": 8.0,
            "temporal_center_seconds": 13.0,
            "member_timestamps": [8.0, 18.0],
        }
        gaps = temporal.cross_cluster_temporal_gaps(left, right)
        self.assertEqual(gaps["medoid_gap_seconds"], 8.0)
        self.assertEqual(gaps["center_gap_seconds"], 8.0)
        self.assertEqual(gaps["interval_gap_seconds"], 0.0)

    def test_center_gate_rejects_far_high_cosine_pair(self) -> None:
        left_frames = _frames(0.0)
        right_frames = _frames(30.0)
        embeddings = [[1.0, 0.0]]
        left_clusters = temporal.time_aware_clustered_frame_representatives(
            left_frames,
            embeddings,
            cluster_count=1,
            time_weight=0.1,
            temporal_unit_seconds=30.0,
        )
        right_clusters = temporal.time_aware_clustered_frame_representatives(
            right_frames,
            embeddings,
            cluster_count=1,
            time_weight=0.1,
            temporal_unit_seconds=30.0,
        )
        pruning = temporal.prune_time_aware_cluster_pair(
            left_frames,
            right_frames,
            embeddings,
            embeddings,
            left_clusters,
            right_clusters,
            full_frame_matrix=[[1.0]],
            start_seconds=0.0,
            duration_seconds=31.0,
            sample_interval_seconds=1.0,
            high_similarity_threshold=0.82,
            min_pruned_video_seconds=0.0,
            pruning_protection_mode="reject",
            min_pruned_video_percent=None,
            cross_gap_mode="center",
            max_cross_gap_seconds=15.0,
        )
        self.assertEqual(pruning["ungated_high_similarity_representative_pair_count"], 1)
        self.assertEqual(pruning["high_similarity_representative_pair_count"], 0)
        self.assertEqual(pruning["cross_gap_rejected_high_similarity_pair_count"], 1)
        self.assertEqual(pruning["removed_duration_seconds"], 0.0)

    def test_interval_gate_accepts_overlapping_cluster_intervals(self) -> None:
        left_frames = _frames(0.0, 10.0)
        right_frames = _frames(8.0, 18.0)
        embeddings = [[1.0, 0.0], [1.0, 0.0]]
        left_clusters = temporal.time_aware_clustered_frame_representatives(
            left_frames,
            embeddings,
            cluster_count=1,
            time_weight=0.1,
            temporal_unit_seconds=30.0,
        )
        right_clusters = temporal.time_aware_clustered_frame_representatives(
            right_frames,
            embeddings,
            cluster_count=1,
            time_weight=0.1,
            temporal_unit_seconds=30.0,
        )
        common = {
            "left_frames": left_frames,
            "right_frames": right_frames,
            "left_embeddings": embeddings,
            "right_embeddings": embeddings,
            "left_clusters": left_clusters,
            "right_clusters": right_clusters,
            "full_frame_matrix": [[1.0, 1.0], [1.0, 1.0]],
            "start_seconds": 0.0,
            "duration_seconds": 19.0,
            "sample_interval_seconds": 1.0,
            "high_similarity_threshold": 0.82,
            "min_pruned_video_seconds": 0.0,
            "pruning_protection_mode": "reject",
            "min_pruned_video_percent": None,
            "max_cross_gap_seconds": 0.0,
        }
        center = temporal.prune_time_aware_cluster_pair(
            **common, cross_gap_mode="center"
        )
        interval = temporal.prune_time_aware_cluster_pair(
            **common, cross_gap_mode="interval"
        )
        self.assertEqual(center["high_similarity_representative_pair_count"], 0)
        self.assertEqual(interval["high_similarity_representative_pair_count"], 1)
        self.assertEqual(
            interval["high_similarity_representative_pairs"][0][
                "interval_gap_seconds"
            ],
            0.0,
        )

    def test_default_grid_has_production_temporal_and_two_gate_modes(self) -> None:
        variants = gate_grid.build_cross_user_gap_variants(
            gate_grid.DEFAULT_DURATIONS_SECONDS,
            gate_grid.DEFAULT_SECONDS_PER_CLUSTER,
            gate_grid.DEFAULT_SIMILARITY_THRESHOLDS,
            gate_grid.DEFAULT_CROSS_GAP_MODES,
            gate_grid.DEFAULT_CROSS_GAP_SECONDS,
            within_time_weight=0.1,
        )
        self.assertEqual(len(variants), 64)
        production = [
            row for row in variants if row["variant_kind"] == "production_baseline"
        ]
        temporal_ungated = [
            row for row in variants if row["variant_kind"] == "temporal_ungated"
        ]
        self.assertEqual(len(production), 4)
        self.assertEqual(len(temporal_ungated), 4)
        self.assertTrue(all(row["within_time_weight"] == 0.0 for row in production))
        self.assertTrue(
            all(row["within_time_weight"] == 0.1 for row in temporal_ungated)
        )

    def test_cached_one_pair_run_and_metric_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "source"
            source_pair = source / "pairs" / "pair_1"
            source_pair.mkdir(parents=True)
            (source / "summary.json").write_text(
                json.dumps({"model_id": "test/clip"}), encoding="utf-8"
            )
            pair = {
                "pair_id": "pair_1",
                "day": "DAY1",
                "time_token": "12000000",
                "left_agent": "A1_JAKE",
                "right_agent": "A2_ALICE",
            }
            (source / "cohort.jsonl").write_text(
                json.dumps(pair) + "\n", encoding="utf-8"
            )
            cache_metadata = {
                "model_id": "test/clip",
                "sides": [
                    _frames(0.0, 1.0, 2.0, 3.0),
                    _frames(0.0, 1.0, 2.0, 3.0),
                ],
            }
            (source_pair / "embedding_cache.json").write_text(
                json.dumps(cache_metadata), encoding="utf-8"
            )
            np.savez_compressed(
                source_pair / "embedding_cache.npz",
                left=np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
                right=np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
            )

            first_output = root / "first"
            first = gate_grid.run_cross_user_temporal_gate_grid(
                source_experiment_dir=source,
                output_dir=first_output,
                pair_count=1,
                durations_seconds=[4.0],
                seconds_per_cluster_values=[2.0],
                similarity_thresholds=[0.82],
                within_time_weight=0.1,
                cross_gap_modes=["center", "interval"],
                cross_gap_seconds=[1.0],
                temporal_unit_seconds=30.0,
                sample_interval_seconds=1.0,
                trace_pair_limit=0,
            )
            self.assertTrue(first["target_met"])
            self.assertEqual(first["metric_count"], 4)
            self.assertEqual(first["aggregate_count"], 4)

            second_output = root / "second"
            second = gate_grid.run_cross_user_temporal_gate_grid(
                source_experiment_dir=source,
                output_dir=second_output,
                pair_count=1,
                durations_seconds=[4.0],
                seconds_per_cluster_values=[2.0],
                similarity_thresholds=[0.82],
                within_time_weight=0.1,
                cross_gap_modes=["center", "interval"],
                cross_gap_seconds=[1.0],
                temporal_unit_seconds=30.0,
                sample_interval_seconds=1.0,
                trace_pair_limit=0,
                resume_experiment_dir=first_output,
            )
            self.assertTrue(second["target_met"])
            self.assertEqual(second["metric_count"], 4)
            marker = json.loads(
                (second_output / "pairs" / "pair_1" / "pair_complete.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(marker["metrics_reused"])

    def test_hpc_wrapper_selects_cross_gate_mode(self) -> None:
        root = Path(__file__).resolve().parents[1]
        wrapper = (
            root
            / "hpc"
            / "qa"
            / "preprocessing"
            / "run_cross_user_temporal_gate_grid_50.sbatch"
        )
        core = (
            root
            / "hpc"
            / "qa"
            / "preprocessing"
            / "run_temporal_kmeans_grid_50.sbatch"
        )
        wrapper_text = wrapper.read_text(encoding="utf-8")
        core_text = core.read_text(encoding="utf-8")
        self.assertNotIn(b"\r\n", wrapper.read_bytes())
        self.assertIn(
            "EGOLIFE2U_PREPROCESS_EXPERIMENT_MODE=cross_user_gate", wrapper_text
        )
        self.assertIn("run_temporal_kmeans_grid_50.sbatch", wrapper_text)
        self.assertIn("cross_user_temporal_gate_grid_sidecar", core_text)
        self.assertIn("CROSS_GATE_SOURCE_EXPERIMENT_DIR", core_text)
        self.assertIn("CROSS_GATE_RESUME_EXPERIMENT_DIR", core_text)
        self.assertIn('CROSS_GATE_GAPS="${CROSS_GATE_GAPS:-0,5,10,15,30,60,120}"', core_text)
        self.assertIn("training.torch_storage_preflight", core_text)
        self.assertIn('"${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}"', core_text)


if __name__ == "__main__":
    unittest.main()
