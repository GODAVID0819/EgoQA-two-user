from __future__ import annotations

import math
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa import cross_user_temporal_gate_grid_sidecar as gate_grid  # noqa: E402
from egolife_two_user_qa import group_relative_clip_sampling as sampling  # noqa: E402
from egolife_two_user_qa import temporal_kmeans_grid_sidecar as sidecar  # noqa: E402


class ZipTemporalPruningTests(unittest.TestCase):
    def test_ten_minute_path_uses_k240_and_explicit_thirty_second_cross_gap(self) -> None:
        frames_by_video = [
            [{"timestamp_seconds": 0.0, "path": f"user-{index}.jpg"}]
            for index in range(6)
        ]
        embeddings_by_video = [[[1.0, 0.0]] for _ in range(6)]

        with (
            mock.patch.object(
                sidecar,
                "time_aware_clustered_frame_representatives",
                return_value={"cluster_count": 1, "representatives": []},
            ) as cluster,
            mock.patch.object(
                sidecar,
                "prune_time_aware_cluster_pair",
                return_value={
                    "high_similarity_representative_pairs": [],
                    "right_marked_frame_indices": [],
                    "right_remove_intervals": [],
                    "right_keep_intervals": [[0.0, 600.0]],
                    "right_kept_duration_seconds": 600.0,
                    "right_removed_duration_seconds": 0.0,
                    "passed": True,
                },
            ) as prune,
        ):
            result = sampling.clustered_six_user_zip_temporal_pruning(
                frames_by_video,
                embeddings_by_video,
                speaker_index=0,
                start_seconds=0.0,
                duration_seconds=600.0,
                sample_interval_seconds=1.0,
                seconds_per_cluster=2.5,
                cross_gap_mode="center",
                max_cross_gap_seconds=30.0,
            )

        self.assertEqual(result["cluster_count"], 240)
        self.assertEqual(result["max_cross_gap_seconds"], 30.0)
        self.assertEqual(cluster.call_count, 6)
        self.assertTrue(
            all(call.kwargs["cluster_count"] == 240 for call in cluster.call_args_list)
        )
        self.assertEqual(prune.call_count, 5)
        self.assertTrue(
            all(
                call.kwargs["max_cross_gap_seconds"] == 30.0
                for call in prune.call_args_list
            )
        )

    def test_zero_weight_matches_current_cosine_clustering(self) -> None:
        embeddings = [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ]

        sidecar.assert_zero_weight_compatibility(
            embeddings,
            2,
            timestamps_seconds=[0.0, 1.0, 20.0, 21.0],
        )

    def test_production_weight_splits_identical_early_and_late_frames(self) -> None:
        labels, medoids, diagnostics = sidecar.temporal_spherical_kmeans_medoids(
            [[1.0, 0.0] for _ in range(4)],
            [0.0, 1.0, 29.0, 30.0],
            2,
            time_weight=0.1,
            temporal_unit_seconds=30.0,
        )

        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(set(medoids), {0, 2})
        self.assertEqual(diagnostics["time_weight"], 0.1)

    def test_center_gate_uses_inclusive_ten_second_boundary(self) -> None:
        variants = gate_grid.build_cross_user_gap_variants(
            [180.0],
            [2.5],
            [0.82],
            ["center"],
            [10.0],
            within_time_weight=0.1,
        )

        selected = [
            row
            for row in variants
            if row["variant_kind"] == "temporal_center_gate"
        ]
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["k"], 72)
        self.assertEqual(selected[0]["max_cross_gap_seconds"], 10.0)
        self.assertEqual(selected[0]["within_time_weight"], 0.1)

    def test_center_gap_is_distinct_from_interval_gap(self) -> None:
        left = {
            "timestamp_seconds": 2.0,
            "member_timestamps": [0.0, 4.0],
            "temporal_center_seconds": 2.0,
        }
        right = {
            "timestamp_seconds": 10.0,
            "member_timestamps": [4.0, 16.0],
            "temporal_center_seconds": 10.0,
        }

        gaps = sidecar.cross_cluster_temporal_gaps(left, right)

        self.assertEqual(gaps["center_gap_seconds"], 8.0)
        self.assertEqual(gaps["interval_gap_seconds"], 0.0)

    def test_similarity_and_center_gap_boundaries_are_inclusive(self) -> None:
        right_vector = [0.82, math.sqrt(1.0 - 0.82**2)]
        left_frames = [{"timestamp_seconds": 0.0, "path": "left.jpg"}]
        right_frames = [{"timestamp_seconds": 10.0, "path": "right.jpg"}]
        left_embeddings = [[1.0, 0.0]]
        right_embeddings = [right_vector]
        left_clusters = sidecar.time_aware_clustered_frame_representatives(
            left_frames,
            left_embeddings,
            cluster_count=1,
            time_weight=0.1,
            temporal_unit_seconds=30.0,
        )
        right_clusters = sidecar.time_aware_clustered_frame_representatives(
            right_frames,
            right_embeddings,
            cluster_count=1,
            time_weight=0.1,
            temporal_unit_seconds=30.0,
        )

        pruning = sidecar.prune_time_aware_cluster_pair(
            left_frames,
            right_frames,
            left_embeddings,
            right_embeddings,
            left_clusters,
            right_clusters,
            full_frame_matrix=[[0.82]],
            start_seconds=0.0,
            duration_seconds=20.0,
            sample_interval_seconds=1.0,
            high_similarity_threshold=0.82,
            min_pruned_video_seconds=0.0,
            pruning_protection_mode="min_percent",
            min_pruned_video_percent=20.0,
            cross_gap_mode="center",
            max_cross_gap_seconds=10.0,
        )

        self.assertEqual(pruning["high_similarity_representative_pair_count"], 1)
        self.assertEqual(
            pruning["high_similarity_representative_pairs"][0][
                "selected_cross_gap_seconds"
            ],
            10.0,
        )
        self.assertTrue(pruning["passed"])

    def test_unified_cli_propagates_zip_temporal_parameters(self) -> None:
        from egolife_two_user_qa import cli

        with mock.patch.object(
            cli,
            "mine_group_relative_clip_candidates",
            return_value=[],
        ) as mine:
            result = cli.main(
                [
                    "prepare_clip_pruned_benchmark",
                    "--manifest",
                    "manifest.json",
                    "--output",
                    "candidates.jsonl",
                    "--output-dir",
                    "output",
                    "--selected-count",
                    "6",
                    "--pruning-seconds-per-cluster",
                    "3",
                    "--pruning-time-weight",
                    "0.2",
                    "--pruning-temporal-unit-seconds",
                    "20",
                    "--pruning-max-iterations",
                    "7",
                    "--pruning-cross-gap-mode",
                    "interval",
                    "--pruning-max-cross-gap-seconds",
                    "15",
                    "--min-pruned-video-percent",
                    "25",
                ]
            )

        self.assertEqual(result, 0)
        kwargs = mine.call_args.kwargs
        self.assertEqual(kwargs["pruning_seconds_per_cluster"], 3.0)
        self.assertEqual(kwargs["pruning_time_weight"], 0.2)
        self.assertEqual(kwargs["pruning_temporal_unit_seconds"], 20.0)
        self.assertEqual(kwargs["pruning_max_iterations"], 7)
        self.assertEqual(kwargs["pruning_cross_gap_mode"], "interval")
        self.assertEqual(kwargs["pruning_max_cross_gap_seconds"], 15.0)
        self.assertEqual(kwargs["min_pruned_video_percent"], 25.0)


if __name__ == "__main__":
    unittest.main()
