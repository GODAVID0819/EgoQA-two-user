from __future__ import annotations

import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


try:
    from egolife_two_user_qa import temporal_kmeans_grid_sidecar as sidecar
    from egolife_two_user_qa.clip_gap_demo import cluster_embedding_medoids
except ImportError:
    # This research copy lives in a local directory named ``multi-user`` before
    # it is synchronized into the cluster's egolife_two_user_qa package.
    sidecar = importlib.import_module("multi-user.temporal_kmeans_grid_sidecar")
    cluster_embedding_medoids = importlib.import_module(
        "multi-user.clip_gap_demo"
    ).cluster_embedding_medoids


class TemporalKMeansSidecarTests(unittest.TestCase):
    def test_temporal_penalty_grows_quadratically_with_gap(self) -> None:
        close = sidecar.combined_cluster_distance(
            1.0,
            5.0,
            time_weight=1.0,
            temporal_unit_seconds=30.0,
        )
        far = sidecar.combined_cluster_distance(
            1.0,
            10.0,
            time_weight=1.0,
            temporal_unit_seconds=30.0,
        )
        self.assertAlmostEqual(close, (5.0 / 30.0) ** 2)
        self.assertAlmostEqual(far, 4.0 * close)

    def test_zero_weight_matches_current_cosine_clustering(self) -> None:
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.98, 0.2, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.98, 0.2],
            [0.0, 0.0, 1.0],
            [0.2, 0.0, 0.98],
        ]
        expected = cluster_embedding_medoids(embeddings, 3)
        labels, medoids, diagnostics = sidecar.temporal_spherical_kmeans_medoids(
            embeddings,
            [0.0, 1.0, 10.0, 11.0, 20.0, 21.0],
            3,
            time_weight=0.0,
        )
        self.assertEqual((labels, medoids), expected)
        self.assertEqual(diagnostics["time_weight"], 0.0)

    def test_identical_visual_frames_split_into_early_and_late_clusters(self) -> None:
        embeddings = [[1.0, 0.0] for _ in range(4)]
        labels, medoids, _ = sidecar.temporal_spherical_kmeans_medoids(
            embeddings,
            [0.0, 1.0, 29.0, 30.0],
            2,
            time_weight=4.0,
            temporal_unit_seconds=30.0,
        )
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])
        self.assertEqual(set(medoids), {0, 2})

    def test_time_translation_does_not_change_assignments(self) -> None:
        embeddings = [
            [1.0, 0.0],
            [0.99, 0.01],
            [1.0, 0.0],
            [0.99, 0.01],
        ]
        first = sidecar.temporal_spherical_kmeans_medoids(
            embeddings,
            [0.0, 1.0, 59.0, 60.0],
            2,
            time_weight=2.0,
            temporal_unit_seconds=30.0,
        )[:2]
        shifted = sidecar.temporal_spherical_kmeans_medoids(
            embeddings,
            [600.0, 601.0, 659.0, 660.0],
            2,
            time_weight=2.0,
            temporal_unit_seconds=30.0,
        )[:2]
        self.assertEqual(first, shifted)

    def test_time_aware_clusters_feed_existing_pruning_semantics(self) -> None:
        frames = [
            {"timestamp_seconds": value, "path": f"frame_{index}.png"}
            for index, value in enumerate((0.0, 1.0, 29.0, 30.0))
        ]
        embeddings = [[1.0, 0.0] for _ in frames]
        clusters = sidecar.time_aware_clustered_frame_representatives(
            frames,
            embeddings,
            cluster_count=2,
            time_weight=4.0,
            temporal_unit_seconds=30.0,
        )
        pruning = sidecar.prune_time_aware_cluster_pair(
            frames,
            frames,
            embeddings,
            embeddings,
            clusters,
            clusters,
            full_frame_matrix=[[1.0 for _ in frames] for _ in frames],
            start_seconds=0.0,
            duration_seconds=31.0,
            sample_interval_seconds=1.0,
            high_similarity_threshold=0.82,
            min_pruned_video_seconds=0.0,
            pruning_protection_mode="reject",
            min_pruned_video_percent=None,
        )
        self.assertEqual(pruning["method"], "temporal_kmeans_sidecar_clip_pruning_v1")
        self.assertEqual(pruning["left_marked_frame_indices"], [0, 1, 2, 3])
        self.assertEqual(pruning["right_marked_frame_indices"], [0, 1, 2, 3])
        self.assertTrue(pruning["passed"])

    def test_default_grid_has_four_durations_and_current_controls(self) -> None:
        variants = sidecar.build_grid_variants(
            sidecar.DEFAULT_DURATIONS_SECONDS,
            sidecar.DEFAULT_SECONDS_PER_CLUSTER,
            sidecar.DEFAULT_TIME_WEIGHTS,
            sidecar.DEFAULT_SIMILARITY_THRESHOLDS,
        )
        expected = (
            len(sidecar.DEFAULT_DURATIONS_SECONDS)
            * len(sidecar.DEFAULT_SECONDS_PER_CLUSTER)
            * len(sidecar.DEFAULT_TIME_WEIGHTS)
            * len(sidecar.DEFAULT_SIMILARITY_THRESHOLDS)
        )
        self.assertEqual(len(variants), expected)
        controls = [row for row in variants if row["time_weight"] == 0.0]
        self.assertEqual(
            len(controls),
            len(sidecar.DEFAULT_DURATIONS_SECONDS)
            * len(sidecar.DEFAULT_SECONDS_PER_CLUSTER),
        )
        current_density = next(
            row
            for row in controls
            if row["duration_seconds"] == 30.0
            and row["seconds_per_cluster"] == 2.5
        )
        self.assertEqual(current_density["k"], 12)
        ten_minute_density = next(
            row
            for row in controls
            if row["duration_seconds"] == 600.0
            and row["seconds_per_cluster"] == 2.5
        )
        self.assertEqual(ten_minute_density["k"], 240)

    def test_complete_embedding_cache_can_be_loaded_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            pair_dir = Path(temporary_directory) / "pair"
            pair_dir.mkdir()
            metadata = {
                "model_id": "test/clip",
                "sides": [
                    [
                        {"timestamp_seconds": float(index), "path": f"left_{index}.jpg"}
                        for index in range(4)
                    ],
                    [
                        {"timestamp_seconds": float(index), "path": f"right_{index}.jpg"}
                        for index in range(4)
                    ],
                ],
            }
            (pair_dir / "embedding_cache.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            np.savez_compressed(
                pair_dir / "embedding_cache.npz",
                left=np.asarray([[1.0, 0.0]] * 4, dtype=np.float32),
                right=np.asarray([[0.0, 1.0]] * 4, dtype=np.float32),
            )

            frames, embeddings, loaded_metadata = sidecar._load_pair_embedding_cache(
                pair_dir,
                expected_model_id="test/clip",
                expected_frame_count=4,
            )

            self.assertEqual([len(side) for side in frames], [4, 4])
            self.assertEqual([array.shape for array in embeddings], [(4, 2), (4, 2)])
            self.assertEqual(loaded_metadata["model_id"], "test/clip")

    def test_resume_recomputes_grid_and_checkpoints_without_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "day": "DAY1",
                                "time_token": "12000000",
                                "agent_dir": "A1_JAKE",
                            },
                            {
                                "day": "DAY1",
                                "time_token": "12000000",
                                "agent_dir": "A2_ALICE",
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            pair_id = "DAY1_12000000_A1_JAKE_A2_ALICE_temporal_kmeans"
            source_pair = root / "old_pairs" / pair_id
            source_pair.mkdir(parents=True)
            metadata = {
                "model_id": "test/clip",
                "sides": [
                    [
                        {"timestamp_seconds": float(index), "path": f"left_{index}.jpg"}
                        for index in range(30)
                    ],
                    [
                        {"timestamp_seconds": float(index), "path": f"right_{index}.jpg"}
                        for index in range(30)
                    ],
                ],
            }
            (source_pair / "embedding_cache.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            vectors = np.asarray(
                [[1.0, 0.0] if index < 15 else [0.0, 1.0] for index in range(30)],
                dtype=np.float32,
            )
            np.savez_compressed(
                source_pair / "embedding_cache.npz", left=vectors, right=vectors
            )

            output_dir = root / "output"
            summary = sidecar.run_temporal_kmeans_grid(
                manifest_path=manifest_path,
                output_dir=output_dir,
                cache_dir=root / "unused_video_cache",
                pair_count=1,
                durations_seconds=[30.0],
                time_weights=[0.0, 1.0],
                seconds_per_cluster_values=[15.0],
                similarity_thresholds=[0.82],
                temporal_unit_seconds=30.0,
                sample_interval_seconds=1.0,
                model_id="test/clip",
                trace_pair_limit=0,
                resume_pairs_dir=root / "old_pairs",
                encoder=None,
            )

            progress = json.loads(
                (output_dir / "progress.json").read_text(encoding="utf-8")
            )
            self.assertTrue(summary["target_met"])
            self.assertEqual(summary["metric_count"], 2)
            self.assertEqual(summary["resume_cache_pairs_reused"], 1)
            self.assertEqual(progress["status"], "complete")
            self.assertEqual(progress["pair_count_completed"], 1)
            self.assertTrue(
                (output_dir / "pairs" / pair_id / "pair_complete.json").is_file()
            )
            self.assertTrue((output_dir / "grid_metrics.csv").is_file())

            second_output_dir = root / "second_output"
            with mock.patch.object(
                sidecar,
                "time_aware_clustered_frame_representatives",
                side_effect=AssertionError("completed checkpoint should not be recomputed"),
            ):
                second_summary = sidecar.run_temporal_kmeans_grid(
                    manifest_path=manifest_path,
                    output_dir=second_output_dir,
                    cache_dir=root / "unused_video_cache_2",
                    pair_count=1,
                    durations_seconds=[30.0],
                    time_weights=[0.0, 1.0],
                    seconds_per_cluster_values=[15.0],
                    similarity_thresholds=[0.82],
                    temporal_unit_seconds=30.0,
                    sample_interval_seconds=1.0,
                    model_id="test/clip",
                    trace_pair_limit=0,
                    resume_pairs_dir=output_dir / "pairs",
                    encoder=None,
                )
            self.assertEqual(second_summary["metric_count"], 2)
            self.assertEqual(second_summary["resume_cache_pairs_reused"], 1)

    def test_aggregate_reports_matched_zero_weight_deltas_and_pareto(self) -> None:
        base = {
            "duration_seconds": 30.0,
            "seconds_per_cluster": 2.5,
            "k": 12,
            "temporal_unit_seconds": 30.0,
            "high_similarity_threshold": 0.82,
            "passed": True,
            "no_removal": False,
            "mean_cluster_span_seconds": 20.0,
            "p95_cluster_span_seconds": 28.0,
            "max_cluster_span_seconds": 29.0,
            "mean_member_medoid_gap_seconds": 8.0,
            "p95_member_medoid_gap_seconds": 20.0,
            "member_gap_gt_unit_fraction": 0.2,
            "member_gap_gt_quarter_duration_fraction": 0.4,
            "mean_member_medoid_visual_similarity": 0.95,
            "pruned_member_gap_gt_unit_fraction": 0.3,
            "pruned_member_gap_gt_quarter_duration_fraction": 0.5,
            "mean_removed_percent": 40.0,
            "left_keep_segment_count": 2,
            "right_keep_segment_count": 2,
            "high_similarity_representative_pair_count": 4,
        }
        rows = [
            {**base, "pair_id": "p1", "time_weight": 0.0},
            {
                **base,
                "pair_id": "p1",
                "time_weight": 1.0,
                "member_gap_gt_unit_fraction": 0.05,
                "member_gap_gt_quarter_duration_fraction": 0.1,
                "mean_member_medoid_visual_similarity": 0.94,
                "pruned_member_gap_gt_unit_fraction": 0.1,
                "pruned_member_gap_gt_quarter_duration_fraction": 0.2,
            },
        ]
        aggregates = sidecar.aggregate_metrics(rows)
        weighted = next(row for row in aggregates if row["time_weight"] == 1.0)
        self.assertAlmostEqual(weighted["member_gap_reduction_vs_w0"], 0.15)
        self.assertAlmostEqual(weighted["quarter_duration_gap_reduction_vs_w0"], 0.3)
        self.assertAlmostEqual(weighted["visual_similarity_delta_vs_w0"], -0.01)
        self.assertAlmostEqual(weighted["pruned_gap_reduction_vs_w0"], 0.2)
        self.assertAlmostEqual(
            weighted["pruned_quarter_duration_gap_reduction_vs_w0"], 0.3
        )
        self.assertTrue(weighted["temporal_visual_pareto"])

    def test_hpc_launcher_has_fixed_cohort_and_runtime_preflights(self) -> None:
        root = Path(__file__).resolve().parents[1]
        launcher = (
            root
            / "hpc"
            / "qa"
            / "preprocessing"
            / "run_temporal_kmeans_grid_50.sbatch"
        )
        script = launcher.read_text(encoding="utf-8")
        keeper_path = root / "hpc" / "shared" / "cuda.py"
        keeper_script = keeper_path.read_text(encoding="utf-8")
        environment_script = (root / "hpc" / "shared" / "env_qwen3vl.sh").read_bytes()
        self.assertNotIn(b"\r\n", launcher.read_bytes())
        self.assertNotIn(b"\r\n", keeper_path.read_bytes())
        self.assertNotIn(b"\r\n", environment_script)
        self.assertIn('PAIR_COUNT="${TEMPORAL_KMEANS_PAIR_COUNT:-50}"', script)
        self.assertIn(
            'RESUME_PAIRS_DIR="${TEMPORAL_KMEANS_RESUME_PAIRS_DIR:-}"', script
        )
        self.assertIn('--resume-pairs-dir "${RESUME_PAIRS_DIR}"', script)
        self.assertIn("error=resume_pairs_dir_missing", script)
        self.assertIn("error=resume_embedding_caches_missing", script)
        self.assertIn('echo "progress=${OUTPUT_DIR}/experiment/progress.json"', script)
        self.assertIn('DURATIONS="${TEMPORAL_KMEANS_DURATIONS:-30,180,360,600}"', script)
        self.assertIn(
            '${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/temporal_kmeans_grid_50',
            script,
        )
        self.assertIn("training.torch_storage_preflight", script)
        self.assertIn("from torchcodec.decoders import VideoDecoder", script)
        self.assertIn('FFMPEG_ENV="${FFMPEG_ENV:-', script)
        self.assertIn('FFMPEG_BINARY="$(command -v ffmpeg || true)"', script)
        self.assertIn('FFPROBE_BINARY="$(command -v ffprobe || true)"', script)
        self.assertIn('${CONDA_PREFIX}/bin/ffmpeg', script)
        self.assertIn('"${FFMPEG_BINARY}" -version', script)
        self.assertIn('"${FFPROBE_BINARY}" -version', script)
        self.assertIn('--ffmpeg-binary "${FFMPEG_BINARY}"', script)
        self.assertIn('error=ffmpeg_runtime_not_found', script)
        self.assertIn('JOB_SCRATCH_ROOT="/scratch/${USER}/', script)
        self.assertIn(
            'CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/shared/cuda.py}"',
            script,
        )
        self.assertIn('echo "stage=start_cuda_keeper"', script)
        self.assertIn('trap cleanup EXIT INT TERM', script)
        self.assertIn('--threshold "${CUDA_KEEPER_THRESHOLD}"', script)
        self.assertIn('--gpus "${CUDA_KEEPER_GPUS}"', script)
        self.assertIn('--reserve "${CUDA_KEEPER_RESERVE}"', script)
        self.assertIn('cuda_keeper_exited_early', script)
        self.assertIn('torch.cuda.is_available()', script)
        self.assertIn('pynvml.nvmlInit()', script)
        self.assertIn("cuda_keeper_log_begin", script)
        self.assertIn("PYTHONUNBUFFERED=1", script)
        self.assertIn('"${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}"', script)
        self.assertIn("_SCRIPT_DIR = os.path.realpath(os.path.dirname(__file__))", keeper_script)
        self.assertIn("sys.path[:] = [", keeper_script)
        self.assertIn('help="Util threshold (%%)"', keeper_script)
        self.assertLess(
            keeper_script.index("sys.path[:] = ["),
            keeper_script.index("    import torch"),
        )
        self.assertLess(
            script.index("training.torch_storage_preflight"),
            script.index("stage=start_cuda_keeper"),
        )
        self.assertLess(
            script.index("stage=start_cuda_keeper"),
            script.index("stage=run_temporal_kmeans_grid"),
        )


if __name__ == "__main__":
    unittest.main()
