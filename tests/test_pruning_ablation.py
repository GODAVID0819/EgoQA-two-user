from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from egolife_two_user_qa import pruning_ablation
from egolife_two_user_qa.group_relative_clip_sampling import (
    clustered_frame_representatives,
    clustered_temporal_similarity_pruning,
    score_video_pairs,
)


class _FakeEncoder:
    model_id = "fake/clip"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, image_paths: list[str]) -> list[list[float]]:
        self.calls.append(list(image_paths))
        values = []
        for index, _ in enumerate(image_paths):
            values.append([1.0, 0.0] if index < len(image_paths) // 2 else [0.0, 1.0])
        return values


class PruningAblationTests(unittest.TestCase):
    def test_300_packet_mixed_launcher_keeps_cuda_keeper_and_cohort_contract(self) -> None:
        script_path = (
            Path(__file__).resolve().parents[1]
            / "hpc"
            / "run_clip_pruned_packets_300_mixed_temporal.sbatch"
        )
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('REGULAR_PACKET_COUNT="${REGULAR_PACKET_COUNT:-200}"', script)
        self.assertIn('TEMPORAL_PACKET_COUNT="${TEMPORAL_PACKET_COUNT:-100}"', script)
        self.assertIn('PRUNING_CLUSTERS_PER_VIDEO="${PRUNING_CLUSTERS_PER_VIDEO:-12}"', script)
        self.assertIn(
            'TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS="${TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS:-9.999999}"',
            script,
        )
        self.assertIn('MIN_PRUNED_VIDEO_SECONDS="${MIN_PRUNED_VIDEO_SECONDS:-8}"', script)
        self.assertIn('PRUNING_PROTECTION_MODE="${PRUNING_PROTECTION_MODE:-min_seconds}"', script)
        self.assertIn(
            '--max-pair-time-difference-seconds "${TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS}"',
            script,
        )
        self.assertIn('echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD}', script)
        self.assertIn('trap cleanup EXIT INT TERM', script)
        self.assertIn('echo "stage=combine_and_verify_packets"', script)
        self.assertLess(
            script.index('echo "stage=start_cuda_keeper"'),
            script.index('echo "stage=prepare_regular_packets'),
        )
        self.assertLess(
            script.index('echo "stage=prepare_regular_packets'),
            script.index('echo "stage=prepare_time_sensitive_packets'),
        )
        heredocs = script.split("<<'PY'\n")[1:]
        self.assertEqual(len(heredocs), 4)
        for index, block in enumerate(heredocs):
            source = block.split("\nPY\n", 1)[0]
            compile(source, f"sbatch-heredoc-{index}", "exec")

    def test_slurm_launcher_requires_cuda_keeper_for_entire_ablation(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "hpc" / "run_pruning_ablation_30s.sbatch"
        script = script_path.read_text(encoding="utf-8")
        self.assertIn('CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"', script)
        self.assertIn('echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD}', script)
        self.assertIn('echo "stage=start_cuda_keeper"', script)
        self.assertIn('python "${CUDA_KEEPER_SCRIPT}"', script)
        self.assertIn('trap cleanup EXIT INT TERM', script)
        self.assertIn('echo "stage=stop_cuda_keeper pid=${CUDA_KEEPER_PID}"', script)
        self.assertLess(script.index('echo "stage=start_cuda_keeper"'), script.index('echo "stage=run_pruning_ablation"'))

    def test_default_variant_design_is_four_separate_sweeps(self) -> None:
        variants = pruning_ablation.build_ablation_variants(
            baseline_fps=1.0,
            baseline_k=12,
            baseline_threshold=0.82,
            baseline_temporal_policy="current",
            fps_values=[0.5, 1.0, 2.0, 4.0],
            k_values=[4, 8, 12, 16, 20, 24, 30],
            threshold_values=[0.78, 0.80, 0.82, 0.84, 0.86, 0.88],
            temporal_policies=list(pruning_ablation.DEFAULT_TEMPORAL_POLICIES),
        )
        self.assertEqual(len(variants), 23)
        self.assertEqual(
            {sweep: sum(row["sweep"] == sweep for row in variants) for sweep in {row["sweep"] for row in variants}},
            {"temporal": 6, "threshold": 6, "sampling": 4, "k": 7},
        )
        for row in variants:
            if row["sweep"] != "sampling":
                self.assertEqual(row["fps"], 1.0)
            if row["sweep"] != "k":
                self.assertEqual(row["k"], 12)
            if row["sweep"] != "threshold":
                self.assertEqual(row["high_similarity_threshold"], 0.82)
            if row["sweep"] != "temporal":
                self.assertEqual(row["temporal_policy"], "current")

    def test_time_gate_blocks_high_similarity_at_wrong_timestamp(self) -> None:
        frames = [
            {"timestamp_seconds": float(index), "path": f"frame_{index}.png"}
            for index in range(3)
        ]
        left_embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        right_embeddings = [[0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
        current = clustered_temporal_similarity_pruning(
            frames,
            frames,
            left_embeddings,
            right_embeddings,
            start_seconds=0.0,
            duration_seconds=3.0,
            sample_interval_seconds=1.0,
            cluster_count=3,
            high_similarity_threshold=0.99,
            min_pruned_video_seconds=0.0,
            pruning_protection_mode="reject",
        )
        gated = clustered_temporal_similarity_pruning(
            frames,
            frames,
            left_embeddings,
            right_embeddings,
            start_seconds=0.0,
            duration_seconds=3.0,
            sample_interval_seconds=1.0,
            cluster_count=3,
            high_similarity_threshold=0.99,
            min_pruned_video_seconds=0.0,
            pruning_protection_mode="reject",
            max_pair_time_difference_seconds=0.5,
        )
        self.assertGreater(current["high_similarity_representative_pair_count"], 0)
        self.assertTrue(
            any(
                row["timestamp_difference_seconds"] > 0.5
                for row in current["high_similarity_representative_pairs"]
            )
        )
        self.assertEqual(gated["high_similarity_representative_pair_count"], 0)
        self.assertEqual(gated["removed_duration_seconds"], 0.0)

    def test_production_pair_scoring_forwards_the_centroid_time_gate(self) -> None:
        frames = [
            {"timestamp_seconds": float(index), "path": f"frame_{index}.png"}
            for index in range(3)
        ]
        left_embeddings = [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]
        right_embeddings = [[0.0, -1.0], [-1.0, 0.0], [1.0, 0.0]]
        result = score_video_pairs(
            [{"frames": frames}, {"frames": frames}],
            [left_embeddings, right_embeddings],
            {"clip_scores": [{}, {}]},
            min_topk_sim=-1.0,
            min_mean_sim=-1.0,
            max_mean_sim=1.0,
            duration_seconds=3.0,
            sample_interval_seconds=1.0,
            pruning_clusters_per_video=3,
            high_similarity_interval_threshold=0.99,
            min_pruned_video_seconds=0.0,
            max_pair_time_difference_seconds=0.5,
        )
        self.assertEqual(result["pair_filter"]["max_pair_time_difference_seconds"], 0.5)
        pruning = result["pair_scores"][0]["temporal_pruning"]
        self.assertEqual(pruning["max_pair_time_difference_seconds"], 0.5)
        self.assertEqual(pruning["high_similarity_representative_pair_count"], 0)

    def test_contiguous_policy_splits_visual_cluster_across_time(self) -> None:
        frames = [
            {"timestamp_seconds": float(index), "path": f"frame_{index}.png"}
            for index in range(3)
        ]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        unsplit = clustered_frame_representatives(frames, embeddings, cluster_count=2)
        split = clustered_frame_representatives(
            frames,
            embeddings,
            cluster_count=2,
            split_noncontiguous_clusters=True,
            max_member_gap_seconds=1.5,
        )
        self.assertEqual(unsplit["cluster_count"], 2)
        self.assertEqual(split["visual_cluster_count"], 2)
        self.assertEqual(split["cluster_count"], 3)
        self.assertEqual(sorted(row["member_count"] for row in split["representatives"]), [1, 1, 1])

    def test_runner_reuses_dense_embeddings_and_materializes_every_sweep(self) -> None:
        workspace_tmp = Path(__file__).resolve().parents[1] / "tmp"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        tmp_path = workspace_tmp / f"pruning_ablation_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, tmp_path, True)
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "clips": [
                        {
                            "clip_id": "left",
                            "day": "DAY1",
                            "time_token": "12000000",
                            "agent_dir": "A1_LEFT",
                            "agent_name": "LEFT",
                        },
                        {
                            "clip_id": "right",
                            "day": "DAY1",
                            "time_token": "12000000",
                            "agent_dir": "A2_RIGHT",
                            "agent_name": "RIGHT",
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        source_paths = {}
        for agent in ("A1_LEFT", "A2_RIGHT"):
            path = tmp_path / f"{agent}.mp4"
            path.write_bytes(b"source")
            source_paths[agent] = path

        sampler_calls = []

        def fake_group_clip_frames(group, output_dir, **kwargs):
            sampler_calls.append((group, output_dir, kwargs))
            rows = []
            interval = float(kwargs["sample_interval_seconds"])
            count = int(round(float(kwargs["duration_seconds"]) / interval))
            for clip in sorted(group["clips"], key=lambda row: row["agent_dir"]):
                frames = []
                for index in range(count):
                    timestamp = index * interval
                    path = tmp_path / f"{clip['agent_dir']}_{index}_{timestamp:.2f}.png"
                    path.write_bytes(b"frame")
                    frames.append({"timestamp_seconds": timestamp, "path": str(path)})
                rows.append(
                    {
                        "user": clip["agent_name"],
                        "clip": {**clip, "local_video": str(source_paths[clip["agent_dir"]])},
                        "frames": frames,
                    }
                )
            return rows

        materialize_calls = []

        def fake_materialize(source_video, output_video, keep_intervals, **kwargs):
            materialize_calls.append((source_video, output_video, keep_intervals, kwargs))
            output = Path(output_video)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"video")
            return {"status": "materialized", "path": str(output), "error": None}

        encoder = _FakeEncoder()
        output_dir = tmp_path / "experiment"
        with (
            mock.patch.object(pruning_ablation, "group_clip_frames", fake_group_clip_frames),
            mock.patch.object(pruning_ablation, "_materialize", fake_materialize),
        ):
            summary = pruning_ablation.run_pruning_ablation(
                manifest_path=manifest_path,
                output_dir=output_dir,
                cache_dir=tmp_path / "cache",
                pair_count=1,
                duration_seconds=4.0,
                baseline_fps=1.0,
                baseline_k=1,
                baseline_threshold=0.8,
                fps_values=[1.0, 2.0],
                k_values=[1, 2],
                threshold_values=[0.8, 0.9],
                temporal_policies=["current", "gate_1s"],
                min_pruned_video_seconds=1.0,
                encoder=encoder,
            )

        self.assertEqual(summary["pair_count"], 1)
        self.assertEqual(summary["settings"]["configuration_count_per_pair"], 8)
        self.assertEqual(summary["variant_count"], 8)
        self.assertEqual(len(sampler_calls), 1)
        self.assertEqual(len(encoder.calls), 2)
        self.assertEqual(len(materialize_calls), 18)  # two originals plus two videos for eight variants
        for name in (
            "summary.json",
            "cohort.jsonl",
            "ablation_metrics.csv",
            "sweep_aggregates.csv",
            "cluster_assignments.csv",
            "trigger_pairs.csv",
            "centroid_frames.csv",
            "review.html",
        ):
            self.assertTrue((output_dir / name).exists(), name)
        review = (output_dir / "review.html").read_text(encoding="utf-8")
        self.assertIn("Temporal sweep", review)
        self.assertIn("Threshold sweep", review)
        self.assertIn("Sampling sweep", review)
        self.assertIn("K sweep", review)


if __name__ == "__main__":
    unittest.main()
