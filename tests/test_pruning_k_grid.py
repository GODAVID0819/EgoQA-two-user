from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from egolife_two_user_qa import pruning_k_grid


class _FakeEncoder:
    model_id = "fake/clip"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, image_paths: list[str]) -> list[list[float]]:
        self.calls.append(list(image_paths))
        return [
            [1.0, 0.0],
            [0.98, 0.02],
            [0.0, 1.0],
            [0.02, 0.98],
        ]


class PruningKGridTests(unittest.TestCase):
    def test_parse_k_values_preserves_order_and_deduplicates(self) -> None:
        self.assertEqual(pruning_k_grid.parse_k_values("12,4,12,30"), [12, 4, 30])
        with self.assertRaisesRegex(ValueError, "positive"):
            pruning_k_grid.parse_k_values("12,0")
        with self.assertRaisesRegex(ValueError, "at least one"):
            pruning_k_grid.parse_k_values("")

    def test_grid_reuses_pair_frames_and_embeddings_across_k(self) -> None:
        workspace_tmp = Path(__file__).resolve().parents[1] / "tmp"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        tmp_path = workspace_tmp / f"pruning_k_grid_test_{uuid.uuid4().hex}"
        tmp_path.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, tmp_path, True)
        try:
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "clips": [
                            {
                                "clip_id": "left-clip",
                                "day": "DAY1",
                                "time_token": "12000000",
                                "agent_dir": "A1_LEFT",
                                "agent_name": "LEFT",
                            },
                            {
                                "clip_id": "right-clip",
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
                source = tmp_path / f"{agent}.mp4"
                source.write_bytes(b"source")
                source_paths[agent] = source

            sampler_calls = []

            def fake_group_clip_frames(group, output_dir, **kwargs):
                sampler_calls.append((group, output_dir, kwargs))
                rows = []
                for clip in sorted(group["clips"], key=lambda row: row["agent_dir"]):
                    frames = []
                    for index in range(4):
                        frame = tmp_path / f"{clip['agent_dir']}_frame_{index}.png"
                        frame.write_bytes(b"frame")
                        frames.append({"timestamp_seconds": float(index), "path": str(frame)})
                    local_clip = {**clip, "local_video": str(source_paths[clip["agent_dir"]])}
                    rows.append({"user": clip["agent_name"], "clip": local_clip, "frames": frames})
                return rows

            materialize_calls = []

            def fake_materialize(source_video, output_video, keep_intervals, **kwargs):
                materialize_calls.append((str(source_video), str(output_video), list(keep_intervals), kwargs))
                output_path = Path(output_video)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"video")
                return output_path

            encoder = _FakeEncoder()
            output_dir = tmp_path / "grid"
            with (
                mock.patch.object(pruning_k_grid, "group_clip_frames", fake_group_clip_frames),
                mock.patch.object(pruning_k_grid, "materialize_pruned_video", fake_materialize),
            ):
                summary = pruning_k_grid.run_pruning_k_grid(
                    manifest_path=manifest_path,
                    output_dir=output_dir,
                    cache_dir=tmp_path / "cache",
                    pair_count=1,
                    k_values=[1, 2, 4],
                    duration_seconds=4.0,
                    sample_interval_seconds=1.0,
                    high_similarity_threshold=0.9,
                    min_pruned_video_seconds=2.0,
                    pruning_protection_mode="min_seconds",
                    encoder=encoder,
                )

            self.assertEqual(summary["pair_count"], 1)
            self.assertEqual(summary["variant_count"], 3)
            self.assertEqual(len(sampler_calls), 1)
            self.assertEqual(len(encoder.calls), 2)
            self.assertEqual(len(materialize_calls), 8)  # two originals plus two videos for each K
            self.assertEqual(summary["settings"]["pruning_half_width_seconds"], 0.5)
            self.assertEqual([row["k"] for row in summary["k_aggregates"]], [1, 2, 4])
            self.assertTrue((output_dir / "cohort.jsonl").exists())
            self.assertTrue((output_dir / "grid_metrics.jsonl").exists())
            self.assertTrue((output_dir / "grid_metrics.csv").exists())
            self.assertTrue((output_dir / "cluster_assignments.csv").exists())
            self.assertTrue((output_dir / "trigger_pairs.csv").exists())
            self.assertTrue((output_dir / "centroid_frames.csv").exists())
            self.assertEqual(summary["cluster_assignment_count"], 24)
            self.assertEqual(summary["centroid_frame_count"], 14)
            self.assertTrue((output_dir / "review.html").exists())
            review = (output_dir / "review.html").read_text(encoding="utf-8")
            self.assertIn("Sampled frames", review)
            self.assertIn("Cluster centers, assignments, and trigger pairs", review)
            for k in (1, 2, 4):
                variant_dir = output_dir / "pairs" / "DAY1_12000000_A1_LEFT_A2_RIGHT" / f"K_{k:02d}"
                self.assertTrue((variant_dir / "left_pruned.mp4").exists())
                self.assertTrue((variant_dir / "right_pruned.mp4").exists())
                self.assertTrue((variant_dir / "pruning.json").exists())
                trace_path = variant_dir / "cluster_trace.json"
                self.assertTrue(trace_path.exists())
                self.assertTrue((variant_dir / "centroid_frames.json").exists())
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                self.assertEqual(len(trace["left_sampled_frames"]), 4)
                self.assertEqual(
                    sum(cluster["member_count"] for cluster in trace["left_clusters"]),
                    4,
                )
                for side in ("left", "right"):
                    centroid_files = list((variant_dir / "centroid_frames" / side).glob("*.png"))
                    self.assertEqual(len(centroid_files), min(k, 4))
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
