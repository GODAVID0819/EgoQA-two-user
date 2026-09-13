from __future__ import annotations

import json
import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from egolife_two_user_qa import paired_evidence_pruning
from egolife_two_user_qa import group_relative_clip_sampling
from egolife_two_user_qa.video_qa_loop import clip_video_path


class _FakeEncoder:
    model_id = "fake/clip"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, image_paths: list[str]) -> list[list[float]]:
        self.calls.append(list(image_paths))
        return [[1.0, 0.0] for _ in image_paths]


class PairedEvidencePruningTests(unittest.TestCase):
    def setUp(self) -> None:
        workspace_tmp = Path(__file__).resolve().parents[1] / "tmp"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        self.tmp_path = workspace_tmp / f"paired_evidence_pruning_{uuid.uuid4().hex}"
        self.tmp_path.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp_path, True)

    def _raw_packet(self) -> dict:
        clips = []
        for side, agent in (("left", "A1_LEFT"), ("right", "A2_RIGHT")):
            source = self.tmp_path / f"{side}.mp4"
            source.write_bytes(b"source")
            clips.append(
                {
                    "clip_id": f"{side}-clip",
                    "agent_dir": agent,
                    "agent_name": side.upper(),
                    "day": "DAY1",
                    "time_token": "12000000",
                    "duration_seconds": 4.0,
                    "local_video": str(source),
                }
            )
        return {
            "evidence_id": "PAIR_1",
            "day": "DAY1",
            "time_token": "12000000",
            "duration_seconds": 4.0,
            "required_users": ["LEFT", "RIGHT"],
            "clips": clips,
        }

    def test_k80_prunes_first_and_preserves_full_video_routing(self) -> None:
        input_path = self.tmp_path / "raw.jsonl"
        input_path.write_text(json.dumps(self._raw_packet()) + "\n", encoding="utf-8")
        output_path = self.tmp_path / "pruned.jsonl"
        output_dir = self.tmp_path / "assets"

        def fake_group_clip_frames(group, output_dir, **kwargs):
            rows = []
            for clip in sorted(group["clips"], key=lambda row: row["agent_dir"]):
                frames = []
                for index in range(4):
                    frame = self.tmp_path / f"{clip['agent_dir']}_{index}.jpg"
                    frame.write_bytes(b"frame")
                    frames.append({"timestamp_seconds": index + 0.5, "path": str(frame)})
                rows.append({"user": clip["agent_name"], "clip": dict(clip), "frames": frames})
            return rows

        def fake_materialize(source_video, output_video, keep_intervals, **kwargs):
            output = Path(output_video)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"pruned")
            return output

        encoder = _FakeEncoder()
        with (
            mock.patch.object(
                paired_evidence_pruning,
                "group_clip_frames",
                fake_group_clip_frames,
            ),
            mock.patch.object(paired_evidence_pruning, "ffprobe_duration", return_value=None),
            mock.patch.object(
                group_relative_clip_sampling,
                "materialize_pruned_video",
                fake_materialize,
            ),
        ):
            summary = paired_evidence_pruning.prune_prepared_evidence_pairs(
                evidence_path=input_path,
                output_path=output_path,
                output_dir=output_dir,
                cache_dir=self.tmp_path / "cache",
                expected_duration_seconds=4.0,
                duration_tolerance_seconds=0.1,
                sample_interval_seconds=1.0,
                clusters_per_video=80,
                high_similarity_threshold=0.82,
                min_pruned_video_seconds=2.0,
                pruning_protection_mode="min_seconds",
                clip_batch_size=2,
                encoder=encoder,
            )

        packet = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(summary["packet_count"], 1)
        self.assertEqual(summary["settings"]["clusters_per_video"], 80)
        self.assertEqual(packet["candidate_type"], "ten_minute_k80_pruned_pair")
        self.assertEqual(packet["paired_video_pruning"]["clusters_per_video"], 80)
        self.assertEqual(len(packet["clips"]), 2)
        for clip in packet["clips"]:
            self.assertEqual(clip["generator_media_mode"], "pruned_video")
            self.assertTrue(Path(clip["local_video"]).is_file())
            self.assertTrue(Path(clip["full_local_video"]).is_file())
            self.assertEqual(clip["full_local_video"], clip["original_local_video"])
            self.assertNotEqual(clip["local_video"], clip["full_local_video"])
            self.assertNotIn("cluster_decisions", clip["temporal_pruning"])
            self.assertEqual(clip_video_path(clip, media_role="generator"), clip["local_video"])
            self.assertEqual(clip_video_path(clip, media_role="full"), clip["full_local_video"])
        self.assertEqual(len(encoder.calls), 4)
        self.assertTrue(all(len(call) <= 2 for call in encoder.calls))
        self.assertTrue((output_dir / "pruning_summary.json").is_file())
        self.assertTrue(Path(packet["paired_video_pruning"]["diagnostics_path"]).is_file())

    def test_rejects_any_packet_that_is_not_exactly_two_videos(self) -> None:
        packet = self._raw_packet()
        packet["clips"] = packet["clips"][:1]
        with self.assertRaisesRegex(ValueError, "exactly two videos"):
            paired_evidence_pruning._validate_raw_two_video_packet(
                packet,
                expected_duration_seconds=4.0,
                duration_tolerance_seconds=0.1,
            )


if __name__ == "__main__":
    unittest.main()
