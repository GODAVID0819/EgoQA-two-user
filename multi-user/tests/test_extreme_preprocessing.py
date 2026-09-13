from __future__ import annotations

import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa import evidence  # noqa: E402
from egolife_two_user_qa import ten_minute_six_user_setup as setup  # noqa: E402


def clips(count: int) -> list[dict[str, object]]:
    return [
        {
            "agent_dir": f"A{index}",
            "agent_id": f"A{index}",
            "agent_name": f"User {index}",
            "day": "DAY1",
            "time_token": "12000000",
            "video_url": f"https://example.invalid/A{index}.mp4",
            "gaze_url": f"https://example.invalid/A{index}.csv",
        }
        for index in range(1, count + 1)
    ]


class ExtremePreprocessingTests(unittest.TestCase):
    def test_launchers_enable_managed_media_and_extreme_preprocessing(self) -> None:
        runtime = (
            REPOSITORY_ROOT
            / "hpc/qa/production/run_six_user_qa_10min_sequential_0p5_fresh30.sbatch"
        ).read_text(encoding="utf-8")
        wrapper = (
            REPOSITORY_ROOT
            / "hpc/qa/experiments/run_six_user_qa_fresh_preprocessing_ab_0p5.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'VLLM_OPENAI_USE_LOCAL_MEDIA_URIS="${VLLM_OPENAI_USE_LOCAL_MEDIA_URIS:-${START_VLLM_SERVER}}"',
            runtime,
        )
        self.assertIn('--media-prepare-workers "${PREPROCESSING_MEDIA_PREPARE_WORKERS}"', runtime)
        self.assertIn('PROFILE_CLIP_BATCHES=("600" "32" "600")', wrapper)
        self.assertIn('PROFILE_VIDEO_WORKERS=("6" "1" "6")', wrapper)
        self.assertIn('PROFILE_MEDIA_PREPARE_WORKERS=("6" "1" "6")', wrapper)
        self.assertIn(
            '"extreme_reuse_p6_g4_r5_e2e"', wrapper
        )
        self.assertIn(
            'export VLLM_OPENAI_USE_LOCAL_MEDIA_URIS="${start_server}"', wrapper
        )

    def test_user_media_preparation_is_parallel_and_ordered(self) -> None:
        active = 0
        peak_active = 0
        lock = threading.Lock()

        def fake_prepare(clip, **kwargs):
            del kwargs
            nonlocal active, peak_active
            with lock:
                active += 1
                peak_active = max(peak_active, active)
            time.sleep(0.04)
            with lock:
                active -= 1
            return {
                "agent_dir": clip["agent_dir"],
                "agent_name": clip["agent_name"],
            }

        group_clips = clips(6)
        with mock.patch.object(
            evidence, "_prepare_evidence_clip", side_effect=fake_prepare
        ):
            packet = evidence.build_evidence_packet(
                {
                    "day": "DAY1",
                    "time_token": "12000000",
                    "duration_seconds": 600.0,
                    "clips": list(reversed(group_clips)),
                },
                cache_dir="cache",
                output_root="output",
                users_per_case=6,
                media_prepare_workers=6,
            )

        self.assertEqual(peak_active, 6)
        self.assertEqual(
            [clip["agent_dir"] for clip in packet["clips"]],
            [f"A{index}" for index in range(1, 7)],
        )

    def test_incomplete_groups_are_skipped_before_media_work(self) -> None:
        groups = [
            {"day": "DAY1", "time_token": "12000000", "clips": clips(5)},
            {"day": "DAY2", "time_token": "13000000", "clips": clips(6)},
        ]
        built: list[str] = []

        def fake_build(group, **kwargs):
            del kwargs
            built.append(group["day"])
            return {"evidence_id": group["day"], "clips": group["clips"]}

        with (
            mock.patch.object(evidence, "read_json", return_value={}),
            mock.patch.object(evidence, "group_manifest_clips", return_value=groups),
            mock.patch.object(evidence, "select_evidence_groups", return_value=groups),
            mock.patch.object(
                evidence, "build_evidence_packet", side_effect=fake_build
            ),
        ):
            rows = list(
                evidence.iter_evidence_packets(
                    manifest_path="manifest.json",
                    cache_dir="cache",
                    output_root="output",
                    target_count=2,
                    users_per_case=6,
                    skip_incomplete_groups=True,
                    media_prepare_workers=6,
                )
            )

        self.assertEqual(built, ["DAY2"])
        self.assertEqual([row["evidence_id"] for row in rows], ["DAY2"])

    def test_prepare_cli_exposes_all_extreme_controls(self) -> None:
        args = setup.build_parser().parse_args(
            [
                "prepare",
                "--manifest",
                "manifest.json",
                "--output-root",
                "output",
                "--cache-dir",
                "cache",
                "--clip-batch-size",
                "600",
                "--video-sample-workers",
                "6",
                "--media-prepare-workers",
                "6",
            ]
        )
        self.assertEqual(
            (
                args.clip_batch_size,
                args.video_sample_workers,
                args.media_prepare_workers,
            ),
            (600, 6, 6),
        )


if __name__ == "__main__":
    unittest.main()
