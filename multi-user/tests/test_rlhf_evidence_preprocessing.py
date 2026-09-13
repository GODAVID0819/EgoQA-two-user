from __future__ import annotations

import io
import os
import shutil
import sys
import types
import unittest
import urllib.error
import uuid
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.io_utils import write_json  # noqa: E402
from egolife_two_user_qa import clip_gap_demo  # noqa: E402
from egolife_two_user_qa import evidence  # noqa: E402
from egolife_two_user_qa import io_utils  # noqa: E402
from egolife_two_user_qa import rlhf_evidence_preprocessing  # noqa: E402
from egolife_two_user_qa.clip_gap_demo import sample_source_segments_rgb  # noqa: E402
from egolife_two_user_qa.rlhf_evidence_preprocessing import (  # noqa: E402
    GENERATOR_MEDIA_MODE,
    SCHEMA_VERSION,
    _write_checksum_manifest,
    build_asker_conditioned_keep_masks,
    build_preprocessing_config,
    fingerprint,
    load_asker_view,
    prepare_packet,
    select_source_groups,
    validate_preprocessed_packet,
)


class RlhfEvidenceMaskTests(unittest.TestCase):
    def test_asker_is_full_and_only_providers_are_pruned(self) -> None:
        results = []
        for asker_index in range(6):
            videos = []
            for video_index in range(6):
                videos.append(
                    {
                        "marked_frame_indices": (
                            [] if video_index == asker_index else [0, 2]
                        )
                    }
                )
            results.append({"speaker_index": asker_index, "videos": videos})

        masks = build_asker_conditioned_keep_masks([4] * 6, results)

        self.assertEqual(masks.shape, (6, 6, 4))
        for asker_index in range(6):
            self.assertTrue(masks[asker_index, asker_index].all())
            for provider_index in range(6):
                if provider_index == asker_index:
                    continue
                self.assertEqual(
                    masks[asker_index, provider_index].tolist(),
                    [False, True, False, True],
                )

    def test_config_freezes_full_video_time_aware_policy_without_gap_split(self) -> None:
        config = build_preprocessing_config()

        self.assertEqual(
            config["clustering"]["scope"],
            "independently_per_user_across_entire_ten_minute_video",
        )
        self.assertEqual(config["clustering"]["global_cluster_count_requested"], 120)
        self.assertEqual(config["clustering"]["temporal_kmeans_time_weight"], 0.1)
        self.assertFalse(config["clustering"]["split_noncontiguous_clusters"])
        self.assertIsNone(config["clustering"]["max_cluster_member_gap_seconds"])
        self.assertEqual(config["generator"]["media_mode"], GENERATOR_MEDIA_MODE)

    def test_offline_sampling_persists_jpeg_and_drops_decoded_rgb_objects(self) -> None:
        test_root = ROOT / "tmp" / f"rlhf_jpeg_test_{uuid.uuid4().hex}"
        source = test_root / "source.mp4"
        output = test_root / "frames"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"synthetic-video")
        images = [Image.new("RGB", (8, 8), (index * 30, 0, 0)) for index in range(3)]
        segments = [
            {
                "segment_index": 0,
                "window_start_seconds": 0.0,
                "window_end_seconds": 6.0,
                "local_video": str(source),
            }
        ]
        try:
            with mock.patch.object(
                clip_gap_demo,
                "decode_source_segments_rgb_frames",
                return_value=images,
            ):
                frames = sample_source_segments_rgb(
                    segments,
                    output,
                    duration_seconds=6.0,
                    sample_interval_seconds=2.0,
                    persist_format="jpeg",
                    persist_jpeg_quality=95,
                    attach_rgb_images=False,
                )
            self.assertEqual(len(frames), 3)
            self.assertTrue(all(Path(frame["path"]).suffix == ".jpg" for frame in frames))
            self.assertTrue(all("_rgb_image" not in frame for frame in frames))
            with Image.open(frames[0]["path"]) as persisted:
                self.assertEqual(persisted.format, "JPEG")
        finally:
            for image in images:
                image.close()
            shutil.rmtree(test_root, ignore_errors=True)

    def test_media_preparation_can_skip_all_gaze_downloads(self) -> None:
        test_root = ROOT / "tmp" / f"rlhf_no_gaze_test_{uuid.uuid4().hex}"
        cache_root = test_root / "cache"
        clip = {
            "agent_dir": "A1",
            "agent_id": "id-1",
            "agent_name": "User 1",
            "day": "DAY1",
            "time_token": "12000000",
            "video_path": "DAY1/A1/source.mp4",
            "video_url": "https://example.invalid/source.mp4",
            "gaze_path": "DAY1/A1/gaze.csv",
            "gaze_url": "https://example.invalid/gaze.csv",
        }
        group = {
            "day": "DAY1",
            "time_token": "12000000",
            "clip_clock": "12:00:00.00",
        }
        downloads = []

        def fake_download(url: str, output_path: str | Path) -> Path:
            downloads.append(url)
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"media")
            return output

        try:
            with (
                mock.patch.object(evidence, "download_file", side_effect=fake_download),
                mock.patch.object(evidence, "ffprobe_duration", return_value=30.0),
            ):
                prepared = evidence._prepare_evidence_clip(
                    clip,
                    group=group,
                    cache_dir=cache_root,
                    packet_dir=test_root / "packet",
                    evidence_duration=30.0,
                    frames_per_clip=0,
                    aria_calibration_dir=None,
                    download_media=True,
                    download_gaze=False,
                    assemble_long_video=False,
                    defer_gaze_summary=True,
                    extract_preview_frames=False,
                )
            self.assertEqual(downloads, [clip["video_url"]])
            self.assertEqual(prepared["local_gazes"], [])
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_download_uses_token_file_and_coordinates_retry_after_429(self) -> None:
        test_root = ROOT / "tmp" / f"rlhf_download_test_{uuid.uuid4().hex}"
        token_path = test_root / "token"
        destination = test_root / "media.mp4"
        token_path.parent.mkdir(parents=True)
        token_path.write_text("test-token\n", encoding="utf-8")
        rate_limit_error = urllib.error.HTTPError(
            "https://example.invalid/media.mp4",
            429,
            "Too Many Requests",
            {"Retry-After": "17"},
            io.BytesIO(b"rate limited"),
        )

        class Response(io.BytesIO):
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                self.close()

        try:
            with (
                mock.patch.dict(
                    os.environ,
                    {
                        "HF_TOKEN": "",
                        "HF_TOKEN_PATH": str(token_path),
                        "EGOLIFE_HTTP_429_BASE_DELAY_SECONDS": "1",
                        "EGOLIFE_HTTP_MAX_RETRY_DELAY_SECONDS": "100",
                    },
                    clear=False,
                ),
                mock.patch.object(
                    io_utils.urllib.request,
                    "urlopen",
                    side_effect=[rate_limit_error, Response(b"video-bytes")],
                ) as urlopen,
                mock.patch.object(io_utils, "_wait_for_download_slot"),
                mock.patch.object(io_utils, "_extend_download_cooldown") as cooldown,
            ):
                result = io_utils.download_file(
                    "https://example.invalid/media.mp4",
                    destination,
                    retries=1,
                )
            self.assertEqual(result.read_bytes(), b"video-bytes")
            self.assertEqual(urlopen.call_count, 2)
            for call in urlopen.call_args_list:
                request = call.args[0]
                self.assertEqual(
                    request.get_header("Authorization"), "Bearer test-token"
                )
            self.assertEqual(cooldown.call_count, 1)
            self.assertGreaterEqual(float(cooldown.call_args.args[0]), 17.0)
        finally:
            shutil.rmtree(test_root, ignore_errors=True)

    def test_hpc_launcher_uses_measured_profile_and_project_root_hpc_path(self) -> None:
        launcher = (
            ROOT
            / "hpc"
            / "qa"
            / "production"
            / "run_six_user_rlhf_evidence_preprocessing.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn('HPC_ROOT="${PROJECT_ROOT}/hpc"', launcher)
        self.assertIn('CLIP_BATCH_SIZE="${CLIP_BATCH_SIZE:-256}"', launcher)
        self.assertIn('VIDEO_SAMPLE_WORKERS="${VIDEO_SAMPLE_WORKERS:-6}"', launcher)
        self.assertIn('MEDIA_PREPARE_WORKERS="${MEDIA_PREPARE_WORKERS:-6}"', launcher)
        self.assertIn('SOURCE_WINDOW_COUNT="${SOURCE_WINDOW_COUNT:-120}"', launcher)
        self.assertIn('LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-', launcher)
        self.assertIn('DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-8}"', launcher)
        self.assertIn('ENABLE_CUDA_KEEPER="${ENABLE_CUDA_KEEPER:-1}"', launcher)
        self.assertIn('CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-1}"', launcher)
        self.assertIn('stage "start_cuda_keeper"', launcher)
        self.assertIn(
            'EGOLIFE_HTTP_429_BASE_DELAY_SECONDS="${HTTP_429_BASE_DELAY_SECONDS}"',
            launcher,
        )
        self.assertIn("--clusters-per-30-seconds 6", launcher)
        self.assertIn("--temporal-kmeans-time-weight 0.1", launcher)
        self.assertIn("--min-provider-retention-percent 40", launcher)

    def test_resume_launcher_reselects_full_cohort_and_requires_keeper(self) -> None:
        launcher = (
            ROOT
            / "hpc"
            / "qa"
            / "production"
            / "resume_six_user_rlhf_evidence_preprocessing.sbatch"
        ).read_text(encoding="utf-8")

        self.assertIn('EXPECTED_MIN_COMPLETE_PACKETS="${EXPECTED_MIN_COMPLETE_PACKETS:-75}"', launcher)
        self.assertIn("export SOURCE_WINDOW_COUNT=120", launcher)
        self.assertIn("export START_INDEX=0", launcher)
        self.assertIn("export RANDOM_SEED=20260902", launcher)
        self.assertIn("export HF_TOKEN_PATH=/dev/null", launcher)
        self.assertIn("export ENABLE_CUDA_KEEPER=1", launcher)
        self.assertIn("export CUDA_KEEPER_START_USED_MIB=1", launcher)
        self.assertIn("production_launcher_missing_cuda_keeper", launcher)
        self.assertIn('exec bash "${BASE_LAUNCHER}"', launcher)

    def test_selection_appends_minimum_overlap_windows_after_primary_windows(self) -> None:
        clips = []
        # Forty minutes of common footage beginning at 00:05 creates three
        # wall-clock 10-minute bins plus two useful five-minute-shifted fillers.
        for user_index in range(6):
            agent_dir = f"A{user_index + 1}"
            for segment_index in range(80):
                clock_seconds = 300 + segment_index * 30
                hours, remainder = divmod(clock_seconds, 3600)
                minutes, seconds = divmod(remainder, 60)
                token = f"{hours:02d}{minutes:02d}{seconds:02d}00"
                clips.append(
                    {
                        "clip_id": f"{agent_dir}-{token}",
                        "day": "DAY1",
                        "agent_dir": agent_dir,
                        "agent_id": agent_dir,
                        "agent_name": agent_dir,
                        "time_token": token,
                        "clip_clock": f"{hours:02d}:{minutes:02d}:{seconds:02d}.00",
                        "clock_seconds": float(clock_seconds),
                        "video_path": f"DAY1/{agent_dir}/{token}.mp4",
                        "video_url": f"https://example.invalid/{agent_dir}/{token}.mp4",
                    }
                )

        groups = select_source_groups(
            {"clips": clips},
            duration_seconds=600.0,
            source_window_count=5,
            random_seed=7,
        )

        self.assertEqual(len(groups), 5)
        self.assertEqual(
            [group["selection"]["tier"] for group in groups[:3]],
            ["primary_non_overlapping_wall_clock"] * 3,
        )
        self.assertEqual(
            [group["selection"]["tier"] for group in groups[3:]],
            ["supplemental_minimum_overlap_sliding"] * 2,
        )
        self.assertEqual(
            {
                group["selection"]["new_source_segment_count_at_selection"]
                for group in groups[3:]
            },
            {10},
        )
        self.assertEqual(len({group["time_token"] for group in groups}), 5)
        self.assertTrue(
            all(
                len(group["clips"]) == 6
                and all(len(clip["segments"]) == 20 for clip in group["clips"])
                for group in groups
            )
        )


class RlhfEvidenceStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = ROOT / "tmp" / f"rlhf_evidence_test_{uuid.uuid4().hex}"
        self.root = self.test_root / "dataset_a"
        self.packet_id = "RLHF6U_DAY1_12000000_A1_A2_A3_A4_A5_A6"
        self.packet_dir = self.root / "packets" / self.packet_id
        self.packet_dir.mkdir(parents=True)
        config = build_preprocessing_config(duration_seconds=6.0, sample_fps=0.5)
        users = []
        for user_index in range(6):
            frames = []
            frame_dir = self.packet_dir / "frames" / f"user_{user_index}_A{user_index + 1}"
            frame_dir.mkdir(parents=True)
            for frame_index in range(3):
                path = frame_dir / f"frame_{frame_index:04d}_{2 * frame_index:.2f}s.jpg"
                path.write_bytes(b"synthetic-jpeg-bytes")
                frames.append(
                    {
                        "frame_index": frame_index,
                        "timestamp_seconds": float(2 * frame_index),
                        "source_segment_index": 0,
                        "path": path.relative_to(self.packet_dir).as_posix(),
                    }
                )
            users.append(
                {
                    "user_index": user_index,
                    "agent_dir": f"A{user_index + 1}",
                    "agent_id": f"id-{user_index + 1}",
                    "agent_name": f"User {user_index + 1}",
                    "frame_directory": frame_dir.relative_to(self.packet_dir).as_posix(),
                    "frame_count": 3,
                    "frames": frames,
                }
            )
        packet = {
            "schema_version": SCHEMA_VERSION,
            "packet_id": self.packet_id,
            "config_fingerprint": fingerprint(config),
            "source_fingerprint": "source-fingerprint",
            "day": "DAY1",
            "time_token": "12000000",
            "duration_seconds": 6.0,
            "generator_media_mode": GENERATOR_MEDIA_MODE,
            "preprocessing": config,
            "users": users,
        }
        write_json(self.packet_dir / "packet.json", packet)
        np.save(
            self.packet_dir / "clip_embeddings.f16.npy",
            np.ones((6, 3, 2), dtype=np.float16),
            allow_pickle=False,
        )
        masks = np.ones((6, 6, 3), dtype=np.bool_)
        for asker_index in range(6):
            for provider_index in range(6):
                if provider_index != asker_index:
                    masks[asker_index, provider_index, 0] = False
        np.savez_compressed(
            self.packet_dir / "keep_masks.npz",
            keep_masks=masks,
            frame_counts=np.asarray([3] * 6, dtype=np.int32),
        )
        cluster_users = []
        for user_index in range(6):
            cluster_users.append(
                {
                    "user_index": user_index,
                    "agent_dir": f"A{user_index + 1}",
                    "cluster_count": 2,
                    "cluster_count_requested": 2,
                    "global_cluster_count_requested": 2,
                    "clustering_scope": "full_duration_global",
                    "split_noncontiguous_clusters": False,
                    "max_cluster_member_gap_seconds": None,
                    "labels": [0, 0, 1],
                    "representatives": [],
                }
            )
        write_json(
            self.packet_dir / "clusters.json",
            {
                "schema_version": SCHEMA_VERSION,
                "clustering_scope": "per_user_across_entire_source_window",
                "users": cluster_users,
            },
        )
        write_json(
            self.packet_dir / "asker_views.json",
            {"schema_version": SCHEMA_VERSION, "views": []},
        )
        _write_checksum_manifest(self.packet_dir)
        (self.packet_dir / "COMPLETE").write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        shutil.rmtree(self.test_root, ignore_errors=True)

    def test_packet_validation_checks_relocatability_and_contract(self) -> None:
        result = validate_preprocessed_packet(
            self.packet_dir, verify_checksums=True
        )

        self.assertEqual(result["frame_counts"], [3] * 6)
        self.assertEqual(result["embedding_shape"], [6, 3, 2])
        self.assertTrue(result["checksums_verified"])

    def test_loader_resolves_full_asker_and_pruned_providers_after_move(self) -> None:
        moved_root = self.test_root / "dataset_b"
        shutil.move(str(self.root), str(moved_root))

        view = load_asker_view(moved_root, self.packet_id, "A3")

        self.assertEqual(view["generator_media_mode"], GENERATOR_MEDIA_MODE)
        self.assertEqual(view["asker_index"], 2)
        self.assertEqual(view["clips"][0]["agent_dir"], "A3")
        self.assertEqual(view["clips"][0]["retained_frame_count"], 3)
        self.assertTrue(
            all(clip["retained_frame_count"] == 2 for clip in view["clips"][1:])
        )
        self.assertTrue(
            all(
                Path(frame["path"]).is_relative_to(moved_root)
                for clip in view["clips"]
                for frame in clip["frames"]
            )
        )


class RlhfEvidenceBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.test_root = ROOT / "tmp" / f"rlhf_builder_test_{uuid.uuid4().hex}"
        self.dataset_root = self.test_root / "dataset"
        self.cache_root = self.test_root / "cache"
        self.cache_root.mkdir(parents=True)
        self.config = build_preprocessing_config(duration_seconds=6.0, sample_fps=0.5)
        self.group = {
            "day": "DAY1",
            "time_token": "12000000",
            "clip_clock": "12:00:00.00",
            "duration_seconds": 6.0,
            "clips": [],
        }
        self.sampled_users = []
        for user_index in range(6):
            agent_dir = f"A{user_index + 1}"
            clip = {
                "agent_dir": agent_dir,
                "agent_id": f"id-{user_index + 1}",
                "agent_name": f"User {user_index + 1}",
                "clip_id": f"clip-{user_index + 1}",
                "time_token": "12000000",
                "video_path": f"DAY1/{agent_dir}/source.mp4",
                "video_url": f"https://example.invalid/{agent_dir}.mp4",
            }
            self.group["clips"].append(clip)
            frames = []
            for frame_index in range(3):
                path = self.cache_root / agent_dir / f"frame_{frame_index}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"cached-jpeg")
                frames.append(
                    {
                        "timestamp_seconds": float(frame_index * 2),
                        "source_segment_index": 0,
                        "path": str(path),
                    }
                )
            self.sampled_users.append(
                {"user": clip["agent_name"], "clip": clip, "frames": frames}
            )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_root, ignore_errors=True)

    @staticmethod
    def clusters() -> list[dict[str, object]]:
        rows = []
        for _ in range(6):
            rows.append(
                {
                    "cluster_count_requested": 2,
                    "cluster_count": 2,
                    "global_cluster_count_requested": 2,
                    "clustering_scope": "full_duration_global",
                    "split_noncontiguous_clusters": False,
                    "max_member_gap_seconds": None,
                    "labels": [0, 0, 1],
                    "representatives": [
                        {
                            "cluster_index": 0,
                            "frame_index": 0,
                            "timestamp_seconds": 0.0,
                            "member_count": 2,
                            "temporal_center_seconds": 1.0,
                        },
                        {
                            "cluster_index": 1,
                            "frame_index": 2,
                            "timestamp_seconds": 4.0,
                            "member_count": 1,
                            "temporal_center_seconds": 4.0,
                        },
                    ],
                    "representative_embeddings": [[1.0, 0.0], [0.0, 1.0]],
                }
            )
        return rows

    @staticmethod
    def pruning_result(asker_index: int) -> dict[str, object]:
        videos = []
        for video_index in range(6):
            videos.append(
                {
                    "marked_frame_indices": (
                        [] if video_index == asker_index else [0]
                    ),
                    "marked_cluster_indices": (
                        [] if video_index == asker_index else [0]
                    ),
                    "restored_cluster_indices": [],
                }
            )
        return {
            "speaker_index": asker_index,
            "videos": videos,
            "applied_event_count": 5,
            "eligible_pairwise_comparison_count": 20,
            "computed_cosine_pair_count": 20,
        }

    def test_prepare_packet_reuses_clusters_for_all_askers_and_writes_loader_contract(
        self,
    ) -> None:
        class FakeEncoder:
            model_id = "fake-clip"

            def encode(self, paths: list[str]) -> list[list[float]]:
                return [[1.0, float(index % 2)] for index, _ in enumerate(paths)]

        shared_clusters = self.clusters()
        pruning_calls = []

        def fake_pruning(*args: object, **kwargs: object) -> dict[str, object]:
            pruning_calls.append(kwargs)
            self.assertEqual(kwargs["pruning_protection_mode"], "min_percent")
            self.assertEqual(kwargs["min_pruned_video_percent"], 40.0)
            self.assertFalse(kwargs["split_noncontiguous_clusters"])
            self.assertIsNone(kwargs["max_cluster_member_gap_seconds"])
            self.assertIs(kwargs["precomputed_clusters_by_video"], shared_clusters)
            return self.pruning_result(int(kwargs["speaker_index"]))

        with (
            mock.patch.object(
                rlhf_evidence_preprocessing,
                "build_evidence_packet",
                return_value={"clips": self.group["clips"]},
            ),
            mock.patch.object(
                rlhf_evidence_preprocessing,
                "group_clip_frames",
                return_value=self.sampled_users,
            ),
            mock.patch.object(
                rlhf_evidence_preprocessing,
                "cluster_six_user_frame_representatives",
                return_value=shared_clusters,
            ) as cluster_mock,
            mock.patch.object(
                rlhf_evidence_preprocessing,
                "clustered_speaker_provider_all_pairs_pruning",
                side_effect=fake_pruning,
            ),
        ):
            result = prepare_packet(
                self.group,
                dataset_root=self.dataset_root,
                cache_dir=self.cache_root,
                config=self.config,
                config_fingerprint=fingerprint(self.config),
                encoder=FakeEncoder(),
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(cluster_mock.call_count, 1)
        self.assertFalse(cluster_mock.call_args.kwargs["split_noncontiguous_clusters"])
        self.assertIsNone(cluster_mock.call_args.kwargs["max_cluster_member_gap_seconds"])
        self.assertEqual(len(pruning_calls), 6)
        packet_id = result["packet_id"]
        validation = validate_preprocessed_packet(
            self.dataset_root / "packets" / packet_id,
            verify_checksums=True,
        )
        self.assertEqual(validation["frame_counts"], [3] * 6)
        view = load_asker_view(self.dataset_root, packet_id, "A2")
        self.assertEqual(view["clips"][0]["frame_mode"], "full_sampled")
        self.assertEqual(view["clips"][0]["retained_frame_count"], 3)
        self.assertTrue(
            all(clip["retained_frame_count"] == 2 for clip in view["clips"][1:])
        )
        resumed = prepare_packet(
            self.group,
            dataset_root=self.dataset_root,
            cache_dir=self.cache_root,
            config=self.config,
            config_fingerprint=fingerprint(self.config),
            encoder=FakeEncoder(),
        )
        self.assertEqual(resumed["status"], "skipped_complete")


if __name__ == "__main__":
    unittest.main()
