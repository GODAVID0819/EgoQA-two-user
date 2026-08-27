from __future__ import annotations

from pathlib import Path
import inspect
from unittest import mock

from egolife_two_user_qa.evidence import group_manifest_clips
from egolife_two_user_qa import group_relative_clip_sampling as sampling


AGENTS = [
    ("A1_JAKE", "A1", "Jake"),
    ("A2_ALICE", "A2", "Alice"),
    ("A3_TASHA", "A3", "Tasha"),
    ("A4_LUCIA", "A4", "Lucia"),
    ("A5_KATRINA", "A5", "Katrina"),
    ("A6_SHURE", "A6", "Shure"),
]


def _time_fields(clock_seconds: int) -> tuple[str, str]:
    hours, remainder = divmod(clock_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:02d}{minutes:02d}{seconds:02d}00",
        f"{hours:02d}:{minutes:02d}:{seconds:02d}.00",
    )


def _clip(
    agent_dir: str,
    agent_id: str,
    name: str,
    clock_seconds: int,
) -> dict[str, object]:
    time_token, clip_clock = _time_fields(clock_seconds)
    stem = f"DAY1_{agent_id}_{name.upper()}_{time_token}"
    return {
        "clip_id": f"DAY1_{agent_dir}_{time_token}",
        "day": "DAY1",
        "agent_dir": agent_dir,
        "agent_id": agent_id,
        "agent_name": name,
        "time_token": time_token,
        "clip_clock": clip_clock,
        "clock_seconds": float(clock_seconds),
        "video_path": f"{agent_dir}/DAY1/{stem}.mp4",
        "video_url": f"https://example.test/{agent_dir}/{stem}.mp4",
    }


def _three_minute_manifest() -> dict[str, object]:
    start = 12 * 3600
    return {
        "clips": [
            _clip(agent_dir, agent_id, name, start + 30 * index)
            for agent_dir, agent_id, name in AGENTS
            for index in range(6)
        ]
    }


def _three_minute_group() -> dict[str, object]:
    return group_manifest_clips(
        _three_minute_manifest(),
        evidence_duration_seconds=180.0,
    )[0]


def test_three_minute_group_has_six_users_and_six_segments_each() -> None:
    group = _three_minute_group()

    assert group["duration_seconds"] == 180.0
    assert group["segment_count"] == 6
    assert len(group["clips"]) == 6
    assert all(len(row["segments"]) == 6 for row in group["clips"])


def test_group_local_video_concatenates_once_and_reuses_cache(tmp_path: Path) -> None:
    group = _three_minute_group()
    sources = [tmp_path / f"segment-{index}.mp4" for index in range(6)]
    for source in sources:
        source.write_bytes(b"segment")
    target = tmp_path / "assembled" / "window.mp4"

    def fake_concat(paths, output, *, duration_seconds):
        assert paths == sources
        assert duration_seconds == 180.0
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_bytes(b"assembled")

    with (
        mock.patch.object(sampling, "_resolve_local_video", side_effect=sources * 2),
        mock.patch.object(sampling, "window_cache_path", return_value=target),
        mock.patch.object(
            sampling,
            "concatenate_video_segments",
            side_effect=fake_concat,
        ) as concat,
    ):
        first = sampling._resolve_group_local_video(
            group["clips"][0],
            group,
            cache_dir=tmp_path,
            download_media=False,
        )
        second = sampling._resolve_group_local_video(
            group["clips"][0],
            group,
            cache_dir=tmp_path,
            download_media=False,
        )

    assert first == target
    assert second == target
    assert concat.call_count == 1


def _frames_for_three_minutes() -> list[list[dict[str, object]]]:
    return [
        [
            {"timestamp_seconds": float(second), "path": f"v{video}-t{second}.jpg"}
            for second in range(180)
        ]
        for video in range(6)
    ]


def test_blockwise_pruning_calls_existing_kernel_once_per_thirty_seconds() -> None:
    frames = _frames_for_three_minutes()
    embeddings = [[[1.0, 0.0] for _ in row] for row in frames]

    def fake_kernel(
        block_frames,
        _block_embeddings,
        *,
        speaker_index,
        start_seconds,
        duration_seconds,
        **_kwargs,
    ):
        assert duration_seconds == 30.0
        assert all(
            start_seconds <= frame["timestamp_seconds"] < start_seconds + 30.0
            for rows in block_frames
            for frame in rows
        )
        block_index = int(start_seconds // 30)
        return {
            "method": "speaker_provider_all_pairs_provider_only",
            "speaker_index": speaker_index,
            "events": [] if block_index == 2 else [{"event_index": 0}],
            "videos": [
                {
                    "video_index": video_index,
                    "keep_intervals": [(start_seconds, start_seconds + 30.0)],
                    "remove_intervals": [],
                    "kept_duration_seconds": 30.0,
                    "removed_duration_seconds": 0.0,
                    "marked_cluster_indices": [],
                    "trigger_event_indices": [],
                    "passed": True,
                }
                for video_index in range(6)
            ],
            "passed": block_index != 2,
        }

    with mock.patch.object(
        sampling,
        "clustered_speaker_provider_all_pairs_pruning",
        side_effect=fake_kernel,
    ) as kernel:
        result = sampling.blockwise_speaker_provider_all_pairs_pruning(
            frames,
            embeddings,
            speaker_index=0,
            start_seconds=0.0,
            duration_seconds=180.0,
            block_duration_seconds=30.0,
            sample_interval_seconds=1.0,
        )

    assert kernel.call_count == 6
    assert len(result["blocks"]) == 6
    assert result["videos"][0]["remove_intervals"] == []
    assert result["passed"] is True


def _run_blockwise_with_provider_kept_seconds(
    provider_kept_seconds_per_block: float,
) -> dict[str, object]:
    frames = _frames_for_three_minutes()
    embeddings = [[[1.0, 0.0] for _ in row] for row in frames]

    def fake_kernel(
        _block_frames,
        _block_embeddings,
        *,
        speaker_index,
        start_seconds,
        duration_seconds,
        **_kwargs,
    ):
        block_end = start_seconds + duration_seconds
        rows = []
        for video_index in range(6):
            kept = (
                duration_seconds
                if video_index != 1
                else provider_kept_seconds_per_block
            )
            keep_end = start_seconds + kept
            rows.append(
                {
                    "video_index": video_index,
                    "keep_intervals": [(start_seconds, keep_end)],
                    "remove_intervals": [] if keep_end == block_end else [(keep_end, block_end)],
                    "kept_duration_seconds": kept,
                    "removed_duration_seconds": duration_seconds - kept,
                    "marked_cluster_indices": [0] if video_index == 1 else [],
                    "trigger_event_indices": [0] if video_index == 1 else [],
                    "passed": kept >= 8.0,
                }
            )
        return {
            "method": "speaker_provider_all_pairs_provider_only",
            "speaker_index": speaker_index,
            "events": [{"event_index": 0, "provider_index": 1}],
            "videos": rows,
            "passed": all(row["passed"] for row in rows),
        }

    with mock.patch.object(
        sampling,
        "clustered_speaker_provider_all_pairs_pruning",
        side_effect=fake_kernel,
    ):
        return sampling.blockwise_speaker_provider_all_pairs_pruning(
            frames,
            embeddings,
            speaker_index=0,
            start_seconds=0.0,
            duration_seconds=180.0,
            block_duration_seconds=30.0,
            sample_interval_seconds=1.0,
            min_pruned_video_seconds=8.0,
        )


def test_blockwise_pass_uses_aggregate_duration_not_each_block_threshold() -> None:
    result = _run_blockwise_with_provider_kept_seconds(4.0)

    assert result["videos"][1]["kept_duration_seconds"] == 24.0
    assert all(
        block["videos"][1]["passed"] is False
        for block in result["blocks"]
    )
    assert result["videos"][1]["passed"] is True
    assert result["passed"] is True


def test_blockwise_pass_still_rejects_short_aggregate_provider() -> None:
    result = _run_blockwise_with_provider_kept_seconds(1.0)

    assert result["videos"][1]["kept_duration_seconds"] == 6.0
    assert result["videos"][1]["passed"] is False
    assert result["passed"] is False


def test_three_minute_analysis_uses_zip_full_window_k72_for_every_speaker(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "user": name,
            "clip": {
                "agent_name": name,
                "agent_dir": agent_dir,
                "local_video": str(tmp_path / f"{agent_dir}.mp4"),
            },
            "frames": [
                {"timestamp_seconds": float(second), "path": f"{agent_dir}-{second}.jpg"}
                for second in range(180)
            ],
        }
        for agent_dir, _agent_id, name in AGENTS
    ]

    class Encoder:
        model_id = "fake/clip"

        def encode(self, paths):
            return [[1.0, 0.0] for _ in paths]

    calls = []

    def fake_zip_pruning(*_args, speaker_index, **kwargs):
        calls.append(kwargs)
        return {
            "method": "zip_temporal_kmeans_center_gate_pair_pruning_v1",
            "speaker_index": speaker_index,
            "events": [],
            "videos": [],
            "pair_results": [],
            "passed": False,
        }

    group = {
        "day": "DAY1",
        "time_token": "12000000",
        "clips": [row["clip"] for row in rows],
    }
    with (
        mock.patch.object(sampling, "group_clip_frames", return_value=rows),
        mock.patch.object(
            sampling,
            "clustered_six_user_zip_temporal_pruning",
            side_effect=fake_zip_pruning,
        ) as zip_kernel,
        mock.patch.object(
            sampling,
            "blockwise_speaker_provider_all_pairs_pruning",
        ) as blockwise,
    ):
        sampling.analyze_group_relative_similarity(
            group,
            output_dir=tmp_path / "out",
            cache_dir=tmp_path / "cache",
            encoder=Encoder(),
            selected_count=6,
            duration_seconds=180.0,
            pruning_block_seconds=30.0,
        )

    assert zip_kernel.call_count == 6
    blockwise.assert_not_called()
    assert len(calls) == 6
    assert all(call["seconds_per_cluster"] == 2.5 for call in calls)
    assert all(call["time_weight"] == 0.1 for call in calls)
    assert all(call["cross_gap_mode"] == "center" for call in calls)
    assert all(call["max_cross_gap_seconds"] == 10.0 for call in calls)


def test_single_candidate_group_keeps_partial_speaker_set(tmp_path: Path) -> None:
    groups = [
        {"day": "DAY1", "time_token": "12000000", "clips": [{}] * 6},
        {"day": "DAY1", "time_token": "12030000", "clips": [{}] * 6},
    ]
    first_candidates = [
        {
            "day": "DAY1",
            "time_token": "12000000",
            "selection": {"speaker_index": speaker_index},
        }
        for speaker_index in (0, 2, 5)
    ]

    class Encoder:
        model_id = "fake/clip"

    def fake_packet(candidate):
        group_id = f"{candidate['day']}::{candidate['time_token']}"
        return {
            "evidence_id": f"speaker-{candidate['selection']['speaker_index']}",
            "generation_group_id": group_id,
            "group_relative_clip_similarity": {},
        }

    with (
        mock.patch.object(sampling, "read_json", return_value={"clips": []}),
        mock.patch.object(sampling, "group_manifest_clips", return_value=groups),
        mock.patch.object(
            sampling,
            "analyze_group_relative_similarity",
            side_effect=[{"speaker_candidates": first_candidates}],
        ) as analyze,
        mock.patch.object(sampling, "write_review_bundle", return_value=tmp_path / "review"),
        mock.patch.object(sampling, "build_candidate_packet", side_effect=fake_packet),
    ):
        candidates = sampling.mine_group_relative_clip_candidates(
            manifest_path=tmp_path / "manifest.json",
            output_path=tmp_path / "candidates.jsonl",
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache",
            selected_count=6,
            target_count=6,
            encoder=Encoder(),
            single_candidate_group=True,
        )

    assert analyze.call_count == 1
    assert [row["evidence_id"] for row in candidates] == [
        "speaker-0",
        "speaker-2",
        "speaker-5",
    ]
    assert {row["generation_group_id"] for row in candidates} == {"DAY1::12000000"}


def test_miner_requires_three_distinct_generation_groups_even_after_candidate_target(
    tmp_path: Path,
) -> None:
    assert "target_generation_groups" in inspect.signature(
        sampling.mine_group_relative_clip_candidates
    ).parameters
    groups = [
        {"day": "DAY1", "time_token": token, "clips": [{}] * 6}
        for token in ("12000000", "12030000", "12060000", "12090000")
    ]

    def analyzed(group, **_kwargs):
        return {
            "speaker_candidates": [
                {
                    "day": group["day"],
                    "time_token": group["time_token"],
                    "selection": {"speaker_index": speaker_index},
                }
                for speaker_index in range(6)
            ]
        }

    def packet(candidate):
        return {
            "evidence_id": f"{candidate['time_token']}-speaker-{candidate['selection']['speaker_index']}",
            "generation_group_id": f"{candidate['day']}::{candidate['time_token']}",
            "group_relative_clip_similarity": {},
        }

    with (
        mock.patch.object(sampling, "read_json", return_value={"clips": []}),
        mock.patch.object(sampling, "group_manifest_clips", return_value=groups),
        mock.patch.object(
            sampling,
            "analyze_group_relative_similarity",
            side_effect=analyzed,
        ) as analyze,
        mock.patch.object(sampling, "write_review_bundle", return_value=tmp_path / "review"),
        mock.patch.object(sampling, "build_candidate_packet", side_effect=packet),
        mock.patch.object(sampling, "write_json") as write_json,
    ):
        candidates = sampling.mine_group_relative_clip_candidates(
            manifest_path=tmp_path / "manifest.json",
            output_path=tmp_path / "candidates.jsonl",
            output_dir=tmp_path / "output",
            cache_dir=tmp_path / "cache",
            selected_count=6,
            target_count=6,
            target_generation_groups=3,
            encoder=mock.Mock(model_id="fake/clip"),
        )

    assert analyze.call_count == 3
    assert len(candidates) == 18
    actual_group_ids = {row["generation_group_id"] for row in candidates}
    assert len(actual_group_ids) == 3
    assert all(
        sum(row["generation_group_id"] == group_id for row in candidates) == 6
        for group_id in actual_group_ids
    )
    summary = write_json.call_args_list[-1].args[1]
    assert summary["generation_group_count"] == 3
    assert set(summary["generation_group_ids"]) == actual_group_ids
