from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from egolife_two_user_qa.evidence import (
    DEFAULT_EVIDENCE_DURATION_SECONDS,
    LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
    build_evidence_packet,
    concatenate_video_segments,
    group_manifest_clips,
)
from egolife_two_user_qa.prompts import temporal_pruning_brief, video_packet_brief
from egolife_two_user_qa.video_qa_loop import video_evidence_for_packet


AGENTS = {
    "A1_JAKE": ("A1", "Jake"),
    "A2_ALICE": ("A2", "Alice"),
}


def _time_fields(clock_seconds: int) -> tuple[str, str]:
    hours, remainder = divmod(clock_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:02d}{minutes:02d}{seconds:02d}00",
        f"{hours:02d}:{minutes:02d}:{seconds:02d}.00",
    )


def _clip(agent_dir: str, clock_seconds: int) -> dict[str, Any]:
    agent_id, agent_name = AGENTS[agent_dir]
    time_token, clip_clock = _time_fields(clock_seconds)
    stem = f"DAY1_{agent_id}_{agent_name.upper()}_{time_token}"
    return {
        "clip_id": f"DAY1_{agent_dir}_{time_token}",
        "day": "DAY1",
        "agent_dir": agent_dir,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "time_token": time_token,
        "clip_clock": clip_clock,
        "clock_seconds": float(clock_seconds),
        "video_path": f"{agent_dir}/DAY1/{stem}.mp4",
        "video_url": f"https://example.test/{agent_dir}/{stem}.mp4",
        "gaze_path": f"EyeGaze/{agent_dir}/DAY1/{stem}.csv",
        "gaze_url": f"https://example.test/EyeGaze/{agent_dir}/{stem}.csv",
        "overlay_url": None,
    }


def _ten_minute_manifest(*, omit: tuple[str, int] | None = None) -> dict[str, Any]:
    start = 12 * 3600
    clips = []
    for agent_dir in AGENTS:
        for index in range(20):
            clock_seconds = start + 30 * index
            if omit == (agent_dir, clock_seconds):
                continue
            clips.append(_clip(agent_dir, clock_seconds))
    return {"clips": clips}


def test_ten_minute_grouping_builds_complete_synchronized_windows() -> None:
    groups = group_manifest_clips(
        _ten_minute_manifest(),
        evidence_duration_seconds=LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
    )

    assert len(groups) == 1
    group = groups[0]
    assert group["time_token"] == "12000000"
    assert group["duration_seconds"] == 600.0
    assert group["segment_count"] == 20
    assert [clip["agent_dir"] for clip in group["clips"]] == ["A1_JAKE", "A2_ALICE"]
    assert all(len(clip["segments"]) == 20 for clip in group["clips"])
    assert group["clips"][0]["segments"][-1]["time_token"] == "12093000"


def test_ten_minute_grouping_rejects_an_agent_with_a_missing_segment() -> None:
    missing_time = 12 * 3600 + 9 * 30

    groups = group_manifest_clips(
        _ten_minute_manifest(omit=("A2_ALICE", missing_time)),
        evidence_duration_seconds=LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
    )

    assert groups == []


def test_thirty_second_override_preserves_exact_timestamp_grouping() -> None:
    clips = [_clip(agent_dir, 12 * 3600 + 12) for agent_dir in AGENTS]

    groups = group_manifest_clips(
        {"clips": clips},
        evidence_duration_seconds=30.0,
    )

    assert len(groups) == 1
    assert groups[0]["time_token"] == clips[0]["time_token"]
    assert groups[0]["segment_count"] == 1


def test_default_evidence_duration_preserves_original_thirty_second_behavior() -> None:
    clips = [_clip(agent_dir, 12 * 3600 + 12) for agent_dir in AGENTS]

    groups = group_manifest_clips(
        {"clips": clips},
        evidence_duration_seconds=DEFAULT_EVIDENCE_DURATION_SECONDS,
    )

    assert DEFAULT_EVIDENCE_DURATION_SECONDS == 30.0
    assert len(groups) == 1
    assert groups[0]["duration_seconds"] == 30.0
    assert groups[0]["segment_count"] == 1


def test_dry_packet_keeps_all_source_provenance_without_claiming_remote_window_url(
    tmp_path,
) -> None:
    group = group_manifest_clips(
        _ten_minute_manifest(),
        evidence_duration_seconds=LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
    )[0]

    packet = build_evidence_packet(
        group,
        cache_dir=tmp_path / "cache",
        output_root=tmp_path / "output",
        download_media=False,
    )

    assert packet["evidence_id"] == "EGOLIFE2U_DAY1_12000000_600S_A1_A2"
    assert packet["duration_seconds"] == 600.0
    assert packet["segment_count"] == 20
    assert len(packet["source_urls"]["videos"]) == 40
    assert all(clip["video_url"] is None for clip in packet["clips"])
    assert all(clip["local_video"] is None for clip in packet["clips"])
    assert all(len(clip["source_segments"]) == 20 for clip in packet["clips"])

    prompt_packet = json.loads(video_packet_brief(packet))
    assert prompt_packet["clips"][0]["duration_seconds"] == 600.0
    assert prompt_packet["clips"][0]["segment_count"] == 20
    audit_rows = video_evidence_for_packet(packet)
    assert len(audit_rows[0]["source_video_urls"]) == 20
    assert len(audit_rows[0]["source_segments"]) == 20


def test_temporal_pruning_brief_exposes_pruned_to_original_time_map() -> None:
    brief = temporal_pruning_brief(
        {
            "keep_intervals": [[2.5, 4.5], [6.5, 7.5]],
            "kept_duration_seconds": 3.0,
            "removed_duration_seconds": 7.0,
        }
    )

    assert brief is not None
    assert brief["pruned_to_original_time_map"] == [
        {
            "pruned_start_seconds": 0.0,
            "pruned_end_seconds": 2.0,
            "original_start_seconds": 2.5,
            "original_end_seconds": 4.5,
        },
        {
            "pruned_start_seconds": 2.0,
            "pruned_end_seconds": 3.0,
            "original_start_seconds": 6.5,
            "original_end_seconds": 7.5,
        },
    ]
    assert "equal pruned playback positions do not prove concurrency" in brief[
        "temporal_alignment_contract"
    ]


def test_evidence_duration_must_be_a_multiple_of_source_clip_duration() -> None:
    with pytest.raises(ValueError, match="multiple of the 30-second"):
        group_manifest_clips(
            _ten_minute_manifest(),
            evidence_duration_seconds=595.0,
        )


def test_concat_uses_twenty_fixed_thirty_second_segments(
    tmp_path,
    monkeypatch,
) -> None:
    sources = []
    for index in range(20):
        source = tmp_path / f"segment_{index:02d}.mp4"
        source.write_bytes(b"source")
        sources.append(source)
    observed: dict[str, Any] = {}

    def fake_run(command, *, check):
        assert check is True
        concat_file = Path(command[command.index("-i") + 1])
        observed["concat_text"] = concat_file.read_text(encoding="utf-8")
        observed["command"] = command
        Path(command[-1]).write_bytes(b"assembled")

    monkeypatch.setattr("egolife_two_user_qa.evidence.shutil.which", lambda _name: "ffmpeg")
    monkeypatch.setattr("egolife_two_user_qa.evidence.subprocess.run", fake_run)
    monkeypatch.setattr("egolife_two_user_qa.evidence.ffprobe_duration", lambda _path: 600.0)
    output = tmp_path / "window.mp4"

    concatenate_video_segments(sources, output, duration_seconds=600.0)

    assert output.read_bytes() == b"assembled"
    assert observed["concat_text"].count("file '") == 20
    assert observed["concat_text"].count("duration 30.000") == 20
    assert observed["command"][observed["command"].index("-t") + 1] == "600.000"
