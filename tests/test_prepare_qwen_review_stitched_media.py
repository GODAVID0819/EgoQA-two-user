from __future__ import annotations

import json
from pathlib import Path

from egolife_two_user_qa.tools.prepare_qwen_review_stitched_media import (
    MediaTask,
    build_media_tasks,
    prepare_media,
    segment_timestamps,
    segment_url,
)


def test_segment_timestamps_cover_ten_minutes_in_thirty_second_steps() -> None:
    values = segment_timestamps("12000000")
    assert len(values) == 20
    assert values[:3] == ("12000000", "12003000", "12010000")
    assert values[-1] == "12093000"


def test_segment_timestamps_cross_hour_boundary() -> None:
    values = segment_timestamps("15523000")
    assert values[0] == "15523000"
    assert values[-1] == "16020000"


def test_segment_url_uses_real_egolife_layout() -> None:
    assert segment_url("DAY4", "Alice", "12003000") == (
        "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main/"
        "A2_ALICE/DAY4/DAY4_A2_ALICE_12003000.mp4"
    )


def test_build_media_tasks_deduplicates_groups_and_expands_six_users(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.jsonl"
    rows = [
        {"qa_id": "Q1", "generation_group_id": "DAY4::12000000"},
        {"qa_id": "Q2", "generation_group_id": "DAY4::12000000"},
        {"qa_id": "Q3", "generation_group_id": "DAY1::11200000"},
    ]
    selection.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    tasks = build_media_tasks(selection, tmp_path / "stitched")
    assert len(tasks) == 12
    assert tasks[0].group_id == "DAY4::12000000"
    assert tasks[0].user == "Jake"
    assert tasks[0].output_path == tmp_path / "stitched" / "DAY4_12000000" / "Jake.mp4"
    assert len(tasks[0].urls) == 20
    assert tasks[-1].group_id == "DAY1::11200000"
    assert tasks[-1].user == "Shure"


def test_prepare_media_manifest_uses_actual_task_count(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from egolife_two_user_qa.tools import prepare_qwen_review_stitched_media as module

    output = tmp_path / "stitched" / "DAY1_11200000" / "Jake.mp4"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"video")
    task = MediaTask(
        group_id="DAY1::11200000",
        group_dir="DAY1_11200000",
        day="DAY1",
        user="Jake",
        agent_dir="A1_JAKE",
        urls=("https://example.invalid/segment.mp4",),
        output_path=output,
    )
    monkeypatch.setattr(module, "_usable_output", lambda ffprobe, path: (True, 600.0))
    manifest = tmp_path / "media_manifest.json"
    prepare_media(
        [task],
        work_root=tmp_path / "work",
        ffmpeg="ffmpeg",
        ffprobe="ffprobe",
        workers=1,
        timeout=1,
        manifest_path=manifest,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["expected_task_count"] == 1
    assert payload["completed_task_count"] == 1
