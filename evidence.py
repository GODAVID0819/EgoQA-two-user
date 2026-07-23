"""Prepare two-user evidence packets from the EgoLife manifest."""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import statistics
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .gaze_projection import (
    find_clip_calibration,
    load_aria_projection_calibration,
    project_gaze_csv_with_projectaria_tools,
    project_gaze_row,
    summarize_projected_gaze,
)
from .io_utils import download_file, read_json, stable_id, write_jsonl
from .manifest import seconds_from_time_token


SOURCE_CLIP_DURATION_SECONDS = 30.0
DEFAULT_EVIDENCE_DURATION_SECONDS = SOURCE_CLIP_DURATION_SECONDS
LONG_CONTEXT_EVIDENCE_DURATION_SECONDS = 10 * 60.0


def _clock_fields(clock_centiseconds: int) -> tuple[str, str]:
    hours, remainder = divmod(clock_centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    time_token = f"{hours:02d}{minutes:02d}{seconds:02d}{centiseconds:02d}"
    clip_clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"
    return time_token, clip_clock


def _duration_centiseconds(evidence_duration_seconds: float) -> int:
    duration_centiseconds = round(float(evidence_duration_seconds) * 100)
    if duration_centiseconds <= 0:
        raise ValueError("evidence_duration_seconds must be positive")
    if abs(duration_centiseconds / 100.0 - float(evidence_duration_seconds)) > 1e-9:
        raise ValueError("evidence_duration_seconds must have at most centisecond precision")
    source_centiseconds = round(SOURCE_CLIP_DURATION_SECONDS * 100)
    if duration_centiseconds % source_centiseconds:
        raise ValueError(
            "evidence_duration_seconds must be a multiple of the 30-second source clip duration"
        )
    return duration_centiseconds


def _group_single_source_clips(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Preserve the original exact-timestamp grouping for 30-second evidence."""

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for clip in manifest.get("clips", []):
        grouped[(clip["day"], clip["time_token"])].append(clip)

    groups = []
    for (day, time_token), clips in sorted(grouped.items()):
        unique_agents = sorted({clip["agent_dir"] for clip in clips})
        if len(unique_agents) < 2:
            continue
        groups.append(
            {
                "day": day,
                "time_token": time_token,
                "clip_clock": clips[0].get("clip_clock"),
                "clock_seconds": clips[0].get("clock_seconds"),
                "duration_seconds": SOURCE_CLIP_DURATION_SECONDS,
                "segment_count": 1,
                "agents": unique_agents,
                "clips": sorted(clips, key=lambda c: c["agent_dir"]),
            }
        )
    return groups


def group_manifest_clips(
    manifest: dict[str, Any],
    *,
    evidence_duration_seconds: float = SOURCE_CLIP_DURATION_SECONDS,
) -> list[dict[str, Any]]:
    """Group complete, synchronized source sequences into evidence windows.

    EgoLife stores the recordings as consecutive 30-second files. Longer
    evidence windows are aligned to clock boundaries and admitted only when an
    agent has every source segment, which prevents silent gaps or cross-user
    temporal drift. The 30-second override retains the historical exact-token
    behavior for smoke tests and comparisons.
    """

    duration_centiseconds = _duration_centiseconds(evidence_duration_seconds)
    source_centiseconds = round(SOURCE_CLIP_DURATION_SECONDS * 100)
    if duration_centiseconds == source_centiseconds:
        return _group_single_source_clips(manifest)

    segments_by_window: dict[
        tuple[str, int], dict[str, dict[int, dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(dict))
    for clip in manifest.get("clips", []):
        try:
            clock_centiseconds = round(
                float(clip.get("clock_seconds", seconds_from_time_token(clip["time_token"]))) * 100
            )
        except (KeyError, TypeError, ValueError):
            continue
        window_start = (clock_centiseconds // duration_centiseconds) * duration_centiseconds
        offset = clock_centiseconds - window_start
        if offset % source_centiseconds:
            continue
        segments_by_window[(str(clip["day"]), window_start)][str(clip["agent_dir"])][
            clock_centiseconds
        ] = clip

    groups = []
    segment_count = duration_centiseconds // source_centiseconds
    for (day, window_start), agent_segments in sorted(segments_by_window.items()):
        expected_starts = [
            window_start + index * source_centiseconds for index in range(segment_count)
        ]
        window_clips = []
        for agent_dir, segments_by_start in sorted(agent_segments.items()):
            if any(start not in segments_by_start for start in expected_starts):
                continue
            segments = [segments_by_start[start] for start in expected_starts]
            window_clip = dict(segments[0])
            window_clip["segments"] = segments
            window_clip["duration_seconds"] = duration_centiseconds / 100.0
            window_clip["segment_count"] = segment_count
            window_clips.append(window_clip)
        if len(window_clips) < 2:
            continue
        time_token, clip_clock = _clock_fields(window_start)
        groups.append(
            {
                "day": day,
                "time_token": time_token,
                "clip_clock": clip_clock,
                "clock_seconds": window_start / 100.0,
                "duration_seconds": duration_centiseconds / 100.0,
                "segment_count": segment_count,
                "agents": [clip["agent_dir"] for clip in window_clips],
                "clips": window_clips,
            }
        )
    return groups


def _safe_rel_path(repo_path: str) -> Path:
    return Path(*Path(repo_path).parts)


def _cache_day_dir_name(day: str) -> str:
    if day.startswith("DAY_"):
        return day
    suffix = day.removeprefix("DAY")
    return f"DAY_{suffix}" if suffix.isdigit() else day


def local_cache_path(cache_dir: str | Path, repo_path: str) -> Path:
    rel_path = _safe_rel_path(repo_path)
    parts = rel_path.parts
    normalized_parts = [
        _cache_day_dir_name(part) if part.startswith("DAY") else part
        for part in parts
    ]
    if normalized_parts:
        return Path(cache_dir).joinpath(*normalized_parts)
    return Path(cache_dir) / rel_path


def window_cache_path(
    cache_dir: str | Path,
    *,
    day: str,
    agent_dir: str,
    agent_id: str,
    agent_name: str,
    time_token: str,
    duration_seconds: float,
) -> Path:
    duration_label = f"{float(duration_seconds):g}".replace(".", "p")
    filename = f"{day}_{agent_id}_{str(agent_name).upper()}_{time_token}_{duration_label}s.mp4"
    return (
        Path(cache_dir)
        / "assembled_evidence"
        / agent_dir
        / _cache_day_dir_name(day)
        / filename
    )


def _ffconcat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def concatenate_video_segments(
    video_paths: list[str | Path],
    output_path: str | Path,
    *,
    duration_seconds: float,
) -> Path:
    """Concatenate a complete nominal window, padding small encoder shortfalls.

    The EgoLife filenames advance in exact 30-second slots, but some encoded
    MP4 streams end a fraction of a second early.  The normal path remains a
    lossless stream copy.  If a complete multi-segment window is no more than
    five percent short (capped at one source-clip duration), re-encode once and
    clone the final video frame so the materialized window reaches its nominal
    duration.  Larger shortfalls still fail as incomplete evidence.
    """

    if not video_paths:
        raise ValueError("video_paths must not be empty")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to assemble multi-segment evidence videos")
    resolved_paths = [Path(path) for path in video_paths]
    missing = [str(path) for path in resolved_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"cannot assemble evidence window; missing source segments: {missing[:3]}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_path: Path | None = None
    temporary_output: Path | None = None
    padded_output: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".ffconcat",
            prefix=f"{output_path.stem}_",
            dir=output_path.parent,
            delete=False,
        ) as concat_file:
            concat_path = Path(concat_file.name)
            concat_file.write("ffconcat version 1.0\n")
            for path in resolved_paths:
                concat_file.write(f"file '{_ffconcat_path(path)}'\n")
                concat_file.write(f"duration {SOURCE_CLIP_DURATION_SECONDS:.3f}\n")

        file_descriptor, temporary_name = tempfile.mkstemp(
            suffix=output_path.suffix or ".mp4",
            prefix=f"{output_path.stem}_",
            dir=output_path.parent,
        )
        temporary_output = Path(temporary_name)
        os.close(file_descriptor)
        temporary_output.unlink(missing_ok=True)
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_path),
                "-t",
                f"{float(duration_seconds):.3f}",
                "-map",
                "0:v:0",
                "-map",
                "0:a?",
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ],
            check=True,
        )
        actual_duration = ffprobe_duration(temporary_output)
        if actual_duration is not None and actual_duration + 0.5 < float(duration_seconds):
            requested_duration = float(duration_seconds)
            shortfall = requested_duration - actual_duration
            max_padding_seconds = min(
                SOURCE_CLIP_DURATION_SECONDS,
                requested_duration * 0.05,
            )
            if (
                requested_duration <= SOURCE_CLIP_DURATION_SECONDS
                or shortfall > max_padding_seconds
            ):
                raise RuntimeError(
                    "assembled evidence video is shorter than requested and exceeds the "
                    "safe padding limit: "
                    f"{actual_duration:.3f}s < {requested_duration:.3f}s "
                    f"(shortfall={shortfall:.3f}s, limit={max_padding_seconds:.3f}s)"
                )

            padded_descriptor, padded_name = tempfile.mkstemp(
                suffix=output_path.suffix or ".mp4",
                prefix=f"{output_path.stem}_padded_",
                dir=output_path.parent,
            )
            padded_output = Path(padded_name)
            os.close(padded_descriptor)
            padded_output.unlink(missing_ok=True)
            print(
                "assembled_evidence_padding "
                f"source_duration_seconds={actual_duration:.3f} "
                f"target_duration_seconds={requested_duration:.3f} "
                f"padding_seconds={shortfall:.3f}",
                flush=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(temporary_output),
                    "-t",
                    f"{requested_duration:.3f}",
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a?",
                    "-vf",
                    (
                        "tpad=stop_mode=clone:"
                        f"stop_duration={shortfall + 1.0:.3f}"
                    ),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-c:a",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(padded_output),
                ],
                check=True,
            )
            padded_duration = ffprobe_duration(padded_output)
            if padded_duration is not None and padded_duration + 0.5 < requested_duration:
                raise RuntimeError(
                    "padded evidence video is still shorter than requested: "
                    f"{padded_duration:.3f}s < {requested_duration:.3f}s"
                )
            padded_output.replace(temporary_output)
        temporary_output.replace(output_path)
    finally:
        if concat_path is not None:
            concat_path.unlink(missing_ok=True)
        if temporary_output is not None:
            temporary_output.unlink(missing_ok=True)
        if padded_output is not None:
            padded_output.unlink(missing_ok=True)
    return output_path


def ffprobe_duration(video_path: str | Path) -> float | None:
    if not shutil.which("ffprobe"):
        return None
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    frames_per_clip: int = 3,
    duration: float | None = None,
) -> list[dict[str, Any]]:
    """Extract evenly spaced frames with ffmpeg."""

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for frame extraction")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = duration if duration is not None else ffprobe_duration(video_path)
    if duration is None or duration <= 0:
        duration = 30.0
    frames_per_clip = max(1, frames_per_clip)
    timestamps = [
        min(duration - 0.05, max(0.0, duration * (idx + 1) / (frames_per_clip + 1)))
        for idx in range(frames_per_clip)
    ]
    rows: list[dict[str, Any]] = []
    for idx, timestamp in enumerate(timestamps, 1):
        frame_path = output_dir / f"frame_{idx:02d}_{timestamp:.2f}s.jpg"
        if not frame_path.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{timestamp:.3f}",
                    "-i",
                    str(video_path),
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(frame_path),
                ],
                check=True,
            )
        rows.append({"timestamp_seconds": round(timestamp, 3), "path": str(frame_path)})
    return rows


def summarize_gaze_csv(
    path: str | Path,
    *,
    max_rows: int = 5000,
    calibration_path: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize EyeGaze CSV and optionally project CPF gaze to image pixels.

    EgoLife gaze CSV values are Aria CPF yaw/pitch/depth values. Without a
    calibration file, this function intentionally does not produce 2D pixel
    gaze coordinates.
    """

    yaw_values: list[float] = []
    pitch_values: list[float] = []
    depth_values: list[float] = []
    projected_points: list[dict[str, Any]] = []
    calibration = None
    projectaria_projection_summary: dict[str, Any] | None = None
    if calibration_path:
        suffix = Path(calibration_path).suffix.lower()
        if suffix in {".vrs", ".jsonl"}:
            try:
                projected_points, projectaria_projection_summary = project_gaze_csv_with_projectaria_tools(
                    path,
                    calibration_path,
                    max_rows=max_rows,
                )
            except RuntimeError as exc:
                projectaria_projection_summary = {
                    "projection_status": "projection_failed",
                    "calibration_path": str(calibration_path),
                    "projection_error": str(exc),
                }
        else:
            calibration = load_aria_projection_calibration(calibration_path)
    projection_errors: list[str] = []
    first_ts = None
    last_ts = None
    total = 0
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for row in reader:
            total += 1
            if first_ts is None:
                first_ts = row.get("tracking_timestamp_us")
            last_ts = row.get("tracking_timestamp_us")
            if total > max_rows:
                continue
            for target, values in [
                ("pitch_rads_cpf", pitch_values),
                ("depth_m", depth_values),
            ]:
                try:
                    values.append(float(row[target]))
                except (KeyError, TypeError, ValueError):
                    pass
            try:
                left = float(row["left_yaw_rads_cpf"])
                right = float(row["right_yaw_rads_cpf"])
                yaw_values.append((left + right) / 2.0)
            except (KeyError, TypeError, ValueError):
                pass
            if calibration is not None:
                try:
                    projected = project_gaze_row(row, calibration)
                    if projected is not None:
                        projected_points.append(projected)
                except ValueError as exc:
                    if len(projection_errors) < 3:
                        projection_errors.append(str(exc))

    def stats(values: list[float]) -> dict[str, float] | None:
        if not values:
            return None
        return {
            "min": round(min(values), 5),
            "max": round(max(values), 5),
            "median": round(statistics.median(values), 5),
        }

    summary = {
        "row_count": total,
        "sampled_rows": min(total, max_rows),
        "first_tracking_timestamp_us": first_ts,
        "last_tracking_timestamp_us": last_ts,
        "columns": fields,
        "yaw_rads_summary": stats(yaw_values),
        "pitch_rads_summary": stats(pitch_values),
        "depth_m_summary": stats(depth_values),
        "gaze_coordinate_frame": "Aria Central Pupil Frame (CPF); not image pixels",
    }
    if calibration_path and projectaria_projection_summary is not None:
        summary["projection_status"] = projectaria_projection_summary["projection_status"]
        summary["projected_gaze_summary"] = projectaria_projection_summary
        summary["projected_gaze_points_sample"] = projected_points[:20]
    elif calibration is None:
        summary["projection_status"] = "missing_calibration"
        summary["projected_gaze_summary"] = None
    else:
        projection_summary = summarize_projected_gaze(projected_points, calibration)
        if projection_errors:
            projection_summary["projection_errors"] = projection_errors
        summary["projection_status"] = projection_summary["projection_status"]
        summary["projected_gaze_summary"] = projection_summary
        summary["projected_gaze_points_sample"] = projected_points[:20]
    return summary


def choose_required_clips(
    group: dict[str, Any],
    users_per_case: int,
    *,
    rng: random.Random | None = None,
) -> list[dict[str, Any]]:
    clips = sorted(group["clips"], key=lambda c: c["agent_dir"])
    count = min(len(clips), max(2, users_per_case))
    if rng is not None and len(clips) > count:
        return sorted(rng.sample(clips, count), key=lambda c: c["agent_dir"])
    return clips[:count]


def select_evidence_groups(
    groups: list[dict[str, Any]],
    *,
    target_count: int,
    random_seed: int | None = None,
    stratify_by_day: bool = False,
) -> list[dict[str, Any]]:
    """Select evidence groups, optionally spreading random samples across days."""

    if random_seed is None:
        return groups[:target_count]
    rng = random.Random(random_seed)
    if not stratify_by_day:
        shuffled = list(groups)
        rng.shuffle(shuffled)
        return shuffled[:target_count]

    by_day: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        by_day[str(group.get("day"))].append(group)
    day_names = sorted(by_day)
    for day_groups in by_day.values():
        rng.shuffle(day_groups)
    rng.shuffle(day_names)

    selected = []
    while len(selected) < target_count:
        added = False
        for day in day_names:
            if by_day[day]:
                selected.append(by_day[day].pop())
                added = True
                if len(selected) >= target_count:
                    break
        if not added:
            break
    return selected


def _combined_gaze_summary(
    segment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [row["gaze_summary"] for row in segment_rows if row.get("gaze_summary")]
    if not summaries:
        return {}
    if len(summaries) == 1:
        return summaries[0]
    statuses = [str(summary.get("projection_status", "unknown")) for summary in summaries]
    unique_statuses = sorted(set(statuses))
    return {
        "row_count": sum(int(summary.get("row_count") or 0) for summary in summaries),
        "sampled_rows": sum(int(summary.get("sampled_rows") or 0) for summary in summaries),
        "first_tracking_timestamp_us": summaries[0].get("first_tracking_timestamp_us"),
        "last_tracking_timestamp_us": summaries[-1].get("last_tracking_timestamp_us"),
        "gaze_coordinate_frame": "Aria Central Pupil Frame (CPF); not image pixels",
        "projection_status": unique_statuses[0] if len(unique_statuses) == 1 else "mixed_segment_status",
        "projected_gaze_summary": None,
        "segment_count": len(segment_rows),
        "segment_projection_statuses": statuses,
        "aggregation_note": (
            "This window spans multiple source gaze CSVs. Projection and distribution summaries "
            "remain available per source segment in source_segments."
        ),
    }


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def build_evidence_packet(
    group: dict[str, Any],
    *,
    cache_dir: str | Path,
    output_root: str | Path,
    users_per_case: int = 2,
    frames_per_clip: int = 3,
    aria_calibration_dir: str | Path | None = None,
    download_media: bool = True,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    selected = choose_required_clips(group, users_per_case, rng=rng)
    evidence_duration = float(group.get("duration_seconds") or SOURCE_CLIP_DURATION_SECONDS)
    packet_id_parts: list[Any] = ["EGOLIFE2U", group["day"], group["time_token"]]
    if evidence_duration != SOURCE_CLIP_DURATION_SECONDS:
        packet_id_parts.append(f"{evidence_duration:g}S")
    packet_id_parts.extend(c["agent_id"] for c in selected)
    packet_id = stable_id(*packet_id_parts)
    packet_dir = Path(output_root) / "evidence_assets" / packet_id
    clips_out = []

    for clip in selected:
        source_clips = list(clip.get("segments") or [clip])
        source_rows = []
        for segment_index, source_clip in enumerate(source_clips):
            source_local_video = local_cache_path(cache_dir, source_clip["video_path"])
            source_local_gaze = local_cache_path(cache_dir, source_clip["gaze_path"])
            if download_media:
                download_file(source_clip["video_url"], source_local_video)
                download_file(source_clip["gaze_url"], source_local_gaze)
            calibration_path = find_clip_calibration(aria_calibration_dir, source_clip)
            gaze_summary = (
                summarize_gaze_csv(source_local_gaze, calibration_path=calibration_path)
                if _nonempty_file(source_local_gaze)
                else {}
            )
            source_rows.append(
                {
                    "segment_index": segment_index,
                    "clip_id": source_clip.get("clip_id"),
                    "day": source_clip.get("day"),
                    "window_start_seconds": round(segment_index * SOURCE_CLIP_DURATION_SECONDS, 3),
                    "window_end_seconds": round(
                        min((segment_index + 1) * SOURCE_CLIP_DURATION_SECONDS, evidence_duration),
                        3,
                    ),
                    "time_token": source_clip["time_token"],
                    "clip_clock": source_clip.get("clip_clock"),
                    "video_url": source_clip["video_url"],
                    "gaze_url": source_clip["gaze_url"],
                    "overlay_url": source_clip.get("overlay_url"),
                    "local_video": (
                        str(source_local_video) if _nonempty_file(source_local_video) else None
                    ),
                    "local_gaze": (
                        str(source_local_gaze) if _nonempty_file(source_local_gaze) else None
                    ),
                    "local_calibration": str(calibration_path) if calibration_path else None,
                    "gaze_summary": gaze_summary,
                }
            )

        if len(source_rows) == 1:
            local_video = Path(source_rows[0]["local_video"]) if source_rows[0]["local_video"] else None
        else:
            assembled_path = window_cache_path(
                cache_dir,
                day=str(group["day"]),
                agent_dir=str(clip["agent_dir"]),
                agent_id=str(clip["agent_id"]),
                agent_name=str(clip["agent_name"]),
                time_token=str(group["time_token"]),
                duration_seconds=evidence_duration,
            )
            cached_duration = (
                ffprobe_duration(assembled_path) if _nonempty_file(assembled_path) else None
            )
            if cached_duration is not None and cached_duration + 0.5 < evidence_duration:
                assembled_path.unlink(missing_ok=True)
            if not _nonempty_file(assembled_path) and all(
                row.get("local_video") for row in source_rows
            ):
                concatenate_video_segments(
                    [str(row["local_video"]) for row in source_rows],
                    assembled_path,
                    duration_seconds=evidence_duration,
                )
            local_video = assembled_path if _nonempty_file(assembled_path) else None

        duration = ffprobe_duration(local_video) if local_video is not None else None
        frame_rows: list[dict[str, Any]] = []
        if local_video is not None:
            frame_dir = packet_dir / clip["agent_dir"]
            try:
                frame_rows = extract_frames(
                    local_video,
                    frame_dir,
                    frames_per_clip=frames_per_clip,
                    duration=duration if duration is not None else evidence_duration,
                )
            except subprocess.CalledProcessError:
                if not download_media:
                    raise
                local_video.unlink(missing_ok=True)
                shutil.rmtree(frame_dir, ignore_errors=True)
                if len(source_rows) == 1:
                    download_file(source_rows[0]["video_url"], local_video)
                else:
                    concatenate_video_segments(
                        [str(row["local_video"]) for row in source_rows],
                        local_video,
                        duration_seconds=evidence_duration,
                    )
                duration = ffprobe_duration(local_video) if _nonempty_file(local_video) else None
                frame_rows = extract_frames(
                    local_video,
                    frame_dir,
                    frames_per_clip=frames_per_clip,
                    duration=duration if duration is not None else evidence_duration,
                )
        gaze_summary = _combined_gaze_summary(source_rows)
        first_source = source_rows[0]
        clips_out.append(
            {
                "agent_dir": clip["agent_dir"],
                "agent_id": clip["agent_id"],
                "agent_name": clip["agent_name"],
                "day": clip["day"],
                "time_token": group["time_token"],
                "clip_clock": group.get("clip_clock"),
                "duration_seconds": duration if duration is not None else evidence_duration,
                "segment_count": len(source_rows),
                "video_url": first_source["video_url"] if len(source_rows) == 1 else None,
                "source_video_urls": [row["video_url"] for row in source_rows],
                "gaze_url": first_source["gaze_url"] if len(source_rows) == 1 else None,
                "source_gaze_urls": [row["gaze_url"] for row in source_rows],
                "overlay_url": first_source.get("overlay_url") if len(source_rows) == 1 else None,
                "local_video": str(local_video) if local_video is not None else None,
                "local_gaze": first_source.get("local_gaze") if len(source_rows) == 1 else None,
                "local_gazes": [row["local_gaze"] for row in source_rows if row.get("local_gaze")],
                "local_calibration": (
                    first_source.get("local_calibration") if len(source_rows) == 1 else None
                ),
                "source_segments": source_rows,
                "frames": frame_rows,
                "gaze_summary": gaze_summary,
            }
        )

    window_start_seconds = float(
        group.get("clock_seconds") or seconds_from_time_token(group["time_token"])
    )
    return {
        "evidence_id": packet_id,
        "day": group["day"],
        "time_token": group["time_token"],
        "clip_clock": group.get("clip_clock"),
        "duration_seconds": evidence_duration,
        "segment_count": int(
            group.get("segment_count") or len(selected[0].get("segments") or [selected[0]])
        ),
        "window": {
            "start_clock_seconds": window_start_seconds,
            "end_clock_seconds": window_start_seconds + evidence_duration,
            "duration_seconds": evidence_duration,
            "source_clip_duration_seconds": SOURCE_CLIP_DURATION_SECONDS,
        },
        "required_users": [clip["agent_name"] for clip in selected],
        "requirement": "The final question must require evidence from at least two listed users; any single listed user alone must be insufficient.",
        "clips": clips_out,
        "source_urls": {
            "videos": [
                source_clip["video_url"]
                for clip in selected
                for source_clip in (clip.get("segments") or [clip])
            ],
            "gazes": [
                source_clip["gaze_url"]
                for clip in selected
                for source_clip in (clip.get("segments") or [clip])
            ],
            "overlays": [
                source_clip.get("overlay_url")
                for clip in selected
                for source_clip in (clip.get("segments") or [clip])
                if source_clip.get("overlay_url")
            ],
        },
    }


def prepare_evidence(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    cache_dir: str | Path,
    output_root: str | Path,
    target_count: int = 20,
    users_per_case: int = 2,
    frames_per_clip: int = 3,
    evidence_duration_seconds: float = DEFAULT_EVIDENCE_DURATION_SECONDS,
    aria_calibration_dir: str | Path | None = None,
    max_groups: int | None = None,
    download_media: bool = True,
    random_seed: int | None = None,
    stratify_by_day: bool = False,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    groups = group_manifest_clips(
        manifest,
        evidence_duration_seconds=evidence_duration_seconds,
    )
    if max_groups is not None:
        groups = groups[:max_groups]
    groups = select_evidence_groups(
        groups,
        target_count=target_count,
        random_seed=random_seed,
        stratify_by_day=stratify_by_day,
    )
    rng = random.Random(random_seed) if random_seed is not None else None

    packets = []
    for group in groups:
        if len(packets) >= target_count:
            break
        packets.append(
            build_evidence_packet(
                group,
                cache_dir=cache_dir,
                output_root=output_root,
                users_per_case=users_per_case,
                frames_per_clip=frames_per_clip,
                aria_calibration_dir=aria_calibration_dir,
                download_media=download_media,
                rng=rng,
            )
        )
    write_jsonl(output_path, packets)
    return packets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare EgoLife two-user evidence packets")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    parser.add_argument("--output-root", default="egolife_two_user_qa/outputs/pilot_20")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--users-per-case", type=int, default=2)
    parser.add_argument("--frames-per-clip", type=int, default=3)
    parser.add_argument(
        "--evidence-duration-seconds",
        type=float,
        default=DEFAULT_EVIDENCE_DURATION_SECONDS,
        help="Complete synchronized evidence-window duration (default: 30 seconds)",
    )
    parser.add_argument("--aria-calibration-dir")
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--no-download-media", action="store_true")
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--stratify-by-day", action="store_true")
    args = parser.parse_args(argv)
    packets = prepare_evidence(
        manifest_path=args.manifest,
        output_path=args.output,
        cache_dir=args.cache_dir,
        output_root=args.output_root,
        target_count=args.target_count,
        users_per_case=args.users_per_case,
        frames_per_clip=args.frames_per_clip,
        evidence_duration_seconds=args.evidence_duration_seconds,
        aria_calibration_dir=args.aria_calibration_dir,
        max_groups=args.max_groups,
        download_media=not args.no_download_media,
        random_seed=args.random_seed,
        stratify_by_day=args.stratify_by_day,
    )
    print(f"wrote {len(packets)} evidence packets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
