"""Prepare two-user evidence packets from the EgoLife manifest."""

from __future__ import annotations

import argparse
import csv
import random
import shutil
import statistics
import subprocess
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


def group_manifest_clips(manifest: dict[str, Any]) -> list[dict[str, Any]]:
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
                "agents": unique_agents,
                "clips": sorted(clips, key=lambda c: c["agent_dir"]),
            }
        )
    return groups


def _safe_rel_path(repo_path: str) -> Path:
    return Path(*Path(repo_path).parts)


def local_cache_path(cache_dir: str | Path, repo_path: str) -> Path:
    return Path(cache_dir) / _safe_rel_path(repo_path)


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
    packet_id = stable_id("EGOLIFE2U", group["day"], group["time_token"], *[c["agent_id"] for c in selected])
    packet_dir = Path(output_root) / "evidence_assets" / packet_id
    clips_out = []

    for clip in selected:
        local_video = local_cache_path(cache_dir, clip["video_path"])
        local_gaze = local_cache_path(cache_dir, clip["gaze_path"])
        if download_media:
            download_file(clip["video_url"], local_video)
            download_file(clip["gaze_url"], local_gaze)
        duration = ffprobe_duration(local_video) if local_video.exists() else None
        frame_rows: list[dict[str, Any]] = []
        if local_video.exists():
            frame_dir = packet_dir / clip["agent_dir"]
            try:
                frame_rows = extract_frames(
                    local_video,
                    frame_dir,
                    frames_per_clip=frames_per_clip,
                    duration=duration,
                )
            except subprocess.CalledProcessError:
                if not download_media:
                    raise
                local_video.unlink(missing_ok=True)
                shutil.rmtree(frame_dir, ignore_errors=True)
                download_file(clip["video_url"], local_video)
                duration = ffprobe_duration(local_video) if local_video.exists() else None
                frame_rows = extract_frames(
                    local_video,
                    frame_dir,
                    frames_per_clip=frames_per_clip,
                    duration=duration,
                )
        calibration_path = find_clip_calibration(aria_calibration_dir, clip)
        gaze_summary = (
            summarize_gaze_csv(local_gaze, calibration_path=calibration_path) if local_gaze.exists() else {}
        )
        clips_out.append(
            {
                "agent_dir": clip["agent_dir"],
                "agent_id": clip["agent_id"],
                "agent_name": clip["agent_name"],
                "day": clip["day"],
                "time_token": clip["time_token"],
                "clip_clock": clip["clip_clock"],
                "duration_seconds": duration,
                "video_url": clip["video_url"],
                "gaze_url": clip["gaze_url"],
                "overlay_url": clip.get("overlay_url"),
                "local_video": str(local_video) if local_video.exists() else None,
                "local_gaze": str(local_gaze) if local_gaze.exists() else None,
                "local_calibration": str(calibration_path) if calibration_path else None,
                "frames": frame_rows,
                "gaze_summary": gaze_summary,
            }
        )

    return {
        "evidence_id": packet_id,
        "day": group["day"],
        "time_token": group["time_token"],
        "clip_clock": group.get("clip_clock"),
        "required_users": [clip["agent_name"] for clip in selected],
        "requirement": "The final question must require evidence from at least two listed users; any single listed user alone must be insufficient.",
        "clips": clips_out,
        "source_urls": {
            "videos": [clip["video_url"] for clip in selected],
            "gazes": [clip["gaze_url"] for clip in selected],
            "overlays": [clip.get("overlay_url") for clip in selected if clip.get("overlay_url")],
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
    aria_calibration_dir: str | Path | None = None,
    max_groups: int | None = None,
    download_media: bool = True,
    random_seed: int | None = None,
    stratify_by_day: bool = False,
) -> list[dict[str, Any]]:
    manifest = read_json(manifest_path)
    groups = group_manifest_clips(manifest)
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
