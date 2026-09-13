"""Restore missing original/pruned MP4s referenced by a saved evidence packet."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .evidence import local_cache_path
from .group_relative_clip_sampling import materialize_pruned_video
from .io_utils import download_file, iter_jsonl, write_json


def _nonempty_file(path: str | Path | None) -> bool:
    if not path:
        return False
    candidate = Path(str(path))
    return candidate.is_file() and candidate.stat().st_size > 0


def _unique_paths(values: list[Any]) -> list[Path]:
    paths = []
    seen = set()
    for value in values:
        if not value:
            continue
        path = Path(str(value))
        key = str(path)
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return paths


def _copy_if_missing(source: Path, destination: Path) -> str:
    if _nonempty_file(destination):
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f"{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copy2(source, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    if not _nonempty_file(destination):
        raise RuntimeError(f"failed to restore original video: {destination}")
    return "restored"


def _materialize_pruned_if_missing(
    source: Path,
    destination: Path,
    keep_intervals: list[list[float]] | list[tuple[float, float]],
    *,
    ffmpeg_binary: str,
) -> str:
    if _nonempty_file(destination):
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.stem}.repair.tmp{destination.suffix}")
    temporary.unlink(missing_ok=True)
    try:
        materialize_pruned_video(
            source,
            temporary,
            keep_intervals,
            ffmpeg_binary=ffmpeg_binary,
        )
        if not _nonempty_file(temporary):
            raise RuntimeError(f"pruned-video repair produced no media: {temporary}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "restored"


def _source_video(
    packet_id: str,
    clip: dict[str, Any],
    *,
    cache_dir: Path | None,
    download_missing_source: bool,
) -> tuple[Path, str]:
    benchmark_media = clip.get("benchmark_media")
    if not isinstance(benchmark_media, dict):
        benchmark_media = {}
    candidates = _unique_paths(
        [
            clip.get("source_local_video"),
            benchmark_media.get("source_cache_video"),
        ]
    )
    for candidate in candidates:
        if _nonempty_file(candidate):
            return candidate, "existing"

    video_path = clip.get("video_path")
    video_url = clip.get("video_url")
    if candidates:
        destination = candidates[0]
    elif cache_dir is not None and video_path:
        destination = local_cache_path(cache_dir, str(video_path))
    else:
        raise FileNotFoundError(
            f"{packet_id}: {clip.get('agent_dir')} has no recoverable source-video path"
        )
    if not download_missing_source:
        raise FileNotFoundError(
            f"{packet_id}: source video is missing for {clip.get('agent_dir')}: {destination}"
        )
    if not video_url:
        raise FileNotFoundError(
            f"{packet_id}: cannot redownload source video for {clip.get('agent_dir')}; "
            "video_url is missing"
        )
    download_file(str(video_url), destination)
    if not _nonempty_file(destination):
        raise RuntimeError(f"source download produced no media: {destination}")
    return destination, "downloaded"


def _keep_intervals(packet_id: str, clip: dict[str, Any]) -> list[list[float]]:
    pruning = clip.get("temporal_pruning")
    if not isinstance(pruning, dict):
        pruning = {}
    intervals = pruning.get("keep_intervals")
    if not isinstance(intervals, list) or not intervals:
        raise ValueError(
            f"{packet_id}: saved keep_intervals are unavailable for "
            f"{clip.get('agent_dir')}"
        )
    return intervals


def repair_pruned_pair_media(
    *,
    evidence_path: str | Path,
    day: str,
    time_token: str,
    evidence_id: str | None = None,
    cache_dir: str | Path | None = None,
    ffmpeg_binary: str = "ffmpeg",
    download_missing_source: bool = True,
    report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Restore the saved pair media without changing packet JSON or its paths."""

    evidence_path = Path(evidence_path)
    matches = [
        packet
        for packet in iter_jsonl(evidence_path)
        if str(packet.get("day") or "") == str(day)
        and str(packet.get("time_token") or "") == str(time_token)
        and (
            evidence_id is None
            or str(packet.get("evidence_id") or "") == str(evidence_id)
        )
    ]
    if not matches:
        suffix = f" and evidence_id={evidence_id}" if evidence_id else ""
        raise ValueError(
            f"no packet found for day={day} time_token={time_token}{suffix}"
        )
    if len(matches) != 1:
        raise ValueError(
            f"expected one packet for {day}/{time_token}, found {len(matches)}; "
            "pass --evidence-id to disambiguate"
        )

    packet = matches[0]
    packet_id = str(packet.get("evidence_id") or "<missing evidence_id>")
    clips = packet.get("clips")
    if not isinstance(clips, list) or len(clips) != 2:
        raise ValueError(f"{packet_id}: expected exactly two selected clips")

    clip_reports = []
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError(f"{packet_id}: selected clip is not an object")
        source, source_status = _source_video(
            packet_id,
            clip,
            cache_dir=Path(cache_dir) if cache_dir is not None else None,
            download_missing_source=download_missing_source,
        )
        benchmark_media = clip.get("benchmark_media")
        if not isinstance(benchmark_media, dict):
            benchmark_media = {}

        original_targets = _unique_paths(
            [
                clip.get("original_local_video"),
                clip.get("full_local_video"),
                benchmark_media.get("judge_video"),
                benchmark_media.get("answerability_video"),
            ]
        )
        pruned_targets = _unique_paths(
            [
                clip.get("generator_local_video"),
                clip.get("local_video"),
                benchmark_media.get("generator_video"),
            ]
        )
        if not original_targets:
            raise ValueError(
                f"{packet_id}: no saved original-video destination for "
                f"{clip.get('agent_dir')}"
            )
        if not pruned_targets:
            raise ValueError(
                f"{packet_id}: no saved pruned-video destination for "
                f"{clip.get('agent_dir')}"
            )

        original_results = [
            {
                "path": str(target),
                "status": _copy_if_missing(source, target),
            }
            for target in original_targets
        ]
        intervals = _keep_intervals(packet_id, clip)
        pruned_results = [
            {
                "path": str(target),
                "status": _materialize_pruned_if_missing(
                    source,
                    target,
                    intervals,
                    ffmpeg_binary=ffmpeg_binary,
                ),
            }
            for target in pruned_targets
        ]
        clip_reports.append(
            {
                "agent_dir": clip.get("agent_dir"),
                "agent_name": clip.get("agent_name"),
                "source_video": str(source),
                "source_status": source_status,
                "keep_intervals": intervals,
                "original_videos": original_results,
                "pruned_videos": pruned_results,
            }
        )

    report = {
        "evidence_id": packet_id,
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "evidence_path": str(evidence_path),
        "packet_json_modified": False,
        "clips": clip_reports,
        "restored_original_video_count": sum(
            row["status"] == "restored"
            for clip in clip_reports
            for row in clip["original_videos"]
        ),
        "restored_pruned_video_count": sum(
            row["status"] == "restored"
            for clip in clip_reports
            for row in clip["pruned_videos"]
        ),
    }
    destination = (
        Path(report_path)
        if report_path is not None
        else evidence_path.parent
        / f"pair_media_repair_{day}_{time_token}.json"
    )
    write_json(destination, report)
    report["report_path"] = str(destination)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore missing benchmark_video_pairs originals and pruned MP4s "
            "from a saved evidence packet"
        )
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--day", required=True)
    parser.add_argument("--time-token", required=True)
    parser.add_argument("--evidence-id")
    parser.add_argument("--cache-dir")
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--no-download-source", action="store_true")
    parser.add_argument("--report")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = repair_pruned_pair_media(
        evidence_path=args.evidence,
        day=args.day,
        time_token=args.time_token,
        evidence_id=args.evidence_id,
        cache_dir=args.cache_dir,
        ffmpeg_binary=args.ffmpeg_binary,
        download_missing_source=not args.no_download_source,
        report_path=args.report,
    )
    print(
        f"repaired {report['evidence_id']}: "
        f"originals={report['restored_original_video_count']} "
        f"pruned={report['restored_pruned_video_count']} "
        f"report={report['report_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
