"""Materialize local videos for manual QA review."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .io_utils import download_file, iter_jsonl, stable_id, write_json


def _existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.exists() and path.is_file() else None


def _video_filename(row: dict[str, Any], index: int) -> str:
    user = stable_id(row.get("agent_dir") or row.get("user") or f"user_{index + 1}")
    source_name = Path(str(row.get("video_url") or row.get("local_video") or "video.mp4")).name
    if not source_name.lower().endswith((".mp4", ".mov", ".mkv", ".avi", ".webm")):
        source_name = "video.mp4"
    return f"{index + 1:02d}_{user}_{source_name}"


def _copy_or_download_video(
    row: dict[str, Any],
    output_path: Path,
    *,
    download_missing: bool,
) -> dict[str, Any]:
    local_video = _existing_path(row.get("local_video"))
    source = "missing"
    status = "missing"
    error = ""

    try:
        if local_video:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if local_video.resolve() != output_path.resolve():
                shutil.copy2(local_video, output_path)
            source = "local_video"
            status = "ok"
        elif download_missing and row.get("video_url"):
            download_file(str(row["video_url"]), output_path)
            source = "video_url"
            status = "ok"
        elif row.get("video_url"):
            status = "not_downloaded"
            source = "video_url"
    except Exception as exc:
        status = "error"
        error = f"{type(exc).__name__}: {exc}"

    return {
        "user": row.get("user") or row.get("agent_name"),
        "agent_dir": row.get("agent_dir"),
        "agent_id": row.get("agent_id"),
        "day": row.get("day"),
        "time_token": row.get("time_token"),
        "clip_clock": row.get("clip_clock"),
        "video_url": row.get("video_url"),
        "source_local_video": row.get("local_video"),
        "review_video": str(output_path) if output_path.exists() else None,
        "status": status,
        "source": source,
        "error": error,
    }


def _packet_video_rows(packet: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for clip in packet.get("clips", []):
        rows.append(
            {
                "user": clip.get("agent_name"),
                "agent_name": clip.get("agent_name"),
                "agent_dir": clip.get("agent_dir"),
                "agent_id": clip.get("agent_id"),
                "day": clip.get("day"),
                "time_token": clip.get("time_token"),
                "clip_clock": clip.get("clip_clock"),
                "video_url": clip.get("video_url"),
                "local_video": clip.get("local_video"),
            }
        )
    return rows


def _qa_video_rows(qa: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for item in qa.get("video_evidence", []):
        rows.append(
            {
                "user": item.get("user"),
                "agent_dir": item.get("agent_dir"),
                "agent_id": item.get("agent_id"),
                "day": item.get("day"),
                "time_token": item.get("time_token"),
                "clip_clock": item.get("clip_clock"),
                "video_url": item.get("video_url"),
                "local_video": item.get("local_video"),
            }
        )
    return rows


def materialize_review_videos(
    *,
    evidence_path: str | Path | None = None,
    qa_path: str | Path | None = None,
    output_dir: str | Path,
    download_missing: bool = True,
) -> dict[str, Any]:
    """Copy/download evidence videos into a review folder and write a manifest."""

    if not evidence_path and not qa_path:
        raise ValueError("one of evidence_path or qa_path is required")

    output_dir = Path(output_dir)
    videos_dir = output_dir / "videos"
    manifest_rows = []
    seen: set[tuple[str, str | None, str | None]] = set()

    if evidence_path:
        for packet in iter_jsonl(evidence_path):
            evidence_id = str(packet.get("evidence_id") or stable_id(packet.get("day"), packet.get("time_token")))
            video_rows = []
            for index, row in enumerate(_packet_video_rows(packet)):
                key = (evidence_id, row.get("user"), row.get("video_url") or row.get("local_video"))
                if key in seen:
                    continue
                seen.add(key)
                target = videos_dir / evidence_id / _video_filename(row, index)
                video_rows.append(
                    _copy_or_download_video(row, target, download_missing=download_missing)
                )
            manifest_rows.append(
                {
                    "source_type": "evidence",
                    "evidence_id": evidence_id,
                    "day": packet.get("day"),
                    "time_token": packet.get("time_token"),
                    "required_users": packet.get("required_users", []),
                    "videos": video_rows,
                }
            )

    if qa_path:
        for qa in iter_jsonl(qa_path):
            evidence_id = str(qa.get("evidence_id") or qa.get("qa_id") or "unknown_evidence")
            video_rows = []
            for index, row in enumerate(_qa_video_rows(qa)):
                key = (evidence_id, row.get("user"), row.get("video_url") or row.get("local_video"))
                if key in seen:
                    continue
                seen.add(key)
                target = videos_dir / evidence_id / _video_filename(row, index)
                video_rows.append(
                    _copy_or_download_video(row, target, download_missing=download_missing)
                )
            manifest_rows.append(
                {
                    "source_type": "qa",
                    "qa_id": qa.get("qa_id"),
                    "evidence_id": evidence_id,
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "correct": qa.get("correct"),
                    "required_users": qa.get("required_users", []),
                    "videos": video_rows,
                }
            )

    ok_count = sum(
        1
        for row in manifest_rows
        for video in row.get("videos", [])
        if video.get("status") == "ok"
    )
    error_count = sum(
        1
        for row in manifest_rows
        for video in row.get("videos", [])
        if video.get("status") == "error"
    )
    missing_count = sum(
        1
        for row in manifest_rows
        for video in row.get("videos", [])
        if video.get("status") in {"missing", "not_downloaded"}
    )
    manifest = {
        "output_dir": str(output_dir),
        "videos_dir": str(videos_dir),
        "download_missing": download_missing,
        "row_count": len(manifest_rows),
        "video_count_ok": ok_count,
        "video_count_error": error_count,
        "video_count_missing": missing_count,
        "rows": manifest_rows,
    }
    write_json(output_dir / "review_video_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Copy/download evidence videos for manual review")
    parser.add_argument("--evidence")
    parser.add_argument("--qa")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args(argv)
    manifest = materialize_review_videos(
        evidence_path=args.evidence,
        qa_path=args.qa,
        output_dir=args.output_dir,
        download_missing=not args.no_download,
    )
    print(
        "materialized "
        f"{manifest['video_count_ok']} videos to {manifest['videos_dir']} "
        f"({manifest['video_count_error']} errors, {manifest['video_count_missing']} missing)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
