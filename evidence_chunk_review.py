"""按用户和原始 30 秒分段准备可审计的 evidence review 媒体。"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _clock_range(time_token: str, *, duration_seconds: int = 30) -> str:
    digits = re.sub(r"\D", "", str(time_token))
    if len(digits) < 6:
        return "unknown"
    hour = int(digits[0:2])
    minute = int(digits[2:4])
    second = int(digits[4:6])
    start = hour * 3600 + minute * 60 + second
    end = start + duration_seconds

    def render(value: int) -> str:
        return f"{value // 3600:02d}:{value % 3600 // 60:02d}:{value % 60:02d}"

    return f"{render(start)}-{render(end)}"


def evidence_segment_specs(
    packet: dict[str, Any],
    full_video_paths: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Return six ordered 30-second specs for every required user."""

    users = [str(user) for user in packet.get("required_users") or []]
    clips = list(packet.get("clips") or [])
    if len(users) != 6 or len(clips) != 6 or len(full_video_paths) != 6:
        raise ValueError("chunked six-user evidence review requires six users, clips, and full videos")
    clips_by_user = {str(clip.get("agent_name") or clip.get("user")): clip for clip in clips}
    result: dict[str, list[dict[str, Any]]] = {}
    for user, full_video_path in zip(users, full_video_paths):
        clip = clips_by_user.get(user)
        if not isinstance(clip, dict):
            raise ValueError(f"missing clip for required user: {user}")
        segments = list(clip.get("segments") or [])
        if len(segments) != 6:
            raise ValueError(f"user {user} must have exactly six original segments")
        rows = []
        for index, segment in enumerate(segments):
            rows.append(
                {
                    "user": user,
                    "segment_index": index,
                    "time_token": str(segment.get("time_token") or ""),
                    "original_time_range": _clock_range(
                        str(segment.get("time_token") or "")
                    ),
                    "video_url": segment.get("video_url"),
                    "full_video_path": str(full_video_path),
                    "start_seconds": float(index * 30),
                    "duration_seconds": 30.0,
                }
            )
        result[user] = rows
    return result


def default_evidence_chunk_cache_dir() -> Path:
    base = Path(os.getenv("QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR", ".qwen_memory_safe_video_cache"))
    return base / "evidence_30s_segments"


def materialize_evidence_segment_paths(
    packet: dict[str, Any],
    specs_by_user: dict[str, list[dict[str, Any]]],
    *,
    cache_dir: str | Path | None = None,
    ffmpeg_binary: str | None = None,
    command_runner: Callable[..., Any] = subprocess.run,
) -> dict[str, list[str]]:
    """Split cached full videos into reusable 30-second review chunks."""

    root = Path(cache_dir) if cache_dir is not None else default_evidence_chunk_cache_dir()
    evidence_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(packet.get("evidence_id") or "evidence"))
    ffmpeg = ffmpeg_binary or os.getenv("FFMPEG_BINARY", "ffmpeg")
    outputs: dict[str, list[str]] = {}
    for user, specs in specs_by_user.items():
        safe_user = re.sub(r"[^A-Za-z0-9_.-]+", "_", user)
        user_dir = root / evidence_id / safe_user
        user_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for spec in specs:
            output = user_dir / f"segment_{int(spec['segment_index']):02d}_{spec['time_token']}.mp4"
            if not output.is_file() or output.stat().st_size <= 0:
                temporary = output.with_name(f".{output.stem}.part.mp4")
                command = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-ss",
                    f"{float(spec['start_seconds']):.3f}",
                    "-i",
                    str(spec["full_video_path"]),
                    "-t",
                    f"{float(spec['duration_seconds']):.3f}",
                    "-map",
                    "0:v:0",
                    "-an",
                    "-c:v",
                    "copy",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(temporary),
                ]
                try:
                    command_runner(command, check=True, capture_output=True, text=True)
                    if not temporary.is_file() or temporary.stat().st_size <= 0:
                        raise RuntimeError(f"ffmpeg did not create segment: {temporary}")
                    temporary.replace(output)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            paths.append(str(output))
        outputs[user] = paths
    return outputs


def validate_segment_observation(
    observation: dict[str, Any],
    *,
    expected_user: str,
    expected_time_tokens: list[str],
) -> list[str]:
    errors = []
    if observation.get("user") != expected_user:
        errors.append(f"user must equal {expected_user}")
    segments = observation.get("segments")
    if not isinstance(segments, list) or len(segments) != len(expected_time_tokens):
        return [*errors, "segments must cover every supplied 30-second video"]
    for index, (segment, expected_token) in enumerate(zip(segments, expected_time_tokens)):
        if not isinstance(segment, dict):
            errors.append(f"segments[{index}] must be an object")
            continue
        if segment.get("segment_index") != index:
            errors.append(f"segments[{index}].segment_index must equal {index}")
        if str(segment.get("time_token") or "") != expected_token:
            errors.append(f"segments[{index}].time_token must equal {expected_token}")
        claims = segment.get("claims")
        if not isinstance(claims, list):
            errors.append(f"segments[{index}].claims must be an array")
            continue
        for claim_index, claim in enumerate(claims):
            if not isinstance(claim, dict):
                errors.append(f"segments[{index}].claims[{claim_index}] must be an object")
                continue
            if claim.get("status") not in {
                "SUPPORTED",
                "CONTRADICTED",
                "NOT_VISIBLE",
                "AMBIGUOUS",
            }:
                errors.append(
                    f"segments[{index}].claims[{claim_index}].status is invalid"
                )
    return errors
