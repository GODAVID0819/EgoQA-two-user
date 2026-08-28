"""按用户和原始 30 秒分段准备可审计的 evidence review 媒体。"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any


OPTION_LETTERS = ("A", "B", "C", "D", "E")


def _time_token_seconds(time_token: str) -> int | None:
    digits = re.sub(r"\D", "", str(time_token))
    if len(digits) < 6:
        return None
    hour = int(digits[0:2])
    minute = int(digits[2:4])
    second = int(digits[4:6])
    if hour > 23 or minute > 59 or second > 59:
        return None
    return hour * 3600 + minute * 60 + second


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
    """Return uniformly ordered 30-second specs for every required user."""

    users = [str(user) for user in packet.get("required_users") or []]
    clips = list(packet.get("clips") or [])
    if len(users) != 6 or len(clips) != 6 or len(full_video_paths) != 6:
        raise ValueError("chunked six-user evidence review requires six users, clips, and full videos")
    clips_by_user = {str(clip.get("agent_name") or clip.get("user")): clip for clip in clips}
    segment_counts = {
        len(list((clips_by_user.get(user) or {}).get("segments") or []))
        for user in users
    }
    if len(segment_counts) != 1 or next(iter(segment_counts), 0) not in {6, 20}:
        raise ValueError(
            "all six users must have the same complete segment count (6 or 20)"
        )
    result: dict[str, list[dict[str, Any]]] = {}
    for user, full_video_path in zip(users, full_video_paths):
        clip = clips_by_user.get(user)
        if not isinstance(clip, dict):
            raise ValueError(f"missing clip for required user: {user}")
        segments = list(clip.get("segments") or [])
        token_seconds = [
            _time_token_seconds(str(segment.get("time_token") or ""))
            for segment in segments
        ]
        if any(value is None for value in token_seconds) or any(
            (int(current) - int(previous)) % 86400 != 30
            for previous, current in zip(token_seconds, token_seconds[1:])
        ):
            raise ValueError(
                f"user {user} must have consecutive 30-second time tokens"
            )
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
            if claim.get("confidence") not in {"HIGH", "MEDIUM", "LOW"}:
                errors.append(
                    f"segments[{index}].claims[{claim_index}].confidence is invalid"
                )
    vote = observation.get("user_vote")
    if not isinstance(vote, dict):
        return [*errors, "user_vote must be an object"]
    visible = vote.get("visible")
    confidence = vote.get("confidence")
    option = vote.get("supported_option")
    indices = vote.get("supporting_segment_indices")
    if not isinstance(visible, bool):
        errors.append("user_vote.visible must be a boolean")
    if confidence not in {"HIGH", "MEDIUM", "LOW"}:
        errors.append("user_vote.confidence is invalid")
    if not isinstance(vote.get("reason"), str) or not vote["reason"].strip():
        errors.append("user_vote.reason must be a non-empty string")
    if not isinstance(indices, list) or any(
        not isinstance(value, int) or value < 0 or value >= len(expected_time_tokens)
        for value in (indices or [])
    ):
        errors.append("user_vote.supporting_segment_indices is invalid")
    high_visible = visible is True and confidence == "HIGH"
    if high_visible:
        if option not in OPTION_LETTERS:
            errors.append("HIGH visible vote must select exactly one A-E option")
        if not indices:
            errors.append("HIGH visible vote must cite at least one supporting segment")
    else:
        if option not in (None, ""):
            errors.append("non-HIGH vote must not select an option")
        if indices:
            errors.append("non-HIGH vote must not cite supporting segments")
    return errors


def aggregate_evidence_user_votes(
    correct: str,
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate one strict high-confidence answer vote per visible user."""

    counts = {letter: 0 for letter in OPTION_LETTERS}
    visible_users = []
    for observation in observations:
        vote = observation.get("user_vote")
        if not isinstance(vote, dict):
            continue
        if vote.get("visible") is not True or vote.get("confidence") != "HIGH":
            continue
        option = vote.get("supported_option")
        if option not in counts:
            continue
        counts[option] += 1
        visible_users.append(str(observation.get("user") or ""))

    visible_count = len(visible_users)
    threshold_options = [
        option
        for option, count in counts.items()
        if count >= 3 or count > visible_count / 2
    ]
    normalized_correct = str(correct or "").strip().upper()
    passed = threshold_options == [normalized_correct]
    return {
        "passed": passed,
        "correct": normalized_correct,
        "visible_user_count": visible_count,
        "visible_users": visible_users,
        "option_support_counts": counts,
        "threshold_options": threshold_options,
        "not_visible_user_count": len(observations) - visible_count,
        "rule": "support >= 3 users or support > half of strict visible users",
    }


def validate_evidence_premise_audit(audit: dict[str, Any]) -> list[str]:
    """Validate the text-only aggregation output without letting it recount votes."""

    expected_keys = {
        "premises_supported",
        "high_confidence_material_conflict",
        "reason",
    }
    errors = []
    if set(audit) != expected_keys:
        errors.append("premise audit must contain exactly the required fields")
    if not isinstance(audit.get("premises_supported"), bool):
        errors.append("premises_supported must be boolean")
    if not isinstance(audit.get("high_confidence_material_conflict"), bool):
        errors.append("high_confidence_material_conflict must be boolean")
    if not isinstance(audit.get("reason"), str) or not audit["reason"].strip():
        errors.append("reason must be a non-empty string")
    return errors
