"""Canonical media binding for retained-frame GRPO and full-video review."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


GENERATOR_MEDIA_MODE = "retained_cluster_frames_only"
REVIEWER_VIDEO_FIELDS = (
    "full_local_video",
    "original_local_video",
    "source_local_video",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def required_users(packet: Mapping[str, Any]) -> list[str]:
    """Return the two distinct required users in their declared order."""

    value = packet.get("required_users")
    if not isinstance(value, list):
        raise ValueError("packet.required_users must be a list")
    users = [str(item).strip() for item in value]
    if len(users) != 2 or any(not user for user in users) or len(set(users)) != 2:
        raise ValueError(
            "packet.required_users must contain exactly two distinct non-empty users"
        )
    return users


def _clip_user(clip: Mapping[str, Any]) -> str:
    agent_name = str(clip.get("agent_name") or "").strip()
    legacy_user = str(clip.get("user") or "").strip()
    if agent_name and legacy_user and agent_name != legacy_user:
        raise ValueError(
            f"clip agent_name/user disagree: {agent_name!r} != {legacy_user!r}"
        )
    return agent_name or legacy_user


def clip_for_user(packet: Mapping[str, Any], user: str) -> Mapping[str, Any]:
    """Bind one required user to exactly one clip, rejecting ambiguous packets."""

    clips = packet.get("clips")
    if not isinstance(clips, list):
        raise ValueError("packet.clips must be a list")
    matches = [
        clip
        for clip in clips
        if isinstance(clip, Mapping) and _clip_user(clip) == user
    ]
    if len(matches) != 1:
        raise ValueError(
            f"required user {user!r} must map to exactly one clip; found {len(matches)}"
        )
    return matches[0]


def _materialized_path(
    value: Any,
    *,
    user: str,
    media_kind: str,
    suffixes: set[str],
    require_file: bool,
) -> str:
    if isinstance(value, (list, tuple, dict)) or not str(value or "").strip():
        raise ValueError(f"{media_kind} for required user {user!r} must be one path")
    path = Path(str(value)).expanduser()
    if path.suffix.lower() not in suffixes:
        expected = ", ".join(sorted(suffixes))
        raise ValueError(
            f"{media_kind} for required user {user!r} has unsupported suffix "
            f"{path.suffix!r}; expected one of: {expected}"
        )
    if require_file and (not path.is_file() or path.stat().st_size <= 0):
        raise FileNotFoundError(
            f"materialized {media_kind} is missing or empty for {user!r}: {path}"
        )
    return str(path.resolve())


def ordered_generator_frame_paths(
    packet: Mapping[str, Any],
    *,
    require_files: bool = True,
) -> list[str]:
    """Flatten retained frame paths in required-user, then chronological, order."""

    if packet.get("generator_media_mode") != GENERATOR_MEDIA_MODE:
        raise ValueError(
            "packet.generator_media_mode must be "
            f"{GENERATOR_MEDIA_MODE!r}"
        )
    clips = packet.get("clips")
    if not isinstance(clips, list) or len(clips) != 2 or not all(
        isinstance(clip, Mapping) for clip in clips
    ):
        raise ValueError("retained-frame packets must contain exactly two clip objects")
    paths: list[str] = []
    for user in required_users(packet):
        clip = clip_for_user(packet, user)
        if clip.get("generator_media_mode") != GENERATOR_MEDIA_MODE:
            raise ValueError(
                f"required user {user!r} generator_media_mode must be "
                f"{GENERATOR_MEDIA_MODE!r}"
            )
        if clip.get("force_frame_inputs") is not True:
            raise ValueError(
                f"required user {user!r} must set force_frame_inputs=true"
            )
        if str(clip.get("local_video") or "").strip() or str(
            clip.get("generator_local_video") or ""
        ).strip():
            raise ValueError(
                f"required user {user!r} exposes generator video media; "
                "retained-frame GRPO requires images only"
            )
        frames = clip.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(
                f"required user {user!r} must have retained frames in clips[*].frames"
            )
        timestamps: list[float] = []
        for index, frame in enumerate(frames, start=1):
            if not isinstance(frame, Mapping):
                raise ValueError(
                    f"retained frame {index} for required user {user!r} must be an object"
                )
            input_order = frame.get("input_order_within_user")
            if input_order is not None and input_order != index:
                raise ValueError(
                    f"retained frames for required user {user!r} are not in their "
                    "declared input_order_within_user"
                )
            timestamp = frame.get("timestamp_seconds")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
                raise ValueError(
                    f"retained frame {index} for required user {user!r} needs a numeric "
                    "timestamp_seconds"
                )
            numeric_timestamp = float(timestamp)
            if not math.isfinite(numeric_timestamp):
                raise ValueError(
                    f"retained frame {index} for required user {user!r} needs a finite "
                    "timestamp_seconds"
                )
            timestamps.append(numeric_timestamp)
            paths.append(
                _materialized_path(
                    frame.get("path"),
                    user=user,
                    media_kind="retained generator frame",
                    suffixes=IMAGE_SUFFIXES,
                    require_file=require_files,
                )
            )
        if timestamps != sorted(timestamps):
            raise ValueError(
                f"retained frames for required user {user!r} must be chronological"
            )
    if len(paths) != len(set(paths)):
        raise ValueError("retained generator frame paths must be unique within a packet")
    return paths


def generator_frame_counts(packet: Mapping[str, Any]) -> list[int]:
    """Return retained-frame counts in required_users order."""

    counts = []
    for user in required_users(packet):
        frames = clip_for_user(packet, user).get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"required user {user!r} has no retained frames")
        counts.append(len(frames))
    return counts


def _reviewer_video_path(
    clip: Mapping[str, Any],
    *,
    user: str,
    require_file: bool,
) -> str:
    selected_field = next(
        (
            field
            for field in REVIEWER_VIDEO_FIELDS
            if str(clip.get(field) or "").strip()
        ),
        None,
    )
    if selected_field is None:
        expected = ", ".join(REVIEWER_VIDEO_FIELDS)
        raise ValueError(
            f"required user {user!r} has no reviewer video; expected one of: {expected}"
        )
    return _materialized_path(
        clip[selected_field],
        user=user,
        media_kind="reviewer video",
        suffixes={".mp4"},
        require_file=require_file,
    )


def ordered_reviewer_video_paths(
    packet: Mapping[str, Any],
    *,
    require_files: bool = True,
) -> list[str]:
    """Resolve one full video per required user in required_users order."""

    return [
        _reviewer_video_path(
            clip_for_user(packet, user),
            user=user,
            require_file=require_files,
        )
        for user in required_users(packet)
    ]


def validate_bound_paths(
    values: Any,
    expected: Sequence[str],
    *,
    field: str,
) -> None:
    """Require a dataset media column to equal packet-derived paths exactly."""

    if not isinstance(values, list) or len(values) != len(expected):
        raise ValueError(f"{field} must contain exactly {len(expected)} paths")
    if any(isinstance(value, (list, tuple, dict)) for value in values):
        raise ValueError(f"{field} must contain paths, not nested media lists")
    actual = [str(Path(str(value)).expanduser().resolve()) for value in values]
    if actual != list(expected):
        raise ValueError(
            f"{field} does not match packet media in required_users order: "
            f"dataset={actual}, packet={list(expected)}"
        )
