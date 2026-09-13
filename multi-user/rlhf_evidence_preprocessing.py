"""Build reusable six-user RLHF evidence without running a generator.

Each synchronized ten-minute source window is sampled once, encoded once, and
clustered once per user.  The packet then stores six asker-conditioned masks:
the asker's own frames are always complete, while redundant provider clusters
are removed relative to that asker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
import types
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# This file is intentionally runnable both as a package module and as a script
# copied into the HPC package directory.
if __package__ in {None, ""}:
    package_root = Path(__file__).resolve().parent
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(package_root)]
    sys.modules.setdefault("egolife_two_user_qa", package)
    __package__ = "egolife_two_user_qa"

from .clip_gap_demo import DEFAULT_CLIP_MODEL
from .evidence import (
    LONG_CONTEXT_EVIDENCE_DURATION_SECONDS,
    build_evidence_packet,
    group_manifest_clips,
)
from .group_relative_clip_sampling import (
    cluster_six_user_frame_representatives,
    clustered_speaker_provider_all_pairs_pruning,
    group_clip_frames,
)
from .io_utils import read_json, stable_id, write_json
from .manifest import seconds_from_time_token
from .ten_minute_six_user_setup import LazyBatchedImageEncoder


SCHEMA_VERSION = "egolife_rlhf_evidence_v1"
GENERATOR_MEDIA_MODE = "asker_full_provider_pruned"
USER_COUNT = 6
DEFAULT_DURATION_SECONDS = LONG_CONTEXT_EVIDENCE_DURATION_SECONDS
DEFAULT_SAMPLE_FPS = 0.5
DEFAULT_CLUSTERS_PER_30_SECONDS = 6
DEFAULT_CLUSTER_DENSITY_WINDOW_SECONDS = 30.0
DEFAULT_TEMPORAL_KMEANS_TIME_WEIGHT = 0.1
DEFAULT_TEMPORAL_UNIT_SECONDS = 30.0
DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS = 30.0
DEFAULT_HIGH_SIMILARITY_THRESHOLD = 0.82
DEFAULT_MIN_PROVIDER_RETENTION_PERCENT = 40.0
DEFAULT_JPEG_QUALITY = 95
DEFAULT_CLIP_BATCH_SIZE = 256
DEFAULT_VIDEO_SAMPLE_WORKERS = 6
DEFAULT_MEDIA_PREPARE_WORKERS = 6
DEFAULT_RANDOM_SEED = 20260902


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    materialized = list(rows)
    body = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in materialized)
    _atomic_write_text(path, body)
    return len(materialized)


def _safe_component(value: Any) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip()).strip("._")
    return cleaned or "unknown"


def build_preprocessing_config(
    *,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    sample_fps: float = DEFAULT_SAMPLE_FPS,
    clusters_per_density_window: int = DEFAULT_CLUSTERS_PER_30_SECONDS,
    cluster_density_window_seconds: float = DEFAULT_CLUSTER_DENSITY_WINDOW_SECONDS,
    temporal_kmeans_time_weight: float = DEFAULT_TEMPORAL_KMEANS_TIME_WEIGHT,
    temporal_unit_seconds: float = DEFAULT_TEMPORAL_UNIT_SECONDS,
    max_pair_time_difference_seconds: float = DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    high_similarity_threshold: float = DEFAULT_HIGH_SIMILARITY_THRESHOLD,
    min_provider_retention_percent: float = DEFAULT_MIN_PROVIDER_RETENTION_PERCENT,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    clip_model_id: str = DEFAULT_CLIP_MODEL,
) -> dict[str, Any]:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    if clusters_per_density_window <= 0:
        raise ValueError("clusters_per_density_window must be positive")
    if cluster_density_window_seconds <= 0 or temporal_unit_seconds <= 0:
        raise ValueError("clustering time units must be positive")
    if temporal_kmeans_time_weight <= 0:
        raise ValueError("offline RLHF preprocessing requires positive time-aware weight")
    if max_pair_time_difference_seconds < 0:
        raise ValueError("max_pair_time_difference_seconds must be non-negative")
    if not 0 <= high_similarity_threshold <= 1:
        raise ValueError("high_similarity_threshold must be between 0 and 1")
    if not 0 < min_provider_retention_percent <= 100:
        raise ValueError("min_provider_retention_percent must be in (0, 100]")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    sample_interval_seconds = 1.0 / float(sample_fps)
    expected_frames = int(math.ceil(float(duration_seconds) / sample_interval_seconds))
    global_cluster_count = max(
        1,
        int(
            math.ceil(
                float(duration_seconds)
                / float(cluster_density_window_seconds)
                * int(clusters_per_density_window)
            )
        ),
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "user_count": USER_COUNT,
        "duration_seconds": float(duration_seconds),
        "sampling": {
            "fps": float(sample_fps),
            "interval_seconds": sample_interval_seconds,
            "expected_frames_per_user": expected_frames,
            "image_format": "jpeg",
            "jpeg_quality": int(jpeg_quality),
            "jpeg_subsampling": 0,
            "clip_reads_persisted_images": True,
        },
        "clip": {
            "model_id": str(clip_model_id),
            "saved_dtype": "float16",
            "scope": "once_per_packet_for_all_six_users",
        },
        "clustering": {
            "method": "time_aware_spherical_kmeans_medoid",
            "scope": "independently_per_user_across_entire_ten_minute_video",
            "clusters_per_density_window": int(clusters_per_density_window),
            "density_window_seconds": float(cluster_density_window_seconds),
            "global_cluster_count_requested": global_cluster_count,
            "temporal_kmeans_time_weight": float(temporal_kmeans_time_weight),
            "temporal_unit_seconds": float(temporal_unit_seconds),
            "split_noncontiguous_clusters": False,
            "max_cluster_member_gap_seconds": None,
            "objective": (
                "2*(1-clip_cosine_similarity) + "
                "time_weight*((frame_time-center_time)/temporal_unit_seconds)^2"
            ),
        },
        "pruning": {
            "asker_frames": "always_all_sampled_frames",
            "provider_frames": "remove_provider_clusters_redundant_with_asker",
            "high_similarity_threshold": float(high_similarity_threshold),
            "max_pair_time_difference_seconds": float(
                max_pair_time_difference_seconds
            ),
            "mutual_nearest_only": False,
            "minimum_provider_retention_percent": float(
                min_provider_retention_percent
            ),
            "no_redundancy_is_valid": True,
        },
        "generator": {
            "media_mode": GENERATOR_MEDIA_MODE,
            "ordering": "asker_first_then_providers_in_canonical_user_order",
        },
    }


def _canonical_group(group: dict[str, Any]) -> dict[str, Any]:
    result = dict(group)
    result["clips"] = sorted(
        (dict(clip) for clip in group.get("clips", [])),
        key=lambda clip: (
            str(clip.get("agent_dir", "")),
            str(clip.get("agent_id", "")),
        ),
    )
    result["agents"] = [str(clip.get("agent_dir")) for clip in result["clips"]]
    return result


def _clock_fields_from_centiseconds(clock_centiseconds: int) -> tuple[str, str]:
    hours, remainder = divmod(int(clock_centiseconds), 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    seconds, centiseconds = divmod(remainder, 100)
    return (
        f"{hours:02d}{minutes:02d}{seconds:02d}{centiseconds:02d}",
        f"{hours:02d}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}",
    )


def _clip_clock_centiseconds(clip: dict[str, Any]) -> int:
    try:
        clock_value = clip.get("clock_seconds")
        if clock_value is None:
            clock_value = seconds_from_time_token(str(clip["time_token"]))
        seconds = float(clock_value)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"clip has no usable clock timestamp: {clip.get('clip_id')}") from exc
    return int(round(seconds * 100.0))


def _group_source_slots(group: dict[str, Any]) -> frozenset[tuple[str, int]]:
    clips = _canonical_group(group)["clips"]
    if not clips:
        return frozenset()
    segments = list(clips[0].get("segments") or [clips[0]])
    return frozenset(
        (str(group.get("day")), _clip_clock_centiseconds(segment))
        for segment in segments
    )


def _sliding_six_user_groups(
    manifest: dict[str, Any],
    *,
    duration_seconds: float,
    required_agents: Sequence[str],
) -> list[dict[str, Any]]:
    """Build all complete six-user windows at the native 30-second stride."""

    source_clip_centiseconds = 3_000
    duration_centiseconds = int(round(float(duration_seconds) * 100.0))
    if duration_centiseconds <= 0 or duration_centiseconds % source_clip_centiseconds:
        raise ValueError("sliding duration must be a positive multiple of 30 seconds")
    if len(required_agents) != USER_COUNT or len(set(required_agents)) != USER_COUNT:
        raise ValueError("sliding six-user selection requires six distinct agents")
    segment_count = duration_centiseconds // source_clip_centiseconds
    clips_by_day_agent: dict[str, dict[str, dict[int, dict[str, Any]]]] = {}
    ordered_clips = sorted(
        manifest.get("clips", []),
        key=lambda clip: (
            str(clip.get("day", "")),
            str(clip.get("agent_dir", "")),
            _clip_clock_centiseconds(clip),
            str(clip.get("clip_id", "")),
        ),
    )
    for clip in ordered_clips:
        agent_dir = str(clip.get("agent_dir", ""))
        if agent_dir not in required_agents:
            continue
        clock_centiseconds = _clip_clock_centiseconds(clip)
        if clock_centiseconds % source_clip_centiseconds:
            continue
        clips_by_day_agent.setdefault(str(clip.get("day")), {}).setdefault(
            agent_dir, {}
        ).setdefault(clock_centiseconds, clip)

    groups = []
    for day, clips_by_agent in sorted(clips_by_day_agent.items()):
        if any(agent not in clips_by_agent for agent in required_agents):
            continue
        common_starts = set.intersection(
            *(set(clips_by_agent[agent]) for agent in required_agents)
        )
        for window_start in sorted(common_starts):
            expected_starts = [
                window_start + index * source_clip_centiseconds
                for index in range(segment_count)
            ]
            if any(
                source_start not in clips_by_agent[agent]
                for agent in required_agents
                for source_start in expected_starts
            ):
                continue
            window_clips = []
            for agent in required_agents:
                segments = [
                    clips_by_agent[agent][source_start]
                    for source_start in expected_starts
                ]
                window_clip = dict(segments[0])
                window_clip["segments"] = segments
                window_clip["duration_seconds"] = float(duration_seconds)
                window_clip["segment_count"] = segment_count
                window_clips.append(window_clip)
            time_token, clip_clock = _clock_fields_from_centiseconds(window_start)
            groups.append(
                {
                    "day": day,
                    "time_token": time_token,
                    "clip_clock": clip_clock,
                    "clock_seconds": window_start / 100.0,
                    "duration_seconds": float(duration_seconds),
                    "segment_count": segment_count,
                    "agents": list(required_agents),
                    "clips": window_clips,
                }
            )
    return groups


def _selection_tie_breaker(
    group: dict[str, Any], random_seed: int | None
) -> int:
    payload = (
        f"{random_seed if random_seed is not None else 'canonical'}\0"
        f"{group.get('day')}\0{group.get('time_token')}"
    )
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16)


def _select_supplemental_groups(
    candidates: Sequence[dict[str, Any]],
    *,
    primary_groups: Sequence[dict[str, Any]],
    target_count: int,
    random_seed: int | None,
) -> list[dict[str, Any]]:
    """Greedily maximize new source slots, then minimize pairwise overlap."""

    if target_count <= 0:
        return []
    covered = set().union(*(_group_source_slots(group) for group in primary_groups))
    selected_slot_sets = [_group_source_slots(group) for group in primary_groups]
    day_counts = {
        day: sum(str(group.get("day")) == day for group in primary_groups)
        for day in {str(group.get("day")) for group in candidates}
    }
    remaining = [dict(group) for group in candidates]
    selected = []
    while remaining and len(selected) < target_count:
        def candidate_score(group: dict[str, Any]) -> tuple[int, int, int, int]:
            slots = _group_source_slots(group)
            new_slot_count = len(slots - covered)
            maximum_pair_overlap = max(
                (len(slots & selected_slots) for selected_slots in selected_slot_sets),
                default=0,
            )
            day = str(group.get("day"))
            return (
                new_slot_count,
                -maximum_pair_overlap,
                -day_counts.get(day, 0),
                -_selection_tie_breaker(group, random_seed),
            )

        best = max(remaining, key=candidate_score)
        remaining.remove(best)
        slots = _group_source_slots(best)
        new_slot_count = len(slots - covered)
        maximum_pair_overlap = max(
            (len(slots & selected_slots) for selected_slots in selected_slot_sets),
            default=0,
        )
        selection_rank = len(selected) + 1
        best["selection"] = {
            "tier": "supplemental_minimum_overlap_sliding",
            "selection_rank_within_tier": selection_rank,
            "candidate_stride_seconds": 30.0,
            "source_segment_count": len(slots),
            "new_source_segment_count_at_selection": new_slot_count,
            "overlap_source_segment_count_at_selection": len(slots) - new_slot_count,
            "overlap_percent_at_selection": round(
                100.0 * (len(slots) - new_slot_count) / len(slots), 3
            ),
            "maximum_pair_overlap_segments_at_selection": maximum_pair_overlap,
            "objective": (
                "maximize previously uncovered 30-second source segments, then "
                "minimize overlap with any previously selected window, then balance days"
            ),
        }
        selected.append(best)
        covered.update(slots)
        selected_slot_sets.append(slots)
        day = str(best.get("day"))
        day_counts[day] = day_counts.get(day, 0) + 1
    return selected


def select_source_groups(
    manifest: dict[str, Any],
    *,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    source_window_count: int | None = None,
    start_index: int = 0,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[dict[str, Any]]:
    """Select primary non-overlapping windows, then minimum-overlap fillers."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if source_window_count is not None and source_window_count < 0:
        raise ValueError("source_window_count must be non-negative")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    primary_groups = []
    for raw_group in group_manifest_clips(
        manifest, evidence_duration_seconds=duration_seconds
    ):
        group = _canonical_group(raw_group)
        distinct = {str(clip.get("agent_dir")) for clip in group["clips"]}
        if len(group["clips"]) != USER_COUNT or len(distinct) != USER_COUNT:
            continue
        group["selection"] = {
            "tier": "primary_non_overlapping_wall_clock",
            "wall_clock_aligned": True,
            "non_overlapping_within_tier": True,
        }
        primary_groups.append(group)
    primary_groups.sort(
        key=lambda group: (
            str(group.get("day", "")),
            str(group.get("time_token", "")),
            tuple(str(clip.get("agent_dir", "")) for clip in group["clips"]),
        )
    )
    if random_seed is not None:
        random.Random(int(random_seed)).shuffle(primary_groups)

    requested_end = (
        len(primary_groups)
        if source_window_count is None
        else start_index + source_window_count
    )
    cohort = list(primary_groups)
    if requested_end > len(cohort) and primary_groups:
        agent_sets = {
            tuple(str(clip.get("agent_dir")) for clip in group["clips"])
            for group in primary_groups
        }
        if len(agent_sets) != 1:
            raise ValueError(
                "primary six-user windows do not share one canonical participant set"
            )
        required_agents = next(iter(agent_sets))
        primary_ids = {packet_id_for_group(group) for group in primary_groups}
        sliding_candidates = [
            group
            for group in _sliding_six_user_groups(
                manifest,
                duration_seconds=duration_seconds,
                required_agents=required_agents,
            )
            if packet_id_for_group(group) not in primary_ids
        ]
        cohort.extend(
            _select_supplemental_groups(
                sliding_candidates,
                primary_groups=primary_groups,
                target_count=requested_end - len(primary_groups),
                random_seed=random_seed,
            )
        )
    cohort = cohort[start_index:requested_end]
    return [
        group
        for global_index, group in enumerate(cohort)
        if global_index % shard_count == shard_index
    ]


def packet_id_for_group(group: dict[str, Any]) -> str:
    clips = _canonical_group(group)["clips"]
    return stable_id(
        "RLHF6U",
        group.get("day"),
        group.get("time_token"),
        *(clip.get("agent_id") or clip.get("agent_dir") for clip in clips),
    )


def _source_identity(group: dict[str, Any]) -> dict[str, Any]:
    users = []
    for clip in _canonical_group(group)["clips"]:
        segments = []
        for segment in clip.get("segments") or [clip]:
            segments.append(
                {
                    "clip_id": segment.get("clip_id"),
                    "time_token": segment.get("time_token"),
                    "video_url": segment.get("video_url"),
                }
            )
        users.append(
            {
                "agent_dir": clip.get("agent_dir"),
                "agent_id": clip.get("agent_id"),
                "agent_name": clip.get("agent_name"),
                "segments": segments,
            }
        )
    return {
        "day": group.get("day"),
        "time_token": group.get("time_token"),
        "clip_clock": group.get("clip_clock"),
        "duration_seconds": group.get("duration_seconds"),
        "selection": group.get("selection"),
        "users": users,
    }


def _ensure_dataset_metadata(dataset_root: Path, config: dict[str, Any]) -> str:
    config_fingerprint = fingerprint(config)
    metadata_path = dataset_root / "dataset.json"
    dataset_root.mkdir(parents=True, exist_ok=True)
    if metadata_path.is_file():
        existing = read_json(metadata_path)
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"dataset schema mismatch in {metadata_path}")
        if existing.get("config_fingerprint") != config_fingerprint:
            raise ValueError(
                "dataset preprocessing configuration differs from this run; "
                "use a new dataset root"
            )
        return config_fingerprint
    _atomic_write_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": utc_now(),
            "config_fingerprint": config_fingerprint,
            "configuration": config,
            "storage": {
                "packet_root": "packets",
                "packet_index": "index.jsonl",
                "packet_completion_marker": "COMPLETE",
                "paths_are_relative_to_packet": True,
            },
        },
    )
    return config_fingerprint


def _archive_existing_packet(dataset_root: Path, packet_dir: Path) -> Path:
    archive_root = dataset_root / "stale"
    archive_root.mkdir(parents=True, exist_ok=True)
    label = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = archive_root / f"{packet_dir.name}.{label}.{uuid.uuid4().hex[:8]}"
    packet_dir.replace(destination)
    return destination


def _link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"sampled frame is unavailable: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _persist_packet_frames(
    sampled_users: Sequence[dict[str, Any]], packet_dir: Path
) -> tuple[list[list[dict[str, Any]]], list[dict[str, Any]]]:
    stored_frames: list[list[dict[str, Any]]] = []
    user_rows: list[dict[str, Any]] = []
    for user_index, sampled in enumerate(sampled_users):
        clip = sampled["clip"]
        agent_dir = str(clip.get("agent_dir") or sampled.get("user"))
        directory = f"user_{user_index}_{_safe_component(agent_dir)}"
        frames = []
        for frame_index, frame in enumerate(sampled.get("frames", [])):
            source = Path(str(frame.get("path") or ""))
            timestamp = float(frame.get("timestamp_seconds", 0.0))
            filename = f"frame_{frame_index:04d}_{timestamp:.2f}s.jpg"
            relative = Path("frames") / directory / filename
            destination = packet_dir / relative
            _link_or_copy(source, destination)
            frames.append(
                {
                    "frame_index": frame_index,
                    "timestamp_seconds": round(timestamp, 3),
                    "source_segment_index": int(
                        frame.get("source_segment_index", 0)
                    ),
                    "path": relative.as_posix(),
                }
            )
        stored_frames.append(frames)
        user_rows.append(
            {
                "user_index": user_index,
                "agent_dir": agent_dir,
                "agent_id": clip.get("agent_id"),
                "agent_name": clip.get("agent_name") or sampled.get("user"),
                "frame_directory": (Path("frames") / directory).as_posix(),
                "frame_count": len(frames),
                "frames": frames,
            }
        )
    return stored_frames, user_rows


def _compact_clusters(
    clusters_by_video: Sequence[dict[str, Any]], users: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    compact_users = []
    for user, clusters in zip(users, clusters_by_video):
        representatives = []
        for row in clusters.get("representatives", []):
            representatives.append(
                {
                    "cluster_index": int(row["cluster_index"]),
                    "frame_index": int(row["frame_index"]),
                    "timestamp_seconds": float(row.get("timestamp_seconds", 0.0)),
                    "member_count": int(row.get("member_count", 0)),
                    "temporal_center_seconds": float(
                        row.get("temporal_center_seconds", 0.0)
                    ),
                }
            )
        compact_users.append(
            {
                "user_index": int(user["user_index"]),
                "agent_dir": user["agent_dir"],
                "cluster_count": int(clusters["cluster_count"]),
                "cluster_count_requested": int(clusters["cluster_count_requested"]),
                "global_cluster_count_requested": int(
                    clusters.get(
                        "global_cluster_count_requested",
                        clusters["cluster_count_requested"],
                    )
                ),
                "clustering_scope": clusters.get("clustering_scope"),
                "split_noncontiguous_clusters": bool(
                    clusters.get("split_noncontiguous_clusters", False)
                ),
                "max_cluster_member_gap_seconds": clusters.get(
                    "max_member_gap_seconds"
                ),
                "labels": [int(value) for value in clusters["labels"]],
                "representatives": representatives,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "clustering_scope": "per_user_across_entire_source_window",
        "users": compact_users,
    }


def build_asker_conditioned_keep_masks(
    frame_counts: Sequence[int],
    pruning_results: Sequence[dict[str, Any]],
) -> np.ndarray:
    """Return ``[asker, video, frame]`` masks with unpruned diagonals."""

    if len(frame_counts) != USER_COUNT or len(pruning_results) != USER_COUNT:
        raise ValueError("exactly six frame counts and pruning results are required")
    if any(int(count) <= 0 for count in frame_counts):
        raise ValueError("every user must have at least one sampled frame")
    max_frames = max(int(count) for count in frame_counts)
    masks = np.zeros((USER_COUNT, USER_COUNT, max_frames), dtype=np.bool_)
    for asker_index, result in enumerate(pruning_results):
        if int(result.get("speaker_index", -1)) != asker_index:
            raise ValueError("pruning results must be ordered by asker index")
        for video_index, count in enumerate(frame_counts):
            masks[asker_index, video_index, : int(count)] = True
        videos = result.get("videos", [])
        if len(videos) != USER_COUNT:
            raise ValueError("each pruning result must contain six video rows")
        if videos[asker_index].get("marked_frame_indices"):
            raise ValueError("the pruning implementation attempted to prune the asker")
        for provider_index in range(USER_COUNT):
            if provider_index == asker_index:
                continue
            for frame_index in videos[provider_index].get(
                "marked_frame_indices", []
            ):
                frame_index = int(frame_index)
                if not 0 <= frame_index < int(frame_counts[provider_index]):
                    raise ValueError(
                        f"invalid marked frame {frame_index} for provider {provider_index}"
                    )
                masks[asker_index, provider_index, frame_index] = False
    return masks


def _compact_asker_views(
    users: Sequence[dict[str, Any]],
    pruning_results: Sequence[dict[str, Any]],
    masks: np.ndarray,
) -> dict[str, Any]:
    views = []
    for asker_index, result in enumerate(pruning_results):
        media = []
        for video_index, user in enumerate(users):
            frame_count = int(user["frame_count"])
            retained = int(masks[asker_index, video_index, :frame_count].sum())
            video_result = result["videos"][video_index]
            media.append(
                {
                    "user_index": video_index,
                    "agent_dir": user["agent_dir"],
                    "role": "asker" if video_index == asker_index else "provider",
                    "original_frame_count": frame_count,
                    "retained_frame_count": retained,
                    "removed_frame_count": frame_count - retained,
                    "retained_percent": round(100.0 * retained / frame_count, 3),
                    "marked_cluster_count": len(
                        video_result.get("marked_cluster_indices", [])
                    ),
                    "restored_cluster_count": len(
                        video_result.get("restored_cluster_indices", [])
                    ),
                }
            )
        views.append(
            {
                "asker_index": asker_index,
                "asker_agent_dir": users[asker_index]["agent_dir"],
                "generator_media_mode": GENERATOR_MEDIA_MODE,
                "applied_event_count": int(result.get("applied_event_count", 0)),
                "eligible_pairwise_comparison_count": int(
                    result.get("eligible_pairwise_comparison_count", 0)
                ),
                "computed_cosine_pair_count": int(
                    result.get("computed_cosine_pair_count", 0)
                ),
                "no_redundancy_found": int(result.get("applied_event_count", 0)) == 0,
                "media": media,
            }
        )
    return {"schema_version": SCHEMA_VERSION, "views": views}


def _write_checksum_manifest(packet_dir: Path) -> None:
    rows = []
    for path in sorted(
        (
            candidate
            for candidate in packet_dir.rglob("*")
            if candidate.is_file()
            and candidate.name not in {"checksums.sha256", "COMPLETE"}
        ),
        key=lambda candidate: candidate.relative_to(packet_dir).as_posix(),
    ):
        rows.append(
            f"{file_sha256(path)}  {path.relative_to(packet_dir).as_posix()}\n"
        )
    (packet_dir / "checksums.sha256").write_text("".join(rows), encoding="utf-8")


def _read_checksum_manifest(packet_dir: Path) -> list[tuple[str, str]]:
    rows = []
    for line_number, line in enumerate(
        (packet_dir / "checksums.sha256").read_text(encoding="utf-8").splitlines(),
        1,
    ):
        if not line:
            continue
        try:
            digest, relative = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"invalid checksum line {line_number}") from exc
        rows.append((digest, relative))
    return rows


def validate_preprocessed_packet(
    packet_dir: str | Path,
    *,
    verify_checksums: bool = False,
) -> dict[str, Any]:
    packet_dir = Path(packet_dir)
    required = [
        "packet.json",
        "clip_embeddings.f16.npy",
        "clusters.json",
        "keep_masks.npz",
        "asker_views.json",
        "checksums.sha256",
        "COMPLETE",
    ]
    missing = [name for name in required if not (packet_dir / name).is_file()]
    if missing:
        raise ValueError(f"incomplete packet {packet_dir}: missing {missing}")
    packet = read_json(packet_dir / "packet.json")
    if packet.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported packet schema in {packet_dir}")
    if packet.get("generator_media_mode") != GENERATOR_MEDIA_MODE:
        raise ValueError("packet has the wrong generator media contract")
    users = packet.get("users", [])
    if len(users) != USER_COUNT:
        raise ValueError("packet must contain exactly six users")
    frame_counts = [int(user.get("frame_count", -1)) for user in users]
    if any(count <= 0 for count in frame_counts):
        raise ValueError("packet contains an empty user frame sequence")
    configured_frame_count = int(
        packet["preprocessing"]["sampling"]["expected_frames_per_user"]
    )
    if frame_counts != [configured_frame_count] * USER_COUNT:
        raise ValueError(
            "packet frame counts do not match the configured sampling contract: "
            f"expected={configured_frame_count} actual={frame_counts}"
        )
    for user, frame_count in zip(users, frame_counts):
        frames = user.get("frames", [])
        if len(frames) != frame_count:
            raise ValueError("stored frame count disagrees with packet metadata")
        for expected_index, frame in enumerate(frames):
            if int(frame.get("frame_index", -1)) != expected_index:
                raise ValueError("stored frame indices are not contiguous")
            relative = Path(str(frame.get("path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"packet frame path is not relocatable: {relative}")
            absolute = packet_dir / relative
            if not absolute.is_file() or absolute.stat().st_size <= 0:
                raise ValueError(f"packet frame is missing or empty: {absolute}")
    embeddings = np.load(packet_dir / "clip_embeddings.f16.npy", allow_pickle=False)
    expected_frames = max(frame_counts)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (USER_COUNT, expected_frames):
        raise ValueError(f"unexpected embedding shape: {embeddings.shape}")
    if embeddings.shape[2] <= 0:
        raise ValueError("CLIP embedding dimension must be positive")
    if embeddings.dtype != np.float16:
        raise ValueError(f"unexpected embedding dtype: {embeddings.dtype}")
    with np.load(packet_dir / "keep_masks.npz", allow_pickle=False) as mask_file:
        masks = mask_file["keep_masks"]
        saved_counts = mask_file["frame_counts"]
    if tuple(masks.shape) != (USER_COUNT, USER_COUNT, expected_frames):
        raise ValueError(f"unexpected keep-mask shape: {masks.shape}")
    if masks.dtype != np.bool_:
        raise ValueError(f"unexpected keep-mask dtype: {masks.dtype}")
    if [int(value) for value in saved_counts] != frame_counts:
        raise ValueError("keep-mask frame counts disagree with packet metadata")
    retention_floor = float(
        packet["preprocessing"]["pruning"]["minimum_provider_retention_percent"]
    )
    for asker_index in range(USER_COUNT):
        if not bool(masks[asker_index, asker_index, : frame_counts[asker_index]].all()):
            raise ValueError(f"asker {asker_index} has pruned frames")
        for video_index, frame_count in enumerate(frame_counts):
            if bool(masks[asker_index, video_index, frame_count:].any()):
                raise ValueError("keep mask retains padded frames")
            if video_index != asker_index:
                retained_percent = (
                    100.0
                    * int(masks[asker_index, video_index, :frame_count].sum())
                    / frame_count
                )
                if retained_percent + 1e-9 < retention_floor:
                    raise ValueError(
                        f"provider retention floor violated: {retained_percent:.3f}%"
                    )
    clusters = read_json(packet_dir / "clusters.json")
    if len(clusters.get("users", [])) != USER_COUNT:
        raise ValueError("cluster artifact must contain exactly six users")
    for user_clusters, frame_count in zip(clusters["users"], frame_counts):
        if user_clusters.get("split_noncontiguous_clusters") is not False:
            raise ValueError("noncontiguous-cluster splitting is forbidden")
        if user_clusters.get("max_cluster_member_gap_seconds") is not None:
            raise ValueError("temporal-component gap splitting must remain disabled")
        if len(user_clusters.get("labels", [])) != frame_count:
            raise ValueError("cluster labels do not cover every sampled frame")
        if int(user_clusters.get("global_cluster_count_requested", -1)) != int(
            packet["preprocessing"]["clustering"][
                "global_cluster_count_requested"
            ]
        ):
            raise ValueError("cluster-count density disagrees with packet configuration")
    if verify_checksums:
        for expected_digest, relative in _read_checksum_manifest(packet_dir):
            path = packet_dir / Path(relative)
            if not path.is_file() or file_sha256(path) != expected_digest:
                raise ValueError(f"checksum mismatch: {relative}")
    return {
        "packet_id": packet["packet_id"],
        "packet_dir": str(packet_dir),
        "frame_counts": frame_counts,
        "embedding_shape": list(embeddings.shape),
        "checksums_verified": bool(verify_checksums),
    }


def _resolve_asker_index(users: Sequence[dict[str, Any]], asker: str | int) -> int:
    if isinstance(asker, int) or str(asker).isdigit():
        index = int(asker)
        if 0 <= index < len(users):
            return index
    query = str(asker).casefold()
    matches = [
        int(user["user_index"])
        for user in users
        if query
        in {
            str(user.get("agent_dir", "")).casefold(),
            str(user.get("agent_id", "")).casefold(),
            str(user.get("agent_name", "")).casefold(),
        }
    ]
    if len(matches) != 1:
        raise ValueError(f"asker {asker!r} did not uniquely identify one packet user")
    return matches[0]


def load_asker_view(
    dataset_root: str | Path,
    packet_id: str,
    asker: str | int,
) -> dict[str, Any]:
    """Resolve one generation-ready view without decoding or reclustering media."""

    packet_dir = Path(dataset_root) / "packets" / packet_id
    packet = read_json(packet_dir / "packet.json")
    users = packet["users"]
    asker_index = _resolve_asker_index(users, asker)
    with np.load(packet_dir / "keep_masks.npz", allow_pickle=False) as mask_file:
        masks = mask_file["keep_masks"][asker_index]
    order = [asker_index] + [index for index in range(USER_COUNT) if index != asker_index]
    clips = []
    for user_index in order:
        user = users[user_index]
        role = "asker" if user_index == asker_index else "provider"
        frames = []
        for frame in user["frames"]:
            frame_index = int(frame["frame_index"])
            if role == "provider" and not bool(masks[user_index, frame_index]):
                continue
            frames.append(
                {
                    **frame,
                    "path": str((packet_dir / Path(frame["path"])).resolve()),
                }
            )
        clips.append(
            {
                "user_index": user_index,
                "agent_dir": user["agent_dir"],
                "agent_id": user.get("agent_id"),
                "agent_name": user.get("agent_name"),
                "media_role": role,
                "frame_mode": "full_sampled" if role == "asker" else "asker_pruned",
                "original_frame_count": int(user["frame_count"]),
                "retained_frame_count": len(frames),
                "frames": frames,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet_id,
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "duration_seconds": packet.get("duration_seconds"),
        "asker_index": asker_index,
        "asker_agent_dir": users[asker_index]["agent_dir"],
        "generator_media_mode": GENERATOR_MEDIA_MODE,
        "clips": clips,
    }


def _packet_index_row(packet_dir: Path) -> dict[str, Any]:
    packet = read_json(packet_dir / "packet.json")
    views = read_json(packet_dir / "asker_views.json")["views"]
    provider_counts = [
        int(media["retained_frame_count"])
        for view in views
        for media in view["media"]
        if media["role"] == "provider"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "packet_id": packet["packet_id"],
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "selection_tier": (packet.get("selection") or {}).get("tier"),
        "relative_path": (Path("packets") / packet_dir.name).as_posix(),
        "users": [user["agent_dir"] for user in packet["users"]],
        "frames_per_user": [int(user["frame_count"]) for user in packet["users"]],
        "asker_view_count": USER_COUNT,
        "provider_retained_frame_count_min": min(provider_counts),
        "provider_retained_frame_count_max": max(provider_counts),
        "config_fingerprint": packet["config_fingerprint"],
        "source_fingerprint": packet["source_fingerprint"],
        "status": "complete",
    }


def rebuild_dataset_index(dataset_root: str | Path) -> list[dict[str, Any]]:
    dataset_root = Path(dataset_root)
    rows = []
    for packet_dir in sorted((dataset_root / "packets").glob("*")):
        if packet_dir.is_dir() and (packet_dir / "COMPLETE").is_file():
            rows.append(_packet_index_row(packet_dir))
    rows.sort(key=lambda row: (str(row["day"]), str(row["time_token"]), row["packet_id"]))
    _atomic_write_jsonl(dataset_root / "index.jsonl", rows)
    return rows


def prepare_packet(
    group: dict[str, Any],
    *,
    dataset_root: str | Path,
    cache_dir: str | Path,
    config: dict[str, Any],
    config_fingerprint: str,
    device: str = "cuda",
    ffmpeg_binary: str = "ffmpeg",
    clip_batch_size: int = DEFAULT_CLIP_BATCH_SIZE,
    video_sample_workers: int = DEFAULT_VIDEO_SAMPLE_WORKERS,
    media_prepare_workers: int = DEFAULT_MEDIA_PREPARE_WORKERS,
    overwrite_stale: bool = False,
    encoder: Any | None = None,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    cache_dir = Path(cache_dir)
    packet_id = packet_id_for_group(group)
    source_identity = _source_identity(group)
    source_fingerprint = fingerprint(source_identity)
    packet_dir = dataset_root / "packets" / packet_id
    if packet_dir.exists():
        if (packet_dir / "COMPLETE").is_file() and (packet_dir / "packet.json").is_file():
            existing = read_json(packet_dir / "packet.json")
            if (
                existing.get("config_fingerprint") == config_fingerprint
                and existing.get("source_fingerprint") == source_fingerprint
            ):
                validated = validate_preprocessed_packet(packet_dir)
                return {**validated, "status": "skipped_complete"}
        if not overwrite_stale:
            raise ValueError(
                f"stale or incomplete packet exists at {packet_dir}; "
                "pass --overwrite-stale to archive and rebuild it"
            )
        archived = _archive_existing_packet(dataset_root, packet_dir)
        print(
            f"rlhf_preprocess packet={packet_id} status=archived_stale path={archived}",
            flush=True,
        )

    staging_root = dataset_root / ".staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    stage = staging_root / f"{packet_id}.{uuid.uuid4().hex}"
    stage.mkdir(parents=True)
    started = time.monotonic()
    try:
        print(f"rlhf_preprocess packet={packet_id} status=preparing_media", flush=True)
        evidence_packet = build_evidence_packet(
            _canonical_group(group),
            cache_dir=cache_dir,
            output_root=stage / "transient_evidence",
            users_per_case=USER_COUNT,
            frames_per_clip=0,
            download_media=True,
            download_gaze=False,
            media_prepare_workers=media_prepare_workers,
            assemble_long_video=False,
            defer_gaze_summary=True,
            extract_preview_frames=False,
        )
        sample_interval_seconds = float(config["sampling"]["interval_seconds"])
        sampled_users = group_clip_frames(
            evidence_packet,
            stage / "transient_sampling",
            cache_dir=cache_dir,
            duration_seconds=float(config["duration_seconds"]),
            sample_interval_seconds=sample_interval_seconds,
            start_seconds=0.0,
            ffmpeg_binary=ffmpeg_binary,
            download_media=False,
            video_sample_workers=video_sample_workers,
            direct_segment_rgb=True,
            frame_cache_dir=cache_dir / "offline_rlhf_sampled_frames",
            frame_format="jpeg",
            jpeg_quality=int(config["sampling"]["jpeg_quality"]),
            attach_rgb_images=False,
        )
        expected_frames = int(config["sampling"]["expected_frames_per_user"])
        if len(sampled_users) != USER_COUNT:
            raise ValueError(f"expected six sampled users, got {len(sampled_users)}")
        if any(len(user.get("frames", [])) != expected_frames for user in sampled_users):
            counts = [len(user.get("frames", [])) for user in sampled_users]
            raise ValueError(f"expected {expected_frames} frames per user, got {counts}")

        print(f"rlhf_preprocess packet={packet_id} status=encoding_clip", flush=True)
        encoder = encoder or LazyBatchedImageEncoder(
            str(config["clip"]["model_id"]),
            device=device,
            batch_size=clip_batch_size,
        )
        flat_paths = [
            str(frame["path"])
            for sampled in sampled_users
            for frame in sampled["frames"]
        ]
        encoded = np.asarray(encoder.encode(flat_paths), dtype=np.float32)
        if encoded.ndim != 2 or encoded.shape[0] != USER_COUNT * expected_frames:
            raise ValueError(f"unexpected CLIP embedding shape: {encoded.shape}")
        embedding_dim = int(encoded.shape[1])
        embeddings = encoded.reshape(USER_COUNT, expected_frames, embedding_dim)
        np.save(
            stage / "clip_embeddings.f16.npy",
            embeddings.astype(np.float16),
            allow_pickle=False,
        )
        embeddings_by_video = [embeddings[index].tolist() for index in range(USER_COUNT)]
        frames_by_video = [list(sampled["frames"]) for sampled in sampled_users]

        cluster_config = config["clustering"]
        print(f"rlhf_preprocess packet={packet_id} status=clustering", flush=True)
        clusters_by_video = cluster_six_user_frame_representatives(
            frames_by_video,
            embeddings_by_video,
            start_seconds=0.0,
            duration_seconds=float(config["duration_seconds"]),
            cluster_count=int(cluster_config["clusters_per_density_window"]),
            split_noncontiguous_clusters=False,
            max_cluster_member_gap_seconds=None,
            cluster_window_seconds=float(cluster_config["density_window_seconds"]),
            temporal_time_weight=float(
                cluster_config["temporal_kmeans_time_weight"]
            ),
            temporal_unit_seconds=float(cluster_config["temporal_unit_seconds"]),
        )
        if any(bool(result.get("split_noncontiguous_clusters")) for result in clusters_by_video):
            raise ValueError("offline clustering unexpectedly split temporal components")

        pruning_config = config["pruning"]
        print(f"rlhf_preprocess packet={packet_id} status=building_six_masks", flush=True)
        pruning_results = []
        for asker_index in range(USER_COUNT):
            pruning_results.append(
                clustered_speaker_provider_all_pairs_pruning(
                    frames_by_video,
                    embeddings_by_video,
                    speaker_index=asker_index,
                    start_seconds=0.0,
                    duration_seconds=float(config["duration_seconds"]),
                    sample_interval_seconds=sample_interval_seconds,
                    cluster_count=int(cluster_config["clusters_per_density_window"]),
                    high_similarity_threshold=float(
                        pruning_config["high_similarity_threshold"]
                    ),
                    min_pruned_video_seconds=0.0,
                    pruning_protection_mode="min_percent",
                    min_pruned_video_percent=float(
                        pruning_config["minimum_provider_retention_percent"]
                    ),
                    max_pair_time_difference_seconds=float(
                        pruning_config["max_pair_time_difference_seconds"]
                    ),
                    mutual_nearest_only=False,
                    split_noncontiguous_clusters=False,
                    max_cluster_member_gap_seconds=None,
                    cluster_window_seconds=float(
                        cluster_config["density_window_seconds"]
                    ),
                    precomputed_clusters_by_video=clusters_by_video,
                    temporal_time_weight=float(
                        cluster_config["temporal_kmeans_time_weight"]
                    ),
                    temporal_unit_seconds=float(cluster_config["temporal_unit_seconds"]),
                )
            )
        frame_counts = [len(frames) for frames in frames_by_video]
        masks = build_asker_conditioned_keep_masks(frame_counts, pruning_results)
        np.savez_compressed(
            stage / "keep_masks.npz",
            keep_masks=masks,
            frame_counts=np.asarray(frame_counts, dtype=np.int32),
        )

        stored_frames, users = _persist_packet_frames(sampled_users, stage)
        # The CLIP embeddings were computed from byte-identical cache files.  The
        # packet-owned copies are now the sole paths exposed to downstream code.
        if [len(frames) for frames in stored_frames] != frame_counts:
            raise ValueError("packet frame persistence changed frame counts")
        write_json(stage / "clusters.json", _compact_clusters(clusters_by_video, users))
        write_json(
            stage / "asker_views.json",
            _compact_asker_views(users, pruning_results, masks),
        )
        packet_metadata = {
            "schema_version": SCHEMA_VERSION,
            "packet_id": packet_id,
            "config_fingerprint": config_fingerprint,
            "source_fingerprint": source_fingerprint,
            "day": group.get("day"),
            "time_token": group.get("time_token"),
            "clip_clock": group.get("clip_clock"),
            "duration_seconds": float(config["duration_seconds"]),
            "selection": group.get("selection"),
            "generator_media_mode": GENERATOR_MEDIA_MODE,
            "preprocessing": config,
            "source": source_identity,
            "users": users,
            "artifacts": {
                "clip_embeddings": "clip_embeddings.f16.npy",
                "clusters": "clusters.json",
                "keep_masks": "keep_masks.npz",
                "asker_views": "asker_views.json",
                "checksums": "checksums.sha256",
            },
        }
        write_json(stage / "packet.json", packet_metadata)
        _write_checksum_manifest(stage)
        (stage / "COMPLETE").write_text(
            json.dumps(
                {
                    "schema_version": SCHEMA_VERSION,
                    "packet_id": packet_id,
                    "completed_at_utc": utc_now(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        validate_preprocessed_packet(stage)
        packet_dir.parent.mkdir(parents=True, exist_ok=True)
        stage.replace(packet_dir)
        elapsed = time.monotonic() - started
        print(
            f"rlhf_preprocess packet={packet_id} status=complete seconds={elapsed:.3f}",
            flush=True,
        )
        return {
            "packet_id": packet_id,
            "packet_dir": str(packet_dir),
            "frame_counts": frame_counts,
            "embedding_shape": [USER_COUNT, expected_frames, embedding_dim],
            "status": "complete",
            "seconds": elapsed,
        }
    except BaseException as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "packet_id": packet_id,
            "failed_at_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        try:
            write_json(stage / "FAILED.json", failure)
        except OSError:
            pass
        raise


def prepare_dataset(
    *,
    manifest_path: str | Path,
    dataset_root: str | Path,
    cache_dir: str | Path,
    config: dict[str, Any],
    source_window_count: int | None,
    start_index: int = 0,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    shard_index: int = 0,
    shard_count: int = 1,
    device: str = "cuda",
    ffmpeg_binary: str = "ffmpeg",
    clip_batch_size: int = DEFAULT_CLIP_BATCH_SIZE,
    video_sample_workers: int = DEFAULT_VIDEO_SAMPLE_WORKERS,
    media_prepare_workers: int = DEFAULT_MEDIA_PREPARE_WORKERS,
    overwrite_stale: bool = False,
    continue_on_error: bool = False,
) -> dict[str, Any]:
    dataset_root = Path(dataset_root)
    config_fingerprint = _ensure_dataset_metadata(dataset_root, config)
    manifest = read_json(manifest_path)
    groups = select_source_groups(
        manifest,
        duration_seconds=float(config["duration_seconds"]),
        source_window_count=source_window_count,
        start_index=start_index,
        random_seed=random_seed,
        shard_index=shard_index,
        shard_count=shard_count,
    )
    print(
        "rlhf_preprocess status=cohort_selected "
        f"windows={len(groups)} "
        f"primary={sum((group.get('selection') or {}).get('tier') == 'primary_non_overlapping_wall_clock' for group in groups)} "
        f"supplemental={sum((group.get('selection') or {}).get('tier') == 'supplemental_minimum_overlap_sliding' for group in groups)} "
        f"shard={shard_index}/{shard_count} "
        f"config={config_fingerprint[:12]}",
        flush=True,
    )
    encoder = LazyBatchedImageEncoder(
        str(config["clip"]["model_id"]),
        device=device,
        batch_size=clip_batch_size,
    )
    results = []
    failures = []
    for ordinal, group in enumerate(groups, 1):
        packet_id = packet_id_for_group(group)
        try:
            result = prepare_packet(
                group,
                dataset_root=dataset_root,
                cache_dir=cache_dir,
                config=config,
                config_fingerprint=config_fingerprint,
                device=device,
                ffmpeg_binary=ffmpeg_binary,
                clip_batch_size=clip_batch_size,
                video_sample_workers=video_sample_workers,
                media_prepare_workers=media_prepare_workers,
                overwrite_stale=overwrite_stale,
                encoder=encoder,
            )
            results.append(result)
            print(
                f"rlhf_preprocess progress={ordinal}/{len(groups)} "
                f"packet={packet_id} status={result['status']}",
                flush=True,
            )
        except Exception as exc:
            failure = {
                "packet_id": packet_id,
                "day": group.get("day"),
                "time_token": group.get("time_token"),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            failures.append(failure)
            print(
                f"rlhf_preprocess progress={ordinal}/{len(groups)} "
                f"packet={packet_id} status=failed error={exc}",
                flush=True,
            )
            if not continue_on_error:
                raise
    rows = rebuild_dataset_index(dataset_root)
    if failures:
        _atomic_write_jsonl(
            dataset_root / f"failures.shard_{shard_index:04d}.jsonl", failures
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": config_fingerprint,
        "selected_window_count": len(groups),
        "selected_primary_window_count": sum(
            (group.get("selection") or {}).get("tier")
            == "primary_non_overlapping_wall_clock"
            for group in groups
        ),
        "selected_supplemental_window_count": sum(
            (group.get("selection") or {}).get("tier")
            == "supplemental_minimum_overlap_sliding"
            for group in groups
        ),
        "supplemental_new_source_segment_count": sum(
            int(
                (group.get("selection") or {}).get(
                    "new_source_segment_count_at_selection", 0
                )
            )
            for group in groups
        ),
        "supplemental_overlap_source_segment_count": sum(
            int(
                (group.get("selection") or {}).get(
                    "overlap_source_segment_count_at_selection", 0
                )
            )
            for group in groups
        ),
        "completed_this_run": sum(row["status"] == "complete" for row in results),
        "skipped_complete_this_run": sum(
            row["status"] == "skipped_complete" for row in results
        ),
        "failed_this_run": len(failures),
        "complete_packet_count_in_dataset": len(rows),
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    _atomic_write_json(
        dataset_root / f"run_summary.shard_{shard_index:04d}.json", summary
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="build reusable RLHF evidence packets")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--dataset-root", required=True)
    prepare.add_argument("--cache-dir", required=True)
    prepare.add_argument(
        "--source-window-count",
        type=int,
        default=120,
        help=(
            "Total packet target: use all strict non-overlapping windows first, "
            "then deterministically fill from minimum-overlap 30-second-stride windows"
        ),
    )
    prepare.add_argument("--start-index", type=int, default=0)
    prepare.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    prepare.add_argument("--no-shuffle", action="store_true")
    prepare.add_argument("--shard-index", type=int, default=0)
    prepare.add_argument("--shard-count", type=int, default=1)
    prepare.add_argument("--duration-seconds", type=float, default=DEFAULT_DURATION_SECONDS)
    prepare.add_argument("--sample-fps", type=float, default=DEFAULT_SAMPLE_FPS)
    prepare.add_argument(
        "--clusters-per-30-seconds", type=int, default=DEFAULT_CLUSTERS_PER_30_SECONDS
    )
    prepare.add_argument(
        "--temporal-kmeans-time-weight",
        type=float,
        default=DEFAULT_TEMPORAL_KMEANS_TIME_WEIGHT,
    )
    prepare.add_argument(
        "--temporal-unit-seconds", type=float, default=DEFAULT_TEMPORAL_UNIT_SECONDS
    )
    prepare.add_argument(
        "--max-pair-time-difference-seconds",
        type=float,
        default=DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    )
    prepare.add_argument(
        "--high-similarity-threshold",
        type=float,
        default=DEFAULT_HIGH_SIMILARITY_THRESHOLD,
    )
    prepare.add_argument(
        "--min-provider-retention-percent",
        type=float,
        default=DEFAULT_MIN_PROVIDER_RETENTION_PERCENT,
    )
    prepare.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    prepare.add_argument("--clip-model-id", default=DEFAULT_CLIP_MODEL)
    prepare.add_argument("--clip-batch-size", type=int, default=DEFAULT_CLIP_BATCH_SIZE)
    prepare.add_argument(
        "--video-sample-workers", type=int, default=DEFAULT_VIDEO_SAMPLE_WORKERS
    )
    prepare.add_argument(
        "--media-prepare-workers", type=int, default=DEFAULT_MEDIA_PREPARE_WORKERS
    )
    prepare.add_argument("--device", default="cuda")
    prepare.add_argument("--ffmpeg-binary", default="ffmpeg")
    prepare.add_argument("--overwrite-stale", action="store_true")
    prepare.add_argument("--continue-on-error", action="store_true")

    validate = subparsers.add_parser("validate", help="validate one packet or a dataset")
    validate.add_argument("--dataset-root", required=True)
    validate.add_argument("--packet-id")
    validate.add_argument("--verify-checksums", action="store_true")

    resolve = subparsers.add_parser(
        "resolve-view", help="materialize one asker's generation-time frame list"
    )
    resolve.add_argument("--dataset-root", required=True)
    resolve.add_argument("--packet-id", required=True)
    resolve.add_argument("--asker", required=True)
    resolve.add_argument("--output")

    finalize = subparsers.add_parser(
        "finalize-index", help="rebuild index.jsonl from completed packet directories"
    )
    finalize.add_argument("--dataset-root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        config = build_preprocessing_config(
            duration_seconds=args.duration_seconds,
            sample_fps=args.sample_fps,
            clusters_per_density_window=args.clusters_per_30_seconds,
            temporal_kmeans_time_weight=args.temporal_kmeans_time_weight,
            temporal_unit_seconds=args.temporal_unit_seconds,
            max_pair_time_difference_seconds=args.max_pair_time_difference_seconds,
            high_similarity_threshold=args.high_similarity_threshold,
            min_provider_retention_percent=args.min_provider_retention_percent,
            jpeg_quality=args.jpeg_quality,
            clip_model_id=args.clip_model_id,
        )
        summary = prepare_dataset(
            manifest_path=args.manifest,
            dataset_root=args.dataset_root,
            cache_dir=args.cache_dir,
            config=config,
            source_window_count=args.source_window_count,
            start_index=args.start_index,
            random_seed=None if args.no_shuffle else args.random_seed,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            device=args.device,
            ffmpeg_binary=args.ffmpeg_binary,
            clip_batch_size=args.clip_batch_size,
            video_sample_workers=args.video_sample_workers,
            media_prepare_workers=args.media_prepare_workers,
            overwrite_stale=args.overwrite_stale,
            continue_on_error=args.continue_on_error,
        )
        print(json.dumps(summary, indent=2))
        return 1 if summary["failed_this_run"] else 0
    if args.command == "validate":
        root = Path(args.dataset_root)
        if args.packet_id:
            packet_dirs = [root / "packets" / args.packet_id]
        else:
            packet_dirs = sorted(
                packet_dir
                for packet_dir in (root / "packets").glob("*")
                if packet_dir.is_dir() and (packet_dir / "COMPLETE").is_file()
            )
        results = [
            validate_preprocessed_packet(
                packet_dir, verify_checksums=args.verify_checksums
            )
            for packet_dir in packet_dirs
        ]
        print(json.dumps({"validated_packet_count": len(results), "packets": results}, indent=2))
        return 0
    if args.command == "resolve-view":
        view = load_asker_view(args.dataset_root, args.packet_id, args.asker)
        if args.output:
            write_json(args.output, view)
        else:
            print(json.dumps(view, ensure_ascii=False, indent=2))
        return 0
    if args.command == "finalize-index":
        rows = rebuild_dataset_index(args.dataset_root)
        print(json.dumps({"complete_packet_count": len(rows)}, indent=2))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
