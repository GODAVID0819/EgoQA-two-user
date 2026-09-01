"""Sidecar random-pair CLIP-pruned sampler for synchronized EgoLife clips.

This is intentionally separate from the main evidence pipeline. It starts from
the full manifest, randomly selects a synchronized two-video pair by default,
and emits candidate packets with paired original/pruned videos. The selected
videos are sampled at one frame per second, embedded with CLIP, clustered within
each video, compared through cluster medoids, and high-similarity clusters are
removed as temporal intervals. Comparing all videos in a synchronized group is
available only as an explicit slow opt-in.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from .clip_gap_demo import (
        DEFAULT_CLIP_MODEL,
        ImageEncoder,
        TransformersClipEncoder,
        cluster_embedding_medoids,
        cosine_similarity,
    )
    from .clip_gap_demo import sample_short_video
    from .evidence import (
        concatenate_video_segments,
        group_manifest_clips,
        local_cache_path,
        window_cache_path,
    )
    from .io_utils import download_file, read_json, stable_id, write_json, write_jsonl
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from egolife_two_user_qa.clip_gap_demo import (
        DEFAULT_CLIP_MODEL,
        ImageEncoder,
        TransformersClipEncoder,
        cluster_embedding_medoids,
    )
    from egolife_two_user_qa.clip_gap_demo import cosine_similarity, sample_short_video
    from egolife_two_user_qa.evidence import (
        concatenate_video_segments,
        group_manifest_clips,
        local_cache_path,
        window_cache_path,
    )
    from egolife_two_user_qa.io_utils import download_file, read_json, stable_id, write_json, write_jsonl


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [value / norm for value in vector]


def _safe_filename_part(value: Any) -> str:
    text = str(value or "unknown").strip()
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)
    return safe.strip("_") or "unknown"


def mean_embedding(embeddings: list[list[float]]) -> list[float]:
    """Average frame embeddings into one normalized clip-level embedding."""

    if not embeddings:
        raise ValueError("cannot average an empty embedding list")
    width = len(embeddings[0])
    if any(len(vector) != width for vector in embeddings):
        raise ValueError("embedding dimensions must match")
    averaged = [
        sum(vector[index] for vector in embeddings) / len(embeddings)
        for index in range(width)
    ]
    return _normalize(averaged)


def group_similarity_matrix(clip_embeddings: list[list[float]]) -> list[list[float]]:
    """Return rounded pairwise cosine similarities for all clips in a group."""

    return [
        [
            round(1.0 if left_index == right_index else cosine_similarity(left, right), 6)
            for right_index, right in enumerate(clip_embeddings)
        ]
        for left_index, left in enumerate(clip_embeddings)
    ]


def relative_group_scores(
    clip_rows: list[dict[str, Any]],
    clip_embeddings: list[list[float]],
) -> dict[str, Any]:
    """Score each clip by how typical it is relative to the synchronized group."""

    if len(clip_rows) != len(clip_embeddings):
        raise ValueError("clip row and embedding counts must match")
    if len(clip_rows) < 2:
        raise ValueError("at least two clips are required")

    matrix = group_similarity_matrix(clip_embeddings)
    centroid = mean_embedding(clip_embeddings)
    scored = []
    for index, row in enumerate(clip_rows):
        other_similarities = [
            matrix[index][other_index]
            for other_index in range(len(clip_rows))
            if other_index != index
        ]
        mean_to_others = sum(other_similarities) / len(other_similarities)
        clip = row["clip"]
        scored.append(
            {
                "index": index,
                "agent_dir": clip.get("agent_dir"),
                "agent_id": clip.get("agent_id"),
                "agent_name": clip.get("agent_name") or row.get("user"),
                "mean_similarity_to_group": round(mean_to_others, 6),
                "min_similarity_to_group": round(min(other_similarities), 6),
                "max_similarity_to_group": round(max(other_similarities), 6),
                "centroid_similarity": round(cosine_similarity(clip_embeddings[index], centroid), 6),
                "frames": row.get("frames", []),
            }
        )

    ranked = sorted(scored, key=lambda item: (item["mean_similarity_to_group"], str(item["agent_dir"])))
    n = len(ranked)
    for rank, item in enumerate(ranked, 1):
        item["relative_rank"] = rank
        item["typicality_percentile"] = round((rank - 1) / (n - 1), 6) if n > 1 else 0.5
        item["typicality_middle_score"] = round(
            1.0 - abs(float(item["typicality_percentile"]) - 0.5) * 2.0,
            6,
        )
    return {
        "similarity_matrix": matrix,
        "clip_scores": sorted(scored, key=lambda item: int(item["index"])),
        "ranked_by_group_similarity": ranked,
    }


def frame_similarity_matrix(
    left_embeddings: list[list[float]],
    right_embeddings: list[list[float]],
) -> list[list[float]]:
    """Return pairwise CLIP cosine similarities for two embedding lists."""

    if not left_embeddings or not right_embeddings:
        raise ValueError("both videos need at least one frame embedding")
    return [
        [round(cosine_similarity(left, right), 6) for right in right_embeddings]
        for left in left_embeddings
    ]


def _flatten_matrix(matrix: list[list[float]]) -> list[float]:
    return [float(value) for row in matrix for value in row]


def _topk_mean(values: list[float], k: int) -> float:
    if not values:
        raise ValueError("cannot compute top-k mean for an empty list")
    k = max(1, min(k, len(values)))
    return sum(sorted(values, reverse=True)[:k]) / k


def _bounded_frame_indices(
    decisions: list[dict[str, Any]],
    *,
    max_frames: int | None,
) -> list[int]:
    kept = [item for item in decisions if item["status"] == "kept"]
    if max_frames is not None and max_frames > 0:
        kept = sorted(
            kept,
            key=lambda item: (
                -float(item.get("best_match_similarity", 0.0)),
                int(item.get("frame_index", 0)),
            ),
        )[:max_frames]
    return sorted(int(item["frame_index"]) for item in kept)


def relative_frame_pruning(
    matrix: list[list[float]],
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    *,
    min_frame_sim: float,
    max_frame_sim: float,
    min_frames_per_clip: int = 1,
    max_frames_per_clip: int | None = None,
) -> dict[str, Any]:
    """Keep frames with cross-video similarity in a useful middle band.

    Frames whose closest cross-video match is above max_frame_sim are treated as
    near-duplicates and removed before generation. Frames below min_frame_sim are
    too unrelated to anchor a cross-video question. The remaining frames are
    similar enough to share context without inviting questions about duplicate
    views.
    """

    if min_frame_sim > max_frame_sim:
        raise ValueError("min_frame_sim must be <= max_frame_sim")
    if len(matrix) != len(left_frames):
        raise ValueError("left frame count must match frame similarity matrix rows")
    if any(len(row) != len(right_frames) for row in matrix):
        raise ValueError("right frame count must match frame similarity matrix columns")
    if not left_frames or not right_frames:
        raise ValueError("both selected clips need sampled frames before pruning")

    def decide(value: float) -> str:
        if value > max_frame_sim:
            return "dropped_too_close"
        if value < min_frame_sim:
            return "dropped_too_dissimilar"
        return "kept"

    left_decisions = []
    for left_index, row in enumerate(matrix):
        best_right_index, best = max(enumerate(row), key=lambda item: item[1])
        left_decisions.append(
            {
                "frame_index": left_index,
                "timestamp_seconds": left_frames[left_index].get("timestamp_seconds"),
                "best_match_index": int(best_right_index),
                "best_match_timestamp_seconds": right_frames[best_right_index].get("timestamp_seconds"),
                "best_match_similarity": round(float(best), 6),
                "status": decide(float(best)),
            }
        )

    right_decisions = []
    for right_index, _frame in enumerate(right_frames):
        candidates = [(left_index, matrix[left_index][right_index]) for left_index in range(len(left_frames))]
        best_left_index, best = max(candidates, key=lambda item: item[1])
        right_decisions.append(
            {
                "frame_index": right_index,
                "timestamp_seconds": right_frames[right_index].get("timestamp_seconds"),
                "best_match_index": int(best_left_index),
                "best_match_timestamp_seconds": left_frames[best_left_index].get("timestamp_seconds"),
                "best_match_similarity": round(float(best), 6),
                "status": decide(float(best)),
            }
        )

    left_kept_indices = _bounded_frame_indices(left_decisions, max_frames=max_frames_per_clip)
    right_kept_indices = _bounded_frame_indices(right_decisions, max_frames=max_frames_per_clip)
    left_status_counts = {
        status: sum(1 for item in left_decisions if item["status"] == status)
        for status in ("kept", "dropped_too_close", "dropped_too_dissimilar")
    }
    right_status_counts = {
        status: sum(1 for item in right_decisions if item["status"] == status)
        for status in ("kept", "dropped_too_close", "dropped_too_dissimilar")
    }
    passed = len(left_kept_indices) >= min_frames_per_clip and len(right_kept_indices) >= min_frames_per_clip
    return {
        "method": "bandpass_best_cross_video_frame_similarity",
        "min_frame_sim": min_frame_sim,
        "max_frame_sim": max_frame_sim,
        "min_frames_per_clip": min_frames_per_clip,
        "max_frames_per_clip": max_frames_per_clip,
        "left_kept_indices": left_kept_indices,
        "right_kept_indices": right_kept_indices,
        "left_kept_count": len(left_kept_indices),
        "right_kept_count": len(right_kept_indices),
        "left_status_counts": left_status_counts,
        "right_status_counts": right_status_counts,
        "dropped_too_close_frame_count": (
            left_status_counts["dropped_too_close"] + right_status_counts["dropped_too_close"]
        ),
        "dropped_too_dissimilar_frame_count": (
            left_status_counts["dropped_too_dissimilar"] + right_status_counts["dropped_too_dissimilar"]
        ),
        "passed": passed,
        "left_frame_decisions": left_decisions,
        "right_frame_decisions": right_decisions,
    }


def _merge_intervals(intervals: list[tuple[float, float]], *, gap_tolerance: float = 1e-6) -> list[tuple[float, float]]:
    cleaned = sorted((float(start), float(end)) for start, end in intervals if end > start)
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end + gap_tolerance:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def _subtract_intervals(
    base: tuple[float, float],
    remove: list[tuple[float, float]],
    *,
    min_interval_seconds: float = 0.2,
) -> list[tuple[float, float]]:
    start, end = base
    keep = []
    cursor = float(start)
    for remove_start, remove_end in _merge_intervals(remove):
        remove_start = max(float(start), remove_start)
        remove_end = min(float(end), remove_end)
        if remove_end <= cursor:
            continue
        if remove_start - cursor >= min_interval_seconds:
            keep.append((cursor, remove_start))
        cursor = max(cursor, remove_end)
    if end - cursor >= min_interval_seconds:
        keep.append((cursor, float(end)))
    return [(round(left, 3), round(right, 3)) for left, right in keep if right - left >= min_interval_seconds]


def _sampled_frame_interval(
    frame: dict[str, Any],
    *,
    window_start: float,
    window_end: float,
    sample_interval_seconds: float,
) -> tuple[float, float] | None:
    timestamp = float(frame.get("timestamp_seconds", window_start))
    half_width = float(sample_interval_seconds) / 2.0
    start = max(float(window_start), timestamp - half_width)
    end = min(float(window_end), timestamp + half_width)
    if end <= start:
        return None
    return (start, end)


def _intervals_for_frame_indices(
    frames: list[dict[str, Any]],
    frame_indices: set[int],
    *,
    window_start: float,
    window_end: float,
    sample_interval_seconds: float,
) -> list[tuple[float, float]]:
    intervals = []
    for frame_index in sorted(frame_indices):
        if frame_index < 0 or frame_index >= len(frames):
            continue
        interval = _sampled_frame_interval(
            frames[frame_index],
            window_start=window_start,
            window_end=window_end,
            sample_interval_seconds=sample_interval_seconds,
        )
        if interval is not None:
            intervals.append(interval)
    return _merge_intervals(intervals)


def _side_best_frame_matches(
    matrix: list[list[float]],
    *,
    side: str,
    left_frames: list[dict[str, Any]] | None = None,
    right_frames: list[dict[str, Any]] | None = None,
    max_pair_time_difference_seconds: float | None = None,
) -> dict[int, dict[str, Any]]:
    """Return each sampled frame's best cross-video match from a similarity matrix."""

    if max_pair_time_difference_seconds is not None:
        if max_pair_time_difference_seconds < 0:
            raise ValueError("max_pair_time_difference_seconds must be non-negative")
        if left_frames is None or right_frames is None:
            raise ValueError("frame timestamps are required for time-gated matching")

    def eligible(left_index: int, right_index: int) -> bool:
        if max_pair_time_difference_seconds is None:
            return True
        left_timestamp = float(left_frames[left_index].get("timestamp_seconds", 0.0))
        right_timestamp = float(right_frames[right_index].get("timestamp_seconds", 0.0))
        return abs(left_timestamp - right_timestamp) <= max_pair_time_difference_seconds + 1e-9

    if side == "left":
        matches = {}
        for left_index, row in enumerate(matrix):
            choices = [
                (right_index, similarity)
                for right_index, similarity in enumerate(row)
                if eligible(left_index, right_index)
            ]
            if not choices:
                continue
            right_index, similarity = max(choices, key=lambda item: item[1])
            matches[left_index] = {
                "best_match_index": int(right_index),
                "best_match_similarity": float(similarity),
            }
        return matches
    if side == "right":
        if not matrix:
            return {}
        width = len(matrix[0])
        matches = {}
        for right_index in range(width):
            choices = [
                (left_index, matrix[left_index][right_index])
                for left_index in range(len(matrix))
                if eligible(left_index, right_index)
            ]
            if not choices:
                continue
            left_index, similarity = max(choices, key=lambda item: item[1])
            matches[right_index] = {
                "best_match_index": int(left_index),
                "best_match_similarity": float(similarity),
            }
        return matches
    raise ValueError(f"unknown side: {side}")


def _filter_preserved_intervals(
    intervals: list[tuple[float, float]],
    preserved_intervals: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not preserved_intervals:
        return intervals
    return [
        interval
        for interval in intervals
        if not any(interval[0] < preserved[1] and interval[1] > preserved[0] for preserved in preserved_intervals)
    ]


def _apply_pruning_duration_protection(
    frames: list[dict[str, Any]],
    marked_indices: set[int],
    best_matches: dict[int, dict[str, Any]],
    *,
    side: str,
    window_start: float,
    window_end: float,
    sample_interval_seconds: float,
    high_similarity_threshold: float,
    target_kept_seconds: float | None,
    preserved_intervals: list[tuple[float, float]],
) -> dict[str, Any]:
    """Unmark least-similar high-threshold frames until the kept duration floor is met."""

    current_marked = set(marked_indices)

    def compute(marked: set[int]) -> tuple[list[tuple[float, float]], list[tuple[float, float]], float, float]:
        remove = _intervals_for_frame_indices(
            frames,
            marked,
            window_start=window_start,
            window_end=window_end,
            sample_interval_seconds=sample_interval_seconds,
        )
        remove = _merge_intervals(_filter_preserved_intervals(remove, preserved_intervals))
        keep = _subtract_intervals((window_start, window_end), remove)
        kept = round(sum(end - start for start, end in keep), 3)
        removed = round(sum(end - start for start, end in remove), 3)
        return remove, keep, kept, removed

    remove_intervals, keep_intervals, kept_duration, removed_duration = compute(current_marked)
    restored = []
    target = None if target_kept_seconds is None else max(0.0, round(float(target_kept_seconds), 3))
    if target is not None and kept_duration < target:
        candidates = []
        for frame_index in sorted(current_marked):
            match = best_matches.get(frame_index)
            if not match:
                continue
            similarity = float(match["best_match_similarity"])
            if similarity < high_similarity_threshold:
                continue
            frame = frames[frame_index]
            candidates.append(
                {
                    "side": side,
                    "frame_index": int(frame_index),
                    "timestamp_seconds": frame.get("timestamp_seconds"),
                    "best_match_index": int(match["best_match_index"]),
                    "best_match_similarity": round(similarity, 6),
                }
            )
        candidates.sort(
            key=lambda row: (
                float(row["best_match_similarity"]),
                float(row["timestamp_seconds"] if row["timestamp_seconds"] is not None else window_start),
                int(row["frame_index"]),
            )
        )
        for candidate in candidates:
            if kept_duration >= target:
                break
            frame_index = int(candidate["frame_index"])
            if frame_index not in current_marked:
                continue
            before = kept_duration
            current_marked.remove(frame_index)
            remove_intervals, keep_intervals, kept_duration, removed_duration = compute(current_marked)
            restored.append({**candidate, "kept_duration_before_seconds": before, "kept_duration_after_seconds": kept_duration})

    return {
        "marked_indices": current_marked,
        "remove_intervals": remove_intervals,
        "keep_intervals": keep_intervals,
        "kept_duration_seconds": kept_duration,
        "removed_duration_seconds": removed_duration,
        "restored_frames": restored,
        "target_kept_seconds": target,
        "target_met": True if target is None else kept_duration >= target,
    }


def _protected_duration_target_seconds(
    *,
    mode: str,
    duration_seconds: float,
    min_pruned_video_seconds: float,
    min_pruned_video_percent: float | None,
) -> float | None:
    if mode == "reject":
        return None
    if mode == "min_seconds":
        if min_pruned_video_seconds < 0:
            raise ValueError("min_pruned_video_seconds must be non-negative")
        return min(float(duration_seconds), float(min_pruned_video_seconds))
    if mode == "min_percent":
        if min_pruned_video_percent is None:
            raise ValueError("min_pruned_video_percent is required when pruning_protection_mode is min_percent")
        if min_pruned_video_percent < 0 or min_pruned_video_percent > 100:
            raise ValueError("min_pruned_video_percent must be between 0 and 100")
        return min(float(duration_seconds), float(duration_seconds) * float(min_pruned_video_percent) / 100.0)
    raise ValueError(f"unknown pruning_protection_mode: {mode}")


def clustered_frame_representatives(
    frames: list[dict[str, Any]],
    embeddings: list[list[float]],
    *,
    cluster_count: int,
    split_noncontiguous_clusters: bool = False,
    max_member_gap_seconds: float | None = None,
) -> dict[str, Any]:
    """Cluster one video's sampled frame embeddings and expose medoid frames."""

    if len(frames) != len(embeddings):
        raise ValueError("frame and embedding counts must match")
    if not frames:
        raise ValueError("cannot cluster an empty frame list")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if split_noncontiguous_clusters and (
        max_member_gap_seconds is None or max_member_gap_seconds <= 0
    ):
        raise ValueError("a positive max_member_gap_seconds is required when splitting clusters")

    labels, medoids = cluster_embedding_medoids(embeddings, cluster_count)
    representatives = []
    representative_embeddings = []
    output_labels = [-1 for _ in frames]
    for visual_cluster_index, frame_index in enumerate(medoids):
        visual_member_indices = [
            index
            for index, label in enumerate(labels)
            if int(label) == int(visual_cluster_index)
        ]
        components = [visual_member_indices]
        if split_noncontiguous_clusters:
            ordered = sorted(
                visual_member_indices,
                key=lambda index: (float(frames[index].get("timestamp_seconds", 0.0)), index),
            )
            components = []
            for member_index in ordered:
                if not components:
                    components.append([member_index])
                    continue
                previous_index = components[-1][-1]
                previous_timestamp = float(frames[previous_index].get("timestamp_seconds", 0.0))
                timestamp = float(frames[member_index].get("timestamp_seconds", 0.0))
                if timestamp - previous_timestamp > float(max_member_gap_seconds) + 1e-9:
                    components.append([member_index])
                else:
                    components[-1].append(member_index)

        for component_index, member_indices in enumerate(components):
            if split_noncontiguous_clusters:
                component_embeddings = [embeddings[index] for index in member_indices]
                _, component_medoids = cluster_embedding_medoids(component_embeddings, 1)
                frame_index = member_indices[component_medoids[0]]
            cluster_index = len(representatives)
            for member_index in member_indices:
                output_labels[member_index] = cluster_index
            frame = frames[frame_index]
            representatives.append(
                {
                    "cluster_index": int(cluster_index),
                    "visual_cluster_index": int(visual_cluster_index),
                    "temporal_component_index": int(component_index),
                    "frame_index": int(frame_index),
                    "timestamp_seconds": frame.get("timestamp_seconds"),
                    "path": frame.get("path"),
                    "member_indices": member_indices,
                    "member_timestamps": [
                        frames[index].get("timestamp_seconds")
                        for index in member_indices
                    ],
                    "member_count": len(member_indices),
                }
            )
            representative_embeddings.append(embeddings[frame_index])

    return {
        "cluster_count_requested": cluster_count,
        "cluster_count": len(representatives),
        "visual_cluster_count": len(medoids),
        "split_noncontiguous_clusters": split_noncontiguous_clusters,
        "max_member_gap_seconds": max_member_gap_seconds,
        "labels": output_labels,
        "representatives": representatives,
        "representative_embeddings": representative_embeddings,
    }




def clustered_speaker_provider_all_pairs_pruning(
    frames_by_video: list[list[dict[str, Any]]],
    embeddings_by_video: list[list[list[float]]],
    *,
    speaker_index: int,
    start_seconds: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    cluster_count: int = 12,
    high_similarity_threshold: float = 0.82,
    min_pruned_video_seconds: float = 8.0,
) -> dict[str, Any]:
    """比较全部 speaker-provider cluster 对，仅裁剪过阈值的 provider cluster。"""

    if len(frames_by_video) != 6 or len(embeddings_by_video) != 6:
        raise ValueError("speaker provider pruning requires exactly 6 videos")
    if speaker_index < 0 or speaker_index >= 6:
        raise ValueError(f"speaker_index must be between 0 and 5, got {speaker_index}")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    clusters_by_video = [
        clustered_frame_representatives(frames, embeddings, cluster_count=cluster_count)
        for frames, embeddings in zip(frames_by_video, embeddings_by_video)
    ]
    provider_indices = [index for index in range(6) if index != speaker_index]
    speaker_clusters = clusters_by_video[speaker_index]
    matrices = {
        provider_index: frame_similarity_matrix(
            speaker_clusters["representative_embeddings"],
            clusters_by_video[provider_index]["representative_embeddings"],
        )
        for provider_index in provider_indices
    }

    events: list[dict[str, Any]] = []
    marked_clusters = [set() for _ in range(6)]
    trigger_events = [set() for _ in range(6)]
    pairwise_comparison_count = 0
    for provider_index in provider_indices:
        matrix = matrices[provider_index]
        pairwise_comparison_count += sum(len(row) for row in matrix)
        provider_cluster_count = int(clusters_by_video[provider_index]["cluster_count"])
        for provider_cluster_index in range(provider_cluster_count):
            speaker_matches = [
                {
                    "speaker_cluster_index": speaker_cluster_index,
                    "similarity": round(float(row[provider_cluster_index]), 6),
                }
                for speaker_cluster_index, row in enumerate(matrix)
                if float(row[provider_cluster_index]) >= float(high_similarity_threshold)
            ]
            if not speaker_matches:
                continue
            event_index = len(events)
            events.append(
                {
                    "event_index": event_index,
                    "provider_index": provider_index,
                    "provider_cluster_index": provider_cluster_index,
                    "speaker_matches": speaker_matches,
                    "max_similarity": max(
                        float(match["similarity"]) for match in speaker_matches
                    ),
                    "deleted_clusters": [
                        {
                            "video_index": provider_index,
                            "cluster_index": provider_cluster_index,
                        }
                    ],
                }
            )
            marked_clusters[provider_index].add(provider_cluster_index)
            trigger_events[provider_index].add(event_index)

    window_start = float(start_seconds)
    window_end = round(window_start + float(duration_seconds), 3)
    video_results = []
    for video_index, (frames, clusters) in enumerate(zip(frames_by_video, clusters_by_video)):
        marked_frame_indices: set[int] = set()
        for marked_cluster_index in marked_clusters[video_index]:
            marked_frame_indices.update(
                int(index)
                for index in clusters["representatives"][marked_cluster_index].get(
                    "member_indices", []
                )
            )
        remove_intervals = _intervals_for_frame_indices(
            frames,
            marked_frame_indices,
            window_start=window_start,
            window_end=window_end,
            sample_interval_seconds=sample_interval_seconds,
        )
        keep_intervals = _subtract_intervals((window_start, window_end), remove_intervals)
        kept_duration = round(sum(end - start for start, end in keep_intervals), 3)
        removed_duration = round(sum(end - start for start, end in remove_intervals), 3)
        video_results.append(
            {
                "video_index": video_index,
                "cluster_count": int(clusters["cluster_count"]),
                "clusters": clusters["representatives"],
                "marked_cluster_indices": sorted(marked_clusters[video_index]),
                "marked_frame_indices": sorted(marked_frame_indices),
                "trigger_event_indices": sorted(trigger_events[video_index]),
                "remove_intervals": remove_intervals,
                "keep_intervals": keep_intervals,
                "kept_duration_seconds": kept_duration,
                "removed_duration_seconds": removed_duration,
                "passed": kept_duration >= float(min_pruned_video_seconds),
            }
        )

    return {
        "method": "speaker_provider_all_pairs_provider_only",
        "speaker_index": speaker_index,
        "provider_indices": provider_indices,
        "cluster_count": cluster_count,
        "high_similarity_threshold": high_similarity_threshold,
        "pairwise_comparison_count": pairwise_comparison_count,
        "events": events,
        "videos": video_results,
        "passed": bool(events) and all(video["passed"] for video in video_results),
    }


def clustered_six_user_zip_temporal_pruning(
    frames_by_video: list[list[dict[str, Any]]],
    embeddings_by_video: list[list[list[float]]],
    *,
    speaker_index: int,
    start_seconds: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    seconds_per_cluster: float = 2.5,
    time_weight: float = 0.1,
    temporal_unit_seconds: float = 30.0,
    max_iterations: int = 25,
    high_similarity_threshold: float = 0.82,
    cross_gap_mode: str = "center",
    max_cross_gap_seconds: float = 10.0,
    min_pruned_video_seconds: float = 8.0,
    min_pruned_video_percent: float = 20.0,
) -> dict[str, Any]:
    """按 ZIP sidecar 对五个 speaker-provider pair 执行时间感知双侧剪枝。"""

    if __package__:
        from .temporal_kmeans_grid_sidecar import (
            prune_time_aware_cluster_pair,
            time_aware_clustered_frame_representatives,
        )
    else:
        from egolife_two_user_qa.temporal_kmeans_grid_sidecar import (
            prune_time_aware_cluster_pair,
            time_aware_clustered_frame_representatives,
        )

    if len(frames_by_video) != 6 or len(embeddings_by_video) != 6:
        raise ValueError("ZIP temporal pruning requires exactly 6 videos")
    if speaker_index not in range(6):
        raise ValueError("speaker_index must be between 0 and 5")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if seconds_per_cluster <= 0:
        raise ValueError("seconds_per_cluster must be positive")

    cluster_count = max(1, math.ceil(duration_seconds / seconds_per_cluster))
    clusters_by_video = [
        time_aware_clustered_frame_representatives(
            frames,
            embeddings,
            cluster_count=cluster_count,
            time_weight=time_weight,
            temporal_unit_seconds=temporal_unit_seconds,
            max_iterations=max_iterations,
        )
        for frames, embeddings in zip(frames_by_video, embeddings_by_video)
    ]
    provider_indices = [index for index in range(6) if index != speaker_index]
    pair_results: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    videos: list[dict[str, Any] | None] = [None] * 6
    window_start = float(start_seconds)
    window_end = round(window_start + float(duration_seconds), 3)
    videos[speaker_index] = {
        "video_index": speaker_index,
        "cluster_count": int(clusters_by_video[speaker_index]["cluster_count"]),
        "clusters": clusters_by_video[speaker_index]["representatives"],
        "marked_cluster_indices": [],
        "marked_frame_indices": [],
        "trigger_event_indices": [],
        "remove_intervals": [],
        "keep_intervals": [[window_start, window_end]],
        "kept_duration_seconds": float(duration_seconds),
        "removed_duration_seconds": 0.0,
        "passed": True,
        "qa_media_uses_full_original": True,
    }

    for pair_index, provider_index in enumerate(provider_indices):
        full_matrix = frame_similarity_matrix(
            embeddings_by_video[speaker_index],
            embeddings_by_video[provider_index],
        )
        pruning = prune_time_aware_cluster_pair(
            frames_by_video[speaker_index],
            frames_by_video[provider_index],
            embeddings_by_video[speaker_index],
            embeddings_by_video[provider_index],
            clusters_by_video[speaker_index],
            clusters_by_video[provider_index],
            full_frame_matrix=full_matrix,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            high_similarity_threshold=high_similarity_threshold,
            min_pruned_video_seconds=min_pruned_video_seconds,
            pruning_protection_mode="min_percent",
            min_pruned_video_percent=min_pruned_video_percent,
            cross_gap_mode=cross_gap_mode,
            max_cross_gap_seconds=max_cross_gap_seconds,
        )
        provider_event_indices: list[int] = []
        for high_pair in pruning["high_similarity_representative_pairs"]:
            event_index = len(events)
            provider_event_indices.append(event_index)
            events.append(
                {
                    "event_index": event_index,
                    "pair_index": pair_index,
                    "speaker_index": speaker_index,
                    "provider_index": provider_index,
                    "speaker_cluster_index": int(high_pair["left_cluster_index"]),
                    "provider_cluster_index": int(high_pair["right_cluster_index"]),
                    "similarity": float(high_pair["similarity"]),
                    "deleted_clusters": [
                        {
                            "video_index": speaker_index,
                            "cluster_index": int(high_pair["left_cluster_index"]),
                        },
                        {
                            "video_index": provider_index,
                            "cluster_index": int(high_pair["right_cluster_index"]),
                        },
                    ],
                }
            )
        pair_results.append(
            {
                "pair_index": pair_index,
                "speaker_index": speaker_index,
                "provider_index": provider_index,
                "pruning": pruning,
            }
        )
        videos[provider_index] = {
            "video_index": provider_index,
            "cluster_count": int(clusters_by_video[provider_index]["cluster_count"]),
            "clusters": clusters_by_video[provider_index]["representatives"],
            "marked_cluster_indices": sorted(
                {
                    int(row["right_cluster_index"])
                    for row in pruning["high_similarity_representative_pairs"]
                }
            ),
            "marked_frame_indices": list(pruning["right_marked_frame_indices"]),
            "trigger_event_indices": provider_event_indices,
            "remove_intervals": list(pruning["right_remove_intervals"]),
            "keep_intervals": list(pruning["right_keep_intervals"]),
            "kept_duration_seconds": pruning["right_kept_duration_seconds"],
            "removed_duration_seconds": pruning["right_removed_duration_seconds"],
            "passed": bool(pruning["passed"]),
        }

    if any(video is None for video in videos):
        raise RuntimeError("ZIP temporal pruning did not produce all six video diagnostics")
    completed_videos = [video for video in videos if video is not None]
    return {
        "method": f"zip_temporal_kmeans_{cross_gap_mode}_gate_pair_pruning_v1",
        "speaker_index": speaker_index,
        "provider_indices": provider_indices,
        "cluster_count": cluster_count,
        "seconds_per_cluster": float(seconds_per_cluster),
        "time_weight": float(time_weight),
        "temporal_unit_seconds": float(temporal_unit_seconds),
        "max_iterations": int(max_iterations),
        "high_similarity_threshold": float(high_similarity_threshold),
        "cross_gap_mode": cross_gap_mode,
        "max_cross_gap_seconds": float(max_cross_gap_seconds),
        "pruning_protection_mode": "min_percent",
        "min_pruned_video_percent": float(min_pruned_video_percent),
        "pair_results": pair_results,
        "events": events,
        "videos": completed_videos,
        "passed": all(pair["pruning"]["passed"] for pair in pair_results),
        "speaker_qa_media_uses_full_original": True,
    }


def blockwise_speaker_provider_all_pairs_pruning(
    frames_by_video: list[list[dict[str, Any]]],
    embeddings_by_video: list[list[list[float]]],
    *,
    speaker_index: int,
    start_seconds: float,
    duration_seconds: float,
    block_duration_seconds: float,
    sample_interval_seconds: float,
    cluster_count: int = 12,
    high_similarity_threshold: float = 0.82,
    min_pruned_video_seconds: float = 8.0,
) -> dict[str, Any]:
    """在独立时间块内复用现有 provider-only 内核并聚合全局区间。"""

    if block_duration_seconds <= 0:
        raise ValueError("block_duration_seconds must be positive")
    block_count_float = duration_seconds / block_duration_seconds
    block_count = round(block_count_float)
    if abs(block_count_float - block_count) > 1e-9:
        raise ValueError("duration_seconds must be a positive multiple of block_duration_seconds")
    if len(frames_by_video) != 6 or len(embeddings_by_video) != 6:
        raise ValueError("blockwise speaker-provider pruning requires exactly 6 videos")

    blocks: list[dict[str, Any]] = []
    global_events: list[dict[str, Any]] = []
    for block_index in range(block_count):
        block_start = start_seconds + block_index * block_duration_seconds
        block_end = block_start + block_duration_seconds
        block_frames: list[list[dict[str, Any]]] = []
        block_embeddings: list[list[list[float]]] = []
        for frames, embeddings in zip(frames_by_video, embeddings_by_video):
            selected = [
                index
                for index, frame in enumerate(frames)
                if block_start <= float(frame["timestamp_seconds"]) < block_end
            ]
            if not selected:
                raise ValueError(f"video block {block_index} has no sampled frames")
            block_frames.append([frames[index] for index in selected])
            block_embeddings.append([embeddings[index] for index in selected])

        result = clustered_speaker_provider_all_pairs_pruning(
            block_frames,
            block_embeddings,
            speaker_index=speaker_index,
            start_seconds=block_start,
            duration_seconds=block_duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            cluster_count=cluster_count,
            high_similarity_threshold=high_similarity_threshold,
            min_pruned_video_seconds=min_pruned_video_seconds,
        )
        result["block_index"] = block_index
        result["block_start_seconds"] = block_start
        result["block_end_seconds"] = block_end
        for event in result.get("events", []):
            global_events.append(
                {
                    **event,
                    "event_index": len(global_events),
                    "block_event_index": event.get("event_index"),
                    "block_index": block_index,
                }
            )
        blocks.append(result)

    videos = []
    window_end = start_seconds + duration_seconds
    for video_index in range(6):
        remove_intervals = _merge_intervals(
            [
                interval
                for block in blocks
                for interval in block["videos"][video_index].get("remove_intervals", [])
            ]
        )
        if video_index == speaker_index:
            remove_intervals = []
        keep_intervals = _subtract_intervals((start_seconds, window_end), remove_intervals)
        kept_duration = round(sum(end - start for start, end in keep_intervals), 3)
        aggregate_passed = kept_duration >= float(min_pruned_video_seconds)
        videos.append(
            {
                "video_index": video_index,
                "keep_intervals": keep_intervals,
                "remove_intervals": remove_intervals,
                "kept_duration_seconds": kept_duration,
                "removed_duration_seconds": round(duration_seconds - kept_duration, 3),
                "marked_cluster_indices": [],
                "trigger_event_indices": [],
                "block_diagnostics": [
                    block["videos"][video_index] for block in blocks
                ],
                "block_local_passes": [
                    bool(block["videos"][video_index]["passed"])
                    for block in blocks
                ],
                "aggregate_min_pruned_video_seconds": float(
                    min_pruned_video_seconds
                ),
                "aggregate_passed": aggregate_passed,
                "passed": aggregate_passed,
            }
        )

    return {
        "method": "blockwise_speaker_provider_all_pairs_provider_only",
        "speaker_index": speaker_index,
        "block_duration_seconds": block_duration_seconds,
        "block_count": block_count,
        "events": global_events,
        "blocks": blocks,
        "videos": videos,
        "aggregate_short_video_indices": [
            video["video_index"] for video in videos if not video["aggregate_passed"]
        ],
        "passed": bool(global_events) and all(video["passed"] for video in videos),
    }


def clustered_temporal_similarity_pruning(
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    left_embeddings: list[list[float]],
    right_embeddings: list[list[float]],
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    cluster_count: int = 12,
    high_similarity_threshold: float = 0.82,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "reject",
    min_pruned_video_percent: float | None = None,
    max_pair_time_difference_seconds: float | None = None,
    mutual_nearest_only: bool = False,
    split_noncontiguous_clusters: bool = False,
    max_cluster_member_gap_seconds: float | None = None,
) -> dict[str, Any]:
    """Prune high-similarity clusters using representative sampled frames.

    Each video is sampled independently, clustered, and represented by medoid
    frames. High-similarity medoid pairs mark both source clusters for pruning;
    every sampled frame assigned to a marked cluster removes an equal-width
    interval centered on that frame.
    """

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if max_pair_time_difference_seconds is not None and max_pair_time_difference_seconds < 0:
        raise ValueError("max_pair_time_difference_seconds must be non-negative")
    if split_noncontiguous_clusters and max_cluster_member_gap_seconds is None:
        max_cluster_member_gap_seconds = 1.5 * float(sample_interval_seconds)

    window_start = float(start_seconds)
    window_end = round(window_start + float(duration_seconds), 3)
    target_kept_seconds = _protected_duration_target_seconds(
        mode=pruning_protection_mode,
        duration_seconds=duration_seconds,
        min_pruned_video_seconds=min_pruned_video_seconds,
        min_pruned_video_percent=min_pruned_video_percent,
    )
    left_clusters = clustered_frame_representatives(
        left_frames,
        left_embeddings,
        cluster_count=cluster_count,
        split_noncontiguous_clusters=split_noncontiguous_clusters,
        max_member_gap_seconds=max_cluster_member_gap_seconds,
    )
    right_clusters = clustered_frame_representatives(
        right_frames,
        right_embeddings,
        cluster_count=cluster_count,
        split_noncontiguous_clusters=split_noncontiguous_clusters,
        max_member_gap_seconds=max_cluster_member_gap_seconds,
    )
    matrix = frame_similarity_matrix(
        left_clusters["representative_embeddings"],
        right_clusters["representative_embeddings"],
    )

    def representative_time_difference(left_index: int, right_index: int) -> float:
        left_timestamp = float(
            left_clusters["representatives"][left_index].get("timestamp_seconds", 0.0)
        )
        right_timestamp = float(
            right_clusters["representatives"][right_index].get("timestamp_seconds", 0.0)
        )
        return abs(left_timestamp - right_timestamp)

    def representative_pair_is_eligible(left_index: int, right_index: int) -> bool:
        return (
            max_pair_time_difference_seconds is None
            or representative_time_difference(left_index, right_index)
            <= max_pair_time_difference_seconds + 1e-9
        )

    left_best: dict[int, int] = {}
    for left_cluster_index, row in enumerate(matrix):
        choices = [
            (right_cluster_index, float(similarity))
            for right_cluster_index, similarity in enumerate(row)
            if representative_pair_is_eligible(left_cluster_index, right_cluster_index)
        ]
        if choices:
            left_best[left_cluster_index] = max(choices, key=lambda item: item[1])[0]
    right_best: dict[int, int] = {}
    if matrix:
        for right_cluster_index in range(len(matrix[0])):
            choices = [
                (left_cluster_index, float(matrix[left_cluster_index][right_cluster_index]))
                for left_cluster_index in range(len(matrix))
                if representative_pair_is_eligible(left_cluster_index, right_cluster_index)
            ]
            if choices:
                right_best[right_cluster_index] = max(choices, key=lambda item: item[1])[0]

    high_pairs = []
    left_marked_clusters: set[int] = set()
    right_marked_clusters: set[int] = set()
    for left_cluster_index, row in enumerate(matrix):
        for right_cluster_index, similarity in enumerate(row):
            if not representative_pair_is_eligible(left_cluster_index, right_cluster_index):
                continue
            if float(similarity) < high_similarity_threshold:
                continue
            if mutual_nearest_only and not (
                left_best.get(left_cluster_index) == right_cluster_index
                and right_best.get(right_cluster_index) == left_cluster_index
            ):
                continue
            left_marked_clusters.add(left_cluster_index)
            right_marked_clusters.add(right_cluster_index)
            left_rep = left_clusters["representatives"][left_cluster_index]
            right_rep = right_clusters["representatives"][right_cluster_index]
            high_pairs.append(
                {
                    "left_cluster_index": int(left_cluster_index),
                    "right_cluster_index": int(right_cluster_index),
                    "similarity": round(float(similarity), 6),
                    "left_representative_frame_index": left_rep["frame_index"],
                    "right_representative_frame_index": right_rep["frame_index"],
                    "left_representative_timestamp_seconds": left_rep.get("timestamp_seconds"),
                    "right_representative_timestamp_seconds": right_rep.get("timestamp_seconds"),
                    "timestamp_difference_seconds": round(
                        representative_time_difference(left_cluster_index, right_cluster_index),
                        6,
                    ),
                }
            )

    high_pairs.sort(key=lambda row: float(row["similarity"]), reverse=True)
    preserved_intervals: list[tuple[float, float]] = []
    if high_pairs and preserve_shared_anchor_seconds > 0:
        strongest = high_pairs[0]
        left_center = float(strongest["left_representative_timestamp_seconds"])
        right_center = float(strongest["right_representative_timestamp_seconds"])
        center = (left_center + right_center) / 2.0
        half_preserve = min(float(preserve_shared_anchor_seconds), float(duration_seconds)) / 2.0
        preserved_intervals = [
            (
                max(window_start, center - half_preserve),
                min(window_end, center + half_preserve),
            )
        ]

    def marked_frame_indices(clusters: dict[str, Any], marked_clusters: set[int]) -> set[int]:
        indices: set[int] = set()
        for cluster_index in marked_clusters:
            representative = clusters["representatives"][cluster_index]
            indices.update(int(index) for index in representative.get("member_indices", []))
        return indices

    left_marked_indices = marked_frame_indices(left_clusters, left_marked_clusters)
    right_marked_indices = marked_frame_indices(right_clusters, right_marked_clusters)
    full_frame_matrix = frame_similarity_matrix(left_embeddings, right_embeddings)
    left_protection = _apply_pruning_duration_protection(
        left_frames,
        left_marked_indices,
        _side_best_frame_matches(
            full_frame_matrix,
            side="left",
            left_frames=left_frames,
            right_frames=right_frames,
            max_pair_time_difference_seconds=max_pair_time_difference_seconds,
        ),
        side="left",
        window_start=window_start,
        window_end=window_end,
        sample_interval_seconds=sample_interval_seconds,
        high_similarity_threshold=high_similarity_threshold,
        target_kept_seconds=target_kept_seconds,
        preserved_intervals=preserved_intervals,
    )
    right_protection = _apply_pruning_duration_protection(
        right_frames,
        right_marked_indices,
        _side_best_frame_matches(
            full_frame_matrix,
            side="right",
            left_frames=left_frames,
            right_frames=right_frames,
            max_pair_time_difference_seconds=max_pair_time_difference_seconds,
        ),
        side="right",
        window_start=window_start,
        window_end=window_end,
        sample_interval_seconds=sample_interval_seconds,
        high_similarity_threshold=high_similarity_threshold,
        target_kept_seconds=target_kept_seconds,
        preserved_intervals=preserved_intervals,
    )
    left_marked_indices = left_protection["marked_indices"]
    right_marked_indices = right_protection["marked_indices"]
    left_remove_intervals = left_protection["remove_intervals"]
    right_remove_intervals = right_protection["remove_intervals"]
    left_keep_intervals = left_protection["keep_intervals"]
    right_keep_intervals = right_protection["keep_intervals"]
    left_kept_duration = left_protection["kept_duration_seconds"]
    right_kept_duration = right_protection["kept_duration_seconds"]
    left_removed_duration = left_protection["removed_duration_seconds"]
    right_removed_duration = right_protection["removed_duration_seconds"]
    removed_duration = round(left_removed_duration + right_removed_duration, 3)
    kept_duration = round(min(left_kept_duration, right_kept_duration), 3)
    required_kept_duration = (
        float(min_pruned_video_seconds)
        if pruning_protection_mode == "reject"
        else float(target_kept_seconds or 0.0)
    )
    passed = (
        left_kept_duration >= required_kept_duration
        and right_kept_duration >= required_kept_duration
        and left_protection["target_met"]
        and right_protection["target_met"]
        and removed_duration > 0.0
    )

    def cluster_decisions(clusters: dict[str, Any], marked_clusters: set[int]) -> list[dict[str, Any]]:
        rows = []
        for representative in clusters["representatives"]:
            cluster_index = int(representative["cluster_index"])
            rows.append(
                {
                    **representative,
                    "status": "marked_for_pruning" if cluster_index in marked_clusters else "kept",
                }
            )
        return rows

    return {
        "method": "cluster_representative_high_similarity_interval_pruning",
        "high_similarity_threshold": high_similarity_threshold,
        "cluster_count": cluster_count,
        "left_cluster_count": left_clusters["cluster_count"],
        "right_cluster_count": right_clusters["cluster_count"],
        "max_pair_time_difference_seconds": max_pair_time_difference_seconds,
        "mutual_nearest_only": mutual_nearest_only,
        "split_noncontiguous_clusters": split_noncontiguous_clusters,
        "max_cluster_member_gap_seconds": max_cluster_member_gap_seconds,
        "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
        "min_pruned_video_seconds": min_pruned_video_seconds,
        "pruning_protection_mode": pruning_protection_mode,
        "min_pruned_video_percent": min_pruned_video_percent,
        "protection_target_kept_seconds": target_kept_seconds,
        "required_kept_duration_seconds": round(required_kept_duration, 3),
        "window": {
            "start_seconds": round(window_start, 3),
            "end_seconds": window_end,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "representative_similarity_matrix": matrix,
        "high_similarity_representative_pairs": high_pairs,
        "high_similarity_representative_pair_count": len(high_pairs),
        "left_marked_cluster_count": len(left_marked_clusters),
        "right_marked_cluster_count": len(right_marked_clusters),
        "left_marked_frame_indices": sorted(left_marked_indices),
        "right_marked_frame_indices": sorted(right_marked_indices),
        "left_restored_frame_indices": [int(row["frame_index"]) for row in left_protection["restored_frames"]],
        "right_restored_frame_indices": [int(row["frame_index"]) for row in right_protection["restored_frames"]],
        "left_restored_frames": left_protection["restored_frames"],
        "right_restored_frames": right_protection["restored_frames"],
        "duration_protection": {
            "mode": pruning_protection_mode,
            "target_kept_seconds": target_kept_seconds,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "min_pruned_video_percent": min_pruned_video_percent,
            "left_target_met": left_protection["target_met"],
            "right_target_met": right_protection["target_met"],
            "selection_rule": (
                "When protection is enabled, restore least-similar sampled-frame intervals whose "
                "best cross-video CLIP similarity is still at or above high_similarity_threshold."
            ),
        },
        "left_remove_intervals": [[round(start, 3), round(end, 3)] for start, end in left_remove_intervals],
        "right_remove_intervals": [[round(start, 3), round(end, 3)] for start, end in right_remove_intervals],
        "left_keep_intervals": [[round(start, 3), round(end, 3)] for start, end in left_keep_intervals],
        "right_keep_intervals": [[round(start, 3), round(end, 3)] for start, end in right_keep_intervals],
        "remove_intervals": {
            "left": [[round(start, 3), round(end, 3)] for start, end in left_remove_intervals],
            "right": [[round(start, 3), round(end, 3)] for start, end in right_remove_intervals],
        },
        "keep_intervals": {
            "left": [[round(start, 3), round(end, 3)] for start, end in left_keep_intervals],
            "right": [[round(start, 3), round(end, 3)] for start, end in right_keep_intervals],
        },
        "preserved_shared_intervals": [
            [round(start, 3), round(end, 3)] for start, end in _merge_intervals(preserved_intervals)
        ],
        "left_removed_duration_seconds": left_removed_duration,
        "right_removed_duration_seconds": right_removed_duration,
        "removed_duration_seconds": removed_duration,
        "left_kept_duration_seconds": left_kept_duration,
        "right_kept_duration_seconds": right_kept_duration,
        "kept_duration_seconds": kept_duration,
        "passed": passed,
        "left_cluster_decisions": cluster_decisions(left_clusters, left_marked_clusters),
        "right_cluster_decisions": cluster_decisions(right_clusters, right_marked_clusters),
    }


def temporal_similarity_pruning(
    matrix: list[list[float]],
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    *,
    start_seconds: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    high_similarity_threshold: float = 0.82,
    temporal_neighborhood_seconds: float | None = None,
    preserve_shared_anchor_seconds: float = 4.0,
    min_pruned_video_seconds: float = 8.0,
) -> dict[str, Any]:
    """Turn nearby high-similarity checkpoints into video intervals to remove.

    Similarity is computed on sampled checkpoints, but pruning is applied to
    time intervals in the original video window. A short strongest shared span
    can be preserved so the generator still has common evidence to anchor a
    natural cross-video question.
    """

    if len(matrix) != len(left_frames):
        raise ValueError("left frame count must match frame similarity matrix rows")
    if any(len(row) != len(right_frames) for row in matrix):
        raise ValueError("right frame count must match frame similarity matrix columns")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")

    window_start = float(start_seconds)
    window_end = round(window_start + float(duration_seconds), 3)
    neighborhood = (
        float(temporal_neighborhood_seconds)
        if temporal_neighborhood_seconds is not None
        else float(sample_interval_seconds) / 2.0 + 1e-6
    )
    half_width = float(sample_interval_seconds) / 2.0

    checkpoint_rows = []
    for left_index, left_frame in enumerate(left_frames):
        left_time = float(left_frame.get("timestamp_seconds", window_start))
        nearby = [
            (right_index, matrix[left_index][right_index])
            for right_index, right_frame in enumerate(right_frames)
            if abs(float(right_frame.get("timestamp_seconds", window_start)) - left_time) <= neighborhood
        ]
        if not nearby:
            continue
        right_index, similarity = max(nearby, key=lambda item: item[1])
        interval = (
            max(window_start, left_time - half_width),
            min(window_end, left_time + half_width),
        )
        if interval[1] <= interval[0]:
            continue
        right_time = float(right_frames[right_index].get("timestamp_seconds", window_start))
        checkpoint_rows.append(
            {
                "left_index": left_index,
                "right_index": int(right_index),
                "timestamp_seconds": round(left_time, 3),
                "right_timestamp_seconds": round(right_time, 3),
                "nearby_similarity": round(float(similarity), 6),
                "interval": [round(interval[0], 3), round(interval[1], 3)],
                "is_high_similarity": float(similarity) >= high_similarity_threshold,
            }
        )

    high_rows = [row for row in checkpoint_rows if row["is_high_similarity"]]
    remove_intervals = [tuple(row["interval"]) for row in high_rows]
    preserved_intervals: list[tuple[float, float]] = []
    if high_rows and preserve_shared_anchor_seconds > 0:
        strongest = max(
            high_rows,
            key=lambda row: (float(row["nearby_similarity"]), -abs(float(row["timestamp_seconds"]) - window_start)),
        )
        center = float(strongest["timestamp_seconds"])
        half_preserve = min(float(preserve_shared_anchor_seconds), float(duration_seconds)) / 2.0
        preserved_intervals = [
            (
                max(window_start, center - half_preserve),
                min(window_end, center + half_preserve),
            )
        ]
        remove_intervals = [
            interval
            for interval in remove_intervals
            if not any(interval[0] < preserved[1] and interval[1] > preserved[0] for preserved in preserved_intervals)
        ]

    remove_intervals = _merge_intervals(remove_intervals)
    keep_intervals = _subtract_intervals((window_start, window_end), remove_intervals)
    kept_duration = round(sum(end - start for start, end in keep_intervals), 3)
    removed_duration = round(sum(end - start for start, end in remove_intervals), 3)
    passed = kept_duration >= min_pruned_video_seconds and removed_duration > 0.0
    return {
        "method": "remove_nearby_high_similarity_time_intervals",
        "high_similarity_threshold": high_similarity_threshold,
        "temporal_neighborhood_seconds": temporal_neighborhood_seconds,
        "effective_temporal_neighborhood_seconds": round(neighborhood, 3),
        "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
        "min_pruned_video_seconds": min_pruned_video_seconds,
        "window": {
            "start_seconds": round(window_start, 3),
            "end_seconds": window_end,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "checkpoint_count": len(checkpoint_rows),
        "high_similarity_checkpoint_count": len(high_rows),
        "remove_intervals": [[round(start, 3), round(end, 3)] for start, end in remove_intervals],
        "keep_intervals": [[round(start, 3), round(end, 3)] for start, end in keep_intervals],
        "preserved_shared_intervals": [
            [round(start, 3), round(end, 3)] for start, end in _merge_intervals(preserved_intervals)
        ],
        "removed_duration_seconds": removed_duration,
        "kept_duration_seconds": kept_duration,
        "passed": passed,
        "checkpoint_decisions": checkpoint_rows,
    }


def _resolve_ffmpeg_binary(ffmpeg_binary: str) -> str:
    ffmpeg = shutil.which(ffmpeg_binary)
    if not ffmpeg:
        explicit = Path(ffmpeg_binary)
        if explicit.exists():
            ffmpeg = str(explicit)
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to materialize pruned videos")
    return ffmpeg


def materialize_pruned_video(
    source_video: str | Path,
    output_video: str | Path,
    keep_intervals: list[list[float]] | list[tuple[float, float]],
    *,
    ffmpeg_binary: str = "ffmpeg",
) -> Path:
    """Write a new MP4 by concatenating the requested source-video intervals."""

    if not keep_intervals:
        raise ValueError("cannot materialize a pruned video with no keep intervals")
    source = Path(source_video)
    if not source.exists():
        raise FileNotFoundError(f"source video does not exist: {source}")
    output = Path(output_video)
    output.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = _resolve_ffmpeg_binary(ffmpeg_binary)
    trim_parts = []
    concat_inputs = []
    for index, interval in enumerate(keep_intervals):
        start, end = float(interval[0]), float(interval[1])
        if end <= start:
            continue
        trim_parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},setpts=PTS-STARTPTS[v{index}]")
        concat_inputs.append(f"[v{index}]")
    if not trim_parts:
        raise ValueError("all keep intervals were empty")
    if len(trim_parts) == 1:
        filter_complex = trim_parts[0].replace("[v0]", "[outv]")
    else:
        filter_complex = ";".join(trim_parts)
        filter_complex += f";{''.join(concat_inputs)}concat=n={len(trim_parts)}:v=1:a=0[outv]"

    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            filter_complex,
            "-map",
            "[outv]",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(output),
        ],
        check=True,
    )
    return output


def score_video_pairs(
    clip_rows: list[dict[str, Any]],
    frame_embeddings_by_clip: list[list[list[float]]],
    scoring: dict[str, Any],
    *,
    topk: int = 3,
    min_topk_sim: float = 0.65,
    min_mean_sim: float = 0.25,
    max_mean_sim: float = 0.90,
    start_seconds: float = 0.0,
    duration_seconds: float = 30.0,
    sample_interval_seconds: float = 1.0,
    pruning_clusters_per_video: int = 12,
    high_similarity_interval_threshold: float = 0.82,
    temporal_neighborhood_seconds: float | None = None,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "reject",
    min_pruned_video_percent: float | None = None,
    max_pair_time_difference_seconds: float | None = None,
) -> dict[str, Any]:
    """Filter video pairs using clustered-frame representatives and overlap metrics."""

    clip_scores = scoring.get("clip_scores", [])
    if len(clip_rows) != len(frame_embeddings_by_clip) or len(clip_rows) != len(clip_scores):
        raise ValueError("clip rows, frame embeddings, and clip scores must align")
    if len(clip_rows) < 2:
        raise ValueError("at least two clips are required")

    pairs = []
    for left_index in range(len(clip_rows)):
        for right_index in range(left_index + 1, len(clip_rows)):
            left = clip_scores[left_index]
            right = clip_scores[right_index]
            temporal_pruning = clustered_temporal_similarity_pruning(
                clip_rows[left_index].get("frames", []),
                clip_rows[right_index].get("frames", []),
                frame_embeddings_by_clip[left_index],
                frame_embeddings_by_clip[right_index],
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                cluster_count=pruning_clusters_per_video,
                high_similarity_threshold=high_similarity_interval_threshold,
                preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
                min_pruned_video_seconds=min_pruned_video_seconds,
                pruning_protection_mode=pruning_protection_mode,
                min_pruned_video_percent=min_pruned_video_percent,
                max_pair_time_difference_seconds=max_pair_time_difference_seconds,
            )
            matrix = temporal_pruning["representative_similarity_matrix"]
            values = _flatten_matrix(matrix)
            mean_sim = sum(values) / len(values)
            topk_sim = _topk_mean(values, topk)
            rejection_reasons = []
            if topk_sim < min_topk_sim:
                rejection_reasons.append("topk_sim_too_low_no_shared_anchor")
            if mean_sim < min_mean_sim:
                rejection_reasons.append("mean_sim_too_low_unrelated")
            if mean_sim > max_mean_sim:
                rejection_reasons.append("mean_sim_too_high_redundant")
            if not temporal_pruning["passed"]:
                rejection_reasons.append("pruned_video_too_short_after_removing_high_similarity_intervals")

            pairs.append(
                {
                    "pair_key": f"{left_index}-{right_index}",
                    "left_index": left_index,
                    "right_index": right_index,
                    "left_agent_dir": left.get("agent_dir"),
                    "left_agent_name": left.get("agent_name"),
                    "right_agent_dir": right.get("agent_dir"),
                    "right_agent_name": right.get("agent_name"),
                    "mean_sim": round(mean_sim, 6),
                    "topk_sim": round(topk_sim, 6),
                    "topk": max(1, min(topk, len(values))),
                    "representative_similarity_matrix": matrix,
                    "temporal_pruning": temporal_pruning,
                    "left_frame_count": len(frame_embeddings_by_clip[left_index]),
                    "right_frame_count": len(frame_embeddings_by_clip[right_index]),
                    "left_mean_similarity_to_group": left.get("mean_similarity_to_group"),
                    "right_mean_similarity_to_group": right.get("mean_similarity_to_group"),
                    "mean_clip_typicality_middle_score": round(
                        (
                            float(left.get("typicality_middle_score", 0.0))
                            + float(right.get("typicality_middle_score", 0.0))
                        )
                        / 2.0,
                        6,
                    ),
                    "status": "kept" if not rejection_reasons else "rejected",
                    "rejection_reasons": rejection_reasons,
                    "rejection_reason": ";".join(rejection_reasons) if rejection_reasons else None,
                }
            )

    pair_scores = sorted(
        pairs,
        key=lambda item: (
            item["status"] != "kept",
            -float(item["topk_sim"]),
            abs(float(item["mean_sim"]) - ((min_mean_sim + max_mean_sim) / 2.0)),
            str(item["pair_key"]),
        ),
    )
    kept_pairs = [pair for pair in pair_scores if pair["status"] == "kept"]
    for rank, pair in enumerate(pair_scores, 1):
        pair["trace_rank"] = rank
    return {
        "pair_filter": {
            "method": "random_pair_cluster_representative_shared_anchor_and_overlap_thresholds",
            "topk": topk,
            "min_topk_sim": min_topk_sim,
            "min_mean_sim": min_mean_sim,
            "max_mean_sim": max_mean_sim,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "pruning_clusters_per_video": pruning_clusters_per_video,
            "high_similarity_interval_threshold": high_similarity_interval_threshold,
            "temporal_neighborhood_seconds": temporal_neighborhood_seconds,
            "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_percent": min_pruned_video_percent,
            "max_pair_time_difference_seconds": max_pair_time_difference_seconds,
            "pair_count": len(pair_scores),
            "kept_pair_count": len(kept_pairs),
            "interpretation": (
                "Each selected video is sampled once per second, clustered, and represented by medoid "
                "frames. Pair scores are computed from representative CLIP similarities; high "
                "representative matches mark their source clusters for uniform interval pruning. "
                "topk_sim captures strongest shared anchors; mean_sim captures representative overlap. "
                "Pairs are rejected when shared anchors are too weak, global overlap is too low, "
                "global overlap is too high, or high-similarity interval removal would leave too "
                "little video. By default the synchronized pair is sampled before CLIP embedding; "
                "all-pairs group comparison only runs when explicitly requested. Selected videos are "
                "materialized as paired original/pruned MP4s. QA generation uses pruned MP4s; "
                "judges and answerability gates use the original 30-second MP4s."
            ),
        },
        "pair_scores": pair_scores,
        "surviving_pairs": kept_pairs,
        "ranked_pairs": kept_pairs,
        "rejected_pairs": [pair for pair in pair_scores if pair["status"] == "rejected"],
    }


def compact_pair_rejection_summary(pair_analysis: dict[str, Any], *, limit: int = 3) -> list[dict[str, Any]]:
    """Return compact diagnostics for rejected pair-filter decisions."""

    rows = []
    for pair in pair_analysis.get("pair_scores", [])[:limit]:
        pruning = pair.get("temporal_pruning") if isinstance(pair.get("temporal_pruning"), dict) else {}
        rows.append(
            {
                "pair_key": pair.get("pair_key"),
                "status": pair.get("status"),
                "rejection_reasons": pair.get("rejection_reasons", []),
                "mean_sim": pair.get("mean_sim"),
                "topk_sim": pair.get("topk_sim"),
                "high_similarity_representative_pair_count": pruning.get(
                    "high_similarity_representative_pair_count"
                ),
                "left_marked_cluster_count": pruning.get("left_marked_cluster_count"),
                "right_marked_cluster_count": pruning.get("right_marked_cluster_count"),
                "left_kept_duration_seconds": pruning.get("left_kept_duration_seconds"),
                "right_kept_duration_seconds": pruning.get("right_kept_duration_seconds"),
                "left_removed_duration_seconds": pruning.get("left_removed_duration_seconds"),
                "right_removed_duration_seconds": pruning.get("right_removed_duration_seconds"),
            }
        )
    return rows


def build_six_user_role_structures(
    pair_scores: list[dict[str, Any]],
    *,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """从六人全部 pair 结果中构造可依次尝试的双锚点星型角色结构。"""

    expected_keys = {
        (left_index, right_index)
        for left_index in range(6)
        for right_index in range(left_index + 1, 6)
    }
    pair_by_indices: dict[tuple[int, int], dict[str, Any]] = {}
    for pair in pair_scores:
        left_index = int(pair.get("left_index", -1))
        right_index = int(pair.get("right_index", -1))
        key = tuple(sorted((left_index, right_index)))
        if key not in expected_keys:
            raise ValueError(f"invalid six-user pair edge: {key!r}")
        if key in pair_by_indices:
            raise ValueError(f"duplicate six-user pair edge: {key!r}")
        pair_by_indices[key] = pair

    if set(pair_by_indices) != expected_keys:
        missing = sorted(expected_keys - set(pair_by_indices))
        raise ValueError(
            "six-user role selection requires all 15 pair edges; "
            f"received {len(pair_by_indices)}, missing {missing}"
        )

    kept_neighbors: dict[int, list[int]] = {index: [] for index in range(6)}
    for (left_index, right_index), pair in pair_by_indices.items():
        if pair.get("status") == "kept":
            kept_neighbors[left_index].append(right_index)
            kept_neighbors[right_index].append(left_index)
    for neighbors in kept_neighbors.values():
        neighbors.sort()

    role_structures = []
    for speaker_index in range(6):
        neighbors = kept_neighbors[speaker_index]
        for first_offset, first_anchor in enumerate(neighbors):
            for second_anchor in neighbors[first_offset + 1 :]:
                anchor_indices = [first_anchor, second_anchor]
                additional_indices = [
                    index
                    for index in range(6)
                    if index != speaker_index and index not in anchor_indices
                ]
                selected_anchor_edges = [
                    pair_by_indices[tuple(sorted((speaker_index, anchor_index)))]
                    for anchor_index in anchor_indices
                ]
                role_structures.append(
                    {
                        "speaker_index": speaker_index,
                        "anchor_indices": anchor_indices,
                        "additional_indices": additional_indices,
                        "selected_anchor_edges": selected_anchor_edges,
                    }
                )

    active_rng = rng or random.Random()
    active_rng.shuffle(role_structures)
    for candidate_rank, structure in enumerate(role_structures, start=1):
        structure["candidate_rank"] = candidate_rank

    kept_degrees = [len(kept_neighbors[index]) for index in range(6)]
    return {
        "role_structures": role_structures,
        "eligible_speaker_indices": [
            index for index, degree in enumerate(kept_degrees) if degree >= 2
        ],
        "kept_degrees": kept_degrees,
        "diagnostic_pair_edges": list(pair_scores),
    }


def _resolve_local_video(
    clip: dict[str, Any],
    *,
    cache_dir: str | Path,
    download_media: bool,
) -> Path:
    local_video = clip.get("local_video")
    if local_video and Path(local_video).exists():
        return Path(local_video)

    video_path = clip.get("video_path")
    if not video_path:
        raise FileNotFoundError(f"clip is missing video_path/local_video: {clip.get('clip_id')}")
    candidate = local_cache_path(cache_dir, str(video_path))
    if candidate.exists():
        return candidate
    if download_media:
        video_url = clip.get("video_url")
        if not video_url:
            raise FileNotFoundError(f"clip is missing video_url: {clip.get('clip_id')}")
        return download_file(str(video_url), candidate)
    raise FileNotFoundError(f"local video is unavailable for {clip.get('agent_dir')}: {candidate}")


def _resolve_group_local_video(
    clip: dict[str, Any],
    group: dict[str, Any],
    *,
    cache_dir: str | Path,
    download_media: bool,
) -> Path:
    segments = list(clip.get("segments") or [])
    if len(segments) <= 1:
        return _resolve_local_video(
            segments[0] if segments else clip,
            cache_dir=cache_dir,
            download_media=download_media,
        )

    source_paths = [
        _resolve_local_video(
            segment,
            cache_dir=cache_dir,
            download_media=download_media,
        )
        for segment in segments
    ]
    output = window_cache_path(
        cache_dir,
        day=str(group["day"]),
        agent_dir=str(clip["agent_dir"]),
        agent_id=str(clip["agent_id"]),
        agent_name=str(clip["agent_name"]),
        time_token=str(group["time_token"]),
        duration_seconds=float(group["duration_seconds"]),
    )
    if not output.is_file() or output.stat().st_size == 0:
        concatenate_video_segments(
            source_paths,
            output,
            duration_seconds=float(group["duration_seconds"]),
        )
    return output


def group_clip_frames(
    group: dict[str, Any],
    output_dir: str | Path,
    *,
    cache_dir: str | Path,
    duration_seconds: float,
    sample_interval_seconds: float,
    start_seconds: float,
    ffmpeg_binary: str,
    download_media: bool,
) -> list[dict[str, Any]]:
    """Sample the same temporal window from every clip in a synchronized group."""

    rows = []
    group_dir = Path(output_dir) / stable_id(group.get("day"), group.get("time_token"))
    for clip in sorted(group.get("clips", []), key=lambda item: str(item.get("agent_dir"))):
        user = str(clip.get("agent_name") or clip.get("agent_dir"))
        local_video = _resolve_group_local_video(
            clip,
            group,
            cache_dir=cache_dir,
            download_media=download_media,
        )
        frames = sample_short_video(
            local_video,
            group_dir / "sampled_frames" / str(clip.get("agent_dir") or user),
            duration_seconds=duration_seconds,
            sample_interval_seconds=sample_interval_seconds,
            start_seconds=start_seconds,
            ffmpeg_binary=ffmpeg_binary,
        )
        clip_with_local = dict(clip)
        clip_with_local["local_video"] = str(local_video)
        clip_with_local["source_segments"] = list(clip.get("segments") or [clip])
        clip_with_local["segment_count"] = int(group.get("segment_count") or 1)
        clip_with_local["duration_seconds"] = float(
            group.get("duration_seconds") or duration_seconds
        )
        rows.append({"user": user, "clip": clip_with_local, "frames": frames})
    return rows


def _sample_group_clips_for_pair(
    group: dict[str, Any],
    *,
    selected_count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    clips = sorted(group.get("clips", []), key=lambda item: str(item.get("agent_dir")))
    if len(clips) < selected_count:
        raise ValueError(f"group needs at least {selected_count} clips")
    if len(clips) == selected_count:
        return clips
    return rng.sample(clips, selected_count)


def _clip_with_pruned_video(
    clip: dict[str, Any],
    *,
    side: str,
    pair: dict[str, Any],
    output_dir: str | Path,
    ffmpeg_binary: str,
) -> dict[str, Any]:
    pruned = dict(clip)
    source_video = pruned.get("local_video")
    if not source_video:
        raise FileNotFoundError(f"selected clip is missing local_video: {pruned.get('clip_id')}")
    pruning = pair.get("temporal_pruning", {})
    side_keep_key = f"{side}_keep_intervals"
    side_remove_key = f"{side}_remove_intervals"
    keep_intervals = pruning.get(side_keep_key)
    if keep_intervals is None:
        keep_intervals = pruning.get("keep_intervals", [])
        if isinstance(keep_intervals, dict):
            keep_intervals = keep_intervals.get(side, [])
    remove_intervals = pruning.get(side_remove_key)
    if remove_intervals is None:
        remove_intervals = pruning.get("remove_intervals", [])
        if isinstance(remove_intervals, dict):
            remove_intervals = remove_intervals.get(side, [])
    cluster_decisions = pruning.get(f"{side}_cluster_decisions", [])
    kept_cluster_representatives = [
        dict(row)
        for row in cluster_decisions
        if row.get("status") == "kept"
    ]
    pair_key = _safe_filename_part(pair.get("pair_key"))
    agent = _safe_filename_part(pruned.get("agent_dir") or pruned.get("agent_name") or side)
    pair_dir = Path(output_dir) / "benchmark_video_pairs" / pair_key
    pair_dir.mkdir(parents=True, exist_ok=True)
    source_suffix = Path(source_video).suffix or ".mp4"
    original_video = pair_dir / f"{side}_{agent}_original{source_suffix}"
    shutil.copy2(source_video, original_video)
    output_video = pair_dir / f"{side}_{agent}_pruned.mp4"
    materialize_pruned_video(
        source_video,
        output_video,
        keep_intervals,
        ffmpeg_binary=ffmpeg_binary,
    )
    pruned["source_local_video"] = source_video
    pruned["original_local_video"] = str(original_video)
    pruned["full_local_video"] = str(original_video)
    pruned["local_video"] = str(output_video)
    pruned["generator_media_mode"] = "pruned_video"
    pruned["generator_local_video"] = str(output_video)
    pruned["temporal_pruning"] = {
        "side": side,
        "pair_key": pair.get("pair_key"),
        "source_local_video": source_video,
        "original_local_video": str(original_video),
        "pruned_local_video": str(output_video),
        "method": pruning.get("method"),
        "high_similarity_threshold": pruning.get("high_similarity_threshold"),
        "pruning_protection_mode": pruning.get("pruning_protection_mode"),
        "min_pruned_video_percent": pruning.get("min_pruned_video_percent"),
        "protection_target_kept_seconds": pruning.get("protection_target_kept_seconds"),
        "required_kept_duration_seconds": pruning.get("required_kept_duration_seconds"),
        "keep_intervals": keep_intervals,
        "remove_intervals": remove_intervals,
        "cluster_decisions": cluster_decisions,
        "kept_cluster_representatives": kept_cluster_representatives,
        "kept_cluster_count": len(kept_cluster_representatives),
        "restored_frame_indices": pruning.get(f"{side}_restored_frame_indices", []),
        "restored_frames": pruning.get(f"{side}_restored_frames", []),
        "preserved_shared_intervals": pruning.get("preserved_shared_intervals", []),
        "kept_duration_seconds": pruning.get(f"{side}_kept_duration_seconds", pruning.get("kept_duration_seconds")),
        "removed_duration_seconds": pruning.get(
            f"{side}_removed_duration_seconds",
            pruning.get("removed_duration_seconds"),
        ),
    }
    pruned["benchmark_media"] = {
        "generator_video": str(output_video),
        "judge_video": str(original_video),
        "answerability_video": str(original_video),
        "source_cache_video": source_video,
    }
    return pruned


def selected_clips_for_pair_from_rows(
    rows: list[dict[str, Any]],
    pair: dict[str, Any],
    *,
    output_dir: str | Path,
    ffmpeg_binary: str,
) -> list[dict[str, Any]]:
    """Return selected clips whose local_video points to pruned MP4s."""

    left_index = int(pair["left_index"])
    right_index = int(pair["right_index"])
    return [
        _clip_with_pruned_video(
            rows[left_index]["clip"],
            side="left",
            pair=pair,
            output_dir=output_dir,
            ffmpeg_binary=ffmpeg_binary,
        ),
        _clip_with_pruned_video(
            rows[right_index]["clip"],
            side="right",
            pair=pair,
            output_dir=output_dir,
            ffmpeg_binary=ffmpeg_binary,
        ),
    ]


def selected_clips_for_pair_from_group_result(
    group_result: dict[str, Any],
    pair: dict[str, Any],
    *,
    output_dir: str | Path,
    ffmpeg_binary: str,
) -> list[dict[str, Any]]:
    """Return selected group clips with the pair's temporal pruning applied."""

    group_clips = group_result["group_clips"]
    left_index = int(pair["left_index"])
    right_index = int(pair["right_index"])
    return [
        _clip_with_pruned_video(
            group_clips[left_index],
            side="left",
            pair=pair,
            output_dir=output_dir,
            ffmpeg_binary=ffmpeg_binary,
        ),
        _clip_with_pruned_video(
            group_clips[right_index],
            side="right",
            pair=pair,
            output_dir=output_dir,
            ffmpeg_binary=ffmpeg_binary,
        ),
    ]


def _pair_side_for_clip(pair: dict[str, Any], clip_index: int) -> str:
    if int(pair.get("left_index", -1)) == clip_index:
        return "left"
    if int(pair.get("right_index", -1)) == clip_index:
        return "right"
    raise ValueError(
        f"pair {pair.get('pair_key')!r} does not contain clip index {clip_index}"
    )


def _pair_side_intervals(
    pair: dict[str, Any],
    clip_index: int,
) -> tuple[list[list[float]], list[list[float]]]:
    side = _pair_side_for_clip(pair, clip_index)
    pruning = pair.get("temporal_pruning")
    if not isinstance(pruning, dict):
        raise ValueError(f"pair {pair.get('pair_key')!r} is missing temporal_pruning")
    keep_intervals = pruning.get(f"{side}_keep_intervals")
    remove_intervals = pruning.get(f"{side}_remove_intervals")
    if not isinstance(keep_intervals, list) or not isinstance(remove_intervals, list):
        raise ValueError(
            f"pair {pair.get('pair_key')!r} is missing {side}-side pruning intervals"
        )
    return keep_intervals, remove_intervals


def _materialize_six_user_clip(
    clip: dict[str, Any],
    *,
    media_role: str,
    position: int,
    output_dir: str | Path,
    keep_intervals: list[list[float]] | list[tuple[float, float]] | None,
    remove_intervals: list[list[float]] | list[tuple[float, float]],
    source_edges: list[dict[str, Any]],
    ffmpeg_binary: str,
) -> dict[str, Any]:
    result = dict(clip)
    source_video = result.get("local_video")
    if not source_video or not Path(source_video).is_file():
        raise FileNotFoundError(
            f"six-user selected clip is missing local_video: {result.get('clip_id')}"
        )

    role_dir = Path(output_dir)
    role_dir.mkdir(parents=True, exist_ok=True)
    agent = _safe_filename_part(result.get("agent_dir") or result.get("agent_name") or position)
    source_suffix = Path(source_video).suffix or ".mp4"
    full_video = role_dir / f"{position:02d}_{agent}_full{source_suffix}"
    shutil.copy2(source_video, full_video)

    is_pruned = keep_intervals is not None
    if is_pruned:
        generator_video = role_dir / f"{position:02d}_{agent}_pruned.mp4"
        materialize_pruned_video(
            source_video,
            generator_video,
            keep_intervals,
            ffmpeg_binary=ffmpeg_binary,
        )
        generator_media_mode = "pruned_video"
    else:
        generator_video = full_video
        generator_media_mode = "full_video"

    result.update(
        {
            "source_local_video": str(source_video),
            "original_local_video": str(full_video),
            "full_local_video": str(full_video),
            "local_video": str(generator_video),
            "generator_local_video": str(generator_video),
            "generator_media_mode": generator_media_mode,
            "media_role": media_role,
            "is_pruned": is_pruned,
            "benchmark_media": {
                "generator_video": str(generator_video),
                "judge_video": str(full_video),
                "answerability_video": str(full_video),
                "source_cache_video": str(source_video),
            },
        }
    )
    if is_pruned:
        kept_duration = round(
            sum(float(end) - float(start) for start, end in keep_intervals),
            3,
        )
        removed_duration = round(
            sum(float(end) - float(start) for start, end in remove_intervals),
            3,
        )
        result["temporal_pruning"] = {
            "method": (
                "speaker_two_anchor_remove_interval_union"
                if media_role == "speaker_pruned"
                else "anchor_pair_specific_pruning"
            ),
            "keep_intervals": list(keep_intervals),
            "remove_intervals": list(remove_intervals),
            "kept_duration_seconds": kept_duration,
            "removed_duration_seconds": removed_duration,
            "source_pair_keys": [edge.get("pair_key") for edge in source_edges],
        }
    return result


def materialize_six_user_role_structure(
    rows: list[dict[str, Any]],
    structure: dict[str, Any],
    *,
    output_dir: str | Path,
    start_seconds: float,
    duration_seconds: float,
    min_pruned_video_seconds: float,
    ffmpeg_binary: str,
) -> list[dict[str, Any]]:
    """按 speaker、anchors、additionals 顺序物化三段裁剪和三段完整视频。"""

    if len(rows) != 6:
        raise ValueError(f"six-user materialization requires 6 rows, got {len(rows)}")
    speaker_index = int(structure["speaker_index"])
    anchor_indices = [int(index) for index in structure.get("anchor_indices", [])]
    additional_indices = [int(index) for index in structure.get("additional_indices", [])]
    selected_edges = list(structure.get("selected_anchor_edges") or [])
    if len(anchor_indices) != 2 or len(additional_indices) != 3 or len(selected_edges) != 2:
        raise ValueError("six-user role structure must contain two anchors and three additionals")

    window_start = float(start_seconds)
    window_end = round(window_start + float(duration_seconds), 3)
    speaker_remove_rows = []
    for edge in selected_edges:
        _keep, remove = _pair_side_intervals(edge, speaker_index)
        speaker_remove_rows.extend(remove)
    speaker_remove = _merge_intervals(
        [
            (max(window_start, float(start)), min(window_end, float(end)))
            for start, end in speaker_remove_rows
            if min(window_end, float(end)) > max(window_start, float(start))
        ]
    )
    speaker_keep = _subtract_intervals((window_start, window_end), speaker_remove)
    speaker_kept_seconds = sum(end - start for start, end in speaker_keep)
    if speaker_kept_seconds < float(min_pruned_video_seconds):
        raise ValueError(
            "speaker pruned video is too short after merging two anchor-edge removals: "
            f"kept {speaker_kept_seconds:.3f}s, need {float(min_pruned_video_seconds):.3f}s"
        )

    role_dir = (
        Path(output_dir)
        / "six_user_role_structures"
        / f"candidate_{int(structure.get('candidate_rank') or 1):03d}"
    )
    clips = [
        _materialize_six_user_clip(
            rows[speaker_index]["clip"],
            media_role="speaker_pruned",
            position=0,
            output_dir=role_dir,
            keep_intervals=speaker_keep,
            remove_intervals=speaker_remove,
            source_edges=selected_edges,
            ffmpeg_binary=ffmpeg_binary,
        )
    ]

    for position, anchor_index in enumerate(anchor_indices, start=1):
        matching_edge = next(
            (
                edge
                for edge in selected_edges
                if anchor_index
                in (int(edge.get("left_index", -1)), int(edge.get("right_index", -1)))
            ),
            None,
        )
        if matching_edge is None:
            raise ValueError(f"anchor index {anchor_index} has no selected speaker edge")
        keep_intervals, remove_intervals = _pair_side_intervals(matching_edge, anchor_index)
        kept_seconds = sum(float(end) - float(start) for start, end in keep_intervals)
        if kept_seconds < float(min_pruned_video_seconds):
            raise ValueError(
                f"anchor {anchor_index} pruned video is too short: kept {kept_seconds:.3f}s, "
                f"need {float(min_pruned_video_seconds):.3f}s"
            )
        clips.append(
            _materialize_six_user_clip(
                rows[anchor_index]["clip"],
                media_role="anchor_provider_pruned",
                position=position,
                output_dir=role_dir,
                keep_intervals=keep_intervals,
                remove_intervals=remove_intervals,
                source_edges=[matching_edge],
                ffmpeg_binary=ffmpeg_binary,
            )
        )

    for position, additional_index in enumerate(additional_indices, start=3):
        clips.append(
            _materialize_six_user_clip(
                rows[additional_index]["clip"],
                media_role="additional_provider_full",
                position=position,
                output_dir=role_dir,
                keep_intervals=None,
                remove_intervals=[],
                source_edges=[],
                ffmpeg_binary=ffmpeg_binary,
            )
        )
    return clips


def materialize_six_user_consensus_candidate(
    rows: list[dict[str, Any]],
    consensus: dict[str, Any],
    *,
    output_dir: str | Path,
    ffmpeg_binary: str,
) -> list[dict[str, Any]]:
    """按 speaker、五个 provider 顺序物化六段 consensus-pruned 视频。"""

    if len(rows) != 6:
        raise ValueError(f"six-user consensus materialization requires 6 rows, got {len(rows)}")
    if not consensus.get("passed"):
        raise ValueError(
            "speaker consensus pruning did not pass: "
            f"speaker_index={consensus.get('speaker_index')} events={len(consensus.get('events') or [])}"
        )
    speaker_index = int(consensus["speaker_index"])
    ordered_indices = [speaker_index, *[index for index in range(6) if index != speaker_index]]
    diagnostics_by_video = {
        int(video["video_index"]): video for video in consensus.get("videos", [])
    }
    if set(diagnostics_by_video) != set(range(6)):
        raise ValueError("speaker consensus diagnostics must contain all 6 video indices")

    candidate_dir = Path(output_dir) / f"speaker_{speaker_index + 1:02d}"
    clips = []
    for position, source_index in enumerate(ordered_indices):
        diagnostics = diagnostics_by_video[source_index]
        if not diagnostics.get("passed", True):
            raise ValueError(
                "consensus-pruned video is too short: "
                f"speaker_index={speaker_index} video_index={source_index} "
                f"kept={diagnostics.get('kept_duration_seconds')}"
            )
        clip = _materialize_six_user_clip(
            rows[source_index]["clip"],
            media_role=(
                "speaker_reference_unpruned"
                if position == 0
                else "provider_similarity_pruned"
            ),
            position=position,
            output_dir=candidate_dir,
            keep_intervals=(
                None
                if position == 0
                else list(diagnostics.get("keep_intervals") or [])
            ),
            remove_intervals=list(diagnostics.get("remove_intervals") or []),
            source_edges=[],
            ffmpeg_binary=ffmpeg_binary,
        )
        clip.setdefault("temporal_pruning", {}).update(
            {
                "method": consensus.get("method"),
                "speaker_index": speaker_index,
                "source_video_index": source_index,
                "marked_cluster_indices": list(
                    diagnostics.get("marked_cluster_indices") or []
                ),
                "trigger_event_indices": list(
                    diagnostics.get("trigger_event_indices") or []
                ),
                "block_duration_seconds": consensus.get("block_duration_seconds"),
                "block_count": consensus.get("block_count"),
                "block_diagnostics": list(
                    diagnostics.get("block_diagnostics") or []
                ),
            }
        )
        clips.append(clip)
    return clips


def analyze_group_relative_similarity(
    group: dict[str, Any],
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    encoder: ImageEncoder,
    duration_seconds: float = 30.0,
    pruning_block_seconds: float = 30.0,
    sample_interval_seconds: float = 1.0,
    start_seconds: float = 0.0,
    selected_count: int = 2,
    pairs_per_group: int = 1,
    topk: int = 3,
    min_topk_sim: float = 0.65,
    min_mean_sim: float = 0.25,
    max_mean_sim: float = 0.90,
    high_similarity_interval_threshold: float = 0.82,
    pruning_clusters_per_video: int = 12,
    pruning_seconds_per_cluster: float = 2.5,
    pruning_time_weight: float = 0.1,
    pruning_temporal_unit_seconds: float = 30.0,
    pruning_max_iterations: int = 25,
    pruning_cross_gap_mode: str = "center",
    pruning_max_cross_gap_seconds: float = 10.0,
    temporal_neighborhood_seconds: float | None = None,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "reject",
    min_pruned_video_percent: float | None = None,
    max_pair_time_difference_seconds: float | None = None,
    random_pair_first: bool = True,
    rng: random.Random | None = None,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
) -> dict[str, Any]:
    """分析一个同步组，并按双用户或六用户合同物化候选。"""

    if selected_count not in {2, 6}:
        raise ValueError("selected_count must be 2 or 6")
    if pairs_per_group < 1:
        raise ValueError("pairs_per_group must be positive")
    rng = rng or random.Random()
    original_group_size = len(group.get("clips", []))
    sampled_source_clips = (
        _sample_group_clips_for_pair(group, selected_count=selected_count, rng=rng)
        if random_pair_first or selected_count == 6
        else sorted(group.get("clips", []), key=lambda item: str(item.get("agent_dir")))
    )
    if len(sampled_source_clips) < selected_count:
        raise ValueError(f"group needs at least {selected_count} clips")
    sampled_group = {**group, "clips": sampled_source_clips}

    rows = group_clip_frames(
        sampled_group,
        output_dir,
        cache_dir=cache_dir,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        start_seconds=start_seconds,
        ffmpeg_binary=ffmpeg_binary,
        download_media=download_media,
    )
    clip_embeddings = []
    frame_embeddings_by_clip = []
    for row in rows:
        frame_embeddings = encoder.encode([str(frame["path"]) for frame in row["frames"]])
        frame_embeddings_by_clip.append(frame_embeddings)
        clip_embeddings.append(mean_embedding(frame_embeddings))

    group_output_dir = Path(output_dir) / (
        _safe_filename_part(f"{group.get('day')}_{group.get('time_token')}")
        if selected_count == 6
        else stable_id(group.get("day"), group.get("time_token"))
    )
    if selected_count == 6:
        speaker_attempts = []
        speaker_candidates = []
        effective_min_pruned_video_percent = (
            20.0
            if min_pruned_video_percent is None
            else float(min_pruned_video_percent)
        )
        for speaker_index in range(6):
            consensus = clustered_six_user_zip_temporal_pruning(
                [row["frames"] for row in rows],
                frame_embeddings_by_clip,
                speaker_index=speaker_index,
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                seconds_per_cluster=pruning_seconds_per_cluster,
                time_weight=pruning_time_weight,
                temporal_unit_seconds=pruning_temporal_unit_seconds,
                max_iterations=pruning_max_iterations,
                high_similarity_threshold=high_similarity_interval_threshold,
                cross_gap_mode=pruning_cross_gap_mode,
                max_cross_gap_seconds=pruning_max_cross_gap_seconds,
                min_pruned_video_seconds=min_pruned_video_seconds,
                min_pruned_video_percent=effective_min_pruned_video_percent,
            )
            if not consensus.get("passed"):
                short_videos = [
                    {
                        "video_index": video.get("video_index"),
                        "kept_duration_seconds": video.get("kept_duration_seconds"),
                    }
                    for video in consensus.get("videos", [])
                    if not video.get("passed", False)
                ]
                failed_pairs = [
                    pair
                    for pair in consensus.get("pair_results", [])
                    if not pair.get("pruning", {}).get("passed", False)
                ]
                failure_reason = (
                    "zip_pair_no_accepted_cluster_match"
                    if any(
                        int(
                            pair.get("pruning", {}).get(
                                "high_similarity_representative_pair_count",
                                0,
                            )
                        )
                        == 0
                        for pair in failed_pairs
                    )
                    else "zip_pair_pruning_contract_failed"
                )
                speaker_attempts.append(
                    {
                        "speaker_index": speaker_index,
                        "speaker_user": rows[speaker_index]["clip"].get("agent_name"),
                        "status": "failed",
                        "failure_reason": failure_reason,
                        "short_videos": short_videos,
                        "consensus": consensus,
                    }
                )
                continue
            try:
                selected_clips = materialize_six_user_consensus_candidate(
                    rows,
                    consensus,
                    output_dir=group_output_dir / "six_user_speaker_consensus",
                    ffmpeg_binary=ffmpeg_binary,
                )
            except Exception as exc:
                speaker_attempts.append(
                    {
                        "speaker_index": speaker_index,
                        "speaker_user": rows[speaker_index]["clip"].get("agent_name"),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                        "consensus": consensus,
                    }
                )
                continue

            selection = {
                "method": "six_user_speaker_consensus",
                "pruning_method": consensus.get("method"),
                "speaker_index": speaker_index,
                "speaker_user": selected_clips[0].get("agent_name"),
                "provider_indices": [index for index in range(6) if index != speaker_index],
                "provider_users": [clip.get("agent_name") for clip in selected_clips[1:]],
                "selected_indices": [
                    speaker_index,
                    *[index for index in range(6) if index != speaker_index],
                ],
                "selected_agents": [clip.get("agent_dir") for clip in selected_clips],
                "selected_users": [clip.get("agent_name") for clip in selected_clips],
            }
            speaker_attempts.append(
                {
                    "speaker_index": speaker_index,
                    "speaker_user": rows[speaker_index]["clip"].get("agent_name"),
                    "status": "succeeded",
                    "candidate_index": len(speaker_candidates),
                    "consensus": consensus,
                }
            )
            speaker_candidates.append(
                {
                    "day": group.get("day"),
                    "time_token": group.get("time_token"),
                    "clip_clock": group.get("clip_clock"),
                    "model_id": encoder.model_id,
                    "window": {
                        "start_seconds": start_seconds,
                        "duration_seconds": duration_seconds,
                        "sample_interval_seconds": sample_interval_seconds,
                    },
                    "group_size": original_group_size,
                    "embedded_clip_count": len(rows),
                    "selection": selection,
                    "speaker_consensus_pruning": consensus,
                    "group_clips": [row["clip"] for row in rows],
                    "selected_clips": selected_clips,
                }
            )

        for candidate in speaker_candidates:
            candidate["speaker_attempts"] = speaker_attempts

        return {
            "day": group.get("day"),
            "time_token": group.get("time_token"),
            "clip_clock": group.get("clip_clock"),
            "model_id": encoder.model_id,
            "window": {
                "start_seconds": start_seconds,
                "duration_seconds": duration_seconds,
                "sample_interval_seconds": sample_interval_seconds,
            },
            "group_size": original_group_size,
            "embedded_clip_count": len(rows),
            "selection": {
                "method": "six_user_speaker_consensus_all_speakers",
                "selected_count": 6,
                "speaker_order": [0, 1, 2, 3, 4, 5],
                "original_group_size": original_group_size,
                "embedded_clip_count": len(rows),
                "sampled_source_agents": [row["clip"].get("agent_dir") for row in rows],
                "sampled_source_users": [row["clip"].get("agent_name") for row in rows],
                "pruning_method": (
                    f"zip_temporal_kmeans_{pruning_cross_gap_mode}_gate_pair_pruning_v1"
                ),
                "pruning_seconds_per_cluster": pruning_seconds_per_cluster,
                "pruning_time_weight": pruning_time_weight,
                "pruning_temporal_unit_seconds": pruning_temporal_unit_seconds,
                "pruning_max_iterations": pruning_max_iterations,
                "pruning_cross_gap_mode": pruning_cross_gap_mode,
                "pruning_max_cross_gap_seconds": pruning_max_cross_gap_seconds,
                "pruning_min_video_percent": effective_min_pruned_video_percent,
                "legacy_pruning_clusters_per_video": pruning_clusters_per_video,
                "legacy_pruning_block_seconds": pruning_block_seconds,
                "high_similarity_interval_threshold": high_similarity_interval_threshold,
                "min_pruned_video_seconds": min_pruned_video_seconds,
            },
            "speaker_attempts": speaker_attempts,
            "speaker_candidates": speaker_candidates,
            "group_clips": [row["clip"] for row in rows],
        }

    scoring = relative_group_scores(rows, clip_embeddings)
    pair_analysis = score_video_pairs(
        rows,
        frame_embeddings_by_clip,
        scoring,
        topk=topk,
        min_topk_sim=min_topk_sim,
        min_mean_sim=min_mean_sim,
        max_mean_sim=max_mean_sim,
        start_seconds=start_seconds,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        pruning_clusters_per_video=pruning_clusters_per_video,
        high_similarity_interval_threshold=high_similarity_interval_threshold,
        temporal_neighborhood_seconds=temporal_neighborhood_seconds,
        preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
        min_pruned_video_seconds=min_pruned_video_seconds,
        pruning_protection_mode=pruning_protection_mode,
        min_pruned_video_percent=min_pruned_video_percent,
        max_pair_time_difference_seconds=max_pair_time_difference_seconds,
    )
    surviving_pairs = pair_analysis["surviving_pairs"]
    role_selection: dict[str, Any] | None = None
    if selected_count == 2:
        if not surviving_pairs:
            diagnostics = compact_pair_rejection_summary(pair_analysis)
            raise ValueError(f"no video pairs survived the frame-matrix pair filters: {diagnostics}")
        sampled_pairs = rng.sample(surviving_pairs, min(pairs_per_group, len(surviving_pairs)))
        for sample_rank, pair in enumerate(sampled_pairs, 1):
            pair["sample_rank"] = sample_rank
        selected_pair = sampled_pairs[0]
        selected_indices = [int(selected_pair["left_index"]), int(selected_pair["right_index"])]
        selected_clips = selected_clips_for_pair_from_rows(
            rows,
            selected_pair,
            output_dir=group_output_dir,
            ffmpeg_binary=ffmpeg_binary,
        )
        selection_details = {
            "method": "random_synchronized_pair_then_cluster_prune",
            "selected_pair": selected_pair,
            "selected_pair_mean_sim": selected_pair["mean_sim"],
            "selected_pair_topk_sim": selected_pair["topk_sim"],
            "rationale": (
                "The sampler first randomly selects two videos from the synchronized group, then "
                "takes one frame per second only from those videos, clusters each selected video "
                "with CLIP embeddings, compares representative frames, removes uniform intervals "
                "around frames assigned to high-similarity clusters, and materializes paired "
                "original/pruned videos. Generators consume pruned videos; judges and "
                "answerability gates consume the original videos."
            ),
        }
    else:
        role_selection = build_six_user_role_structures(pair_analysis["pair_scores"], rng=rng)
        if not role_selection["role_structures"]:
            raise ValueError(
                "no six-user speaker has two kept anchor neighbors: "
                f"kept_degrees={role_selection['kept_degrees']}"
            )
        materialization_attempts = []
        selected_structure = None
        selected_clips = None
        for structure in role_selection["role_structures"]:
            try:
                candidate_clips = materialize_six_user_role_structure(
                    rows,
                    structure,
                    output_dir=group_output_dir,
                    start_seconds=start_seconds,
                    duration_seconds=duration_seconds,
                    min_pruned_video_seconds=min_pruned_video_seconds,
                    ffmpeg_binary=ffmpeg_binary,
                )
            except Exception as exc:
                materialization_attempts.append(
                    {
                        "candidate_rank": structure.get("candidate_rank"),
                        "speaker_index": structure.get("speaker_index"),
                        "anchor_indices": structure.get("anchor_indices"),
                        "status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            selected_structure = structure
            selected_clips = candidate_clips
            materialization_attempts.append(
                {
                    "candidate_rank": structure.get("candidate_rank"),
                    "speaker_index": structure.get("speaker_index"),
                    "anchor_indices": structure.get("anchor_indices"),
                    "status": "selected",
                }
            )
            break
        if selected_structure is None or selected_clips is None:
            raise ValueError(
                "all six-user role structures failed media materialization: "
                f"{materialization_attempts}"
            )
        selected_indices = [
            int(selected_structure["speaker_index"]),
            *[int(index) for index in selected_structure["anchor_indices"]],
            *[int(index) for index in selected_structure["additional_indices"]],
        ]
        sampled_pairs = list(selected_structure["selected_anchor_edges"])
        selection_details = {
            "method": "six_user_two_anchor_star",
            "selected_role_structure": selected_structure,
            "selected_anchor_edges": list(selected_structure["selected_anchor_edges"]),
            "role_materialization_attempts": materialization_attempts,
            "eligible_speaker_indices": role_selection["eligible_speaker_indices"],
            "kept_degrees": role_selection["kept_degrees"],
            "rationale": (
                "Six synchronized videos are scored across all 15 pair edges. A speaker with "
                "at least two kept neighbors is paired with two anchor providers. The speaker "
                "uses the union of both anchor-edge removals, each anchor uses pair-specific "
                "pruning, and three additional providers remain full for generation."
            ),
        }

    return {
        "day": group.get("day"),
        "time_token": group.get("time_token"),
        "clip_clock": group.get("clip_clock"),
        "model_id": encoder.model_id,
        "window": {
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "group_size": original_group_size,
        "embedded_clip_count": len(rows),
        "selection": {
            **selection_details,
            "selected_count": selected_count,
            "pairs_per_group": pairs_per_group,
            "random_pair_first": random_pair_first,
            "original_group_size": original_group_size,
            "embedded_clip_count": len(rows),
            "sampled_source_agents": [row["clip"].get("agent_dir") for row in rows],
            "sampled_source_users": [row["clip"].get("agent_name") for row in rows],
            "topk": topk,
            "min_topk_sim": min_topk_sim,
            "min_mean_sim": min_mean_sim,
            "max_mean_sim": max_mean_sim,
            "pruning_clusters_per_video": pruning_clusters_per_video,
            "high_similarity_interval_threshold": high_similarity_interval_threshold,
            "temporal_neighborhood_seconds": temporal_neighborhood_seconds,
            "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_percent": min_pruned_video_percent,
            "max_pair_time_difference_seconds": max_pair_time_difference_seconds,
            "selected_indices": selected_indices,
            "selected_agents": [clip.get("agent_dir") for clip in selected_clips],
            "selected_users": [clip.get("agent_name") for clip in selected_clips],
        },
        **scoring,
        **pair_analysis,
        "sampled_pairs": sampled_pairs,
        **({"six_user_role_selection": role_selection} if role_selection is not None else {}),
        "group_clips": [row["clip"] for row in rows],
        "selected_clips": selected_clips,
    }


def _write_csv(path: str | Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_review_bundle(group_result: dict[str, Any], review_root: str | Path) -> Path:
    """Copy all group videos and write comparison traces for manual inspection."""

    selection_method = group_result.get("selection", {}).get("method")
    bundle_id = (
        _safe_filename_part(
            f"{group_result.get('day')}_{group_result.get('time_token')}_six_user_consensus"
        )
        if selection_method == "six_user_speaker_consensus_all_speakers"
        else stable_id(group_result.get("day"), group_result.get("time_token"))
    )
    bundle_dir = Path(review_root) / bundle_id
    videos_dir = bundle_dir / "videos"
    traces_dir = bundle_dir / "comparison_traces"
    videos_dir.mkdir(parents=True, exist_ok=True)
    traces_dir.mkdir(parents=True, exist_ok=True)

    selected_agents = set(group_result.get("selection", {}).get("selected_agents", []))
    review_clips = []
    for index, clip in enumerate(group_result.get("group_clips", []), 1):
        agent_dir = clip.get("agent_dir")
        agent_name = clip.get("agent_name")
        selected = agent_dir in selected_agents
        local_video = clip.get("local_video")
        if not local_video or not Path(local_video).exists():
            raise FileNotFoundError(f"review video is unavailable for {agent_dir}: {local_video}")

        suffix = Path(local_video).suffix or ".mp4"
        flag = "SELECTED" if selected else "context"
        video_name = (
            f"{index:02d}_{_safe_filename_part(agent_dir)}_"
            f"{_safe_filename_part(agent_name)}_{flag}{suffix}"
        )
        review_video_path = videos_dir / video_name
        shutil.copy2(local_video, review_video_path)
        review_clips.append(
            {
                "index": index - 1,
                "agent_dir": agent_dir,
                "agent_name": agent_name,
                "selected": selected,
                "source_local_video": local_video,
                "review_video_path": str(review_video_path),
            }
        )

    trace = dict(group_result)
    trace["review_bundle"] = {
        "bundle_dir": str(bundle_dir),
        "videos_dir": str(videos_dir),
        "comparison_traces_dir": str(traces_dir),
        "clips": review_clips,
    }
    write_json(traces_dir / "comparison_trace.json", trace)

    score_fields = [
        "relative_rank",
        "index",
        "agent_dir",
        "agent_name",
        "mean_similarity_to_group",
        "centroid_similarity",
        "min_similarity_to_group",
        "max_similarity_to_group",
        "selected",
        "review_video_path",
    ]
    review_by_index = {int(row["index"]): row for row in review_clips}
    score_rows = []
    for score in group_result.get("clip_scores", []):
        row = dict(score)
        review_row = review_by_index.get(int(score["index"]), {})
        row["selected"] = review_row.get("selected", False)
        row["review_video_path"] = review_row.get("review_video_path")
        row.pop("frames", None)
        score_rows.append(row)
    _write_csv(traces_dir / "clip_scores_by_manifest_order.csv", score_rows, score_fields)
    _write_csv(
        traces_dir / "clip_scores_ranked_by_group_similarity.csv",
        sorted(score_rows, key=lambda item: int(item.get("relative_rank") or 0)),
        score_fields,
    )

    selection = group_result.get("selection", {})
    selected_pair_keys = {
        str(pair.get("pair_key"))
        for pair in selection.get("selected_anchor_edges", [])
        if pair.get("pair_key") is not None
    }
    selected_pair_key = selection.get("selected_pair", {}).get("pair_key")
    if selected_pair_key is not None:
        selected_pair_keys.add(str(selected_pair_key))
    pair_fields = [
        "trace_rank",
        "sample_rank",
        "pair_key",
        "status",
        "rejection_reason",
        "mean_sim",
        "topk_sim",
        "topk",
        "left_index",
        "left_agent_dir",
        "left_agent_name",
        "right_index",
        "right_agent_dir",
        "right_agent_name",
        "left_mean_similarity_to_group",
        "right_mean_similarity_to_group",
        "mean_clip_typicality_middle_score",
        "selected",
    ]
    pair_rows = []
    for pair in group_result.get("pair_scores", []):
        row = dict(pair)
        row["selected"] = str(row.get("pair_key")) in selected_pair_keys
        pair_rows.append(row)
    _write_csv(traces_dir / "pair_scores_ranked_for_qa.csv", pair_rows, pair_fields)
    _write_csv(
        traces_dir / "pair_scores_by_mean_sim.csv",
        sorted(pair_rows, key=lambda item: (float(item.get("mean_sim") or 0.0), str(item.get("pair_key")))),
        pair_fields,
    )
    _write_csv(
        traces_dir / "surviving_pairs_sample_pool.csv",
        [row for row in pair_rows if row.get("status") == "kept"],
        pair_fields,
    )

    labels = [
        f"{score.get('agent_dir')}:{score.get('agent_name')}"
        for score in group_result.get("clip_scores", [])
    ]
    matrix_rows = []
    for label, row in zip(labels, group_result.get("similarity_matrix", [])):
        matrix_rows.append({"clip": label, **{labels[index]: value for index, value in enumerate(row)}})
    _write_csv(traces_dir / "pairwise_similarity_matrix.csv", matrix_rows, ["clip", *labels])

    selected_users = ", ".join(selection.get("selected_users", []))
    selected_agents_text = ", ".join(selection.get("selected_agents", []))
    selected_relation_label = (
        ", ".join(sorted(selected_pair_keys))
        if selection.get("method") == "six_user_two_anchor_star"
        else str(selected_pair_key)
    )
    readme = (
        f"# {bundle_id}\n\n"
        f"- Day/time: {group_result.get('day')} {group_result.get('clip_clock')}\n"
        f"- Selected users: {selected_users} ({selected_agents_text})\n"
        f"- Selected pair edges: {selected_relation_label}\n"
        f"- Videos: `videos/` contains the sampled synchronized inputs; selected files end with `_SELECTED.mp4`.\n"
        f"- Pair trace: `comparison_traces/pair_scores_ranked_for_qa.csv` shows kept/rejected pair decisions.\n"
        f"- Sample pool: `comparison_traces/surviving_pairs_sample_pool.csv` shows all pairs eligible for random sampling.\n"
        f"- Clip trace: `comparison_traces/clip_scores_ranked_by_group_similarity.csv` shows per-video typicality.\n"
    )
    (bundle_dir / "README.md").write_text(readme, encoding="utf-8")
    return bundle_dir


def result_for_sampled_pair(
    group_result: dict[str, Any],
    pair: dict[str, Any],
    *,
    output_dir: str | Path,
    ffmpeg_binary: str,
) -> dict[str, Any]:
    """Return a group result view whose selected clips are one sampled pair."""

    selected_indices = [int(pair["left_index"]), int(pair["right_index"])]
    selected_clips = selected_clips_for_pair_from_group_result(
        group_result,
        pair,
        output_dir=output_dir,
        ffmpeg_binary=ffmpeg_binary,
    )
    result = dict(group_result)
    selection = dict(group_result.get("selection", {}))
    selection.update(
        {
            "selected_indices": selected_indices,
            "selected_agents": [clip.get("agent_dir") for clip in selected_clips],
            "selected_users": [clip.get("agent_name") for clip in selected_clips],
            "selected_pair": pair,
            "selected_pair_mean_sim": pair["mean_sim"],
            "selected_pair_topk_sim": pair["topk_sim"],
        }
    )
    result["selection"] = selection
    result["selected_clips"] = selected_clips
    return result


def build_candidate_packet(group_result: dict[str, Any]) -> dict[str, Any]:
    selected_clips = group_result["selected_clips"]
    required_users = [clip.get("agent_name") for clip in selected_clips]
    if len(required_users) == 6:
        selection = group_result.get("selection", {})
        media_roles = {
            str(clip.get("agent_name")): str(clip.get("media_role"))
            for clip in selected_clips
        }
        if selection.get("method") == "six_user_speaker_consensus":
            pruning_diagnostics = group_result.get("speaker_consensus_pruning", {})
            time_weight = pruning_diagnostics.get("time_weight", 0.1)
            gap_mode = pruning_diagnostics.get("cross_gap_mode", "center")
            max_gap_seconds = pruning_diagnostics.get("max_cross_gap_seconds", 10.0)
            packet_id = _safe_filename_part(
                "EGOLIFE6U_CONSENSUS_"
                f"{group_result.get('day')}_{group_result.get('time_token')}_"
                f"S{int(selection.get('speaker_index', 0)) + 1}"
            )
            return {
                "evidence_id": packet_id,
                "generation_group_id": (
                    f"{group_result.get('day')}::{group_result.get('time_token')}"
                ),
                "candidate_type": "six_user_speaker_consensus",
                "day": group_result.get("day"),
                "time_token": group_result.get("time_token"),
                "clip_clock": group_result.get("clip_clock"),
                "input_users": required_users,
                "required_users": required_users,
                "speaker_index": int(selection["speaker_index"]),
                "speaker_user": required_users[0],
                "provider_users": required_users[1:],
                "evidence_provider_user": required_users[1],
                "evidence_provider_users": required_users[1:],
                "media_roles": media_roles,
                "speaker_consensus_pruning": pruning_diagnostics,
                "speaker_attempts": list(group_result.get("speaker_attempts") or []),
                "requirement": (
                    "Six synchronized input videos are ordered as one speaker and five providers. "
                    f"ZIP temporal pair pruning evaluates and marks both sides with w={time_weight} "
                    f"and a {max_gap_seconds}-second {gap_mode}-gap gate. Generation still uses "
                    "the full speaker video and "
                    "the five temporally pruned provider videos. Groundedness uses all six full "
                    "originals. Answerability requires the "
                    "speaker-only condition to choose a valid wrong option and the all-six condition "
                    "to choose the declared correct option."
                ),
                "generator_media_mode": "speaker_full_five_provider_pruned_videos",
                "clips": selected_clips,
                "source_urls": {
                    "videos": [clip.get("video_url") for clip in selected_clips],
                    "gazes": [clip.get("gaze_url") for clip in selected_clips],
                    "overlays": [
                        clip.get("overlay_url")
                        for clip in selected_clips
                        if clip.get("overlay_url")
                    ],
                },
                "group_relative_clip_similarity": {
                    key: group_result[key]
                    for key in [
                        "model_id",
                        "window",
                        "group_size",
                        "selection",
                        "speaker_consensus_pruning",
                        "speaker_attempts",
                        "review_bundle",
                    ]
                    if key in group_result
                },
            }
        packet_id = stable_id(
            "EGOLIFE6U_TWO_ANCHOR_MIXED",
            group_result.get("day"),
            group_result.get("time_token"),
            *required_users,
            selection.get("selected_role_structure", {}).get("candidate_rank"),
        )
        return {
            "evidence_id": packet_id,
            "candidate_type": "six_user_two_anchor_mixed_media",
            "day": group_result.get("day"),
            "time_token": group_result.get("time_token"),
            "clip_clock": group_result.get("clip_clock"),
            "input_users": required_users,
            "required_users": required_users,
            "speaker_user": required_users[0],
            "anchor_provider_users": required_users[1:3],
            "additional_provider_users": required_users[3:6],
            "evidence_provider_user": required_users[1],
            "evidence_provider_users": required_users[1:3],
            "media_roles": media_roles,
            "selected_anchor_edges": list(selection.get("selected_anchor_edges") or []),
            "diagnostic_pair_edges": list(group_result.get("pair_scores") or []),
            "requirement": (
                "Six synchronized input videos are ordered as one speaker, two anchor providers, "
                "and three additional providers. Generation uses the pruned speaker and anchors "
                "plus three full additional videos. Groundedness uses all six full videos. "
                "Answerability requires the speaker-only condition to choose incorrectly and the "
                "all-six condition to choose correctly; providers need not all contribute."
            ),
            "generator_media_mode": "three_pruned_three_full_videos",
            "clips": selected_clips,
            "source_urls": {
                "videos": [clip.get("video_url") for clip in selected_clips],
                "gazes": [clip.get("gaze_url") for clip in selected_clips],
                "overlays": [
                    clip.get("overlay_url")
                    for clip in selected_clips
                    if clip.get("overlay_url")
                ],
            },
            "group_relative_clip_similarity": {
                key: group_result[key]
                for key in [
                    "model_id",
                    "window",
                    "group_size",
                    "selection",
                    "clip_scores",
                    "ranked_by_group_similarity",
                    "similarity_matrix",
                    "pair_filter",
                    "pair_scores",
                    "surviving_pairs",
                    "sampled_pairs",
                    "six_user_role_selection",
                    "review_bundle",
                ]
                if key in group_result
            },
        }

    packet_id = stable_id(
        "EGOLIFE2U_RANDOM_PAIR_CLIP_PRUNED",
        group_result.get("day"),
        group_result.get("time_token"),
        *[clip.get("agent_id") for clip in selected_clips],
        group_result.get("selection", {}).get("selected_pair", {}).get("pair_key"),
    )
    return {
        "evidence_id": packet_id,
        "candidate_type": "random_synchronized_pair_cluster_pruned_video",
        "day": group_result.get("day"),
        "time_token": group_result.get("time_token"),
        "clip_clock": group_result.get("clip_clock"),
        "required_users": required_users,
        "speaker_user": required_users[0] if required_users else None,
        "evidence_provider_user": required_users[1] if len(required_users) > 1 else None,
        "requirement": (
            "Sidecar candidate: a random time-synchronized pair was selected first, then each "
            "selected 30-second video was sampled once per second, clustered with CLIP embeddings, "
            "and compared through representative frames. This pair survived shared-anchor, "
            "unrelatedness, and redundancy filters, then frames assigned to high-similarity "
            "representative clusters were removed as uniform temporal intervals from both selected "
            "videos. Generation should use the pruned videos; judgers and "
            "answerability gates should use the original 30-second videos. Treat "
            "required_users[0] as the asker and required_users[1] as the evidence "
            "provider, then verify shared context, asymmetric evidence, asker-side "
            "insufficiency, and answerability."
        ),
        "generator_media_mode": "pruned_video",
        "clips": selected_clips,
        "source_urls": {
            "videos": [clip.get("video_url") for clip in selected_clips],
            "gazes": [clip.get("gaze_url") for clip in selected_clips],
            "overlays": [clip.get("overlay_url") for clip in selected_clips if clip.get("overlay_url")],
        },
        "group_relative_clip_similarity": {
            key: group_result[key]
            for key in [
                "model_id",
                "window",
                "group_size",
                "selection",
                "clip_scores",
                "ranked_by_group_similarity",
                "similarity_matrix",
                "pair_filter",
                "pair_scores",
                "surviving_pairs",
                "sampled_pairs",
                "review_bundle",
            ]
            if key in group_result
        },
    }

def mine_group_relative_clip_candidates(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    model_id: str = DEFAULT_CLIP_MODEL,
    target_count: int = 100,
    max_groups: int | None = None,
    min_group_size: int = 2,
    duration_seconds: float = 30.0,
    pruning_block_seconds: float = 30.0,
    sample_interval_seconds: float = 1.0,
    start_seconds: float = 0.0,
    selected_count: int = 2,
    pairs_per_group: int = 1,
    topk: int = 3,
    min_topk_sim: float = 0.65,
    min_mean_sim: float = 0.25,
    max_mean_sim: float = 0.90,
    high_similarity_interval_threshold: float = 0.82,
    pruning_clusters_per_video: int = 12,
    pruning_seconds_per_cluster: float = 2.5,
    pruning_time_weight: float = 0.1,
    pruning_temporal_unit_seconds: float = 30.0,
    pruning_max_iterations: int = 25,
    pruning_cross_gap_mode: str = "center",
    pruning_max_cross_gap_seconds: float = 10.0,
    temporal_neighborhood_seconds: float | None = None,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "reject",
    min_pruned_video_percent: float | None = None,
    max_pair_time_difference_seconds: float | None = None,
    random_pair_first: bool = True,
    random_seed: int | None = 42,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
    review_dir: str | Path | None = None,
    encoder: ImageEncoder | None = None,
    single_candidate_group: bool = False,
    target_generation_groups: int | None = None,
) -> list[dict[str, Any]]:
    """写出双用户 pair 或六用户双锚点 CLIP 裁剪候选。"""

    if selected_count not in {2, 6}:
        raise ValueError("selected_count must be 2 or 6")
    if target_generation_groups is not None and target_generation_groups < 1:
        raise ValueError("target_generation_groups must be positive when provided")
    manifest = read_json(manifest_path)
    rng = random.Random(random_seed) if random_seed is not None else random.Random()
    effective_min_group_size = max(int(min_group_size), selected_count)
    groups = [
        group
        for group in group_manifest_clips(
            manifest,
            evidence_duration_seconds=duration_seconds,
        )
        if len(group.get("clips", [])) >= effective_min_group_size
    ]
    rng.shuffle(groups)
    if max_groups is not None:
        groups = groups[:max_groups]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_root = Path(review_dir) if review_dir is not None else output_dir / "review_bundles"
    encoder = encoder or TransformersClipEncoder(model_id)

    candidates = []
    skipped = []
    generation_group_ids: list[str] = []
    for index, group in enumerate(groups):
        candidate_target_reached = len(candidates) >= target_count
        group_target_reached = (
            target_generation_groups is None
            or len(generation_group_ids) >= target_generation_groups
        )
        if candidate_target_reached and group_target_reached:
            break
        try:
            result = analyze_group_relative_similarity(
                group,
                output_dir=output_dir,
                cache_dir=cache_dir,
                encoder=encoder,
                duration_seconds=duration_seconds,
                pruning_block_seconds=pruning_block_seconds,
                sample_interval_seconds=sample_interval_seconds,
                start_seconds=start_seconds,
                selected_count=selected_count,
                pairs_per_group=pairs_per_group,
                topk=topk,
                min_topk_sim=min_topk_sim,
                min_mean_sim=min_mean_sim,
                max_mean_sim=max_mean_sim,
                high_similarity_interval_threshold=high_similarity_interval_threshold,
                pruning_clusters_per_video=pruning_clusters_per_video,
                pruning_seconds_per_cluster=pruning_seconds_per_cluster,
                pruning_time_weight=pruning_time_weight,
                pruning_temporal_unit_seconds=pruning_temporal_unit_seconds,
                pruning_max_iterations=pruning_max_iterations,
                pruning_cross_gap_mode=pruning_cross_gap_mode,
                pruning_max_cross_gap_seconds=pruning_max_cross_gap_seconds,
                temporal_neighborhood_seconds=temporal_neighborhood_seconds,
                preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
                min_pruned_video_seconds=min_pruned_video_seconds,
                pruning_protection_mode=pruning_protection_mode,
                min_pruned_video_percent=min_pruned_video_percent,
                max_pair_time_difference_seconds=max_pair_time_difference_seconds,
                random_pair_first=random_pair_first,
                rng=rng,
                ffmpeg_binary=ffmpeg_binary,
                download_media=download_media,
            )
        except Exception as exc:
            skipped.append(
                {
                    "index": index,
                    "day": group.get("day"),
                    "time_token": group.get("time_token"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        result_stem = (
            _safe_filename_part(f"{group.get('day')}_{group.get('time_token')}")
            if selected_count == 6
            else stable_id(group.get("day"), group.get("time_token"))
        )
        result_path = output_dir / f"{result_stem}_group_relative_clip.json"
        bundle_dir = write_review_bundle(result, review_root)
        result["review_bundle"] = str(bundle_dir)
        write_json(result_path, result)
        if selected_count == 6:
            if not result.get("speaker_candidates"):
                skipped.append(
                    {
                        "index": index,
                        "day": group.get("day"),
                        "time_token": group.get("time_token"),
                        "error": "no speaker consensus candidate passed",
                        "speaker_attempts": result.get("speaker_attempts", []),
                    }
                )
                continue
            for candidate_result in result["speaker_candidates"]:
                candidate_result["review_bundle"] = str(bundle_dir)
                packet = build_candidate_packet(candidate_result)
                packet["group_relative_clip_similarity"]["result_path"] = str(result_path)
                candidates.append(packet)
            generation_group_id = (
                f"{group.get('day')}::{group.get('time_token')}"
            )
            if generation_group_id not in generation_group_ids:
                generation_group_ids.append(generation_group_id)
            if single_candidate_group:
                break
        else:
            for pair in result.get("sampled_pairs", []):
                packet_result = result_for_sampled_pair(
                    result,
                    pair,
                    output_dir=output_dir / stable_id(group.get("day"), group.get("time_token")),
                    ffmpeg_binary=ffmpeg_binary,
                )
                packet = build_candidate_packet(packet_result)
                packet["group_relative_clip_similarity"]["result_path"] = str(result_path)
                candidates.append(packet)
                if len(candidates) >= target_count:
                    break

    write_jsonl(output_path, candidates)
    write_json(
        output_dir / "group_relative_clip_summary.json",
        {
            "manifest_path": str(manifest_path),
            "output_path": str(output_path),
            "review_dir": str(review_root),
            "group_count_considered": len(groups),
            "candidate_count": len(candidates),
            "generation_group_count": len(generation_group_ids),
            "generation_group_ids": generation_group_ids,
            "skipped_count": len(skipped),
            "skipped": skipped,
            "settings": {
                "model_id": encoder.model_id,
                "target_count": target_count,
                "max_groups": max_groups,
                "min_group_size": min_group_size,
                "effective_min_group_size": effective_min_group_size,
                "group_order": "randomized_before_max_groups",
                "duration_seconds": duration_seconds,
                "pruning_block_seconds": pruning_block_seconds,
                "sample_interval_seconds": sample_interval_seconds,
                "start_seconds": start_seconds,
                "selected_count": selected_count,
                "pairs_per_group": pairs_per_group,
                "topk": topk,
                "min_topk_sim": min_topk_sim,
                "min_mean_sim": min_mean_sim,
                "max_mean_sim": max_mean_sim,
                "high_similarity_interval_threshold": high_similarity_interval_threshold,
                "pruning_clusters_per_video": pruning_clusters_per_video,
                "pruning_seconds_per_cluster": pruning_seconds_per_cluster,
                "pruning_time_weight": pruning_time_weight,
                "pruning_temporal_unit_seconds": pruning_temporal_unit_seconds,
                "pruning_max_iterations": pruning_max_iterations,
                "pruning_cross_gap_mode": pruning_cross_gap_mode,
                "pruning_max_cross_gap_seconds": pruning_max_cross_gap_seconds,
                "temporal_neighborhood_seconds": temporal_neighborhood_seconds,
                "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
                "min_pruned_video_seconds": min_pruned_video_seconds,
                "pruning_protection_mode": pruning_protection_mode,
                "min_pruned_video_percent": min_pruned_video_percent,
                "max_pair_time_difference_seconds": max_pair_time_difference_seconds,
                "random_pair_first": random_pair_first,
                "random_seed": random_seed,
                "download_media": download_media,
                "review_dir": str(review_root),
                "single_candidate_group": single_candidate_group,
                "target_generation_groups": target_generation_groups,
            },
        },
    )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="CLIP-pruned synchronized candidate sampler for two or six input videos"
    )
    parser.add_argument("--manifest", required=True, help="Input EgoLife manifest JSON")
    parser.add_argument("--output", required=True, help="Output candidate JSONL")
    parser.add_argument("--output-dir", required=True, help="Directory for frame samples and diagnostics")
    parser.add_argument(
        "--review-dir",
        help="Separate human-review folder for selected pair videos and comparison traces",
    )
    parser.add_argument("--cache-dir", required=True, help="Local video cache root")
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    parser.add_argument(
        "--pruning-block-seconds",
        type=float,
        default=30.0,
        help="Legacy compatibility only; six-user ZIP pruning uses the full duration",
    )
    parser.add_argument(
        "--single-candidate-group",
        action="store_true",
        help="Stop after the first synchronized group that yields any speaker candidates",
    )
    parser.add_argument(
        "--target-generation-groups",
        type=int,
        help=(
            "Continue mining until this many distinct synchronized groups have yielded "
            "speaker candidates, even when --target-count was reached earlier"
        ),
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument(
        "--selected-count",
        type=int,
        choices=[2, 6],
        default=2,
        help="Use 2 for the legacy pair path or 6 for one speaker, two anchors, and three additionals",
    )
    parser.add_argument("--pairs-per-group", type=int, default=1)
    parser.add_argument("--topk", type=int, default=3, help="Number of strongest frame matches averaged into topk_sim")
    parser.add_argument(
        "--min-topk-sim",
        type=float,
        default=0.65,
        help="Reject pairs whose strongest shared-anchor score is below this value",
    )
    parser.add_argument(
        "--min-mean-sim",
        type=float,
        default=0.25,
        help="Reject pairs whose representative-similarity mean is below this value",
    )
    parser.add_argument(
        "--max-mean-sim",
        type=float,
        default=0.90,
        help="Reject pairs whose representative-similarity mean is above this value",
    )
    parser.add_argument(
        "--high-similarity-interval-threshold",
        type=float,
        default=0.82,
        help="Remove clusters whose representative frame similarities reach this value",
    )
    parser.add_argument(
        "--pruning-clusters-per-video",
        type=int,
        default=12,
        help="Legacy pair-path cluster count; six-user ZIP pruning uses seconds per cluster",
    )
    parser.add_argument(
        "--pruning-seconds-per-cluster",
        type=float,
        default=2.5,
        help="Six-user ZIP pruning cluster density over the full video duration",
    )
    parser.add_argument("--pruning-time-weight", type=float, default=0.1)
    parser.add_argument(
        "--pruning-temporal-unit-seconds",
        type=float,
        default=30.0,
    )
    parser.add_argument("--pruning-max-iterations", type=int, default=25)
    parser.add_argument(
        "--pruning-cross-gap-mode",
        choices=["center", "interval"],
        default="center",
    )
    parser.add_argument(
        "--pruning-max-cross-gap-seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--temporal-neighborhood-seconds",
        type=float,
        help="Deprecated for cluster pruning; retained for compatibility with older runs",
    )
    parser.add_argument(
        "--preserve-shared-anchor-seconds",
        type=float,
        default=0.0,
        help="Optionally keep this many seconds around the strongest high-similarity representative pair",
    )
    parser.add_argument(
        "--min-pruned-video-seconds",
        type=float,
        default=8.0,
        help="Minimum retained video seconds for reject mode or min_seconds protection mode",
    )
    parser.add_argument(
        "--pruning-protection-mode",
        choices=["reject", "min_seconds", "min_percent"],
        default="reject",
        help=(
            "reject keeps legacy behavior; min_seconds restores least-similar high-threshold "
            "sampled-frame intervals until --min-pruned-video-seconds remain; min_percent uses "
            "--min-pruned-video-percent instead"
        ),
    )
    parser.add_argument(
        "--min-pruned-video-percent",
        type=float,
        help=(
            "Minimum retained percentage for six-user ZIP pair pruning and for the "
            "legacy path when --pruning-protection-mode=min_percent"
        ),
    )
    parser.add_argument(
        "--max-pair-time-difference-seconds",
        type=float,
        help=(
            "Only prune high-similarity centroid pairs whose timestamps differ by at most "
            "this many seconds; omit for timestamp-agnostic pruning"
        ),
    )
    parser.add_argument(
        "--compare-all-pairs",
        action="store_true",
        help="Embed every video in each synchronized group and compare all pairs; slower than the default random-pair-first path",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--download-media", action="store_true")
    args = parser.parse_args(argv)

    candidates = mine_group_relative_clip_candidates(
        manifest_path=args.manifest,
        output_path=args.output,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        target_count=args.target_count,
        max_groups=args.max_groups,
        min_group_size=args.min_group_size,
        duration_seconds=args.duration_seconds,
        pruning_block_seconds=args.pruning_block_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        start_seconds=args.start_seconds,
        selected_count=args.selected_count,
        pairs_per_group=args.pairs_per_group,
        topk=args.topk,
        min_topk_sim=args.min_topk_sim,
        min_mean_sim=args.min_mean_sim,
        max_mean_sim=args.max_mean_sim,
        high_similarity_interval_threshold=args.high_similarity_interval_threshold,
        pruning_clusters_per_video=args.pruning_clusters_per_video,
        pruning_seconds_per_cluster=args.pruning_seconds_per_cluster,
        pruning_time_weight=args.pruning_time_weight,
        pruning_temporal_unit_seconds=args.pruning_temporal_unit_seconds,
        pruning_max_iterations=args.pruning_max_iterations,
        pruning_cross_gap_mode=args.pruning_cross_gap_mode,
        pruning_max_cross_gap_seconds=args.pruning_max_cross_gap_seconds,
        temporal_neighborhood_seconds=args.temporal_neighborhood_seconds,
        preserve_shared_anchor_seconds=args.preserve_shared_anchor_seconds,
        min_pruned_video_seconds=args.min_pruned_video_seconds,
        pruning_protection_mode=args.pruning_protection_mode,
        min_pruned_video_percent=args.min_pruned_video_percent,
        max_pair_time_difference_seconds=args.max_pair_time_difference_seconds,
        random_pair_first=not args.compare_all_pairs,
        random_seed=args.random_seed,
        ffmpeg_binary=args.ffmpeg_binary,
        download_media=args.download_media,
        review_dir=args.review_dir,
        single_candidate_group=args.single_candidate_group,
        target_generation_groups=args.target_generation_groups,
    )
    print(f"wrote {len(candidates)} CLIP-pruned candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
