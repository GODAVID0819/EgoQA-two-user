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
    from .evidence import group_manifest_clips, local_cache_path
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
    from egolife_two_user_qa.evidence import group_manifest_clips, local_cache_path
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
) -> dict[int, dict[str, Any]]:
    """Return each sampled frame's best cross-video match from a similarity matrix."""

    if side == "left":
        return {
            left_index: {
                "best_match_index": int(max(enumerate(row), key=lambda item: item[1])[0]),
                "best_match_similarity": float(max(row)),
            }
            for left_index, row in enumerate(matrix)
            if row
        }
    if side == "right":
        if not matrix:
            return {}
        width = len(matrix[0])
        return {
            right_index: {
                "best_match_index": int(max(
                    ((left_index, matrix[left_index][right_index]) for left_index in range(len(matrix))),
                    key=lambda item: item[1],
                )[0]),
                "best_match_similarity": float(max(matrix[left_index][right_index] for left_index in range(len(matrix)))),
            }
            for right_index in range(width)
        }
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
) -> dict[str, Any]:
    """Cluster one video's sampled frame embeddings and expose medoid frames."""

    if len(frames) != len(embeddings):
        raise ValueError("frame and embedding counts must match")
    if not frames:
        raise ValueError("cannot cluster an empty frame list")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")

    labels, medoids = cluster_embedding_medoids(embeddings, cluster_count)
    representatives = []
    representative_embeddings = []
    for cluster_index, frame_index in enumerate(medoids):
        member_indices = [
            index
            for index, label in enumerate(labels)
            if int(label) == int(cluster_index)
        ]
        frame = frames[frame_index]
        representatives.append(
            {
                "cluster_index": int(cluster_index),
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
        "labels": [int(label) for label in labels],
        "representatives": representatives,
        "representative_embeddings": representative_embeddings,
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
    )
    right_clusters = clustered_frame_representatives(
        right_frames,
        right_embeddings,
        cluster_count=cluster_count,
    )
    matrix = frame_similarity_matrix(
        left_clusters["representative_embeddings"],
        right_clusters["representative_embeddings"],
    )

    high_pairs = []
    left_marked_clusters: set[int] = set()
    right_marked_clusters: set[int] = set()
    for left_cluster_index, row in enumerate(matrix):
        for right_cluster_index, similarity in enumerate(row):
            if float(similarity) < high_similarity_threshold:
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
        _side_best_frame_matches(full_frame_matrix, side="left"),
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
        _side_best_frame_matches(full_frame_matrix, side="right"),
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
        local_video = _resolve_local_video(clip, cache_dir=cache_dir, download_media=download_media)
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


def analyze_group_relative_similarity(
    group: dict[str, Any],
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    encoder: ImageEncoder,
    duration_seconds: float = 30.0,
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
    temporal_neighborhood_seconds: float | None = None,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "reject",
    min_pruned_video_percent: float | None = None,
    random_pair_first: bool = True,
    rng: random.Random | None = None,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
) -> dict[str, Any]:
    """Analyze one synchronized group after optionally sampling a two-video pair first."""

    if selected_count != 2:
        raise ValueError("pair-ranking mode currently selects exactly two clips")
    if pairs_per_group < 1:
        raise ValueError("pairs_per_group must be positive")
    rng = rng or random.Random()
    original_group_size = len(group.get("clips", []))
    sampled_source_clips = (
        _sample_group_clips_for_pair(group, selected_count=selected_count, rng=rng)
        if random_pair_first
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
    )
    surviving_pairs = pair_analysis["surviving_pairs"]
    if not surviving_pairs:
        diagnostics = compact_pair_rejection_summary(pair_analysis)
        raise ValueError(f"no video pairs survived the frame-matrix pair filters: {diagnostics}")
    sampled_pairs = rng.sample(surviving_pairs, min(pairs_per_group, len(surviving_pairs)))
    for sample_rank, pair in enumerate(sampled_pairs, 1):
        pair["sample_rank"] = sample_rank
    selected_pair = sampled_pairs[0]
    selected_indices = [int(selected_pair["left_index"]), int(selected_pair["right_index"])]
    group_output_dir = Path(output_dir) / stable_id(group.get("day"), group.get("time_token"))
    selected_clips = selected_clips_for_pair_from_rows(
        rows,
        selected_pair,
        output_dir=group_output_dir,
        ffmpeg_binary=ffmpeg_binary,
    )

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
            "method": "random_synchronized_pair_then_cluster_prune",
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
            "selected_indices": selected_indices,
            "selected_agents": [clip.get("agent_dir") for clip in selected_clips],
            "selected_users": [clip.get("agent_name") for clip in selected_clips],
            "selected_pair": selected_pair,
            "selected_pair_mean_sim": selected_pair["mean_sim"],
            "selected_pair_topk_sim": selected_pair["topk_sim"],
            "rationale": (
                "The sampler first randomly selects two videos from the synchronized group, then "
                "takes one frame per second only from those videos, clusters each selected video "
                "with CLIP embeddings, compares representative frames, removes uniform intervals "
                "around frames assigned to high-similarity clusters, and materializes paired "
                "original/pruned videos. Generators consume pruned videos; judges and "
                "answerability gates consume the original 30-second videos."
            ),
        },
        **scoring,
        **pair_analysis,
        "sampled_pairs": sampled_pairs,
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

    bundle_id = stable_id(group_result.get("day"), group_result.get("time_token"))
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

    selected_pair_key = group_result.get("selection", {}).get("selected_pair", {}).get("pair_key")
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
        row["selected"] = row.get("pair_key") == selected_pair_key
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

    selected_users = ", ".join(group_result.get("selection", {}).get("selected_users", []))
    selected_agents_text = ", ".join(group_result.get("selection", {}).get("selected_agents", []))
    readme = (
        f"# {bundle_id}\n\n"
        f"- Day/time: {group_result.get('day')} {group_result.get('clip_clock')}\n"
        f"- Selected pair: {selected_users} ({selected_agents_text})\n"
        f"- Selected mean_sim: {group_result.get('selection', {}).get('selected_pair_mean_sim')}\n"
        f"- Selected topk_sim: {group_result.get('selection', {}).get('selected_pair_topk_sim')}\n"
        f"- Videos: `videos/` contains the sampled synchronized pair; selected files end with `_SELECTED.mp4`.\n"
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
    temporal_neighborhood_seconds: float | None = None,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "reject",
    min_pruned_video_percent: float | None = None,
    random_pair_first: bool = True,
    random_seed: int | None = 42,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
    review_dir: str | Path | None = None,
    encoder: ImageEncoder | None = None,
) -> list[dict[str, Any]]:
    """Write CLIP-pruned candidates from random synchronized two-video pairs."""

    manifest = read_json(manifest_path)
    rng = random.Random(random_seed) if random_seed is not None else random.Random()
    groups = [
        group
        for group in group_manifest_clips(manifest)
        if len(group.get("clips", [])) >= min_group_size
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
    for index, group in enumerate(groups):
        if len(candidates) >= target_count:
            break
        try:
            result = analyze_group_relative_similarity(
                group,
                output_dir=output_dir,
                cache_dir=cache_dir,
                encoder=encoder,
                duration_seconds=duration_seconds,
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
                temporal_neighborhood_seconds=temporal_neighborhood_seconds,
                preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
                min_pruned_video_seconds=min_pruned_video_seconds,
                pruning_protection_mode=pruning_protection_mode,
                min_pruned_video_percent=min_pruned_video_percent,
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
        result_path = output_dir / f"{stable_id(group.get('day'), group.get('time_token'))}_group_relative_clip.json"
        bundle_dir = write_review_bundle(result, review_root)
        result["review_bundle"] = str(bundle_dir)
        write_json(result_path, result)
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
            "skipped_count": len(skipped),
            "skipped": skipped,
            "settings": {
                "model_id": encoder.model_id,
                "target_count": target_count,
                "max_groups": max_groups,
                "min_group_size": min_group_size,
                "group_order": "randomized_before_max_groups",
                "duration_seconds": duration_seconds,
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
                "temporal_neighborhood_seconds": temporal_neighborhood_seconds,
                "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
                "min_pruned_video_seconds": min_pruned_video_seconds,
                "pruning_protection_mode": pruning_protection_mode,
                "min_pruned_video_percent": min_pruned_video_percent,
                "random_pair_first": random_pair_first,
                "random_seed": random_seed,
                "download_media": download_media,
                "review_dir": str(review_root),
            },
        },
    )
    return candidates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sidecar sampler that CLIP-prunes random synchronized two-video pairs"
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
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--selected-count", type=int, default=2)
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
        help="Cluster each video's sampled frames into this many CLIP medoid groups before pruning",
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
        help="Minimum retained percentage of the input window when --pruning-protection-mode=min_percent",
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
        temporal_neighborhood_seconds=args.temporal_neighborhood_seconds,
        preserve_shared_anchor_seconds=args.preserve_shared_anchor_seconds,
        min_pruned_video_seconds=args.min_pruned_video_seconds,
        pruning_protection_mode=args.pruning_protection_mode,
        min_pruned_video_percent=args.min_pruned_video_percent,
        random_pair_first=not args.compare_all_pairs,
        random_seed=args.random_seed,
        ffmpeg_binary=args.ffmpeg_binary,
        download_media=args.download_media,
        review_dir=args.review_dir,
    )
    print(f"wrote {len(candidates)} random-pair CLIP-pruned candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
