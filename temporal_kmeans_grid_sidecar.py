"""Time-aware spherical K-means and a fixed-cohort pruning grid sidecar.

This module is intentionally additive.  It does not patch or replace the
production clustering/pruning functions.  A time weight of zero reproduces the
existing deterministic cosine K-means objective; positive weights add a soft
quadratic cost for assigning temporally distant frames to the same cluster.
Cross-video representative matching remains pure CLIP cosine similarity so the
experiment isolates the effect of time inside each video's clustering stage.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import random
import shutil
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .clip_gap_demo import (
    DEFAULT_CLIP_MODEL,
    ImageEncoder,
    TransformersClipEncoder,
    cluster_embedding_medoids,
    sample_short_video,
)
from .evidence import SOURCE_CLIP_DURATION_SECONDS, group_manifest_clips
from .group_relative_clip_sampling import (
    _apply_pruning_duration_protection,
    _protected_duration_target_seconds,
    _resolve_local_video,
    _side_best_frame_matches,
)
from .io_utils import iter_jsonl, read_json, stable_id, write_json, write_jsonl
from .paired_evidence_pruning import encode_paths_in_batches


DEFAULT_DURATIONS_SECONDS = (30.0, 180.0, 360.0, 600.0)
DEFAULT_TIME_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
# Keep the current 30-second K=12 intensity at every duration:
# 30/2.5=12, 180/2.5=72, 360/2.5=144, 600/2.5=240.
DEFAULT_SECONDS_PER_CLUSTER = (2.5,)
DEFAULT_SIMILARITY_THRESHOLDS = (0.82,)
DEFAULT_PAIR_COUNT = 50
DEFAULT_RANDOM_SEED = 20260825
DEFAULT_TEMPORAL_UNIT_SECONDS = 30.0

CLUSTER_DISTANCE_FORMULA = (
    "2 * (1 - clip_cosine_similarity) + "
    "time_weight * ((frame_time - center_time) / temporal_unit_seconds) ** 2"
)

METRIC_FIELDS = (
    "pair_id",
    "day",
    "time_token",
    "left_agent",
    "right_agent",
    "duration_seconds",
    "sample_interval_seconds",
    "seconds_per_cluster",
    "k",
    "time_weight",
    "temporal_unit_seconds",
    "high_similarity_threshold",
    "left_cluster_count",
    "right_cluster_count",
    "mean_cluster_span_seconds",
    "p95_cluster_span_seconds",
    "max_cluster_span_seconds",
    "mean_member_medoid_gap_seconds",
    "p95_member_medoid_gap_seconds",
    "member_gap_gt_unit_fraction",
    "member_gap_gt_quarter_duration_fraction",
    "mean_member_medoid_visual_similarity",
    "pruned_member_gap_gt_unit_fraction",
    "pruned_member_gap_gt_quarter_duration_fraction",
    "p95_pruned_member_medoid_gap_seconds",
    "high_similarity_representative_pair_count",
    "mean_trigger_time_difference_seconds",
    "max_trigger_time_difference_seconds",
    "left_marked_frame_count",
    "right_marked_frame_count",
    "left_removed_percent",
    "right_removed_percent",
    "mean_removed_percent",
    "left_keep_segment_count",
    "right_keep_segment_count",
    "no_removal",
    "passed",
    "diagnostics_path",
)

AGGREGATE_FIELDS = (
    "duration_seconds",
    "seconds_per_cluster",
    "k",
    "time_weight",
    "temporal_unit_seconds",
    "high_similarity_threshold",
    "pair_count",
    "pass_rate",
    "no_removal_rate",
    "mean_cluster_span_seconds",
    "mean_p95_cluster_span_seconds",
    "mean_max_cluster_span_seconds",
    "mean_member_medoid_gap_seconds",
    "mean_p95_member_medoid_gap_seconds",
    "mean_member_gap_gt_unit_fraction",
    "mean_member_gap_gt_quarter_duration_fraction",
    "mean_member_medoid_visual_similarity",
    "mean_pruned_member_gap_gt_unit_fraction",
    "mean_pruned_member_gap_gt_quarter_duration_fraction",
    "mean_removed_percent",
    "mean_keep_segment_count",
    "mean_trigger_pair_count",
    "member_gap_reduction_vs_w0",
    "quarter_duration_gap_reduction_vs_w0",
    "visual_similarity_delta_vs_w0",
    "pruned_gap_reduction_vs_w0",
    "pruned_quarter_duration_gap_reduction_vs_w0",
    "removed_percent_delta_vs_w0",
    "temporal_visual_pareto",
)


def parse_float_grid(
    value: str | Iterable[float],
    *,
    name: str,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> list[float]:
    """Parse, validate, and de-duplicate a comma-separated float grid."""

    raw_values: Iterable[Any]
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = value
    parsed: list[float] = []
    for raw in raw_values:
        try:
            number = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid {name} value: {raw!r}") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name} values must be finite")
        if strictly_positive and number <= 0:
            raise ValueError(f"{name} values must be positive")
        if minimum is not None and number < minimum:
            raise ValueError(f"{name} values must be at least {minimum}")
        if maximum is not None and number > maximum:
            raise ValueError(f"{name} values must be at most {maximum}")
        if number not in parsed:
            parsed.append(number)
    if not parsed:
        raise ValueError(f"at least one {name} value is required")
    return parsed


def _normalized_embedding_matrix(embeddings: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("embedding vectors must have positive norm")
    return matrix / norms


def combined_cluster_distance(
    clip_cosine_similarity: float,
    time_difference_seconds: float,
    *,
    time_weight: float,
    temporal_unit_seconds: float,
) -> float:
    """Return the visual-plus-temporal distance used by the sidecar."""

    if not math.isfinite(time_weight) or time_weight < 0:
        raise ValueError("time_weight must be finite and non-negative")
    if not math.isfinite(temporal_unit_seconds) or temporal_unit_seconds <= 0:
        raise ValueError("temporal_unit_seconds must be finite and positive")
    normalized_gap = float(time_difference_seconds) / float(temporal_unit_seconds)
    return 2.0 * (1.0 - float(clip_cosine_similarity)) + time_weight * normalized_gap**2


def temporal_spherical_kmeans_medoids(
    embeddings: Sequence[Sequence[float]],
    timestamps_seconds: Sequence[float],
    cluster_count: int,
    *,
    time_weight: float,
    temporal_unit_seconds: float = DEFAULT_TEMPORAL_UNIT_SECONDS,
    max_iterations: int = 25,
) -> tuple[list[int], list[int], dict[str, Any]]:
    """Cluster normalized CLIP vectors with a soft quadratic time dimension.

    Assignment minimizes ``CLUSTER_DISTANCE_FORMULA``.  Visual centers are
    normalized means (spherical K-means) and temporal centers are arithmetic
    means.  At ``time_weight=0`` the initialization, assignment, center update,
    and medoid rule match ``cluster_embedding_medoids``.
    """

    vectors = _normalized_embedding_matrix(embeddings)
    times = np.asarray(timestamps_seconds, dtype=np.float64)
    if times.ndim != 1 or len(times) != len(vectors):
        raise ValueError("timestamps must align one-to-one with embeddings")
    if not np.isfinite(times).all():
        raise ValueError("timestamps must contain only finite values")
    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if not math.isfinite(time_weight) or time_weight < 0:
        raise ValueError("time_weight must be finite and non-negative")
    if not math.isfinite(temporal_unit_seconds) or temporal_unit_seconds <= 0:
        raise ValueError("temporal_unit_seconds must be finite and positive")

    # Subtracting the first timestamp improves numeric conditioning while
    # preserving every temporal difference and therefore every assignment.
    scaled_times = (times - times[0]) / float(temporal_unit_seconds)
    k = max(1, min(int(cluster_count), len(vectors)))

    center_indices = [0]
    min_distances = np.full(len(vectors), np.inf, dtype=np.float64)
    while len(center_indices) < k:
        latest = center_indices[-1]
        visual_distance = 2.0 * (1.0 - vectors @ vectors[latest])
        temporal_distance = time_weight * (scaled_times - scaled_times[latest]) ** 2
        min_distances = np.minimum(min_distances, visual_distance + temporal_distance)
        candidate_distances = min_distances.copy()
        candidate_distances[np.asarray(center_indices, dtype=np.int64)] = -np.inf
        next_index = int(np.argmax(candidate_distances))
        center_indices.append(next_index)

    visual_centers = vectors[np.asarray(center_indices, dtype=np.int64)].copy()
    temporal_centers = scaled_times[np.asarray(center_indices, dtype=np.int64)].copy()
    labels = np.zeros(len(vectors), dtype=np.int64)
    iterations_run = 0
    for iteration in range(max_iterations):
        visual_similarity = vectors @ visual_centers.T
        temporal_penalty = 0.5 * time_weight * (
            scaled_times[:, None] - temporal_centers[None, :]
        ) ** 2
        new_labels = np.argmax(visual_similarity - temporal_penalty, axis=1).astype(np.int64)
        # This mirrors the existing implementation and prevents duplicate
        # initial centers from collapsing a requested cluster through a tie.
        new_labels[np.asarray(center_indices, dtype=np.int64)] = np.arange(k, dtype=np.int64)

        new_visual_centers = np.empty_like(visual_centers)
        new_temporal_centers = np.empty_like(temporal_centers)
        for cluster_index in range(k):
            members = np.flatnonzero(new_labels == cluster_index)
            mean_visual = vectors[members].mean(axis=0)
            norm = float(np.linalg.norm(mean_visual))
            new_visual_centers[cluster_index] = (
                mean_visual / norm if norm > 0 else visual_centers[cluster_index]
            )
            new_temporal_centers[cluster_index] = float(scaled_times[members].mean())
        iterations_run = iteration + 1
        converged = np.array_equal(new_labels, labels)
        labels = new_labels
        visual_centers = new_visual_centers
        temporal_centers = new_temporal_centers
        if converged:
            break

    medoids: list[int] = []
    for cluster_index in range(k):
        members = np.flatnonzero(labels == cluster_index)
        scores = (
            vectors[members] @ visual_centers[cluster_index]
            - 0.5
            * time_weight
            * (scaled_times[members] - temporal_centers[cluster_index]) ** 2
        )
        medoids.append(int(members[int(np.argmax(scores))]))

    diagnostics = {
        "method": "temporal_spherical_kmeans_medoids_v1",
        "distance_formula": CLUSTER_DISTANCE_FORMULA,
        "time_weight": float(time_weight),
        "temporal_unit_seconds": float(temporal_unit_seconds),
        "cluster_count_requested": int(cluster_count),
        "cluster_count": k,
        "max_iterations": int(max_iterations),
        "iterations_run": iterations_run,
        "center_seed_indices": [int(index) for index in center_indices],
        "temporal_center_seconds": [
            round(float(times[0] + value * temporal_unit_seconds), 6)
            for value in temporal_centers
        ],
    }
    return labels.tolist(), medoids, diagnostics


def assert_zero_weight_compatibility(
    embeddings: Sequence[Sequence[float]],
    cluster_count: int,
    *,
    timestamps_seconds: Sequence[float] | None = None,
) -> None:
    """Fail if the sidecar's zero-weight result drifts from the current code."""

    timestamps = (
        list(timestamps_seconds)
        if timestamps_seconds is not None
        else [float(index) for index in range(len(embeddings))]
    )
    expected_labels, expected_medoids = cluster_embedding_medoids(
        [list(row) for row in embeddings], cluster_count
    )
    labels, medoids, _ = temporal_spherical_kmeans_medoids(
        embeddings,
        timestamps,
        cluster_count,
        time_weight=0.0,
    )
    if labels != expected_labels or medoids != expected_medoids:
        raise AssertionError(
            "time_weight=0 is not compatible with current cosine clustering: "
            f"labels={labels}/{expected_labels} medoids={medoids}/{expected_medoids}"
        )


def time_aware_clustered_frame_representatives(
    frames: Sequence[dict[str, Any]],
    embeddings: Sequence[Sequence[float]],
    *,
    cluster_count: int,
    time_weight: float,
    temporal_unit_seconds: float,
    max_iterations: int = 25,
) -> dict[str, Any]:
    """Expose medoid frames from the sidecar's time-aware clustering."""

    if len(frames) != len(embeddings):
        raise ValueError("frame and embedding counts must match")
    if not frames:
        raise ValueError("cannot cluster an empty frame list")
    timestamps = [float(frame["timestamp_seconds"]) for frame in frames]
    labels, medoids, clustering = temporal_spherical_kmeans_medoids(
        embeddings,
        timestamps,
        cluster_count,
        time_weight=time_weight,
        temporal_unit_seconds=temporal_unit_seconds,
        max_iterations=max_iterations,
    )
    normalized_embeddings = _normalized_embedding_matrix(embeddings)
    representatives = []
    representative_embeddings = []
    for cluster_index, frame_index in enumerate(medoids):
        member_indices = [
            index for index, label in enumerate(labels) if int(label) == cluster_index
        ]
        member_timestamps = [timestamps[index] for index in member_indices]
        frame = frames[frame_index]
        representatives.append(
            {
                "cluster_index": cluster_index,
                "visual_cluster_index": cluster_index,
                "temporal_component_index": 0,
                "frame_index": int(frame_index),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "path": frame.get("path"),
                "member_indices": member_indices,
                "member_timestamps": member_timestamps,
                "member_count": len(member_indices),
                "temporal_center_seconds": clustering["temporal_center_seconds"][cluster_index],
                "temporal_span_seconds": round(
                    max(member_timestamps) - min(member_timestamps), 6
                ),
            }
        )
        representative_embeddings.append(normalized_embeddings[frame_index].tolist())
    return {
        "cluster_count_requested": int(cluster_count),
        "cluster_count": len(representatives),
        "visual_cluster_count": len(representatives),
        "split_noncontiguous_clusters": False,
        "max_member_gap_seconds": None,
        "labels": labels,
        "representatives": representatives,
        "representative_embeddings": representative_embeddings,
        "clustering": clustering,
    }


def _cosine_matrix(
    left_embeddings: Sequence[Sequence[float]],
    right_embeddings: Sequence[Sequence[float]],
) -> list[list[float]]:
    left = _normalized_embedding_matrix(left_embeddings)
    right = _normalized_embedding_matrix(right_embeddings)
    return np.round(np.clip(left @ right.T, -1.0, 1.0), 6).tolist()


def _representative_temporal_bounds(
    representative: dict[str, Any],
) -> tuple[float, float, float]:
    """Return temporal center/start/end for one cluster representative."""

    member_times = [
        float(value) for value in representative.get("member_timestamps", [])
    ]
    medoid_time = float(representative["timestamp_seconds"])
    start = min(member_times) if member_times else medoid_time
    end = max(member_times) if member_times else medoid_time
    center_value = representative.get("temporal_center_seconds")
    center = (
        float(center_value)
        if center_value is not None
        else float(statistics.fmean(member_times))
        if member_times
        else medoid_time
    )
    return center, start, end


def cross_cluster_temporal_gaps(
    left_representative: dict[str, Any],
    right_representative: dict[str, Any],
) -> dict[str, float]:
    """Measure medoid, center, and closest-interval gaps for two clusters."""

    left_center, left_start, left_end = _representative_temporal_bounds(
        left_representative
    )
    right_center, right_start, right_end = _representative_temporal_bounds(
        right_representative
    )
    interval_gap = max(
        0.0,
        right_start - left_end,
        left_start - right_end,
    )
    return {
        "medoid_gap_seconds": abs(
            float(left_representative["timestamp_seconds"])
            - float(right_representative["timestamp_seconds"])
        ),
        "center_gap_seconds": abs(left_center - right_center),
        "interval_gap_seconds": interval_gap,
        "left_temporal_center_seconds": left_center,
        "right_temporal_center_seconds": right_center,
        "left_interval_start_seconds": left_start,
        "left_interval_end_seconds": left_end,
        "right_interval_start_seconds": right_start,
        "right_interval_end_seconds": right_end,
    }


def cross_cluster_temporal_gap_matrices(
    left_representatives: Sequence[dict[str, Any]],
    right_representatives: Sequence[dict[str, Any]],
) -> dict[str, list[list[float]]]:
    """Vectorize the three cross-cluster temporal gaps for grid reuse."""

    left_bounds = [
        _representative_temporal_bounds(representative)
        for representative in left_representatives
    ]
    right_bounds = [
        _representative_temporal_bounds(representative)
        for representative in right_representatives
    ]
    left_centers = np.asarray([value[0] for value in left_bounds], dtype=np.float64)
    left_starts = np.asarray([value[1] for value in left_bounds], dtype=np.float64)
    left_ends = np.asarray([value[2] for value in left_bounds], dtype=np.float64)
    right_centers = np.asarray([value[0] for value in right_bounds], dtype=np.float64)
    right_starts = np.asarray([value[1] for value in right_bounds], dtype=np.float64)
    right_ends = np.asarray([value[2] for value in right_bounds], dtype=np.float64)
    left_medoids = np.asarray(
        [float(row["timestamp_seconds"]) for row in left_representatives],
        dtype=np.float64,
    )
    right_medoids = np.asarray(
        [float(row["timestamp_seconds"]) for row in right_representatives],
        dtype=np.float64,
    )
    center = np.abs(left_centers[:, None] - right_centers[None, :])
    medoid = np.abs(left_medoids[:, None] - right_medoids[None, :])
    interval = np.maximum.reduce(
        [
            np.zeros((len(left_bounds), len(right_bounds)), dtype=np.float64),
            right_starts[None, :] - left_ends[:, None],
            left_starts[:, None] - right_ends[None, :],
        ]
    )
    return {
        "medoid": np.round(medoid, 6).tolist(),
        "center": np.round(center, 6).tolist(),
        "interval": np.round(interval, 6).tolist(),
    }


def prune_time_aware_cluster_pair(
    left_frames: list[dict[str, Any]],
    right_frames: list[dict[str, Any]],
    left_embeddings: Sequence[Sequence[float]],
    right_embeddings: Sequence[Sequence[float]],
    left_clusters: dict[str, Any],
    right_clusters: dict[str, Any],
    *,
    full_frame_matrix: list[list[float]],
    start_seconds: float,
    duration_seconds: float,
    sample_interval_seconds: float,
    high_similarity_threshold: float,
    min_pruned_video_seconds: float,
    pruning_protection_mode: str,
    min_pruned_video_percent: float | None,
    cross_gap_mode: str = "none",
    max_cross_gap_seconds: float | None = None,
    representative_similarity_matrix: list[list[float]] | None = None,
    representative_temporal_gap_matrices: dict[str, list[list[float]]] | None = None,
    left_best_frame_matches: dict[int, dict[str, Any]] | None = None,
    right_best_frame_matches: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Apply current pruning semantics to sidecar-produced clusters."""

    if duration_seconds <= 0 or sample_interval_seconds <= 0:
        raise ValueError("duration and sample interval must be positive")
    if len(left_frames) != len(left_embeddings):
        raise ValueError("left frame and embedding counts must match")
    if len(right_frames) != len(right_embeddings):
        raise ValueError("right frame and embedding counts must match")
    if len(full_frame_matrix) != len(left_frames) or any(
        len(row) != len(right_frames) for row in full_frame_matrix
    ):
        raise ValueError("full-frame similarity matrix must cover both timelines")
    normalized_cross_gap_mode = str(cross_gap_mode).strip().lower()
    if normalized_cross_gap_mode not in {"none", "center", "interval"}:
        raise ValueError("cross gap mode must be one of: none, center, interval")
    if normalized_cross_gap_mode == "none":
        effective_max_cross_gap = None
    else:
        if max_cross_gap_seconds is None:
            raise ValueError("max cross gap seconds is required for a temporal gate")
        effective_max_cross_gap = float(max_cross_gap_seconds)
        if not math.isfinite(effective_max_cross_gap) or effective_max_cross_gap < 0:
            raise ValueError("max cross gap seconds must be finite and non-negative")
    window_start = float(start_seconds)
    window_end = round(window_start + float(duration_seconds), 3)
    target_kept_seconds = _protected_duration_target_seconds(
        mode=pruning_protection_mode,
        duration_seconds=duration_seconds,
        min_pruned_video_seconds=min_pruned_video_seconds,
        min_pruned_video_percent=min_pruned_video_percent,
    )
    representative_matrix = (
        representative_similarity_matrix
        if representative_similarity_matrix is not None
        else _cosine_matrix(
            left_clusters["representative_embeddings"],
            right_clusters["representative_embeddings"],
        )
    )
    if len(representative_matrix) != int(left_clusters["cluster_count"]) or any(
        len(row) != int(right_clusters["cluster_count"])
        for row in representative_matrix
    ):
        raise ValueError("representative similarity matrix must cover both cluster sets")
    temporal_gap_matrices = (
        representative_temporal_gap_matrices
        if representative_temporal_gap_matrices is not None
        else cross_cluster_temporal_gap_matrices(
            left_clusters["representatives"], right_clusters["representatives"]
        )
    )
    expected_shape = (
        int(left_clusters["cluster_count"]),
        int(right_clusters["cluster_count"]),
    )
    temporal_gap_arrays: dict[str, np.ndarray] = {}
    for name in ("medoid", "center", "interval"):
        if name not in temporal_gap_matrices:
            raise ValueError(f"representative temporal gap matrices missing {name}")
        current = np.asarray(temporal_gap_matrices[name], dtype=np.float64)
        if current.shape != expected_shape:
            raise ValueError(
                f"representative temporal {name} gap matrix has shape "
                f"{current.shape}, expected {expected_shape}"
            )
        temporal_gap_arrays[name] = current
    representative_array = np.asarray(representative_matrix, dtype=np.float64)
    ungated_high_mask = representative_array >= float(high_similarity_threshold)
    if normalized_cross_gap_mode == "none":
        eligible_mask = np.ones(expected_shape, dtype=bool)
    else:
        eligible_mask = (
            temporal_gap_arrays[normalized_cross_gap_mode]
            <= float(effective_max_cross_gap)
        )
    accepted_indices = np.argwhere(ungated_high_mask & eligible_mask)
    high_pairs = []
    left_marked_clusters: set[int] = set()
    right_marked_clusters: set[int] = set()
    eligible_pair_count = int(np.count_nonzero(eligible_mask))
    ungated_high_similarity_pair_count = int(np.count_nonzero(ungated_high_mask))
    for left_cluster_index_value, right_cluster_index_value in accepted_indices:
        left_cluster_index = int(left_cluster_index_value)
        right_cluster_index = int(right_cluster_index_value)
        similarity = float(representative_array[left_cluster_index, right_cluster_index])
        left_rep = left_clusters["representatives"][left_cluster_index]
        right_rep = right_clusters["representatives"][right_cluster_index]
        temporal_gaps = cross_cluster_temporal_gaps(left_rep, right_rep)
        selected_gap = (
            temporal_gaps[f"{normalized_cross_gap_mode}_gap_seconds"]
            if normalized_cross_gap_mode != "none"
            else None
        )
        left_marked_clusters.add(left_cluster_index)
        right_marked_clusters.add(right_cluster_index)
        left_time = float(left_rep["timestamp_seconds"])
        right_time = float(right_rep["timestamp_seconds"])
        high_pairs.append(
            {
                    "left_cluster_index": left_cluster_index,
                    "right_cluster_index": right_cluster_index,
                    "similarity": round(float(similarity), 6),
                    "left_representative_frame_index": left_rep["frame_index"],
                    "right_representative_frame_index": right_rep["frame_index"],
                    "left_representative_timestamp_seconds": left_time,
                    "right_representative_timestamp_seconds": right_time,
                    "timestamp_difference_seconds": round(abs(left_time - right_time), 6),
                    "center_gap_seconds": round(
                        float(temporal_gaps["center_gap_seconds"]), 6
                    ),
                    "interval_gap_seconds": round(
                        float(temporal_gaps["interval_gap_seconds"]), 6
                    ),
                    "selected_cross_gap_seconds": (
                        round(float(selected_gap), 6)
                        if selected_gap is not None
                        else None
                    ),
                    "left_temporal_center_seconds": round(
                        float(temporal_gaps["left_temporal_center_seconds"]), 6
                    ),
                    "right_temporal_center_seconds": round(
                        float(temporal_gaps["right_temporal_center_seconds"]), 6
                    ),
                    "left_interval_start_seconds": round(
                        float(temporal_gaps["left_interval_start_seconds"]), 6
                    ),
                    "left_interval_end_seconds": round(
                        float(temporal_gaps["left_interval_end_seconds"]), 6
                    ),
                    "right_interval_start_seconds": round(
                        float(temporal_gaps["right_interval_start_seconds"]), 6
                    ),
                    "right_interval_end_seconds": round(
                        float(temporal_gaps["right_interval_end_seconds"]), 6
                    ),
            }
        )
    high_pairs.sort(key=lambda row: float(row["similarity"]), reverse=True)

    def marked_indices(clusters: dict[str, Any], selected: set[int]) -> set[int]:
        output: set[int] = set()
        for cluster_index in selected:
            output.update(
                int(index)
                for index in clusters["representatives"][cluster_index]["member_indices"]
            )
        return output

    left_initial_marked = marked_indices(left_clusters, left_marked_clusters)
    right_initial_marked = marked_indices(right_clusters, right_marked_clusters)
    left_best_matches = left_best_frame_matches or _side_best_frame_matches(
        full_frame_matrix,
        side="left",
        left_frames=left_frames,
        right_frames=right_frames,
        max_pair_time_difference_seconds=None,
    )
    right_best_matches = right_best_frame_matches or _side_best_frame_matches(
        full_frame_matrix,
        side="right",
        left_frames=left_frames,
        right_frames=right_frames,
        max_pair_time_difference_seconds=None,
    )
    left_protection = _apply_pruning_duration_protection(
        left_frames,
        left_initial_marked,
        left_best_matches,
        side="left",
        window_start=window_start,
        window_end=window_end,
        sample_interval_seconds=sample_interval_seconds,
        high_similarity_threshold=high_similarity_threshold,
        target_kept_seconds=target_kept_seconds,
        preserved_intervals=[],
    )
    right_protection = _apply_pruning_duration_protection(
        right_frames,
        right_initial_marked,
        right_best_matches,
        side="right",
        window_start=window_start,
        window_end=window_end,
        sample_interval_seconds=sample_interval_seconds,
        high_similarity_threshold=high_similarity_threshold,
        target_kept_seconds=target_kept_seconds,
        preserved_intervals=[],
    )
    left_marked_indices = set(left_protection["marked_indices"])
    right_marked_indices = set(right_protection["marked_indices"])
    required_kept_duration = (
        float(min_pruned_video_seconds)
        if pruning_protection_mode == "reject"
        else float(target_kept_seconds or 0.0)
    )
    removed_duration = round(
        float(left_protection["removed_duration_seconds"])
        + float(right_protection["removed_duration_seconds"]),
        3,
    )
    passed = (
        float(left_protection["kept_duration_seconds"]) >= required_kept_duration
        and float(right_protection["kept_duration_seconds"]) >= required_kept_duration
        and bool(left_protection["target_met"])
        and bool(right_protection["target_met"])
        and removed_duration > 0
    )

    def decisions(clusters: dict[str, Any], selected: set[int]) -> list[dict[str, Any]]:
        return [
            {
                **representative,
                "status": (
                    "marked_for_pruning"
                    if int(representative["cluster_index"]) in selected
                    else "kept"
                ),
            }
            for representative in clusters["representatives"]
        ]

    return {
        "method": (
            "temporal_kmeans_sidecar_clip_pruning_v1"
            if normalized_cross_gap_mode == "none"
            else "temporal_kmeans_sidecar_hard_cross_gap_pruning_v1"
        ),
        "isolation_contract": (
            "within-video time affects clustering; cross-video representatives must "
            "pass the unchanged CLIP threshold and the configured hard temporal gate; "
            "full-frame duration protection remains unchanged"
            if normalized_cross_gap_mode != "none"
            else "time affects within-video clustering only; cross-video representative "
            "and full-frame matching remain pure CLIP cosine similarity"
        ),
        "high_similarity_threshold": float(high_similarity_threshold),
        "cross_gap_mode": normalized_cross_gap_mode,
        "max_cross_gap_seconds": effective_max_cross_gap,
        "representative_pair_count": int(left_clusters["cluster_count"])
        * int(right_clusters["cluster_count"]),
        "cross_gap_eligible_pair_count": eligible_pair_count,
        "ungated_high_similarity_representative_pair_count": (
            ungated_high_similarity_pair_count
        ),
        "cross_gap_rejected_high_similarity_pair_count": (
            ungated_high_similarity_pair_count - len(high_pairs)
        ),
        "cluster_count": int(left_clusters["cluster_count_requested"]),
        "left_cluster_count": int(left_clusters["cluster_count"]),
        "right_cluster_count": int(right_clusters["cluster_count"]),
        "time_weight": float(left_clusters["clustering"]["time_weight"]),
        "temporal_unit_seconds": float(
            left_clusters["clustering"]["temporal_unit_seconds"]
        ),
        "distance_formula": CLUSTER_DISTANCE_FORMULA,
        "pruning_protection_mode": pruning_protection_mode,
        "min_pruned_video_seconds": float(min_pruned_video_seconds),
        "min_pruned_video_percent": min_pruned_video_percent,
        "protection_target_kept_seconds": target_kept_seconds,
        "window": {
            "start_seconds": window_start,
            "duration_seconds": float(duration_seconds),
            "sample_interval_seconds": float(sample_interval_seconds),
        },
        "representative_similarity_matrix": representative_matrix,
        "high_similarity_representative_pairs": high_pairs,
        "high_similarity_representative_pair_count": len(high_pairs),
        "left_marked_cluster_count": len(left_marked_clusters),
        "right_marked_cluster_count": len(right_marked_clusters),
        "left_marked_frame_indices": sorted(left_marked_indices),
        "right_marked_frame_indices": sorted(right_marked_indices),
        "left_restored_frame_indices": [
            int(row["frame_index"]) for row in left_protection["restored_frames"]
        ],
        "right_restored_frame_indices": [
            int(row["frame_index"]) for row in right_protection["restored_frames"]
        ],
        "left_restored_frames": left_protection["restored_frames"],
        "right_restored_frames": right_protection["restored_frames"],
        "left_remove_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in left_protection["remove_intervals"]
        ],
        "right_remove_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in right_protection["remove_intervals"]
        ],
        "left_keep_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in left_protection["keep_intervals"]
        ],
        "right_keep_intervals": [
            [round(start, 3), round(end, 3)]
            for start, end in right_protection["keep_intervals"]
        ],
        "left_removed_duration_seconds": left_protection["removed_duration_seconds"],
        "right_removed_duration_seconds": right_protection["removed_duration_seconds"],
        "removed_duration_seconds": removed_duration,
        "left_kept_duration_seconds": left_protection["kept_duration_seconds"],
        "right_kept_duration_seconds": right_protection["kept_duration_seconds"],
        "kept_duration_seconds": round(
            min(
                float(left_protection["kept_duration_seconds"]),
                float(right_protection["kept_duration_seconds"]),
            ),
            3,
        ),
        "passed": passed,
        "left_cluster_decisions": decisions(left_clusters, left_marked_clusters),
        "right_cluster_decisions": decisions(right_clusters, right_marked_clusters),
    }


def _expected_frame_count(duration_seconds: float, sample_interval_seconds: float) -> int:
    return len(
        np.arange(0.0, float(duration_seconds) - 1e-9, float(sample_interval_seconds))
    )


def _sample_segmented_clip(
    clip: dict[str, Any],
    *,
    output_dir: Path,
    cache_dir: str | Path,
    duration_seconds: float,
    sample_interval_seconds: float,
    ffmpeg_binary: str,
    download_media: bool,
) -> list[dict[str, Any]]:
    segments = clip.get("segments")
    source_segments = list(segments) if isinstance(segments, list) and segments else [clip]
    needed_segments = int(math.ceil(duration_seconds / SOURCE_CLIP_DURATION_SECONDS))
    if len(source_segments) < needed_segments:
        raise ValueError(
            f"clip has {len(source_segments)} source segments but {needed_segments} are required"
        )
    frames: list[dict[str, Any]] = []
    remaining = float(duration_seconds)
    for segment_index, segment in enumerate(source_segments[:needed_segments]):
        segment_duration = min(SOURCE_CLIP_DURATION_SECONDS, remaining)
        local_video = _resolve_local_video(
            segment,
            cache_dir=cache_dir,
            download_media=download_media,
        )
        sampled = sample_short_video(
            local_video,
            output_dir / f"segment_{segment_index:03d}",
            duration_seconds=segment_duration,
            sample_interval_seconds=sample_interval_seconds,
            start_seconds=0.0,
            ffmpeg_binary=ffmpeg_binary,
        )
        expected = _expected_frame_count(segment_duration, sample_interval_seconds)
        if len(sampled) != expected:
            raise RuntimeError(
                "incomplete source-segment sampling: "
                f"segment={segment_index} expected={expected} actual={len(sampled)}"
            )
        offset = segment_index * SOURCE_CLIP_DURATION_SECONDS
        for frame in sampled:
            frames.append(
                {
                    **frame,
                    "timestamp_seconds": round(
                        offset + float(frame["timestamp_seconds"]), 6
                    ),
                    "source_segment_index": segment_index,
                    "source_timestamp_seconds": frame["timestamp_seconds"],
                    "source_video": str(local_video),
                }
            )
        remaining -= segment_duration
    expected_total = _expected_frame_count(duration_seconds, sample_interval_seconds)
    if len(frames) != expected_total:
        raise RuntimeError(
            f"incomplete long-window sampling: expected={expected_total} actual={len(frames)}"
        )
    return frames


def _select_pair(group: dict[str, Any], rng: random.Random) -> list[dict[str, Any]]:
    clips = sorted(group.get("clips", []), key=lambda row: str(row.get("agent_dir")))
    if len(clips) < 2:
        raise ValueError("synchronized group contains fewer than two videos")
    selected = clips if len(clips) == 2 else rng.sample(clips, 2)
    return sorted(selected, key=lambda row: str(row.get("agent_dir")))


def _embedding_cache_signature(
    frames_by_side: Sequence[Sequence[dict[str, Any]]], model_id: str
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "sides": [
            [
                {
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "path": str(frame["path"]),
                }
                for frame in frames
            ]
            for frames in frames_by_side
        ],
    }


def _load_or_encode_pair(
    pair_dir: Path,
    frames_by_side: Sequence[Sequence[dict[str, Any]]],
    *,
    encoder: ImageEncoder,
    clip_batch_size: int,
) -> list[np.ndarray]:
    metadata_path = pair_dir / "embedding_cache.json"
    array_path = pair_dir / "embedding_cache.npz"
    signature = _embedding_cache_signature(frames_by_side, encoder.model_id)
    if metadata_path.exists() and array_path.exists():
        try:
            if read_json(metadata_path) == signature:
                with np.load(array_path, allow_pickle=False) as cached:
                    arrays = [cached["left"].copy(), cached["right"].copy()]
                if all(len(array) == len(frames) for array, frames in zip(arrays, frames_by_side)):
                    return arrays
        except Exception:
            pass

    arrays = []
    for frames in frames_by_side:
        encoded = encode_paths_in_batches(
            encoder,
            [str(frame["path"]) for frame in frames],
            batch_size=clip_batch_size,
        )
        arrays.append(_normalized_embedding_matrix(encoded).astype(np.float32))
    pair_dir.mkdir(parents=True, exist_ok=True)
    temporary = array_path.with_name(array_path.name + ".tmp.npz")
    np.savez_compressed(temporary, left=arrays[0], right=arrays[1])
    temporary.replace(array_path)
    write_json(metadata_path, signature)
    return arrays


def _load_pair_embedding_cache(
    pair_dir: str | Path,
    *,
    expected_model_id: str | None = None,
    expected_frame_count: int | None = None,
) -> tuple[list[list[dict[str, Any]]], list[np.ndarray], dict[str, Any]]:
    """Load one complete two-sided cache without opening sampled images."""

    pair_dir = Path(pair_dir)
    metadata_path = pair_dir / "embedding_cache.json"
    array_path = pair_dir / "embedding_cache.npz"
    if not metadata_path.is_file() or not array_path.is_file():
        raise FileNotFoundError(f"incomplete embedding cache: {pair_dir}")

    metadata = read_json(metadata_path)
    if not isinstance(metadata, dict):
        raise ValueError(f"embedding cache metadata must be an object: {metadata_path}")
    model_id = str(metadata.get("model_id") or "")
    if expected_model_id is not None and model_id != expected_model_id:
        raise ValueError(
            f"embedding cache model mismatch: expected={expected_model_id!r} "
            f"actual={model_id!r} path={metadata_path}"
        )
    raw_sides = metadata.get("sides")
    if not isinstance(raw_sides, list) or len(raw_sides) != 2:
        raise ValueError(f"embedding cache must contain exactly two sides: {metadata_path}")

    frames_by_side: list[list[dict[str, Any]]] = []
    for side_index, raw_frames in enumerate(raw_sides):
        if not isinstance(raw_frames, list):
            raise ValueError(f"cache side {side_index} is not a frame list: {metadata_path}")
        frames = []
        for frame_index, raw_frame in enumerate(raw_frames):
            if not isinstance(raw_frame, dict):
                raise ValueError(
                    f"cache frame {side_index}:{frame_index} is not an object: {metadata_path}"
                )
            timestamp = float(raw_frame["timestamp_seconds"])
            if not math.isfinite(timestamp):
                raise ValueError(f"non-finite cached timestamp: {metadata_path}")
            frames.append(
                {
                    "timestamp_seconds": timestamp,
                    "path": str(raw_frame.get("path") or ""),
                }
            )
        if expected_frame_count is not None and len(frames) != expected_frame_count:
            raise ValueError(
                f"cached frame count mismatch: side={side_index} "
                f"expected={expected_frame_count} actual={len(frames)} path={metadata_path}"
            )
        frames_by_side.append(frames)

    with np.load(array_path, allow_pickle=False) as cached:
        if "left" not in cached or "right" not in cached:
            raise ValueError(f"embedding cache is missing left/right arrays: {array_path}")
        embeddings_by_side = [
            _normalized_embedding_matrix(cached["left"]).astype(np.float32),
            _normalized_embedding_matrix(cached["right"]).astype(np.float32),
        ]
    for side_index, (frames, embeddings) in enumerate(
        zip(frames_by_side, embeddings_by_side)
    ):
        if len(frames) != len(embeddings):
            raise ValueError(
                f"cached frame/embedding mismatch: side={side_index} "
                f"frames={len(frames)} embeddings={len(embeddings)} path={pair_dir}"
            )
    return frames_by_side, embeddings_by_side, metadata


def _copy_pair_embedding_cache(source_pair_dir: Path, target_pair_dir: Path) -> None:
    """Make a resumed output independently reusable without copying frame JPEGs."""

    target_pair_dir.mkdir(parents=True, exist_ok=True)
    for name in ("embedding_cache.json", "embedding_cache.npz"):
        source = source_pair_dir / name
        target = target_pair_dir / name
        if source.resolve() == target.resolve():
            continue
        temporary = target.with_name(target.name + ".tmp")
        shutil.copy2(source, temporary)
        temporary.replace(target)


def _duration_prefix(
    frames: Sequence[dict[str, Any]],
    embeddings: np.ndarray,
    duration_seconds: float,
    sample_interval_seconds: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    indices = [
        index
        for index, frame in enumerate(frames)
        if float(frame["timestamp_seconds"]) < duration_seconds - 1e-9
    ]
    expected = _expected_frame_count(duration_seconds, sample_interval_seconds)
    if len(indices) != expected:
        raise RuntimeError(
            f"duration prefix is incomplete: duration={duration_seconds} "
            f"expected={expected} actual={len(indices)}"
        )
    return [dict(frames[index]) for index in indices], embeddings[np.asarray(indices)]


def cluster_quality_metrics(
    clusters: dict[str, Any], embeddings: Sequence[Sequence[float]]
) -> dict[str, Any]:
    matrix = _normalized_embedding_matrix(embeddings)
    spans: list[float] = []
    gaps: list[float] = []
    visual_similarities: list[float] = []
    for representative in clusters["representatives"]:
        medoid_index = int(representative["frame_index"])
        member_indices = [int(index) for index in representative["member_indices"]]
        member_times = [float(value) for value in representative["member_timestamps"]]
        medoid_time = float(representative["timestamp_seconds"])
        spans.append(max(member_times) - min(member_times))
        gaps.extend(abs(value - medoid_time) for value in member_times)
        visual_similarities.extend(
            float(value) for value in matrix[member_indices] @ matrix[medoid_index]
        )
    return {
        "cluster_spans": spans,
        "member_medoid_gaps": gaps,
        "member_medoid_visual_similarities": visual_similarities,
    }


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(np.asarray(values, dtype=np.float64), percentile)), 6)


def _mean(values: Sequence[float]) -> float | None:
    return round(float(statistics.fmean(values)), 6) if values else None


def _pruned_member_gaps(
    cluster_decisions: Sequence[dict[str, Any]], marked_frame_indices: set[int]
) -> list[float]:
    gaps = []
    for cluster in cluster_decisions:
        medoid_time = float(cluster["timestamp_seconds"])
        for member_index, timestamp in zip(
            cluster["member_indices"], cluster["member_timestamps"]
        ):
            if int(member_index) in marked_frame_indices:
                gaps.append(abs(float(timestamp) - medoid_time))
    return gaps


def _metric_row(
    *,
    pair: dict[str, Any],
    duration_seconds: float,
    sample_interval_seconds: float,
    seconds_per_cluster: float,
    k: int,
    time_weight: float,
    temporal_unit_seconds: float,
    threshold: float,
    left_quality: dict[str, Any],
    right_quality: dict[str, Any],
    pruning: dict[str, Any],
    diagnostics_path: str | None,
) -> dict[str, Any]:
    spans = left_quality["cluster_spans"] + right_quality["cluster_spans"]
    gaps = left_quality["member_medoid_gaps"] + right_quality["member_medoid_gaps"]
    visual = (
        left_quality["member_medoid_visual_similarities"]
        + right_quality["member_medoid_visual_similarities"]
    )
    pruned_gaps = _pruned_member_gaps(
        pruning["left_cluster_decisions"], set(pruning["left_marked_frame_indices"])
    ) + _pruned_member_gaps(
        pruning["right_cluster_decisions"], set(pruning["right_marked_frame_indices"])
    )
    trigger_differences = [
        float(row["timestamp_difference_seconds"])
        for row in pruning["high_similarity_representative_pairs"]
    ]
    left_removed_percent = (
        100.0 * float(pruning["left_removed_duration_seconds"]) / duration_seconds
    )
    right_removed_percent = (
        100.0 * float(pruning["right_removed_duration_seconds"]) / duration_seconds
    )
    return {
        "pair_id": pair["pair_id"],
        "day": pair.get("day"),
        "time_token": pair.get("time_token"),
        "left_agent": pair["left_agent"],
        "right_agent": pair["right_agent"],
        "duration_seconds": float(duration_seconds),
        "sample_interval_seconds": float(sample_interval_seconds),
        "seconds_per_cluster": float(seconds_per_cluster),
        "k": int(k),
        "time_weight": float(time_weight),
        "temporal_unit_seconds": float(temporal_unit_seconds),
        "high_similarity_threshold": float(threshold),
        "left_cluster_count": pruning["left_cluster_count"],
        "right_cluster_count": pruning["right_cluster_count"],
        "mean_cluster_span_seconds": _mean(spans),
        "p95_cluster_span_seconds": _percentile(spans, 95),
        "max_cluster_span_seconds": round(max(spans), 6) if spans else None,
        "mean_member_medoid_gap_seconds": _mean(gaps),
        "p95_member_medoid_gap_seconds": _percentile(gaps, 95),
        "member_gap_gt_unit_fraction": (
            round(sum(value > temporal_unit_seconds for value in gaps) / len(gaps), 6)
            if gaps
            else None
        ),
        "member_gap_gt_quarter_duration_fraction": (
            round(sum(value > duration_seconds / 4.0 for value in gaps) / len(gaps), 6)
            if gaps
            else None
        ),
        "mean_member_medoid_visual_similarity": _mean(visual),
        "pruned_member_gap_gt_unit_fraction": (
            round(
                sum(value > temporal_unit_seconds for value in pruned_gaps)
                / len(pruned_gaps),
                6,
            )
            if pruned_gaps
            else None
        ),
        "pruned_member_gap_gt_quarter_duration_fraction": (
            round(
                sum(value > duration_seconds / 4.0 for value in pruned_gaps)
                / len(pruned_gaps),
                6,
            )
            if pruned_gaps
            else None
        ),
        "p95_pruned_member_medoid_gap_seconds": _percentile(pruned_gaps, 95),
        "high_similarity_representative_pair_count": len(trigger_differences),
        "mean_trigger_time_difference_seconds": _mean(trigger_differences),
        "max_trigger_time_difference_seconds": (
            round(max(trigger_differences), 6) if trigger_differences else None
        ),
        "left_marked_frame_count": len(pruning["left_marked_frame_indices"]),
        "right_marked_frame_count": len(pruning["right_marked_frame_indices"]),
        "left_removed_percent": round(left_removed_percent, 6),
        "right_removed_percent": round(right_removed_percent, 6),
        "mean_removed_percent": round(
            (left_removed_percent + right_removed_percent) / 2.0, 6
        ),
        "left_keep_segment_count": len(pruning["left_keep_intervals"]),
        "right_keep_segment_count": len(pruning["right_keep_intervals"]),
        "no_removal": float(pruning["removed_duration_seconds"]) <= 0,
        "passed": bool(pruning["passed"]),
        "diagnostics_path": diagnostics_path,
    }


def build_grid_variants(
    durations_seconds: Sequence[float],
    seconds_per_cluster_values: Sequence[float],
    time_weights: Sequence[float],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    variants = []
    for duration in durations_seconds:
        for seconds_per_cluster in seconds_per_cluster_values:
            k = max(1, int(math.ceil(float(duration) / float(seconds_per_cluster))))
            for time_weight in time_weights:
                for threshold in thresholds:
                    variants.append(
                        {
                            "duration_seconds": float(duration),
                            "seconds_per_cluster": float(seconds_per_cluster),
                            "k": k,
                            "time_weight": float(time_weight),
                            "high_similarity_threshold": float(threshold),
                        }
                    )
    return variants


def aggregate_metrics(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float, int, float, float, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in rows:
        key = (
            float(row["duration_seconds"]),
            float(row["seconds_per_cluster"]),
            int(row["k"]),
            float(row["time_weight"]),
            float(row["temporal_unit_seconds"]),
            float(row["high_similarity_threshold"]),
        )
        grouped[key].append(row)

    def average(selected: Sequence[dict[str, Any]], field: str) -> float | None:
        values = [float(row[field]) for row in selected if row.get(field) is not None]
        return _mean(values)

    aggregates = []
    for key, selected in sorted(grouped.items()):
        duration, seconds_per_cluster, k, weight, unit, threshold = key
        aggregates.append(
            {
                "duration_seconds": duration,
                "seconds_per_cluster": seconds_per_cluster,
                "k": k,
                "time_weight": weight,
                "temporal_unit_seconds": unit,
                "high_similarity_threshold": threshold,
                "pair_count": len(selected),
                "pass_rate": round(sum(bool(row["passed"]) for row in selected) / len(selected), 6),
                "no_removal_rate": round(
                    sum(bool(row["no_removal"]) for row in selected) / len(selected), 6
                ),
                "mean_cluster_span_seconds": average(selected, "mean_cluster_span_seconds"),
                "mean_p95_cluster_span_seconds": average(
                    selected, "p95_cluster_span_seconds"
                ),
                "mean_max_cluster_span_seconds": average(
                    selected, "max_cluster_span_seconds"
                ),
                "mean_member_medoid_gap_seconds": average(
                    selected, "mean_member_medoid_gap_seconds"
                ),
                "mean_p95_member_medoid_gap_seconds": average(
                    selected, "p95_member_medoid_gap_seconds"
                ),
                "mean_member_gap_gt_unit_fraction": average(
                    selected, "member_gap_gt_unit_fraction"
                ),
                "mean_member_gap_gt_quarter_duration_fraction": average(
                    selected, "member_gap_gt_quarter_duration_fraction"
                ),
                "mean_member_medoid_visual_similarity": average(
                    selected, "mean_member_medoid_visual_similarity"
                ),
                "mean_pruned_member_gap_gt_unit_fraction": average(
                    selected, "pruned_member_gap_gt_unit_fraction"
                ),
                "mean_pruned_member_gap_gt_quarter_duration_fraction": average(
                    selected, "pruned_member_gap_gt_quarter_duration_fraction"
                ),
                "mean_removed_percent": average(selected, "mean_removed_percent"),
                "mean_keep_segment_count": _mean(
                    [
                        (
                            float(row["left_keep_segment_count"])
                            + float(row["right_keep_segment_count"])
                        )
                        / 2.0
                        for row in selected
                    ]
                ),
                "mean_trigger_pair_count": average(
                    selected, "high_similarity_representative_pair_count"
                ),
            }
        )

    baselines = {
        (
            row["duration_seconds"],
            row["seconds_per_cluster"],
            row["k"],
            row["temporal_unit_seconds"],
            row["high_similarity_threshold"],
        ): row
        for row in aggregates
        if float(row["time_weight"]) == 0.0
    }
    for row in aggregates:
        baseline = baselines.get(
            (
                row["duration_seconds"],
                row["seconds_per_cluster"],
                row["k"],
                row["temporal_unit_seconds"],
                row["high_similarity_threshold"],
            )
        )

        def delta(field: str, *, reduction: bool = False) -> float | None:
            if baseline is None or baseline.get(field) is None or row.get(field) is None:
                return None
            value = (
                float(baseline[field]) - float(row[field])
                if reduction
                else float(row[field]) - float(baseline[field])
            )
            return round(value, 6)

        row["member_gap_reduction_vs_w0"] = delta(
            "mean_member_gap_gt_unit_fraction", reduction=True
        )
        row["quarter_duration_gap_reduction_vs_w0"] = delta(
            "mean_member_gap_gt_quarter_duration_fraction", reduction=True
        )
        row["visual_similarity_delta_vs_w0"] = delta(
            "mean_member_medoid_visual_similarity"
        )
        row["pruned_gap_reduction_vs_w0"] = delta(
            "mean_pruned_member_gap_gt_unit_fraction", reduction=True
        )
        row["pruned_quarter_duration_gap_reduction_vs_w0"] = delta(
            "mean_pruned_member_gap_gt_quarter_duration_fraction", reduction=True
        )
        row["removed_percent_delta_vs_w0"] = delta("mean_removed_percent")

    pareto_groups: dict[tuple[float, float, int, float, float], list[dict[str, Any]]] = (
        defaultdict(list)
    )
    for row in aggregates:
        pareto_groups[
            (
                row["duration_seconds"],
                row["seconds_per_cluster"],
                row["k"],
                row["temporal_unit_seconds"],
                row["high_similarity_threshold"],
            )
        ].append(row)
    for selected in pareto_groups.values():
        for candidate in selected:
            candidate_leakage = candidate.get(
                "mean_member_gap_gt_quarter_duration_fraction"
            )
            candidate_visual = candidate.get("mean_member_medoid_visual_similarity")
            dominated = False
            if candidate_leakage is not None and candidate_visual is not None:
                for other in selected:
                    if other is candidate:
                        continue
                    other_leakage = other.get(
                        "mean_member_gap_gt_quarter_duration_fraction"
                    )
                    other_visual = other.get("mean_member_medoid_visual_similarity")
                    if other_leakage is None or other_visual is None:
                        continue
                    if (
                        float(other_leakage) <= float(candidate_leakage)
                        and float(other_visual) >= float(candidate_visual)
                        and (
                            float(other_leakage) < float(candidate_leakage)
                            or float(other_visual) > float(candidate_visual)
                        )
                    ):
                        dominated = True
                        break
            candidate["temporal_visual_pareto"] = not dominated
    return aggregates


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def _write_json_atomic(path: Path, data: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_json(temporary, data)
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write_jsonl(temporary, rows)
    temporary.replace(path)


def _write_csv_atomic(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    _write_csv(temporary, rows, fields)
    temporary.replace(path)


def write_summary_html(output_dir: Path, aggregates: Sequence[dict[str, Any]]) -> Path:
    lines = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>Temporal K-means grid</title>",
        "<style>body{font-family:system-ui;margin:24px;background:#f7f7f8;color:#222}",
        "table{border-collapse:collapse;width:100%;background:white;font-size:13px}",
        "th,td{border:1px solid #ddd;padding:6px;text-align:right}th{position:sticky;top:0;background:#eee}",
        "td:first-child,th:first-child{text-align:left}.pareto{background:#e8f7e8}</style></head><body>",
        "<h1>Temporal K-means fixed-cohort grid</h1>",
        "<p>Green rows are non-dominated on lower long-range cluster membership and higher visual coherence. "
        "Cross-video matching remains pure CLIP in every row.</p>",
        "<table><thead><tr>",
    ]
    display_fields = [
        "duration_seconds",
        "seconds_per_cluster",
        "k",
        "time_weight",
        "pair_count",
        "mean_member_gap_gt_quarter_duration_fraction",
        "quarter_duration_gap_reduction_vs_w0",
        "mean_member_medoid_visual_similarity",
        "visual_similarity_delta_vs_w0",
        "mean_pruned_member_gap_gt_quarter_duration_fraction",
        "mean_removed_percent",
        "pass_rate",
        "no_removal_rate",
    ]
    lines.extend(f"<th>{html.escape(field)}</th>" for field in display_fields)
    lines.append("</tr></thead><tbody>")
    for row in aggregates:
        css = " class='pareto'" if row.get("temporal_visual_pareto") else ""
        lines.append(f"<tr{css}>")
        for field in display_fields:
            value = row.get(field)
            rendered = "" if value is None else str(value)
            lines.append(f"<td>{html.escape(rendered)}</td>")
        lines.append("</tr>")
    lines.extend(["</tbody></table></body></html>"])
    path = output_dir / "summary.html"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_progress_checkpoint(
    output_root: Path,
    *,
    cohort_rows: Sequence[dict[str, Any]],
    metric_rows: Sequence[dict[str, Any]],
    skipped: Sequence[dict[str, Any]],
    pair_count_target: int,
    configuration_count_per_pair: int,
    resume_pairs_dir: str | None,
    resume_cache_candidates: int,
    status: str,
) -> list[dict[str, Any]]:
    """Persist every completed pair so timeouts leave analyzable results."""

    aggregates = aggregate_metrics(metric_rows)
    _write_jsonl_atomic(output_root / "cohort.jsonl", cohort_rows)
    _write_jsonl_atomic(output_root / "grid_metrics.jsonl", metric_rows)
    _write_jsonl_atomic(output_root / "aggregate_metrics.jsonl", aggregates)
    _write_csv_atomic(output_root / "grid_metrics.csv", metric_rows, METRIC_FIELDS)
    _write_csv_atomic(
        output_root / "aggregate_metrics.csv", aggregates, AGGREGATE_FIELDS
    )
    write_summary_html(output_root, aggregates)
    reused_count = sum(bool(row.get("embedding_cache_reused")) for row in cohort_rows)
    progress = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "pair_count_target": int(pair_count_target),
        "pair_count_completed": len(cohort_rows),
        "configuration_count_per_pair": int(configuration_count_per_pair),
        "expected_metric_count": int(pair_count_target)
        * int(configuration_count_per_pair),
        "metric_count_completed": len(metric_rows),
        "aggregate_count": len(aggregates),
        "last_completed_pair_id": (
            cohort_rows[-1]["pair_id"] if cohort_rows else None
        ),
        "resume_pairs_dir": resume_pairs_dir,
        "resume_cache_candidates": int(resume_cache_candidates),
        "resume_cache_pairs_reused": reused_count,
        "new_pairs_completed": len(cohort_rows) - reused_count,
        "skipped_group_count": len(skipped),
        "skipped_groups": list(skipped),
    }
    _write_json_atomic(output_root / "progress.json", progress)
    return aggregates


def _number_slug(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def run_temporal_kmeans_grid(
    *,
    manifest_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    pair_count: int = DEFAULT_PAIR_COUNT,
    durations_seconds: Sequence[float] = DEFAULT_DURATIONS_SECONDS,
    time_weights: Sequence[float] = DEFAULT_TIME_WEIGHTS,
    seconds_per_cluster_values: Sequence[float] = DEFAULT_SECONDS_PER_CLUSTER,
    similarity_thresholds: Sequence[float] = DEFAULT_SIMILARITY_THRESHOLDS,
    temporal_unit_seconds: float = DEFAULT_TEMPORAL_UNIT_SECONDS,
    sample_interval_seconds: float = 1.0,
    model_id: str = DEFAULT_CLIP_MODEL,
    device: str = "auto",
    clip_batch_size: int = 32,
    max_iterations: int = 25,
    min_group_size: int = 2,
    max_groups: int | None = None,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "min_percent",
    min_pruned_video_percent: float | None = 20.0,
    random_seed: int = DEFAULT_RANDOM_SEED,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
    trace_pair_limit: int = 3,
    resume_pairs_dir: str | Path | None = None,
    encoder: ImageEncoder | None = None,
) -> dict[str, Any]:
    """Run a nested-duration grid on one fixed cohort of complete 10-minute pairs."""

    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if clip_batch_size <= 0 or max_iterations <= 0:
        raise ValueError("batch size and max iterations must be positive")
    if trace_pair_limit < 0:
        raise ValueError("trace_pair_limit must be non-negative")
    durations = sorted(
        parse_float_grid(durations_seconds, name="duration", strictly_positive=True)
    )
    weights = parse_float_grid(time_weights, name="time weight", minimum=0.0)
    if 0.0 not in weights:
        raise ValueError("time weight grid must include 0 for the exact current baseline")
    cluster_densities = parse_float_grid(
        seconds_per_cluster_values,
        name="seconds per cluster",
        strictly_positive=True,
    )
    thresholds = parse_float_grid(
        similarity_thresholds,
        name="similarity threshold",
        minimum=-1.0,
        maximum=1.0,
    )
    if temporal_unit_seconds <= 0:
        raise ValueError("temporal_unit_seconds must be positive")
    longest_duration = max(durations)
    source_ratio = longest_duration / SOURCE_CLIP_DURATION_SECONDS
    if abs(source_ratio - round(source_ratio)) > 1e-9:
        raise ValueError("maximum duration must be a multiple of the 30-second source clips")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = read_json(manifest_path)
    groups = [
        group
        for group in group_manifest_clips(
            manifest, evidence_duration_seconds=longest_duration
        )
        if len(group.get("clips", [])) >= min_group_size
    ]
    rng = random.Random(random_seed)
    rng.shuffle(groups)
    if max_groups is not None:
        groups = groups[:max_groups]
    clip_encoder = encoder
    effective_model_id = encoder.model_id if encoder is not None else model_id

    resume_root = Path(resume_pairs_dir) if resume_pairs_dir else None
    resume_pair_dirs: dict[str, Path] = {}
    if resume_root is not None:
        if not resume_root.is_dir():
            raise FileNotFoundError(f"resume pairs directory does not exist: {resume_root}")
        resume_pair_dirs = {
            child.name: child
            for child in resume_root.iterdir()
            if child.is_dir()
            and (child / "embedding_cache.json").is_file()
            and (child / "embedding_cache.npz").is_file()
        }
        print(
            f"resume_cache_candidates={len(resume_pair_dirs)} "
            f"resume_pairs_dir={resume_root}",
            flush=True,
        )

    variants = build_grid_variants(
        durations, cluster_densities, weights, thresholds
    )
    cohort_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    checkpoint_completed_pair_ids: set[str] = set()
    if resume_root is not None:
        checkpoint_cohort_path = resume_root.parent / "cohort.jsonl"
        checkpoint_metrics_path = resume_root.parent / "grid_metrics.jsonl"
        if checkpoint_cohort_path.is_file() and checkpoint_metrics_path.is_file():
            checkpoint_cohort = list(iter_jsonl(checkpoint_cohort_path))
            checkpoint_metrics = list(iter_jsonl(checkpoint_metrics_path))
            metrics_by_pair: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in checkpoint_metrics:
                metrics_by_pair[str(row.get("pair_id"))].append(row)
            expected_configuration_keys = {
                (
                    float(variant["duration_seconds"]),
                    float(variant["seconds_per_cluster"]),
                    int(variant["k"]),
                    float(variant["time_weight"]),
                    float(temporal_unit_seconds),
                    float(variant["high_similarity_threshold"]),
                )
                for variant in variants
            }
            for cached_pair in checkpoint_cohort:
                pair_id = str(cached_pair.get("pair_id") or "")
                source_pair_dir = resume_pair_dirs.get(pair_id)
                selected_metrics = metrics_by_pair.get(pair_id, [])
                selected_keys = {
                    (
                        float(row["duration_seconds"]),
                        float(row["seconds_per_cluster"]),
                        int(row["k"]),
                        float(row["time_weight"]),
                        float(row["temporal_unit_seconds"]),
                        float(row["high_similarity_threshold"]),
                    )
                    for row in selected_metrics
                    if float(row.get("sample_interval_seconds", -1.0))
                    == float(sample_interval_seconds)
                }
                if (
                    source_pair_dir is None
                    or len(selected_metrics) != len(variants)
                    or selected_keys != expected_configuration_keys
                ):
                    continue
                cache_metadata = read_json(source_pair_dir / "embedding_cache.json")
                if str(cache_metadata.get("model_id") or "") != effective_model_id:
                    continue
                target_pair_dir = output_root / "pairs" / pair_id
                _copy_pair_embedding_cache(source_pair_dir, target_pair_dir)
                resumed_pair = {
                    **cached_pair,
                    "embedding_cache": str(target_pair_dir / "embedding_cache.npz"),
                    "embedding_cache_reused": True,
                    "resume_source_pair_dir": str(source_pair_dir),
                }
                cohort_rows.append(resumed_pair)
                metric_rows.extend(selected_metrics)
                checkpoint_completed_pair_ids.add(pair_id)
                _write_json_atomic(
                    target_pair_dir / "pair_complete.json",
                    {
                        "pair_id": pair_id,
                        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "configuration_count": len(selected_metrics),
                        "embedding_cache_reused": True,
                        "metrics_reused": True,
                        "resume_source_pair_dir": str(source_pair_dir),
                    },
                )
            if checkpoint_completed_pair_ids:
                print(
                    f"resume_checkpoint_pairs={len(checkpoint_completed_pair_ids)} "
                    f"resume_checkpoint_metrics={len(metric_rows)}",
                    flush=True,
                )
    _write_progress_checkpoint(
        output_root,
        cohort_rows=cohort_rows,
        metric_rows=metric_rows,
        skipped=skipped,
        pair_count_target=pair_count,
        configuration_count_per_pair=len(variants),
        resume_pairs_dir=str(resume_root) if resume_root is not None else None,
        resume_cache_candidates=len(resume_pair_dirs),
        status="running",
    )
    for group_index, group in enumerate(groups):
        if len(cohort_rows) >= pair_count:
            break
        try:
            selected = _select_pair(group, rng)
            agents = [str(clip.get("agent_dir")) for clip in selected]
            pair_id = stable_id(
                group.get("day"), group.get("time_token"), *agents, "temporal_kmeans"
            )
            if pair_id in checkpoint_completed_pair_ids:
                continue
            pair_dir = output_root / "pairs" / pair_id
            resume_pair_dir = resume_pair_dirs.get(pair_id)
            embedding_cache_reused = resume_pair_dir is not None
            if resume_pair_dir is not None:
                frames_by_side, embeddings_by_side, _ = _load_pair_embedding_cache(
                    resume_pair_dir,
                    expected_model_id=effective_model_id,
                    expected_frame_count=_expected_frame_count(
                        longest_duration, sample_interval_seconds
                    ),
                )
                _copy_pair_embedding_cache(resume_pair_dir, pair_dir)
            else:
                frames_by_side = [
                    _sample_segmented_clip(
                        clip,
                        output_dir=pair_dir / "sampled_frames" / side,
                        cache_dir=cache_dir,
                        duration_seconds=longest_duration,
                        sample_interval_seconds=sample_interval_seconds,
                        ffmpeg_binary=ffmpeg_binary,
                        download_media=download_media,
                    )
                    for side, clip in zip(("left", "right"), selected)
                ]
                if clip_encoder is None:
                    clip_encoder = TransformersClipEncoder(model_id, device=device)
                    effective_model_id = clip_encoder.model_id
                embeddings_by_side = _load_or_encode_pair(
                    pair_dir,
                    frames_by_side,
                    encoder=clip_encoder,
                    clip_batch_size=clip_batch_size,
                )
            pair = {
                "pair_id": pair_id,
                "day": group.get("day"),
                "time_token": group.get("time_token"),
                "left_agent": agents[0],
                "right_agent": agents[1],
                "duration_seconds_available": longest_duration,
                "sample_interval_seconds": sample_interval_seconds,
                "left_frame_count": len(frames_by_side[0]),
                "right_frame_count": len(frames_by_side[1]),
                "embedding_cache": str(pair_dir / "embedding_cache.npz"),
                "embedding_cache_reused": embedding_cache_reused,
                "resume_source_pair_dir": (
                    str(resume_pair_dir) if resume_pair_dir is not None else None
                ),
                "source_segment_count": int(group.get("segment_count") or 0),
            }
            pair_metric_rows: list[dict[str, Any]] = []

            duration_cache: dict[
                float,
                tuple[
                    list[list[dict[str, Any]]],
                    list[np.ndarray],
                    list[list[float]],
                ],
            ] = {}
            cluster_cache: dict[
                tuple[float, float, float],
                tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
            ] = {}
            for variant in variants:
                duration = float(variant["duration_seconds"])
                if duration not in duration_cache:
                    prefix_frames = []
                    prefix_embeddings = []
                    for frames, embeddings in zip(frames_by_side, embeddings_by_side):
                        current_frames, current_embeddings = _duration_prefix(
                            frames,
                            embeddings,
                            duration,
                            sample_interval_seconds,
                        )
                        prefix_frames.append(current_frames)
                        prefix_embeddings.append(current_embeddings)
                    full_matrix = _cosine_matrix(
                        prefix_embeddings[0], prefix_embeddings[1]
                    )
                    duration_cache[duration] = (
                        prefix_frames,
                        prefix_embeddings,
                        full_matrix,
                    )
                prefix_frames, prefix_embeddings, full_matrix = duration_cache[duration]
                cache_key = (
                    duration,
                    float(variant["seconds_per_cluster"]),
                    float(variant["time_weight"]),
                )
                if cache_key not in cluster_cache:
                    k = int(variant["k"])
                    left_clusters = time_aware_clustered_frame_representatives(
                        prefix_frames[0],
                        prefix_embeddings[0],
                        cluster_count=k,
                        time_weight=float(variant["time_weight"]),
                        temporal_unit_seconds=temporal_unit_seconds,
                        max_iterations=max_iterations,
                    )
                    right_clusters = time_aware_clustered_frame_representatives(
                        prefix_frames[1],
                        prefix_embeddings[1],
                        cluster_count=k,
                        time_weight=float(variant["time_weight"]),
                        temporal_unit_seconds=temporal_unit_seconds,
                        max_iterations=max_iterations,
                    )
                    cluster_cache[cache_key] = (
                        left_clusters,
                        right_clusters,
                        cluster_quality_metrics(left_clusters, prefix_embeddings[0]),
                        cluster_quality_metrics(right_clusters, prefix_embeddings[1]),
                    )
                left_clusters, right_clusters, left_quality, right_quality = cluster_cache[
                    cache_key
                ]
                pruning = prune_time_aware_cluster_pair(
                    prefix_frames[0],
                    prefix_frames[1],
                    prefix_embeddings[0],
                    prefix_embeddings[1],
                    left_clusters,
                    right_clusters,
                    full_frame_matrix=full_matrix,
                    start_seconds=0.0,
                    duration_seconds=duration,
                    sample_interval_seconds=sample_interval_seconds,
                    high_similarity_threshold=float(
                        variant["high_similarity_threshold"]
                    ),
                    min_pruned_video_seconds=min_pruned_video_seconds,
                    pruning_protection_mode=pruning_protection_mode,
                    min_pruned_video_percent=min_pruned_video_percent,
                )
                diagnostics_path: Path | None = None
                if len(cohort_rows) < trace_pair_limit:
                    diagnostics_path = (
                        pair_dir
                        / "diagnostics"
                        / f"duration_{_number_slug(duration)}s"
                        / f"spc_{_number_slug(float(variant['seconds_per_cluster']))}"
                        / f"w_{_number_slug(float(variant['time_weight']))}"
                        / f"threshold_{_number_slug(float(variant['high_similarity_threshold']))}.json"
                    )
                    write_json(
                        diagnostics_path,
                        {
                            "pair": pair,
                            "configuration": {
                                **variant,
                                "temporal_unit_seconds": temporal_unit_seconds,
                                "distance_formula": CLUSTER_DISTANCE_FORMULA,
                            },
                            "left_cluster_quality": left_quality,
                            "right_cluster_quality": right_quality,
                            "temporal_pruning": pruning,
                        },
                    )
                pair_metric_rows.append(
                    _metric_row(
                        pair=pair,
                        duration_seconds=duration,
                        sample_interval_seconds=sample_interval_seconds,
                        seconds_per_cluster=float(variant["seconds_per_cluster"]),
                        k=int(variant["k"]),
                        time_weight=float(variant["time_weight"]),
                        temporal_unit_seconds=temporal_unit_seconds,
                        threshold=float(variant["high_similarity_threshold"]),
                        left_quality=left_quality,
                        right_quality=right_quality,
                        pruning=pruning,
                        diagnostics_path=(
                            str(diagnostics_path) if diagnostics_path is not None else None
                        ),
                    )
                )
            if len(pair_metric_rows) != len(variants):
                raise RuntimeError(
                    f"incomplete pair grid: pair={pair_id} "
                    f"expected={len(variants)} actual={len(pair_metric_rows)}"
                )
            metric_rows.extend(pair_metric_rows)
            cohort_rows.append(pair)
            _write_json_atomic(
                pair_dir / "pair_complete.json",
                {
                    "pair_id": pair_id,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "configuration_count": len(pair_metric_rows),
                    "embedding_cache_reused": embedding_cache_reused,
                    "resume_source_pair_dir": (
                        str(resume_pair_dir) if resume_pair_dir is not None else None
                    ),
                },
            )
            _write_progress_checkpoint(
                output_root,
                cohort_rows=cohort_rows,
                metric_rows=metric_rows,
                skipped=skipped,
                pair_count_target=pair_count,
                configuration_count_per_pair=len(variants),
                resume_pairs_dir=str(resume_root) if resume_root is not None else None,
                resume_cache_candidates=len(resume_pair_dirs),
                status="running",
            )
            print(
                f"progress_pairs={len(cohort_rows)}/{pair_count} "
                f"pair_id={pair_id} cache_reused={int(embedding_cache_reused)} "
                f"metrics={len(metric_rows)}",
                flush=True,
            )
        except Exception as exc:
            skipped.append(
                {
                    "group_index": group_index,
                    "day": group.get("day"),
                    "time_token": group.get("time_token"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            _write_progress_checkpoint(
                output_root,
                cohort_rows=cohort_rows,
                metric_rows=metric_rows,
                skipped=skipped,
                pair_count_target=pair_count,
                configuration_count_per_pair=len(variants),
                resume_pairs_dir=str(resume_root) if resume_root is not None else None,
                resume_cache_candidates=len(resume_pair_dirs),
                status="running",
            )
            print(
                f"progress_skip group_index={group_index} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )

    target_met = len(cohort_rows) >= pair_count
    aggregates = _write_progress_checkpoint(
        output_root,
        cohort_rows=cohort_rows,
        metric_rows=metric_rows,
        skipped=skipped,
        pair_count_target=pair_count,
        configuration_count_per_pair=len(variants),
        resume_pairs_dir=str(resume_root) if resume_root is not None else None,
        resume_cache_candidates=len(resume_pair_dirs),
        status="complete" if target_met else "incomplete",
    )
    review_path = output_root / "summary.html"
    reused_count = sum(bool(row.get("embedding_cache_reused")) for row in cohort_rows)
    summary = {
        "manifest_path": str(manifest_path),
        "output_dir": str(output_root),
        "summary_html": str(review_path),
        "pair_count_requested": pair_count,
        "pair_count": len(cohort_rows),
        "target_met": target_met,
        "same_pair_cohort_for_all_durations": True,
        "nested_duration_prefixes": durations,
        "model_id": effective_model_id,
        "distance_formula": CLUSTER_DISTANCE_FORMULA,
        "isolation_contract": (
            "time changes within-video clustering only; cross-video matching and "
            "the CLIP threshold are unchanged"
        ),
        "settings": {
            "durations_seconds": durations,
            "time_weights": weights,
            "seconds_per_cluster_values": cluster_densities,
            "similarity_thresholds": thresholds,
            "temporal_unit_seconds": temporal_unit_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "clip_batch_size": clip_batch_size,
            "max_iterations": max_iterations,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "min_pruned_video_percent": min_pruned_video_percent,
            "random_seed": random_seed,
            "trace_pair_limit": trace_pair_limit,
            "configuration_count_per_pair": len(variants),
            "resume_pairs_dir": str(resume_root) if resume_root is not None else None,
        },
        "resume_cache_candidates": len(resume_pair_dirs),
        "resume_cache_pairs_reused": reused_count,
        "new_pairs_completed": len(cohort_rows) - reused_count,
        "progress_path": str(output_root / "progress.json"),
        "expected_metric_count": pair_count * len(variants),
        "metric_count": len(metric_rows),
        "aggregate_count": len(aggregates),
        "skipped_group_count": len(skipped),
        "skipped_groups": skipped,
    }
    _write_json_atomic(output_root / "summary.json", summary)
    if not summary["target_met"]:
        raise RuntimeError(
            f"requested {pair_count} complete pairs but produced {len(cohort_rows)}; "
            f"inspect {output_root / 'summary.json'}"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search a time dimension in sidecar cosine K-means pruning"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument(
        "--durations-seconds",
        default=",".join(f"{value:g}" for value in DEFAULT_DURATIONS_SECONDS),
    )
    parser.add_argument(
        "--time-weights",
        default=",".join(f"{value:g}" for value in DEFAULT_TIME_WEIGHTS),
    )
    parser.add_argument(
        "--seconds-per-cluster-values",
        default=",".join(f"{value:g}" for value in DEFAULT_SECONDS_PER_CLUSTER),
    )
    parser.add_argument(
        "--similarity-thresholds",
        default=",".join(f"{value:g}" for value in DEFAULT_SIMILARITY_THRESHOLDS),
    )
    parser.add_argument(
        "--temporal-unit-seconds", type=float, default=DEFAULT_TEMPORAL_UNIT_SECONDS
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--min-group-size", type=int, default=2)
    parser.add_argument("--max-groups", type=int)
    parser.add_argument("--min-pruned-video-seconds", type=float, default=8.0)
    parser.add_argument(
        "--pruning-protection-mode",
        choices=["reject", "min_seconds", "min_percent"],
        default="min_percent",
    )
    parser.add_argument("--min-pruned-video-percent", type=float, default=20.0)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--download-media", action="store_true")
    parser.add_argument("--trace-pair-limit", type=int, default=3)
    parser.add_argument(
        "--resume-pairs-dir",
        help=(
            "Reuse complete embedding_cache.json/.npz pairs from an earlier "
            "experiment/pairs directory"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    durations = parse_float_grid(
        args.durations_seconds, name="duration", strictly_positive=True
    )
    time_weights = parse_float_grid(
        args.time_weights, name="time weight", minimum=0.0
    )
    seconds_per_cluster_values = parse_float_grid(
        args.seconds_per_cluster_values,
        name="seconds per cluster",
        strictly_positive=True,
    )
    similarity_thresholds = parse_float_grid(
        args.similarity_thresholds,
        name="similarity threshold",
        minimum=-1.0,
        maximum=1.0,
    )
    summary = run_temporal_kmeans_grid(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        pair_count=args.pair_count,
        durations_seconds=durations,
        time_weights=time_weights,
        seconds_per_cluster_values=seconds_per_cluster_values,
        similarity_thresholds=similarity_thresholds,
        temporal_unit_seconds=args.temporal_unit_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        model_id=args.model_id,
        device=args.device,
        clip_batch_size=args.clip_batch_size,
        max_iterations=args.max_iterations,
        min_group_size=args.min_group_size,
        max_groups=args.max_groups,
        min_pruned_video_seconds=args.min_pruned_video_seconds,
        pruning_protection_mode=args.pruning_protection_mode,
        min_pruned_video_percent=args.min_pruned_video_percent,
        random_seed=args.random_seed,
        ffmpeg_binary=args.ffmpeg_binary,
        download_media=args.download_media,
        trace_pair_limit=args.trace_pair_limit,
        resume_pairs_dir=args.resume_pairs_dir,
        encoder=None,
    )
    print(
        f"wrote {summary['metric_count']} variants for {summary['pair_count']} fixed pairs "
        f"to {summary['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
