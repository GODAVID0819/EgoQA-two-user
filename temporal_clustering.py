"""Deterministic clustering helpers whose clusters are contiguous in time."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Sequence


COSINE_KMEANS_CLUSTERING = "cosine_kmeans"
TEMPORAL_AGGLOMERATIVE_CLUSTERING = "temporal_agglomerative"
CLUSTERING_METHODS = (
    COSINE_KMEANS_CLUSTERING,
    TEMPORAL_AGGLOMERATIVE_CLUSTERING,
)


def _normalize(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(float(value) * float(value) for value in vector))
    if norm <= 0:
        return [0.0 for _ in vector]
    return [float(value) / norm for value in vector]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(float(a) * float(b) for a, b in zip(left, right))


@dataclass
class _TemporalCluster:
    cluster_id: int
    member_indices: tuple[int, ...]
    vector_sum: list[float]
    centroid: list[float]
    previous_id: int | None
    next_id: int | None
    active: bool = True


def temporal_agglomerative_embedding_medoids(
    embeddings: Sequence[Sequence[float]],
    cluster_count: int,
    *,
    timestamps: Sequence[float] | None = None,
) -> tuple[list[int], list[int]]:
    """Return labels and medoids from adjacency-constrained agglomeration.

    Every sampled frame begins as a singleton. At each step, the most visually
    similar pair of *temporally adjacent* clusters is merged. Because no
    non-neighboring merge is legal, every final cluster is one contiguous run
    in chronological sampled-frame order. Merging stops at exactly ``K`` (or
    the number of frames when fewer than ``K`` frames are available).
    """

    if cluster_count <= 0:
        raise ValueError("cluster_count must be positive")
    if not embeddings:
        return [], []
    width = len(embeddings[0])
    if width <= 0 or any(len(vector) != width for vector in embeddings):
        raise ValueError("embedding dimensions must be non-empty and equal")
    if timestamps is not None and len(timestamps) != len(embeddings):
        raise ValueError("timestamp and embedding counts must match")

    normalized = [_normalize(vector) for vector in embeddings]
    if timestamps is None:
        chronological_indices = list(range(len(normalized)))
    else:
        timestamp_values = [float(value) for value in timestamps]
        if any(not math.isfinite(value) for value in timestamp_values):
            raise ValueError("timestamps must be finite")
        chronological_indices = sorted(
            range(len(normalized)),
            key=lambda index: (timestamp_values[index], index),
        )
    chronological_positions = {
        frame_index: order_index
        for order_index, frame_index in enumerate(chronological_indices)
    }

    clusters: dict[int, _TemporalCluster] = {}
    for order_index, frame_index in enumerate(chronological_indices):
        vector = normalized[frame_index]
        clusters[order_index] = _TemporalCluster(
            cluster_id=order_index,
            member_indices=(frame_index,),
            vector_sum=list(vector),
            centroid=list(vector),
            previous_id=order_index - 1 if order_index > 0 else None,
            next_id=order_index + 1 if order_index + 1 < len(normalized) else None,
        )

    heap: list[tuple[float, int, int]] = []

    def push_pair(left_id: int | None, right_id: int | None) -> None:
        if left_id is None or right_id is None:
            return
        left = clusters[left_id]
        right = clusters[right_id]
        if not left.active or not right.active or left.next_id != right_id:
            return
        similarity = _cosine(left.centroid, right.centroid)
        heapq.heappush(heap, (-similarity, left_id, right_id))

    for cluster_id in range(len(normalized) - 1):
        push_pair(cluster_id, cluster_id + 1)

    target_count = min(cluster_count, len(normalized))
    active_count = len(normalized)
    next_cluster_id = len(normalized)
    while active_count > target_count:
        while heap:
            _, left_id, right_id = heapq.heappop(heap)
            left = clusters[left_id]
            right = clusters[right_id]
            if left.active and right.active and left.next_id == right_id:
                break
        else:  # pragma: no cover - the active chain always has an adjacent pair
            raise RuntimeError("temporal cluster adjacency chain became disconnected")

        vector_sum = [
            left.vector_sum[index] + right.vector_sum[index]
            for index in range(width)
        ]
        merged = _TemporalCluster(
            cluster_id=next_cluster_id,
            member_indices=left.member_indices + right.member_indices,
            vector_sum=vector_sum,
            centroid=_normalize(vector_sum),
            previous_id=left.previous_id,
            next_id=right.next_id,
        )
        left.active = False
        right.active = False
        clusters[next_cluster_id] = merged
        if merged.previous_id is not None:
            clusters[merged.previous_id].next_id = next_cluster_id
        if merged.next_id is not None:
            clusters[merged.next_id].previous_id = next_cluster_id
        push_pair(merged.previous_id, next_cluster_id)
        push_pair(next_cluster_id, merged.next_id)
        next_cluster_id += 1
        active_count -= 1

    final_clusters = sorted(
        (cluster for cluster in clusters.values() if cluster.active),
        key=lambda cluster: chronological_positions[cluster.member_indices[0]],
    )
    labels = [-1 for _ in normalized]
    medoids: list[int] = []
    for output_cluster_index, cluster in enumerate(final_clusters):
        for frame_index in cluster.member_indices:
            labels[frame_index] = output_cluster_index
        medoids.append(
            max(
                cluster.member_indices,
                key=lambda frame_index: (
                    _cosine(normalized[frame_index], cluster.centroid),
                    -chronological_positions[frame_index],
                ),
            )
        )
    return labels, medoids
