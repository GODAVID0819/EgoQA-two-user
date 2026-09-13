"""Sidecar CLIP-pruned sampler for synchronized EgoLife participant pairs.

This is intentionally separate from the main evidence pipeline. It starts from
the full manifest and either randomly selects a synchronized two-video pair or
fills fixed time-bin and participant-pair quotas. It emits candidate packets
with paired original/pruned videos. The selected videos are sampled at one frame
per second, embedded with CLIP, clustered within each video, compared through
cluster medoids, and high-similarity clusters are removed as temporal intervals.
Comparing all videos in a synchronized group is available only as an explicit
slow opt-in.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, deque
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

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
    from .manifest import seconds_from_time_token
    from .schema import extract_json_object
    from .temporal_clustering import (
        CLUSTERING_METHODS,
        COSINE_KMEANS_CLUSTERING,
        TEMPORAL_AGGLOMERATIVE_CLUSTERING,
        temporal_agglomerative_embedding_medoids,
    )
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
    from egolife_two_user_qa.manifest import seconds_from_time_token
    from egolife_two_user_qa.schema import extract_json_object
    from egolife_two_user_qa.temporal_clustering import (
        CLUSTERING_METHODS,
        COSINE_KMEANS_CLUSTERING,
        TEMPORAL_AGGLOMERATIVE_CLUSTERING,
        temporal_agglomerative_embedding_medoids,
    )


RANDOM_SAMPLING_POLICY = "random"
BALANCED_TIME_PAIR_SAMPLING_POLICY = "balanced_time_pair"
SAMPLING_POLICIES = (RANDOM_SAMPLING_POLICY, BALANCED_TIME_PAIR_SAMPLING_POLICY)
DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS = 2.0
DEFAULT_SPLIT_NONCONTIGUOUS_CLUSTERS = True
DEFAULT_CLUSTER_SUMMARY_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


class ClusterSummaryRunner(Protocol):
    model_id: str

    def generate(
        self,
        prompt: str,
        image_paths: list[str] | None = None,
        video_paths: list[str] | None = None,
        decoding_mode: str = "greedy",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int | None = None,
    ) -> str:
        ...


def _group_clock_seconds(group: dict[str, Any]) -> float:
    value = group.get("clock_seconds")
    if value is not None:
        return float(value)
    return seconds_from_time_token(str(group.get("time_token") or ""))


def stable_group_partition(
    group: dict[str, Any],
    *,
    partition_count: int,
) -> int:
    """Assign a synchronized source group to a stable disjoint partition."""

    if partition_count <= 0:
        raise ValueError("partition_count must be positive")
    identity = f"{group.get('day') or ''}\0{group.get('time_token') or ''}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % partition_count


def _group_pair_options(group: dict[str, Any]) -> list[tuple[str, str]]:
    agents = sorted(
        {
            str(clip.get("agent_dir") or "").strip()
            for clip in group.get("clips", [])
            if str(clip.get("agent_dir") or "").strip()
        }
    )
    return list(combinations(agents, 2))


def _pair_label(pair: tuple[str, str]) -> str:
    return "+".join(pair)


@dataclass(frozen=True)
class BalancedSamplingDecision:
    """One scheduled synchronized group and its preselected participant pair."""

    time_bin_index: int
    group: dict[str, Any]
    pair: tuple[str, str]


class BalancedTimePairScheduler:
    """Fill fixed clock-time strata while balancing pair and participant usage."""

    def __init__(
        self,
        groups: list[dict[str, Any]],
        *,
        target_count: int,
        time_bin_count: int,
        random_seed: int | None,
        max_attempts: int | None = None,
        timeline_start_seconds: float | None = None,
        timeline_end_seconds: float | None = None,
        pair_quota_rotation: int = 0,
        coordination_groups: list[dict[str, Any]] | None = None,
        coordination_partition_count: int = 1,
    ) -> None:
        if target_count <= 0:
            raise ValueError("target_count must be positive")
        if time_bin_count <= 0:
            raise ValueError("time_bin_count must be positive")
        if max_attempts is not None and max_attempts <= 0:
            raise ValueError("max_attempts must be positive when provided")
        if coordination_partition_count <= 0:
            raise ValueError("coordination_partition_count must be positive")
        if not 0 <= pair_quota_rotation < coordination_partition_count:
            raise ValueError(
                "pair_quota_rotation must be in [0, coordination_partition_count)"
            )
        if not groups:
            raise ValueError("balanced sampling requires at least one synchronized group")

        clocks = [_group_clock_seconds(group) for group in groups]
        observed_start = min(clocks)
        observed_end = max(clocks) + 0.01
        self.timeline_start_seconds = (
            observed_start if timeline_start_seconds is None else float(timeline_start_seconds)
        )
        self.timeline_end_seconds = (
            observed_end if timeline_end_seconds is None else float(timeline_end_seconds)
        )
        if self.timeline_end_seconds <= self.timeline_start_seconds:
            self.timeline_end_seconds = self.timeline_start_seconds + 0.01

        self.target_count = target_count
        self.time_bin_count = time_bin_count
        self.max_attempts = max_attempts
        self.random_seed = random_seed
        self.pair_quota_rotation = pair_quota_rotation
        self.bin_width_seconds = (
            self.timeline_end_seconds - self.timeline_start_seconds
        ) / time_bin_count
        self.groups_by_bin: list[list[dict[str, Any]]] = [
            [] for _ in range(time_bin_count)
        ]
        for group in groups:
            bin_index = self.time_bin_for_clock(_group_clock_seconds(group))
            self.groups_by_bin[bin_index].append(group)
        planning_groups = coordination_groups or groups
        planning_groups_by_partition_and_bin: list[list[list[dict[str, Any]]]] = [
            [[] for _ in range(time_bin_count)]
            for _ in range(coordination_partition_count)
        ]
        for group in planning_groups:
            partition_index = stable_group_partition(
                group,
                partition_count=coordination_partition_count,
            )
            bin_index = self.time_bin_for_clock(_group_clock_seconds(group))
            planning_groups_by_partition_and_bin[partition_index][bin_index].append(group)
        self.coordination_group_count = len(planning_groups)
        self.coordination_partition_count = coordination_partition_count

        capacities = [len(bin_groups) for bin_groups in self.groups_by_bin]
        if sum(capacities) < target_count:
            raise ValueError(
                f"balanced sampling has only {sum(capacities)} groups for target {target_count}"
            )
        self.bin_quotas = [0 for _ in range(time_bin_count)]
        remaining = target_count
        while remaining:
            allocated = False
            for bin_index, capacity in enumerate(capacities):
                if self.bin_quotas[bin_index] >= capacity:
                    continue
                self.bin_quotas[bin_index] += 1
                remaining -= 1
                allocated = True
                if remaining == 0:
                    break
            if not allocated:
                raise ValueError("could not allocate balanced time-bin quotas")

        self.available_pairs = sorted(
            {pair for group in planning_groups for pair in _group_pair_options(group)}
        )
        if not self.available_pairs:
            raise ValueError("balanced sampling found no two-participant pair options")
        pair_order = sorted(
            self.available_pairs,
            key=lambda pair: hashlib.sha256(
                f"{random_seed}\0pair-quota\0{_pair_label(pair)}".encode("utf-8")
            ).hexdigest(),
        )
        base_pair_quota, extra_pair_quota = divmod(target_count, len(pair_order))

        def pair_quotas_for_rotation(rotation: int) -> dict[tuple[str, str], int]:
            extra_start = (rotation * extra_pair_quota) % len(pair_order)
            extra_pairs = {
                pair_order[(extra_start + offset) % len(pair_order)]
                for offset in range(extra_pair_quota)
            }
            return {
                pair: base_pair_quota + int(pair in extra_pairs) for pair in pair_order
            }

        self.pair_quotas = pair_quotas_for_rotation(pair_quota_rotation)

        available_pairs_by_partition_and_bin = [
            [
                sorted({pair for group in bin_groups for pair in _group_pair_options(group)})
                for bin_groups in partition_groups_by_bin
            ]
            for partition_groups_by_bin in planning_groups_by_partition_and_bin
        ]
        self.available_pairs_by_bin = available_pairs_by_partition_and_bin[
            pair_quota_rotation
        ]
        def allocate_pair_bin_quotas(
            pair_quotas: dict[tuple[str, str], int],
            *,
            rotation: int,
            forbidden_pair_bins: set[tuple[int, tuple[str, str]]],
        ) -> Counter[tuple[int, tuple[str, str]]]:
            source = ("source", rotation)
            sink = ("sink", rotation)
            adjacency: dict[tuple[Any, ...], list[tuple[Any, ...]]] = {}
            residual: dict[tuple[tuple[Any, ...], tuple[Any, ...]], int] = {}

            def add_edge(
                left: tuple[Any, ...],
                right: tuple[Any, ...],
                capacity: int,
            ) -> None:
                adjacency.setdefault(left, []).append(right)
                adjacency.setdefault(right, []).append(left)
                residual[(left, right)] = capacity
                residual[(right, left)] = 0

            pair_nodes = {pair: ("pair", *pair) for pair in self.available_pairs}
            bin_nodes = {
                bin_index: ("bin", rotation, bin_index)
                for bin_index in range(time_bin_count)
            }
            for pair in sorted(
                self.available_pairs,
                key=lambda value: hashlib.sha256(
                    (
                        f"{random_seed}\0pair-bin-flow\0{rotation}"
                        f"\0{_pair_label(value)}"
                    ).encode("utf-8")
                ).hexdigest(),
            ):
                add_edge(source, pair_nodes[pair], pair_quotas[pair])
                candidate_bins = [
                    bin_index
                    for bin_index in range(time_bin_count)
                    if pair in available_pairs_by_partition_and_bin[rotation][bin_index]
                    and (bin_index, pair) not in forbidden_pair_bins
                    and self.bin_quotas[bin_index] > 0
                ]
                candidate_bins.sort(
                    key=lambda bin_index: hashlib.sha256(
                        (
                            f"{random_seed}\0pair-bin-flow\0{rotation}"
                            f"\0{_pair_label(pair)}\0{bin_index}"
                        ).encode("utf-8")
                    ).hexdigest()
                )
                for bin_index in candidate_bins:
                    add_edge(pair_nodes[pair], bin_nodes[bin_index], 1)
            for bin_index in range(time_bin_count):
                add_edge(bin_nodes[bin_index], sink, self.bin_quotas[bin_index])

            flow = 0
            while True:
                parents: dict[tuple[Any, ...], tuple[Any, ...] | None] = {source: None}
                queue = deque([source])
                while queue and sink not in parents:
                    left = queue.popleft()
                    for right in adjacency.get(left, []):
                        if right in parents or residual.get((left, right), 0) <= 0:
                            continue
                        parents[right] = left
                        queue.append(right)
                        if right == sink:
                            break
                if sink not in parents:
                    break
                node = sink
                while parents[node] is not None:
                    parent = parents[node]
                    residual[(parent, node)] -= 1
                    residual[(node, parent)] += 1
                    node = parent
                flow += 1

            required_flow = sum(pair_quotas.values())
            if flow != required_flow:
                raise ValueError(
                    f"coordinated partition {rotation} pair/time quota flow reached "
                    f"{flow}, expected {required_flow}"
                )
            plan: Counter[tuple[int, tuple[str, str]]] = Counter()
            for pair, pair_node in pair_nodes.items():
                for bin_index, bin_node in bin_nodes.items():
                    if residual.get((bin_node, pair_node), 0) > 0:
                        plan[(bin_index, pair)] = 1
            return plan

        forbidden_pair_bins: set[tuple[int, tuple[str, str]]] = set()
        for prior_rotation in range(pair_quota_rotation):
            prior_plan = allocate_pair_bin_quotas(
                pair_quotas_for_rotation(prior_rotation),
                rotation=prior_rotation,
                forbidden_pair_bins=forbidden_pair_bins,
            )
            forbidden_pair_bins.update(
                key for key, count in prior_plan.items() if count > 0
            )
        self.pair_quotas_by_bin = allocate_pair_bin_quotas(
            self.pair_quotas,
            rotation=pair_quota_rotation,
            forbidden_pair_bins=forbidden_pair_bins,
        )

        self.accepted_by_bin: Counter[int] = Counter()
        self.accepted_pair_global: Counter[tuple[str, str]] = Counter()
        self.accepted_pair_by_bin: Counter[tuple[int, tuple[str, str]]] = Counter()
        self.attempted_pair_global: Counter[tuple[str, str]] = Counter()
        self.attempted_pair_by_bin: Counter[tuple[int, tuple[str, str]]] = Counter()
        self.accepted_participant_global: Counter[str] = Counter()
        self.accepted_day_global: Counter[str] = Counter()
        self.accepted_day_by_bin: Counter[tuple[int, str]] = Counter()
        self.attempt_count = 0
        self.accepted_count = 0
        self._bin_cursor = 0

    def time_bin_for_clock(self, clock_seconds: float) -> int:
        relative = (float(clock_seconds) - self.timeline_start_seconds) / (
            self.timeline_end_seconds - self.timeline_start_seconds
        )
        return min(max(int(relative * self.time_bin_count), 0), self.time_bin_count - 1)

    def _tie_key(
        self,
        *,
        group: dict[str, Any],
        pair: tuple[str, str],
    ) -> str:
        value = (
            f"{self.random_seed}\0{group.get('day')}\0{group.get('time_token')}"
            f"\0{_pair_label(pair)}"
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _choice_score(
        self,
        *,
        bin_index: int,
        group: dict[str, Any],
        pair: tuple[str, str],
    ) -> tuple[Any, ...]:
        day = str(group.get("day") or "")
        pair_quota = self.pair_quotas[pair]
        pair_progress = (
            self.accepted_pair_global[pair] / pair_quota
            if pair_quota > 0
            else float("inf")
        )
        participant_counts = [self.accepted_participant_global[value] for value in pair]
        planned_in_bin = self.pair_quotas_by_bin[(bin_index, pair)]
        planned_quota_filled = (
            self.accepted_pair_by_bin[(bin_index, pair)] >= planned_in_bin
        )
        return (
            planned_quota_filled,
            self.accepted_pair_by_bin[(bin_index, pair)],
            pair_progress,
            self.attempted_pair_by_bin[(bin_index, pair)],
            self.attempted_pair_global[pair],
            self.accepted_day_by_bin[(bin_index, day)],
            self.accepted_day_global[day],
            max(participant_counts),
            sum(participant_counts),
            self._tie_key(group=group, pair=pair),
        )

    def next_decision(self) -> BalancedSamplingDecision | None:
        if self.accepted_count >= self.target_count:
            return None
        if self.max_attempts is not None and self.attempt_count >= self.max_attempts:
            return None

        chosen_bin = None
        for offset in range(self.time_bin_count):
            bin_index = (self._bin_cursor + offset) % self.time_bin_count
            if self.accepted_by_bin[bin_index] >= self.bin_quotas[bin_index]:
                continue
            if self.groups_by_bin[bin_index]:
                chosen_bin = bin_index
                break
        if chosen_bin is None:
            return None

        choices = [
            (self._choice_score(bin_index=chosen_bin, group=group, pair=pair), group, pair)
            for group in self.groups_by_bin[chosen_bin]
            for pair in _group_pair_options(group)
        ]
        if not choices:
            return None
        _, chosen_group, chosen_pair = min(choices, key=lambda row: row[0])
        self.groups_by_bin[chosen_bin].remove(chosen_group)
        self.attempted_pair_global[chosen_pair] += 1
        self.attempted_pair_by_bin[(chosen_bin, chosen_pair)] += 1
        self.attempt_count += 1
        self._bin_cursor = (chosen_bin + 1) % self.time_bin_count
        return BalancedSamplingDecision(chosen_bin, chosen_group, chosen_pair)

    def record_success(self, decision: BalancedSamplingDecision) -> None:
        day = str(decision.group.get("day") or "")
        self.accepted_by_bin[decision.time_bin_index] += 1
        self.accepted_pair_global[decision.pair] += 1
        self.accepted_pair_by_bin[(decision.time_bin_index, decision.pair)] += 1
        for participant in decision.pair:
            self.accepted_participant_global[participant] += 1
        self.accepted_day_global[day] += 1
        self.accepted_day_by_bin[(decision.time_bin_index, day)] += 1
        self.accepted_count += 1

    def decision_metadata(self, decision: BalancedSamplingDecision) -> dict[str, Any]:
        bin_index = decision.time_bin_index
        return {
            "policy": BALANCED_TIME_PAIR_SAMPLING_POLICY,
            "time_bin_index": bin_index,
            "time_bin_number": bin_index + 1,
            "time_bin_count": self.time_bin_count,
            "time_bin_start_seconds": round(
                self.timeline_start_seconds + bin_index * self.bin_width_seconds, 6
            ),
            "time_bin_end_seconds": round(
                self.timeline_start_seconds + (bin_index + 1) * self.bin_width_seconds,
                6,
            ),
            "time_bin_target_count": self.bin_quotas[bin_index],
            "preselected_agent_pair": list(decision.pair),
        }

    def strict_balance_errors(self) -> list[str]:
        errors = []
        if self.accepted_count != self.target_count:
            errors.append(
                f"accepted {self.accepted_count} packets, expected {self.target_count}"
            )
        for bin_index, quota in enumerate(self.bin_quotas):
            accepted = self.accepted_by_bin[bin_index]
            if accepted != quota:
                errors.append(
                    f"time bin {bin_index + 1} accepted {accepted}, expected {quota}"
                )
            pair_capacity = len(self.available_pairs_by_bin[bin_index])
            if pair_capacity:
                allowed_repeat = math.ceil(quota / pair_capacity)
                observed_repeat = max(
                    (
                        self.accepted_pair_by_bin[(bin_index, pair)]
                        for pair in self.available_pairs_by_bin[bin_index]
                    ),
                    default=0,
                )
                if observed_repeat > allowed_repeat:
                    errors.append(
                        f"time bin {bin_index + 1} repeats one pair {observed_repeat} times; "
                        f"balanced limit is {allowed_repeat}"
                    )
            for pair in self.available_pairs_by_bin[bin_index]:
                expected_pair_count = self.pair_quotas_by_bin[(bin_index, pair)]
                accepted_pair_count = self.accepted_pair_by_bin[(bin_index, pair)]
                if accepted_pair_count != expected_pair_count:
                    errors.append(
                        f"time bin {bin_index + 1} pair {_pair_label(pair)} accepted "
                        f"{accepted_pair_count}, expected {expected_pair_count}"
                    )
        for pair, quota in self.pair_quotas.items():
            accepted = self.accepted_pair_global[pair]
            if accepted != quota:
                errors.append(
                    f"pair {_pair_label(pair)} accepted {accepted}, expected {quota}"
                )
        return errors

    def summary(self) -> dict[str, Any]:
        return {
            "policy": BALANCED_TIME_PAIR_SAMPLING_POLICY,
            "target_count": self.target_count,
            "accepted_count": self.accepted_count,
            "attempt_count": self.attempt_count,
            "coordination_group_count": self.coordination_group_count,
            "coordination_partition_count": self.coordination_partition_count,
            "timeline_start_seconds": self.timeline_start_seconds,
            "timeline_end_seconds": self.timeline_end_seconds,
            "time_bin_count": self.time_bin_count,
            "time_bins": [
                {
                    "time_bin_index": bin_index,
                    "time_bin_number": bin_index + 1,
                    "start_seconds": round(
                        self.timeline_start_seconds + bin_index * self.bin_width_seconds,
                        6,
                    ),
                    "end_seconds": round(
                        self.timeline_start_seconds + (bin_index + 1) * self.bin_width_seconds,
                        6,
                    ),
                    "source_group_count": (
                        len(self.groups_by_bin[bin_index])
                        + self.accepted_by_bin[bin_index]
                        + sum(
                            self.attempted_pair_by_bin[(bin_index, pair)]
                            - self.accepted_pair_by_bin[(bin_index, pair)]
                            for pair in self.available_pairs_by_bin[bin_index]
                        )
                    ),
                    "target_count": self.bin_quotas[bin_index],
                    "accepted_count": self.accepted_by_bin[bin_index],
                    "pair_target_counts": {
                        _pair_label(pair): self.pair_quotas_by_bin[(bin_index, pair)]
                        for pair in self.available_pairs_by_bin[bin_index]
                        if self.pair_quotas_by_bin[(bin_index, pair)]
                    },
                    "pair_counts": {
                        _pair_label(pair): self.accepted_pair_by_bin[(bin_index, pair)]
                        for pair in self.available_pairs_by_bin[bin_index]
                        if self.accepted_pair_by_bin[(bin_index, pair)]
                    },
                }
                for bin_index in range(self.time_bin_count)
            ],
            "pair_target_counts": {
                _pair_label(pair): quota for pair, quota in self.pair_quotas.items()
            },
            "pair_counts": {
                _pair_label(pair): self.accepted_pair_global[pair]
                for pair in self.available_pairs
            },
            "participant_counts": dict(sorted(self.accepted_participant_global.items())),
            "day_counts": dict(sorted(self.accepted_day_global.items())),
            "strict_balance_errors": self.strict_balance_errors(),
        }


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
    clustering_method: str = COSINE_KMEANS_CLUSTERING,
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
    if clustering_method not in CLUSTERING_METHODS:
        raise ValueError(f"unknown clustering_method: {clustering_method}")
    if clustering_method == TEMPORAL_AGGLOMERATIVE_CLUSTERING and split_noncontiguous_clusters:
        raise ValueError("temporal agglomerative clusters are already contiguous")
    if split_noncontiguous_clusters and (
        max_member_gap_seconds is None or max_member_gap_seconds <= 0
    ):
        raise ValueError("a positive max_member_gap_seconds is required when splitting clusters")

    if clustering_method == COSINE_KMEANS_CLUSTERING:
        labels, medoids = cluster_embedding_medoids(embeddings, cluster_count)
    else:
        labels, medoids = temporal_agglomerative_embedding_medoids(
            embeddings,
            cluster_count,
            timestamps=[
                float(frame.get("timestamp_seconds") or 0.0) for frame in frames
            ],
        )
    representatives = []
    representative_embeddings = []
    output_labels = [-1 for _ in frames]
    for visual_cluster_index, frame_index in enumerate(medoids):
        visual_member_indices = [
            index
            for index, label in enumerate(labels)
            if int(label) == int(visual_cluster_index)
        ]
        if clustering_method == TEMPORAL_AGGLOMERATIVE_CLUSTERING:
            visual_member_indices.sort(
                key=lambda index: (
                    float(frames[index].get("timestamp_seconds") or 0.0),
                    index,
                )
            )
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
            member_timestamps = [
                frames[index].get("timestamp_seconds") for index in member_indices
            ]
            numeric_member_timestamps = [
                float(value) for value in member_timestamps if value is not None
            ]
            representatives.append(
                {
                    "cluster_index": int(cluster_index),
                    "visual_cluster_index": int(visual_cluster_index),
                    "temporal_component_index": int(component_index),
                    "frame_index": int(frame_index),
                    "timestamp_seconds": frame.get("timestamp_seconds"),
                    "path": frame.get("path"),
                    "member_indices": member_indices,
                    "member_timestamps": member_timestamps,
                    "temporal_start_seconds": (
                        min(numeric_member_timestamps)
                        if numeric_member_timestamps
                        else None
                    ),
                    "temporal_end_seconds": (
                        max(numeric_member_timestamps)
                        if numeric_member_timestamps
                        else None
                    ),
                    "member_count": len(member_indices),
                }
            )
            representative_embeddings.append(embeddings[frame_index])

    return {
        "cluster_count_requested": cluster_count,
        "cluster_count": len(representatives),
        "visual_cluster_count": len(medoids),
        "clustering_method": clustering_method,
        "split_noncontiguous_clusters": split_noncontiguous_clusters,
        "max_member_gap_seconds": max_member_gap_seconds,
        "labels": output_labels,
        "representatives": representatives,
        "representative_embeddings": representative_embeddings,
    }


def build_cluster_summary_prompt(
    *,
    cluster_specs: list[dict[str, Any]],
) -> str:
    """Build one batched grounded event-summary request for every cluster in a pair."""

    compact_specs = [
        {
            "cluster_id": spec["cluster_id"],
            "user": spec["user"],
            "image_numbers": spec["image_numbers"],
            "timestamps_seconds": spec["timestamps_seconds"],
        }
        for spec in cluster_specs
    ]
    return f"""You are summarizing ALL temporal visual clusters from a synchronized pair of egocentric videos in ONE batched inference.

The supplied images are ordered globally. The catalog below maps each cluster_id to the 1-based image numbers belonging to that cluster and to their original-video timestamps. Summarize each cluster independently; never merge evidence across cluster_ids while writing an event summary.

Cluster/image catalog:
{json.dumps(compact_specs, ensure_ascii=False, indent=2)}

Return exactly one JSON object with exactly this top-level shape:
{{
  "summaries": [
    {{
      "cluster_id": "copy one cluster_id exactly",
      "event_summary": "one concise grounded event description"
    }}
  ]
}}

Output requirements:
- Return exactly one summary for every cluster_id in the catalog, with no missing or extra cluster_ids.
- Keep event_summary to one concise sentence whenever possible, at most two short sentences.
- Focus on the primary visible event: what the camera wearer does, interacts with, changes, carries, gives, receives, places, opens, uses, or directly observes.
- Preserve distinctive action-relevant attributes of objects or people when they could help match an entity across another user's view (for example color, shape, clothing, or location), but omit generic scene description.
- Include a directly visible resulting state or location when it is important to the event (for example an object is placed on a desk or a lamp is visibly off afterward).
- Mention a nearby person's action only when it is directly visible and important to the event. Keep wearer actions and other-person actions clearly attributed.
- If no manipulation is visible, state the most specific directly visible task, attention target, navigation, posture, or observation instead of inventing an action.
- Treat images as sparse samples. You may describe progression only when the supplied images for that SAME cluster visibly support it. Do not invent motion, handoffs, transitions, intent, identity, causality, or intervening events.
- Do not infer that similar-looking objects or people across different cluster_ids are identical. Cross-cluster linkage is handled later by a separate relation-selection step.
- Do not mention CLIP, cluster mechanics, filenames, frame indices, image numbers, timestamps, or these instructions inside event_summary.
- Do not add a visual_summary or general scene caption.
"""


def build_relation_selection_prompt(summary_rows: list[dict[str, Any]]) -> str:
    """Build one text-only request that retrieves a few promising cross-user relations."""

    compact_rows = [
        {
            "cluster_id": row["cluster_id"],
            "user": row["user"],
            "member_timestamps": row.get("member_timestamps", []),
            "event_summary": row["event_summary"],
        }
        for row in summary_rows
    ]
    return f"""You are a conservative cross-user relation retriever for egocentric video events.

Below are independently grounded event summaries from temporal clusters belonging to two synchronized users. Your job is ONLY to identify a very small number of promising cross-user relations that may be worth visually verifying before question generation. Do not write a question and do not claim that a hypothesized relation is proven.

Event summaries:
{json.dumps(compact_rows, ensure_ascii=False, indent=2)}

Return exactly one JSON object with exactly this shape:
{{
  "candidate_relations": [
    {{
      "cluster_ids": ["cluster_id", "cluster_id"],
      "relation": "short hypothesis such as possible object handoff and continuation",
      "why_promising": "brief reason grounded only in the listed summaries",
      "confidence": "low/medium/high"
    }}
  ]
}}

Selection rules:
- Return at most 3 candidate_relations. Return an empty list when no relation is genuinely promising.
- Every candidate must contain at least one cluster from EACH user. A candidate may also include same-user continuation clusters before or after the cross-user link.
- Prefer distinctive, answer-bearing relations: complementary give/receive or request/response actions; the same distinctive object apparently continuing across users; visible state/location continuation; one user's action followed by another user's concrete outcome; or a strong identity/role linkage supported by distinctive attributes.
- A shared room, shared task, generic object, similar pose, repeated background person, or mere timestamp proximity is NOT enough.
- Do not select a relation merely because both users mention the same common object (cup, phone, flowers, table, etc.) unless actions/attributes make the linkage meaningfully stronger.
- Do not turn concurrency itself into the relation and do not select unrelated simultaneous activities.
- Treat matching descriptions and chronology as hypotheses, never proof of identity, transfer, possession, causality, or continuity.
- Prefer fewer stronger candidates over filling the quota.
- Copy cluster_ids exactly from the input. Do not invent events or cluster_ids.
"""


def _evenly_spaced_indices(count: int, limit: int) -> list[int]:
    if count <= 0:
        return []
    if limit <= 0:
        raise ValueError("cluster_summary_max_images must be positive")
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    return sorted(
        {
            round(position * (count - 1) / (limit - 1))
            for position in range(limit)
        }
    )


def summarize_pair_clusters(
    pair: dict[str, Any],
    clip_rows: list[dict[str, Any]],
    *,
    runner: ClusterSummaryRunner,
    max_images_per_cluster: int = 12,
    max_attempts: int = 2,
) -> list[dict[str, Any]]:
    """Batch cluster descriptions once, then map relations in one text-only call.

    The normal path uses exactly two model inferences per sampled pair:
      1) one multimodal call covering every temporal cluster;
      2) one text-only call selecting up to three cross-user relation hypotheses.
    A whole call is retried only when its structured response is invalid.
    """

    if max_images_per_cluster <= 0:
        raise ValueError("max_images_per_cluster must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    pruning = pair.get("temporal_pruning")
    if not isinstance(pruning, dict):
        raise ValueError("pair is missing temporal_pruning diagnostics")

    cluster_specs: list[dict[str, Any]] = []
    all_image_paths: list[str] = []
    decision_by_cluster_id: dict[str, dict[str, Any]] = {}

    # Prepare every cluster first so all of their images can be sent together.
    for side in ("left", "right"):
        clip_index = int(pair[f"{side}_index"])
        clip_row = clip_rows[clip_index]
        clip = clip_row.get("clip") if isinstance(clip_row.get("clip"), dict) else {}
        user = str(clip.get("agent_name") or clip_row.get("user") or side)
        agent_dir = clip.get("agent_dir")
        frames = list(clip_row.get("frames") or [])
        decisions = pruning.get(f"{side}_cluster_decisions")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError(f"{side} cluster decisions are unavailable for VLM summaries")
        restored_indices = {
            int(value)
            for value in pruning.get(f"{side}_restored_frame_indices", []) or []
        }

        for decision in decisions:
            if not isinstance(decision, dict):
                continue
            cluster_index = int(decision.get("cluster_index", -1))
            cluster_id = f"{side}:{cluster_index}"
            if cluster_id in decision_by_cluster_id:
                raise ValueError(f"duplicate cluster_id while batching summaries: {cluster_id}")

            member_indices = sorted(
                {int(value) for value in decision.get("member_indices", [])},
                key=lambda index: (
                    float(frames[index].get("timestamp_seconds", 0.0))
                    if 0 <= index < len(frames)
                    else float("inf"),
                    index,
                ),
            )
            if not member_indices:
                raise ValueError(f"{side} cluster {cluster_index} has no member frames")
            invalid_indices = [
                index for index in member_indices if index < 0 or index >= len(frames)
            ]
            if invalid_indices:
                raise IndexError(
                    f"{side} cluster {cluster_index} references invalid frames {invalid_indices[:5]}"
                )
            selected_offsets = _evenly_spaced_indices(
                len(member_indices),
                max_images_per_cluster,
            )
            selected_indices = [member_indices[offset] for offset in selected_offsets]
            image_paths = [str(frames[index].get("path") or "") for index in selected_indices]
            missing_paths = [path for path in image_paths if not path or not Path(path).is_file()]
            if missing_paths:
                raise FileNotFoundError(
                    f"{side} cluster {cluster_index} has unavailable member images: {missing_paths[:3]}"
                )
            timestamps = [
                float(frames[index].get("timestamp_seconds", 0.0))
                for index in selected_indices
            ]
            first_image_number = len(all_image_paths) + 1
            all_image_paths.extend(image_paths)
            last_image_number = len(all_image_paths)
            retained_for_generator = (
                str(decision.get("status") or "") == "kept"
                or any(index in restored_indices for index in member_indices)
            )

            spec = {
                "cluster_id": cluster_id,
                "side": side,
                "agent_dir": agent_dir,
                "user": user,
                "cluster_index": cluster_index,
                "visual_cluster_index": decision.get("visual_cluster_index"),
                "temporal_component_index": decision.get("temporal_component_index"),
                "member_timestamps": list(decision.get("member_timestamps") or []),
                "status": decision.get("status"),
                "retained_for_generator": retained_for_generator,
                "source_member_count": len(member_indices),
                "selected_image_count": len(image_paths),
                "timestamps_seconds": timestamps,
                "image_numbers": list(range(first_image_number, last_image_number + 1)),
            }
            cluster_specs.append(spec)
            decision_by_cluster_id[cluster_id] = decision

    if not cluster_specs:
        raise ValueError("no temporal clusters available for batched VLM summary")

    # Inference 1/2: one multimodal request describes every cluster independently.
    summary_prompt = build_cluster_summary_prompt(cluster_specs=cluster_specs)
    raw_summary_response = ""
    parsed_summaries: dict[str, str] | None = None
    summary_errors: list[str] = []
    expected_cluster_ids = {spec["cluster_id"] for spec in cluster_specs}

    for attempt_index in range(1, max_attempts + 1):
        attempt_prompt = summary_prompt
        if raw_summary_response:
            attempt_prompt += (
                "\nYour previous batched response was invalid. Return only the required JSON object "
                "with exactly one event_summary for every catalog cluster_id. "
                f"Previous response:\n{raw_summary_response}"
            )
        raw_summary_response = runner.generate(
            attempt_prompt,
            image_paths=all_image_paths,
            decoding_mode="greedy",
        )
        try:
            candidate = extract_json_object(raw_summary_response)
            items = candidate.get("summaries")
            if not isinstance(items, list):
                raise ValueError("summaries must be a list")
            by_id: dict[str, str] = {}
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("each summaries item must be an object")
                cluster_id = item.get("cluster_id")
                event_summary = item.get("event_summary")
                if not isinstance(cluster_id, str) or cluster_id not in expected_cluster_ids:
                    raise ValueError(f"unknown cluster_id in summaries: {cluster_id!r}")
                if cluster_id in by_id:
                    raise ValueError(f"duplicate cluster_id in summaries: {cluster_id}")
                if not isinstance(event_summary, str) or not event_summary.strip():
                    raise ValueError(f"event_summary must be non-empty for {cluster_id}")
                by_id[cluster_id] = event_summary.strip()
            missing = sorted(expected_cluster_ids - set(by_id))
            extra = sorted(set(by_id) - expected_cluster_ids)
            if missing or extra:
                raise ValueError(f"summary cluster mismatch; missing={missing}, extra={extra}")
            parsed_summaries = by_id
            break
        except (TypeError, ValueError) as exc:
            summary_errors.append(f"attempt {attempt_index}: {exc}")

    if parsed_summaries is None:
        raise ValueError("batched cluster VLM summary failed: " + "; ".join(summary_errors))

    summary_rows: list[dict[str, Any]] = []
    for spec in cluster_specs:
        cluster_id = spec["cluster_id"]
        event_summary = parsed_summaries[cluster_id]
        decision = decision_by_cluster_id[cluster_id]
        decision["vlm_summary"] = {
            "model_id": runner.model_id,
            "event_summary": event_summary,
            "image_count": spec["selected_image_count"],
            "source_member_count": spec["source_member_count"],
            "source_timestamps_seconds": spec["timestamps_seconds"],
            "batch_attempt_count": len(summary_errors) + 1,
        }
        summary_rows.append(
            {
                "cluster_id": cluster_id,
                "side": spec["side"],
                "agent_dir": spec["agent_dir"],
                "user": spec["user"],
                "cluster_index": spec["cluster_index"],
                "visual_cluster_index": spec["visual_cluster_index"],
                "temporal_component_index": spec["temporal_component_index"],
                "member_timestamps": spec["member_timestamps"],
                "status": spec["status"],
                "retained_for_generator": spec["retained_for_generator"],
                "model_id": runner.model_id,
                "event_summary": event_summary,
            }
        )

    # Inference 2/2: retrieve a few relations using text only.
    relation_prompt = build_relation_selection_prompt(summary_rows)
    raw_relation_response = ""
    parsed_relations: list[dict[str, Any]] | None = None
    relation_errors: list[str] = []
    summary_by_id = {row["cluster_id"]: row for row in summary_rows}
    side_by_id = {row["cluster_id"]: row["side"] for row in summary_rows}

    for attempt_index in range(1, max_attempts + 1):
        attempt_prompt = relation_prompt
        if raw_relation_response:
            attempt_prompt += (
                "\nYour previous relation response was invalid. Return only the required JSON object. "
                f"Previous response:\n{raw_relation_response}"
            )
        raw_relation_response = runner.generate(
            attempt_prompt,
            image_paths=None,
            decoding_mode="greedy",
        )
        try:
            candidate = extract_json_object(raw_relation_response)
            relations = candidate.get("candidate_relations")
            if not isinstance(relations, list):
                raise ValueError("candidate_relations must be a list")
            if len(relations) > 3:
                raise ValueError("candidate_relations must contain at most 3 items")

            validated: list[dict[str, Any]] = []
            for relation_index, relation in enumerate(relations, 1):
                if not isinstance(relation, dict):
                    raise ValueError("each candidate relation must be an object")
                cluster_ids = relation.get("cluster_ids")
                relation_text = relation.get("relation")
                why_promising = relation.get("why_promising")
                confidence = relation.get("confidence")
                if not isinstance(cluster_ids, list) or len(cluster_ids) < 2:
                    raise ValueError("each candidate relation needs at least two cluster_ids")
                normalized_ids: list[str] = []
                for cluster_id in cluster_ids:
                    if not isinstance(cluster_id, str) or cluster_id not in summary_by_id:
                        raise ValueError(f"unknown relation cluster_id: {cluster_id!r}")
                    if cluster_id not in normalized_ids:
                        normalized_ids.append(cluster_id)
                if len({side_by_id[cluster_id] for cluster_id in normalized_ids}) < 2:
                    raise ValueError("each candidate relation must span both users")
                if not isinstance(relation_text, str) or not relation_text.strip():
                    raise ValueError("relation must be a non-empty string")
                if not isinstance(why_promising, str) or not why_promising.strip():
                    raise ValueError("why_promising must be a non-empty string")
                if confidence not in {"low", "medium", "high"}:
                    raise ValueError("confidence must be low, medium, or high")

                # These locators help the generator find the suggested moments. The
                # underlying event summaries stay in audit metadata and are not copied
                # into candidate_relations or the generator prompt.
                selected_events = [
                    {
                        "cluster_id": cluster_id,
                        "user": summary_by_id[cluster_id]["user"],
                        "member_timestamps": summary_by_id[cluster_id].get(
                            "member_timestamps", []
                        ),
                        "retained_for_generator": summary_by_id[cluster_id].get(
                            "retained_for_generator"
                        ),
                    }
                    for cluster_id in normalized_ids
                ]
                validated.append(
                    {
                        "relation_id": f"R{relation_index}",
                        "cluster_ids": normalized_ids,
                        "relation": relation_text.strip(),
                        "why_promising": why_promising.strip(),
                        "confidence": confidence,
                        "selected_events": selected_events,
                    }
                )
            parsed_relations = validated
            break
        except (TypeError, ValueError) as exc:
            relation_errors.append(f"attempt {attempt_index}: {exc}")

    if parsed_relations is None:
        raise ValueError("cross-user relation selection failed: " + "; ".join(relation_errors))

    # Raw summaries remain available for auditing, but prompt rendering exposes
    # only the compact relation-map hypotheses.
    pair["cluster_vlm_summaries"] = summary_rows
    pair["candidate_relations"] = parsed_relations
    pair["cluster_summary_batch_diagnostics"] = {
        "model_id": runner.model_id,
        "summary_inference_count": 1,
        "relation_inference_count": 1,
        "normal_total_inference_count": 2,
        "cluster_count": len(summary_rows),
        "batched_image_count": len(all_image_paths),
        "summary_attempt_count": len(summary_errors) + 1,
        "relation_attempt_count": len(relation_errors) + 1,
        "raw_summary_response": raw_summary_response,
        "raw_relation_response": raw_relation_response,
    }
    pruning["cluster_summary_model_id"] = runner.model_id
    pruning["cluster_summary_count"] = len(summary_rows)
    pruning["cluster_vlm_summaries"] = summary_rows
    pruning["candidate_relation_count"] = len(parsed_relations)
    pruning["candidate_relations"] = parsed_relations
    return summary_rows


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
    clustering_method: str = COSINE_KMEANS_CLUSTERING,
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
        clustering_method=clustering_method,
        split_noncontiguous_clusters=split_noncontiguous_clusters,
        max_member_gap_seconds=max_cluster_member_gap_seconds,
    )
    right_clusters = clustered_frame_representatives(
        right_frames,
        right_embeddings,
        cluster_count=cluster_count,
        clustering_method=clustering_method,
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

    eligibility_matrix = [
        [
            representative_pair_is_eligible(left_cluster_index, right_cluster_index)
            for right_cluster_index in range(len(row))
        ]
        for left_cluster_index, row in enumerate(matrix)
    ]

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
    preserved_cross_time_pairs = []
    left_marked_clusters: set[int] = set()
    right_marked_clusters: set[int] = set()
    for left_cluster_index, row in enumerate(matrix):
        for right_cluster_index, similarity in enumerate(row):
            if not representative_pair_is_eligible(left_cluster_index, right_cluster_index):
                if float(similarity) >= high_similarity_threshold:
                    left_rep = left_clusters["representatives"][left_cluster_index]
                    right_rep = right_clusters["representatives"][right_cluster_index]
                    preserved_cross_time_pairs.append(
                        {
                            "left_cluster_index": int(left_cluster_index),
                            "right_cluster_index": int(right_cluster_index),
                            "similarity": round(float(similarity), 6),
                            "left_representative_frame_index": left_rep["frame_index"],
                            "right_representative_frame_index": right_rep["frame_index"],
                            "left_representative_timestamp_seconds": left_rep.get(
                                "timestamp_seconds"
                            ),
                            "right_representative_timestamp_seconds": right_rep.get(
                                "timestamp_seconds"
                            ),
                            "timestamp_difference_seconds": round(
                                representative_time_difference(
                                    left_cluster_index,
                                    right_cluster_index,
                                ),
                                6,
                            ),
                            "status": "preserved_cross_time",
                        }
                    )
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
    preserved_cross_time_pairs.sort(
        key=lambda row: float(row["similarity"]),
        reverse=True,
    )
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
        "clustering_method": clustering_method,
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
        "representative_pair_eligibility_matrix": eligibility_matrix,
        "temporal_pairing_policy": (
            "timestamp_agnostic"
            if max_pair_time_difference_seconds is None
            else "hard_centroid_time_gate"
        ),
        "high_similarity_representative_pairs": high_pairs,
        "high_similarity_representative_pair_count": len(high_pairs),
        "preserved_cross_time_representative_pairs": preserved_cross_time_pairs,
        "preserved_cross_time_representative_pair_count": len(
            preserved_cross_time_pairs
        ),
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
    max_pair_time_difference_seconds: float | None = DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    split_noncontiguous_clusters: bool = DEFAULT_SPLIT_NONCONTIGUOUS_CLUSTERS,
    max_cluster_member_gap_seconds: float | None = None,
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
                split_noncontiguous_clusters=split_noncontiguous_clusters,
                max_cluster_member_gap_seconds=max_cluster_member_gap_seconds,
            )
            matrix = temporal_pruning["representative_similarity_matrix"]
            eligibility_matrix = temporal_pruning[
                "representative_pair_eligibility_matrix"
            ]
            values = [
                float(similarity)
                for row_index, row in enumerate(matrix)
                for column_index, similarity in enumerate(row)
                if eligibility_matrix[row_index][column_index]
            ]
            mean_sim = sum(values) / len(values) if values else 0.0
            topk_sim = _topk_mean(values, topk) if values else 0.0
            rejection_reasons = []
            if not values:
                rejection_reasons.append("no_temporally_eligible_representative_pairs")
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
                    "topk": min(topk, len(values)),
                    "temporally_eligible_representative_pair_count": len(values),
                    "all_representative_pair_count": sum(len(row) for row in matrix),
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
            "split_noncontiguous_clusters": split_noncontiguous_clusters,
            "max_cluster_member_gap_seconds": max_cluster_member_gap_seconds,
            "pair_count": len(pair_scores),
            "kept_pair_count": len(kept_pairs),
            "interpretation": (
                "Each selected video is sampled once per second, clustered, and represented by medoid "
                "frames. Visual clusters are split into temporal runs, and pair scores are computed "
                "only from representative CLIP similarities inside the configured timestamp gate. "
                "High eligible representative matches mark their source temporal clusters for "
                "uniform interval pruning; visually similar representatives outside the gate are "
                "preserved. "
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


def _select_group_clips_for_agent_pair(
    group: dict[str, Any],
    *,
    agent_pair: tuple[str, str],
) -> list[dict[str, Any]]:
    """Select an exact scheduler-chosen pair from a synchronized group."""

    wanted = set(agent_pair)
    selected = sorted(
        [
            clip
            for clip in group.get("clips", [])
            if str(clip.get("agent_dir") or "") in wanted
        ],
        key=lambda item: str(item.get("agent_dir")),
    )
    if len(selected) != 2 or {
        str(clip.get("agent_dir") or "") for clip in selected
    } != wanted:
        raise ValueError(
            f"scheduled agent pair {agent_pair!r} is unavailable in "
            f"{group.get('day')} {group.get('time_token')}"
        )
    return selected


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
        "clustering_method": pruning.get("clustering_method"),
        "high_similarity_threshold": pruning.get("high_similarity_threshold"),
        "max_pair_time_difference_seconds": pruning.get(
            "max_pair_time_difference_seconds"
        ),
        "temporal_pairing_policy": pruning.get("temporal_pairing_policy"),
        "split_noncontiguous_clusters": pruning.get(
            "split_noncontiguous_clusters"
        ),
        "max_cluster_member_gap_seconds": pruning.get(
            "max_cluster_member_gap_seconds"
        ),
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
        "preserved_cross_time_representative_pairs": pruning.get(
            "preserved_cross_time_representative_pairs",
            [],
        ),
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
    max_pair_time_difference_seconds: float | None = DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    split_noncontiguous_clusters: bool = DEFAULT_SPLIT_NONCONTIGUOUS_CLUSTERS,
    max_cluster_member_gap_seconds: float | None = None,
    random_pair_first: bool = True,
    preselected_agent_pair: tuple[str, str] | None = None,
    rng: random.Random | None = None,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
    cluster_summary_runner: ClusterSummaryRunner | None = None,
    cluster_summary_max_images: int = 12,
    cluster_summary_max_attempts: int = 2,
) -> dict[str, Any]:
    """Analyze one synchronized group after optionally sampling a two-video pair first."""

    if selected_count != 2:
        raise ValueError("pair-ranking mode currently selects exactly two clips")
    if pairs_per_group < 1:
        raise ValueError("pairs_per_group must be positive")
    rng = rng or random.Random()
    original_group_size = len(group.get("clips", []))
    if preselected_agent_pair is not None:
        sampled_source_clips = _select_group_clips_for_agent_pair(
            group,
            agent_pair=preselected_agent_pair,
        )
    elif random_pair_first:
        sampled_source_clips = _sample_group_clips_for_pair(
            group,
            selected_count=selected_count,
            rng=rng,
        )
    else:
        sampled_source_clips = sorted(
            group.get("clips", []), key=lambda item: str(item.get("agent_dir"))
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
        max_pair_time_difference_seconds=max_pair_time_difference_seconds,
        split_noncontiguous_clusters=split_noncontiguous_clusters,
        max_cluster_member_gap_seconds=max_cluster_member_gap_seconds,
    )
    surviving_pairs = pair_analysis["surviving_pairs"]
    if not surviving_pairs:
        diagnostics = compact_pair_rejection_summary(pair_analysis)
        raise ValueError(f"no video pairs survived the frame-matrix pair filters: {diagnostics}")
    sampled_pairs = rng.sample(surviving_pairs, min(pairs_per_group, len(surviving_pairs)))
    for sample_rank, pair in enumerate(sampled_pairs, 1):
        pair["sample_rank"] = sample_rank
        if cluster_summary_runner is not None:
            summarize_pair_clusters(
                pair,
                rows,
                runner=cluster_summary_runner,
                max_images_per_cluster=cluster_summary_max_images,
                max_attempts=cluster_summary_max_attempts,
            )
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
            "method": (
                "balanced_time_pair_preselection_then_cluster_prune"
                if preselected_agent_pair is not None
                else "random_synchronized_pair_then_cluster_prune"
            ),
            "selected_count": selected_count,
            "pairs_per_group": pairs_per_group,
            "random_pair_first": random_pair_first,
            "preselected_agent_pair": (
                list(preselected_agent_pair) if preselected_agent_pair is not None else None
            ),
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
            "split_noncontiguous_clusters": split_noncontiguous_clusters,
            "max_cluster_member_gap_seconds": max_cluster_member_gap_seconds,
            "cluster_summary_model_id": (
                cluster_summary_runner.model_id
                if cluster_summary_runner is not None
                else None
            ),
            "selected_indices": selected_indices,
            "selected_agents": [clip.get("agent_dir") for clip in selected_clips],
            "selected_users": [clip.get("agent_name") for clip in selected_clips],
            "selected_pair": selected_pair,
            "selected_pair_mean_sim": selected_pair["mean_sim"],
            "selected_pair_topk_sim": selected_pair["topk_sim"],
            "rationale": (
                "The sampler first selects two videos from the synchronized group using the "
                "recorded pair-selection policy, then "
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

    selected_pair = group_result.get("selection", {}).get("selected_pair", {})
    cluster_summary_rows = [
        dict(row)
        for row in selected_pair.get("cluster_vlm_summaries", [])
        if isinstance(row, dict)
    ]
    if cluster_summary_rows:
        for row in cluster_summary_rows:
            row["member_timestamps"] = ",".join(
                str(value) for value in row.get("member_timestamps", [])
            )
        _write_csv(
            traces_dir / "cluster_vlm_summaries.csv",
            cluster_summary_rows,
            [
                "side",
                "agent_dir",
                "user",
                "cluster_index",
                "visual_cluster_index",
                "temporal_component_index",
                "member_timestamps",
                "status",
                "retained_for_generator",
                "event_summary",
                "model_id",
            ],
        )

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
        + (
            "- Cluster summaries: `comparison_traces/cluster_vlm_summaries.csv` contains one compact event summary per temporal cluster; relation candidates are stored in packet metadata.\n"
            if cluster_summary_rows
            else ""
        )
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
    packet = {
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
    selected_pair = group_result.get("selection", {}).get("selected_pair", {})
    cluster_summaries = selected_pair.get("cluster_vlm_summaries")
    if isinstance(cluster_summaries, list) and cluster_summaries:
        packet["cluster_vlm_summaries"] = [
            dict(row) for row in cluster_summaries if isinstance(row, dict)
        ]
    candidate_relations = selected_pair.get("candidate_relations")
    if isinstance(candidate_relations, list):
        packet["candidate_relations"] = [
            dict(row) for row in candidate_relations if isinstance(row, dict)
        ]
    batch_diagnostics = selected_pair.get("cluster_summary_batch_diagnostics")
    if isinstance(batch_diagnostics, dict):
        packet["cluster_summary_batch_diagnostics"] = dict(batch_diagnostics)
    if isinstance(group_result.get("balanced_sampling"), dict):
        packet["balanced_sampling"] = dict(group_result["balanced_sampling"])
    return packet

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
    max_pair_time_difference_seconds: float | None = DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    split_noncontiguous_clusters: bool = DEFAULT_SPLIT_NONCONTIGUOUS_CLUSTERS,
    max_cluster_member_gap_seconds: float | None = None,
    random_pair_first: bool = True,
    random_seed: int | None = 42,
    sampling_policy: str = RANDOM_SAMPLING_POLICY,
    time_bin_count: int = 10,
    strict_balance: bool = False,
    group_partition_count: int = 1,
    group_partition_index: int = 0,
    ffmpeg_binary: str = "ffmpeg",
    download_media: bool = False,
    review_dir: str | Path | None = None,
    encoder: ImageEncoder | None = None,
    summarize_clusters: bool = False,
    cluster_summary_runner: ClusterSummaryRunner | None = None,
    cluster_summary_backend: str = "transformers-local",
    cluster_summary_model_id: str = DEFAULT_CLUSTER_SUMMARY_MODEL_ID,
    cluster_summary_base_url: str = "http://127.0.0.1:8000/v1",
    cluster_summary_api_key: str | None = None,
    cluster_summary_dtype: str = "bfloat16",
    cluster_summary_max_new_tokens: int = 4096,
    cluster_summary_max_images: int = 12,
    cluster_summary_max_attempts: int = 2,
    cluster_summary_allow_cpu: bool = False,
    cluster_summary_disable_thinking: bool = True,
) -> list[dict[str, Any]]:
    """Write CLIP-pruned candidates using random or balanced time/pair sampling."""

    manifest = read_json(manifest_path)
    rng = random.Random(random_seed) if random_seed is not None else random.Random()
    if sampling_policy not in SAMPLING_POLICIES:
        raise ValueError(
            f"sampling_policy must be one of {', '.join(SAMPLING_POLICIES)}"
        )
    if group_partition_count <= 0:
        raise ValueError("group_partition_count must be positive")
    if not 0 <= group_partition_index < group_partition_count:
        raise ValueError("group_partition_index must be in [0, group_partition_count)")
    if sampling_policy == BALANCED_TIME_PAIR_SAMPLING_POLICY and pairs_per_group != 1:
        raise ValueError("balanced_time_pair sampling requires pairs_per_group=1")

    all_groups = [
        group
        for group in group_manifest_clips(manifest)
        if len(group.get("clips", [])) >= min_group_size
    ]
    if not all_groups:
        raise ValueError("manifest contains no eligible synchronized groups")
    timeline_start_seconds = min(_group_clock_seconds(group) for group in all_groups)
    timeline_end_seconds = max(_group_clock_seconds(group) for group in all_groups) + 0.01
    groups = [
        group
        for group in all_groups
        if stable_group_partition(group, partition_count=group_partition_count)
        == group_partition_index
    ]
    if not groups:
        raise ValueError(
            f"group partition {group_partition_index}/{group_partition_count} is empty"
        )

    scheduler: BalancedTimePairScheduler | None = None
    if sampling_policy == BALANCED_TIME_PAIR_SAMPLING_POLICY:
        scheduler = BalancedTimePairScheduler(
            groups,
            target_count=target_count,
            time_bin_count=time_bin_count,
            random_seed=random_seed,
            max_attempts=max_groups,
            timeline_start_seconds=timeline_start_seconds,
            timeline_end_seconds=timeline_end_seconds,
            pair_quota_rotation=group_partition_index,
            coordination_groups=all_groups,
            coordination_partition_count=group_partition_count,
        )
    else:
        rng.shuffle(groups)
        if max_groups is not None:
            groups = groups[:max_groups]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    review_root = Path(review_dir) if review_dir is not None else output_dir / "review_bundles"
    encoder = encoder or TransformersClipEncoder(model_id)
    if cluster_summary_runner is not None:
        summarize_clusters = True
    if summarize_clusters and cluster_summary_runner is None:
        try:
            from .qwen3vl_runner import make_runner
        except ImportError:
            from egolife_two_user_qa.qwen3vl_runner import make_runner

        cluster_summary_runner = make_runner(
            cluster_summary_backend,
            model_id=cluster_summary_model_id,
            base_url=cluster_summary_base_url,
            max_new_tokens=cluster_summary_max_new_tokens,
            dtype=cluster_summary_dtype,
            allow_cpu=cluster_summary_allow_cpu,
            disable_thinking=cluster_summary_disable_thinking,
            api_key=cluster_summary_api_key,
        )

    candidates = []
    skipped = []
    random_group_index = 0
    attempt_index = 0
    while len(candidates) < target_count:
        decision: BalancedSamplingDecision | None = None
        if scheduler is not None:
            decision = scheduler.next_decision()
            if decision is None:
                break
            group = decision.group
            preselected_agent_pair = decision.pair
        else:
            if random_group_index >= len(groups):
                break
            group = groups[random_group_index]
            random_group_index += 1
            preselected_agent_pair = None
        attempt_index += 1
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
                max_pair_time_difference_seconds=max_pair_time_difference_seconds,
                split_noncontiguous_clusters=split_noncontiguous_clusters,
                max_cluster_member_gap_seconds=max_cluster_member_gap_seconds,
                random_pair_first=random_pair_first,
                preselected_agent_pair=preselected_agent_pair,
                rng=rng,
                ffmpeg_binary=ffmpeg_binary,
                download_media=download_media,
                cluster_summary_runner=cluster_summary_runner,
                cluster_summary_max_images=cluster_summary_max_images,
                cluster_summary_max_attempts=cluster_summary_max_attempts,
            )
        except Exception as exc:
            skipped.append(
                {
                    "index": attempt_index - 1,
                    "day": group.get("day"),
                    "time_token": group.get("time_token"),
                    "time_bin_index": (
                        decision.time_bin_index if decision is not None else None
                    ),
                    "preselected_agent_pair": (
                        list(decision.pair) if decision is not None else None
                    ),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if scheduler is not None and decision is not None:
            scheduler.record_success(decision)
            result["balanced_sampling"] = scheduler.decision_metadata(decision)
            result["balanced_sampling"]["group_partition_count"] = group_partition_count
            result["balanced_sampling"]["group_partition_index"] = group_partition_index
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

    balance_summary = scheduler.summary() if scheduler is not None else None
    balance_errors = (
        list(balance_summary.get("strict_balance_errors") or [])
        if balance_summary is not None
        else []
    )
    write_jsonl(output_path, candidates)
    write_json(
        output_dir / "group_relative_clip_summary.json",
        {
            "manifest_path": str(manifest_path),
            "output_path": str(output_path),
            "review_dir": str(review_root),
            "eligible_group_count_before_partition": len(all_groups),
            "group_count_considered": len(groups),
            "group_attempt_count": attempt_index,
            "candidate_count": len(candidates),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "balanced_sampling": balance_summary,
            "settings": {
                "model_id": encoder.model_id,
                "target_count": target_count,
                "max_groups": max_groups,
                "min_group_size": min_group_size,
                "sampling_policy": sampling_policy,
                "time_bin_count": time_bin_count,
                "strict_balance": strict_balance,
                "group_partition_count": group_partition_count,
                "group_partition_index": group_partition_index,
                "group_order": (
                    "balanced_time_bin_round_robin_with_pair_quotas"
                    if scheduler is not None
                    else "randomized_before_max_groups"
                ),
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
                "max_pair_time_difference_seconds": max_pair_time_difference_seconds,
                "split_noncontiguous_clusters": split_noncontiguous_clusters,
                "max_cluster_member_gap_seconds": max_cluster_member_gap_seconds,
                "summarize_clusters": summarize_clusters,
                "cluster_summary_backend": (
                    cluster_summary_backend if summarize_clusters else None
                ),
                "cluster_summary_model_id": (
                    cluster_summary_runner.model_id
                    if cluster_summary_runner is not None
                    else None
                ),
                "cluster_summary_max_images": cluster_summary_max_images,
                "cluster_summary_max_attempts": cluster_summary_max_attempts,
                "random_pair_first": random_pair_first,
                "random_seed": random_seed,
                "download_media": download_media,
                "review_dir": str(review_root),
            },
        },
    )
    if strict_balance and balance_errors:
        raise ValueError("strict balanced sampling failed: " + "; ".join(balance_errors))
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
        "--max-pair-time-difference-seconds",
        type=float,
        default=DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
        help=(
            "Only prune high-similarity centroid pairs whose timestamps differ by at most "
            "this many seconds and use only those pairs for pair scoring"
        ),
    )
    parser.add_argument(
        "--timestamp-agnostic-pruning",
        action="store_const",
        const=None,
        dest="max_pair_time_difference_seconds",
        help="Disable the centroid time gate for a timestamp-agnostic ablation",
    )
    parser.add_argument(
        "--split-noncontiguous-clusters",
        action="store_true",
        dest="split_noncontiguous_clusters",
        help="Split each visual cluster into separate temporal runs before comparison",
    )
    parser.add_argument(
        "--no-split-noncontiguous-clusters",
        action="store_false",
        dest="split_noncontiguous_clusters",
        help="Keep visually similar frames in one cluster even across temporal gaps",
    )
    parser.set_defaults(
        split_noncontiguous_clusters=DEFAULT_SPLIT_NONCONTIGUOUS_CLUSTERS
    )
    parser.add_argument(
        "--max-cluster-member-gap-seconds",
        type=float,
        help="Maximum adjacent timestamp gap inside one temporal cluster; defaults to 1.5x sampling interval",
    )
    parser.add_argument("--summarize-clusters", action="store_true")
    parser.add_argument(
        "--cluster-summary-backend",
        default="transformers-local",
        choices=[
            "transformers-local",
            "transformers-local-memory-safe",
            "openai-compatible-local",
            "openrouter",
            "gemini",
            "dry-run",
        ],
    )
    parser.add_argument(
        "--cluster-summary-model-id",
        default=DEFAULT_CLUSTER_SUMMARY_MODEL_ID,
    )
    parser.add_argument(
        "--cluster-summary-base-url",
        default="http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--cluster-summary-api-key")
    parser.add_argument("--cluster-summary-dtype", default="bfloat16")
    parser.add_argument(
        "--cluster-summary-max-new-tokens",
        type=int,
        default=4096,
        help="Maximum new tokens for the batched all-cluster summary and relation calls",
    )
    parser.add_argument("--cluster-summary-max-images", type=int, default=12)
    parser.add_argument("--cluster-summary-max-attempts", type=int, default=2)
    parser.add_argument("--cluster-summary-allow-cpu", action="store_true")
    parser.add_argument(
        "--cluster-summary-enable-thinking",
        action="store_true",
        help="Leave model thinking enabled; disabled by default for structured summaries",
    )
    parser.add_argument(
        "--compare-all-pairs",
        action="store_true",
        help="Embed every video in each synchronized group and compare all pairs; slower than the default random-pair-first path",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--sampling-policy",
        choices=SAMPLING_POLICIES,
        default=RANDOM_SAMPLING_POLICY,
    )
    parser.add_argument("--time-bin-count", type=int, default=10)
    parser.add_argument("--strict-balance", action="store_true")
    parser.add_argument("--group-partition-count", type=int, default=1)
    parser.add_argument("--group-partition-index", type=int, default=0)
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
        max_pair_time_difference_seconds=args.max_pair_time_difference_seconds,
        split_noncontiguous_clusters=args.split_noncontiguous_clusters,
        max_cluster_member_gap_seconds=args.max_cluster_member_gap_seconds,
        random_pair_first=not args.compare_all_pairs,
        random_seed=args.random_seed,
        sampling_policy=args.sampling_policy,
        time_bin_count=args.time_bin_count,
        strict_balance=args.strict_balance,
        group_partition_count=args.group_partition_count,
        group_partition_index=args.group_partition_index,
        ffmpeg_binary=args.ffmpeg_binary,
        download_media=args.download_media,
        review_dir=args.review_dir,
        summarize_clusters=args.summarize_clusters,
        cluster_summary_backend=args.cluster_summary_backend,
        cluster_summary_model_id=args.cluster_summary_model_id,
        cluster_summary_base_url=args.cluster_summary_base_url,
        cluster_summary_api_key=args.cluster_summary_api_key,
        cluster_summary_dtype=args.cluster_summary_dtype,
        cluster_summary_max_new_tokens=args.cluster_summary_max_new_tokens,
        cluster_summary_max_images=args.cluster_summary_max_images,
        cluster_summary_max_attempts=args.cluster_summary_max_attempts,
        cluster_summary_allow_cpu=args.cluster_summary_allow_cpu,
        cluster_summary_disable_thinking=not args.cluster_summary_enable_thinking,
    )
    print(f"wrote {len(candidates)} random-pair CLIP-pruned candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
