"""Paired evidence experiment for global and temporally contiguous clustering.

This module is deliberately separate from the production evidence builders. It
samples and embeds each input video pair once, then applies two pruning arms to
those identical inputs:

* global cosine K-means followed by cross-video timestamp-gated pruning;
* adjacency-constrained temporal agglomeration followed by the same pruning.

Both arms emit ordinary pruned evidence packets so they can be sent through the
same downstream QA generation and judging pipeline.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from .clip_gap_demo import DEFAULT_CLIP_MODEL, ImageEncoder, TransformersClipEncoder
from .group_relative_clip_sampling import (
    DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    clustered_temporal_similarity_pruning,
    group_clip_frames,
    selected_clips_for_pair_from_rows,
)
from .io_utils import iter_jsonl, stable_id, write_json, write_jsonl
from .paired_evidence_pruning import (
    DEFAULT_CLIP_BATCH_SIZE,
    _compact_clip_pruning,
    _release_packet_memory,
    encode_paths_in_batches,
)
from .temporal_clustering import (
    COSINE_KMEANS_CLUSTERING,
    TEMPORAL_AGGLOMERATIVE_CLUSTERING,
)


BASELINE_ARM = "kmeans_time_gate"
TEMPORAL_ARM = "temporal_agglomerative_time_gate"
ARM_METHODS = {
    BASELINE_ARM: COSINE_KMEANS_CLUSTERING,
    TEMPORAL_ARM: TEMPORAL_AGGLOMERATIVE_CLUSTERING,
}
DEFAULT_DURATION_SECONDS = 30.0


def _source_clips(packet: dict[str, Any]) -> list[dict[str, Any]]:
    clips = packet.get("clips")
    packet_id = str(packet.get("evidence_id") or "<missing evidence_id>")
    if not isinstance(clips, list) or len(clips) != 2:
        count = len(clips) if isinstance(clips, list) else 0
        raise ValueError(f"{packet_id}: expected exactly two clips, found {count}")

    source_clips = []
    for clip_index, clip in enumerate(clips):
        source_path = next(
            (
                Path(str(clip[field]))
                for field in (
                    "full_local_video",
                    "original_local_video",
                    "source_local_video",
                    "local_video",
                )
                if clip.get(field) and Path(str(clip[field])).is_file()
            ),
            None,
        )
        if source_path is None:
            raise FileNotFoundError(
                f"{packet_id}: clip {clip_index} has no available full/source local video"
            )
        source_clip = dict(clip)
        source_clip["local_video"] = str(source_path)
        source_clip.pop("generator_local_video", None)
        source_clip.pop("generator_media_mode", None)
        source_clips.append(source_clip)
    return source_clips


def _packet_duration_seconds(
    packet: dict[str, Any],
    clips: list[dict[str, Any]],
    configured_duration_seconds: float | None,
) -> float:
    if configured_duration_seconds is not None:
        duration = float(configured_duration_seconds)
    else:
        clip_durations = [
            float(clip.get("duration_seconds") or 0.0) for clip in clips
        ]
        positive_clip_durations = [value for value in clip_durations if value > 0]
        duration = float(
            packet.get("duration_seconds")
            or (min(positive_clip_durations) if positive_clip_durations else 0.0)
            or DEFAULT_DURATION_SECONDS
        )
    if duration <= 0:
        raise ValueError("duration_seconds must be positive")
    return duration


def _compact_pruning(
    pruning: dict[str, Any],
    *,
    diagnostics_path: Path,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    return {
        "method": pruning.get("method"),
        "clustering_method": pruning.get("clustering_method"),
        "sampling_fps": round(1.0 / sample_interval_seconds, 6),
        "sample_interval_seconds": sample_interval_seconds,
        "clusters_per_video": pruning.get("cluster_count"),
        "left_cluster_count": pruning.get("left_cluster_count"),
        "right_cluster_count": pruning.get("right_cluster_count"),
        "high_similarity_threshold": pruning.get("high_similarity_threshold"),
        "max_pair_time_difference_seconds": pruning.get(
            "max_pair_time_difference_seconds"
        ),
        "temporal_pairing_policy": pruning.get("temporal_pairing_policy"),
        "high_similarity_representative_pair_count": pruning.get(
            "high_similarity_representative_pair_count"
        ),
        "preserved_cross_time_representative_pair_count": pruning.get(
            "preserved_cross_time_representative_pair_count"
        ),
        "pruning_protection_mode": pruning.get("pruning_protection_mode"),
        "min_pruned_video_seconds": pruning.get("min_pruned_video_seconds"),
        "min_pruned_video_percent": pruning.get("min_pruned_video_percent"),
        "left_kept_duration_seconds": pruning.get("left_kept_duration_seconds"),
        "right_kept_duration_seconds": pruning.get("right_kept_duration_seconds"),
        "left_removed_duration_seconds": pruning.get("left_removed_duration_seconds"),
        "right_removed_duration_seconds": pruning.get("right_removed_duration_seconds"),
        "removed_duration_seconds": pruning.get("removed_duration_seconds"),
        "passed": pruning.get("passed"),
        "diagnostics_path": str(diagnostics_path),
        "media_routing": {
            "generator": "clips[*].local_video (arm-specific pruned MP4)",
            "judges_and_answerability": (
                "clips[*].full_local_video (the same unpruned source in both arms)"
            ),
        },
    }


def _noncontiguous_cluster_count(
    decisions: list[dict[str, Any]],
    *,
    sample_interval_seconds: float,
) -> int:
    max_gap = 1.5 * float(sample_interval_seconds)
    count = 0
    for decision in decisions:
        timestamps = sorted(
            float(value)
            for value in decision.get("member_timestamps", [])
            if value is not None
        )
        if any(
            right - left > max_gap + 1e-9
            for left, right in zip(timestamps, timestamps[1:])
        ):
            count += 1
    return count


def _arm_metrics(
    pruning: dict[str, Any],
    *,
    sample_interval_seconds: float,
) -> dict[str, Any]:
    left_marked = set(int(value) for value in pruning["left_marked_frame_indices"])
    right_marked = set(int(value) for value in pruning["right_marked_frame_indices"])
    return {
        "clustering_method": pruning["clustering_method"],
        "left_cluster_count": pruning["left_cluster_count"],
        "right_cluster_count": pruning["right_cluster_count"],
        "left_noncontiguous_cluster_count": _noncontiguous_cluster_count(
            pruning["left_cluster_decisions"],
            sample_interval_seconds=sample_interval_seconds,
        ),
        "right_noncontiguous_cluster_count": _noncontiguous_cluster_count(
            pruning["right_cluster_decisions"],
            sample_interval_seconds=sample_interval_seconds,
        ),
        "high_similarity_representative_pair_count": pruning[
            "high_similarity_representative_pair_count"
        ],
        "preserved_cross_time_representative_pair_count": pruning[
            "preserved_cross_time_representative_pair_count"
        ],
        "left_marked_frame_indices": sorted(left_marked),
        "right_marked_frame_indices": sorted(right_marked),
        "left_marked_frame_count": len(left_marked),
        "right_marked_frame_count": len(right_marked),
        "left_kept_duration_seconds": pruning["left_kept_duration_seconds"],
        "right_kept_duration_seconds": pruning["right_kept_duration_seconds"],
        "removed_duration_seconds": pruning["removed_duration_seconds"],
        "passed": pruning["passed"],
    }


def _comparison_row(
    *,
    packet_id: str,
    pair_id: str,
    packet: dict[str, Any],
    arm_pruning: dict[str, dict[str, Any]],
    sample_interval_seconds: float,
) -> dict[str, Any]:
    baseline = _arm_metrics(
        arm_pruning[BASELINE_ARM],
        sample_interval_seconds=sample_interval_seconds,
    )
    temporal = _arm_metrics(
        arm_pruning[TEMPORAL_ARM],
        sample_interval_seconds=sample_interval_seconds,
    )
    baseline_left = set(baseline["left_marked_frame_indices"])
    baseline_right = set(baseline["right_marked_frame_indices"])
    temporal_left = set(temporal["left_marked_frame_indices"])
    temporal_right = set(temporal["right_marked_frame_indices"])
    return {
        "evidence_id": packet_id,
        "pair_id": pair_id,
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "arms": {
            BASELINE_ARM: baseline,
            TEMPORAL_ARM: temporal,
        },
        "delta_temporal_minus_kmeans": {
            "left_cluster_count": temporal["left_cluster_count"]
            - baseline["left_cluster_count"],
            "right_cluster_count": temporal["right_cluster_count"]
            - baseline["right_cluster_count"],
            "left_noncontiguous_cluster_count": temporal[
                "left_noncontiguous_cluster_count"
            ]
            - baseline["left_noncontiguous_cluster_count"],
            "right_noncontiguous_cluster_count": temporal[
                "right_noncontiguous_cluster_count"
            ]
            - baseline["right_noncontiguous_cluster_count"],
            "left_marked_frame_count": temporal["left_marked_frame_count"]
            - baseline["left_marked_frame_count"],
            "right_marked_frame_count": temporal["right_marked_frame_count"]
            - baseline["right_marked_frame_count"],
            "removed_duration_seconds": round(
                float(temporal["removed_duration_seconds"])
                - float(baseline["removed_duration_seconds"]),
                6,
            ),
        },
        "changed_pruning_assignments": {
            "left_frame_indices": sorted(baseline_left ^ temporal_left),
            "right_frame_indices": sorted(baseline_right ^ temporal_right),
        },
    }


def run_temporal_clustering_evidence_sidecar(
    *,
    evidence_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    model_id: str = DEFAULT_CLIP_MODEL,
    duration_seconds: float | None = None,
    sample_interval_seconds: float = 1.0,
    start_seconds: float = 0.0,
    clusters_per_video: int = 12,
    high_similarity_threshold: float = 0.82,
    max_pair_time_difference_seconds: float = DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    preserve_shared_anchor_seconds: float = 0.0,
    min_pruned_video_seconds: float = 8.0,
    pruning_protection_mode: str = "min_seconds",
    min_pruned_video_percent: float | None = None,
    clip_batch_size: int = DEFAULT_CLIP_BATCH_SIZE,
    max_packets: int | None = None,
    ffmpeg_binary: str = "ffmpeg",
    device: str = "auto",
    encoder: ImageEncoder | None = None,
) -> dict[str, Any]:
    """Generate two paired evidence sets while sharing sampling and embeddings."""

    if sample_interval_seconds <= 0:
        raise ValueError("sample_interval_seconds must be positive")
    if clusters_per_video <= 0:
        raise ValueError("clusters_per_video must be positive")
    if max_pair_time_difference_seconds < 0:
        raise ValueError("max_pair_time_difference_seconds must be non-negative")
    if max_packets is not None and max_packets <= 0:
        raise ValueError("max_packets must be positive when provided")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    comparison_path = output_root / "clustering_comparisons.jsonl"
    arm_output_paths = {
        arm: output_root / arm / "evidence.jsonl" for arm in ARM_METHODS
    }
    active_encoder = encoder or TransformersClipEncoder(model_id, device=device)
    arm_packets: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_METHODS}
    comparison_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    arm_removed_seconds: Counter[str] = Counter()

    for packet_index, packet in enumerate(iter_jsonl(evidence_path)):
        if max_packets is not None and packet_index >= max_packets:
            break
        packet_id = str(packet.get("evidence_id") or f"packet_{packet_index:05d}")
        try:
            clips = _source_clips(packet)
            packet_duration = _packet_duration_seconds(packet, clips, duration_seconds)
            agents = [
                str(clip.get("agent_dir") or clip.get("agent_name") or index)
                for index, clip in enumerate(clips)
            ]
            pair_id = stable_id(packet_id, *sorted(agents), f"K{clusters_per_video}")
            group = {
                "day": packet.get("day"),
                "time_token": packet.get("time_token"),
                "clip_clock": packet.get("clip_clock"),
                "clips": clips,
            }
            rows = group_clip_frames(
                group,
                output_root / "shared_sampling" / pair_id,
                cache_dir=cache_dir,
                duration_seconds=packet_duration,
                sample_interval_seconds=sample_interval_seconds,
                start_seconds=start_seconds,
                ffmpeg_binary=ffmpeg_binary,
                download_media=False,
            )
            if len(rows) != 2:
                raise RuntimeError(f"{packet_id}: frame sampler did not return two videos")
            embeddings = [
                encode_paths_in_batches(
                    active_encoder,
                    [str(frame["path"]) for frame in row["frames"]],
                    batch_size=clip_batch_size,
                )
                for row in rows
            ]

            current_arm_packets: dict[str, dict[str, Any]] = {}
            arm_pruning: dict[str, dict[str, Any]] = {}
            for arm, clustering_method in ARM_METHODS.items():
                pruning = clustered_temporal_similarity_pruning(
                    rows[0]["frames"],
                    rows[1]["frames"],
                    embeddings[0],
                    embeddings[1],
                    start_seconds=start_seconds,
                    duration_seconds=packet_duration,
                    sample_interval_seconds=sample_interval_seconds,
                    cluster_count=clusters_per_video,
                    clustering_method=clustering_method,
                    high_similarity_threshold=high_similarity_threshold,
                    preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
                    min_pruned_video_seconds=min_pruned_video_seconds,
                    pruning_protection_mode=pruning_protection_mode,
                    min_pruned_video_percent=min_pruned_video_percent,
                    max_pair_time_difference_seconds=max_pair_time_difference_seconds,
                    split_noncontiguous_clusters=False,
                )
                arm_pair_id = f"{pair_id}_{arm}"
                pair = {
                    "pair_key": arm_pair_id,
                    "left_index": 0,
                    "right_index": 1,
                    "temporal_pruning": pruning,
                }
                pruned_clips = selected_clips_for_pair_from_rows(
                    rows,
                    pair,
                    output_dir=output_root / arm,
                    ffmpeg_binary=ffmpeg_binary,
                )
                diagnostics_path = output_root / arm / "diagnostics" / f"{pair_id}.json"
                write_json(
                    diagnostics_path,
                    {
                        "evidence_id": packet_id,
                        "pair_id": pair_id,
                        "arm": arm,
                        "clustering_method": clustering_method,
                        "model_id": getattr(active_encoder, "model_id", model_id),
                        "shared_sampled_frames": {
                            "left": rows[0]["frames"],
                            "right": rows[1]["frames"],
                        },
                        "temporal_pruning": pruning,
                    },
                )
                output_packet = dict(packet)
                output_packet["clips"] = [
                    _compact_clip_pruning(clip) for clip in pruned_clips
                ]
                output_packet["paired_video_pruning"] = _compact_pruning(
                    pruning,
                    diagnostics_path=diagnostics_path,
                    sample_interval_seconds=sample_interval_seconds,
                )
                output_packet["generator_media_mode"] = "pruned_video"
                output_packet["candidate_type"] = (
                    f"temporal_clustering_experiment_{arm}"
                )
                output_packet["temporal_clustering_experiment"] = {
                    "arm": arm,
                    "clustering_method": clustering_method,
                    "paired_evidence_id": packet_id,
                    "comparison_path": str(comparison_path),
                    "shared_sampling_and_embeddings": True,
                    "clusters_must_be_contiguous": arm == TEMPORAL_ARM,
                }
                current_arm_packets[arm] = output_packet
                arm_pruning[arm] = pruning

            comparison_rows.append(
                _comparison_row(
                    packet_id=packet_id,
                    pair_id=pair_id,
                    packet=packet,
                    arm_pruning=arm_pruning,
                    sample_interval_seconds=sample_interval_seconds,
                )
            )
            for arm, output_packet in current_arm_packets.items():
                arm_packets[arm].append(output_packet)
                arm_removed_seconds[arm] += float(
                    arm_pruning[arm].get("removed_duration_seconds") or 0.0
                )
            del embeddings, rows, arm_pruning, current_arm_packets
            _release_packet_memory(active_encoder)
        except Exception as exc:
            skipped.append(
                {
                    "packet_index": packet_index,
                    "evidence_id": packet_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    for arm, output_path in arm_output_paths.items():
        write_jsonl(output_path, arm_packets[arm])
    write_jsonl(comparison_path, comparison_rows)

    paired_count = len(comparison_rows)
    summary = {
        "experiment": "global_kmeans_vs_temporal_agglomerative_clustering",
        "input_evidence": str(evidence_path),
        "output_dir": str(output_root),
        "paired_packet_count": paired_count,
        "skipped_packet_count": len(skipped),
        "skipped_packets": skipped,
        "comparison_path": str(comparison_path),
        "arms": {
            arm: {
                "clustering_method": ARM_METHODS[arm],
                "evidence_path": str(arm_output_paths[arm]),
                "packet_count": len(arm_packets[arm]),
                "total_removed_video_seconds": round(arm_removed_seconds[arm], 3),
            }
            for arm in ARM_METHODS
        },
        "settings": {
            "model_id": getattr(active_encoder, "model_id", model_id),
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "start_seconds": start_seconds,
            "clusters_per_video": clusters_per_video,
            "high_similarity_threshold": high_similarity_threshold,
            "max_pair_time_difference_seconds": max_pair_time_difference_seconds,
            "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_percent": min_pruned_video_percent,
            "clip_batch_size": clip_batch_size,
            "max_packets": max_packets,
            "fairness_contract": (
                "Each paired packet shares source videos, sampled frames, CLIP embeddings, K, "
                "cross-video timestamp gate, similarity threshold, and duration protection; only "
                "the within-video clustering method changes."
            ),
        },
    }
    write_json(output_root / "experiment_summary.json", summary)
    if paired_count == 0:
        raise ValueError(
            "no paired evidence packets were generated; inspect experiment_summary.json"
        )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate paired K-means and contiguous temporal-agglomerative CLIP-pruned evidence"
        )
    )
    parser.add_argument("--evidence", required=True, help="Input two-video evidence JSONL")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    parser.add_argument(
        "--duration-seconds",
        type=float,
        help="Window length; by default use each packet's duration_seconds",
    )
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--clusters-per-video", type=int, default=12)
    parser.add_argument("--high-similarity-threshold", type=float, default=0.82)
    parser.add_argument(
        "--max-pair-time-difference-seconds",
        type=float,
        default=DEFAULT_MAX_PAIR_TIME_DIFFERENCE_SECONDS,
    )
    parser.add_argument("--preserve-shared-anchor-seconds", type=float, default=0.0)
    parser.add_argument("--min-pruned-video-seconds", type=float, default=8.0)
    parser.add_argument(
        "--pruning-protection-mode",
        choices=("reject", "min_seconds", "min_percent"),
        default="min_seconds",
    )
    parser.add_argument("--min-pruned-video-percent", type=float)
    parser.add_argument("--clip-batch-size", type=int, default=DEFAULT_CLIP_BATCH_SIZE)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_temporal_clustering_evidence_sidecar(
        evidence_path=args.evidence,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        duration_seconds=args.duration_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        start_seconds=args.start_seconds,
        clusters_per_video=args.clusters_per_video,
        high_similarity_threshold=args.high_similarity_threshold,
        max_pair_time_difference_seconds=args.max_pair_time_difference_seconds,
        preserve_shared_anchor_seconds=args.preserve_shared_anchor_seconds,
        min_pruned_video_seconds=args.min_pruned_video_seconds,
        pruning_protection_mode=args.pruning_protection_mode,
        min_pruned_video_percent=args.min_pruned_video_percent,
        clip_batch_size=args.clip_batch_size,
        max_packets=args.max_packets,
        ffmpeg_binary=args.ffmpeg_binary,
        device=args.device,
    )
    print(
        f"wrote {summary['paired_packet_count']} paired evidence packets to "
        f"{summary['output_dir']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
