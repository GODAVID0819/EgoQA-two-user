"""Memory-bounded CLIP pruning for already-prepared two-video evidence packets.

This path is intentionally separate from the historical 30-second candidate
miners.  It consumes synchronized evidence packets whose videos have already
been assembled, prunes each pair, and routes only the pruned MP4s to the QA
generator.  Full original MP4s remain attached for judges and answerability.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from typing import Any, Iterable

from .clip_gap_demo import DEFAULT_CLIP_MODEL, ImageEncoder, TransformersClipEncoder
from .evidence import ffprobe_duration
from .group_relative_clip_sampling import (
    clustered_temporal_similarity_pruning,
    group_clip_frames,
    selected_clips_for_pair_from_rows,
)
from .io_utils import iter_jsonl, stable_id, write_json, write_jsonl


DEFAULT_EXPECTED_DURATION_SECONDS = 600.0
DEFAULT_SAMPLE_INTERVAL_SECONDS = 1.0
DEFAULT_CLUSTERS_PER_VIDEO = 80
DEFAULT_CLIP_BATCH_SIZE = 32


def encode_paths_in_batches(
    encoder: ImageEncoder,
    image_paths: Iterable[str],
    *,
    batch_size: int = DEFAULT_CLIP_BATCH_SIZE,
) -> list[list[float]]:
    """Encode a long sampled timeline without one all-frames processor batch."""

    if batch_size <= 0:
        raise ValueError("CLIP batch size must be positive")
    paths = list(image_paths)
    embeddings: list[list[float]] = []
    for start in range(0, len(paths), batch_size):
        batch = paths[start : start + batch_size]
        encoded = encoder.encode(batch)
        if len(encoded) != len(batch):
            raise RuntimeError(
                "CLIP encoder returned an unexpected number of embeddings: "
                f"expected={len(batch)} actual={len(encoded)}"
            )
        embeddings.extend(encoded)
    return embeddings


def _release_packet_memory(encoder: ImageEncoder) -> None:
    gc.collect()
    torch = getattr(encoder, "torch", None)
    cuda = getattr(torch, "cuda", None)
    if cuda is not None and cuda.is_available():
        cuda.empty_cache()


def _validate_raw_two_video_packet(
    packet: dict[str, Any],
    *,
    expected_duration_seconds: float,
    duration_tolerance_seconds: float,
) -> list[dict[str, Any]]:
    packet_id = str(packet.get("evidence_id") or "<missing evidence_id>")
    clips = packet.get("clips")
    if not isinstance(clips, list) or len(clips) != 2:
        count = len(clips) if isinstance(clips, list) else 0
        raise ValueError(f"{packet_id}: expected exactly two videos, found {count}")

    packet_duration = float(packet.get("duration_seconds") or 0.0)
    if abs(packet_duration - expected_duration_seconds) > duration_tolerance_seconds:
        raise ValueError(
            f"{packet_id}: evidence duration must be {expected_duration_seconds:g}s, "
            f"found {packet_duration:g}s"
        )

    for index, clip in enumerate(clips):
        if clip.get("generator_media_mode") == "pruned_video":
            raise ValueError(
                f"{packet_id}: clip {index} is already pruned; the input must contain the two "
                "unpruned synchronized videos"
            )
        local_video = clip.get("local_video")
        if not local_video or not Path(str(local_video)).is_file():
            raise FileNotFoundError(f"{packet_id}: clip {index} has no local source video")
        metadata_duration = float(clip.get("duration_seconds") or 0.0)
        if abs(metadata_duration - expected_duration_seconds) > duration_tolerance_seconds:
            raise ValueError(
                f"{packet_id}: clip {index} metadata duration must be "
                f"{expected_duration_seconds:g}s, found {metadata_duration:g}s"
            )
        measured_duration = ffprobe_duration(local_video)
        if (
            measured_duration is not None
            and abs(measured_duration - expected_duration_seconds) > duration_tolerance_seconds
        ):
            raise ValueError(
                f"{packet_id}: clip {index} measured duration must be "
                f"{expected_duration_seconds:g}s (+/-{duration_tolerance_seconds:g}s), "
                f"found {measured_duration:g}s"
            )
    return clips


def _compact_clip_pruning(clip: dict[str, Any]) -> dict[str, Any]:
    """Keep routing and interval facts in evidence; leave frame traces on disk."""

    compact = dict(clip)
    temporal = compact.get("temporal_pruning")
    if isinstance(temporal, dict):
        temporal = dict(temporal)
        for key in (
            "cluster_decisions",
            "kept_cluster_representatives",
            "restored_frames",
        ):
            temporal.pop(key, None)
        compact["temporal_pruning"] = temporal
    return compact


def _compact_packet_pruning(
    pruning: dict[str, Any],
    *,
    diagnostics_path: Path,
) -> dict[str, Any]:
    return {
        "method": pruning.get("method"),
        "clusters_per_video": pruning.get("cluster_count"),
        "left_cluster_count": pruning.get("left_cluster_count"),
        "right_cluster_count": pruning.get("right_cluster_count"),
        "high_similarity_threshold": pruning.get("high_similarity_threshold"),
        "high_similarity_representative_pair_count": pruning.get(
            "high_similarity_representative_pair_count"
        ),
        "pruning_protection_mode": pruning.get("pruning_protection_mode"),
        "min_pruned_video_seconds": pruning.get("min_pruned_video_seconds"),
        "min_pruned_video_percent": pruning.get("min_pruned_video_percent"),
        "left_kept_duration_seconds": pruning.get("left_kept_duration_seconds"),
        "right_kept_duration_seconds": pruning.get("right_kept_duration_seconds"),
        "left_removed_duration_seconds": pruning.get("left_removed_duration_seconds"),
        "right_removed_duration_seconds": pruning.get("right_removed_duration_seconds"),
        "removed_duration_seconds": pruning.get("removed_duration_seconds"),
        "no_high_similarity_intervals": float(pruning.get("removed_duration_seconds") or 0.0)
        == 0.0,
        "diagnostics_path": str(diagnostics_path),
        "media_routing": {
            "generator": "clips[*].local_video (K-pruned MP4)",
            "judges_and_answerability": "clips[*].full_local_video (unpruned source MP4)",
        },
    }


def prune_prepared_evidence_pairs(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    cache_dir: str | Path,
    model_id: str = DEFAULT_CLIP_MODEL,
    expected_duration_seconds: float = DEFAULT_EXPECTED_DURATION_SECONDS,
    duration_tolerance_seconds: float = 1.0,
    sample_interval_seconds: float = DEFAULT_SAMPLE_INTERVAL_SECONDS,
    start_seconds: float = 0.0,
    clusters_per_video: int = DEFAULT_CLUSTERS_PER_VIDEO,
    high_similarity_threshold: float = 0.82,
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
    """Prune every prepared two-video packet without pair-selection filtering."""

    if expected_duration_seconds <= 0:
        raise ValueError("expected duration must be positive")
    if sample_interval_seconds <= 0:
        raise ValueError("sample interval must be positive")
    if clusters_per_video <= 0:
        raise ValueError("clusters per video must be positive")
    if max_packets is not None and max_packets <= 0:
        raise ValueError("max packets must be positive when provided")

    output_path = Path(output_path)
    output_dir = Path(output_dir)
    diagnostics_dir = output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    active_encoder = encoder or TransformersClipEncoder(model_id, device=device)
    stats = {
        "packet_count": 0,
        "video_count": 0,
        "packet_count_with_removal": 0,
        "packet_count_without_removal": 0,
        "total_removed_video_seconds": 0.0,
        "total_kept_video_seconds": 0.0,
    }

    def pruned_rows() -> Iterable[dict[str, Any]]:
        for packet_index, packet in enumerate(iter_jsonl(evidence_path)):
            if max_packets is not None and packet_index >= max_packets:
                break
            clips = _validate_raw_two_video_packet(
                packet,
                expected_duration_seconds=expected_duration_seconds,
                duration_tolerance_seconds=duration_tolerance_seconds,
            )
            packet_id = str(packet.get("evidence_id") or f"packet_{packet_index:05d}")
            pair_id = stable_id(
                packet_id,
                clips[0].get("agent_dir") or clips[0].get("agent_name") or "left",
                clips[1].get("agent_dir") or clips[1].get("agent_name") or "right",
                f"K{clusters_per_video}",
            )
            group = {
                "day": packet.get("day"),
                "time_token": packet.get("time_token"),
                "clip_clock": packet.get("clip_clock"),
                "clips": clips,
            }
            rows = group_clip_frames(
                group,
                output_dir / "sampling" / pair_id,
                cache_dir=cache_dir,
                duration_seconds=expected_duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                start_seconds=start_seconds,
                ffmpeg_binary=ffmpeg_binary,
                download_media=False,
            )
            if len(rows) != 2:
                raise RuntimeError(f"{packet_id}: frame sampler did not return exactly two videos")

            embeddings = [
                encode_paths_in_batches(
                    active_encoder,
                    [str(frame["path"]) for frame in row["frames"]],
                    batch_size=clip_batch_size,
                )
                for row in rows
            ]
            pruning = clustered_temporal_similarity_pruning(
                rows[0]["frames"],
                rows[1]["frames"],
                embeddings[0],
                embeddings[1],
                start_seconds=start_seconds,
                duration_seconds=expected_duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                cluster_count=clusters_per_video,
                high_similarity_threshold=high_similarity_threshold,
                preserve_shared_anchor_seconds=preserve_shared_anchor_seconds,
                min_pruned_video_seconds=min_pruned_video_seconds,
                pruning_protection_mode=pruning_protection_mode,
                min_pruned_video_percent=min_pruned_video_percent,
            )
            pair = {
                "pair_key": pair_id,
                "left_index": 0,
                "right_index": 1,
                "temporal_pruning": pruning,
            }
            pruned_clips = selected_clips_for_pair_from_rows(
                rows,
                pair,
                output_dir=output_dir,
                ffmpeg_binary=ffmpeg_binary,
            )
            diagnostics_path = diagnostics_dir / f"{pair_id}.json"
            write_json(
                diagnostics_path,
                {
                    "evidence_id": packet_id,
                    "pair_id": pair_id,
                    "model_id": getattr(active_encoder, "model_id", model_id),
                    "sampled_frames": {
                        "left": rows[0]["frames"],
                        "right": rows[1]["frames"],
                    },
                    "temporal_pruning": pruning,
                },
            )

            removed = float(pruning.get("removed_duration_seconds") or 0.0)
            stats["packet_count"] += 1
            stats["video_count"] += 2
            stats["total_removed_video_seconds"] += removed
            stats["total_kept_video_seconds"] += float(
                pruning.get("left_kept_duration_seconds") or 0.0
            ) + float(pruning.get("right_kept_duration_seconds") or 0.0)
            if removed > 0:
                stats["packet_count_with_removal"] += 1
            else:
                stats["packet_count_without_removal"] += 1

            output_packet = dict(packet)
            output_packet["clips"] = [_compact_clip_pruning(clip) for clip in pruned_clips]
            output_packet["paired_video_pruning"] = _compact_packet_pruning(
                pruning,
                diagnostics_path=diagnostics_path,
            )
            output_packet["candidate_type"] = f"ten_minute_k{clusters_per_video}_pruned_pair"
            yield output_packet

            del embeddings, pruning, rows, pruned_clips, output_packet
            _release_packet_memory(active_encoder)

    temporary_output = output_path.with_name(f"{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        written_count = write_jsonl(temporary_output, pruned_rows())
        if written_count == 0:
            raise ValueError(f"no evidence packets were found in {evidence_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    summary = {
        **stats,
        "total_removed_video_seconds": round(stats["total_removed_video_seconds"], 3),
        "total_kept_video_seconds": round(stats["total_kept_video_seconds"], 3),
        "input_evidence": str(evidence_path),
        "output_evidence": str(output_path),
        "settings": {
            "expected_video_count_per_packet": 2,
            "expected_duration_seconds": expected_duration_seconds,
            "duration_tolerance_seconds": duration_tolerance_seconds,
            "sample_interval_seconds": sample_interval_seconds,
            "start_seconds": start_seconds,
            "clusters_per_video": clusters_per_video,
            "high_similarity_threshold": high_similarity_threshold,
            "preserve_shared_anchor_seconds": preserve_shared_anchor_seconds,
            "min_pruned_video_seconds": min_pruned_video_seconds,
            "pruning_protection_mode": pruning_protection_mode,
            "min_pruned_video_percent": min_pruned_video_percent,
            "clip_batch_size": clip_batch_size,
            "model_id": getattr(active_encoder, "model_id", model_id),
            "pair_filtering": False,
        },
        "media_routing": {
            "generator": "pruned videos in clips[*].local_video",
            "judges_and_answerability": "unpruned ten-minute videos in clips[*].full_local_video",
        },
    }
    write_json(output_dir / "pruning_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prune prepared synchronized two-video evidence packets before QA generation"
    )
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", default=".cache/egolife_two_user_qa")
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--expected-duration-seconds", type=float, default=600.0)
    parser.add_argument("--duration-tolerance-seconds", type=float, default=1.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.0)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--clusters-per-video", type=int, default=80)
    parser.add_argument("--high-similarity-threshold", type=float, default=0.82)
    parser.add_argument("--preserve-shared-anchor-seconds", type=float, default=0.0)
    parser.add_argument("--min-pruned-video-seconds", type=float, default=8.0)
    parser.add_argument(
        "--pruning-protection-mode",
        choices=["reject", "min_seconds", "min_percent"],
        default="min_seconds",
    )
    parser.add_argument("--min-pruned-video-percent", type=float)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--device", default="auto")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = prune_prepared_evidence_pairs(
        evidence_path=args.evidence,
        output_path=args.output,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        expected_duration_seconds=args.expected_duration_seconds,
        duration_tolerance_seconds=args.duration_tolerance_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        start_seconds=args.start_seconds,
        clusters_per_video=args.clusters_per_video,
        high_similarity_threshold=args.high_similarity_threshold,
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
        f"wrote {summary['packet_count']} K={summary['settings']['clusters_per_video']} "
        f"pruned evidence pairs to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
