"""Sidecar evidence pipeline that sends retained CLIP centroids to the generator.

The production pruning pipeline materializes retained one-second intervals as a
new MP4.  This module deliberately leaves that pipeline untouched.  It consumes
its evidence JSONL, copies only the representative frames for clusters that
remain after pruning, and writes a separate evidence JSONL whose generator
media is an ordered image list.  Full original videos remain routed to visual
judges and answerability checks.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .io_utils import iter_jsonl, stable_id, write_json, write_jsonl


GENERATOR_MEDIA_MODE = "centroid_frames_only"


def _safe_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._")
    return text[:80] or "unknown"


def _existing_file(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        path = Path(str(value))
        if path.is_file():
            return path
    return None


def _load_packet_diagnostics(packet: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    pruning = packet.get("paired_video_pruning")
    diagnostics_value = pruning.get("diagnostics_path") if isinstance(pruning, dict) else None
    if not diagnostics_value:
        return {}, None
    diagnostics_path = Path(str(diagnostics_value))
    if not diagnostics_path.is_file():
        raise FileNotFoundError(
            f"{packet.get('evidence_id')}: pruning diagnostics are missing: {diagnostics_path}"
        )
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{packet.get('evidence_id')}: pruning diagnostics must be a JSON object")
    temporal = payload.get("temporal_pruning")
    return (temporal if isinstance(temporal, dict) else {}), diagnostics_path


def _retained_representatives(
    clip: dict[str, Any],
    *,
    side: str,
    packet_pruning: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return one medoid for every cluster with content left after protection.

    A duration-protection floor can restore individual sampled intervals from a
    cluster initially marked for pruning.  Such a cluster still has retained
    content, so its medoid remains useful even when the medoid's own interval
    was not the restored member.
    """

    temporal = clip.get("temporal_pruning")
    temporal = temporal if isinstance(temporal, dict) else {}
    decisions = temporal.get("cluster_decisions")
    if not isinstance(decisions, list) or not decisions:
        decisions = packet_pruning.get(f"{side}_cluster_decisions")

    restored_values = temporal.get("restored_frame_indices")
    if not isinstance(restored_values, list):
        restored_values = packet_pruning.get(f"{side}_restored_frame_indices", [])
    restored_indices = {int(value) for value in restored_values or []}

    if not isinstance(decisions, list) or not decisions:
        decisions = temporal.get("kept_cluster_representatives")
        if not isinstance(decisions, list) or not decisions:
            raise ValueError(
                f"{clip.get('clip_id') or clip.get('agent_dir') or side}: "
                "no cluster representative decisions are available"
            )
        decisions = [{**row, "status": "kept"} for row in decisions if isinstance(row, dict)]

    retained = []
    seen_cluster_indices: set[int] = set()
    for row in decisions:
        if not isinstance(row, dict):
            continue
        cluster_index = int(row.get("cluster_index", len(seen_cluster_indices)))
        if cluster_index in seen_cluster_indices:
            continue
        member_indices = {int(value) for value in row.get("member_indices", [])}
        kept_by_pruning = str(row.get("status") or "") == "kept"
        kept_by_protection = bool(member_indices & restored_indices)
        if not kept_by_pruning and not kept_by_protection:
            continue
        retained.append(
            {
                **row,
                "cluster_index": cluster_index,
                "retention_reason": (
                    "cluster_kept_by_pruning"
                    if kept_by_pruning
                    else "cluster_has_duration_protection_restore"
                ),
            }
        )
        seen_cluster_indices.add(cluster_index)

    retained.sort(
        key=lambda row: (
            float(row.get("timestamp_seconds") or 0.0),
            int(row.get("cluster_index") or 0),
        )
    )
    if not retained:
        raise ValueError(
            f"{clip.get('clip_id') or clip.get('agent_dir') or side}: "
            "pruning left no cluster centroid frames for the generator"
        )
    return retained


def _copy_representative_frames(
    representatives: list[dict[str, Any]],
    *,
    destination_dir: Path,
    evidence_parent: Path,
    diagnostics_parent: Path | None,
) -> list[dict[str, Any]]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for order, representative in enumerate(representatives, 1):
        source_value = representative.get("path")
        candidates = [Path(str(source_value))] if source_value else []
        if source_value and not Path(str(source_value)).is_absolute():
            candidates.append(evidence_parent / str(source_value))
            if diagnostics_parent is not None:
                candidates.append(diagnostics_parent / str(source_value))
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is None:
            raise FileNotFoundError(f"retained centroid frame is missing: {source_value!r}")

        cluster_index = int(representative["cluster_index"])
        timestamp = float(representative.get("timestamp_seconds") or 0.0)
        timestamp_ms = round(timestamp * 1000.0)
        suffix = source.suffix.lower() or ".jpg"
        destination = destination_dir / (
            f"{order:03d}_cluster_{cluster_index:03d}_t{timestamp_ms:09d}ms{suffix}"
        )
        shutil.copy2(source, destination)
        copied.append(
            {
                "path": str(destination),
                "source_path": str(source),
                "frame_role": "retained_clip_cluster_centroid",
                "input_order_within_user": order,
                "cluster_index": cluster_index,
                "timestamp_seconds": representative.get("timestamp_seconds"),
                "member_count": representative.get("member_count"),
                "retention_reason": representative.get("retention_reason"),
            }
        )
    return copied


def _frame_only_clip(
    clip: dict[str, Any],
    *,
    side: str,
    evidence_id: str,
    output_dir: Path,
    evidence_parent: Path,
    packet_pruning: dict[str, Any],
    diagnostics_parent: Path | None,
) -> dict[str, Any]:
    if clip.get("generator_media_mode") == GENERATOR_MEDIA_MODE:
        raise ValueError(f"{evidence_id}: input clip is already a centroid-frame sidecar")

    full_video = _existing_file(
        clip.get("full_local_video"),
        clip.get("original_local_video"),
        clip.get("source_local_video"),
        (clip.get("benchmark_media") or {}).get("judge_video")
        if isinstance(clip.get("benchmark_media"), dict)
        else None,
    )
    if full_video is None:
        raise FileNotFoundError(
            f"{evidence_id}: {side} clip has no existing full original video for judge routing"
        )

    representatives = _retained_representatives(
        clip,
        side=side,
        packet_pruning=packet_pruning,
    )
    packet_key = _safe_part(stable_id("centroid_frame_sidecar", evidence_id))
    agent = _safe_part(clip.get("agent_dir") or clip.get("agent_name") or side)
    frames = _copy_representative_frames(
        representatives,
        destination_dir=output_dir / "centroid_frames" / packet_key / f"{side}_{agent}",
        evidence_parent=evidence_parent,
        diagnostics_parent=diagnostics_parent,
    )

    current = dict(clip)
    source_pruned_video = current.get("generator_local_video")
    if not source_pruned_video and current.get("generator_media_mode") == "pruned_video":
        source_pruned_video = current.get("local_video")
    current.pop("local_video", None)
    current.pop("generator_local_video", None)
    current["source_pruned_video"] = source_pruned_video
    current["source_local_video"] = str(full_video)
    current["original_local_video"] = str(full_video)
    current["full_local_video"] = str(full_video)
    current["generator_media_mode"] = GENERATOR_MEDIA_MODE
    current["force_frame_inputs"] = True
    current["frames"] = frames

    temporal = current.get("temporal_pruning")
    temporal = dict(temporal) if isinstance(temporal, dict) else {}
    temporal.pop("cluster_decisions", None)
    temporal["side"] = side
    temporal["kept_cluster_representatives"] = frames
    temporal["kept_cluster_count"] = len(frames)
    temporal["generator_representation"] = "retained_cluster_centroid_frames"
    current["temporal_pruning"] = temporal

    benchmark_media = current.get("benchmark_media")
    benchmark_media = dict(benchmark_media) if isinstance(benchmark_media, dict) else {}
    benchmark_media.pop("generator_video", None)
    benchmark_media["generator_frames"] = [frame["path"] for frame in frames]
    benchmark_media["judge_video"] = str(full_video)
    benchmark_media["answerability_video"] = str(full_video)
    current["benchmark_media"] = benchmark_media
    return current


def prepare_centroid_frame_evidence(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    start_index: int = 0,
    max_packets: int | None = None,
    expected_packet_count: int | None = None,
) -> dict[str, Any]:
    """Create an isolated frame-only generator evidence set from pruned packets."""

    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    if max_packets is not None and max_packets <= 0:
        raise ValueError("max_packets must be positive when provided")
    if expected_packet_count is not None and expected_packet_count <= 0:
        raise ValueError("expected_packet_count must be positive when provided")

    evidence_path = Path(evidence_path)
    output_path = Path(output_path)
    output_dir = Path(output_dir)
    if evidence_path.resolve() == output_path.resolve():
        raise ValueError("sidecar output must not overwrite the source evidence JSONL")
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "packet_count": 0,
        "clip_count": 0,
        "centroid_frame_count": 0,
        "minimum_centroid_frames_per_clip": None,
        "maximum_centroid_frames_per_clip": 0,
    }

    def converted_rows() -> Iterable[dict[str, Any]]:
        source_rows = iter_jsonl(evidence_path)
        for source_index, packet in enumerate(source_rows):
            if source_index < start_index:
                continue
            if max_packets is not None and stats["packet_count"] >= max_packets:
                break
            evidence_id = str(packet.get("evidence_id") or "")
            if not evidence_id:
                raise ValueError(f"source row {source_index} has no evidence_id")
            clips = packet.get("clips")
            if not isinstance(clips, list) or len(clips) != 2:
                raise ValueError(f"{evidence_id}: expected exactly two clips")

            packet_pruning, diagnostics_path = _load_packet_diagnostics(packet)
            converted_clips = []
            for clip_index, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    raise ValueError(f"{evidence_id}: clip {clip_index} must be an object")
                clip_pruning = clip.get("temporal_pruning")
                side = str(
                    clip_pruning.get("side")
                    if isinstance(clip_pruning, dict)
                    else ""
                )
                if side not in {"left", "right"}:
                    side = "left" if clip_index == 0 else "right"
                converted = _frame_only_clip(
                    clip,
                    side=side,
                    evidence_id=evidence_id,
                    output_dir=output_dir,
                    evidence_parent=evidence_path.parent,
                    packet_pruning=packet_pruning,
                    diagnostics_parent=diagnostics_path.parent if diagnostics_path else None,
                )
                frame_count = len(converted["frames"])
                stats["clip_count"] += 1
                stats["centroid_frame_count"] += frame_count
                current_minimum = stats["minimum_centroid_frames_per_clip"]
                stats["minimum_centroid_frames_per_clip"] = (
                    frame_count if current_minimum is None else min(current_minimum, frame_count)
                )
                stats["maximum_centroid_frames_per_clip"] = max(
                    int(stats["maximum_centroid_frames_per_clip"]),
                    frame_count,
                )
                converted_clips.append(converted)

            output_packet = dict(packet)
            output_packet["clips"] = converted_clips
            output_packet["generator_media_mode"] = GENERATOR_MEDIA_MODE
            output_packet["candidate_type"] = (
                f"{packet.get('candidate_type') or 'clip_pruned_pair'}_centroid_frame_sidecar"
            )
            output_packet["centroid_frame_sidecar"] = {
                "version": 1,
                "source_evidence_path": str(evidence_path),
                "source_row_index": source_index,
                "generator_media_mode": GENERATOR_MEDIA_MODE,
                "generator_frame_count": sum(len(clip["frames"]) for clip in converted_clips),
                "per_user_frame_counts": {
                    str(clip.get("agent_name") or clip.get("agent_dir") or index): len(
                        clip["frames"]
                    )
                    for index, clip in enumerate(converted_clips)
                },
                "selection_rule": (
                    "one CLIP medoid per cluster with content retained after pruning and "
                    "duration protection"
                ),
                "generator_media": "ordered centroid images only; no MP4",
                "judge_media": "full original synchronized videos",
            }
            output_packet["requirement"] = (
                "Centroid-frame sidecar experiment. The generator receives only the retained "
                "CLIP cluster medoids, ordered by original timestamp within each required user. "
                "It must not infer motion or events between the supplied still images. Visual "
                "judges and answerability checks continue to use the full original videos."
            )
            stats["packet_count"] += 1
            yield output_packet

    temporary_output = output_path.with_name(f"{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        written_count = write_jsonl(temporary_output, converted_rows())
        if written_count == 0:
            raise ValueError(
                f"no source packets were selected from {evidence_path} at index {start_index}"
            )
        if expected_packet_count is not None and written_count != expected_packet_count:
            raise ValueError(
                f"expected {expected_packet_count} sidecar packets, converted {written_count}"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)

    summary = {
        **stats,
        "source_evidence_path": str(evidence_path),
        "output_evidence_path": str(output_path),
        "output_dir": str(output_dir),
        "source_start_index": start_index,
        "requested_max_packets": max_packets,
        "expected_packet_count": expected_packet_count,
        "generator_media_mode": GENERATOR_MEDIA_MODE,
        "video_materialization": False,
        "media_routing": {
            "generator": "clips[*].frames (retained CLIP centroid images)",
            "judges_and_answerability": "clips[*].full_local_video (full original MP4)",
        },
    }
    write_json(output_dir / "centroid_frame_sidecar_summary.json", summary)
    return summary


def verify_centroid_frame_generation(
    *,
    evidence_path: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    expected_packet_count: int | None = None,
    expected_judge_media_role: str = "full",
) -> dict[str, Any]:
    """Verify that prompts honored image-only generation and full-video judging."""

    evidence = list(iter_jsonl(evidence_path))
    prompts = list(iter_jsonl(prompts_path))
    if expected_packet_count is not None and len(evidence) != expected_packet_count:
        raise ValueError(
            f"expected {expected_packet_count} sidecar evidence packets, found {len(evidence)}"
        )

    expected_images: dict[str, list[str]] = {}
    for packet in evidence:
        evidence_id = str(packet.get("evidence_id") or "")
        clips = packet.get("clips")
        if (
            packet.get("generator_media_mode") != GENERATOR_MEDIA_MODE
            or not isinstance(clips, list)
            or len(clips) != 2
        ):
            raise ValueError(f"{evidence_id}: invalid centroid-frame evidence contract")
        paths = [
            str(frame.get("path"))
            for clip in clips
            for frame in clip.get("frames", [])
            if isinstance(frame, dict)
        ]
        if not paths or any(not Path(path).is_file() for path in paths):
            raise FileNotFoundError(f"{evidence_id}: generator centroid frames are incomplete")
        expected_images[evidence_id] = paths

    generation_rows = [row for row in prompts if row.get("stage") == "generation"]
    visual_judge_rows = [
        row
        for row in prompts
        if row.get("stage") in {"evidence_groundedness_judge", "answerability"}
    ]
    observed_generation_ids = set()
    for row in generation_rows:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id not in expected_images:
            continue
        observed_generation_ids.add(evidence_id)
        if list(row.get("image_paths") or []) != expected_images[evidence_id]:
            raise ValueError(f"{evidence_id}: generation prompt did not use all centroid frames")
        if row.get("video_paths"):
            raise ValueError(f"{evidence_id}: generation prompt unexpectedly received an MP4")
    missing_generation = sorted(set(expected_images) - observed_generation_ids)
    if missing_generation:
        raise ValueError(
            "sidecar packets missing generation prompts: " + ", ".join(missing_generation[:5])
        )

    for row in visual_judge_rows:
        evidence_id = str(row.get("evidence_id") or "")
        if evidence_id not in expected_images:
            continue
        if row.get("media_role") != expected_judge_media_role:
            raise ValueError(
                f"{evidence_id}: visual judge media role is {row.get('media_role')!r}, "
                f"expected {expected_judge_media_role!r}"
            )
        if row.get("image_paths"):
            raise ValueError(f"{evidence_id}: visual judge unexpectedly received centroid images")
        if not row.get("video_paths"):
            raise ValueError(f"{evidence_id}: visual judge did not receive full video media")

    report = {
        "verified": True,
        "evidence_packet_count": len(evidence),
        "generation_prompt_count": len(generation_rows),
        "visual_judge_prompt_count": len(visual_judge_rows),
        "generator_media_mode": GENERATOR_MEDIA_MODE,
        "generator_image_only": True,
        "generator_video_count": 0,
        "judge_media_role": expected_judge_media_role,
        "judge_full_video_only": True,
    }
    write_json(output_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or verify the retained-centroid-frame generator sidecar"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Convert existing CLIP-pruned evidence to frame-only generator evidence",
    )
    prepare.add_argument("--evidence", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--start-index", type=int, default=0)
    prepare.add_argument("--max-packets", type=int)
    prepare.add_argument("--expected-packet-count", type=int)

    verify = subparsers.add_parser(
        "verify-generation",
        help="Verify generator image routing and full-video judge routing",
    )
    verify.add_argument("--evidence", required=True)
    verify.add_argument("--prompts", required=True)
    verify.add_argument("--output", required=True)
    verify.add_argument("--expected-packet-count", type=int)
    verify.add_argument("--expected-judge-media-role", default="full")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        summary = prepare_centroid_frame_evidence(
            evidence_path=args.evidence,
            output_path=args.output,
            output_dir=args.output_dir,
            start_index=args.start_index,
            max_packets=args.max_packets,
            expected_packet_count=args.expected_packet_count,
        )
        print(
            f"wrote {summary['packet_count']} centroid-frame sidecar packets with "
            f"{summary['centroid_frame_count']} generator images to {args.output}"
        )
        return 0
    if args.command == "verify-generation":
        report = verify_centroid_frame_generation(
            evidence_path=args.evidence,
            prompts_path=args.prompts,
            output_path=args.output,
            expected_packet_count=args.expected_packet_count,
            expected_judge_media_role=args.expected_judge_media_role,
        )
        print(
            f"verified {report['evidence_packet_count']} centroid-frame generator packets "
            f"against {report['generation_prompt_count']} generation prompts"
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
