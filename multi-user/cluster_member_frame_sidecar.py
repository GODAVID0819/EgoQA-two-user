"""Sidecar that sends every retained CLIP-sampled cluster member to the generator.

This is an additive alternative to both the production pruned-MP4 pipeline and
the centroid-only sidecar. It reads the pruning diagnostics already written by
``paired_evidence_pruning``, copies the sampled frames that remain after
cluster pruning and duration protection, and routes them to the generator as
an ordered image sequence. Full original videos remain routed to visual judges
and answerability checks.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from .io_utils import iter_jsonl, stable_id, write_json, write_jsonl


GENERATOR_MEDIA_MODE = "retained_cluster_frames_only"


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


def _load_packet_diagnostics(
    packet: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], Path | None]:
    pruning = packet.get("paired_video_pruning")
    diagnostics_value = pruning.get("diagnostics_path") if isinstance(pruning, dict) else None
    if not diagnostics_value:
        # The older 30-second group-relative pruner stores cluster decisions
        # directly on each clip and leaves sampled images beside the medoids.
        # It predates packet-level diagnostics, so recover those images below.
        return {}, {}, None
    diagnostics_path = Path(str(diagnostics_value))
    if not diagnostics_path.is_file():
        return {}, {}, None
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{packet.get('evidence_id')}: pruning diagnostics must be an object")
    temporal = payload.get("temporal_pruning")
    sampled = payload.get("sampled_frames")
    if not isinstance(temporal, dict):
        raise ValueError(f"{packet.get('evidence_id')}: diagnostics lack temporal pruning")
    if not isinstance(sampled, dict):
        raise ValueError(f"{packet.get('evidence_id')}: diagnostics lack sampled frames")
    sampled_by_side: dict[str, list[dict[str, Any]]] = {}
    for side in ("left", "right"):
        rows = sampled.get(side)
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"{packet.get('evidence_id')}: diagnostics lack sampled {side} frames"
            )
        if not all(isinstance(row, dict) for row in rows):
            raise ValueError(
                f"{packet.get('evidence_id')}: sampled {side} frames must be objects"
            )
        sampled_by_side[side] = rows
    return temporal, sampled_by_side, diagnostics_path


def _cluster_decisions(
    clip: dict[str, Any],
    *,
    side: str,
    packet_pruning: dict[str, Any],
) -> list[dict[str, Any]]:
    temporal = clip.get("temporal_pruning")
    temporal = temporal if isinstance(temporal, dict) else {}
    decisions = temporal.get("cluster_decisions")
    if not isinstance(decisions, list) or not decisions:
        decisions = packet_pruning.get(f"{side}_cluster_decisions")
    if not isinstance(decisions, list) or not decisions:
        raise ValueError(
            f"{clip.get('clip_id') or clip.get('agent_dir') or side}: "
            "cluster-member decisions are unavailable"
        )
    return [row for row in decisions if isinstance(row, dict)]


_SAMPLED_FRAME_NAME = re.compile(
    r"^frame_(?P<index>\d+)_(?P<timestamp>-?\d+(?:\.\d+)?)s\.[^.]+$"
)


def _recover_sampled_frames_from_cluster_paths(
    clip: dict[str, Any],
    *,
    side: str,
    packet_pruning: dict[str, Any],
    evidence_parent: Path,
) -> list[dict[str, Any]]:
    """Recover the legacy sampled-frame list from medoid sibling files."""

    decisions = _cluster_decisions(
        clip,
        side=side,
        packet_pruning=packet_pruning,
    )
    required_indices = {
        int(value)
        for decision in decisions
        for value in decision.get("member_indices", [])
    }
    if not required_indices:
        raise ValueError(
            f"{clip.get('clip_id') or side}: cluster decisions have no member indices"
        )

    candidate_directories: list[Path] = []
    for decision in decisions:
        source_value = decision.get("source_path") or decision.get("path")
        if not source_value:
            continue
        source = Path(str(source_value))
        candidates = [source]
        if not source.is_absolute():
            candidates.append(evidence_parent / source)
        existing = next((candidate for candidate in candidates if candidate.is_file()), None)
        if existing is not None and existing.parent not in candidate_directories:
            candidate_directories.append(existing.parent)

    frames_by_index: dict[int, dict[str, Any]] = {}
    for directory in candidate_directories:
        for path in directory.glob("frame_*_*s.*"):
            match = _SAMPLED_FRAME_NAME.match(path.name)
            if not match or not path.is_file():
                continue
            frame_index = int(match.group("index"))
            frames_by_index.setdefault(
                frame_index,
                {
                    "timestamp_seconds": float(match.group("timestamp")),
                    "path": str(path),
                },
            )

    missing = sorted(required_indices - set(frames_by_index))
    if missing:
        searched = ", ".join(str(path) for path in candidate_directories[:3])
        raise FileNotFoundError(
            f"{clip.get('clip_id') or side}: legacy pruning packet has no diagnostics and "
            f"sampled member files {missing[:8]} were not found beside its medoids; "
            f"searched={searched or '<no existing medoid paths>'}"
        )

    maximum_index = max(required_indices)
    return [
        frames_by_index.get(
            index,
            {
                "timestamp_seconds": float(index),
                "path": "",
            },
        )
        for index in range(maximum_index + 1)
    ]


def _retained_member_rows(
    clip: dict[str, Any],
    *,
    side: str,
    packet_pruning: dict[str, Any],
    sampled_frames: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    temporal = clip.get("temporal_pruning")
    temporal = temporal if isinstance(temporal, dict) else {}
    decisions = _cluster_decisions(
        clip,
        side=side,
        packet_pruning=packet_pruning,
    )

    restored_values = temporal.get("restored_frame_indices")
    if not isinstance(restored_values, list):
        restored_values = packet_pruning.get(f"{side}_restored_frame_indices", [])
    restored_indices = {int(value) for value in restored_values or []}

    selected: list[dict[str, Any]] = []
    seen_frame_indices: set[int] = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        cluster_index = int(decision.get("cluster_index", -1))
        member_indices = sorted(
            {int(value) for value in decision.get("member_indices", [])},
            key=lambda index: (
                float(sampled_frames[index].get("timestamp_seconds") or 0.0)
                if 0 <= index < len(sampled_frames)
                else float("inf"),
                index,
            ),
        )
        if not member_indices:
            raise ValueError(
                f"{clip.get('clip_id') or side}: cluster {cluster_index} has no member indices"
            )
        invalid = [
            index for index in member_indices if index < 0 or index >= len(sampled_frames)
        ]
        if invalid:
            raise IndexError(
                f"{clip.get('clip_id') or side}: cluster {cluster_index} references "
                f"invalid sampled frame indices {invalid[:5]}"
            )

        kept_cluster = str(decision.get("status") or "") == "kept"
        retained_indices = (
            member_indices
            if kept_cluster
            else [index for index in member_indices if index in restored_indices]
        )
        retention_reason = (
            "cluster_kept_by_pruning"
            if kept_cluster
            else "duration_protection_restored_member"
        )
        for member_order, frame_index in enumerate(member_indices, 1):
            if frame_index not in retained_indices or frame_index in seen_frame_indices:
                continue
            sampled = sampled_frames[frame_index]
            selected.append(
                {
                    **sampled,
                    "cluster_index": cluster_index,
                    "sampled_frame_index": frame_index,
                    "cluster_member_order": member_order,
                    "cluster_member_count": len(member_indices),
                    "is_cluster_medoid": frame_index
                    == int(decision.get("frame_index", -1)),
                    "retention_reason": retention_reason,
                }
            )
            seen_frame_indices.add(frame_index)

    selected.sort(
        key=lambda row: (
            float(row.get("timestamp_seconds") or 0.0),
            int(row.get("sampled_frame_index") or 0),
        )
    )
    if not selected:
        raise ValueError(
            f"{clip.get('clip_id') or clip.get('agent_dir') or side}: "
            "pruning left no sampled cluster-member frames for the generator"
        )
    return selected


def _copy_member_frames(
    frames: list[dict[str, Any]],
    *,
    destination_dir: Path,
    evidence_parent: Path,
    diagnostics_parent: Path,
) -> list[dict[str, Any]]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for order, frame in enumerate(frames, 1):
        source_value = frame.get("path")
        candidates = [Path(str(source_value))] if source_value else []
        if source_value and not Path(str(source_value)).is_absolute():
            candidates.extend(
                [
                    evidence_parent / str(source_value),
                    diagnostics_parent / str(source_value),
                ]
            )
        source = next((candidate for candidate in candidates if candidate.is_file()), None)
        if source is None:
            raise FileNotFoundError(
                f"retained cluster-member frame is missing: {source_value!r}"
            )

        cluster_index = int(frame["cluster_index"])
        sampled_index = int(frame["sampled_frame_index"])
        timestamp = float(frame.get("timestamp_seconds") or 0.0)
        timestamp_ms = round(timestamp * 1000.0)
        suffix = source.suffix.lower() or ".jpg"
        destination = destination_dir / (
            f"{order:04d}_cluster_{cluster_index:03d}_member_{sampled_index:04d}_"
            f"t{timestamp_ms:09d}ms{suffix}"
        )
        shutil.copy2(source, destination)
        copied.append(
            {
                "path": str(destination),
                "source_path": str(source),
                "frame_role": "retained_clip_cluster_member",
                "input_order_within_user": order,
                "cluster_index": cluster_index,
                "sampled_frame_index": sampled_index,
                "cluster_member_order": frame.get("cluster_member_order"),
                "cluster_member_count": frame.get("cluster_member_count"),
                "is_cluster_medoid": bool(frame.get("is_cluster_medoid")),
                "timestamp_seconds": frame.get("timestamp_seconds"),
                "retention_reason": frame.get("retention_reason"),
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
    sampled_frames: list[dict[str, Any]],
    diagnostics_parent: Path,
) -> dict[str, Any]:
    if clip.get("generator_media_mode") == GENERATOR_MEDIA_MODE:
        raise ValueError(f"{evidence_id}: input is already a cluster-member sidecar")

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
            f"{evidence_id}: {side} clip has no full original video for judge routing"
        )

    selected = _retained_member_rows(
        clip,
        side=side,
        packet_pruning=packet_pruning,
        sampled_frames=sampled_frames,
    )
    packet_key = _safe_part(stable_id("cluster_member_frame_sidecar", evidence_id))
    agent = _safe_part(clip.get("agent_dir") or clip.get("agent_name") or side)
    frames = _copy_member_frames(
        selected,
        destination_dir=output_dir
        / "cluster_member_frames"
        / packet_key
        / f"{side}_{agent}",
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
    temporal["retained_cluster_member_frames"] = frames
    temporal["retained_cluster_member_frame_count"] = len(frames)
    temporal["generator_representation"] = "retained_cluster_member_frames"
    current["temporal_pruning"] = temporal

    benchmark_media = current.get("benchmark_media")
    benchmark_media = dict(benchmark_media) if isinstance(benchmark_media, dict) else {}
    benchmark_media.pop("generator_video", None)
    benchmark_media["generator_frames"] = [frame["path"] for frame in frames]
    benchmark_media["judge_video"] = str(full_video)
    benchmark_media["answerability_video"] = str(full_video)
    current["benchmark_media"] = benchmark_media
    return current


def prepare_cluster_member_frame_evidence(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    start_index: int = 0,
    max_packets: int | None = None,
    expected_packet_count: int | None = None,
) -> dict[str, Any]:
    """Create frame-only generator evidence from all retained sampled members."""

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

    stats: dict[str, Any] = {
        "packet_count": 0,
        "clip_count": 0,
        "cluster_member_frame_count": 0,
        "minimum_frames_per_clip": None,
        "maximum_frames_per_clip": 0,
        "diagnostics_backed_clip_count": 0,
        "legacy_recovered_clip_count": 0,
    }

    def converted_rows() -> Iterable[dict[str, Any]]:
        for source_index, packet in enumerate(iter_jsonl(evidence_path)):
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

            packet_pruning, sampled_by_side, diagnostics_path = _load_packet_diagnostics(
                packet
            )
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
                sampled_frames = sampled_by_side.get(side)
                if sampled_frames:
                    stats["diagnostics_backed_clip_count"] += 1
                else:
                    sampled_frames = _recover_sampled_frames_from_cluster_paths(
                        clip,
                        side=side,
                        packet_pruning=packet_pruning,
                        evidence_parent=evidence_path.parent,
                    )
                    stats["legacy_recovered_clip_count"] += 1
                converted = _frame_only_clip(
                    clip,
                    side=side,
                    evidence_id=evidence_id,
                    output_dir=output_dir,
                    evidence_parent=evidence_path.parent,
                    packet_pruning=packet_pruning,
                    sampled_frames=sampled_frames,
                    diagnostics_parent=(
                        diagnostics_path.parent
                        if diagnostics_path is not None
                        else evidence_path.parent
                    ),
                )
                frame_count = len(converted["frames"])
                stats["clip_count"] += 1
                stats["cluster_member_frame_count"] += frame_count
                current_minimum = stats["minimum_frames_per_clip"]
                stats["minimum_frames_per_clip"] = (
                    frame_count
                    if current_minimum is None
                    else min(current_minimum, frame_count)
                )
                stats["maximum_frames_per_clip"] = max(
                    int(stats["maximum_frames_per_clip"]),
                    frame_count,
                )
                converted_clips.append(converted)

            output_packet = dict(packet)
            output_packet["clips"] = converted_clips
            output_packet["generator_media_mode"] = GENERATOR_MEDIA_MODE
            output_packet["candidate_type"] = (
                f"{packet.get('candidate_type') or 'clip_pruned_pair'}_"
                "cluster_member_frame_sidecar"
            )
            output_packet["cluster_member_frame_sidecar"] = {
                "version": 1,
                "source_evidence_path": str(evidence_path),
                "source_row_index": source_index,
                "generator_media_mode": GENERATOR_MEDIA_MODE,
                "generator_frame_count": sum(
                    len(clip["frames"]) for clip in converted_clips
                ),
                "per_user_frame_counts": {
                    str(
                        clip.get("agent_name") or clip.get("agent_dir") or index
                    ): len(clip["frames"])
                    for index, clip in enumerate(converted_clips)
                },
                "selection_rule": (
                    "all CLIP-sampled members of clusters kept by pruning, plus only "
                    "explicitly restored members of otherwise pruned clusters"
                ),
                "generator_media": "ordered retained cluster-member images; no MP4",
                "judge_media": "full original synchronized videos",
            }
            output_packet["requirement"] = (
                "Cluster-member-frame sidecar experiment. The generator receives every "
                "retained CLIP-sampled cluster member, ordered by original timestamp within "
                "each required user. These are denser still-image sequences, not continuous "
                "video; it must not infer unseen motion between supplied images. Visual judges "
                "and answerability checks continue to use the full original videos."
            )
            stats["packet_count"] += 1
            yield output_packet

    temporary_output = output_path.with_name(f"{output_path.name}.tmp")
    temporary_output.unlink(missing_ok=True)
    try:
        written_count = write_jsonl(temporary_output, converted_rows())
        if written_count == 0:
            raise ValueError(f"no source packets were selected from {evidence_path}")
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
            "generator": "clips[*].frames (retained CLIP-sampled cluster members)",
            "judges_and_answerability": "clips[*].full_local_video (full original MP4)",
        },
    }
    write_json(output_dir / "cluster_member_frame_sidecar_summary.json", summary)
    return summary


def verify_cluster_member_frame_generation(
    *,
    evidence_path: str | Path,
    prompts_path: str | Path,
    output_path: str | Path,
    expected_packet_count: int | None = None,
    expected_judge_media_role: str = "full",
) -> dict[str, Any]:
    """Verify image-only generation and full-video visual judging."""

    evidence = list(iter_jsonl(evidence_path))
    prompts = list(iter_jsonl(prompts_path))
    if expected_packet_count is not None and len(evidence) != expected_packet_count:
        raise ValueError(
            f"expected {expected_packet_count} sidecar packets, found {len(evidence)}"
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
            raise ValueError(f"{evidence_id}: invalid cluster-member evidence contract")
        paths = [
            str(frame.get("path"))
            for clip in clips
            for frame in clip.get("frames", [])
            if isinstance(frame, dict)
        ]
        if not paths or any(not Path(path).is_file() for path in paths):
            raise FileNotFoundError(
                f"{evidence_id}: generator cluster-member frames are incomplete"
            )
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
            raise ValueError(
                f"{evidence_id}: generation prompt did not use every retained member frame"
            )
        if row.get("video_paths"):
            raise ValueError(f"{evidence_id}: generator unexpectedly received an MP4")
    missing_generation = sorted(set(expected_images) - observed_generation_ids)
    if missing_generation:
        raise ValueError(
            "sidecar packets missing generation prompts: "
            + ", ".join(missing_generation[:5])
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
            raise ValueError(
                f"{evidence_id}: visual judge unexpectedly received member images"
            )
        if not row.get("video_paths"):
            raise ValueError(f"{evidence_id}: visual judge did not receive full videos")

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
        description="Prepare or verify the retained-cluster-member generator sidecar"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="Convert existing CLIP-pruned evidence to retained-member image evidence",
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
        summary = prepare_cluster_member_frame_evidence(
            evidence_path=args.evidence,
            output_path=args.output,
            output_dir=args.output_dir,
            start_index=args.start_index,
            max_packets=args.max_packets,
            expected_packet_count=args.expected_packet_count,
        )
        print(
            f"wrote {summary['packet_count']} cluster-member sidecar packets with "
            f"{summary['cluster_member_frame_count']} images to {args.output}"
        )
        return 0
    if args.command == "verify-generation":
        report = verify_cluster_member_frame_generation(
            evidence_path=args.evidence,
            prompts_path=args.prompts,
            output_path=args.output,
            expected_packet_count=args.expected_packet_count,
            expected_judge_media_role=args.expected_judge_media_role,
        )
        print(
            f"verified {report['evidence_packet_count']} cluster-member generator packets "
            f"against {report['generation_prompt_count']} generation prompts"
        )
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
