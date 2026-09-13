"""Rank evidence packets by CLIP-based cross-user exclusiveness.

This is an optional retrieval stage between ``prepare_evidence`` and
``generate_video_qa_loop``. It does not change question generation; it only
annotates and ranks packets whose two POVs appear visually/semantically far
apart under CLIP frame embeddings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .clip_gap_demo import (
    DEFAULT_CLIP_MODEL,
    ImageEncoder,
    TransformersClipEncoder,
    _cluster_user_frames,
    mine_anchors_and_gaps,
    packet_frames,
    write_contact_sheet,
)
from .io_utils import iter_jsonl, stable_id, write_json, write_jsonl


def _matrix_values(result: dict[str, Any]) -> list[float]:
    return [
        float(value)
        for row in result.get("similarity_matrix", [])
        for value in row
    ]


def summarize_exclusiveness(result: dict[str, Any]) -> dict[str, Any]:
    """Return compact ranking metrics from a CLIP gap result."""

    matrix_values = _matrix_values(result)
    mean_similarity = sum(matrix_values) / len(matrix_values) if matrix_values else None
    min_similarity = min(matrix_values) if matrix_values else None
    left_gaps = result.get("left_evidence_gaps", [])
    right_gaps = result.get("right_evidence_gaps", [])
    left_novelties = [float(row["novelty"]) for row in left_gaps]
    right_novelties = [float(row["novelty"]) for row in right_gaps]
    largest_novelty = max(left_novelties + right_novelties, default=None)
    cross_user_dissimilarity = 1.0 - mean_similarity if mean_similarity is not None else None
    score_parts = [
        value
        for value in [cross_user_dissimilarity, largest_novelty]
        if value is not None
    ]
    score = sum(score_parts) / len(score_parts) if score_parts else 0.0
    return {
        "score": round(score, 6),
        "cross_user_dissimilarity": (
            round(cross_user_dissimilarity, 6) if cross_user_dissimilarity is not None else None
        ),
        "mean_cross_user_similarity": (
            round(mean_similarity, 6) if mean_similarity is not None else None
        ),
        "minimum_cross_user_similarity": (
            round(min_similarity, 6) if min_similarity is not None else None
        ),
        "largest_user_novelty": (
            round(largest_novelty, 6) if largest_novelty is not None else None
        ),
        f"max_{result['left_user']}_novelty": (
            round(max(left_novelties), 6) if left_novelties else None
        ),
        f"max_{result['right_user']}_novelty": (
            round(max(right_novelties), 6) if right_novelties else None
        ),
        "anchor_count": len(result.get("anchors", [])),
    }


def analyze_clip_exclusiveness(
    packet: dict[str, Any],
    *,
    output_dir: str | Path,
    encoder: ImageEncoder,
    duration_seconds: float = 12.0,
    sample_interval_seconds: float = 1.5,
    start_seconds: float = 0.0,
    clusters_per_user: int = 4,
    anchor_threshold: float = 0.75,
    top_k: int = 3,
    ffmpeg_binary: str = "ffmpeg",
    resample_videos: bool = True,
) -> dict[str, Any]:
    """Compute CLIP exclusiveness metadata for one two-user packet."""

    packet_id = str(packet.get("evidence_id") or stable_id(packet.get("day"), packet.get("time_token")))
    packet_dir = Path(output_dir) / packet_id
    users = packet_frames(
        packet,
        packet_dir,
        duration_seconds=duration_seconds,
        sample_interval_seconds=sample_interval_seconds,
        start_seconds=start_seconds,
        ffmpeg_binary=ffmpeg_binary,
        resample_videos=resample_videos,
    )
    encoded_users = []
    for row in users:
        embeddings = encoder.encode([str(frame["path"]) for frame in row["frames"]])
        representatives, representative_embeddings, groups = _cluster_user_frames(
            row["frames"],
            embeddings,
            clusters_per_user,
        )
        encoded_users.append(
            {
                **row,
                "representatives": representatives,
                "representative_embeddings": representative_embeddings,
                "groups": groups,
            }
        )

    left, right = encoded_users
    mined = mine_anchors_and_gaps(
        left["representatives"],
        right["representatives"],
        left["representative_embeddings"],
        right["representative_embeddings"],
        anchor_threshold=anchor_threshold,
        top_k=top_k,
    )
    result = {
        "evidence_id": packet.get("evidence_id"),
        "day": packet.get("day"),
        "time_token": packet.get("time_token"),
        "model_id": encoder.model_id,
        "left_user": left["user"],
        "right_user": right["user"],
        "window": {
            "start_seconds": start_seconds,
            "duration_seconds": duration_seconds,
            "sample_interval_seconds": sample_interval_seconds,
        },
        "clustering": {
            "clusters_per_user": clusters_per_user,
            left["user"]: left["groups"],
            right["user"]: right["groups"],
        },
        "representative_frames": {
            left["user"]: left["representatives"],
            right["user"]: right["representatives"],
        },
        **mined,
    }
    result["metrics"] = summarize_exclusiveness(result)
    return result


def _exclusive_frames_by_user(result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        result["left_user"]: result.get("left_evidence_gaps", []),
        result["right_user"]: result.get("right_evidence_gaps", []),
    }


def annotate_packet(packet: dict[str, Any], result: dict[str, Any], *, rank: int | None = None) -> dict[str, Any]:
    annotated = dict(packet)
    metadata = {
        "rank": rank,
        "model_id": result.get("model_id"),
        "left_user": result.get("left_user"),
        "right_user": result.get("right_user"),
        "window": result.get("window"),
        "metrics": result.get("metrics", {}),
        "score": result.get("metrics", {}).get("score"),
        "representative_frames": result.get("representative_frames", {}),
        "exclusive_frames_by_user": _exclusive_frames_by_user(result),
        "anchors": result.get("anchors", []),
        "interpretation_warning": result.get("interpretation_warning"),
    }
    annotated["clip_exclusiveness"] = metadata
    return annotated


def mine_clip_exclusive_candidates(
    *,
    evidence_path: str | Path,
    output_path: str | Path,
    output_dir: str | Path,
    model_id: str = DEFAULT_CLIP_MODEL,
    duration_seconds: float = 12.0,
    sample_interval_seconds: float = 1.5,
    start_seconds: float = 0.0,
    clusters_per_user: int = 4,
    anchor_threshold: float = 0.75,
    top_k: int = 3,
    max_packets: int | None = None,
    min_score: float | None = None,
    contact_sheet_count: int = 5,
    ffmpeg_binary: str = "ffmpeg",
    resample_videos: bool = True,
    preserve_order: bool = False,
    encoder: ImageEncoder | None = None,
) -> list[dict[str, Any]]:
    """Write evidence packets ranked by CLIP exclusiveness."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    encoder = encoder or TransformersClipEncoder(model_id)
    ranked: list[tuple[dict[str, Any], dict[str, Any]]] = []
    skipped = []
    for index, packet in enumerate(iter_jsonl(evidence_path)):
        if max_packets is not None and index >= max_packets:
            break
        try:
            result = analyze_clip_exclusiveness(
                packet,
                output_dir=output_dir,
                encoder=encoder,
                duration_seconds=duration_seconds,
                sample_interval_seconds=sample_interval_seconds,
                start_seconds=start_seconds,
                clusters_per_user=clusters_per_user,
                anchor_threshold=anchor_threshold,
                top_k=top_k,
                ffmpeg_binary=ffmpeg_binary,
                resample_videos=resample_videos,
            )
        except Exception as exc:
            skipped.append(
                {
                    "index": index,
                    "evidence_id": packet.get("evidence_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        score = float(result.get("metrics", {}).get("score") or 0.0)
        if min_score is not None and score < min_score:
            continue
        ranked.append((packet, result))

    ranked_by_score = sorted(
        ranked,
        key=lambda item: (
            float(item[1].get("metrics", {}).get("score") or 0.0),
            str(item[0].get("evidence_id") or ""),
        ),
        reverse=True,
    )
    rank_metadata = {}
    for rank, (packet, result) in enumerate(ranked_by_score, 1):
        full_result_path = output_dir / f"rank_{rank:03d}_{packet.get('evidence_id')}_clip_exclusive.json"
        write_json(full_result_path, result)
        metadata = {"rank": rank, "result_path": str(full_result_path)}
        if rank <= contact_sheet_count:
            contact_sheet_path = output_dir / f"rank_{rank:03d}_{packet.get('evidence_id')}_contact_sheet.jpg"
            write_contact_sheet(result, contact_sheet_path)
            metadata["contact_sheet_path"] = str(contact_sheet_path)
        rank_metadata[id(result)] = metadata

    output_pairs = ranked if preserve_order else ranked_by_score
    annotated = []
    for packet, result in output_pairs:
        metadata = rank_metadata[id(result)]
        row = annotate_packet(packet, result, rank=metadata["rank"])
        row["clip_exclusiveness"]["result_path"] = metadata["result_path"]
        if metadata.get("contact_sheet_path"):
            row["clip_exclusiveness"]["contact_sheet_path"] = metadata["contact_sheet_path"]
        annotated.append(row)

    write_jsonl(output_path, annotated)
    write_json(
        output_dir / "clip_exclusive_summary.json",
        {
            "input": str(evidence_path),
            "output": str(output_path),
            "candidate_count": len(annotated),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "model_id": encoder.model_id,
            "parameters": {
                "duration_seconds": duration_seconds,
                "sample_interval_seconds": sample_interval_seconds,
                "start_seconds": start_seconds,
                "clusters_per_user": clusters_per_user,
                "anchor_threshold": anchor_threshold,
                "top_k": top_k,
                "max_packets": max_packets,
                "min_score": min_score,
                "resample_videos": resample_videos,
                "preserve_order": preserve_order,
            },
        },
    )
    return annotated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rank evidence packets by CLIP cross-user exclusiveness")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--duration-seconds", type=float, default=12.0)
    parser.add_argument("--sample-interval-seconds", type=float, default=1.5)
    parser.add_argument("--start-seconds", type=float, default=0.0)
    parser.add_argument("--clusters-per-user", type=int, default=4)
    parser.add_argument("--anchor-threshold", type=float, default=0.75)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-packets", type=int)
    parser.add_argument("--min-score", type=float)
    parser.add_argument("--contact-sheet-count", type=int, default=5)
    parser.add_argument("--ffmpeg-binary", default="ffmpeg")
    parser.add_argument("--use-existing-frames", action="store_true")
    parser.add_argument("--preserve-order", action="store_true")
    args = parser.parse_args(argv)
    rows = mine_clip_exclusive_candidates(
        evidence_path=args.evidence,
        output_path=args.output,
        output_dir=args.output_dir,
        model_id=args.model_id,
        duration_seconds=args.duration_seconds,
        sample_interval_seconds=args.sample_interval_seconds,
        start_seconds=args.start_seconds,
        clusters_per_user=args.clusters_per_user,
        anchor_threshold=args.anchor_threshold,
        top_k=args.top_k,
        max_packets=args.max_packets,
        min_score=args.min_score,
        contact_sheet_count=args.contact_sheet_count,
        ffmpeg_binary=args.ffmpeg_binary,
        resample_videos=not args.use_existing_frames,
        preserve_order=args.preserve_order,
    )
    print(f"wrote {len(rows)} CLIP-exclusive evidence packets to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
