"""Prepare and verify a targeted repair collection for discarded reward packets.

The repair collection is intentionally separate from the two original collection
roots.  It reuses their frozen sampled-frame evidence and full original videos,
regenerates only discarded packets, and records enough source provenance to treat
the original retained candidates plus the repair candidates as one complete set.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPAIR_SCHEMA_VERSION = "reward_candidate_repair_v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one JSON object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _collection_groups(root: Path) -> list[Path]:
    index_path = root / "packet_groups.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing packet-group index: {index_path}")
    groups: list[Path] = []
    for row in _read_jsonl(index_path):
        relative_dir = str(row.get("relative_dir") or "")
        if not relative_dir:
            raise ValueError(f"{index_path}: group row is missing relative_dir")
        group = (root / relative_dir).resolve()
        if not group.is_dir():
            raise FileNotFoundError(f"missing packet-group directory: {group}")
        groups.append(group)
    if not groups:
        raise ValueError(f"{index_path}: packet-group index is empty")
    return groups


def _candidate_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _collection_groups(root):
        path = group / "candidate_details.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing candidate details: {path}")
        rows.extend(_read_jsonl(path))
    return rows


def _discarded_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in _collection_groups(root):
        path = group / "discarded_packets.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing discarded-packet log: {path}")
        rows.extend(_read_jsonl(path))
    return rows


def _validate_packet_media(packet: dict[str, Any]) -> None:
    evidence_id = str(packet.get("evidence_id") or "")
    clips = packet.get("clips")
    if not isinstance(clips, list) or len(clips) != 2:
        raise ValueError(f"{evidence_id}: expected exactly two clips")
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError(f"{evidence_id}: clip entry is not an object")
        frames = [Path(str(row.get("path") or "")) for row in clip.get("frames") or []]
        video = Path(str(clip.get("full_local_video") or ""))
        if not frames or any(not path.is_file() for path in frames):
            raise FileNotFoundError(f"{evidence_id}: sampled-frame evidence is incomplete")
        if not video.is_file():
            raise FileNotFoundError(f"{evidence_id}: full original video is missing: {video}")


def prepare_repair(
    *,
    source_dirs: list[Path],
    output_dir: Path,
    expected_source_packets: int,
    expected_discarded_packets: int,
) -> dict[str, Any]:
    source_dirs = [path.resolve() for path in source_dirs]
    if len(source_dirs) < 1 or len(source_dirs) != len(set(source_dirs)):
        raise ValueError("source collection directories must be non-empty and unique")
    output_dir = output_dir.resolve()
    evidence_by_id: dict[str, dict[str, Any]] = {}
    repair_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []

    for source in source_dirs:
        summary_path = source / "collection_summary.json"
        evidence_path = source / "evidence_manifest.jsonl"
        if not summary_path.is_file() or not evidence_path.is_file():
            raise FileNotFoundError(f"incomplete source collection root: {source}")
        summary = _read_json(summary_path)
        evidence_rows = _read_jsonl(evidence_path)
        if len(evidence_rows) != int(summary.get("selected_packet_count") or -1):
            raise ValueError(f"{source}: evidence count does not match collection summary")
        for packet in evidence_rows:
            evidence_id = str(packet.get("evidence_id") or "")
            if not evidence_id or evidence_id in evidence_by_id:
                raise ValueError(f"duplicate or empty source evidence ID: {evidence_id!r}")
            evidence_by_id[evidence_id] = packet

        discarded = _discarded_rows(source)
        if len(discarded) != int(summary.get("discarded_packet_count") or 0):
            raise ValueError(f"{source}: discarded record count does not match summary")
        for record in discarded:
            evidence_id = str(record.get("evidence_id") or "")
            if evidence_id not in evidence_by_id:
                raise ValueError(f"{source}: discarded evidence is absent from manifest: {evidence_id}")
            if record.get("retained") is not False:
                raise ValueError(f"{source}: discarded record is unexpectedly retained: {evidence_id}")
            packet = evidence_by_id[evidence_id]
            _validate_packet_media(packet)
            repair_rows.append(packet)

        source_summaries.append(
            {
                "source_dir": str(source),
                "config_sha256": summary.get("config_sha256"),
                "selected_packet_count": summary.get("selected_packet_count"),
                "retained_packet_count": summary.get("retained_packet_count"),
                "discarded_packet_count": summary.get("discarded_packet_count"),
                "candidate_count": summary.get("candidate_count"),
            }
        )

    if len(evidence_by_id) != expected_source_packets:
        raise ValueError(
            f"expected {expected_source_packets} source packets, found {len(evidence_by_id)}"
        )
    repair_ids = [str(row["evidence_id"]) for row in repair_rows]
    if len(repair_ids) != expected_discarded_packets or len(repair_ids) != len(set(repair_ids)):
        raise ValueError(
            f"expected {expected_discarded_packets} unique discarded packets, found {len(repair_ids)}"
        )
    repair_rows.sort(
        key=lambda row: (
            str(row.get("day") or ""),
            str(row.get("time_token") or ""),
            str(row.get("evidence_id") or ""),
        )
    )
    repair_ids = [str(row["evidence_id"]) for row in repair_rows]
    evidence_output = output_dir / "repair_evidence.jsonl"
    plan_output = output_dir / "repair_plan.json"
    _write_jsonl(evidence_output, repair_rows)
    plan = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "regenerate only discarded packets from their frozen sampled-frame evidence; "
            "leave both source collections unchanged"
        ),
        "source_collections": source_summaries,
        "source_packet_count": len(evidence_by_id),
        "repair_packet_count": len(repair_rows),
        "repair_evidence_ids": repair_ids,
        "repair_evidence_path": str(evidence_output),
        "repair_collection_dir": str(output_dir / "collection"),
    }
    _write_json(plan_output, plan)
    return plan


def verify_repair(*, repair_dir: Path) -> dict[str, Any]:
    repair_dir = repair_dir.resolve()
    plan = _read_json(repair_dir / "repair_plan.json")
    repair_collection = Path(str(plan["repair_collection_dir"])).resolve()
    repair_summary = _read_json(repair_collection / "collection_summary.json")
    repair_ids = set(str(value) for value in plan.get("repair_evidence_ids") or [])
    repair_candidates = _candidate_rows(repair_collection)
    repaired_counts = Counter(str(row.get("evidence_id") or "") for row in repair_candidates)
    if set(repaired_counts) != repair_ids or any(count != 6 for count in repaired_counts.values()):
        raise ValueError("repair collection does not contain exactly six candidates per repair packet")
    if repair_summary.get("quota_complete") is not True:
        raise ValueError("repair collection quota is incomplete")

    source_ids: set[str] = set()
    retained_ids: set[str] = set()
    source_candidate_count = 0
    discarded_ids: set[str] = set()
    for source_row in plan.get("source_collections") or []:
        source = Path(str(source_row["source_dir"])).resolve()
        evidence_rows = _read_jsonl(source / "evidence_manifest.jsonl")
        current_ids = {str(row.get("evidence_id") or "") for row in evidence_rows}
        if not current_ids or source_ids.intersection(current_ids):
            raise ValueError("source collections contain empty or duplicate evidence IDs")
        source_ids.update(current_ids)
        discarded_ids.update(str(row.get("evidence_id") or "") for row in _discarded_rows(source))
        candidates = _candidate_rows(source)
        counts = Counter(str(row.get("evidence_id") or "") for row in candidates)
        if any(count != 6 for count in counts.values()):
            raise ValueError(f"{source}: retained packet does not contain exactly six candidates")
        retained_ids.update(counts)
        source_candidate_count += len(candidates)

    if repair_ids != discarded_ids:
        raise ValueError("repair evidence IDs do not exactly match the source discarded IDs")
    if retained_ids.intersection(repair_ids):
        raise ValueError("a repaired evidence ID was already retained in a source collection")
    combined_retained_ids = retained_ids | repair_ids
    if combined_retained_ids != source_ids:
        raise ValueError("source retained plus repaired packets do not cover every selected packet")
    combined_candidate_count = source_candidate_count + len(repair_candidates)
    expected_candidate_count = len(source_ids) * 6
    if combined_candidate_count != expected_candidate_count:
        raise ValueError(
            f"expected {expected_candidate_count} combined candidates, found {combined_candidate_count}"
        )

    summary = {
        "schema_version": REPAIR_SCHEMA_VERSION,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "repair_complete": True,
        "source_packet_count": len(source_ids),
        "source_retained_packet_count": len(retained_ids),
        "repaired_packet_count": len(repair_ids),
        "combined_retained_packet_count": len(combined_retained_ids),
        "source_candidate_count": source_candidate_count,
        "repair_candidate_count": len(repair_candidates),
        "combined_candidate_count": combined_candidate_count,
        "expected_candidate_count": expected_candidate_count,
        "repair_evidence_ids": sorted(repair_ids),
        "source_collections": [row["source_dir"] for row in plan["source_collections"]],
        "repair_collection_dir": str(repair_collection),
        "integration_policy": (
            f"use the {len(retained_ids)} retained source packets and replace the "
            f"{len(repair_ids)} source discarded packets with the corresponding repair "
            "collection packets by evidence_id"
        ),
    }
    _write_json(repair_dir / "repair_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="extract discarded packets for repair")
    prepare.add_argument("--source-dir", action="append", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--expected-source-packets", type=int, default=100)
    prepare.add_argument("--expected-discarded-packets", type=int, default=4)

    verify = subparsers.add_parser("verify", help="verify repaired plus retained candidate totals")
    verify.add_argument("--repair-dir", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_repair(
            source_dirs=[Path(value) for value in args.source_dir],
            output_dir=Path(args.output_dir),
            expected_source_packets=args.expected_source_packets,
            expected_discarded_packets=args.expected_discarded_packets,
        )
        print(
            "reward_repair_prepared "
            f"source_packets={result['source_packet_count']} "
            f"repair_packets={result['repair_packet_count']} "
            f"evidence={result['repair_evidence_path']}"
        )
        return 0
    result = verify_repair(repair_dir=Path(args.repair_dir))
    print(
        "reward_repair_verified "
        f"packets={result['combined_retained_packet_count']} "
        f"candidates={result['combined_candidate_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
