from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


REQUIRED_FIELDS = frozenset(
    {"packet_json", "evidence_id", "question_type", "generation_mode"}
)
REQUIRED_QUESTION_TYPES = frozenset({"commonality", "difference"})


def audit_dataset_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    heldout_evidence_ids: set[str],
    heldout_rows: Iterable[Mapping[str, Any]] | None = None,
    min_evidence_ids: int = 8,
    min_heldout_evidence_ids: int = 2,
) -> dict[str, Any]:
    materialized = list(rows)
    if not materialized:
        raise ValueError("dataset must contain at least one row")
    evidence_counts: Counter[str] = Counter()
    question_type_counts: Counter[str] = Counter()
    generation_mode_counts: Counter[str] = Counter()
    for index, row in enumerate(materialized):
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            raise ValueError(f"row {index} missing fields: {', '.join(sorted(missing))}")
        evidence_id = str(row["evidence_id"]).strip()
        question_type = str(row["question_type"]).strip()
        generation_mode = str(row["generation_mode"]).strip()
        if not evidence_id or not question_type or not generation_mode:
            raise ValueError(f"row {index} contains an empty required field")
        packet = row["packet_json"]
        packet = json.loads(packet) if isinstance(packet, str) else packet
        if not isinstance(packet, Mapping):
            raise ValueError(f"row {index} packet_json must decode to an object")
        packet_evidence_id = str(packet.get("evidence_id", evidence_id)).strip()
        if packet_evidence_id != evidence_id:
            raise ValueError(f"row {index} packet_json evidence_id mismatch")
        evidence_counts[evidence_id] += 1
        question_type_counts[question_type] += 1
        generation_mode_counts[generation_mode] += 1

    if len(evidence_counts) < min_evidence_ids:
        raise ValueError(
            f"dataset needs at least {min_evidence_ids} distinct evidence_id; "
            f"found {len(evidence_counts)}"
        )
    heldout = {str(item).strip() for item in heldout_evidence_ids if str(item).strip()}
    if len(heldout) < min_heldout_evidence_ids:
        raise ValueError(
            f"need at least {min_heldout_evidence_ids} heldout evidence_id; found {len(heldout)}"
        )
    overlap = sorted(set(evidence_counts) & heldout)
    if overlap:
        raise ValueError(f"train/heldout evidence_id overlap: {', '.join(overlap)}")
    missing_types = REQUIRED_QUESTION_TYPES - set(question_type_counts)
    if missing_types:
        raise ValueError(
            f"dataset missing required question_type: {', '.join(sorted(missing_types))}"
        )
    if heldout_rows is not None:
        heldout_materialized = list(heldout_rows)
        actual_heldout_ids: set[str] = set()
        heldout_question_types: set[str] = set()
        for index, row in enumerate(heldout_materialized):
            missing = REQUIRED_FIELDS - set(row)
            if missing:
                raise ValueError(
                    f"heldout row {index} missing fields: {', '.join(sorted(missing))}"
                )
            evidence_id = str(row["evidence_id"]).strip()
            packet = row["packet_json"]
            packet = json.loads(packet) if isinstance(packet, str) else packet
            if not isinstance(packet, Mapping):
                raise ValueError(f"heldout row {index} packet_json must decode to an object")
            if str(packet.get("evidence_id", evidence_id)).strip() != evidence_id:
                raise ValueError(f"heldout row {index} packet_json evidence_id mismatch")
            actual_heldout_ids.add(evidence_id)
            heldout_question_types.add(str(row["question_type"]).strip())
        if actual_heldout_ids != heldout:
            raise ValueError(
                "heldout dataset IDs do not match declared heldout evidence IDs"
            )
        missing_heldout_types = REQUIRED_QUESTION_TYPES - heldout_question_types
        if missing_heldout_types:
            raise ValueError(
                "heldout dataset missing required question_type: "
                + ", ".join(sorted(missing_heldout_types))
            )
    return {
        "passed": True,
        "row_count": len(materialized),
        "distinct_evidence_id_count": len(evidence_counts),
        "heldout_evidence_id_count": len(heldout),
        "question_type_counts": dict(sorted(question_type_counts.items())),
        "generation_mode_counts": dict(sorted(generation_mode_counts.items())),
        "per_evidence_group_counts": dict(sorted(evidence_counts.items())),
        "heldout_evidence_ids": sorted(heldout),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit cross-view GRPO dataset coverage.")
    parser.add_argument("--dataset", type=Path, required=True)
    heldout_group = parser.add_mutually_exclusive_group(required=True)
    heldout_group.add_argument("--heldout-evidence-ids", type=Path)
    heldout_group.add_argument("--split-manifest", type=Path)
    parser.add_argument("--heldout-dataset", type=Path)
    parser.add_argument("--min-evidence-ids", type=int, default=8)
    parser.add_argument("--min-heldout-evidence-ids", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.split_manifest:
        split = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        heldout_ids = {
            str(item) for item in split.get("eval_evidence_ids", [])
        }
    else:
        heldout_ids = {
            line.strip()
            for line in args.heldout_evidence_ids.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    result = audit_dataset_rows(
        _read_jsonl(args.dataset),
        heldout_evidence_ids=heldout_ids,
        heldout_rows=_read_jsonl(args.heldout_dataset) if args.heldout_dataset else None,
        min_evidence_ids=args.min_evidence_ids,
        min_heldout_evidence_ids=args.min_heldout_evidence_ids,
    )
    if args.split_manifest:
        declared_train = {str(item) for item in split.get("train_evidence_ids", [])}
        actual_train = set(result["per_evidence_group_counts"])
        if declared_train != actual_train:
            raise ValueError("training dataset IDs do not match split manifest")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
