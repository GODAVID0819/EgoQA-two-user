"""Preflight a score-only retained-frame GRPO JSONL before GPU use."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from ..baseline.gate3_dataset import (
    PRODUCTION_GENERATION_MODE,
    PRODUCTION_QUESTION_TYPE,
    SPLIT_SCHEMA_VERSION,
)
from .data import (
    PRODUCTION_PROMPT_CONTRACT,
    production_prompt_source_contract,
    validate_swift_row,
)
from .media import GENERATOR_MEDIA_MODE


SplitName = Literal["train", "eval", "test", "heldout"]
REQUIRED_FIELDS = {
    "messages",
    "images",
    "generator_image_paths",
    "reviewer_video_paths",
    "evidence_id",
    "packet_json",
    "question_type",
    "generation_mode",
    "prompt_contract",
    "prompt_sha256",
    "prompt_builder",
    "prompt_source_path",
    "prompt_source_sha256",
    "required_users",
    "image_order",
    "generator_frame_counts",
    "reviewer_video_order",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_ids(value: Any, field: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"split manifest {field} must be a list")
    ids = [str(item).strip() for item in value]
    if any(not item for item in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"split manifest {field} must contain unique non-empty IDs")
    return ids


def _validate_split_manifest(
    value: Any,
    *,
    require_current_prompt_source: bool = True,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("split manifest must be a JSON object")
    manifest = dict(value)
    if manifest.get("schema_version") != SPLIT_SCHEMA_VERSION:
        raise ValueError(
            "unsupported split manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )
    train_ids = _manifest_ids(manifest.get("train_evidence_ids"), "train_evidence_ids")
    eval_ids = _manifest_ids(manifest.get("eval_evidence_ids"), "eval_evidence_ids")
    test_ids = _manifest_ids(manifest.get("test_evidence_ids", []), "test_evidence_ids")
    counts = {
        "train": manifest.get("train_count"),
        "eval": manifest.get("eval_count"),
        "test": manifest.get("test_count", 0),
    }
    for name, count in counts.items():
        minimum = 0 if name == "test" else 1
        if isinstance(count, bool) or not isinstance(count, int) or count < minimum:
            qualifier = "non-negative" if name == "test" else "positive"
            raise ValueError(
                f"split manifest {name}_count must be a {qualifier} integer"
            )
    if any(
        counts[name] != len(ids)
        for name, ids in (("train", train_ids), ("eval", eval_ids), ("test", test_ids))
    ):
        raise ValueError("split manifest counts do not match their evidence ID lists")
    if manifest.get("question_type") != PRODUCTION_QUESTION_TYPE:
        raise ValueError("split manifest question_type must be 'neutral'")
    if manifest.get("generation_mode") != PRODUCTION_GENERATION_MODE:
        raise ValueError("split manifest generation_mode must be 'baseline'")
    if manifest.get("generator_media_mode") != GENERATOR_MEDIA_MODE:
        raise ValueError(
            "split manifest generator_media_mode must be "
            f"{GENERATOR_MEDIA_MODE!r}"
        )
    if manifest.get("prompt_contract") != PRODUCTION_PROMPT_CONTRACT:
        raise ValueError(
            f"split manifest prompt_contract must be {PRODUCTION_PROMPT_CONTRACT!r}"
        )
    if require_current_prompt_source:
        for field, expected in production_prompt_source_contract().items():
            if manifest.get(field) != expected:
                raise ValueError(
                    f"split manifest {field} does not match the current production prompt"
                )
    overlap = sorted(
        (set(train_ids) & set(eval_ids))
        | (set(train_ids) & set(test_ids))
        | (set(eval_ids) & set(test_ids))
    )
    if overlap:
        raise ValueError(
            f"split manifest has train/eval leakage or held-out test leakage: {overlap}"
        )
    manifest["train_evidence_ids"] = train_ids
    manifest["eval_evidence_ids"] = eval_ids
    manifest["test_evidence_ids"] = test_ids
    return manifest


def validate_dataset(
    dataset: Path,
    *,
    expected_rows: int | None = None,
    split_manifest: Path | None = None,
    split: SplitName = "train",
    require_current_prompt_source: bool = True,
) -> dict[str, Any]:
    if split not in {"train", "eval", "test", "heldout"}:
        raise ValueError("split must be 'train', 'eval', 'test', or 'heldout'")
    if expected_rows is not None and (
        isinstance(expected_rows, bool)
        or not isinstance(expected_rows, int)
        or expected_rows <= 0
    ):
        raise ValueError("expected_rows must be a positive integer")

    rows: list[dict[str, Any]] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{dataset}:{line_number} is not valid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{dataset}:{line_number} is not a JSON object")
            missing = sorted(REQUIRED_FIELDS - set(row))
            if missing:
                raise ValueError(f"{dataset}:{line_number} is missing fields: {missing}")
            try:
                validate_swift_row(
                    row,
                    require_files=True,
                    require_current_prompt_source=require_current_prompt_source,
                )
            except (ValueError, FileNotFoundError) as error:
                raise type(error)(f"{dataset}:{line_number}: {error}") from error
            if row.get("question_type") != PRODUCTION_QUESTION_TYPE:
                raise ValueError(
                    f"{dataset}:{line_number}: question_type must be 'neutral'"
                )
            if row.get("generation_mode") != PRODUCTION_GENERATION_MODE:
                raise ValueError(
                    f"{dataset}:{line_number}: generation_mode must be 'baseline'"
                )
            rows.append(row)

    if not rows:
        raise ValueError("dataset is empty")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"dataset must contain {expected_rows} rows; found {len(rows)}")

    evidence_ids = [str(row["evidence_id"]) for row in rows]
    if len(set(evidence_ids)) != len(evidence_ids):
        raise ValueError("dataset contains duplicate evidence_id values")

    split_sha256 = None
    manifest_count = None
    if split_manifest is not None:
        try:
            raw_manifest = json.loads(split_manifest.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("split manifest is not valid JSON") from error
        manifest = _validate_split_manifest(
            raw_manifest,
            require_current_prompt_source=require_current_prompt_source,
        )
        if split == "heldout":
            expected_ids = (
                manifest["eval_evidence_ids"] + manifest["test_evidence_ids"]
            )
            manifest_count = manifest["eval_count"] + manifest["test_count"]
        else:
            expected_ids = manifest[f"{split}_evidence_ids"]
            manifest_count = manifest[f"{split}_count"]
        if expected_ids != evidence_ids:
            raise ValueError(
                f"split manifest {split}_evidence_ids does not match dataset order"
            )
        if len(rows) != manifest_count:
            raise ValueError(
                f"dataset row count does not match split manifest {split}_count"
            )
        split_sha256 = _sha256(split_manifest)

    return {
        "status": "passed",
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "split": split,
        "row_count": len(rows),
        "evidence_count": len(evidence_ids),
        "generator_image_reference_count": sum(
            len(row["generator_image_paths"]) for row in rows
        ),
        "reviewer_video_reference_count": 2 * len(rows),
        "split_manifest": str(split_manifest.resolve()) if split_manifest else None,
        "split_manifest_sha256": split_sha256,
        "split_manifest_count": manifest_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a score-only retained-frame GRPO JSONL"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument(
        "--split", choices=("train", "eval", "test", "heldout"), default="train"
    )
    parser.add_argument(
        "--allow-stale-prompt-source",
        action="store_true",
        help=(
            "Allow stale prompt source-file provenance only when every serialized "
            "prompt exactly matches the current production builder output."
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = validate_dataset(
        args.dataset,
        expected_rows=args.expected_rows,
        split_manifest=args.split_manifest,
        split=args.split,
        require_current_prompt_source=not args.allow_stale_prompt_source,
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(
        "DATASET_CONTRACT_PASSED "
        f"split={result['split']} rows={result['row_count']} "
        f"evidence={result['evidence_count']} sha256={result['dataset_sha256']}"
    )


if __name__ == "__main__":
    main()
