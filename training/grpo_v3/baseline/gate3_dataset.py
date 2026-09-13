"""Build retained-frame Gate 3 datasets from random or named evidence splits."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..shared.data import packet_to_swift_row, read_jsonl, write_jsonl
from ..shared.media import GENERATOR_MEDIA_MODE


SPLIT_SCHEMA_VERSION = "grpo_v3_gate3_retained_frame_split_v6"
PRODUCTION_QUESTION_TYPE = "neutral"
PRODUCTION_GENERATION_MODE = "baseline"


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _unique_packets(packets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for packet in packets:
        evidence_id = str(packet.get("evidence_id") or "").strip()
        if not evidence_id:
            raise ValueError("packet is missing evidence_id")
        if evidence_id in by_id:
            raise ValueError(f"duplicate evidence_id: {evidence_id}")
        by_id[evidence_id] = packet
    return [by_id[key] for key in sorted(by_id)]


def _merge_packet_sources(
    packet_sources: Iterable[Iterable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Merge manifests, accepting only byte-equivalent duplicate packet objects."""

    by_id: dict[str, dict[str, Any]] = {}
    canonical: dict[str, str] = {}
    for packets in packet_sources:
        for packet in packets:
            evidence_id = str(packet.get("evidence_id") or "").strip()
            if not evidence_id:
                raise ValueError("packet is missing evidence_id")
            serialized = json.dumps(
                packet, ensure_ascii=False, allow_nan=False, sort_keys=True
            )
            if evidence_id in by_id and canonical[evidence_id] != serialized:
                raise ValueError(
                    "conflicting duplicate evidence packet across manifests: "
                    f"{evidence_id}"
                )
            by_id[evidence_id] = packet
            canonical[evidence_id] = serialized
    return [by_id[key] for key in sorted(by_id)]


def _path_with_rebased_root(value: Any, source_root: str, target_root: str) -> Any:
    if not isinstance(value, str):
        return value
    source = source_root.rstrip("/\\")
    target = target_root.rstrip("/\\")
    if value == source:
        return target
    for separator in ("/", "\\"):
        prefix = source + separator
        if value.startswith(prefix):
            return target + value[len(source):]
    return value


def rebase_active_media_paths(
    packet: dict[str, Any],
    *,
    source_project_root: str,
    target_project_root: str,
) -> dict[str, Any]:
    """Rebase only media paths consumed by the generator or frozen reviewer."""

    if not str(source_project_root).strip() or not str(target_project_root).strip():
        raise ValueError("source and target project roots must be non-empty")
    rebased = copy.deepcopy(packet)
    clips = rebased.get("clips")
    if not isinstance(clips, list):
        raise ValueError("packet.clips must be a list")
    for clip in clips:
        if not isinstance(clip, dict):
            raise ValueError("packet.clips must contain JSON objects")
        frames = clip.get("frames")
        if isinstance(frames, list):
            for frame in frames:
                if isinstance(frame, dict) and "path" in frame:
                    frame["path"] = _path_with_rebased_root(
                        frame["path"], source_project_root, target_project_root
                    )
        for field in (
            "full_local_video",
            "original_local_video",
            "source_local_video",
        ):
            if field in clip:
                clip[field] = _path_with_rebased_root(
                    clip[field], source_project_root, target_project_root
                )
    return rebased


def _convert_packets(
    packets: Iterable[dict[str, Any]], *, generation_mode: str
) -> list[dict[str, Any]]:
    return [
        packet_to_swift_row(
            packet,
            question_type=PRODUCTION_QUESTION_TYPE,
            generation_mode=generation_mode,
        )
        for packet in packets
    ]


def _base_manifest(
    train: list[dict[str, Any]],
    evaluation: list[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    train_ids = [row["evidence_id"] for row in train]
    eval_ids = [row["evidence_id"] for row in evaluation]
    if set(train_ids) & set(eval_ids):
        raise RuntimeError("train/eval evidence leakage")
    prompt_fields = (
        "prompt_contract",
        "prompt_builder",
        "prompt_source_path",
        "prompt_source_sha256",
    )
    all_rows = train + evaluation
    for field in prompt_fields:
        values = {str(row[field]) for row in all_rows}
        if len(values) != 1:
            raise RuntimeError(f"train/eval rows disagree on {field}")
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": seed,
        "train_count": len(train),
        "eval_count": len(evaluation),
        "question_type": PRODUCTION_QUESTION_TYPE,
        "generation_mode": PRODUCTION_GENERATION_MODE,
        **{field: train[0][field] for field in prompt_fields},
        "train_evidence_ids": train_ids,
        "eval_evidence_ids": eval_ids,
        "train_question_type_counts": dict(
            Counter(row["question_type"] for row in train)
        ),
        "eval_question_type_counts": dict(
            Counter(row["question_type"] for row in evaluation)
        ),
        "generator_media_mode": GENERATOR_MEDIA_MODE,
        "generator_media_field": "clips[*].frames[*].path",
        "reviewer_media_fields": [
            "clips[*].full_local_video",
            "clips[*].original_local_video",
            "clips[*].source_local_video",
        ],
    }


def build_gate3_split(
    packets: Iterable[dict[str, Any]],
    *,
    seed: int = 42,
    train_count: int = 20,
    eval_count: int = 8,
    generation_mode: str = "baseline",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_count = _positive_count(train_count, "train_count")
    eval_count = _positive_count(eval_count, "eval_count")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if generation_mode != PRODUCTION_GENERATION_MODE:
        raise ValueError(
            "score-only GRPO supports only the baseline generation mode; "
            f"got {generation_mode!r}"
        )

    unique = _unique_packets(packets)
    required = train_count + eval_count
    if len(unique) < required:
        raise ValueError(
            f"at least {required} distinct evidence packets are required; found {len(unique)}"
        )
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    train_packets = shuffled[:train_count]
    eval_packets = shuffled[train_count:required]

    train = _convert_packets(train_packets, generation_mode=generation_mode)
    evaluation = _convert_packets(eval_packets, generation_mode=generation_mode)
    manifest = _base_manifest(train, evaluation, seed=seed)
    return train, evaluation, manifest


def build_named_split(
    packets: Iterable[dict[str, Any]],
    *,
    train_evidence_ids: Iterable[str],
    eval_evidence_ids: Iterable[str],
    test_evidence_ids: Iterable[str] = (),
    seed: int = 42,
    generation_mode: str = "baseline",
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Build datasets in an externally fixed evidence-ID order."""

    if generation_mode != PRODUCTION_GENERATION_MODE:
        raise ValueError(
            "score-only GRPO supports only the baseline generation mode; "
            f"got {generation_mode!r}"
        )
    ids_by_split = {
        "train": [str(value).strip() for value in train_evidence_ids],
        "eval": [str(value).strip() for value in eval_evidence_ids],
        "test": [str(value).strip() for value in test_evidence_ids],
    }
    for name, ids in ids_by_split.items():
        if name != "test" and not ids:
            raise ValueError(f"named {name} split must not be empty")
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"named {name} split IDs must be unique and non-empty")
    assigned: dict[str, str] = {}
    for name, ids in ids_by_split.items():
        for evidence_id in ids:
            if evidence_id in assigned:
                raise ValueError(
                    f"named split leakage for {evidence_id}: "
                    f"{assigned[evidence_id]} and {name}"
                )
            assigned[evidence_id] = name

    by_id = {packet["evidence_id"]: packet for packet in _unique_packets(packets)}
    missing = {
        name: [evidence_id for evidence_id in ids if evidence_id not in by_id]
        for name, ids in ids_by_split.items()
    }
    missing = {name: ids for name, ids in missing.items() if ids}
    if missing:
        raise ValueError(f"named split evidence packets are missing: {missing}")

    rows_by_split = {
        name: _convert_packets(
            [by_id[evidence_id] for evidence_id in ids],
            generation_mode=generation_mode,
        )
        for name, ids in ids_by_split.items()
    }
    manifest = _base_manifest(
        rows_by_split["train"], rows_by_split["eval"], seed=seed
    )
    manifest.update(
        {
            "selection_strategy": "external_named_evidence_split",
            "test_count": len(rows_by_split["test"]),
            "test_evidence_ids": [
                row["evidence_id"] for row in rows_by_split["test"]
            ],
            "test_question_type_counts": dict(
                Counter(row["question_type"] for row in rows_by_split["test"])
            ),
        }
    )
    return (
        rows_by_split["train"],
        rows_by_split["eval"],
        rows_by_split["test"],
        manifest,
    )


def _reviewer_split_ids(path: Path) -> tuple[dict[str, list[str]], int]:
    value = json.loads(path.read_text(encoding="utf-8"))
    split = value.get("split_manifest", value) if isinstance(value, dict) else None
    if not isinstance(split, dict):
        raise ValueError("reviewer split audit must contain a split_manifest object")
    result = {
        "train": split.get("train_evidence_ids"),
        "eval": split.get("validation_evidence_ids"),
        "test": split.get("locked_test_evidence_ids"),
    }
    if not all(isinstance(ids, list) for ids in result.values()):
        raise ValueError(
            "reviewer split audit needs train_evidence_ids, "
            "validation_evidence_ids, and locked_test_evidence_ids"
        )
    seed = split.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("reviewer split seed must be an integer")
    return result, seed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build deterministic GRPO v3 Gate 3 train/eval datasets"
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        action="append",
        required=True,
        help="Evidence JSONL; repeat this option to merge multiple manifests",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument(
        "--generation-mode",
        choices=(PRODUCTION_GENERATION_MODE,),
        default=PRODUCTION_GENERATION_MODE,
    )
    parser.add_argument(
        "--reviewer-split-audit",
        type=Path,
        help="Preserve the annotated reviewer's named 60/10/20 evidence split",
    )
    parser.add_argument("--source-project-root")
    parser.add_argument("--target-project-root")
    args = parser.parse_args()
    if bool(args.source_project_root) != bool(args.target_project_root):
        parser.error(
            "--source-project-root and --target-project-root must be provided together"
        )
    packets = _merge_packet_sources(read_jsonl(path) for path in args.evidence)
    if args.source_project_root:
        packets = [
            rebase_active_media_paths(
                packet,
                source_project_root=args.source_project_root,
                target_project_root=args.target_project_root,
            )
            for packet in packets
        ]

    test: list[dict[str, Any]] = []
    if args.reviewer_split_audit:
        named_ids, split_seed = _reviewer_split_ids(args.reviewer_split_audit)
        train, evaluation, test, manifest = build_named_split(
            packets,
            train_evidence_ids=named_ids["train"],
            eval_evidence_ids=named_ids["eval"],
            test_evidence_ids=named_ids["test"],
            seed=split_seed,
            generation_mode=args.generation_mode,
        )
        manifest.update(
            {
                "reviewer_split_audit": str(args.reviewer_split_audit.resolve()),
                "reviewer_split_audit_sha256": _sha256(args.reviewer_split_audit),
            }
        )
    else:
        train, evaluation, manifest = build_gate3_split(
            packets,
            seed=args.seed,
            train_count=args.train_count,
            eval_count=args.eval_count,
            generation_mode=args.generation_mode,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "gate3_v3_train_retained_frames.jsonl"
    eval_path = args.output_dir / "gate3_v3_eval_retained_frames.jsonl"
    test_path = args.output_dir / "gate3_v3_test_retained_frames.jsonl"
    manifest_path = args.output_dir / "gate3_v3_split_manifest.json"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, evaluation)
    if test:
        write_jsonl(test_path, test)
    manifest.update(
        {
            "source_evidence": [str(path.resolve()) for path in args.evidence],
            "source_sha256": {
                str(path.resolve()): _sha256(path) for path in args.evidence
            },
            "source_project_root": args.source_project_root,
            "target_project_root": args.target_project_root,
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "gate3_v3_dataset_preview.json").write_text(
        json.dumps({"train": train, "eval": evaluation}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "train": str(train_path),
                "eval": str(eval_path),
                "test": str(test_path) if test else None,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
