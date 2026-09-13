"""为新版 Gate 3 构造 20 个训练 evidence 与 8 个 held-out greedy 评估 evidence。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

from training.grpo_v3.shared.data import packet_to_swift_row, read_jsonl, write_jsonl


def _unique_packets(packets: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for packet in packets:
        evidence_id = str(packet.get("evidence_id") or "").strip()
        if not evidence_id:
            raise ValueError("packet 缺少 evidence_id")
        if evidence_id in by_id:
            raise ValueError(f"重复 evidence_id: {evidence_id}")
        by_id[evidence_id] = packet
    return [by_id[key] for key in sorted(by_id)]


def build_gate3_split(
    packets: Iterable[dict[str, Any]], *, seed: int = 42, train_count: int = 20,
    eval_count: int = 8, generation_mode: str = "baseline",
    prompt_builder: Callable[..., str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    unique = _unique_packets(packets)
    required = train_count + eval_count
    if len(unique) < required:
        raise ValueError(f"至少需要 {required} 个不同 evidence，实际只有 {len(unique)}")
    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)
    train_packets = shuffled[:train_count]
    eval_packets = shuffled[train_count:required]

    def convert(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            packet_to_swift_row(
                packet,
                question_type="commonality" if index % 2 == 0 else "difference",
                generation_mode=generation_mode,
                prompt_builder=prompt_builder,
            )
            for index, packet in enumerate(rows)
        ]

    train = convert(train_packets)
    evaluation = convert(eval_packets)
    manifest = {
        "schema_version": "grpo_v3_gate3_v2_split_v1",
        "seed": seed,
        "train_count": len(train),
        "eval_count": len(evaluation),
        "train_evidence_ids": [row["evidence_id"] for row in train],
        "eval_evidence_ids": [row["evidence_id"] for row in evaluation],
        "train_question_type_counts": dict(Counter(row["question_type"] for row in train)),
        "eval_question_type_counts": dict(Counter(row["question_type"] for row in evaluation)),
    }
    return train, evaluation, manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="构造 GRPO v3 Gate 3 v2 train/eval 数据")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-count", type=int, default=20)
    parser.add_argument("--eval-count", type=int, default=8)
    parser.add_argument("--generation-mode", default="baseline")
    args = parser.parse_args()
    train, evaluation, manifest = build_gate3_split(
        read_jsonl(args.evidence), seed=args.seed, train_count=args.train_count,
        eval_count=args.eval_count, generation_mode=args.generation_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "gate3_v2_train_native_video.jsonl"
    eval_path = args.output_dir / "gate3_v2_eval_native_video.jsonl"
    manifest_path = args.output_dir / "gate3_v2_split_manifest.json"
    write_jsonl(train_path, train)
    write_jsonl(eval_path, evaluation)
    manifest.update({"source_evidence": str(args.evidence.resolve()), "source_sha256": _sha256(args.evidence)})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "gate3_v2_dataset_preview.json").write_text(
        json.dumps({"train": train, "eval": evaluation}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"train": str(train_path), "eval": str(eval_path), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
