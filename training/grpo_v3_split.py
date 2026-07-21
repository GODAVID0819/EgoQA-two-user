"""为 Gate 4 生成确定性、按 evidence_id 隔离的原生双视频拆分。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from training.grpo_v3_contract import GATE4_EVAL_EVIDENCE, GATE4_TRAIN_EVIDENCE
from training.grpo_v3_data import packet_to_swift_row, read_jsonl, write_jsonl


def _json_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_packets(
    packets: list[dict[str, Any]],
    *,
    seed: int = 42,
    question_type: str = "commonality",
    generation_mode: str = "baseline",
    prompt_builder: Callable[..., str] | None = None,
) -> dict[str, Any]:
    required = GATE4_TRAIN_EVIDENCE + GATE4_EVAL_EVIDENCE
    by_id: dict[str, dict[str, Any]] = {}
    for packet in packets:
        evidence_id = str(packet.get("evidence_id") or "").strip()
        if not evidence_id:
            raise ValueError("packet 缺少 evidence_id")
        if evidence_id in by_id:
            raise ValueError(f"duplicate/重复 evidence_id: {evidence_id}")
        by_id[evidence_id] = packet
    if len(by_id) < required:
        raise ValueError(f"Gate 4 严格需要至少 {required} 个唯一 evidence，实际 {len(by_id)}")

    ranked_ids = sorted(by_id, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())
    selected = ranked_ids[:required]
    train_ids = selected[:GATE4_TRAIN_EVIDENCE]
    eval_ids = selected[GATE4_TRAIN_EVIDENCE:]

    def convert(ids: list[str]) -> list[dict[str, Any]]:
        return [
            packet_to_swift_row(
                by_id[evidence_id],
                question_type=question_type,
                generation_mode=generation_mode,
                prompt_builder=prompt_builder,
            )
            for evidence_id in ids
        ]

    train_rows = convert(train_ids)
    eval_rows = convert(eval_ids)
    source_packets = [by_id[evidence_id] for evidence_id in sorted(by_id)]
    manifest = {
        "schema_version": "gate4_split_v1",
        "seed": seed,
        "question_type": question_type,
        "generation_mode": generation_mode,
        "source_count": len(by_id),
        "selected_count": required,
        "train_count": len(train_ids),
        "eval_count": len(eval_ids),
        "intersection_count": len(set(train_ids) & set(eval_ids)),
        "train_evidence_ids": train_ids,
        "eval_evidence_ids": eval_ids,
        "source_sha256": _json_sha256(source_packets),
        "train_sha256": _json_sha256(train_rows),
        "eval_sha256": _json_sha256(eval_rows),
    }
    return {"train_rows": train_rows, "eval_rows": eval_rows, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 Gate 4 的 40-train/10-eval 原生双视频拆分")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question-type", default="commonality")
    parser.add_argument("--generation-mode", default="baseline")
    args = parser.parse_args()
    result = split_packets(
        read_jsonl(args.evidence),
        seed=args.seed,
        question_type=args.question_type,
        generation_mode=args.generation_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "train_native_video.jsonl"
    eval_path = args.output_dir / "eval_native_video.jsonl"
    manifest_path = args.output_dir / "split_manifest.json"
    write_jsonl(train_path, result["train_rows"])
    write_jsonl(eval_path, result["eval_rows"])
    manifest_path.write_text(json.dumps(result["manifest"], ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"train": str(train_path), "eval": str(eval_path), "manifest": str(manifest_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
