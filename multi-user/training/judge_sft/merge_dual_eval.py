#!/usr/bin/env python3
"""Merge base and LoRA prediction files and report paired deltas."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    out = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            eid = str(row["example_id"])
            if eid in out:
                raise RuntimeError(f"Duplicate ID in {path}: {eid}")
            out[eid] = row
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--lora", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    base = read_jsonl(args.base)
    lora = read_jsonl(args.lora)
    if set(base) != set(lora):
        raise RuntimeError(
            f"Base/LoRA ID mismatch: base={len(base)} lora={len(lora)} "
            f"only_base={len(set(base)-set(lora))} only_lora={len(set(lora)-set(base))}"
        )

    groups = sorted({base[e]["group"] for e in base})
    paired = []
    for eid in sorted(base):
        b, l = base[eid], lora[eid]
        if int(b["gold"]) != int(l["gold"]):
            raise RuntimeError(f"Gold mismatch for {eid}")
        paired.append(
            {
                "example_id": eid,
                "group": b["group"],
                "gold": int(b["gold"]),
                "base_pred": int(b["pred"]),
                "lora_pred": int(l["pred"]),
                "base_margin": float(b["margin"]),
                "lora_margin": float(l["margin"]),
                "base_bce": float(b["raw_bce"]),
                "lora_bce": float(l["raw_bce"]),
            }
        )

    def calc(xs):
        n = len(xs)
        base_acc = sum(x["base_pred"] == x["gold"] for x in xs) / n
        lora_acc = sum(x["lora_pred"] == x["gold"] for x in xs) / n
        base_bce = sum(x["base_bce"] for x in xs) / n
        lora_bce = sum(x["lora_bce"] for x in xs) / n
        return {
            "n": n,
            "base_accuracy": base_acc,
            "lora_accuracy": lora_acc,
            "accuracy_delta_lora_minus_base": lora_acc - base_acc,
            "base_raw_bce": base_bce,
            "lora_raw_bce": lora_bce,
            "raw_bce_delta_lora_minus_base": lora_bce - base_bce,
            "base_wrong_lora_right": sum(
                x["base_pred"] != x["gold"] and x["lora_pred"] == x["gold"] for x in xs
            ),
            "base_right_lora_wrong": sum(
                x["base_pred"] == x["gold"] and x["lora_pred"] != x["gold"] for x in xs
            ),
            "prediction_disagreements": sum(
                x["base_pred"] != x["lora_pred"] for x in xs
            ),
        }

    report = {
        "overall": calc(paired),
        "by_group": {
            g: calc([x for x in paired if x["group"] == g])
            for g in groups
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
