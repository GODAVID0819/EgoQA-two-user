"""生成并验证冻结的 text-only 训练/评估数据。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from training.grpo_v3.experiments.text_only_a_density.domain import (
    EXPERIMENT_VERSION,
    prompt_for,
)


COUNTS = {"train": 10, "eval": 32}
ALLOWED_KEYS = {"messages", "trial_id", "phase"}


def build_records(phase: str) -> list[dict[str, Any]]:
    if phase not in COUNTS:
        raise ValueError("phase 只能是 train 或 eval")
    return [
        {
            "messages": [{"role": "user", "content": prompt_for(f"{phase}-{index:02d}")}],
            "trial_id": f"{phase}-{index:02d}",
            "phase": phase,
        }
        for index in range(COUNTS[phase])
    ]


def _payload(records: list[dict[str, Any]]) -> bytes:
    text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in records
    )
    return text.encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_dataset_bundle(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}
    for phase in ("train", "eval"):
        records = build_records(phase)
        payload = _payload(records)
        filename = f"{phase}.jsonl"
        (root / filename).write_bytes(payload)
        datasets[phase] = {
            "path": filename,
            "sha256": _sha256(payload),
            "record_count": len(records),
            "trial_ids": [row["trial_id"] for row in records],
        }
    manifest = {
        "schema_version": "text_only_a_density_dataset_v1",
        "experiment_version": EXPERIMENT_VERSION,
        "datasets": datasets,
    }
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def validate_dataset(
    path: Path,
    manifest: dict[str, Any],
    *,
    phase: str,
) -> dict[str, Any]:
    if phase not in COUNTS:
        raise ValueError("phase 只能是 train 或 eval")
    raw = path.read_bytes()
    expected = manifest["datasets"][phase]
    if _sha256(raw) != expected["sha256"]:
        raise ValueError("数据 SHA-256 与冻结 manifest 不一致")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    ids = [row.get("trial_id") for row in rows]
    expected_ids = [f"{phase}-{index:02d}" for index in range(COUNTS[phase])]
    if len(rows) != COUNTS[phase] or ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError("记录数或 trial ID 不符合冻结规格")
    for row in rows:
        if set(row) != ALLOWED_KEYS or row.get("phase") != phase:
            raise ValueError("数据 schema 不符合 text-only 冻结规格")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 1:
            raise ValueError("每条记录必须恰好包含一条 user message")
        if messages[0] != {"role": "user", "content": prompt_for(str(row["trial_id"]))}:
            raise ValueError("prompt 与 trial ID 不符合冻结模板")
    return {"status": "passed", "phase": phase, "record_count": len(rows), "sha256": _sha256(raw)}


def main() -> None:
    parser = argparse.ArgumentParser(description="生成/验证冻结 A-density text-only 数据")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = write_dataset_bundle(args.output_dir)
    results = {
        phase: validate_dataset(args.output_dir / f"{phase}.jsonl", manifest, phase=phase)
        for phase in ("train", "eval")
    }
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
