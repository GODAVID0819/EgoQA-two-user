#!/usr/bin/env python3
"""汇总固定 30 槽六用户 one-pass 生成结果。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from one_pass_summary import (  # noqa: E402
    summarize_one_pass_rows,
    update_one_pass_manifest,
)


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    if not file_path.is_file():
        return []
    return [
        json.loads(line)
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--accepted", required=True)
    parser.add_argument("--rejected", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--attempts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest")
    parser.add_argument("--expected-slots", type=int, default=30)
    parser.add_argument("--generation-exit-code", type=int, default=0)
    args = parser.parse_args(argv)
    result = summarize_one_pass_rows(
        evidence_rows=_read_jsonl(args.evidence),
        accepted_rows=_read_jsonl(args.accepted),
        rejected_rows=_read_jsonl(args.rejected),
        prompt_rows=_read_jsonl(args.prompts),
        attempt_rows=_read_jsonl(args.attempts),
        expected_slot_count=args.expected_slots,
        generation_exit_code=args.generation_exit_code,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.manifest:
        update_one_pass_manifest(args.manifest, output, status=result["status"])
    print(
        "one_pass_summary_ready "
        f"status={result['status']} slots={result['slot_count']} "
        f"completed={result['completed_slot_count']} accepted={result['accepted_count']} "
        f"rejected={result['rejected_count']}",
        flush=True,
    )
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
