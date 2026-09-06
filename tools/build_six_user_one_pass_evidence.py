#!/usr/bin/env python3
"""构造六用户 one-pass 实验的紧凑 evidence JSONL。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from one_pass_evidence import write_one_pass_evidence  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", action="append", required=True)
    parser.add_argument("--compact-output", required=True)
    parser.add_argument("--expanded-output", required=True)
    parser.add_argument("--source-job-id", required=True)
    args = parser.parse_args(argv)
    if len(args.asset) != 3:
        raise SystemExit(f"one-pass requires exactly three group assets, got {len(args.asset)}")
    summary = write_one_pass_evidence(
        args.asset,
        compact_output=args.compact_output,
        expanded_output=args.expanded_output,
        source_job_id=args.source_job_id,
    )
    print(
        "one_pass_evidence_ready "
        f"groups={len(summary['generation_groups'])} "
        f"packets={summary['compact_packet_count']} "
        f"slots={summary['expanded_slot_count']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
