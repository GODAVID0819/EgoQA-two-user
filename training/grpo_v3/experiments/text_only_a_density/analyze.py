"""轻量分析 10-step 在线 reward trace。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def analyze_trace(path: Path, *, expected_steps: int = 10) -> dict[str, Any]:
    rows = _rows(path)
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if isinstance(row.get("reward_call_index"), int):
            groups[int(row["reward_call_index"])].append(row)
    group_stats = []
    for index in sorted(groups):
        rewards = [float(row["reward"]) for row in groups[index] if isinstance(row.get("reward"), (int, float))]
        group_stats.append(
            {
                "reward_call_index": index,
                "count": len(groups[index]),
                "mean": statistics.fmean(rewards) if rewards else None,
                "std": statistics.pstdev(rewards) if len(rewards) == 4 else None,
            }
        )
    means = [float(item["mean"]) for item in group_stats if item["mean"] is not None]
    early = statistics.fmean(means[:3]) if len(means) >= 3 else None
    late = statistics.fmean(means[-3:]) if len(means) >= 3 else None
    delta = None if early is None or late is None else late - early
    checks = {
        "exact_40_rows": len(rows) == expected_steps * 4,
        "exact_10_groups": set(groups) == set(range(expected_steps)),
        "four_candidates_per_group": all(len(group) == 4 for group in groups.values()),
        "all_rewards_finite_bounded": all(
            isinstance(row.get("reward"), (int, float))
            and math.isfinite(float(row["reward"]))
            and -1 <= float(row["reward"]) <= 1
            for row in rows
        ),
        "first_three_groups_have_variance": all(
            item["std"] is not None and float(item["std"]) > 0 for item in group_stats[:3]
        ),
        "late_mean_higher_than_early_mean": delta is not None and delta > 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    integrity_names = {
        "exact_40_rows",
        "exact_10_groups",
        "four_candidates_per_group",
        "all_rewards_finite_bounded",
    }
    integrity_failed = [name for name in failed if name in integrity_names]
    status = "invalid" if integrity_failed else ("passed" if not failed else "not_converged")
    return {
        "schema_version": "text_only_a_density_quick_trace_v1",
        "status": status,
        "checks": checks,
        "failed_checks": failed,
        "row_count": len(rows),
        "group_count": len(groups),
        "early_mean": early,
        "late_mean": late,
        "late_mean_minus_early_mean": delta,
        "groups": group_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="轻量分析 A-density 10-step reward trace")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, default=10)
    args = parser.parse_args()
    result = analyze_trace(args.trace, expected_steps=args.expected_steps)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] == "invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
