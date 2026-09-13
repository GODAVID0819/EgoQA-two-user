from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .domain import REWARD_COMPONENT, REWARD_REVISION


def _mean(values: Iterable[float]) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    xs = list(range(len(values)))
    x_mean = _mean(xs)
    y_mean = _mean(values)
    denom = sum((x - x_mean) ** 2 for x in xs)
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values)) / denom if denom else 0.0


def load_trace(path: str | Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def analyze_trace(
    rows: list[dict[str, Any]],
    *,
    expected_groups: int = 40,
    expected_reward_revision: str = REWARD_REVISION,
) -> dict[str, Any]:
    reward_rows = [
        row for row in rows
        if row.get("reward_kind") == "qa_cross_view_relation"
        and row.get("failure_stage") in (None, "")
    ]
    by_group: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in reward_rows:
        by_group[int(row["reward_call_index"])].append(row)
    groups = [by_group[key] for key in sorted(by_group)]
    group_means = [_mean(float(row["reward"]) for row in group) for group in groups]
    group_stds = []
    unrecoverable_rates = []
    for group in groups:
        rewards = [float(row["reward"]) for row in group]
        mean = _mean(rewards)
        group_stds.append(math.sqrt(_mean((item - mean) ** 2 for item in rewards)))
        unrecoverable_rates.append(
            _mean(
                1.0 if ((row.get("record") or {}).get("deterministic") or {}).get("format_status") == "unrecoverable" else 0.0
                for row in group
            )
        )
    first10 = group_means[:10]
    last10 = group_means[-10:] if len(group_means) >= 10 else group_means
    reward_delta = _mean(last10) - _mean(first10)
    slope = _slope(group_means)
    positive_std_ratio = _mean(1.0 if value > 0 else 0.0 for value in group_stds)
    first_unrec = _mean(unrecoverable_rates[:10])
    last_unrec = _mean(unrecoverable_rates[-10:] if len(unrecoverable_rates) >= 10 else unrecoverable_rates)
    finite = all(math.isfinite(float(row["reward"])) for row in reward_rows)
    components_ok = all(
        set(((row.get("record") or {}).get("reward_components") or {})) == {REWARD_COMPONENT}
        and (row.get("record") or {}).get("reward_revision") == expected_reward_revision
        for row in reward_rows
    )
    failed_checks = []
    checks = {
        "expected_group_count": len(groups) == expected_groups,
        "expected_reward_count": len(reward_rows) == expected_groups * 4,
        "all_rewards_finite": finite,
        "only_cross_view_relation_component": components_ok,
        "last10_mean_improved": reward_delta > 0,
        "reward_slope_positive": slope > 0,
        "positive_variance_group_ratio": positive_std_ratio >= 0.8,
        "unrecoverable_rate_not_increased": last_unrec <= first_unrec + 0.02,
    }
    failed_checks = [name for name, ok in checks.items() if not ok]
    return {
        "status": "passed" if not failed_checks else "failed",
        "reward_revision": expected_reward_revision,
        "group_count": len(groups),
        "reward_count": len(reward_rows),
        "first10_mean": _mean(first10),
        "last10_mean": _mean(last10),
        "reward_delta": reward_delta,
        "reward_slope": slope,
        "positive_variance_group_ratio": positive_std_ratio,
        "first10_unrecoverable_rate": first_unrec,
        "last10_unrecoverable_rate": last_unrec,
        "checks": checks,
        "failed_checks": failed_checks,
        "group_means": group_means,
        "group_stds": group_stds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-groups", type=int, default=40)
    parser.add_argument("--expected-reward-revision", default=REWARD_REVISION)
    args = parser.parse_args()
    result = analyze_trace(
        load_trace(args.trace),
        expected_groups=args.expected_groups,
        expected_reward_revision=args.expected_reward_revision,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
