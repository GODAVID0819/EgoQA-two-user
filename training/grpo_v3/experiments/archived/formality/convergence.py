"""分析 qa_formality-only smoke/probe 的收敛证据。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from training.grpo_v3.experiments.archived.formality.reward import FORMALITY_COMPONENT


WINDOW_GROUPS = 10


def linear_slope(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    xs = list(range(1, len(values) + 1))
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(values)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(xs, values)
    ) / denominator


def analyze_formality_convergence(
    rows: list[dict[str, Any]],
    *,
    expected_steps: int,
) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["reward_call_index"])].append(row)

    groups: list[dict[str, Any]] = []
    all_rewards_finite = True
    masked_count = 0
    only_component = True
    only_judge = True
    for call_index in sorted(grouped):
        members = sorted(
            grouped[call_index],
            key=lambda item: int(item.get("candidate_index", -1)),
        )
        rewards: list[float] = []
        for row in members:
            value = row.get("reward")
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                all_rewards_finite = False
            else:
                rewards.append(float(value))
            record = row.get("record") if isinstance(row.get("record"), dict) else {}
            masked_count += int(record.get("masked") is True)
            components = record.get("reward_components")
            if not isinstance(components, dict) or set(components) != {FORMALITY_COMPONENT}:
                only_component = False
            traces = record.get("judge_trace")
            traces = traces if isinstance(traces, dict) else {}
            if record.get("judge_called") is True:
                only_judge = only_judge and set(traces) == {"qa_formality"}
            elif record.get("reward_source") == "deterministic_unjudgeable_floor":
                only_judge = only_judge and not traces
            else:
                only_judge = False
        complete = (
            len(members) == 4
            and [int(item.get("candidate_index", -1)) for item in members] == [0, 1, 2, 3]
            and len(rewards) == 4
        )
        groups.append(
            {
                "reward_call_index": call_index,
                "cardinality": len(members),
                "complete": complete,
                "reward_mean": statistics.fmean(rewards) if rewards else None,
                "reward_std": statistics.pstdev(rewards) if len(rewards) == 4 else None,
                "unjudgeable_count": sum(
                    (item.get("record") or {}).get("reward_source")
                    == "deterministic_unjudgeable_floor"
                    for item in members
                ),
            }
        )

    group_means = [
        float(group["reward_mean"])
        for group in groups
        if group["complete"] and group["reward_mean"] is not None
    ]
    early_groups = groups[:WINDOW_GROUPS]
    late_groups = groups[-WINDOW_GROUPS:]
    early_values = [
        float(group["reward_mean"])
        for group in early_groups
        if group["complete"] and group["reward_mean"] is not None
    ]
    late_values = [
        float(group["reward_mean"])
        for group in late_groups
        if group["complete"] and group["reward_mean"] is not None
    ]
    early_mean = statistics.fmean(early_values) if len(early_values) == WINDOW_GROUPS else None
    late_mean = statistics.fmean(late_values) if len(late_values) == WINDOW_GROUPS else None
    reward_delta = None if early_mean is None or late_mean is None else late_mean - early_mean
    slope = linear_slope(group_means) if len(group_means) == len(groups) else None
    positive_std_count = sum(
        group["reward_std"] is not None and float(group["reward_std"]) > 0
        for group in groups
    )
    positive_std_ratio = positive_std_count / len(groups) if groups else 0.0
    early_unjudgeable = sum(group["unjudgeable_count"] for group in early_groups)
    late_unjudgeable = sum(group["unjudgeable_count"] for group in late_groups)
    early_denominator = sum(group["cardinality"] for group in early_groups)
    late_denominator = sum(group["cardinality"] for group in late_groups)
    early_unjudgeable_rate = early_unjudgeable / early_denominator if early_denominator else 0.0
    late_unjudgeable_rate = late_unjudgeable / late_denominator if late_denominator else 0.0

    checks = {
        "required_group_count": len(groups) == expected_steps,
        "all_groups_have_four_candidates": all(group["complete"] for group in groups),
        "all_rewards_finite": all_rewards_finite and len(group_means) == len(groups),
        "infrastructure_mask_count_zero": masked_count == 0,
        "only_formality_reward_component": only_component,
        "only_formality_judge_called": only_judge,
        "positive_std_ratio_at_least_0_8": positive_std_ratio >= 0.8,
        "last_window_reward_improved": reward_delta is not None and reward_delta > 0,
        "reward_slope_positive": slope is not None and slope > 0,
        "unjudgeable_rate_not_increased": late_unjudgeable_rate <= early_unjudgeable_rate,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "grpo_v3_formality_convergence_v1",
        "status": "passed" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "expected_steps": expected_steps,
        "row_count": len(rows),
        "group_count": len(groups),
        "finite_reward_count": sum(
            isinstance(row.get("reward"), (int, float))
            and math.isfinite(float(row["reward"]))
            for row in rows
        ),
        "masked_count": masked_count,
        "positive_std_group_count": positive_std_count,
        "positive_std_ratio": positive_std_ratio,
        "early_reward_mean": early_mean,
        "late_reward_mean": late_mean,
        "reward_delta": reward_delta,
        "reward_slope": slope,
        "early_unjudgeable_rate": early_unjudgeable_rate,
        "late_unjudgeable_rate": late_unjudgeable_rate,
        "reward_components": [FORMALITY_COMPONENT],
        "groups": groups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 qa_formality-only 收敛证据")
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [
        json.loads(line)
        for line in args.trace.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    result = analyze_formality_convergence(rows, expected_steps=args.expected_steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
