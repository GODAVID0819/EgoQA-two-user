"""汇总 answer-margin step 0/40 固定种子配对评估。"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


EVAL_SEEDS = tuple(2026072100 + index for index in range(32))
CHECKPOINT_STEPS = (0, 40)
TEMPERATURE = 0.5
BOOTSTRAP_SEED = 20260721
BOOTSTRAP_REPLICATES = 10_000
PARENT_JOB = "gate2_14119442"
PARENT_CHECKPOINT = "checkpoint-1"
SCHEMA_VERSION = "combined_video_answer_margin_fixed_eval_v1"
TIE_TOLERANCE = 1e-6


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))


def _percentile(values: Sequence[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    weight = position - lower
    return float(values[lower] * (1.0 - weight) + values[upper] * weight)


def paired_bootstrap_interval(
    differences: Sequence[float],
    *,
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[float]:
    if not differences or replicates <= 0:
        raise ValueError("配对 bootstrap 需要非空差值和正重复次数")
    generator = random.Random(seed)
    count = len(differences)
    means = sorted(
        _mean([differences[generator.randrange(count)] for _ in range(count)])
        for _ in range(replicates)
    )
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _invalid(reason: str, *, row_count: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_status": "invalid",
        "experiment_conclusion": "invalid",
        "row_count": row_count,
        "failed_integrity_checks": [reason],
        "checks": {},
    }


def analyze_fixed_eval(
    rows: Iterable[Mapping[str, Any]],
    training: Mapping[str, Any],
    adapter_reload: Mapping[str, Any],
    *,
    seeds: Sequence[int] = EVAL_SEEDS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    """严格验收 64 行并按预注册八项标准给出三态结论。"""

    materialized = list(rows)
    seed_values = tuple(int(seed) for seed in seeds)
    if seed_values != EVAL_SEEDS:
        return _invalid("eval_seeds_not_frozen", row_count=len(materialized))
    if len(materialized) != 64:
        return _invalid("fixed_eval_not_64_rows", row_count=len(materialized))

    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    step40_adapters: set[str] = set()
    for row in materialized:
        try:
            step = int(row["checkpoint_step"])
            seed = int(row["seed"])
            reward_value = row["reward"]
            if isinstance(reward_value, bool):
                raise ValueError
            reward = float(reward_value)
        except (KeyError, TypeError, ValueError, OverflowError):
            return _invalid("row_schema_invalid", row_count=len(materialized))
        key = (step, seed)
        if key in by_key:
            return _invalid("duplicate_step_seed_key", row_count=len(materialized))
        if not math.isfinite(reward):
            return _invalid("nonfinite_fixed_eval_reward", row_count=len(materialized))
        if row.get("temperature") != TEMPERATURE:
            return _invalid("temperature_not_0_5", row_count=len(materialized))
        if not isinstance(row.get("top1_hit"), bool) or not isinstance(
            row.get("core_qa_extracted"), bool
        ):
            return _invalid("row_metrics_invalid", row_count=len(materialized))
        if step == 0:
            if row.get("source_job") != PARENT_JOB or row.get("checkpoint") != PARENT_CHECKPOINT:
                return _invalid("step0_parent_not_frozen", row_count=len(materialized))
        elif step == 40:
            adapter = row.get("adapter_dir")
            if row.get("source_mode") != "probe40" or not isinstance(adapter, str) or not adapter:
                return _invalid("step40_probe_adapter_invalid", row_count=len(materialized))
            step40_adapters.add(adapter)
        else:
            return _invalid("checkpoint_step_invalid", row_count=len(materialized))
        by_key[key] = row

    expected = {(step, seed) for step in CHECKPOINT_STEPS for seed in EVAL_SEEDS}
    if set(by_key) != expected:
        return _invalid("step_seed_keyspace_incomplete", row_count=len(materialized))
    if len(step40_adapters) != 1:
        return _invalid("step40_adapter_not_unique", row_count=len(materialized))

    if (
        training.get("run_status") != "passed"
        or training.get("mode") != "probe40"
        or training.get("trace_count") != 160
        or training.get("finite_reward_count") != 160
        or training.get("masked_reward_count") != 0
    ):
        return _invalid("probe40_training_integrity_failed", row_count=len(materialized))
    if adapter_reload.get("status") != "passed":
        return _invalid("adapter_reload_failed", row_count=len(materialized))

    step_rewards = {
        step: [float(by_key[(step, seed)]["reward"]) for seed in EVAL_SEEDS]
        for step in CHECKPOINT_STEPS
    }
    differences = [
        float(by_key[(40, seed)]["reward"]) - float(by_key[(0, seed)]["reward"])
        for seed in EVAL_SEEDS
    ]
    delta = _mean(differences)
    interval = paired_bootstrap_interval(
        differences, seed=bootstrap_seed, replicates=bootstrap_replicates
    )
    top1_rates = {
        step: sum(bool(by_key[(step, seed)]["top1_hit"]) for seed in EVAL_SEEDS) / 32
        for step in CHECKPOINT_STEPS
    }
    extraction_rates = {
        step: sum(bool(by_key[(step, seed)]["core_qa_extracted"]) for seed in EVAL_SEEDS) / 32
        for step in CHECKPOINT_STEPS
    }
    positive_groups = training.get("positive_variance_group_count")
    positive_variance_passed = (
        isinstance(positive_groups, int)
        and not isinstance(positive_groups, bool)
        and positive_groups >= 32
    )
    checks = {
        "step40_mean_strictly_higher": _mean(step_rewards[40]) > _mean(step_rewards[0]),
        "paired_bootstrap_ci_lower_positive": interval[0] > 0.0,
        "top1_rate_not_decreased": top1_rates[40] >= top1_rates[0],
        "training_positive_variance_groups_at_least_80_percent": positive_variance_passed,
        "fixed_eval_64_of_64": True,
        "core_qa_extraction_drop_at_most_5pp": extraction_rates[40] - extraction_rates[0] >= -0.05,
        "training_160_of_160_finite_and_zero_mask": True,
        "adapter_reload_passed": True,
    }
    conclusion = "passed" if all(checks.values()) else "not_converged"
    return {
        "schema_version": SCHEMA_VERSION,
        "run_status": "passed",
        "experiment_conclusion": conclusion,
        "row_count": 64,
        "pair_count": 32,
        "temperature": TEMPERATURE,
        "seeds": list(EVAL_SEEDS),
        "step40_adapter_dir": next(iter(step40_adapters)),
        "checkpoint_summaries": {
            str(step): {
                "reward_mean": _mean(step_rewards[step]),
                "top1_hit_rate": top1_rates[step],
                "core_qa_extraction_rate": extraction_rates[step],
            }
            for step in CHECKPOINT_STEPS
        },
        "paired_reward_delta_mean": delta,
        "paired_bootstrap_95_ci": interval,
        "paired_comparison": {
            "wins": sum(value > TIE_TOLERANCE for value in differences),
            "ties": sum(abs(value) <= TIE_TOLERANCE for value in differences),
            "losses": sum(value < -TIE_TOLERANCE for value in differences),
        },
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_replicates": bootstrap_replicates,
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "failed_integrity_checks": [],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"JSON 包含 {item}")))
    if not isinstance(value, dict):
        raise ValueError(f"{path} 必须是 JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line, parse_constant=lambda item: (_ for _ in ()).throw(ValueError(f"第 {number} 行包含 {item}")))
        if not isinstance(value, dict):
            raise ValueError(f"第 {number} 行必须是 JSON object")
        rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 answer-margin step 0/40 固定配对评估")
    parser.add_argument("--results", type=Path, required=True, help="严格 64 行的 JSONL")
    parser.add_argument("--training-summary", type=Path, required=True)
    parser.add_argument("--adapter-reload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = analyze_fixed_eval(
        _read_jsonl(args.results),
        _read_json(args.training_summary),
        _read_json(args.adapter_reload),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, allow_nan=False))
    if summary["experiment_conclusion"] == "invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
