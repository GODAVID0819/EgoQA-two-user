"""重建 GRPO completion group，并生成 Gate 3/4 收敛验收指标。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from training.grpo_v3.shared.contract import (
    CONVERGENCE_WINDOW_GROUPS,
    GATE3_STEPS,
    GATE4_EVAL_EVIDENCE,
    GATE4_STEPS,
    HOLDOUT_MAX_DROP,
)

TARGET_COMPONENTS = ("groundedness", "combined_answerability", "qa_formality")


def _mean(values: Iterable[float]) -> float | None:
    finite = [float(value) for value in values if isinstance(value, (int, float)) and math.isfinite(float(value))]
    return statistics.fmean(finite) if finite else None


def _window_component_means(groups: list[dict[str, Any]], *, first: bool) -> dict[str, float]:
    window = groups[:CONVERGENCE_WINDOW_GROUPS] if first else groups[-CONVERGENCE_WINDOW_GROUPS:]
    values: dict[str, list[float]] = defaultdict(list)
    for group in window:
        for row in group["rows"]:
            components = row.get("record", {}).get("reward_components", {})
            if isinstance(components, dict):
                for key, value in components.items():
                    if isinstance(value, (int, float)) and math.isfinite(float(value)):
                        values[str(key)].append(float(value))
    return {key: statistics.fmean(items) for key, items in values.items() if items}


def _window_format_summary(groups: list[dict[str, Any]], *, first: bool) -> tuple[dict[str, int], dict[str, float]]:
    window = groups[:CONVERGENCE_WINDOW_GROUPS] if first else groups[-CONVERGENCE_WINDOW_GROUPS:]
    counts: Counter[str] = Counter()
    for group in window:
        counts.update(group["format_counts"])
    total = sum(counts.values())
    return dict(counts), ({key: value / total for key, value in counts.items()} if total else {})


def _phase_metrics(groups: list[dict[str, Any]]) -> dict[str, Any]:
    means = [group["reward_mean"] for group in groups if group["reward_mean"] is not None]
    early = _mean(means[:CONVERGENCE_WINDOW_GROUPS])
    late = _mean(means[-CONVERGENCE_WINDOW_GROUPS:])
    early_components = _window_component_means(groups, first=True)
    late_components = _window_component_means(groups, first=False)
    early_format_counts, early_format_rates = _window_format_summary(groups, first=True)
    late_format_counts, late_format_rates = _window_format_summary(groups, first=False)
    component_delta = {
        key: late_components[key] - early_components[key]
        for key in sorted(set(early_components) & set(late_components))
    }
    return {
        "group_count": len(groups),
        "early_reward_mean": early,
        "late_reward_mean": late,
        "reward_delta": None if early is None or late is None else late - early,
        "positive_std_groups": sum(group["reward_std"] is not None and group["reward_std"] > 0 for group in groups),
        "positive_std_ratio": (
            sum(group["reward_std"] is not None and group["reward_std"] > 0 for group in groups) / len(groups)
            if groups else 0.0
        ),
        "early_component_means": early_components,
        "late_component_means": late_components,
        "component_delta": component_delta,
        "early_format_counts": early_format_counts,
        "late_format_counts": late_format_counts,
        "early_format_rates": early_format_rates,
        "late_format_rates": late_format_rates,
        "group_reward_series": [
            {
                "reward_call_index": group["reward_call_index"],
                "evidence_ids": group["evidence_ids"],
                "reward_mean": group["reward_mean"],
                "reward_std": group["reward_std"],
                "format_counts": group["format_counts"],
            }
            for group in groups
        ],
    }


def _trainer_metrics(trainer_state: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    output: dict[str, list[float]] = defaultdict(list)
    nonfinite: list[str] = []
    for entry in (trainer_state or {}).get("log_history", []) or []:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            normalized = str(key).lower()
            if not any(
                token in normalized
                for token in ("grad_norm", "clip", "entropy", "kl", "completion_length", "completions/")
            ):
                continue
            if isinstance(value, (int, float)):
                if math.isfinite(float(value)):
                    output[str(key)].append(float(value))
                else:
                    nonfinite.append(str(key))
    return dict(output), nonfinite


def analyze_convergence(
    rows: list[dict[str, Any]],
    *,
    gate: int,
    trainer_state: dict[str, Any] | None = None,
    expected_eval_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    if gate not in {3, 4}:
        raise ValueError(f"收敛分析仅支持 Gate 3/4，收到 Gate {gate}")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["reward_call_index"])].append(row)

    groups: list[dict[str, Any]] = []
    all_finite = True
    masked_count = 0
    formats: Counter[str] = Counter()
    for call_index in sorted(grouped):
        members = sorted(grouped[call_index], key=lambda item: int(item.get("candidate_index", -1)))
        rewards = [item.get("reward") for item in members]
        finite_rewards = [float(value) for value in rewards if isinstance(value, (int, float)) and math.isfinite(float(value))]
        all_finite = all_finite and len(finite_rewards) == len(rewards)
        masked_count += sum(bool(item.get("record", {}).get("masked")) for item in members)
        group_formats: Counter[str] = Counter()
        for item in members:
            status = item.get("record", {}).get("format_validation", {}).get("status")
            formats[str(status or "missing")] += 1
            group_formats[str(status or "missing")] += 1
        phases = {str(item.get("phase")) for item in members}
        evidence_ids = {str(item.get("evidence_id")) for item in members}
        groups.append(
            {
                "reward_call_index": call_index,
                "phase": next(iter(phases)) if len(phases) == 1 else "mixed",
                "evidence_ids": sorted(evidence_ids),
                "cardinality": len(members),
                "reward_mean": _mean(finite_rewards),
                "reward_std": statistics.pstdev(finite_rewards) if len(finite_rewards) == len(members) and len(members) > 1 else None,
                "format_counts": dict(group_formats),
                "rows": members,
            }
        )

    train_groups = [group for group in groups if group["phase"] == "train"]
    eval_groups = [group for group in groups if group["phase"] == "eval"]
    train = _phase_metrics(train_groups)
    eval_metrics: dict[str, Any] = _phase_metrics(eval_groups)
    trainer_metrics, nonfinite_trainer_metrics = _trainer_metrics(trainer_state)
    checks: dict[str, bool] = {
        "all_rewards_finite": all_finite,
        "masked_count_zero": masked_count == 0,
        "all_groups_have_four_candidates": all(group["cardinality"] == 4 for group in groups),
        "all_groups_single_phase": all(group["phase"] in {"train", "eval"} for group in groups),
        "train_reward_improved": bool(train["reward_delta"] is not None and train["reward_delta"] > 0),
        "train_positive_std_majority": train["positive_std_ratio"] >= (0.8 if gate == 3 else 0.5),
        "required_train_groups": len(train_groups) >= (GATE3_STEPS if gate == 3 else GATE4_STEPS),
        "trainer_logged_metrics_finite": not nonfinite_trainer_metrics,
    }

    if gate == 4:
        baseline = eval_groups[:GATE4_EVAL_EVIDENCE]
        final = eval_groups[-GATE4_EVAL_EVIDENCE:]
        baseline_mean = _mean(group["reward_mean"] for group in baseline)
        final_mean = _mean(group["reward_mean"] for group in final)
        reward_delta = None if baseline_mean is None or final_mean is None else final_mean - baseline_mean
        baseline_ids = {item for group in baseline for item in group["evidence_ids"]}
        final_ids = {item for group in final for item in group["evidence_ids"]}
        expected = set(str(item) for item in (expected_eval_ids or []))
        eval_metrics.update(
            {
                "baseline_group_count": len(baseline),
                "final_group_count": len(final),
                "baseline_reward_mean": baseline_mean,
                "final_reward_mean": final_mean,
                "reward_delta": reward_delta,
                "baseline_evidence_ids": sorted(baseline_ids),
                "final_evidence_ids": sorted(final_ids),
            }
        )
        checks.update(
            {
                "eval_has_baseline_and_final": len(eval_groups) >= 2 * GATE4_EVAL_EVIDENCE,
                "eval_ids_match_manifest": bool(expected) and baseline_ids == expected and final_ids == expected,
                "holdout_drop_within_limit": reward_delta is not None and reward_delta >= -HOLDOUT_MAX_DROP,
                "target_component_improved": any(train["component_delta"].get(key, 0.0) > 0 for key in TARGET_COMPONENTS),
            }
        )

    failed_checks = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "grpo_v3_convergence_v1",
        "gate": gate,
        "status": "passed" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "row_count": len(rows),
        "group_count": len(groups),
        "masked_count": masked_count,
        "format_counts": dict(formats),
        "train": train,
        "eval": eval_metrics,
        "trainer_global_step": int((trainer_state or {}).get("global_step") or 0),
        "trainer_metrics": trainer_metrics,
        "nonfinite_trainer_metrics": nonfinite_trainer_metrics,
        "completion_length_chars_mean": _mean(
            row.get("completion_length_chars") for row in rows
        ),
        "kl_disabled": True,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_trainer_state(output_dir: Path) -> dict[str, Any]:
    paths = list(output_dir.rglob("trainer_state.json"))
    if not paths:
        return {}
    path = max(paths, key=lambda item: item.stat().st_mtime_ns)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="分析 Gate 3/4 GRPO 收敛证据")
    parser.add_argument("--gate", type=int, choices=(3, 4), required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path)
    args = parser.parse_args()
    expected_eval_ids = None
    if args.split_manifest:
        expected_eval_ids = json.loads(args.split_manifest.read_text(encoding="utf-8"))["eval_evidence_ids"]
    result = analyze_convergence(
        _read_jsonl(args.trace),
        gate=args.gate,
        trainer_state=_latest_trainer_state(args.training_output),
        expected_eval_ids=expected_eval_ids,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": result["status"], "failed_checks": result["failed_checks"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
