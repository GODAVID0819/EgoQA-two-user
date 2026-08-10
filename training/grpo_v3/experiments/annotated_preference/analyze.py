"""汇总 ms-swift DPO 训练日志，并执行 smoke、overfit、train、validation Gate。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


_METRICS = {
    "loss": ("loss",),
    "eval_loss": ("eval_loss",),
    "reward_margin": ("rewards/margins",),
    "pair_accuracy": ("rewards/accuracies",),
    "eval_reward_margin": ("eval_rewards/margins", "eval_rewards/margin"),
    "eval_pair_accuracy": ("eval_rewards/accuracies", "eval_rewards/accuracy"),
    "chosen_logp": ("rewards/logps/chosen",),
    "rejected_logp": ("rewards/logps/rejected",),
    "grad_norm": ("grad_norm",),
}


def _failed(mode: str, reasons: list[str]) -> dict[str, Any]:
    return {
        "status": "failed",
        "mode": mode,
        "reasons": reasons,
        "finite_metrics": False,
    }


def _load_object(path: str | Path, label: str) -> tuple[dict[str, Any] | None, str | None]:
    input_path = Path(path)
    try:
        with input_path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        return None, f"{label} unreadable: {error}"
    if not isinstance(value, dict):
        return None, f"{label} must be a JSON object"
    return value, None


def _metric_values(history: Sequence[Mapping[str, Any]], names: Sequence[str]) -> list[float]:
    values: list[float] = []
    for row in history:
        for name in names:
            if name in row:
                value = row[name]
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    values.append(math.nan)
                else:
                    values.append(float(value))
                break
    return values


def _final_value(values: Sequence[float]) -> float | None:
    return values[-1] if values else None


def analyze_paths(
    trainer_state: str | Path,
    parameter_audit: str | Path,
    dataset_manifest: str | Path,
    *,
    mode: str,
) -> dict[str, Any]:
    """返回可持久化的 Gate 结果；任何输入或合同问题都以 failed/reasons 表示。"""
    if mode not in {"smoke", "overfit", "train", "validation"}:
        return _failed(mode, ["mode must be one of smoke, overfit, train, validation"])

    state, state_error = _load_object(trainer_state, "trainer_state")
    audit, audit_error = _load_object(parameter_audit, "parameter_audit")
    manifest, manifest_error = _load_object(dataset_manifest, "dataset_manifest")
    load_errors = [error for error in (state_error, audit_error, manifest_error) if error]
    if load_errors:
        return _failed(mode, load_errors)
    assert state is not None and audit is not None and manifest is not None

    reasons: list[str] = []
    history = state.get("log_history")
    if not isinstance(history, list) or not history or not all(isinstance(row, dict) for row in history):
        return _failed(mode, ["trainer_state.log_history must be a non-empty list of objects"])
    counts = manifest.get("counts")
    if not isinstance(counts, dict) or isinstance(counts.get("validation_pair_count"), bool) or not isinstance(counts.get("validation_pair_count"), int):
        return _failed(mode, ["dataset_manifest.counts.validation_pair_count must be an integer"])
    for key in ("lora_delta_nonzero", "non_lora_delta_zero", "checkpoint_exists"):
        if not isinstance(audit.get(key), bool):
            reasons.append(f"parameter_audit.{key} must be a boolean")

    typed_history = [row for row in history if isinstance(row, Mapping)]
    metric_values = {
        name: _metric_values(typed_history, aliases) for name, aliases in _METRICS.items()
    }
    observed_values = [value for values in metric_values.values() for value in values]
    finite_metrics = bool(observed_values) and all(math.isfinite(value) for value in observed_values)
    if not finite_metrics:
        reasons.append("all observed training metrics must be finite")

    global_step = state.get("global_step", 0)
    if isinstance(global_step, bool) or not isinstance(global_step, int):
        reasons.append("trainer_state.global_step must be an integer")
        global_step = 0
    eval_pair_count = state.get("eval_pair_count")
    if eval_pair_count is None:
        for row in reversed(typed_history):
            if "eval_pair_count" in row:
                eval_pair_count = row["eval_pair_count"]
                break
    if isinstance(eval_pair_count, bool) or not isinstance(eval_pair_count, int):
        eval_pair_count = None

    result: dict[str, Any] = {
        "status": "passed",
        "mode": mode,
        "reasons": reasons,
        "global_step": global_step,
        "metric_counts": {name: len(values) for name, values in metric_values.items()},
        "finite_metrics": finite_metrics,
        "initial_reward_margin": metric_values["reward_margin"][0] if metric_values["reward_margin"] else None,
        "final_reward_margin": _final_value(metric_values["reward_margin"]),
        "final_pair_accuracy": _final_value(metric_values["pair_accuracy"]),
        "final_eval_loss": _final_value(metric_values["eval_loss"]),
        "final_eval_reward_margin": _final_value(metric_values["eval_reward_margin"]),
        "final_eval_pair_accuracy": _final_value(metric_values["eval_pair_accuracy"]),
        "eval_pair_count": eval_pair_count,
        "manifest_validation_pair_count": counts["validation_pair_count"],
    }
    if mode == "smoke":
        for name in ("loss", "grad_norm"):
            if not metric_values[name]:
                reasons.append(f"smoke requires {name}")
        if audit.get("lora_delta_nonzero") is not True:
            reasons.append("smoke requires lora_delta_nonzero")
        if audit.get("non_lora_delta_zero") is not True:
            reasons.append("smoke requires non_lora_delta_zero")
        if audit.get("checkpoint_exists") is not True:
            reasons.append("smoke requires checkpoint_exists")
    elif mode == "overfit":
        margins = metric_values["reward_margin"]
        accuracy = _final_value(metric_values["pair_accuracy"])
        if len(margins) < 2 or margins[-1] <= margins[0]:
            reasons.append("overfit requires final reward margin greater than initial margin")
        if accuracy is None or accuracy <= 0.8:
            reasons.append("overfit requires final pair accuracy greater than 0.8")
    else:
        required_eval = ("eval_loss", "eval_reward_margin", "eval_pair_accuracy")
        for name in required_eval:
            if not metric_values[name]:
                reasons.append(f"{mode} requires {name}")
        if mode == "train" and global_step <= 0:
            reasons.append("train requires a non-zero global_step")
        if mode == "validation":
            if eval_pair_count is None:
                reasons.append("validation requires eval_pair_count")
            elif eval_pair_count != counts["validation_pair_count"]:
                reasons.append("validation eval_pair_count does not match dataset manifest")
    if reasons:
        result["status"] = "failed"
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer-state", required=True, type=Path)
    parser.add_argument("--parameter-audit", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("smoke", "overfit", "train", "validation"))
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = analyze_paths(
        args.trainer_state, args.parameter_audit, args.dataset_manifest, mode=args.mode
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    print(f"dpo_gate_result={args.output}")
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
