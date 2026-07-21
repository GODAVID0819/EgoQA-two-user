"""验证 Gate 1/2 训练产物，不把 smoke 误报为收敛。"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from training.grpo_v3_json_format import summarize_format_traces


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _latest_trainer_state(output_dir: Path) -> dict[str, Any]:
    states = [_json(path) for path in output_dir.rglob("trainer_state.json")]
    return max(states, key=lambda value: int(value.get("global_step") or 0), default={})


def validate_gate_artifacts(output_dir: Path, *, gate: int) -> dict[str, Any]:
    if gate not in {1, 2, 3, 4}:
        raise ValueError("产物验证器只接受 Gate 1/2/3/4")
    trainer_state = _latest_trainer_state(output_dir)
    reload_result = _json(output_dir / "adapter_reload.json")
    trace_path = output_dir / "reward_trace.jsonl"
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()] if trace_path.is_file() else []
    values = [row.get("reward") for row in traces]
    valid = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    masked_records = [
        row for row in traces
        if isinstance(row.get("record"), dict) and row["record"].get("masked") is True
    ]
    format_stats = summarize_format_traces(traces)
    format_trace_count = (
        format_stats["format_raw_valid_count"]
        + format_stats["format_repaired_count"]
        + format_stats["format_unrecoverable_count"]
    )
    reward_std = statistics.pstdev(valid) if len(valid) >= 2 else 0.0
    adapter_configs = list(output_dir.rglob("adapter_config.json"))
    complete_adapters = [
        path.parent
        for path in adapter_configs
        if (path.parent / "adapter_model.safetensors").is_file()
        or (path.parent / "adapter_model.bin").is_file()
    ]
    complete_adapter_paths = {str(path.resolve()) for path in complete_adapters}
    reloaded_adapter = str(Path(str(reload_result.get("adapter_dir") or ".")).resolve())
    processor_dir = Path(str(reload_result.get("processor_dir") or output_dir / "processor"))
    processor_configs = list(processor_dir.glob("*processor*config*.json")) + list(
        processor_dir.glob("preprocessor_config.json")
    )
    checks = {
        "global_step_at_least_1": int(trainer_state.get("global_step") or 0) >= 1,
        "at_least_4_traces": len(traces) >= 4,
        "at_least_2_valid_rewards": len(valid) >= 2,
        "reward_std_positive": reward_std > 0,
        "no_masked_rewards": len(valid) == len(values) and not masked_records,
        "adapter_complete": bool(complete_adapters),
        "adapter_reload_passed": (
            reload_result.get("status") == "passed"
            and int(reload_result.get("lora_parameters") or 0) > 0
            and reloaded_adapter in complete_adapter_paths
        ),
        "processor_config_saved": bool(processor_configs),
        "processor_reload_passed": reload_result.get("processor_reloaded") is True,
    }
    if gate == 2:
        checks["four_finite_rewards"] = len(traces) == 4 and len(valid) == 4
        checks["format_trace_complete"] = format_trace_count == len(traces)
    if gate in {3, 4}:
        convergence = _json(output_dir / "convergence_metrics.json")
        required_steps = 20 if gate == 3 else 40
        checks["convergence_metrics_passed"] = convergence.get("status") == "passed"
        checks["required_global_steps"] = int(trainer_state.get("global_step") or 0) >= required_steps
    if gate == 4:
        split = _json(output_dir / "split_manifest.json")
        train_ids = {str(item) for item in split.get("train_evidence_ids") or []}
        eval_ids = {str(item) for item in split.get("eval_evidence_ids") or []}
        checks["split_counts_40_10"] = (
            int(split.get("train_count") or 0) == 40
            and int(split.get("eval_count") or 0) == 10
            and len(train_ids) == 40
            and len(eval_ids) == 10
        )
        checks["split_ids_disjoint"] = not (train_ids & eval_ids) and int(split.get("intersection_count") or 0) == 0
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "gate": gate,
        "status": "passed" if not failed else "failed",
        "checks": checks,
        "failed_checks": failed,
        "trace_count": len(traces),
        "valid_reward_count": len(valid),
        "masked_reward_count": len(values) - len(valid),
        "reward_std": reward_std,
        "global_step": int(trainer_state.get("global_step") or 0),
        **format_stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="验证 GRPO v3 Gate 1/2/3/4 产物")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    result = validate_gate_artifacts(args.output_dir, gate=args.gate)
    destination = args.result or args.output_dir / f"gate{args.gate}_result.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
