"""验证并汇总 qa_formality-only smoke/probe 训练产物。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from training.grpo_v3_formality_reward import (
    FORMALITY_COMPONENT,
    FORMALITY_MARGIN_CLIP,
    FORMALITY_REWARD_REVISION,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _read_trace(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / "reward_trace.jsonl"
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_trainer_state(output_dir: Path) -> dict[str, Any]:
    states = [_read_json(path) for path in output_dir.rglob("trainer_state.json")]
    return max(states, key=lambda value: int(value.get("global_step") or 0), default={})


def validate_formality_artifacts(output_dir: Path, *, mode: str) -> dict[str, Any]:
    if mode not in {"smoke", "probe"}:
        raise ValueError("formality mode 只能是 smoke 或 probe")
    rows = _read_trace(output_dir)
    values = [row.get("reward") for row in rows]
    finite = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and math.isfinite(float(value))
    ]
    trainer_state = _latest_trainer_state(output_dir)
    global_step = int(trainer_state.get("global_step") or 0)
    reload_result = _read_json(output_dir / "adapter_reload.json")
    adapter_dirs = [
        path.parent
        for path in output_dir.rglob("adapter_config.json")
        if (path.parent / "adapter_model.safetensors").is_file()
        or (path.parent / "adapter_model.bin").is_file()
    ]
    adapter_paths = {str(path.resolve()) for path in adapter_dirs}
    reload_adapter = str(Path(str(reload_result.get("adapter_dir") or ".")).resolve())
    processor_dir = Path(str(reload_result.get("processor_dir") or output_dir / "processor"))
    processor_configs = list(processor_dir.glob("*processor*config*.json")) + list(
        processor_dir.glob("preprocessor_config.json")
    )
    only_component = all(
        set((row.get("record") or {}).get("reward_components") or {})
        == {FORMALITY_COMPONENT}
        for row in rows
    )
    only_judge = all(
        (
            set((row.get("record") or {}).get("judge_trace") or {})
            == {"qa_formality"}
            if (row.get("record") or {}).get("judge_called") is True
            else (
                (row.get("record") or {}).get("reward_source")
                == "deterministic_unjudgeable_floor"
                and not ((row.get("record") or {}).get("judge_trace") or {})
            )
        )
        for row in rows
    )
    expected_rows = 4 if mode == "smoke" else 160
    expected_steps = 1 if mode == "smoke" else 40
    checks = {
        "exact_trace_count": len(rows) == expected_rows,
        "all_rewards_finite": len(finite) == len(rows) == expected_rows,
        "reward_std_positive": len(finite) >= 2 and statistics.pstdev(finite) > 0,
        "infrastructure_mask_count_zero": all(
            (row.get("record") or {}).get("masked") is not True for row in rows
        ),
        "only_formality_reward_component": only_component,
        "only_formality_judge_called": only_judge,
        "required_global_step": global_step == expected_steps,
        "adapter_complete": bool(adapter_dirs),
        "adapter_reload_passed": (
            reload_result.get("status") == "passed"
            and int(reload_result.get("lora_parameters") or 0) > 0
            and reload_adapter in adapter_paths
        ),
        "processor_config_saved": bool(processor_configs),
        "processor_reload_passed": reload_result.get("processor_reloaded") is True,
    }
    if mode == "probe":
        checks["convergence_metrics_passed"] = (
            _read_json(output_dir / "convergence_metrics.json").get("status") == "passed"
        )
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "grpo_v3_formality_artifacts_v1",
        "mode": mode,
        "status": "passed" if not failed_checks else "failed",
        "checks": checks,
        "failed_checks": failed_checks,
        "trace_count": len(rows),
        "finite_reward_count": len(finite),
        "global_step": global_step,
        "reward_mean": statistics.fmean(finite) if finite else None,
        "reward_std": statistics.pstdev(finite) if len(finite) >= 2 else 0.0,
        "adapter_dir": reload_result.get("adapter_dir"),
        "processor_dir": reload_result.get("processor_dir"),
    }


def summarize_formality_run(
    *,
    output_dir: Path,
    mode: str,
    dataset: Path,
    parent_run: Path,
    policy_model: str,
    reviewer_model: str,
    job_id: str | None,
) -> dict[str, Any]:
    result_name = f"formality_{mode}_result.json"
    result = _read_json(output_dir / result_name)
    rows = _read_trace(output_dir)
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    convergence = _read_json(output_dir / "convergence_metrics.json")
    adapter_reload = _read_json(output_dir / "adapter_reload.json")
    resolved_config = {
        "mode": mode,
        "policy_model": policy_model,
        "reviewer_model": reviewer_model,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "parent_run": str(parent_run.resolve()),
        "reward_revision": FORMALITY_REWARD_REVISION,
        "reward_components": [FORMALITY_COMPONENT],
        "margin_clip": FORMALITY_MARGIN_CLIP,
        "calls_video_reviewer": False,
        "num_generations": 4,
        "temperature": 0.7,
        "learning_rate": 1e-5,
        "lr_scheduler_type": "constant",
        "beta": 0.0,
    }
    manifest = {
        "schema_version": "grpo_v3_formality_run_manifest_v1",
        "mode": mode,
        "status": result.get("status", "unknown"),
        "reward_revision": FORMALITY_REWARD_REVISION,
        "reward_components": [FORMALITY_COMPONENT],
        "margin_clip": FORMALITY_MARGIN_CLIP,
        "calls_video_reviewer": False,
        "policy_input": "native_video",
        "policy_model": policy_model,
        "reviewer_model": reviewer_model,
        "parent_run": str(parent_run.resolve()),
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "slurm_job_id": job_id,
        "trace_count": len(rows),
        "reward_trace": str((output_dir / "reward_trace.jsonl").resolve()),
        "adapter_dir": adapter_reload.get("adapter_dir"),
        "result": result,
        "convergence_metrics": convergence or None,
        "resolved_config": resolved_config,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="验证或汇总 qa_formality-only 产物")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--mode", choices=("smoke", "probe"), required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--mode", choices=("smoke", "probe"), required=True)
    summarize.add_argument("--output-dir", type=Path, required=True)
    summarize.add_argument("--dataset", type=Path, required=True)
    summarize.add_argument("--parent-run", type=Path, required=True)
    summarize.add_argument("--policy-model", required=True)
    summarize.add_argument("--reviewer-model", required=True)
    summarize.add_argument("--job-id")
    args = parser.parse_args()
    if args.command == "validate":
        result = validate_formality_artifacts(args.output_dir, mode=args.mode)
        destination = args.output_dir / f"formality_{args.mode}_result.json"
        destination.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(json.dumps(result, ensure_ascii=False))
        if result["status"] != "passed":
            raise SystemExit(2)
        return
    result = summarize_formality_run(
        output_dir=args.output_dir,
        mode=args.mode,
        dataset=args.dataset,
        parent_run=args.parent_run,
        policy_model=args.policy_model,
        reviewer_model=args.reviewer_model,
        job_id=args.job_id,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
