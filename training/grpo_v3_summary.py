"""生成 GRPO v3 可审计 run manifest。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from training.grpo_v3_contract import DEFAULTS
from training.grpo_v3_data import read_jsonl
from training.grpo_v3_json_format import FORMAT_REWARD_REVISION, summarize_format_traces


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def summarize_run(
    *,
    output_dir: Path,
    gate: int,
    dataset: Path,
    policy_model: str,
    reviewer_model: str | None,
    job_id: str | None,
    parent_run: Path | None,
    eval_dataset: Path | None = None,
    split_manifest: Path | None = None,
    slurm_stdout: Path | None = None,
    slurm_stderr: Path | None = None,
    reviewer_log: Path | None = None,
) -> dict[str, Any]:
    gate_result = _read(output_dir / f"gate{gate}_result.json")
    rows = read_jsonl(dataset)
    eval_rows = read_jsonl(eval_dataset) if eval_dataset else []
    trace_path = output_dir / "reward_trace.jsonl"
    traces = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()] if trace_path.is_file() else []
    rewards = [float(row["reward"]) for row in traces if row.get("reward") is not None and math.isfinite(float(row["reward"]))]
    format_stats = summarize_format_traces(traces)
    reward_revision = {
        0: "none_environment_probe",
        1: "gate1_controlled_non_formal",
        2: FORMAT_REWARD_REVISION,
        3: FORMAT_REWARD_REVISION,
        4: FORMAT_REWARD_REVISION,
    }[gate]
    content_reward_revision = os.environ.get("EGOQA_CONTENT_REWARD_REVISION", "repo_native_v1")
    audit_value = os.environ.get("EGOQA_GROUNDEDNESS_AUDIT_SUMMARY")
    audit_path = Path(audit_value).resolve() if audit_value else None
    audit_sha256 = (
        hashlib.sha256(audit_path.read_bytes()).hexdigest()
        if audit_path is not None and audit_path.is_file()
        else None
    )
    adapter_reload = _read(output_dir / "adapter_reload.json")
    trainer_states = list(output_dir.rglob("trainer_state.json"))
    dependency_path = output_dir / "dependencies.txt"
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    eval_dataset_sha256 = hashlib.sha256(eval_dataset.read_bytes()).hexdigest() if eval_dataset else None
    split_data = _read(split_manifest) if split_manifest else {}
    convergence_metrics = _read(output_dir / "convergence_metrics.json")
    parent_manifest = _read(parent_run / "run_manifest.json") if parent_run else {}
    swift_config_paths = sorted(
        {
            path.resolve()
            for pattern in ("args.json", "*_args.json")
            for path in output_dir.rglob(pattern)
            if path.is_file()
        }
    )
    swift_configs = [
        {"path": str(path), "config": _read(path)} for path in swift_config_paths
    ]
    upstream_gate0_media_metadata = (
        parent_manifest.get("upstream_gate0_media_metadata")
        or (parent_manifest.get("gate_result") or {}).get("media_metadata")
        or gate_result.get("media_metadata")
    )
    logs = {
        "slurm_stdout": str(slurm_stdout.resolve()) if slurm_stdout else None,
        "slurm_stderr": str(slurm_stderr.resolve()) if slurm_stderr else None,
        "reviewer": str(reviewer_log.resolve()) if reviewer_log else None,
    }
    resolved_config = {
        **DEFAULTS.to_dict(),
        "gate": gate,
        "policy_model": policy_model,
        "reviewer_model": reviewer_model,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "eval_dataset": str(eval_dataset.resolve()) if eval_dataset else None,
        "eval_dataset_sha256": eval_dataset_sha256,
        "split_manifest": split_data or None,
        "convergence_metrics": convergence_metrics or None,
        "parent_run": str(parent_run.resolve()) if parent_run else None,
        "video_fps": os.environ.get("FPS"),
        "fps_min_frames": os.environ.get("FPS_MIN_FRAMES"),
        "fps_max_frames": os.environ.get("FPS_MAX_FRAMES"),
        "video_max_pixels": os.environ.get("VIDEO_MAX_PIXELS"),
        "reward_trace": str(trace_path.resolve()) if trace_path.is_file() else None,
        "content_reward_revision": content_reward_revision,
        "groundedness_audit_summary": str(audit_path) if audit_path else None,
        "groundedness_audit_summary_sha256": audit_sha256,
        "adapter_dir": adapter_reload.get("adapter_dir"),
        "trainer_state_paths": [str(path.resolve()) for path in trainer_states],
        "dependencies": str(dependency_path.resolve()) if dependency_path.is_file() else None,
        "upstream_gate0_media_metadata": upstream_gate0_media_metadata,
        "logs": logs,
        "swift_configs": swift_configs,
        **(format_stats if gate >= 2 else {}),
    }
    manifest = {
        "gate": gate,
        "gate_status": gate_result.get("status", "unknown"),
        "formal_reward_result": gate >= 2 and gate_result.get("status") == "passed",
        "policy_input": DEFAULTS.policy_input,
        "policy_model": policy_model,
        "review_model": reviewer_model,
        "train_type": DEFAULTS.train_type,
        "torch_dtype": DEFAULTS.torch_dtype,
        "framework": DEFAULTS.framework,
        "framework_version": DEFAULTS.framework_version,
        "num_generations": DEFAULTS.num_generations,
        "use_policy_vllm": DEFAULTS.use_vllm,
        "freeze_vit": DEFAULTS.freeze_vit,
        "reward_revision": reward_revision,
        "content_reward_revision": content_reward_revision,
        "groundedness_audit_summary": str(audit_path) if audit_path else None,
        "groundedness_audit_summary_sha256": audit_sha256,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": dataset_sha256,
        "eval_dataset": str(eval_dataset.resolve()) if eval_dataset else None,
        "eval_dataset_sha256": eval_dataset_sha256,
        "evidence_ids": [str(row.get("evidence_id")) for row in rows],
        "eval_evidence_ids": [str(row.get("evidence_id")) for row in eval_rows],
        "split_manifest": split_data or None,
        "convergence_metrics": convergence_metrics or None,
        "video_order": [row.get("video_order") for row in rows],
        "video_fps": os.environ.get("FPS"),
        "fps_min_frames": os.environ.get("FPS_MIN_FRAMES"),
        "fps_max_frames": os.environ.get("FPS_MAX_FRAMES"),
        "video_max_pixels": os.environ.get("VIDEO_MAX_PIXELS"),
        "reward_trace_count": len(traces),
        "valid_reward_count": len(rewards),
        "reward_mean": statistics.fmean(rewards) if rewards else None,
        "reward_std": statistics.pstdev(rewards) if len(rewards) >= 2 else None,
        "slurm_job_id": job_id,
        "parent_run": str(parent_run.resolve()) if parent_run else None,
        "adapter_dir": adapter_reload.get("adapter_dir"),
        "trainer_state_paths": [str(path.resolve()) for path in trainer_states],
        "reward_trace": str(trace_path.resolve()) if trace_path.is_file() else None,
        "dependencies": str(dependency_path.resolve()) if dependency_path.is_file() else None,
        "resolved_config": resolved_config,
        "upstream_gate0_media_metadata": upstream_gate0_media_metadata,
        "logs": logs,
        "swift_configs": swift_configs,
        "gate_result": gate_result,
        **(format_stats if gate >= 2 else {}),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_config.json").write_text(
        json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="汇总 GRPO v3 Gate 运行产物")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gate", type=int, choices=(0, 1, 2, 3, 4), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--eval-dataset", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--policy-model", required=True)
    parser.add_argument("--reviewer-model")
    parser.add_argument("--job-id")
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--slurm-stdout", type=Path)
    parser.add_argument("--slurm-stderr", type=Path)
    parser.add_argument("--reviewer-log", type=Path)
    args = parser.parse_args()
    result = summarize_run(
        output_dir=args.output_dir,
        gate=args.gate,
        dataset=args.dataset,
        policy_model=args.policy_model,
        reviewer_model=args.reviewer_model,
        job_id=args.job_id,
        parent_run=args.parent_run,
        eval_dataset=args.eval_dataset,
        split_manifest=args.split_manifest,
        slurm_stdout=args.slurm_stdout,
        slurm_stderr=args.slurm_stderr,
        reviewer_log=args.reviewer_log,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
