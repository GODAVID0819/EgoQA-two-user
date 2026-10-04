"""复用失败作业已完成的数据，在新 allocation 中继续同一训练设置。"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from .workflow import attempt_suffix, training_config, write_json


def phase_sequence(c):
    formal = ("formal", c["formal_max_steps"])
    return [formal] if c.get("skip_smoke") else [("smoke", 1), formal]


def latest_complete_checkpoint(outputs, target_steps):
    required = ("adapter_model.safetensors", "adapter_config.json", "optimizer.pt",
                "scheduler.pt", "rng_state.pth", "args.json", "training_args.bin")
    candidates = []
    for output in outputs:
        for checkpoint in Path(output).glob("swift/*/checkpoint-*"):
            if not checkpoint.is_dir():
                continue
            try:
                step = int(checkpoint.name.removeprefix("checkpoint-"))
                state = json.loads((checkpoint / "trainer_state.json").read_text(encoding="utf-8"))
                if not (0 < step <= target_steps and state.get("global_step") == step
                        and state.get("max_steps") == target_steps):
                    continue
                if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0
                           for name in required):
                    continue
            except (ValueError, OSError, json.JSONDecodeError):
                continue
            candidates.append((step, checkpoint.stat().st_mtime, checkpoint))
    return max(candidates)[2] if candidates else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or Path(c["model"]).name != "Qwen3.8-27B":
        raise RuntimeError("恢复任务必须在 Slurm 中执行，且两个模型均为 Qwen3.8-27B")
    root = Path(c["project_root"])
    suffix = attempt_suffix(c)
    output = root / "outputs" / ("workflow_" + job + suffix)
    output.mkdir(parents=True, exist_ok=False)
    status = {"job_id": job, "step_id": os.environ.get("SLURM_STEP_ID"), "status": "validating_reused_data",
              "data_provenance_job_id": c["data_provenance_job_id"], "prepared_data_root": c["prepared_data_root"],
              "model": c["model"], "judge_mode": "baseline", "run_attempt": c.get("run_attempt"),
              "started_epoch": time.time()}
    write_json(output / "workflow_status.json", status)
    try:
        from .launch import scratch_environment
        from .data import read_rows, validate_splits
        from training.torch_storage_preflight import validate_storage_environment
        base = training_config(c, job_id=job, phase="smoke", max_steps=1)
        scratch, env = scratch_environment(base, job)
        os.environ.update(env)
        os.environ["PATH"] = str(Path(c["ffmpeg"]).parent) + os.pathsep + os.environ["PATH"]
        os.environ["LD_LIBRARY_PATH"] = str(Path(c["ffmpeg"]).parent.parent / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
        storage = validate_storage_environment(allowed_root=scratch, environ=os.environ)
        write_json(output / "storage_preflight.json", storage)
        if storage["status"] != "passed":
            raise RuntimeError("存储预检失败")
        source = Path(c["prepared_data_root"])
        expected = c.get("expected_split_counts", {"train":12,"validation":6,"test":6})
        splits = {name: read_rows(source / (name + ".jsonl")) for name in expected}
        validate_splits(splits)
        if {k: len(v) for k, v in splits.items()} != expected:
            raise ValueError(f"恢复数据计数不一致，期望{expected}")
        status["data"] = {"rows": {k: len(v) for k, v in splits.items()}, "reused_without_preprocessing": True}
        for phase, steps in phase_sequence({**c, "formal_max_steps": c.get("formal_max_steps", len(splits["train"]))}):
            path = root / "configs" / f"{phase}_{job}_27b{suffix}.json"
            write_json(path, training_config(c, job_id=job, phase=phase, max_steps=steps))
            status.update(status="running", phase=phase, run_config=str(path))
            write_json(output / "workflow_status.json", status)
            subprocess.run([c["train_python"], "-m", "training.grpo_v3.six_user_binary.launch", "--config", str(path)],
                           cwd=root / "multi-user", check=True)
        status.update(status="completed", finished_epoch=time.time())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_epoch=time.time())
        raise
    finally:
        write_json(output / "workflow_status.json", status)


if __name__ == "__main__":
    main()
