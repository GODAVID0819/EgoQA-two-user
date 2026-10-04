"""在同一 allocation 内复用成片、生成六用户帧数据并启动 27B GRPO。"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path, PurePosixPath
import subprocess
import time


def unique_windows(rows):
    unique, signatures = {}, {}
    for row in rows:
        key = (row["day"], row["time_token"])
        clips = sorted(row["clips"], key=lambda c: c["agent_dir"])
        if len(clips) != 6 or len({c["agent_dir"] for c in clips}) != 6:
            raise ValueError("每个窗口必须包含六名不同用户")
        signature = [(c["agent_dir"], c.get("full_local_video") or c.get("local_video")) for c in clips]
        if key in signatures and signatures[key] != signature:
            raise ValueError("同一窗口的成片映射不一致")
        signatures[key] = signature
        unique.setdefault(key, {**row, "clips": clips, "duration_seconds": 600.})
    return list(unique.values())


def attempt_suffix(workflow):
    attempt = workflow.get("run_attempt")
    if not attempt:
        return ""
    if not isinstance(attempt, str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,31}", attempt):
        raise ValueError("run_attempt 必须是安全的短标识")
    return "_" + attempt


def training_config(workflow, *, job_id, phase, max_steps):
    root = PurePosixPath(workflow["project_root"])
    suffix = attempt_suffix(workflow)
    data = PurePosixPath(workflow["prepared_data_root"]) if workflow.get("prepared_data_root") else root / "data" / ("grpo_" + job_id)
    result = {"project_root": str(root), "train_python": workflow["train_python"],
        "judge_python": workflow["judge_python"], "policy_model": workflow["model"],
        "judge_config": str(root / "configs/judge_baseline_27b.json"),
        "train_dataset": str(data / ("smoke_train.jsonl" if phase == "smoke" else "train.jsonl")),
        "val_dataset": str(data / ("smoke_validation.jsonl" if phase == "smoke" else "validation.jsonl")),
        "output_root": str(root / "outputs" / (phase + suffix)), "scratch_root": "/scratch/xl6775/job_scratch",
        "allocation_manifest": str(root / "outputs" / ("allocation_" + job_id) / "allocation.json"),
        "reward_mode": "continuous", "policy_gpu_indices": [0], "judge_gpu_indices": [1],
        "walltime_seconds": workflow.get("walltime_seconds", 86400), "keeper_reserve_gib": 8., "keeper_max_prealloc_gib": .25,
        "max_steps": max_steps, "num_generations": 4, "num_generations_eval": 4,
        "per_device_train_batch_size": 1, "gradient_accumulation_steps": 4, "per_device_eval_batch_size": 4,
        "max_length": 65536, "max_completion_length": workflow.get("max_completion_length", 1024), "max_pixels": 24576,
        "judge_port": 8766, "learning_rate": 1e-5, "beta": .04, "temperature": .85,
        "top_p": .95, "top_k": 40, "lora_rank": 8, "lora_alpha": 16}
    for key in ("use_vllm", "attn_impl", "vllm_gpu_memory_utilization", "acceleration_packages", "compiler_environment", "policy_cuda_home", "policy_image_cache_gb"):
        if key in workflow:
            result[key] = workflow[key]
    if workflow.get("judge_config"):
        result["judge_config"] = workflow["judge_config"]
    if workflow.get('shared_gpu'):
        result.update(shared_gpu=True, policy_gpu_indices=[0], judge_gpu_indices=[0])
        if phase == 'smoke':
            # 新显存复用边界只做一次完整最小运行，并实际重载其adapter。
            result.update(paired_validation=True, baseline_after_training=True)
    if workflow.get("execution_mode") == "direct":
        result["execution_mode"] = "direct"
        result.pop("allocation_manifest", None)
    if phase == "formal":
        for key in ("eval_steps", "save_steps", "save_total_limit", "paired_validation", "validation_seed",
                    "resume_from_checkpoint", "baseline_validation_source", "baseline_after_training"):
            if key in workflow:
                result[key] = workflow[key]
    return result


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    job = os.environ.get("SLURM_JOB_ID", "")
    if not job.isdigit() or Path(c["model"]).name != "Qwen3.8-27B":
        raise RuntimeError("必须在 Slurm 中运行且 policy / Judge 均使用已确认的 Qwen3.8-27B")
    root = Path(c["project_root"])
    output = root / "outputs" / ("workflow_" + job)
    output.mkdir(parents=True, exist_ok=True)
    status = {"job_id": job, "step_id": os.environ.get("SLURM_STEP_ID"), "model": c["model"],
              "judge_mode": "baseline", "status": "preparing", "started_epoch": time.time()}
    from .launch import scratch_environment, allocation_keeper
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
    status["allocation_keeper"] = allocation_keeper(base, job)
    write_json(output / "workflow_status.json", status)
    try:
        from .data import read_rows, make_row, validate_splits
        from training.judge_sft.prepare_real_data import _repo_module
        prep = _repo_module("rlhf_evidence_preprocessing")
        windows = unique_windows(read_rows(c["candidates"]))
        if len(windows) != 4:
            raise ValueError(f"本轮已核验四个独立窗口，实际为 {len(windows)}")
        dataset = root / "data" / ("frames_" + job)
        cache = root / "data" / ("frame_cache_" + job)
        config = prep.build_preprocessing_config(clip_model_id=c["clip_model"])
        fingerprint = prep._ensure_dataset_metadata(dataset, config)
        encoder = prep.LazyBatchedImageEncoder(c["clip_model"], device="cuda:0", batch_size=64)
        prepared = []
        for window in windows:
            for clip in window["clips"]:
                video = Path(clip.get("full_local_video") or clip["local_video"])
                if not video.is_file() or not video.stat().st_size:
                    raise FileNotFoundError(video)
                probe = subprocess.check_output([c["ffprobe"], "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)], text=True)
                duration = float(json.loads(probe)["format"]["duration"])
                if not 599 <= duration <= 601:
                    raise ValueError(f"非完整十分钟成片：{video} duration={duration}")
            window["selection"] = {"source_manifest": c["candidates"],
                "stitched_paths": [x.get("full_local_video") or x["local_video"] for x in window["clips"]]}
            prepared.append(prep.prepare_packet(window, dataset_root=dataset, cache_dir=cache,
                config=config, config_fingerprint=fingerprint, encoder=encoder, device="cuda:0",
                clip_batch_size=64, ffmpeg_binary=c["ffmpeg"], materialized_packet=window))
        prep.rebuild_dataset_index(dataset)
        del encoder
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
        target = root / "data" / ("grpo_" + job)
        target.mkdir(parents=True, exist_ok=True)
        splits = {"train": prepared[:2] + prepared[3:], "validation": prepared[2:3]}
        rows = {name: [make_row(dataset, item["packet_id"], asker) for item in packets for asker in range(6)]
                for name, packets in splits.items()}
        validate_splits(rows)
        selection = {name: [{"source_packet_id": row["source_packet_id"], "asker_index": row["asker_index"]} for row in values] for name, values in rows.items()}
        write_json(target / "split_manifest.json", selection)
        for name, values in {**rows, "smoke_train": rows["train"][:1], "smoke_validation": rows["validation"][:1]}.items():
            with (target / (name + ".jsonl")).open("x", encoding="utf-8") as stream:
                for value in values:
                    stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        status["data"] = {"source_windows": 4, "videos": 24, "rows": {k: len(v) for k, v in rows.items()},
                          "split_manifest": str(target / "split_manifest.json")}
        write_json(root / "configs/judge_baseline_27b.json", {"model_id": c["model"], "judge_mode": "baseline",
            "tensor_parallel_size": 1, "max_input_tokens": 262144, "max_pixels": 262144,
            "min_pixels": 3136, "gpu_memory_utilization": .85})
        for phase, steps in (("smoke", 1), ("formal", c.get("formal_max_steps", 60))):
            config_path = root / "configs" / f"{phase}_{job}_27b.json"
            write_json(config_path, training_config(c, job_id=job, phase=phase, max_steps=steps))
            status.update(status="running", phase=phase, run_config=str(config_path))
            write_json(output / "workflow_status.json", status)
            subprocess.run([c["train_python"], "-m", "training.grpo_v3.six_user_binary.launch", "--config", str(config_path)],
                           cwd=root / "multi-user", check=True)
        status.update(status="completed", finished_epoch=time.time())
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}", finished_epoch=time.time())
        raise
    finally:
        write_json(output / "workflow_status.json", status)


if __name__ == "__main__":
    main()
