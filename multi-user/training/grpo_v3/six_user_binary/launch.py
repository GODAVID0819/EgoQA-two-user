"""在一个 Slurm 作业内运行冻结 Judge 和在线 GRPO，资源由实际配置提供。"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path, PurePosixPath
import signal
import shutil
import socket
import subprocess
import time
import uuid


def validate_config(c):
    if c.get('policy_allocator_config') not in (None, 'expandable_segments:True'):
        raise ValueError('Policy分配器只支持已核验的expandable_segments:True')
    from .policy_image_cache import configured_cache_bytes
    configured_cache_bytes(c.get('policy_image_cache_gb'))
    modules = c.get('lora_target_modules', ['q_proj', 'v_proj'])
    if (not isinstance(modules, list) or not modules or any(not isinstance(v, str) for v in modules)
            or len(set(modules)) != len(modules)
            or set(modules) not in ({'q_proj', 'v_proj'}, {'q_proj', 'v_proj', 'in_proj_qkv'})):
        raise ValueError('LoRA覆盖必须为原q_proj/v_proj或已确认的q_proj/v_proj/in_proj_qkv')
    if c.get('utilization_guard'):
        from .utilization_guard import checked_config
        checked_config(c['utilization_guard'])
    for key in ("project_root", "train_python", "judge_python", "policy_model", "judge_config",
                "train_dataset", "val_dataset", "output_root", "scratch_root"):
        if not isinstance(c.get(key), str) or not PurePosixPath(c[key]).is_absolute():
            raise ValueError(f"{key} 必须是 Torch 上实际绝对路径")
    for key in ("output_root", "scratch_root"):
        if not PurePosixPath(c[key]).is_relative_to("/scratch"):
            raise ValueError(f"{key} 必须位于 /scratch")
    if c.get("policy_cuda_home") and not PurePosixPath(c["policy_cuda_home"]).is_relative_to("/scratch"):
        raise ValueError("Policy CUDA工具必须位于scratch")
    if c.get("early_keeper_until_seconds") is not None:
        raise ValueError("训练入口不允许配置人为 GPU 空转负载")
    if c.get("reward_mode") not in {"continuous", "binary", "formality", "groundedness"}:
        raise ValueError("必须明确 reward_mode")
    policy, judge = c.get("policy_gpu_indices", []), c.get("judge_gpu_indices", [])
    if len(policy) != 1 or not judge or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in policy + judge):
        raise ValueError("当前入口要求一个 policy GPU，至少一个 Judge GPU；索引相对本次分配")
    if c.get('shared_gpu'):
        if policy != [0] or judge != [0] or not c.get('use_vllm'):
            raise ValueError('共享模式必须明确使用单GPU索引0和vLLM')
    elif len(set(policy + judge)) != len(policy + judge):
        raise ValueError("policy 与 Judge GPU 必须互不重叠")
    for key in ("walltime_seconds", "max_steps", "num_generations", "num_generations_eval", "max_length", "max_completion_length", "max_pixels", "judge_port"):
        if isinstance(c.get(key), bool) or not isinstance(c.get(key), int) or c[key] <= 0:
            raise ValueError(f"{key} 必须显式提供正整数")
    if c["num_generations"] < 2 or c["num_generations_eval"] < 2:
        raise ValueError("GRPO 每组至少两次生成")
    batch = c.get("per_device_train_batch_size", c["num_generations"])
    accumulation = c.get("gradient_accumulation_steps", 1)
    evaluation_batch = c.get("per_device_eval_batch_size", c["num_generations_eval"])
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (batch, accumulation, evaluation_batch)):
        raise ValueError("batch 和 gradient_accumulation_steps 必须是正整数")
    if batch * accumulation % c["num_generations"] or evaluation_batch % c["num_generations_eval"]:
        raise ValueError("有效 batch 必须能被对应生成组大小整除")
    for key in ("learning_rate", "temperature", "top_p"):
        if not isinstance(c.get(key), (int, float)) or not math.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f"{key} 必须显式提供有限正数")
    if not isinstance(c.get("beta"), (int, float)) or not math.isfinite(c["beta"]) or c["beta"] < 0:
        raise ValueError("beta 必须显式提供非负有限数")
    if not 0 < c["top_p"] <= 1 or not isinstance(c.get("top_k"), int):
        raise ValueError("top_p 或 top_k 不合法")
    stop = c.get("stop_after_steps", c["max_steps"])
    if isinstance(stop, bool) or not isinstance(stop, int) or not 0 < stop <= c["max_steps"]:
        raise ValueError("分阶段终点必须是正整数且不超过调度总步数")
    warmup = c.get("warmup_steps", 0)
    if isinstance(warmup, bool) or not isinstance(warmup, int) or not 0 <= warmup < c["max_steps"]:
        raise ValueError("预热步数必须小于调度总步数")
    if c.get("lr_scheduler_type", "constant") not in {"constant", "linear", "cosine", "cosine_with_min_lr"}:
        raise ValueError("不支持的学习率调度方式")
    scheduler_kwargs = c.get("lr_scheduler_kwargs", {})
    if not isinstance(scheduler_kwargs, dict):
        raise ValueError("学习率调度附加参数必须是字典")
    if c.get("lr_scheduler_type") == "cosine_with_min_lr":
        rate = scheduler_kwargs.get("min_lr_rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate < 1:
            raise ValueError("余弦学习率下限比例必须位于[0,1)")


def swift_command(c, output):
    validate_config(c)
    package = PurePosixPath(c["project_root"]) / "multi-user"
    cmd = [str(PurePosixPath(c["train_python"]).parent / "swift"), "rlhf"]
    options = {"rlhf_type": "grpo", "advantage_estimator": "grpo", "scale_rewards": "group",
        "model": c["policy_model"], "dataset": c["train_dataset"], "enable_thinking": "false", "attn_impl": c.get("attn_impl", "sdpa"),
        "val_dataset": c["val_dataset"], "external_plugins": str(package / "training/grpo_v3/six_user_binary/plugin.py"),
        "reward_funcs": "egoqa_six_user_binary_v1", "tuner_type": "lora", "torch_dtype": "bfloat16",
        "freeze_vit": "true", "freeze_aligner": "true", "lora_rank": c.get("lora_rank", 8),
        "lora_alpha": c.get("lora_alpha", 16), "use_vllm": str(c.get("use_vllm", False)).lower(), "num_generations": c["num_generations"],
        "num_generations_eval": c["num_generations_eval"], "per_device_train_batch_size": c.get("per_device_train_batch_size", c["num_generations"]),
        "per_device_eval_batch_size": c.get("per_device_eval_batch_size", c["num_generations_eval"]), "gradient_accumulation_steps": c.get("gradient_accumulation_steps", 1),
        "gradient_checkpointing": "true", "max_steps": c["max_steps"], "seed": c.get("seed", 42), "data_seed": c.get("data_seed", 42),
        "max_length": c["max_length"], "max_completion_length": c["max_completion_length"], "max_pixels": c["max_pixels"],
        "learning_rate": c["learning_rate"], "beta": c["beta"], "temperature": c["temperature"],
        "top_p": c["top_p"], "top_k": c["top_k"], "lr_scheduler_type": c.get("lr_scheduler_type", "constant"),
        "save_strategy": "steps", "save_steps": c.get("save_steps", c["max_steps"]),
        "eval_strategy": "steps", "eval_steps": c.get("eval_steps", c["max_steps"]),
        "save_total_limit": c.get("save_total_limit", 2), "logging_steps": 1, "log_completions": "true", "dataset_shuffle": "true",
        "split_dataset_ratio": 0, "dataset_num_proc": 1, "dataloader_num_workers": 0,
        "remove_unused_columns": "false", "strict": "true", "truncation_strategy": "delete",
        "report_to": "none", "output_dir": str(PurePosixPath(output) / "swift")}
    if c.get("use_vllm"):
        options.update(vllm_mode="colocate", vllm_tensor_parallel_size=1,
            vllm_enable_lora="true", sleep_level=1, offload_model="true", offload_optimizer="true",
            vllm_gpu_memory_utilization=c.get("vllm_gpu_memory_utilization", .55),
            vllm_max_model_len=c["max_length"], vllm_max_num_seqs=c["num_generations"],
            vllm_enable_prefix_caching="true", vllm_enforce_eager="false",
            vllm_mm_processor_cache_gb=4,
            vllm_limit_mm_per_prompt=json.dumps({"image": 1800}),
            vllm_engine_kwargs=json.dumps({"enable_chunked_prefill": True, "max_num_batched_tokens": 8192,
                "additional_config": {"gdn_prefill_backend": "triton"}}))
    if c.get("resume_from_checkpoint"):
        options["resume_from_checkpoint"] = c["resume_from_checkpoint"]
    if "warmup_steps" in c:
        options["warmup_steps"] = c["warmup_steps"]
    if "lr_scheduler_kwargs" in c:
        options["lr_scheduler_kwargs"] = json.dumps(c["lr_scheduler_kwargs"])
    if "stop_after_steps" in c:
        options["callbacks"] = "egoqa_stage_stop"
    for key, value in options.items():
        cmd.extend(["--" + key, str(value)])
    return cmd + ["--target_modules", *c.get('lora_target_modules', ['q_proj', 'v_proj'])]


def checked_baseline_source(path, judge_health, candidates_per_input):
    from .evaluation import same_frozen_judge
    path = Path(path)
    baseline = json.loads(path.read_text(encoding="utf-8"))
    inputs, rows = baseline.get("input_bindings", []), baseline.get("rows", [])
    if (baseline.get("status") != "completed" or baseline.get("adapter") is not None
            or not inputs or len(rows) != len(inputs) * candidates_per_input
            or not same_frozen_judge(baseline.get("judge", {}), judge_health)):
        raise ValueError("外部baseline必须完整且与当前冻结Judge一致")
    return path


def final_validation(c, output, evaluate):
    if not c.get('paired_validation'):
        return
    if c.get('baseline_after_training') and not c.get('baseline_validation_source'):
        evaluate()
    checked = json.loads((Path(output) / 'training_result.json').read_text(encoding='utf-8'))
    evaluate(checked['checkpoint'])


def run_validation_only(c, evaluate):
    """只恢复已保存adapter的固定评分，不重新执行优化器更新。"""
    adapter = c.get('validation_only_adapter')
    if not adapter:
        return False
    if not c.get('paired_validation') or not c.get('baseline_validation_source'):
        raise ValueError('独立评分恢复必须沿用已有完整基座配对验证')
    evaluate(adapter)
    return True


def scratch_environment(c, job_id):
    from training.torch_storage_preflight import REQUIRED_STORAGE_VARIABLES
    scratch = Path(c["scratch_root"]) / ("six_user_grpo_" + job_id)
    env = os.environ.copy()
    for name in REQUIRED_STORAGE_VARIABLES + ("PIP_CACHE_DIR",):
        env[name] = str(scratch / name.lower())
    env.update(PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", VLLM_NO_USAGE_STATS="1",
               PYTHONPATH=str(Path(c["project_root"]) / "multi-user"))
    env.update(c.get("compiler_environment", {}))
    return scratch, env


def role_environment(python, base, *, allocator_config=None):
    # 绝对 Python 路径不会激活环境；ninja 等子进程仍通过 PATH 寻找。
    env = {**base, "PATH": str(PurePosixPath(python).parent) + os.pathsep + base.get("PATH", ""),
           "VLLM_USE_FLASHINFER_SAMPLER": "0"}
    if allocator_config is not None:
        if allocator_config != 'expandable_segments:True':
            raise ValueError('未核验的Policy分配器配置')
        # Swift按旧变量启用原生分阶段切换；避免新别名优先级遮蔽该切换。
        env.pop('PYTORCH_ALLOC_CONF', None)
        env['PYTORCH_CUDA_ALLOC_CONF'] = allocator_config
    return env


def select_master_port():
    """由当前计算节点选择可绑定的 TCP 端口，避免共享节点的 29500 冲突。"""
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as candidate:
            candidate.bind(("0.0.0.0", 0))
            port = candidate.getsockname()[1]
        if port != 29500:
            return port


def cleanup_processes(children, *, kill_group=None):
    kill_group = kill_group or os.killpg
    errors = []
    for process in reversed(children):
        try:
            # launcher 先退出时，vLLM worker 仍可能留在原进程组中占用显存。
            try:
                kill_group(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    kill_group(process.pid, getattr(signal, "SIGKILL", 9))
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            try:
                kill_group(process.pid, getattr(signal, "SIGKILL", 9))
            except ProcessLookupError:
                pass
        except Exception as exc:
            errors.append(f"pid={process.pid}: {type(exc).__name__}: {exc}")
    return errors


def validate_allocation_keeper(record, *, job_id, hostname, command):
    if str(record.get("job_id")) != str(job_id) or record.get("hostname") != hostname:
        raise ValueError("共享 keeper 不属于当前 allocation / 主机")
    if record.get("start_after_seconds") != 7200 or record.get("keeper_script") not in command:
        raise ValueError("共享 keeper 的启动参数或进程身份不一致")


def allocation_keeper(c, job_id):
    record = json.loads(Path(c["allocation_manifest"]).read_text(encoding="utf-8"))
    hostname = socket.gethostname()
    validate_allocation_keeper(record, job_id=job_id, hostname=hostname, command=[record.get("keeper_script")])
    pid = int(record["keeper_pid"])
    try:
        os.kill(pid, 0)
        command = Path(f"/proc/{pid}/cmdline").read_bytes().decode().split("\x00")
    except (ProcessLookupError, FileNotFoundError, PermissionError) as exc:
        return {**record, "status": "not_running", "reason": type(exc).__name__}
    if record["keeper_script"] not in command:
        return {**record, "status": "not_running", "reason": "keeper_pid_reused"}
    return {**record, "status": "running"}


def main():
    parser = argparse.ArgumentParser(description="六用户在线 GRPO 作业驱动")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(c)
    if c.get("execution_mode") != "direct":
        raise ValueError("当前训练必须由直接执行的批处理作业启动，不能依赖独立资源保留作业")
    # 共享节点上的固定 HTTP 端口可能已占用，采用同一套已验证的空闲端口选择。
    c["judge_port"] = select_master_port()
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("本入口只在已分配的 Slurm 作业内运行")
    allocation = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if not all(allocation) or max(c["policy_gpu_indices"] + c["judge_gpu_indices"]) >= len(allocation):
        raise RuntimeError("分配 GPU 与配置索引不匹配")
    if c.get('shared_gpu') and len(allocation) != 1:
        raise RuntimeError('共享模式只能申请一张GPU')
    package = Path(c["project_root"]) / "multi-user"
    output = Path(c["output_root"]) / ("train_" + job_id)
    output.mkdir(parents=True, exist_ok=False)
    scratch, env = scratch_environment(c, job_id)
    from training.torch_storage_preflight import validate_storage_environment
    storage = validate_storage_environment(allowed_root=scratch, environ=env)
    (output / "storage_preflight.json").write_text(json.dumps(storage, indent=2), encoding="utf-8")
    if storage["status"] != "passed":
        raise RuntimeError("scratch 存储预检失败")
    (output / "run_config.json").write_text(json.dumps(c, indent=2), encoding="utf-8")
    children, logs = [], []
    result = {"status": "running", "job_id": job_id, "config": c}
    def child(command, name, child_env):
        stream = (output / name).open("w", encoding="utf-8")
        logs.append(stream)
        process = subprocess.Popen(command, cwd=package, env=child_env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        children.append(process)
        return process
    def interrupted(_signum, _frame):
        raise RuntimeError("作业收到终止信号")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        subprocess.run([c["train_python"], "-m", "training.grpo_v3.six_user_binary.data", "--validate", c["train_dataset"], c["val_dataset"]], cwd=package, env=env, check=True)
        tool_audit = {}
        for role in ("train", "judge"):
            role_env = role_environment(c[role + "_python"], env)
            tool_audit[role] = {name: shutil.which(name, path=role_env["PATH"]) for name in ("ninja", "gcc", "g++", "nvcc", "ffmpeg")}
            with (output / f"{role}_dependencies.txt").open("w") as stream:
                subprocess.run([c[role + "_python"], "-m", "pip", "freeze"], cwd=package, env=role_env, stdout=stream, check=True)
        (output / "runtime_tools.json").write_text(json.dumps(tool_audit, indent=2), encoding="utf-8")
        child(["nvidia-smi", "--id=" + ",".join(allocation),
            "--query-gpu=timestamp,index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv", "-l", "1"], "gpu_metrics.csv", env)
        startup_guard = None
        guard_environment = {}
        if c.get('utilization_guard'):
            if not c.get('shared_gpu'):
                raise ValueError('当前利用率保护交接只适用于已验证的单GPU模式')
            guard_config = json.dumps(c['utilization_guard'])
            handoff = str(output/'utilization_handoff.signal')
            guard_environment = {'EGOQA_UTILIZATION_GUARD': guard_config,
                'EGOQA_UTILIZATION_HANDOFF': handoff,
                'EGOQA_UTILIZATION_LOG': str(output/'utilization_trainer.jsonl'),
                'EGOQA_UTILIZATION_METRICS': str(output/'gpu_metrics.csv')}
            guard_env = {**role_environment(c['train_python'],env),
                **guard_environment, 'CUDA_VISIBLE_DEVICES': allocation[0]}
            guard_env['PYTHONPATH'] += os.pathsep+c['project_root']
            startup_guard = child([c['train_python'],'-m','training.grpo_v3.six_user_binary.utilization_runtime',
                '--config',guard_config,'--log',str(output/'utilization_startup.jsonl'),
                '--handoff',handoff],'utilization_startup.log',guard_env)
        judge_config = json.loads(Path(c["judge_config"]).read_text(encoding="utf-8"))
        judge_config['shared_gpu'] = bool(c.get('shared_gpu'))
        if judge_config.get("tensor_parallel_size", 1) != len(c["judge_gpu_indices"]):
            raise ValueError("Judge TP 与实际 GPU 数不一致")
        (output / "judge_config.json").write_text(json.dumps(judge_config, indent=2), encoding="utf-8")
        judge_env = {**role_environment(c["judge_python"], env), "CUDA_VISIBLE_DEVICES": ",".join(allocation[i] for i in c["judge_gpu_indices"])}
        instance_id = uuid.uuid4().hex
        service = child([c["judge_python"], "-m", "training.grpo_v3.six_user_binary.service", "--config", str(output / "judge_config.json"), "--port", str(c["judge_port"]), "--instance-id", instance_id], "judge_service.log", judge_env)
        from .service import JudgeClient
        client = JudgeClient(f"http://127.0.0.1:{c['judge_port']}", timeout_seconds=2., expected_instance=instance_id)
        deadline = time.monotonic() + c.get("judge_startup_timeout_seconds", 900)
        while True:
            if startup_guard is not None and startup_guard.poll() not in (None,0):
                raise RuntimeError('启动期利用率保护退出，见utilization_startup.log')
            if service.poll() is not None:
                raise RuntimeError("Judge 服务提前退出，见 judge_service.log")
            try:
                health = client.health()
                break
            except (OSError, RuntimeError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("Judge 服务启动超时")
                time.sleep(2.)
        (output / "judge_health.json").write_text(json.dumps(health, indent=2), encoding="utf-8")
        policy_env = {**role_environment(c["train_python"], env, allocator_config=c.get('policy_allocator_config')), "CUDA_VISIBLE_DEVICES": allocation[c["policy_gpu_indices"][0]],
            "EGOQA_GRPO_STOP_AT_STEP": str(c.get("stop_after_steps", c["max_steps"])),
            "EGOQA_GRPO_STAGE_AUDIT": str(output / "stage_state.json"),
            "EGOQA_POLICY_IMAGE_CACHE_GB": str(c.get("policy_image_cache_gb", 0)),
            "EGOQA_SHARED_GPU": '1' if c.get('shared_gpu') else '0',
            "EGOQA_SHARED_GPU_TRACE": str(output / 'gpu_phase_trace.jsonl'),
            "NPROC_PER_NODE": "1", "EGOQA_SIX_USER_JUDGE_URL": client.base_url,
            "EGOQA_SIX_USER_JUDGE_INSTANCE": instance_id,
            "EGOQA_SIX_USER_REWARD_MODE": c["reward_mode"], "EGOQA_GRPO_V3_REWARD_TRACE": str(output / "reward_trace.jsonl")}
        policy_env['EGOQA_QWEN_PARTIAL_PACKED_LORA'] = (
            '1' if 'in_proj_qkv' in c.get('lora_target_modules', []) else '0')
        if guard_environment:
            policy_env.update(guard_environment)
            policy_env['PYTHONPATH'] += os.pathsep+c['project_root']
        if c.get("acceleration_packages"):
            policy_env["PYTHONPATH"] += os.pathsep + c["acceleration_packages"]
        if c.get("policy_cuda_home"):
            cuda_home = Path(c["policy_cuda_home"])
            if not (cuda_home / "bin/nvcc").is_file() or not (cuda_home / "include/cuda_runtime.h").is_file():
                raise FileNotFoundError("Policy CUDA工具链不完整：" + str(cuda_home))
            policy_env["CUDA_HOME"] = str(cuda_home)
            policy_env["CUDA_PATH"] = str(cuda_home)
        baseline_source = (checked_baseline_source(c["baseline_validation_source"], health, c["num_generations_eval"])
            if c.get("baseline_validation_source") else output / "validation_baseline.json")
        def evaluate_policy(adapter=None):
            label = "policy" if adapter else "baseline"
            destination = output / ("validation_" + label + ".json")
            eval_command = [c["train_python"], "-m", "training.grpo_v3.six_user_binary.evaluation",
                "--config", str(output / "run_config.json"), "--output", str(destination),
                "--judge-url", client.base_url, "--judge-instance", instance_id]
            if adapter:
                eval_command += ["--adapter", str(adapter), "--baseline", str(baseline_source)]
            process = child(eval_command, "validation_" + label + ".log", policy_env)
            while process.poll() is None:
                if service.poll() is not None:
                    raise RuntimeError("固定验证期间 Judge 服务退出")
                time.sleep(2.)
            errors = cleanup_processes([process])
            children.remove(process)
            if process.returncode or errors:
                raise RuntimeError(f"固定验证 {label} 失败：exit={process.returncode}, cleanup={errors}")
        if run_validation_only(c, evaluate_policy):
            result.update(status='completed', phase='validation_only',
                          source_checkpoint=c['validation_only_adapter'])
            return
        if c.get("paired_validation") and not c.get("baseline_validation_source") and not c.get("baseline_after_training"):
            evaluate_policy()
        selected_port = select_master_port()
        policy_env.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(selected_port))
        (output / "policy_rendezvous.json").write_text(json.dumps({"master_addr": "127.0.0.1",
            "master_port": selected_port, "job_id": job_id}, indent=2), encoding="utf-8")
        command = swift_command(c, str(output))
        (output / "train_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        trainer = child(command, "trainer.log", policy_env)
        while trainer.poll() is None:
            if startup_guard is not None:
                if startup_guard.poll() not in (None,0):
                    raise RuntimeError('启动期利用率保护交接失败')
            if service.poll() is not None:
                raise RuntimeError("训练期间 Judge 服务退出")
            time.sleep(2.)
        if trainer.returncode:
            raise RuntimeError(f"GRPO 训练失败：exit={trainer.returncode}")
        errors = cleanup_processes([trainer])
        children.remove(trainer)
        if errors:
            raise RuntimeError(f"训练子进程清理失败：{errors}")
        subprocess.run([c["train_python"], "-m", "training.grpo_v3.six_user_binary.validate_run", "--output", str(output)], cwd=package, env=env, check=True)
        final_validation(c, output, evaluate_policy)
        result["status"] = "completed"
    except BaseException as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            result["cleanup_errors"] = cleanup_processes(children)
            for stream in logs:
                try:
                    stream.close()
                except OSError as exc:
                    result["cleanup_errors"].append(str(exc))
        finally:
            (output / "run_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
