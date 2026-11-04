"""实际安装环境的三人CPU检查：不加载模型，不初始化分布式或GPU。"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import MISSING, fields, is_dataclass
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time
import traceback
from types import SimpleNamespace


def require_sampling_flags(command, supported_fields, shuffle_field):
    """按现场Swift参数定义检查两个不同层次的shuffle开关。"""
    if shuffle_field not in supported_fields:
        raise ValueError(f"实际采样器读取的参数未出现在Swift参数定义中：{shuffle_field}")
    for name in {"dataset_shuffle", shuffle_field} | ({"shuffle_dataset"} & set(supported_fields)):
        flag = "--" + name
        if flag not in command or command[command.index(flag)+1].lower() != "false":
            raise ValueError(f"正式CLI必须显式设置 {flag} false，不能只在CPU检查中关闭shuffle")
    return shuffle_field


def sampler_shuffle_source(trainer_class):
    """从现场实际方法定位 self.shuffle_dataset 的赋值来源。"""
    methods = []
    prepare = getattr(trainer_class, "_prepare_algorithm_params", None)
    if prepare is not None:
        methods.append(prepare)
    methods.extend(cls.__dict__["__init__"] for cls in trainer_class.__mro__ if "__init__" in cls.__dict__)
    for method in methods:
        try:
            source = textwrap.dedent(inspect.getsource(method))
            tree = ast.parse(source)
        except (OSError, TypeError, IndentationError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            matches = any(isinstance(target, ast.Attribute) and target.attr == "shuffle_dataset"
                and isinstance(target.value, ast.Name) and target.value.id == "self" for target in node.targets)
            if matches and isinstance(node.value, ast.Attribute) and node.value.attr in {"dataset_shuffle", "shuffle_dataset"}:
                return {"argument": node.value.attr, "method": method.__qualname__,
                        "file": inspect.getfile(method), "assignment": ast.get_source_segment(source, node)}
    raise ValueError("无法从现场Swift/TRL源码确认shuffle参数来源，禁止假设采样顺序")


def check_native_batches(rows, ranks):
    """核对真实采样器返回值；此函数不生成或替换采样结果。"""
    from dual_runtime import condition_binding, validate_rank_bindings
    if len(rows) != 48 or len({condition_binding(row) for row in rows}) != 48 or len(ranks) != 2:
        raise ValueError("必须使用48个唯一条件和两个rank的实际采样结果")
    visits = []
    for rank, batches in enumerate(ranks):
        if len(batches) != 96 or any(len(batch) != 4 or len(set(batch)) != 1 for batch in batches):
            raise ValueError("每rank必须有384个完整G4微批")
        for start in range(0, 96, 4):
            if batches[start:start+4] != [batches[start]]*4:
                raise ValueError("同一次生成的G4必须缓存供4个微批使用")
        actual = [batch[0] for batch in batches[::4]]
        expected = [(2*step+rank) % 48 for step in range(24)]
        if actual != expected:
            raise ValueError("真实采样顺序没有按48条件循环四遍")
        visits.append(actual)
    if Counter(visits[0]+visits[1]) != Counter({index: 1 for index in range(48)}):
        raise ValueError("真实采样访问次数不是每条件4次")
    for left, right in zip(*visits):
        validate_rank_bindings([[rows[left]]*4, [rows[right]]*4])
    return {"epochs": 1, "optimizer_updates_per_rank": 24, "local_microbatches_per_rank": 96,
            "global_unique_conditions": 48, "visits_per_condition": 1,
            "global_training_candidates": 192, "global_condition_visits": 48,
            "sampler_index_emissions_including_cached_reuse": 768,
            "same_window_pairs": 0, "rank_condition_indices": visits}


def check_c_prompt(prompt, alternate_prompt, qa):
    if prompt != alternate_prompt:
        raise ValueError("C提示词随正确答案或选项变化，存在答案泄漏")
    if any(option in prompt for option in qa["options"]) or '"correct"' in prompt or '"options"' in prompt:
        raise ValueError("C提示词泄漏了正确答案字段或选项")


def template_truncation_check(arguments_class, parsed):
    """执行现场Arguments的真实映射方法，不执行会初始化模型的后处理。"""
    if getattr(parsed, "truncation_strategy", None) != "delete":
        raise ValueError("正式CLI必须使用当前Swift支持的truncation_strategy=delete")
    native = arguments_class.__new__(arguments_class)
    if is_dataclass(arguments_class):
        for field in fields(arguments_class):
            if field.default is not MISSING:
                setattr(native, field.name, field.default)
            elif field.default_factory is not MISSING:
                setattr(native, field.name, field.default_factory())
    native.__dict__.update(vars(parsed))
    method = native.get_template_kwargs
    kwargs = method()
    if kwargs.get("truncation_strategy") != "raise":
        raise ValueError("现场Swift没有把CLI delete映射为训练template的raise，禁止提交")
    return {"status": "passed", "cli_truncation_strategy": parsed.truncation_strategy,
            "internal_template_truncation_strategy": kwargs["truncation_strategy"],
            "native_method": method.__qualname__, "source_file": inspect.getfile(method),
            "native_method_source": inspect.getsource(method), "dataclass_post_init_called": False,
            "boundary": "真实Arguments方法的CPU执行证据；训练开始仍检查实际48行和template.raise"}


def setup(config):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU检查必须在显式禁用GPU的Slurm作业中执行")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("CPU检查不允许启动分布式进程")
    root = Path(config["project_root"]).resolve()
    sys.path[:0] = [str(root/"runtime_deps_train"), str(root/"multi-user"), str(root)]
    from training.torch_storage_preflight import REQUIRED_STORAGE_VARIABLES, validate_storage_environment
    scratch = Path(config["scratch_root"])/("aria3_cpu_"+os.environ["SLURM_JOB_ID"])
    for name in REQUIRED_STORAGE_VARIABLES+("PIP_CACHE_DIR",):
        os.environ[name] = str(scratch/name.lower())
    os.environ.update(config.get("compiler_environment", {}))
    os.environ.update(PYTHONDONTWRITEBYTECODE="1", TOKENIZERS_PARALLELISM="false", VLLM_NO_USAGE_STATS="1",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="8",
        PYTHONPATH=os.pathsep.join(map(str, [root/"runtime_deps_train", root/"multi-user", root])))
    tools_path = root/"preflight/media_tools.json"
    if tools_path.is_file():
        tools = json.loads(tools_path.read_text())
        os.environ["PATH"] = str(Path(tools["ffmpeg"]).parent)+os.pathsep+os.environ.get("PATH", "")
        os.environ["LD_LIBRARY_PATH"] = tools["ffmpeg_lib"]+os.pathsep+os.environ.get("LD_LIBRARY_PATH", "")
    report = validate_storage_environment(allowed_root=scratch, environ=os.environ)
    (root/"preflight").mkdir(parents=True, exist_ok=True)
    (root/"preflight/cpu_validation_storage.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise RuntimeError("CPU检查scratch存储封闭失败")
    return root, scratch


def command_and_native_sampler(config, rows, root, scratch):
    import torch
    from torch.utils.data import BatchSampler
    from accelerate.data_loader import BatchSamplerShard
    from trl import GRPOConfig
    from trl.trainer.utils import RepeatSampler
    from swift.arguments import RLHFArguments
    from swift.rlhf_trainers.grpo_trainer import GRPOTrainer
    from transformers import HfArgumentParser
    from training.grpo_v3.six_user_binary.launch import swift_command

    command = swift_command({**config, "policy_gpu_indices": [0], "judge_gpu_indices": [0]}, str(scratch/"cli"))
    callbacks = command.index("--callbacks")
    command.insert(callbacks+2, "egoqa_dual_audit")
    command.extend(["--steps_per_generation", "4", "--ddp_find_unused_parameters", "false"])
    supported = {field.name for field in fields(RLHFArguments)}
    mapping = sampler_shuffle_source(GRPOTrainer)
    require_sampling_flags(command, supported, mapping["argument"])
    parser = HfArgumentParser(RLHFArguments)
    # 真实Arguments生成的parser验证参数；不调用会加载模型配置的dataclass后处理。
    parsed, unknown = parser.parse_known_args(command[2:])
    if unknown:
        raise ValueError("正式Swift CLI存在未识别参数："+repr(unknown))
    for name, expected in {"max_steps": 24, "num_generations": 4, "gradient_accumulation_steps": 4,
        "per_device_train_batch_size": 1, "steps_per_generation": 4, "num_iterations": 1,
        "max_pixels": 50176, "max_length": 131072, "max_completion_length": 2048,
        "eval_strategy": "no", "truncation_strategy": "delete", "dataset_shuffle": False,
        "save_only_model": False, "resume_only_model": False, "ignore_data_skip": False}.items():
        if getattr(parsed, name, None) != expected:
            raise ValueError(f"真实CLI参数不一致：{name}={getattr(parsed, name, None)!r}")
    truncation = template_truncation_check(RLHFArguments, parsed)
    if any(flag in command for flag in ("--resume_from_checkpoint", "--val_dataset")):
        raise ValueError("正式CLI不允许恢复旧训练状态或验证数据")
    if config.get('initialization_mode')=='adapter_warm_start':
        from warm_start import validate_initialization,validate_adapter_files
        validate_initialization(config)
        for key,value in [('adapters',config['initial_adapter']),('ref_adapters',config['reference_adapter']),
                          ('load_args','false'),('load_data_args','false')]:
            if command[command.index('--'+key)+1]!=value:raise ValueError('348初始化CLI不匹配：'+key)
        validate_adapter_files(config['initial_adapter'],config)
    elif any(flag in command for flag in ('--adapters','--ref_adapters')):
        raise ValueError('fresh_base不能载入历史adapter')
    options = json.loads(getattr(parsed, "vllm_limit_mm_per_prompt"))
    if options != {"image": 900}:
        raise ValueError("Policy必须允许900张三人完整帧")
    (root/"preflight/effective_train_command.json").write_text(json.dumps(command, indent=2))

    args = GRPOConfig(output_dir=str(scratch/"cpu_parser_args"), use_cpu=True, bf16=False, fp16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=4, num_generations=4,
        steps_per_generation=4, max_steps=24, shuffle_dataset=False, report_to="none",
        scale_rewards="group", loss_type="grpo")
    if args.world_size != 1 or args.shuffle_dataset is not False:
        raise ValueError("单进程CPU GRPOConfig未保持实际CPU世界大小或关闭shuffle")
    ranks = []
    sampler_parameters = []
    for rank in range(2):
        # 不调用模型/分布式构造器，仅调用现场Swift类的真实采样器方法。
        trainer = GRPOTrainer.__new__(GRPOTrainer)
        trainer.args = SimpleNamespace(**{**vars(args), **vars(parsed), "generation_batch_size": 8})
        trainer.train_dataset = rows
        trainer.template = SimpleNamespace(sequence_parallel_size=1)
        trainer.num_generations = parsed.num_generations
        trainer.num_iterations = parsed.num_iterations
        trainer.shuffle_dataset = getattr(parsed, mapping["argument"])
        sampler = trainer._get_train_sampler()
        if not isinstance(sampler, RepeatSampler) or getattr(sampler, "shuffle", None) is not False:
            raise ValueError("现场Swift真实采样器不是关闭shuffle的RepeatSampler")
        observed = {key: getattr(sampler, key) for key in ("mini_repeat_count", "batch_size", "repeat_count", "shuffle")}
        if observed != {"mini_repeat_count": 4, "batch_size": 2, "repeat_count": 4, "shuffle": False}:
            raise ValueError("现场真实采样器参数与两rank G4不一致："+repr(observed))
        sampler_parameters.append(observed)
        sharded = BatchSamplerShard(BatchSampler(sampler, batch_size=4, drop_last=False), num_processes=2,
            process_index=rank, split_batches=False, even_batches=True)
        batches = []
        for epoch in range(1):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            batches.extend(list(sharded))
        ranks.append(batches)
    report = check_native_batches(rows, ranks)
    report.update(status="passed", native_sampler_parameters=sampler_parameters,
        shuffle_assignment=mapping, cpu_parser_world_size=args.world_size,
        actual_rank_local_batches=ranks, versions={name: version(name) for name in ("torch", "trl", "accelerate", "transformers", "ms-swift")},
        implementation_files={"RepeatSampler": inspect.getfile(RepeatSampler), "BatchSamplerShard": inspect.getfile(BatchSamplerShard),
            "SwiftGRPOTrainer": inspect.getfile(GRPOTrainer)}, sampler_method_source=inspect.getsource(GRPOTrainer._get_train_sampler),
        swift_argument_fields_checked=sorted(set(["dataset_shuffle", "shuffle_dataset"]) & supported),
        template_truncation=truncation,
        GPU_models_loaded=False, distributed_initialized=False)
    if torch.cuda.is_initialized() or torch.distributed.is_initialized():
        raise RuntimeError("CPU采样器检查意外初始化了GPU或分布式")
    return report


def scheduler_check(config):
    import torch
    from transformers.optimization import get_scheduler
    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=config["learning_rate"])
    scheduler = get_scheduler(config["lr_scheduler_type"], optimizer, num_warmup_steps=config["warmup_steps"],
        num_training_steps=config["max_steps"], scheduler_specific_kwargs=config["lr_scheduler_kwargs"])
    values = []
    for _ in range(24):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()
        values.append(scheduler.get_last_lr()[0])
    expected = config["learning_rate"]*config["lr_scheduler_kwargs"]["min_lr_rate"]
    if scheduler.last_epoch != 24 or not math.isclose(values[-1], expected, abs_tol=1e-12):
        raise ValueError("96步完整退火没有达到指定学习率下限")
    return {"status": "passed", "scalar_fixture_only": True, "scheduler_last_epoch": scheduler.last_epoch,
            "peak_lr": max(values), "final_lr": values[-1], "learning_rates": values}


def judge_routing(config, rows, root):
    from training.grpo_v3.six_user_binary.judge import build_examples
    from training.judge_sft.collator import model_visible_prompt, _apply_chat_template, _assert_thinking_disabled
    from transformers import AutoProcessor
    from PIL import Image
    packet = rows[0]["source_packet_id"]
    selected = sorted((row for row in rows if row["source_packet_id"] == packet), key=lambda row: row["asker_index"])
    if [row["asker_index"] for row in selected] != [0, 1, 2]:
        raise ValueError("真实媒体路由检查必须覆盖同packet三个提问者")
    processor = AutoProcessor.from_pretrained(config["policy_model"], local_files_only=True, trust_remote_code=True)
    qa = {"question": "Which visible object was placed beside my cup?",
          "options": ["CPU_OPTION_A_BOOK", "CPU_OPTION_B_BAG", "CPU_OPTION_C_PLATE", "CPU_OPTION_D_BOWL", "CPU_OPTION_E_HAT"],
          "correct": "A", "answer": "CPU_OPTION_A_BOOK"}
    previews = []
    decode_paths = []
    for row in selected:
        if (row.get("alignment") or {}).get("cross_device_sync_verified") is not False:
            raise ValueError("数据行必须明确记录物理同步尚未核实")
        request = {**row, "request_id": f"CPU_ROUTING_FIXTURE_{row['asker_index']}", "completion": json.dumps(qa)}
        examples = build_examples(request)
        other = {**qa, "correct": "B", "answer": qa["options"][1], "options": list(reversed(qa["options"]))}
        other["answer"] = other["options"][1]
        alternate = build_examples({**request, "completion": json.dumps(other)})
        if set(examples) != {"formality", "groundedness", "combined_answerability"}:
            raise ValueError("Judge路由不是三项F/E/C")
        for task, example in examples.items():
            expected_count = 0 if task == "formality" else 900
            if example.frame_count != expected_count:
                raise ValueError("Judge图像数量不符合0/900/900")
            if task != "formality":
                if len(example.frame_sets) != 3 or any(len(item.frames) != 300 for item in example.frame_sets):
                    raise ValueError("Judge必须保留三人各300完整帧")
                if example.frame_sets[0].label != "speaker: "+row["required_users"][0]:
                    raise ValueError("Judge提问者顺序错位")
                if "Cross-device synchronization is not verified" not in example.prompt:
                    raise ValueError("Judge未收到物理同步尚未核实提示")
            visible = model_visible_prompt(example)
            if task == "combined_answerability":
                check_c_prompt(visible, model_visible_prompt(alternate[task]), qa)
                if not decode_paths:
                    decode_paths = [example.frames[index] for index in (0, 149, 299, 300, 449, 599, 600, 899)]
            content = [{"type": "image", "image": path, "min_pixels": 3136, "max_pixels": config["max_pixels"]}
                       for path in example.frames]+[{"type": "text", "text": visible}]
            rendered = _apply_chat_template(processor, [{"role": "user", "content": content}])
            _assert_thinking_disabled(rendered)
            if rendered.count("<|image_pad|>") != expected_count:
                raise ValueError("实际chat模板的图像占位符与媒体数量不一致")
            previews.append({"asker_index": row["asker_index"], "task": task, "frame_count": example.frame_count,
                "frame_labels": [item.label for item in example.frame_sets], "model_visible_prompt": visible,
                "cross_device_sync_verified": False, "test_fixture_only_not_a_model_score": True})
    decoded = []
    for path in decode_paths:
        with Image.open(path) as image:
            image.load()
            if image.format != "JPEG" or min(image.size) <= 0:
                raise ValueError("真实采样图片不是可解码JPEG")
            decoded.append({"path": str(path), "size": list(image.size), "format": image.format})
    if len(decoded) != 8:
        raise ValueError("必须实际解码8张JPEG")
    (root/"preflight/model_visible_three_speakers.json").write_text(json.dumps(previews, ensure_ascii=False, indent=2))
    return {"status": "passed", "asker_count": 3, "model_visible_examples": len(previews),
            "frame_counts_per_asker": [0, 900, 900], "answer_leakage_checked": True,
            "decoded_real_frames": decoded, "cross_device_sync_verified": False, "GPU_models_loaded": False}


def role_imports(config, root):
    from training.grpo_v3.six_user_binary.launch import role_environment
    probe = """import json,sys,torch
from training.grpo_v3.six_user_binary import predictor
c=predictor.validate_config(json.load(open(sys.argv[1])))
o=predictor.engine_options(c)
assert o['enable_lora'] and o['tensor_parallel_size']==1
assert set(c['adapters'])=={'formality','groundedness','answerability'}
assert o['limit_mm_per_prompt']=={'image':900}
assert not torch.cuda.is_initialized()
print('ARIA3_ROLE_IMPORT '+json.dumps({'status':'passed','python':sys.version,'predictor':predictor.__file__,
    'engine_options':o,'GPU_models_loaded':False}))
"""
    report = {}
    for role in ("train", "judge"):
        env = role_environment(config[role+"_python"], os.environ.copy())
        env["PYTHONPATH"] = os.pathsep.join(map(str, ([root/"runtime_deps_train"] if role == "train" else [])+[root/"multi-user", root]))
        process = subprocess.run([config[role+"_python"], "-c", probe, config["judge_config"]], env=env,
            cwd=root/"multi-user", capture_output=True, text=True, timeout=240)
        (root/f"preflight/cpu_role_{role}.log").write_text(process.stdout+process.stderr)
        process.check_returncode()
        marked = [line.removeprefix("ARIA3_ROLE_IMPORT ") for line in process.stdout.splitlines() if line.startswith("ARIA3_ROLE_IMPORT ")]
        if len(marked) != 1:
            raise RuntimeError("角色环境没有返回唯一的真实导入结果："+role)
        report[role] = json.loads(marked[0])
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    root = Path(config["project_root"])
    output = root/"preflight/cpu_validation_complete.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        root, scratch = setup(config)
        from dual_runtime import validate_config, condition_binding
        from training.grpo_v3.six_user_binary.data import read_rows, validate_row
        validate_config(config)
        rows = read_rows(config["train_dataset"])
        if len(rows) != 48 or len({condition_binding(row) for row in rows}) != 48:
            raise ValueError("实际训练文件必须包含48个唯一三人条件")
        for index, row in enumerate(rows):
            validate_row(row)
            if len(row["required_users"]) != 3 or row["generator_frame_counts"][0] != 300:
                raise ValueError("实际行的三人角色或提问者完整300帧错误")
            if index % 12 == 0:
                print("ARIA3_CPU_DATA_PROGRESS", index, flush=True)
        sampler = command_and_native_sampler(config, rows, root, scratch)
        (root/"preflight/native_sampler.json").write_text(json.dumps(sampler, ensure_ascii=False, indent=2))
        schedule = scheduler_check(config)
        (root/"preflight/scheduler_96.json").write_text(json.dumps(schedule, indent=2))
        routing = judge_routing(config, rows, root)
        roles = role_imports(config, root)
        import torch
        if torch.cuda.is_initialized() or torch.distributed.is_initialized():
            raise RuntimeError("CPU检查意外初始化GPU或分布式")
        result = {"status": "passed", "job_id": os.environ["SLURM_JOB_ID"], "GPU_operations": 0,
            "elapsed_seconds": time.time()-started, "actual_data_conditions": 48,
            "real_sampler_global_candidates": sampler["global_training_candidates"],
            "sampler": {key: sampler[key] for key in ("epochs", "visits_per_condition", "global_condition_visits", "shuffle_assignment", "template_truncation")},
            "scheduler_final_lr": schedule["final_lr"], "judge_routing": routing, "startup_import_roles": roles,
            "history_pool_loaded": False, "MiniLM_loaded": False, "decoded_real_frames": 8,
            "deleted_rows": 0, "truncated_rows": 0,
            "boundary": "实际CPU数据/采样/CLI/模板与导入检查；未加载GPU模型，未进行Judge评分或训练更新"}
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print("ARIA3_CPU_VALIDATION_COMPLETE", json.dumps(result, ensure_ascii=False), flush=True)
    except BaseException as error:
        output.write_text(json.dumps({"status": "failed", "job_id": os.environ.get("SLURM_JOB_ID"),
            "GPU_operations": 0, "elapsed_seconds": time.time()-started,
            "error": repr(error), "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2))
        raise


if __name__ == "__main__":
    main()
