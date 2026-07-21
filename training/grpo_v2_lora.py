"""Qwen3-VL-8B 的最小在线 GRPO QLoRA 训练入口。

GPU0 由 TRL 负责 policy 生成和更新；GPU1 上的冻结 vLLM reviewer
通过仓库原生 judge/answerability 管线产生 reward。
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def completion_text(completion: Any) -> str:
    """兼容 TRL 的普通 completion 和对话式 completion。"""

    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return completion_text(completion.get("content", ""))
    if isinstance(completion, (list, tuple)):
        chunks: list[str] = []
        for item in completion:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    chunks.append(str(item["text"]))
                elif "content" in item:
                    chunks.append(completion_text(item["content"]))
                elif item.get("text") is not None:
                    chunks.append(str(item["text"]))
        return "".join(chunks)
    return str(completion or "")


def expand_to_length(values: Any, length: int, *, name: str) -> list[Any]:
    """把 prompt 级元数据展开到 completion 数量。"""

    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        rows = [values]
    else:
        rows = list(values)
    if len(rows) == length:
        return rows
    if len(rows) == 1:
        return rows * length
    raise ValueError(f"{name} 数量为 {len(rows)}，无法展开到 {length}")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON object")
            rows.append(value)
    return rows


def evenly_limit_paths(paths: Sequence[str], limit: int) -> list[str]:
    """按时间顺序均匀保留至多 limit 个帧路径。"""

    values = [str(path) for path in paths]
    if limit < 1:
        raise ValueError("max_policy_frames_per_clip 必须 >= 1")
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[len(values) // 2]]
    indices = [round(index * (len(values) - 1) / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]


def build_training_rows(
    packets: Iterable[dict[str, Any]],
    *,
    max_prompts: int,
    question_type: str,
    generation_mode: str = "baseline",
    policy_media_mode: str = "native_video",
    max_policy_frames_per_clip: int = 4,
    prompt_builder: Callable[..., str] | None = None,
    media_selector: Callable[..., tuple[list[str], list[str]]] | None = None,
    frame_selector: Callable[[dict[str, Any]], list[str]] | None = None,
) -> list[dict[str, Any]]:
    """把 evidence packet 转成 TRL 多模态对话数据。"""

    if prompt_builder is None or media_selector is None or frame_selector is None:
        modules = _repo_modules()
        prompt_builder = prompt_builder or modules["build_video_generation_prompt"]
        media_selector = media_selector or modules["media_for_clips"]
        frame_selector = frame_selector or modules["clip_image_paths"]

    if policy_media_mode not in {"native_video", "sampled_frames"}:
        raise ValueError(f"不支持的 policy_media_mode: {policy_media_mode}")

    rows: list[dict[str, Any]] = []
    for packet in packets:
        if len(rows) >= max_prompts:
            break
        evidence_id = str(packet.get("evidence_id") or "").strip()
        clips = packet.get("clips")
        if not evidence_id or not isinstance(clips, list) or not clips:
            continue
        prompt = prompt_builder(
            packet,
            question_type,
            generation_mode=generation_mode,
        )
        if policy_media_mode == "sampled_frames":
            images = [
                path
                for clip in clips
                for path in evenly_limit_paths(
                    frame_selector(clip),
                    max_policy_frames_per_clip,
                )
            ]
            videos = []
            if not images:
                raise ValueError(f"{evidence_id} 没有可用于 policy 的采样帧")
        else:
            images, videos = media_selector(
                clips,
                backend="openai-compatible-local",
                allow_openai_video_input=True,
                media_role="generator",
            )
        content = [
            *({"type": "image", "image": str(path)} for path in images),
            *({"type": "video", "video": str(path)} for path in videos),
            {"type": "text", "text": prompt},
        ]
        rows.append(
            {
                "prompt": [{"role": "user", "content": content}],
                "packet_json": json.dumps(packet, ensure_ascii=False),
                "evidence_id": evidence_id,
                "question_type": question_type,
                "generation_mode": generation_mode,
            }
        )
    if not rows:
        raise ValueError("evidence 中没有可用于训练的 packet")
    return rows


def _repo_modules() -> dict[str, Any]:
    """延迟导入仓库模块，使登录节点上的纯逻辑测试不依赖训练环境。"""

    try:
        video_loop = importlib.import_module("egolife_two_user_qa.video_qa_loop")
        prompts = importlib.import_module("egolife_two_user_qa.prompts")
        schema = importlib.import_module("egolife_two_user_qa.schema")
        runner = importlib.import_module("egolife_two_user_qa.qwen3vl_runner")
    except ModuleNotFoundError:
        # 本地仓库有时以源码根目录保存；为相对导入建立只读别名。
        import types

        package_name = "_egoqa_repo"
        if package_name not in sys.modules:
            package = types.ModuleType(package_name)
            package.__path__ = [str(PROJECT_ROOT)]
            package.__package__ = package_name
            sys.modules[package_name] = package
        video_loop = importlib.import_module(f"{package_name}.video_qa_loop")
        prompts = importlib.import_module(f"{package_name}.prompts")
        schema = importlib.import_module(f"{package_name}.schema")
        runner = importlib.import_module(f"{package_name}.qwen3vl_runner")

    scoring = importlib.import_module("grpo_judge_reward.scoring")
    return {
        "build_video_generation_prompt": prompts.build_video_generation_prompt,
        "clip_image_paths": video_loop.clip_image_paths,
        "media_for_clips": video_loop.media_for_clips,
        "complete_generator_metadata": video_loop.complete_generator_metadata,
        "video_evidence_for_packet": video_loop.video_evidence_for_packet,
        "human_audit_packet": video_loop.human_audit_packet,
        "run_parallel_review_judges": video_loop.run_parallel_review_judges,
        "build_review_from_gates": video_loop.build_review_from_gates,
        "extract_json_object": schema.extract_json_object,
        "validate_qa_item": schema.validate_qa_item,
        "OpenAICompatibleLocalRunner": runner.OpenAICompatibleLocalRunner,
        "compute_judge_reward": scoring.compute_judge_reward,
    }


def make_repo_score_fn(
    *,
    review_model_id: str,
    review_base_url: str,
    policy_model_id: str,
    review_max_new_tokens: int,
) -> Callable[..., dict[str, Any]]:
    """构造复用仓库原生 judge 的 reward 计算函数。"""

    modules = _repo_modules()
    reviewer = modules["OpenAICompatibleLocalRunner"](
        model_id=review_model_id,
        base_url=review_base_url,
        max_new_tokens=review_max_new_tokens,
        timeout=900,
        allow_video_input=True,
    )

    def score(
        *,
        raw_completion: str,
        packet: dict[str, Any],
        evidence_id: str,
        question_type: str,
        generation_mode: str,
        candidate_index: int,
    ) -> dict[str, Any]:
        candidate_id = f"{evidence_id}::grpo::{candidate_index}"
        try:
            qa = modules["extract_json_object"](raw_completion)
        except Exception as exc:
            return {
                "reward": None,
                "record": {
                    "candidate_id": candidate_id,
                    "evidence_id": evidence_id,
                    "masked": True,
                    "mask_reason": f"completion_json_parse_fail: {exc}",
                    "raw_qa": raw_completion,
                },
            }

        qa["qa_id"] = str(qa.get("qa_id") or f"GRPO_{evidence_id}_{candidate_index}")
        qa["question_type"] = question_type
        qa["generation_mode"] = generation_mode
        qa["required_users"] = list(packet.get("required_users") or qa.get("required_users") or [])
        qa["model_id"] = policy_model_id
        qa["source_urls"] = packet.get("source_urls", {})
        qa.setdefault("review", {})
        qa["video_evidence"] = modules["video_evidence_for_packet"](packet)
        qa["human_audit"] = modules["human_audit_packet"](packet)
        qa.setdefault("generation_trace", [])
        modules["complete_generator_metadata"](
            qa,
            packet=packet,
            question_type=question_type,
        )
        schema_errors = modules["validate_qa_item"](qa)
        full_images, full_videos = modules["media_for_clips"](
            packet.get("clips", []),
            backend="openai-compatible-local",
            allow_openai_video_input=True,
            media_role="full",
        )
        prompt_rows: list[dict[str, Any]] = []
        judge, answerability, judge_trace = modules["run_parallel_review_judges"](
            qa_item=qa,
            packet=packet,
            schema_errors=schema_errors,
            runner=reviewer,
            media_backend="openai-compatible-local",
            allow_openai_video_input=True,
            prompt_rows=prompt_rows,
            full_image_paths=full_images,
            full_video_paths=full_videos,
            attempt=candidate_index + 1,
        )
        judge_passed = bool((judge.get("gate") or {}).get("passed"))
        answerability_passed = bool((answerability.get("gate") or {}).get("passed"))
        accepted = not schema_errors and judge_passed and answerability_passed
        if schema_errors:
            rejection_stage = "schema"
        elif not judge_passed:
            rejection_stage = "judger"
        elif not answerability_passed:
            rejection_stage = "answerability"
        else:
            rejection_stage = None
        review = modules["build_review_from_gates"](
            judge=judge,
            answerability=answerability,
            schema_errors=schema_errors,
            accepted=accepted,
            rejection_stage=rejection_stage,
            final_reason=(judge.get("gate") or {}).get("reason"),
        )
        qa["review"] = review
        data = {
            "candidate_id": candidate_id,
            "group_id": evidence_id,
            "evidence_id": evidence_id,
            "qa_id": qa.get("qa_id"),
            "attempt": candidate_index + 1,
            "raw_qa": raw_completion,
            "qa": qa,
            "schema_errors": schema_errors,
            "review": review,
            "answerability": answerability,
        }
        record = modules["compute_judge_reward"](data)
        record_dict = record.to_dict()
        record_dict["review_model_id"] = review_model_id
        record_dict["judge_trace"] = judge_trace
        record_dict["judge_prompts"] = prompt_rows
        return {"reward": record.reward_total, "record": record_dict}

    return score


class RepoJudgeReward:
    """TRL reward callable：masked completion 返回 None。"""

    __name__ = "repo_judge_reward"

    def __init__(
        self,
        *,
        trace_path: str | Path,
        score_fn: Callable[..., dict[str, Any]],
    ) -> None:
        self.trace_path = Path(trace_path)
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.trace_path.write_text("", encoding="utf-8")
        self.score_fn = score_fn
        self._lock = threading.Lock()

    def __call__(self, completions: Sequence[Any], **kwargs: Any) -> list[float | None]:
        count = len(completions)
        packets = expand_to_length(kwargs["packet_json"], count, name="packet_json")
        evidence_ids = expand_to_length(kwargs["evidence_id"], count, name="evidence_id")
        question_types = expand_to_length(kwargs["question_type"], count, name="question_type")
        generation_modes = expand_to_length(kwargs["generation_mode"], count, name="generation_mode")
        rewards: list[float | None] = []
        for index, completion in enumerate(completions):
            raw = completion_text(completion)
            packet_value = packets[index]
            packet = json.loads(packet_value) if isinstance(packet_value, str) else packet_value
            result = self.score_fn(
                raw_completion=raw,
                packet=packet,
                evidence_id=str(evidence_ids[index]),
                question_type=str(question_types[index]),
                generation_mode=str(generation_modes[index]),
                candidate_index=index,
            )
            reward = result.get("reward")
            if reward is not None:
                reward = float(reward)
                if not math.isfinite(reward):
                    raise ValueError(f"reward 非有限值: {reward}")
            trace = {
                "evidence_id": str(evidence_ids[index]),
                "candidate_index": index,
                "reward": reward,
                "record": result.get("record", {}),
            }
            with self._lock:
                with self.trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
            rewards.append(reward)
        return rewards


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def train(args: argparse.Namespace) -> None:
    """加载 4-bit policy，注入 LoRA，并执行最小在线 GRPO。"""

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig
    try:
        from transformers import Qwen3VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForImageTextToText as ModelClass
    from trl import GRPOConfig, GRPOTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("训练必须在 GPU 计算节点运行；登录节点 cuda_available=False 是正常现象")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    packets = read_jsonl(args.evidence)
    rows = build_training_rows(
        packets,
        max_prompts=args.max_prompts,
        question_type=args.question_type,
        generation_mode=args.generation_mode,
        policy_media_mode=args.policy_media_mode,
        max_policy_frames_per_clip=args.max_policy_frames_per_clip,
    )
    dataset = Dataset.from_list(rows)
    (output_dir / "dataset_preview.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    model = ModelClass.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        quantization_config=quantization,
        device_map={"": 0},
        local_files_only=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )
    model = get_peft_model(model, lora_config)
    for name, parameter in model.named_parameters():
        lowered = name.lower()
        if "visual" in lowered or "vision" in lowered:
            parameter.requires_grad = False
    trainable = [(name, parameter.numel()) for name, parameter in model.named_parameters() if parameter.requires_grad]
    visual_trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad and ("visual" in name.lower() or "vision" in name.lower())]
    if not trainable:
        raise RuntimeError("LoRA trainable parameters 为 0")
    if visual_trainable:
        raise RuntimeError(f"视觉参数仍可训练: {visual_trainable[:5]}")
    print(f"preflight_trainable_parameters={sum(size for _, size in trainable)}", flush=True)
    print("preflight_visual_trainable=0", flush=True)

    score_fn = make_repo_score_fn(
        review_model_id=args.review_model_id,
        review_base_url=args.review_base_url,
        policy_model_id=args.model_path,
        review_max_new_tokens=args.review_max_new_tokens,
    )
    reward = RepoJudgeReward(
        trace_path=output_dir / "reward_trace.jsonl",
        score_fn=score_fn,
    )
    config = GRPOConfig(
        output_dir=str(output_dir / "checkpoints"),
        max_steps=args.max_steps,
        # 单卡全局生成批次必须可被 num_generations 整除；这里正好是一组。
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        beta=0.0,
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        use_vllm=False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        log_completions=True,
        dataloader_num_workers=0,
        seed=args.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=config,
        train_dataset=dataset,
        processing_class=processor,
    )
    result = trainer.train()
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    processor.save_pretrained(str(adapter_dir))
    metrics = _json_safe(result.metrics)
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    trainer.state.save_to_json(str(output_dir / "trainer_state.json"))
    (output_dir / "training_args.json").write_text(
        json.dumps(_json_safe(vars(args)), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "status": "completed",
        "model_path": args.model_path,
        "review_model_id": args.review_model_id,
        "evidence": str(Path(args.evidence).resolve()),
        "max_prompts": args.max_prompts,
        "num_generations": args.num_generations,
        "max_steps": args.max_steps,
        "policy_media_mode": args.policy_media_mode,
        "max_policy_frames_per_clip": args.max_policy_frames_per_clip,
        "adapter": str(adapter_dir),
        "reward_trace": str(output_dir / "reward_trace.jsonl"),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--review-model-id", required=True)
    parser.add_argument("--review-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-prompts", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--question-type", choices=("commonality", "difference", "neutral"), default="commonality")
    parser.add_argument("--generation-mode", default="baseline")
    parser.add_argument(
        "--policy-media-mode",
        choices=("sampled_frames", "native_video"),
        default="sampled_frames",
        help="sampled_frames 可避开当前 TRL 原生视频批处理的占位符错误",
    )
    parser.add_argument("--max-policy-frames-per-clip", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--review-max-new-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if (
        args.max_prompts < 1
        or args.num_generations < 2
        or args.max_steps < 1
        or args.max_policy_frames_per_clip < 1
    ):
        parser.error(
            "max-prompts >= 1、num-generations >= 2、max-steps >= 1、"
            "max-policy-frames-per-clip >= 1"
        )
    return args


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
