"""Train a no-head Qwen3-VL judge from normalized PASS/FAIL invocations."""

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

from .collator import (
    DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
    DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    JudgeFrameCollator,
    QWEN_VISION_TOKEN_PIXEL_AREA,
    adaptive_image_max_pixels,
)
from .contracts import (
    DEFAULTS,
    JudgeTask,
    Verdict,
    validate_task_weights,
)
from .data import (
    JudgeDataset,
    load_normalized_manifest,
    manifest_summary,
)
from .loss import (
    class_weights_by_task,
    resolve_verdict_token_ids,
    task_sampling_scales,
)
from .trainer import audit_rank_aligned_sampler, build_verdict_trainer_class


VISION_MARKERS = ("visual", "vision_tower", "vision_model")
ALIGNER_MARKERS = ("aligner", "projector", "merger")


def _matches_any(name: str, markers: tuple[str, ...]) -> bool:
    lowered = name.lower()
    return any(marker in lowered for marker in markers)


def _parameter_count(parameter: Any) -> int:
    """Count full ZeRO-3 parameters even when the local tensor is partitioned."""

    return int(getattr(parameter, "ds_numel", parameter.numel()))


def freeze_multimodal_adapters(
    model: Any,
    *,
    freeze_vision: bool,
    freeze_aligner: bool,
) -> dict[str, int]:
    for name, parameter in model.named_parameters():
        if freeze_vision and _matches_any(name, VISION_MARKERS):
            parameter.requires_grad = False
        if freeze_aligner and _matches_any(name, ALIGNER_MARKERS):
            parameter.requires_grad = False
    counts = {
        "total": sum(_parameter_count(parameter) for parameter in model.parameters()),
        "trainable": sum(
            _parameter_count(parameter)
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "vision_trainable": sum(
            _parameter_count(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and _matches_any(name, VISION_MARKERS)
        ),
        "aligner_trainable": sum(
            _parameter_count(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and _matches_any(name, ALIGNER_MARKERS)
        ),
        "lora_trainable": sum(
            _parameter_count(parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name.lower()
        ),
    }
    if counts["trainable"] <= 0 or counts["lora_trainable"] <= 0:
        raise RuntimeError(f"LoRA injection produced no trainable parameters: {counts}")
    if freeze_vision and counts["vision_trainable"]:
        raise RuntimeError(f"vision parameters are unexpectedly trainable: {counts}")
    if freeze_aligner and counts["aligner_trainable"]:
        raise RuntimeError(f"aligner parameters are unexpectedly trainable: {counts}")
    unexpected = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora_" not in name.lower()
    ]
    if unexpected:
        raise RuntimeError(
            "default judge plan permits only LoRA parameters to train; unexpected="
            + ",".join(unexpected[:20])
        )
    return counts


def audit_language_lora_targets(
    model: Any,
    requested_targets: list[str] | tuple[str, ...],
) -> dict[str, int]:
    """Require every requested adapter family to exist on the language side.

    PEFT matches target names by suffix and can therefore also inject adapters
    into similarly named vision modules. Those modules are frozen separately;
    this audit proves that each requested attention/MLP family also has
    trainable language-side LoRA parameters.
    """

    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and "lora_" in name.lower()
        and not _matches_any(name, VISION_MARKERS)
        and not _matches_any(name, ALIGNER_MARKERS)
    ]
    counts = {
        target: sum(
            _parameter_count(parameter)
            for name, parameter in trainable
            if f".{target}." in name
        )
        for target in requested_targets
    }
    missing = [target for target, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(
            "requested language LoRA targets produced no trainable parameters: "
            + ",".join(missing)
        )
    return counts


def load_model_and_processor(args: argparse.Namespace) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    import transformers
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
    }
    dtype = torch.bfloat16
    model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_class is None:
        model_class = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_class is None:
        raise RuntimeError(
            "transformers must provide AutoModelForMultimodalLM or "
            "AutoModelForImageTextToText"
        )
    model_parameters = inspect.signature(model_class.from_pretrained).parameters
    if "dtype" in model_parameters:
        load_kwargs["dtype"] = dtype
    else:
        load_kwargs["torch_dtype"] = dtype
    model = model_class.from_pretrained(args.model_id, **load_kwargs)
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(args.lora_target_modules),
        ),
    )
    if args.gradient_checkpointing:
        enable_input_grads = getattr(model, "enable_input_require_grads", None)
        if callable(enable_input_grads):
            enable_input_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    counts = freeze_multimodal_adapters(
        model,
        freeze_vision=True,
        freeze_aligner=True,
    )
    counts["language_lora_by_target"] = audit_language_lora_targets(
        model,
        args.lora_target_modules,
    )
    return model, processor, counts


def _training_argument_kwargs(
    args: argparse.Namespace,
    parameter_names: set[str],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "adam_beta1": args.adam_beta1,
        "adam_beta2": args.adam_beta2,
        "adam_epsilon": args.adam_epsilon,
        "max_grad_norm": args.max_grad_norm,
        "lr_scheduler_type": args.lr_scheduler_type,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": True,
        "logging_steps": 1,
        "save_strategy": "steps" if args.max_steps > 0 else "epoch",
        "save_steps": args.max_steps if args.max_steps > 0 else 500,
        "max_steps": args.max_steps,
        "seed": args.seed,
        "data_seed": args.seed,
        "report_to": [],
        "label_names": ["labels"],
        "optim": "adamw_torch_fused",
        "load_best_model_at_end": False,
    }
    if "warmup_ratio" in parameter_names:
        kwargs["warmup_ratio"] = args.warmup_ratio
    elif "warmup_steps" in parameter_names:
        # Transformers v5.2+ removed warmup_ratio and accepts a float ratio
        # directly through warmup_steps.
        kwargs["warmup_steps"] = args.warmup_ratio
    else:
        raise RuntimeError(
            "installed transformers TrainingArguments supports neither "
            "warmup_ratio nor warmup_steps"
        )
    eval_name = "eval_strategy" if "eval_strategy" in parameter_names else "evaluation_strategy"
    kwargs[eval_name] = "no"
    if args.deepspeed is not None:
        kwargs["deepspeed"] = str(args.deepspeed)
    return kwargs


def training_arguments(args: argparse.Namespace) -> Any:
    from transformers import TrainingArguments

    parameter_names = set(inspect.signature(TrainingArguments.__init__).parameters)
    kwargs = _training_argument_kwargs(args, parameter_names)
    return TrainingArguments(**kwargs)


def _jsonable_weights(values: dict[Any, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        label = key.value if hasattr(key, "value") else str(key)
        if hasattr(value, "__dict__"):
            result[label] = dict(value.__dict__)
        else:
            result[label] = value
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="No-head PASS/FAIL token training for EgoLife judges"
    )
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=DEFAULTS.model_id)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--min-pixels", type=int, default=DEFAULTS.min_pixels)
    parser.add_argument("--max-pixels", type=int, default=DEFAULTS.max_pixels)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULTS.max_input_tokens)
    parser.add_argument(
        "--image-context-target-fraction",
        type=float,
        default=DEFAULT_IMAGE_CONTEXT_TARGET_FRACTION,
    )
    parser.add_argument(
        "--image-text-token-reserve",
        type=int,
        default=DEFAULT_IMAGE_TEXT_TOKEN_RESERVE,
    )
    parser.add_argument(
        "--image-item-token-overhead",
        type=int,
        default=DEFAULT_IMAGE_ITEM_TOKEN_OVERHEAD,
    )
    parser.add_argument("--lora-rank", type=int, default=DEFAULTS.lora_rank)
    parser.add_argument("--lora-alpha", type=int, default=DEFAULTS.lora_alpha)
    parser.add_argument("--lora-dropout", type=float, default=DEFAULTS.lora_dropout)
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=list(DEFAULTS.lora_target_modules),
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULTS.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS.weight_decay)
    parser.add_argument("--adam-beta1", type=float, default=DEFAULTS.adam_beta1)
    parser.add_argument("--adam-beta2", type=float, default=DEFAULTS.adam_beta2)
    parser.add_argument("--adam-epsilon", type=float, default=DEFAULTS.adam_epsilon)
    parser.add_argument("--epochs", type=float, default=DEFAULTS.epochs)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive values override epochs; intended for bounded smoke runs.",
    )
    parser.add_argument(
        "--train-example-id",
        help=(
            "Optimize only this example after computing class/task weights from the full "
            "manifest. Intended for deterministic one-example smoke runs."
        ),
    )
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULTS.warmup_ratio)
    parser.add_argument("--lr-scheduler-type", default=DEFAULTS.lr_scheduler_type)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=DEFAULTS.gradient_accumulation_steps,
    )
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULTS.max_grad_norm)
    parser.add_argument(
        "--class-weight-smoothing",
        type=float,
        default=DEFAULTS.class_weight_smoothing,
    )
    parser.add_argument(
        "--max-class-weight", type=float, default=DEFAULTS.max_class_weight
    )
    parser.add_argument("--formality-weight", type=float, default=0.2)
    parser.add_argument("--groundedness-weight", type=float, default=0.4)
    parser.add_argument("--answerability-weight", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=DEFAULTS.seed)
    parser.add_argument("--deepspeed", type=Path)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=DEFAULTS.gradient_checkpointing,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if args.max_steps == 0 or args.max_steps < -1:
        raise ValueError("max_steps must be -1 or a positive integer")
    train_examples = load_normalized_manifest(args.train_manifest)
    task_weights = {
        JudgeTask.FORMALITY: args.formality_weight,
        JudgeTask.GROUNDEDNESS: args.groundedness_weight,
        JudgeTask.ANSWERABILITY: args.answerability_weight,
    }
    task_weights = validate_task_weights(task_weights)
    class_weights = class_weights_by_task(
        train_examples,
        smoothing=args.class_weight_smoothing,
        max_weight=args.max_class_weight,
    )
    task_scales = task_sampling_scales(train_examples, task_weights)
    optimization_examples = train_examples
    if args.train_example_id:
        optimization_examples = [
            example
            for example in train_examples
            if example.example_id == args.train_example_id
        ]
        if len(optimization_examples) != 1:
            raise ValueError(
                "train_example_id must match exactly one manifest row; "
                f"id={args.train_example_id!r} matches={len(optimization_examples)}"
            )
    launch_world_size = int(os.environ.get("WORLD_SIZE", "1"))
    sampler_audit = audit_rank_aligned_sampler(
        JudgeDataset(optimization_examples),
        world_size=launch_world_size,
        seed=args.seed,
    )
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "rank_aligned_sampler_preflight="
            + json.dumps(sampler_audit, sort_keys=True),
            flush=True,
        )
        visual_budget_audit = {
            "max_input_tokens": args.max_input_tokens,
            "target_fraction": args.image_context_target_fraction,
            "all_six_1800_frame_max_pixels": adaptive_image_max_pixels(
                image_count=1_800,
                configured_max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                max_input_tokens=args.max_input_tokens,
                target_fraction=args.image_context_target_fraction,
                text_token_reserve=args.image_text_token_reserve,
                item_token_overhead=args.image_item_token_overhead,
                vision_token_pixel_area=QWEN_VISION_TOKEN_PIXEL_AREA,
            ),
            "speaker_300_frame_max_pixels": adaptive_image_max_pixels(
                image_count=300,
                configured_max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                max_input_tokens=args.max_input_tokens,
                target_fraction=args.image_context_target_fraction,
                text_token_reserve=args.image_text_token_reserve,
                item_token_overhead=args.image_item_token_overhead,
                vision_token_pixel_area=QWEN_VISION_TOKEN_PIXEL_AREA,
            ),
        }
        print(
            "judge_visual_budget_preflight="
            + json.dumps(visual_budget_audit, sort_keys=True),
            flush=True,
        )
    # Constructing TrainingArguments first activates Transformers' ZeRO-3 model
    # initialization path before the 27B checkpoint is loaded on every rank.
    hf_training_args = training_arguments(args)
    if int(hf_training_args.world_size) != launch_world_size:
        raise RuntimeError(
            "TrainingArguments world size disagrees with torchrun: "
            f"arguments={hf_training_args.world_size} launch={launch_world_size}"
        )
    model, processor, parameter_counts = load_model_and_processor(args)
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = resolve_verdict_token_ids(tokenizer)
    collator = JudgeFrameCollator(
        processor=processor,
        class_weights=class_weights,
        task_scales=task_scales,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_input_tokens=args.max_input_tokens,
        image_context_target_fraction=args.image_context_target_fraction,
        image_text_token_reserve=args.image_text_token_reserve,
        image_item_token_overhead=args.image_item_token_overhead,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_contract = {
        "contract_version": "verdict_token_bce_sampled_frames_v4",
        "model_id": args.model_id,
        "media_contract": (
            "packet-owned 300-frame user timelines sampled at 0.5 FPS; "
            "each timeline is one pre-sampled Qwen video block; one speaker "
            "block or six complete video blocks"
        ),
        "assistant_prefix": '{"verdict":"',
        "inference_generation_contract": (
            "lock pass/fail from first-token logits, then continue the same "
            "autoregressive generation through the complete JSON contract"
        ),
        "verdict_token_ids": {key.value: value for key, value in token_ids.items()},
        "loss": "BCEWithLogits(logit_pass - logit_fail)",
        "distributed_sampler_contract": {
            "policy": "rank-aligned homogeneous model execution paths",
            "signature": "(frame_set_count, frame_count)",
            "world_size": int(hf_training_args.world_size),
            "incomplete_global_microbatch": (
                "repeat a same-signature example with zero loss weight"
            ),
            "preflight": sampler_audit,
        },
        "task_weights": _jsonable_weights(task_weights),
        "class_weights": _jsonable_weights(class_weights),
        "task_sampling_scales": _jsonable_weights(task_scales),
        "train": manifest_summary(train_examples),
        "optimization_train": manifest_summary(optimization_examples),
        "internal_validation": None,
        "checkpoint_policy": (
            "save and retain every epoch checkpoint; select later with separately "
            "labeled standalone validation and test sets"
        ),
        "parameter_counts": parameter_counts,
        "hyperparameters": DEFAULTS.to_dict() | vars(args),
    }
    # Path values are stringified explicitly to keep the audit manifest portable.
    run_contract["hyperparameters"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in run_contract["hyperparameters"].items()
    }
    is_world_process_zero = int(os.environ.get("RANK", "0")) == 0
    if is_world_process_zero:
        (args.output_dir / "training_contract.json").write_text(
            json.dumps(run_contract, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    TrainerClass = build_verdict_trainer_class()
    trainer = TrainerClass(
        model=model,
        args=hf_training_args,
        train_dataset=JudgeDataset(optimization_examples),
        data_collator=collator,
        verdict_token_ids=token_ids,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_state()
    trainer.save_model(str(args.output_dir / "final_adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(str(args.output_dir / "final_adapter"))


if __name__ == "__main__":
    main()
