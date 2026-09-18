"""Processor/tokenizer/1,800-frame smoke probe without loading model weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

from .collator import JudgeFrameCollator, adaptive_image_max_pixels
from .contracts import DEFAULT_TASK_WEIGHTS, JudgeTask, Verdict
from .data import load_normalized_manifest
from .loss import (
    class_weights_by_task,
    resolve_verdict_token_ids,
    task_sampling_scales,
)


def _shape_summary(value: Any) -> dict[str, Any]:
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    device = getattr(value, "device", None)
    return {
        "type": type(value).__name__,
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
        "device": str(device) if device is not None else None,
    }


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers
    from transformers import AutoConfig, AutoProcessor

    examples = load_normalized_manifest(args.manifest)
    selected = (
        [example for example in examples if example.example_id == args.example_id]
        if args.example_id
        else [
            example
            for example in examples
            if example.task is JudgeTask.ANSWERABILITY
            and example.condition_type == "combined_all_six_users"
        ][:1]
    )
    if len(selected) != 1:
        raise ValueError(
            f"example_id={args.example_id!r} must match exactly once; matches={len(selected)}"
        )
    example = selected[0]
    if len(example.frame_sets) != 6 or example.frame_count != 1_800:
        raise ValueError(
            "runtime probe must exercise six complete 300-frame timelines"
        )

    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = resolve_verdict_token_ids(tokenizer)
    class_weights = class_weights_by_task(examples)
    task_scales = task_sampling_scales(examples, DEFAULT_TASK_WEIGHTS)
    collator = JudgeFrameCollator(
        processor=processor,
        class_weights=class_weights,
        task_scales=task_scales,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_input_tokens=args.max_input_tokens,
    )
    batch = collator([example])
    input_tokens = int(batch["input_ids"].shape[-1])
    result = {
        "schema_version": "judge_sft_sampled_frame_runtime_probe_v2",
        "status": "passed",
        "model_id": args.model_id,
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", None) or []),
        "auto_multimodal_model_available": bool(
            getattr(transformers, "AutoModelForMultimodalLM", None)
        ),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "versions": {
            name: _version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "accelerate",
                "deepspeed",
                "qwen-vl-utils",
                "torchcodec",
            )
        },
        "cuda": {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "bf16_supported": bool(
                torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            ),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        },
        "frame_contract": {
            "user_timeline_count": len(example.frame_sets),
            "frames_per_user": [len(value.frames) for value in example.frame_sets],
            "frame_count": example.frame_count,
            "source_fps": 0.5,
            "min_pixels": args.min_pixels,
            "configured_max_pixels": args.max_pixels,
            "effective_max_pixels": adaptive_image_max_pixels(
                image_count=example.frame_count,
                configured_max_pixels=args.max_pixels,
                min_pixels=args.min_pixels,
                max_input_tokens=args.max_input_tokens,
            ),
            "max_input_tokens": args.max_input_tokens,
            "actual_input_tokens": input_tokens,
        },
        "verdict_token_ids": {
            verdict.value: token_ids[verdict]
            for verdict in (Verdict.FAIL, Verdict.PASS)
        },
        "batch": {
            key: _shape_summary(value)
            for key, value in batch.items()
        },
    }
    if not result["cuda"]["available"] or not result["cuda"]["bf16_supported"]:
        raise RuntimeError("runtime probe requires CUDA with BF16 support")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe judge SFT six-timeline preprocessing")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--example-id")
    parser.add_argument("--min-pixels", type=int, default=3_136)
    parser.add_argument("--max-pixels", type=int, default=262_144)
    parser.add_argument("--max-input-tokens", type=int, default=262_144)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = run_probe(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
