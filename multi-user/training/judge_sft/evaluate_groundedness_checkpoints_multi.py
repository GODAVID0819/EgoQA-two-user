#!/usr/bin/env python3
"""TP=2 multi-checkpoint Groundedness evaluation with shared input preprocessing."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor, DistributedConfig

from .collator import JudgeFrameCollator
from .contracts import JudgeTask, Verdict, validate_task_weights
from .data import load_normalized_manifest
from .loss import (
    class_weights_by_task,
    resolve_verdict_token_ids,
    task_sampling_scales,
)
from .trainer import (
    final_logits,
    logits_to_keep_argument,
    media_locality_key,
)
from .train import (
    audit_tensor_parallel_materialization,
    configure_safe_tensor_parallel_plan,
    ensure_tensor_parallel_metadata,
    materialize_lora_tensor_parallelism,
    tensor_parallel_mesh,
)

# Reuse the already-tested single-checkpoint helper logic.
from .evaluate_groundedness_checkpoint import (
    assert_same_example,
    move_to_device,
    raw_bce,
    summarize,
    task_value,
    verdict_value,
)


def validate_adapter_configs(adapter_dirs: list[Path]) -> None:
    """Require all checkpoints to have the same LoRA architecture."""
    reference = None
    reference_path = None

    fields = (
        "r",
        "lora_alpha",
        "target_modules",
        "layers_to_transform",
        "layers_pattern",
        "peft_type",
        "task_type",
    )

    for path in adapter_dirs:
        cfg_path = path / "adapter_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)

        raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        normalized = {
            key: raw.get(key)
            for key in fields
        }

        # target_modules may serialize in a different ordering.
        if isinstance(normalized["target_modules"], list):
            normalized["target_modules"] = sorted(
                normalized["target_modules"]
            )

        if reference is None:
            reference = normalized
            reference_path = path
        elif normalized != reference:
            raise RuntimeError(
                "Adapter architectures differ:\n"
                f"reference={reference_path}\n"
                f"candidate={path}\n"
                f"reference_config={reference}\n"
                f"candidate_config={normalized}"
            )


def load_multi_adapter_model(
    *,
    model_id: str,
    adapter_dirs: list[Path],
    requested_targets: list[str],
    attn_implementation: str,
    tp_size: int,
):
    """
    Load the 27B base exactly once, then attach all LoRA checkpoints.

    The first checkpoint uses PEFT's default adapter name. Subsequent
    checkpoints get ckpt_1, ckpt_2, ... .
    """
    validate_adapter_configs(adapter_dirs)

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )

    config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )

    tp_plan_audit = configure_safe_tensor_parallel_plan(config)

    model_class = getattr(
        transformers,
        "AutoModelForMultimodalLM",
        None,
    )
    if model_class is None:
        model_class = getattr(
            transformers,
            "AutoModelForImageTextToText",
            None,
        )
    if model_class is None:
        raise RuntimeError(
            "No supported Qwen multimodal auto-model class"
        )

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
        "config": config,
        "distributed_config": DistributedConfig(
            tp_size=tp_size
        ),
    }

    parameters = inspect.signature(
        model_class.from_pretrained
    ).parameters

    if "dtype" in parameters:
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    base = model_class.from_pretrained(
        model_id,
        **kwargs,
    )
    base.config.use_cache = False

    base_tp_audit = audit_tensor_parallel_materialization(
        base,
        requested_targets,
    )

    tp_metadata = ensure_tensor_parallel_metadata(
        base,
        requested_tp_size=tp_size,
    )

    # First adapter.
    model = PeftModel.from_pretrained(
        base,
        str(adapter_dirs[0]),
        is_trainable=False,
        autocast_adapter_dtype=False,
    )

    internal_names = ["default"]

    # Remaining adapters. Do this BEFORE LoRA TP materialization so all
    # four adapter families are sharded together.
    load_signature = inspect.signature(
        model.load_adapter
    ).parameters

    for index, adapter_dir in enumerate(
        adapter_dirs[1:],
        start=1,
    ):
        internal_name = f"ckpt_{index}"

        load_kwargs: dict[str, Any] = {
            "adapter_name": internal_name,
        }

        if "is_trainable" in load_signature:
            load_kwargs["is_trainable"] = False

        if "autocast_adapter_dtype" in load_signature:
            load_kwargs[
                "autocast_adapter_dtype"
            ] = False

        model.load_adapter(
            str(adapter_dir),
            **load_kwargs,
        )
        internal_names.append(internal_name)

    base_model = model.get_base_model()

    mesh_source = getattr(
        base_model,
        "_device_mesh",
        None,
    )
    if mesh_source is None:
        mesh_source = getattr(
            base,
            "_device_mesh",
            None,
        )
    if mesh_source is None:
        raise RuntimeError(
            "TP model exposes no device mesh"
        )

    tp_mesh = tensor_parallel_mesh(mesh_source)

    # Important: this now materializes LoRA factors for ALL loaded
    # adapters, not merely the first one.
    lora_tp_audit = materialize_lora_tensor_parallelism(
        model,
        tp_mesh=tp_mesh,
        tp_size=tp_size,
        requested_targets=requested_targets,
    )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    audit = {
        "tp_plan": tp_plan_audit,
        "base_tp": base_tp_audit,
        "tp_metadata": tp_metadata,
        "lora_tp": lora_tp_audit,
        "adapter_count": len(adapter_dirs),
        "internal_adapter_names": internal_names,
    }

    return (
        model,
        processor,
        internal_names,
        audit,
    )


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--eval-manifest",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--train-manifest",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--training-contract",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--adapter-dirs",
        type=Path,
        nargs="+",
        required=True,
    )
    ap.add_argument(
        "--adapter-labels",
        nargs="+",
        default=None,
    )
    ap.add_argument(
        "--model-id",
        required=True,
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    ap.add_argument(
        "--attn-implementation",
        default="sdpa",
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=0,
    )
    ap.add_argument(
        "--decoded-image-cache-entries",
        type=int,
        default=2,
    )

    args = ap.parse_args()

    if len(args.adapter_dirs) < 2:
        raise ValueError(
            "multi-checkpoint evaluator requires >=2 adapters"
        )

    if args.adapter_labels is None:
        labels = [
            path.name
            for path in args.adapter_dirs
        ]
    else:
        labels = list(args.adapter_labels)
        if len(labels) != len(args.adapter_dirs):
            raise ValueError(
                "--adapter-labels count must match "
                "--adapter-dirs count"
            )

    if len(set(labels)) != len(labels):
        raise ValueError(
            "adapter labels must be unique"
        )

    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
        )

    rank = dist.get_rank()
    world = dist.get_world_size()

    if world != 2:
        raise RuntimeError(
            f"Expected TP2 world size=2, got {world}"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(
        "cuda",
        local_rank,
    )

    contract = json.loads(
        args.training_contract.read_text(
            encoding="utf-8"
        )
    )
    hp = contract["hyperparameters"]

    requested_targets = list(
        hp["lora_target_modules"]
    )

    train_examples = load_normalized_manifest(
        args.train_manifest
    )
    eval_examples = load_normalized_manifest(
        args.eval_manifest
    )

    bad_tasks = [
        getattr(ex, "example_id", "<unknown>")
        for ex in eval_examples
        if task_value(ex) != "groundedness"
    ]
    if bad_tasks:
        raise RuntimeError(
            "Eval manifest is not Groundedness-only; "
            f"{len(bad_tasks)} bad rows"
        )

    train_ids = {
        str(getattr(ex, "example_id"))
        for ex in train_examples
    }
    eval_ids = {
        str(getattr(ex, "example_id"))
        for ex in eval_examples
    }

    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(
            "TRAIN/EVAL LEAKAGE: "
            f"{len(overlap)} overlapping example IDs"
        )

    # Put questions sharing the same visual frame input next to
    # each other, so JudgeFrameCollator's decoded-image cache can
    # actually reuse those images.
    eval_examples = sorted(
        eval_examples,
        key=lambda ex: (
            media_locality_key(ex),
            str(getattr(ex, "example_id")),
        ),
    )

    if args.max_examples > 0:
        eval_examples = eval_examples[
            : args.max_examples
        ]

    contract_task_weights = (
        contract.get("task_weights") or {}
    )

    if set(contract_task_weights) >= {
        "formality",
        "groundedness",
        "answerability",
    }:
        task_weights = validate_task_weights(
            {
                JudgeTask.FORMALITY: float(
                    contract_task_weights[
                        "formality"
                    ]
                ),
                JudgeTask.GROUNDEDNESS: float(
                    contract_task_weights[
                        "groundedness"
                    ]
                ),
                JudgeTask.ANSWERABILITY: float(
                    contract_task_weights[
                        "answerability"
                    ]
                ),
            }
        )
    else:
        task_weights = validate_task_weights(
            {
                JudgeTask.FORMALITY: 0.2,
                JudgeTask.GROUNDEDNESS: 0.4,
                JudgeTask.ANSWERABILITY: 0.4,
            }
        )

    class_weights = class_weights_by_task(
        train_examples,
        smoothing=float(
            hp["class_weight_smoothing"]
        ),
        max_weight=float(
            hp["max_class_weight"]
        ),
    )

    task_scales = task_sampling_scales(
        train_examples,
        task_weights,
    )

    if rank == 0:
        print(
            f"train_examples={len(train_examples)}",
            flush=True,
        )
        print(
            f"eval_examples={len(eval_examples)}",
            flush=True,
        )
        print(
            f"adapters={len(args.adapter_dirs)}",
            flush=True,
        )

        for label, path in zip(
            labels,
            args.adapter_dirs,
        ):
            print(
                f"  {label}: {path}",
                flush=True,
            )

    (
        model,
        processor,
        internal_names,
        model_audit,
    ) = load_multi_adapter_model(
        model_id=args.model_id,
        adapter_dirs=list(args.adapter_dirs),
        requested_targets=requested_targets,
        attn_implementation=(
            args.attn_implementation
        ),
        tp_size=2,
    )

    tokenizer = getattr(
        processor,
        "tokenizer",
        processor,
    )

    token_ids = resolve_verdict_token_ids(
        tokenizer
    )

    keep_arg = logits_to_keep_argument(model)

    collator = JudgeFrameCollator(
        processor=processor,
        class_weights=class_weights,
        task_scales=task_scales,
        min_pixels=int(hp["min_pixels"]),
        max_pixels=int(hp["max_pixels"]),
        max_input_tokens=int(
            hp["max_input_tokens"]
        ),
        image_context_target_fraction=float(
            hp["image_context_target_fraction"]
        ),
        image_text_token_reserve=int(
            hp["image_text_token_reserve"]
        ),
        image_item_token_overhead=int(
            hp["image_item_token_overhead"]
        ),
        decoded_image_cache_entries=(
            args.decoded_image_cache_entries
        ),
    )

    rows_by_label: dict[
        str,
        list[dict[str, Any]],
    ] = {
        label: []
        for label in labels
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_path = (
        args.output_dir
        / "predictions.jsonl"
    )

    if rank == 0:
        prediction_path.write_text(
            "",
            encoding="utf-8",
        )

        (
            args.output_dir
            / "eval_contract.json"
        ).write_text(
            json.dumps(
                {
                    "eval_manifest": str(
                        args.eval_manifest
                    ),
                    "train_manifest": str(
                        args.train_manifest
                    ),
                    "training_contract": str(
                        args.training_contract
                    ),
                    "adapter_dirs": {
                        label: str(path)
                        for label, path in zip(
                            labels,
                            args.adapter_dirs,
                        )
                    },
                    "model_id": args.model_id,
                    "eval_examples": len(
                        eval_examples
                    ),
                    "train_eval_overlap": 0,
                    "decoded_image_cache_entries":
                        args.decoded_image_cache_entries,
                    "verdict_token_ids": {
                        key.value: int(value)
                        for key, value
                        in token_ids.items()
                    },
                    "model_audit": model_audit,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    dist.barrier()

    started = time.time()

    with torch.inference_mode():
        for index, ex in enumerate(
            eval_examples,
            start=1,
        ):
            example_id = str(
                getattr(ex, "example_id")
            )

            assert_same_example(
                example_id,
                device,
            )

            collate_started = time.time()

            # IMPORTANT:
            # CPU decode + processor happen ONCE for this question.
            batch = collator([ex])

            batch.pop("labels", None)
            batch.pop(
                "sample_weights",
                None,
            )
            batch.pop(
                "tp_example_fingerprint",
                None,
            )

            batch = move_to_device(
                batch,
                device,
            )

            batch[keep_arg] = 1

            collate_seconds = (
                time.time()
                - collate_started
            )

            gold = int(
                verdict_value(ex) == "pass"
            )

            display_parts = []

            # IMPORTANT:
            # Same prepared input is now reused across all checkpoints.
            for (
                label,
                internal_name,
            ) in zip(
                labels,
                internal_names,
            ):
                model.set_adapter(
                    internal_name
                )

                forward_started = time.time()

                outputs = model(
                    **batch,
                    return_dict=True,
                )

                logits = final_logits(
                    outputs
                )

                z_pass = float(
                    logits[
                        :,
                        token_ids[
                            Verdict.PASS
                        ],
                    ]
                    .detach()
                    .float()
                    .item()
                )

                z_fail = float(
                    logits[
                        :,
                        token_ids[
                            Verdict.FAIL
                        ],
                    ]
                    .detach()
                    .float()
                    .item()
                )

                margin = (
                    z_pass - z_fail
                )
                pred = int(
                    margin > 0.0
                )

                if margin >= 0:
                    p_pass = (
                        1.0
                        / (
                            1.0
                            + math.exp(
                                -margin
                            )
                        )
                    )
                else:
                    exp_margin = math.exp(
                        margin
                    )
                    p_pass = (
                        exp_margin
                        / (1.0 + exp_margin)
                    )

                result = {
                    "adapter": label,
                    "index": index,
                    "example_id": example_id,
                    "task": task_value(ex),
                    "frame_count": int(
                        getattr(
                            ex,
                            "frame_count",
                            0,
                        )
                    ),
                    "gold": gold,
                    "pred": pred,
                    "z_pass": z_pass,
                    "z_fail": z_fail,
                    "margin": margin,
                    "p_pass": p_pass,
                    "raw_bce": raw_bce(
                        margin,
                        gold,
                    ),
                    "collate_seconds":
                        collate_seconds,
                    "forward_seconds": (
                        time.time()
                        - forward_started
                    ),
                }

                rows_by_label[
                    label
                ].append(result)

                if rank == 0:
                    with prediction_path.open(
                        "a",
                        encoding="utf-8",
                    ) as handle:
                        handle.write(
                            json.dumps(
                                result,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                display_parts.append(
                    f"{label}:"
                    f"{pred}/"
                    f"{margin:+.3f}"
                )

                del outputs, logits

            if rank == 0:
                print(
                    f"[{index:3d}/"
                    f"{len(eval_examples)}] "
                    f"gold={gold} "
                    f"collate="
                    f"{collate_seconds:.1f}s "
                    + " | ".join(
                        display_parts
                    ),
                    flush=True,
                )

            del batch

            if index % 4 == 0:
                torch.cuda.empty_cache()

    dist.barrier()

    if rank == 0:
        wall_seconds = (
            time.time() - started
        )

        model_summaries = {}

        for label in labels:
            summary = summarize(
                rows_by_label[label]
            )
            model_summaries[
                label
            ] = summary

        combined = {
            "models": model_summaries,
            "wall_seconds": wall_seconds,
            "eval_examples": len(
                eval_examples
            ),
            "adapter_count": len(
                labels
            ),
        }

        (
            args.output_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                combined,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 100)
        print(
            "MULTI-CHECKPOINT "
            "GROUNDEDNESS HELD-OUT"
        )
        print("=" * 100)

        print(
            f"{'MODEL':<12} "
            f"{'LOSS':>9} "
            f"{'ACC':>8} "
            f"{'BAL_ACC':>9} "
            f"{'PRED_P':>8} "
            f"{'R_PASS':>8} "
            f"{'R_FAIL':>8}"
        )

        print("-" * 75)

        def fmt(value, width=8):
            if value is None:
                return f"{'N/A':>{width}}"
            return f"{value:>{width}.4f}"

        for label in labels:
            s = model_summaries[label]

            print(
                f"{label:<12} "
                f"{fmt(s['raw_bce'], 9)} "
                f"{fmt(s['accuracy'], 8)} "
                f"{fmt(s['balanced_accuracy'], 9)} "
                f"{fmt(s['pred_pass_rate'], 8)} "
                f"{fmt(s['recall_pass'], 8)} "
                f"{fmt(s['recall_fail'], 8)}"
            )

        print(
            f"\nwall_seconds="
            f"{wall_seconds:.1f}"
        )

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
