#!/usr/bin/env python3
"""TP=2 first-token PASS/FAIL evaluation for Groundedness LoRA checkpoints."""

from __future__ import annotations

import argparse
import hashlib
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
from .trainer import final_logits, logits_to_keep_argument
from .train import (
    audit_tensor_parallel_materialization,
    configure_safe_tensor_parallel_plan,
    ensure_tensor_parallel_metadata,
    materialize_lora_tensor_parallelism,
    tensor_parallel_mesh,
)


def task_value(ex: Any) -> str:
    value = getattr(ex, "task")
    return (value.value if hasattr(value, "value") else str(value)).lower()


def verdict_value(ex: Any) -> str:
    value = getattr(ex, "verdict")
    return (value.value if hasattr(value, "value") else str(value)).lower()


def stable_fingerprint(example_id: str) -> int:
    digest = hashlib.sha256(example_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def assert_same_example(example_id: str, device: torch.device) -> None:
    value = torch.tensor(
        [stable_fingerprint(example_id)],
        dtype=torch.long,
        device=device,
    )
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    fingerprints = [int(x.item()) for x in gathered]
    if len(set(fingerprints)) != 1:
        raise RuntimeError(
            f"TP ranks received different examples: {fingerprints}"
        )


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def load_model(
    *,
    model_id: str,
    adapter_dir: Path,
    requested_targets: list[str],
    attn_implementation: str,
    tp_size: int,
):
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

    model_class = getattr(transformers, "AutoModelForMultimodalLM", None)
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
        "distributed_config": DistributedConfig(tp_size=tp_size),
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

    model = PeftModel.from_pretrained(
        base,
        str(adapter_dir),
        is_trainable=False,
        autocast_adapter_dtype=False,
    )

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

    tp_mesh = tensor_parallel_mesh(
        mesh_source
    )

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
    }

    return model, processor, audit


def raw_bce(
    margin: float,
    gold: int,
) -> float:
    return (
        max(margin, 0.0)
        - margin * float(gold)
        + math.log1p(
            math.exp(-abs(margin))
        )
    )


def summarize(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:

    n = len(rows)

    if n == 0:
        raise RuntimeError(
            "No evaluation rows were produced"
        )

    tp = sum(
        r["gold"] == 1
        and r["pred"] == 1
        for r in rows
    )

    tn = sum(
        r["gold"] == 0
        and r["pred"] == 0
        for r in rows
    )

    fp = sum(
        r["gold"] == 0
        and r["pred"] == 1
        for r in rows
    )

    fn = sum(
        r["gold"] == 1
        and r["pred"] == 0
        for r in rows
    )

    pass_total = tp + fn
    fail_total = tn + fp

    recall_pass = (
        tp / pass_total
        if pass_total
        else None
    )

    recall_fail = (
        tn / fail_total
        if fail_total
        else None
    )

    balanced_accuracy = (
        (recall_pass + recall_fail) / 2
        if (
            recall_pass is not None
            and recall_fail is not None
        )
        else None
    )

    return {
        "n": n,
        "raw_bce": (
            sum(
                r["raw_bce"]
                for r in rows
            ) / n
        ),
        "accuracy": (
            sum(
                r["pred"] == r["gold"]
                for r in rows
            ) / n
        ),
        "balanced_accuracy": balanced_accuracy,
        "gold_pass_rate": (
            sum(
                r["gold"]
                for r in rows
            ) / n
        ),
        "pred_pass_rate": (
            sum(
                r["pred"]
                for r in rows
            ) / n
        ),
        "recall_pass": recall_pass,
        "recall_fail": recall_fail,
        "mean_margin": (
            sum(
                r["margin"]
                for r in rows
            ) / n
        ),
        "mean_abs_margin": (
            sum(
                abs(r["margin"])
                for r in rows
            ) / n
        ),
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


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
        "--adapter-dir",
        type=Path,
        required=True,
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

    args = ap.parse_args()

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

    local_rank = int(
        os.environ["LOCAL_RANK"]
    )

    torch.cuda.set_device(
        local_rank
    )

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

    if args.max_examples > 0:
        eval_examples = eval_examples[
            : args.max_examples
        ]

    bad_tasks = [
        getattr(
            ex,
            "example_id",
            "<unknown>",
        )
        for ex in eval_examples
        if task_value(ex) != "groundedness"
    ]

    if bad_tasks:
        raise RuntimeError(
            "Eval manifest is not "
            "Groundedness-only; "
            f"{len(bad_tasks)} "
            "non-groundedness rows found"
        )

    train_ids = {
        str(
            getattr(
                ex,
                "example_id",
            )
        )
        for ex in train_examples
    }

    eval_ids = {
        str(
            getattr(
                ex,
                "example_id",
            )
        )
        for ex in eval_examples
    }

    overlap = train_ids & eval_ids

    if overlap:
        raise RuntimeError(
            "TRAIN/EVAL LEAKAGE: "
            f"{len(overlap)} "
            "overlapping example IDs"
        )

    contract_task_weights = (
        contract.get("task_weights")
        or {}
    )

    if set(contract_task_weights) >= {
        "formality",
        "groundedness",
        "answerability",
    }:
        task_weights = validate_task_weights(
            {
                JudgeTask.FORMALITY:
                    float(
                        contract_task_weights[
                            "formality"
                        ]
                    ),

                JudgeTask.GROUNDEDNESS:
                    float(
                        contract_task_weights[
                            "groundedness"
                        ]
                    ),

                JudgeTask.ANSWERABILITY:
                    float(
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
            f"train_examples="
            f"{len(train_examples)}",
            flush=True,
        )

        print(
            f"eval_examples="
            f"{len(eval_examples)}",
            flush=True,
        )

        print(
            f"train_eval_overlap="
            f"{len(overlap)}",
            flush=True,
        )

        print(
            f"adapter="
            f"{args.adapter_dir}",
            flush=True,
        )

    model, processor, model_audit = (
        load_model(
            model_id=args.model_id,
            adapter_dir=args.adapter_dir,
            requested_targets=requested_targets,
            attn_implementation=(
                args.attn_implementation
            ),
            tp_size=2,
        )
    )

    tokenizer = getattr(
        processor,
        "tokenizer",
        processor,
    )

    token_ids = (
        resolve_verdict_token_ids(
            tokenizer
        )
    )

    keep_arg = (
        logits_to_keep_argument(
            model
        )
    )

    collator = JudgeFrameCollator(
        processor=processor,
        class_weights=class_weights,
        task_scales=task_scales,

        min_pixels=int(
            hp["min_pixels"]
        ),

        max_pixels=int(
            hp["max_pixels"]
        ),

        max_input_tokens=int(
            hp["max_input_tokens"]
        ),

        image_context_target_fraction=float(
            hp[
                "image_context_target_fraction"
            ]
        ),

        image_text_token_reserve=int(
            hp[
                "image_text_token_reserve"
            ]
        ),

        image_item_token_overhead=int(
            hp[
                "image_item_token_overhead"
            ]
        ),
    )

    prediction_path = (
        args.output_dir
        / "predictions.jsonl"
    )

    if rank == 0:
        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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
                    "eval_manifest":
                        str(
                            args.eval_manifest
                        ),

                    "train_manifest":
                        str(
                            args.train_manifest
                        ),

                    "training_contract":
                        str(
                            args.training_contract
                        ),

                    "adapter_dir":
                        str(
                            args.adapter_dir
                        ),

                    "model_id":
                        args.model_id,

                    "eval_examples":
                        len(eval_examples),

                    "train_eval_overlap":
                        0,

                    "verdict_token_ids":
                        {
                            k.value: int(v)
                            for k, v
                            in token_ids.items()
                        },

                    "model_audit":
                        model_audit,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    rows: list[
        dict[str, Any]
    ] = []

    started = time.time()

    with torch.inference_mode():

        for index, ex in enumerate(
            eval_examples,
            start=1,
        ):

            example_id = str(
                getattr(
                    ex,
                    "example_id",
                )
            )

            assert_same_example(
                example_id,
                device,
            )

            t0 = time.time()

            batch = collator(
                [ex]
            )

            batch.pop(
                "labels",
                None,
            )

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
                z_pass
                - z_fail
            )

            gold = int(
                verdict_value(ex)
                == "pass"
            )

            pred = int(
                margin > 0.0
            )

            result = {
                "index": index,

                "example_id":
                    example_id,

                "task":
                    task_value(ex),

                "frame_count":
                    int(
                        getattr(
                            ex,
                            "frame_count",
                            0,
                        )
                    ),

                "gold":
                    gold,

                "pred":
                    pred,

                "z_pass":
                    z_pass,

                "z_fail":
                    z_fail,

                "margin":
                    margin,

                "p_pass":
                    (
                        1.0
                        /
                        (
                            1.0
                            + math.exp(
                                -margin
                            )
                        )
                    ),

                "raw_bce":
                    raw_bce(
                        margin,
                        gold,
                    ),

                "seconds":
                    time.time()
                    - t0,
            }

            rows.append(
                result
            )

            if rank == 0:

                with prediction_path.open(
                    "a",
                    encoding="utf-8",
                ) as f:

                    f.write(
                        json.dumps(
                            result,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                print(
                    f"[{index:3d}/"
                    f"{len(eval_examples)}] "
                    f"gold={gold} "
                    f"pred={pred} "
                    f"margin="
                    f"{margin:+.4f} "
                    f"sec="
                    f"{result['seconds']:.1f}",
                    flush=True,
                )

            del batch
            del outputs
            del logits

            if index % 4 == 0:
                torch.cuda.empty_cache()

    if rank == 0:

        summary = summarize(
            rows
        )

        summary["wall_seconds"] = (
            time.time()
            - started
        )

        (
            args.output_dir
            / "summary.json"
        ).write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()

        print(
            "=" * 72
        )

        print(
            "GROUNDEDNESS "
            "HELD-OUT RESULT"
        )

        print(
            "=" * 72
        )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

    dist.barrier()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
