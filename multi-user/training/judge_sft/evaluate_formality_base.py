#!/usr/bin/env python3

from __future__ import annotations

import argparse
import inspect
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import transformers
from transformers import AutoConfig, AutoProcessor, DistributedConfig

from .contracts import Verdict
from .loss import resolve_verdict_token_ids
from .trainer import final_logits, logits_to_keep_argument
from .train import (
    audit_tensor_parallel_materialization,
    configure_safe_tensor_parallel_plan,
    ensure_tensor_parallel_metadata,
)
from .evaluate_formality_checkpoint import (
    read_jsonl,
    assert_same_example,
    build_formality_inputs,
    move_to_device,
    raw_bce,
    summarize,
)


TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def load_base_model(
    *,
    model_id: str,
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
        raise RuntimeError("No supported Qwen multimodal model class")

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": attn_implementation,
        "config": config,
        "distributed_config": DistributedConfig(tp_size=tp_size),
    }

    params = inspect.signature(model_class.from_pretrained).parameters

    if "dtype" in params:
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = model_class.from_pretrained(
        model_id,
        **kwargs,
    )

    model.config.use_cache = False

    tp_materialization = audit_tensor_parallel_materialization(
        model,
        TARGETS,
    )

    tp_metadata = ensure_tensor_parallel_metadata(
        model,
        requested_tp_size=tp_size,
    )

    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, processor, {
        "tp_plan": tp_plan_audit,
        "tp_materialization": tp_materialization,
        "tp_metadata": tp_metadata,
    }


def main():
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
            f"Expected TP2, got world_size={world}"
        )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    train_rows = read_jsonl(args.train_manifest)
    eval_rows = read_jsonl(args.eval_manifest)

    train_ids = {str(r["example_id"]) for r in train_rows}
    eval_ids = {str(r["example_id"]) for r in eval_rows}

    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(
            f"TRAIN/EVAL LEAKAGE: {len(overlap)} overlapping examples"
        )

    if rank == 0:
        print(f"train_examples={len(train_rows)}", flush=True)
        print(f"eval_examples={len(eval_rows)}", flush=True)
        print(f"train_eval_overlap={len(overlap)}", flush=True)
        print("model=BASE", flush=True)

    model, processor, model_audit = load_base_model(
        model_id=args.model_id,
        attn_implementation=args.attn_implementation,
        tp_size=2,
    )

    tokenizer = getattr(processor, "tokenizer", processor)

    token_ids = resolve_verdict_token_ids(tokenizer)

    keep_arg = logits_to_keep_argument(model)

    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)

        pred_path = args.output_dir / "predictions.jsonl"
        pred_path.write_text("", encoding="utf-8")

        (args.output_dir / "eval_contract.json").write_text(
            json.dumps(
                {
                    "model": "base",
                    "model_id": args.model_id,
                    "eval_manifest": str(args.eval_manifest),
                    "train_manifest": str(args.train_manifest),
                    "eval_examples": len(eval_rows),
                    "train_eval_overlap": 0,
                    "verdict_token_ids": {
                        k.value: int(v)
                        for k, v in token_ids.items()
                    },
                    "model_audit": model_audit,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    results = []

    with torch.inference_mode():

        for idx, row in enumerate(eval_rows, start=1):

            example_id = str(row["example_id"])

            assert_same_example(
                example_id,
                device,
            )

            batch = build_formality_inputs(
                processor,
                str(row["prompt"]),
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

            logits = final_logits(outputs)

            z_pass = logits[:, token_ids[Verdict.PASS]]
            z_fail = logits[:, token_ids[Verdict.FAIL]]

            margin = float(
                (z_pass - z_fail)
                .detach()
                .float()
                .item()
            )

            gold = int(
                str(row["verdict"]).lower() == "pass"
            )

            pred = int(margin > 0.0)

            p_pass = 1.0 / (1.0 + __import__("math").exp(-margin))

            result = {
                "index": idx,
                "example_id": example_id,
                "gold": gold,
                "pred": pred,
                "margin": margin,
                "p_pass": p_pass,
                "raw_bce": raw_bce(margin, gold),
            }

            results.append(result)

            if rank == 0:
                with pred_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(result) + "\n")

                if idx == 1 or idx % 10 == 0 or idx == len(eval_rows):
                    print(
                        f"[{idx:3d}/{len(eval_rows)}] "
                        f"gold={gold} "
                        f"pred={pred} "
                        f"margin={margin:+.4f}",
                        flush=True,
                    )

            del batch, outputs, logits

    if rank == 0:
        summary = summarize(results)

        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print()
        print("=" * 72)
        print("BASE FORMALITY HELD-OUT RESULT")
        print("=" * 72)
        print(json.dumps(summary, indent=2))

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
