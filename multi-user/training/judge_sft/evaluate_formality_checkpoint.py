#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import transformers
from peft import PeftModel
from transformers import (
    AutoConfig,
    AutoProcessor,
    DistributedConfig,
)

from .collator import (
    _apply_chat_template,
    _assert_thinking_disabled,
)
from .contracts import VERDICT_ASSISTANT_PREFIX, Verdict
from .loss import resolve_verdict_token_ids
from .trainer import final_logits, logits_to_keep_argument
from .train import (
    audit_tensor_parallel_materialization,
    configure_safe_tensor_parallel_plan,
    ensure_tensor_parallel_metadata,
    materialize_lora_tensor_parallelism,
    tensor_parallel_mesh,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)

            if row.get("task") != "formality":
                raise ValueError(
                    f"{path}:{line_number}: expected task=formality, "
                    f"got {row.get('task')!r}"
                )

            verdict = str(row.get("verdict") or "").lower()
            if verdict not in {"pass", "fail"}:
                raise ValueError(
                    f"{path}:{line_number}: bad verdict={verdict!r}"
                )

            if not str(row.get("prompt") or "").strip():
                raise ValueError(
                    f"{path}:{line_number}: empty prompt"
                )

            rows.append(row)

    if not rows:
        raise ValueError(f"no rows loaded from {path}")

    return rows


def stable_fingerprint(example_id: str) -> int:
    digest = hashlib.sha256(
        example_id.encode("utf-8")
    ).digest()
    return (
        int.from_bytes(digest[:8], "big")
        & ((1 << 63) - 1)
    )


def assert_same_example(example_id: str, device: torch.device) -> None:
    x = torch.tensor(
        [stable_fingerprint(example_id)],
        dtype=torch.long,
        device=device,
    )

    gathered = [
        torch.empty_like(x)
        for _ in range(dist.get_world_size())
    ]
    dist.all_gather(gathered, x)

    values = [int(t.item()) for t in gathered]
    if len(set(values)) != 1:
        raise RuntimeError(
            f"TP ranks received different examples: {values}"
        )


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            k: move_to_device(v, device)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [
            move_to_device(v, device)
            for v in value
        ]
    if isinstance(value, tuple):
        return tuple(
            move_to_device(v, device)
            for v in value
        )
    return value


def build_formality_inputs(
    processor: Any,
    prompt: str,
) -> dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    rendered = _apply_chat_template(
        processor,
        messages,
    )
    _assert_thinking_disabled(rendered)

    rendered += VERDICT_ASSISTANT_PREFIX

    batch = processor(
        text=[rendered],
        padding=True,
        return_tensors="pt",
    )

    # Formality must remain purely text-only.
    forbidden = {
        "pixel_values",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
    }
    bad = sorted(forbidden & set(batch.keys()))
    if bad:
        raise RuntimeError(
            f"Formality unexpectedly produced visual tensors: {bad}"
        )

    return dict(batch)


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

    tp_plan_audit = configure_safe_tensor_parallel_plan(
        config
    )

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

    tp_mesh = tensor_parallel_mesh(mesh_source)

    lora_tp_audit = materialize_lora_tensor_parallelism(
        model,
        tp_mesh=tp_mesh,
        tp_size=tp_size,
        requested_targets=requested_targets,
    )

    model.eval()

    for param in model.parameters():
        param.requires_grad_(False)

    audit = {
        "tp_plan": tp_plan_audit,
        "base_tp": base_tp_audit,
        "tp_metadata": tp_metadata,
        "lora_tp": lora_tp_audit,
    }

    return model, processor, audit


def raw_bce(margin: float, gold: int) -> float:
    return (
        max(margin, 0.0)
        - margin * float(gold)
        + math.log1p(math.exp(-abs(margin)))
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)

    tp = sum(
        r["gold"] == 1 and r["pred"] == 1
        for r in rows
    )
    tn = sum(
        r["gold"] == 0 and r["pred"] == 0
        for r in rows
    )
    fp = sum(
        r["gold"] == 0 and r["pred"] == 1
        for r in rows
    )
    fn = sum(
        r["gold"] == 1 and r["pred"] == 0
        for r in rows
    )

    pass_total = tp + fn
    fail_total = tn + fp

    pass_recall = (
        tp / pass_total
        if pass_total else None
    )
    fail_recall = (
        tn / fail_total
        if fail_total else None
    )

    balanced_accuracy = (
        (pass_recall + fail_recall) / 2
        if pass_recall is not None
        and fail_recall is not None
        else None
    )

    return {
        "n": n,
        "raw_bce": (
            sum(r["raw_bce"] for r in rows) / n
        ),
        "accuracy": (
            sum(r["pred"] == r["gold"] for r in rows)
            / n
        ),
        "balanced_accuracy": balanced_accuracy,
        "gold_pass_rate": (
            sum(r["gold"] for r in rows) / n
        ),
        "pred_pass_rate": (
            sum(r["pred"] for r in rows) / n
        ),
        "pass_recall": pass_recall,
        "fail_recall": fail_recall,
        "mean_margin": (
            sum(r["margin"] for r in rows) / n
        ),
        "mean_abs_margin": (
            sum(abs(r["margin"]) for r in rows) / n
        ),
        "confusion": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
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

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(
        "cuda",
        local_rank,
    )

    train_rows = read_jsonl(args.train_manifest)
    eval_rows = read_jsonl(args.eval_manifest)

    train_ids = {
        str(x["example_id"])
        for x in train_rows
    }
    eval_ids = {
        str(x["example_id"])
        for x in eval_rows
    }

    overlap = train_ids & eval_ids
    if overlap:
        raise RuntimeError(
            f"TRAIN/EVAL LEAKAGE: {len(overlap)} overlapping example IDs"
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

    if rank == 0:
        print(
            f"train_examples={len(train_rows)}",
            flush=True,
        )
        print(
            f"eval_examples={len(eval_rows)}",
            flush=True,
        )
        print(
            f"train_eval_overlap={len(overlap)}",
            flush=True,
        )
        print(
            f"adapter={args.adapter_dir}",
            flush=True,
        )

    model, processor, model_audit = load_model(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        requested_targets=requested_targets,
        attn_implementation=args.attn_implementation,
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

    if rank == 0:
        args.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        prediction_path = (
            args.output_dir
            / "predictions.jsonl"
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
                    "eval_manifest": str(
                        args.eval_manifest
                    ),
                    "train_manifest": str(
                        args.train_manifest
                    ),
                    "training_contract": str(
                        args.training_contract
                    ),
                    "adapter_dir": str(
                        args.adapter_dir
                    ),
                    "model_id": args.model_id,
                    "eval_examples": len(
                        eval_rows
                    ),
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

    rows = []

    with torch.inference_mode():

        for index, row in enumerate(
            eval_rows,
            start=1,
        ):
            example_id = str(
                row["example_id"]
            )

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

            z_pass = logits[
                :,
                token_ids[Verdict.PASS],
            ]
            z_fail = logits[
                :,
                token_ids[Verdict.FAIL],
            ]

            margin = float(
                (
                    z_pass - z_fail
                )
                .detach()
                .float()
                .item()
            )

            gold = int(
                str(row["verdict"]).lower()
                == "pass"
            )
            pred = int(margin > 0.0)

            p_pass = (
                1.0
                / (
                    1.0
                    + math.exp(-margin)
                )
            )

            result = {
                "index": index,
                "example_id": example_id,
                "gold": gold,
                "pred": pred,
                "margin": margin,
                "p_pass": p_pass,
                "raw_bce": raw_bce(
                    margin,
                    gold,
                ),
            }

            rows.append(result)

            if rank == 0:
                with prediction_path.open(
                    "a",
                    encoding="utf-8",
                ) as f:
                    f.write(
                        json.dumps(result)
                        + "\n"
                    )

                if (
                    index == 1
                    or index % 10 == 0
                    or index == len(eval_rows)
                ):
                    print(
                        f"[{index:3d}/{len(eval_rows)}] "
                        f"gold={gold} "
                        f"pred={pred} "
                        f"margin={margin:+.4f}",
                        flush=True,
                    )

            del batch, outputs, logits

    if rank == 0:
        summary = summarize(rows)

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
        print("=" * 72)
        print("FORMALITY HELD-OUT RESULT")
        print("=" * 72)
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
