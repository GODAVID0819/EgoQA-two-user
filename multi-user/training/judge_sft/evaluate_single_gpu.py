#!/usr/bin/env python3
"""Single-GPU first-token PASS/FAIL evaluation for base or LoRA judge."""

from __future__ import annotations
import argparse
import inspect
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import transformers
from peft import PeftModel
from transformers import AutoProcessor

from .collator import JudgeFrameCollator
from .contracts import JudgeTask, Verdict, validate_task_weights
from .data import load_normalized_manifest
from .loss import class_weights_by_task, resolve_verdict_token_ids, task_sampling_scales
from .trainer import final_logits, logits_to_keep_argument


GROUPS = (
    "formality",
    "answerability_asker",
    "groundedness",
    "answerability_all",
)


def task_value(ex: Any) -> str:
    v = getattr(ex, "task")
    return (v.value if hasattr(v, "value") else str(v)).lower()


def verdict_value(ex: Any) -> str:
    v = getattr(ex, "verdict")
    return (v.value if hasattr(v, "value") else str(v)).lower()


def group(ex: Any) -> str:
    task = task_value(ex)
    frames = int(getattr(ex, "frame_count", 0))
    if "formality" in task:
        return "formality"
    if "ground" in task or "evidence" in task:
        return "groundedness"
    if "answer" in task:
        return "answerability_asker" if frames <= 300 else "answerability_all"
    raise RuntimeError(f"Unknown task {task!r}")


def to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, list):
        return [to_device(v, device) for v in x]
    if isinstance(x, tuple):
        return tuple(to_device(v, device) for v in x)
    return x


def raw_bce(z: float, y: int) -> float:
    return max(z, 0.0) - z * float(y) + math.log1p(math.exp(-abs(z)))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def calc(xs: list[dict[str, Any]]) -> dict[str, Any]:
        if not xs:
            return {"n": 0}
        tp = sum(r["gold"] == 1 and r["pred"] == 1 for r in xs)
        tn = sum(r["gold"] == 0 and r["pred"] == 0 for r in xs)
        fp = sum(r["gold"] == 0 and r["pred"] == 1 for r in xs)
        fn = sum(r["gold"] == 1 and r["pred"] == 0 for r in xs)
        n = len(xs)
        pos = tp + fn
        neg = tn + fp
        tpr = tp / pos if pos else None
        tnr = tn / neg if neg else None
        bal = (tpr + tnr) / 2 if tpr is not None and tnr is not None else None
        return {
            "n": n,
            "accuracy": (tp + tn) / n,
            "balanced_accuracy": bal,
            "raw_bce": sum(r["raw_bce"] for r in xs) / n,
            "gold_pass_rate": sum(r["gold"] for r in xs) / n,
            "pred_pass_rate": sum(r["pred"] for r in xs) / n,
            "mean_margin": sum(r["margin"] for r in xs) / n,
            "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        }

    return {
        "overall": calc(rows),
        "by_group": {
            g: calc([r for r in rows if r["group"] == g])
            for g in GROUPS
        },
    }


def load_model(model_id: str, mode: str, adapter_dir: Path | None) -> tuple[Any, Any]:
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=True,
    )
    model_cls = getattr(transformers, "AutoModelForMultimodalLM", None)
    if model_cls is None:
        model_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if model_cls is None:
        raise RuntimeError("No supported multimodal auto-model class")

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "device_map": {"": 0},
    }
    params = inspect.signature(model_cls.from_pretrained).parameters
    if "dtype" in params:
        kwargs["dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    base = model_cls.from_pretrained(model_id, **kwargs)
    base.config.use_cache = False

    if mode == "lora":
        if adapter_dir is None:
            raise RuntimeError("--adapter-dir is required for mode=lora")
        model = PeftModel.from_pretrained(
            base,
            str(adapter_dir),
            is_trainable=False,
            autocast_adapter_dtype=False,
        )
    else:
        model = base

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, processor


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("base", "lora"), required=True)
    p.add_argument("--model-id", required=True)
    p.add_argument("--adapter-dir", type=Path)
    p.add_argument("--eval-manifest", type=Path, required=True)
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--training-contract", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"This evaluator expects exactly one visible GPU; got {torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")

    contract = json.loads(args.training_contract.read_text(encoding="utf-8"))
    hp = contract["hyperparameters"]

    train_examples = load_normalized_manifest(args.train_manifest)
    eval_examples = load_normalized_manifest(args.eval_manifest)

    contract_task_weights = contract.get("task_weights") or {}

    # Legacy evaluator expects all three task weights, but single-task
    # Groundedness training contracts may contain only groundedness.
    # Evaluation here is Groundedness-only; these fallback weights are
    # used only to satisfy the old multi-task weighting machinery.
    if set(contract_task_weights) >= {"formality", "groundedness", "answerability"}:
        task_weights = validate_task_weights(
            {
                JudgeTask.FORMALITY: float(contract_task_weights["formality"]),
                JudgeTask.GROUNDEDNESS: float(contract_task_weights["groundedness"]),
                JudgeTask.ANSWERABILITY: float(contract_task_weights["answerability"]),
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
        smoothing=float(hp["class_weight_smoothing"]),
        max_weight=float(hp["max_class_weight"]),
    )
    task_scales = task_sampling_scales(train_examples, task_weights)

    model, processor = load_model(
        args.model_id,
        args.mode,
        args.adapter_dir,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    token_ids = resolve_verdict_token_ids(tokenizer)
    keep_arg = logits_to_keep_argument(model)

    collator = JudgeFrameCollator(
        processor=processor,
        class_weights=class_weights,
        task_scales=task_scales,
        min_pixels=int(hp["min_pixels"]),
        max_pixels=int(hp["max_pixels"]),
        max_input_tokens=int(hp["max_input_tokens"]),
        image_context_target_fraction=float(hp["image_context_target_fraction"]),
        image_text_token_reserve=int(hp["image_text_token_reserve"]),
        image_item_token_overhead=int(hp["image_item_token_overhead"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / "predictions.jsonl"
    pred_path.write_text("", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    started = time.time()

    with torch.inference_mode():
        for i, ex in enumerate(eval_examples, 1):
            t0 = time.time()
            batch = collator([ex])
            labels = batch.pop("labels")
            batch.pop("sample_weights", None)
            batch.pop("tp_example_fingerprint", None)
            batch = to_device(batch, device)
            labels = labels.to(device)
            batch[keep_arg] = 1

            outputs = model(**batch, return_dict=True)
            logits = final_logits(outputs)
            z_pass = float(logits[:, token_ids[Verdict.PASS]].float().item())
            z_fail = float(logits[:, token_ids[Verdict.FAIL]].float().item())
            margin = z_pass - z_fail
            gold = int(verdict_value(ex) == "pass")
            pred = int(margin > 0.0)

            row = {
                "index": i,
                "example_id": getattr(ex, "example_id", None),
                "mode": args.mode,
                "task": task_value(ex),
                "group": group(ex),
                "frame_count": int(getattr(ex, "frame_count", 0)),
                "gold": gold,
                "pred": pred,
                "z_pass": z_pass,
                "z_fail": z_fail,
                "margin": margin,
                "p_pass": 1.0 / (1.0 + math.exp(-margin)),
                "raw_bce": raw_bce(margin, gold),
                "seconds": time.time() - t0,
            }
            rows.append(row)
            with pred_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(
                f"[{args.mode}] {i}/{len(eval_examples)} "
                f"group={row['group']} gold={gold} pred={pred} "
                f"margin={margin:.4f} sec={row['seconds']:.1f}",
                flush=True,
            )

            del batch, labels, outputs, logits
            if i % 4 == 0:
                torch.cuda.empty_cache()

    summary = {
        "mode": args.mode,
        "rows": len(rows),
        "wall_seconds": time.time() - started,
        "metrics": summarize(rows),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
