"""在同一40题上评测旧Judge或完整保存的epoch adapter。"""
import argparse
import json
import math
import time
from pathlib import Path

from .aria12 import CONDITION
from .collator import JudgeFrameCollator
from .contracts import JudgeTask, Verdict
from .data import load_normalized_manifest
from .loss import BinaryClassWeights, resolve_verdict_token_ids
from .metrics import _binary_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--eval-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    examples = load_normalized_manifest(args.eval_manifest)
    if len(examples) != 40 or sum(ex.target for ex in examples) != 10:
        raise ValueError("expected the approved 40-question evaluation set")
    if any(ex.condition_type != CONDITION for ex in examples):
        raise ValueError("evaluation must use the joint C requirement")
    if (args.output_dir / "summary.json").exists():
        raise FileExistsError("completed evaluation already exists")
    import torch
    from .check_linear_backend import validate_linear_backend
    from .evaluate_single_gpu import load_model, to_device
    from .trainer import final_logits, logits_to_keep_argument

    if torch.cuda.device_count() != 1:
        raise ValueError("this evaluator uses one allocated H200")
    validate_linear_backend()
    model, processor = load_model(args.model_id, "lora", args.adapter_dir)
    from .frozen_vision_cache import FrozenVisionCache
    vision_cache = FrozenVisionCache(model)
    tokens = resolve_verdict_token_ids(processor.tokenizer)
    keep_arg = logits_to_keep_argument(model)
    collator = JudgeFrameCollator(
        processor=processor,
        class_weights={JudgeTask.ANSWERABILITY: BinaryClassWeights(1, 1, 30, 10)},
        task_scales={JudgeTask.ANSWERABILITY: 1}, min_pixels=3136,
        max_pixels=119808, max_input_tokens=262144, decoded_image_cache_entries=3,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()
    with (args.output_dir / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        with torch.inference_mode():
            for index, example in enumerate(examples, 1):
                before = time.monotonic()
                vision_cache.key = (example.frames, 3136, 119808)
                batch = collator([example])
                for key in ("labels", "sample_weights", "tp_example_fingerprint"):
                    batch.pop(key)
                batch = to_device(batch, torch.device("cuda:0"))
                batch[keep_arg] = 1
                output = model(**batch, return_dict=True)
                logits = final_logits(output)
                margin = float((logits[0, tokens[Verdict.PASS]] - logits[0, tokens[Verdict.FAIL]]).float().cpu())
                if not math.isfinite(margin):
                    raise RuntimeError(f"nonfinite prediction for {example.example_id}")
                exponential = math.exp(-abs(margin))
                probability = 1 / (1 + exponential) if margin >= 0 else exponential / (1 + exponential)
                row = dict(example_id=example.example_id, gold=example.target, prediction=int(margin >= 0),
                           pass_probability=probability, margin=margin, seconds=time.monotonic()-before)
                rows.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(f"eval {args.label} {index}/40 p_pass={probability:.4f}", flush=True)
                del batch, output, logits
    metrics = _binary_metrics([r["gold"] for r in rows], [r["prediction"] for r in rows])
    metrics["pass_precision"] = sum(r["gold"] == r["prediction"] == 1 for r in rows) / max(1, sum(r["prediction"] for r in rows))
    metrics["confusion"] = {
        name: sum(r["gold"] == gold and r["prediction"] == predicted for r in rows)
        for name, gold, predicted in [("tp", 1, 1), ("tn", 0, 0), ("fp", 0, 1), ("fn", 1, 0)]
    }
    summary = dict(label=args.label, adapter=str(args.adapter_dir), examples=len(rows),
                   wall_seconds=time.monotonic()-started, metrics=metrics,
                   scope="same_scene_checkpoint_selection", threshold=0.5)
    summary["vision_cache"] = dict(hits=vision_cache.hits, misses=vision_cache.misses, bytes=vision_cache.bytes)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
