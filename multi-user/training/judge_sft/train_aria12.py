"""联合C专用训练入口。固定训练5轮，最终模型保存后评估留题测试集。"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import train as base
from .aria12 import CONDITION, validate_split
from .data import load_normalized_manifest
from .metrics import _binary_metrics


def load_splits(train_path, test_path):
    train = load_normalized_manifest(train_path)
    test = load_normalized_manifest(test_path)
    for examples, count, positives in [(train, 160, 40), (test, 40, 10)]:
        if len(examples) != count or sum(ex.target for ex in examples) != positives:
            raise ValueError("unexpected split size or class counts")
        if any(ex.condition_type != CONDITION or ex.frame_count != 1404 for ex in examples):
            raise ValueError("expected complete 12-window joint C examples")
    validate_split(
        [dict(example_id=ex.example_id, split_group_id=ex.split_group_id) for ex in train],
        [dict(example_id=ex.example_id, split_group_id=ex.split_group_id) for ex in test],
    )
    return train, test


def evaluate_final(trainer, examples, output_dir):
    """逐题沿用训练前向。两张TP卡执行相同样本，只有rank 0写结果。"""
    model = trainer.model_wrapped
    model.eval()
    rows = []
    for example in examples:
        batch = trainer.data_collator([example])
        _, logits, _ = trainer.prediction_step(model, batch, prediction_loss_only=False)
        margin = float((logits[0, 1].float() - logits[0, 0].float()).cpu())
        if not math.isfinite(margin):
            raise RuntimeError(f"nonfinite verdict margin: {example.example_id}")
        exp_value = math.exp(-abs(margin))
        probability = 1 / (1 + exp_value) if margin >= 0 else exp_value / (1 + exp_value)
        rows.append({
            "example_id": example.example_id, "split_group_id": example.split_group_id,
            "gold": example.target, "prediction": int(margin >= 0),
            "pass_probability": probability, "margin": margin,
        })
    if trainer.is_world_process_zero():
        metrics = _binary_metrics([r["gold"] for r in rows], [r["prediction"] for r in rows])
        metrics.update(examples=len(rows), epoch=5, global_step=200, threshold=0.5,
                       checkpoint="final_adapter", scope="same_scene_held_out_questions")
        (output_dir / "test_predictions.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        (output_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print("final_test_metrics=" + json.dumps(metrics), flush=True)


def main():
    original_parser = base.build_parser
    original_training_kwargs = base._training_argument_kwargs
    original_trainer = base.build_verdict_trainer_class
    original_loader = base.load_normalized_manifest

    def parser():
        p = original_parser()
        p.add_argument("--test-manifest", type=Path, required=True)
        p.set_defaults(task="answerability", epochs=5, gradient_accumulation_steps=4,
                       lora_rank=128, lora_alpha=128, learning_rate=2e-5, seed=17,
                       max_pixels=119808, no_class_weights=True)
        return p

    args = parser().parse_args()
    if (args.task, args.epochs, args.gradient_accumulation_steps, args.max_steps) != ("answerability", 5, 4, -1):
        raise ValueError("this entry point requires the approved 5-epoch joint C run")
    if args.train_example_id or args.probe_skip_save:
        raise ValueError("formal training requires all 160 examples and checkpoint saving")
    train, test = load_splits(args.train_manifest, args.test_manifest)

    def training_kwargs(*pos, **kw):
        result = original_training_kwargs(*pos, **kw)
        result["save_only_model"] = False
        result["per_device_eval_batch_size"] = 1
        return result

    def trainer_class():
        Parent = original_trainer()

        class Aria12Trainer(Parent):
            def save_model(self, output_dir=None, _internal_call=False):
                super().save_model(output_dir, _internal_call)
                if output_dir is None or Path(output_dir).name != "final_adapter" or _internal_call:
                    return
                if self.state.global_step != 200 or not math.isclose(self.state.epoch, 5.0):
                    raise RuntimeError("final test requires exactly five completed epochs")
                evaluate_final(self, test, Path(output_dir).parent)

        return Aria12Trainer

    base.build_parser = parser
    base._training_argument_kwargs = training_kwargs
    base.build_verdict_trainer_class = trainer_class
    base.load_normalized_manifest = lambda path: train if Path(path).resolve() == args.train_manifest.resolve() else original_loader(path)
    try:
        base.main()
    finally:
        base.build_parser = original_parser
        base._training_argument_kwargs = original_training_kwargs
        base.build_verdict_trainer_class = original_trainer
        base.load_normalized_manifest = original_loader


if __name__ == "__main__":
    main()
