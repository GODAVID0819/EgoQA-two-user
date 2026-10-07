from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from peft import LoraConfig, PeftModel, get_peft_model

from training.judge_sft import train
from training.judge_sft.contracts import JudgeTask
from training.judge_sft.loss import BinaryClassWeights


class _LinearAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj_qkv = torch.nn.Linear(8, 12, bias=False)
        self.out_proj = torch.nn.Linear(4, 8, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_attn = _LinearAttention()
        self.gate_proj = torch.nn.Linear(8, 16, bias=False)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.layers = torch.nn.ModuleList([_Layer(), _Layer()])


def _lora_config() -> LoraConfig:
    return LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=["in_proj_qkv", "out_proj", "gate_proj"],
        layers_to_transform=[1],
        layers_pattern="layers",
    )


def _write_full_state_checkpoint(directory: Path, *, train_lora_b: bool) -> None:
    model = get_peft_model(_TinyModel(), _lora_config())
    if train_lora_b:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_()
    model.save_pretrained(str(directory))
    for name in ("optimizer.pt", "scheduler.pt", "trainer_state.json"):
        (directory / name).write_text("{}", encoding="utf-8")


def _validate(directory: Path, **overrides) -> dict:
    kwargs = {
        "lora_rank": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "lora_target_modules": ["gate_proj", "in_proj_qkv", "out_proj"],
        "trainable_layer_indices": [1],
    }
    kwargs.update(overrides)
    return train.validate_resume_checkpoint(directory, **kwargs)


class LoraTargetPolicyTests(unittest.TestCase):
    def test_split_separates_sharded_and_linear_attention_targets(self) -> None:
        sharded, replicated = train.split_lora_targets(
            ["q_proj", "in_proj_qkv", "down_proj", "out_proj", "in_proj_a"]
        )
        self.assertEqual(sharded, ["q_proj", "down_proj"])
        self.assertEqual(replicated, ["in_proj_qkv", "out_proj", "in_proj_a"])

    def test_split_rejects_unknown_targets(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "no TP-aware LoRA factor policy"):
            train.split_lora_targets(["q_proj", "conv1d"])

    def test_linear_attention_lora_requires_zero_dropout(self) -> None:
        argv = [
            "train.py",
            "--train-manifest", "unused.jsonl",
            "--output-dir", "unused",
            "--lora-target-modules", "q_proj", "in_proj_qkv",
            "--lora-dropout", "0.05",
        ]
        with patch.object(sys, "argv", argv):
            with self.assertRaisesRegex(ValueError, "--lora-dropout 0"):
                train.main()


class UnitClassWeightTests(unittest.TestCase):
    def test_unit_class_weights_keep_counts(self) -> None:
        weighted = {
            JudgeTask.GROUNDEDNESS: BinaryClassWeights(
                fail=1.544, passed=0.74, fail_count=218, pass_count=456
            )
        }
        unit = train.unit_class_weights(weighted)[JudgeTask.GROUNDEDNESS]
        self.assertEqual((unit.fail, unit.passed), (1.0, 1.0))
        self.assertEqual((unit.fail_count, unit.pass_count), (218, 456))


class ResumeCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.checkpoint = Path(self._tmp.name) / "checkpoint-10"
        self.checkpoint.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_validate_accepts_matching_full_state_checkpoint(self) -> None:
        _write_full_state_checkpoint(self.checkpoint, train_lora_b=True)
        audit = _validate(self.checkpoint)
        self.assertEqual(audit["adapter_config"]["r"], 2)

    def test_validate_rejects_model_only_checkpoint(self) -> None:
        _write_full_state_checkpoint(self.checkpoint, train_lora_b=True)
        (self.checkpoint / "optimizer.pt").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "optimizer.pt"):
            _validate(self.checkpoint)

    def test_validate_rejects_mismatched_lora_config(self) -> None:
        _write_full_state_checkpoint(self.checkpoint, train_lora_b=True)
        with self.assertRaisesRegex(ValueError, "lora_alpha"):
            _validate(self.checkpoint, lora_alpha=8)

    def test_resumed_adapter_matches_checkpoint(self) -> None:
        _write_full_state_checkpoint(self.checkpoint, train_lora_b=True)
        model = PeftModel.from_pretrained(
            _TinyModel(), str(self.checkpoint), is_trainable=True
        )
        report = train.verify_resumed_adapter_weights(model, self.checkpoint)
        self.assertEqual(report["tensors_verified"], 6)
        self.assertGreater(report["lora_B_norm"], 0.0)

    def test_fresh_adapter_is_not_accepted_as_resume(self) -> None:
        # Regression: the old resume entry silently trained a fresh adapter
        # (LoRA-B = 0) while restoring the checkpoint optimizer state.
        _write_full_state_checkpoint(self.checkpoint, train_lora_b=True)
        fresh = get_peft_model(_TinyModel(), _lora_config())
        with self.assertRaisesRegex(RuntimeError, "differ from the checkpoint"):
            train.verify_resumed_adapter_weights(fresh, self.checkpoint)

    def test_untrained_checkpoint_is_rejected(self) -> None:
        _write_full_state_checkpoint(self.checkpoint, train_lora_b=False)
        model = PeftModel.from_pretrained(
            _TinyModel(), str(self.checkpoint), is_trainable=True
        )
        with self.assertRaisesRegex(RuntimeError, "all-zero LoRA-B"):
            train.verify_resumed_adapter_weights(model, self.checkpoint)


if __name__ == "__main__":
    unittest.main()
