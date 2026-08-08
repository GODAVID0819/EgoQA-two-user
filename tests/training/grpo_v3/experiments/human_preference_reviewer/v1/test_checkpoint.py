from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.checkpoint import (
    checkpoint_head_names,
    validate_checkpoint_runtime_contract,
)


class CheckpointContractTests(unittest.TestCase):
    def test_stage0_checkpoint_contains_only_evidence_head(self) -> None:
        self.assertEqual(checkpoint_head_names(("evidence_quality",)), ("evidence_head",))

    def test_runtime_rejects_checkpoint_stage_head_lora_or_model_drift(self) -> None:
        contract = {
            "stage": "stage0",
            "active_heads": ["evidence_quality"],
            "lora_enabled": False,
            "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
        }
        validate_checkpoint_runtime_contract(
            contract,
            stage="stage0",
            active_heads=("evidence_quality",),
            lora_enabled=False,
            model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
        )
        for key, bad_value in (
            ("stage", "stage2"),
            ("active_heads", ["evidence_quality", "answerability", "qa_formality"]),
            ("lora_enabled", True),
            ("model_name_or_path", "wrong-model"),
        ):
            altered = dict(contract)
            altered[key] = bad_value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "runtime contract mismatch"):
                validate_checkpoint_runtime_contract(
                    altered,
                    stage="stage0",
                    active_heads=("evidence_quality",),
                    lora_enabled=False,
                    model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
                )

    def test_runtime_rejects_lora_hyperparameter_and_target_drift(self) -> None:
        expected_config = {
            "last_n_shared_blocks": 2,
            "lora_target_modules": ["q_proj", "v_proj"],
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "lora_bias": "none",
        }
        contract = {
            "stage": "stage2",
            "active_heads": ["evidence_quality", "answerability", "qa_formality"],
            "lora_enabled": True,
            "model_name_or_path": "Qwen/Qwen3-VL-8B-Instruct",
            **expected_config,
            "actual_lora_targets": ["layer34.q_proj", "layer35.q_proj"],
        }
        validate_checkpoint_runtime_contract(
            contract,
            stage="stage2",
            active_heads=("evidence_quality", "answerability", "qa_formality"),
            lora_enabled=True,
            model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
            reviewer_config=expected_config,
            actual_lora_targets=("layer34.q_proj", "layer35.q_proj"),
        )
        altered = dict(contract, lora_r=4)
        with self.assertRaisesRegex(ValueError, "runtime contract mismatch"):
            validate_checkpoint_runtime_contract(
                altered,
                stage="stage2",
                active_heads=("evidence_quality", "answerability", "qa_formality"),
                lora_enabled=True,
                model_name_or_path="Qwen/Qwen3-VL-8B-Instruct",
                reviewer_config=expected_config,
                actual_lora_targets=("layer34.q_proj", "layer35.q_proj"),
            )


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is verified in the Torch training environment")
class CheckpointTests(unittest.TestCase):
    def test_stage0_round_trip_has_no_lora_file(self) -> None:
        from torch import nn
        from training.grpo_v3.experiments.human_preference_reviewer.v1.checkpoint import (
            load_classification_heads,
            save_checkpoint,
        )

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.evidence_head = nn.Linear(4, 3)

        directory = Path(tempfile.mkdtemp())
        model = Model()
        save_checkpoint(
            model,
            directory,
            config={"contract_version": "human_preference_reviewer_absolute_v1", "stage": "stage0"},
            csv_sha256="CSV",
            split_sha256="SPLIT",
            parameter_audit={"status": "passed"},
            active_heads=("evidence_quality",),
            lora_enabled=False,
        )
        self.assertFalse((directory / "lora_adapter.pt").exists())
        restored = Model()
        load_classification_heads(restored, directory)

    def test_heads_round_trip_exactly(self) -> None:
        import torch
        from torch import nn
        from training.grpo_v3.experiments.human_preference_reviewer.v1.checkpoint import (
            load_classification_heads,
            load_lora_adapter,
            save_checkpoint,
        )

        class Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.evidence_head = nn.Linear(4, 3)
                self.answerability_head = nn.Linear(4, 3)
                self.formality_head = nn.Linear(4, 3)
                self.q_proj = nn.Module()
                self.q_proj.lora_A = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})
                self.q_proj.lora_B = nn.ModuleDict({"default": nn.Linear(1, 1, bias=False)})

        model = Model()
        directory = Path(tempfile.mkdtemp())
        save_checkpoint(
            model, directory,
            config={"contract_version": "human_preference_reviewer_absolute_v1"},
            csv_sha256="CSV", split_sha256="SPLIT", parameter_audit={"status": "passed"},
        )
        restored = Model()
        load_classification_heads(restored, directory)
        load_lora_adapter(restored, directory)
        for name in ("evidence_head", "answerability_head", "formality_head"):
            for key, value in getattr(model, name).state_dict().items():
                torch.testing.assert_close(value, getattr(restored, name).state_dict()[key])

        adapter_path = directory / "lora_adapter.pt"
        adapter = torch.load(adapter_path, map_location="cpu", weights_only=True)
        adapter.pop(next(iter(adapter)))
        torch.save(adapter, adapter_path)
        with self.assertRaisesRegex(ValueError, "parameter set mismatch"):
            load_lora_adapter(Model(), directory)


if __name__ == "__main__":
    unittest.main()
