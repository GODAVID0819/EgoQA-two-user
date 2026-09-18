from __future__ import annotations

import json
import unittest
from pathlib import Path

from training.judge_sft.contracts import DEFAULTS
from training.judge_sft.train import build_parser


ROOT = Path(__file__).resolve().parents[3]
PREP_SBATCH = ROOT / "hpc" / "judge_sft" / "prepare_real_manifests.sbatch"
RUNTIME_SBATCH = ROOT / "hpc" / "judge_sft" / "runtime_smoke_qwen38_27b.sbatch"
STEP1_SBATCH = ROOT / "hpc" / "judge_sft" / "train_one_step_qwen38_27b.sbatch"
TRAIN_SBATCH = ROOT / "hpc" / "judge_sft" / "train_real_40_packets_qwen38_27b.sbatch"
DEEPSPEED_CONFIG = ROOT / "training" / "judge_sft" / "deepspeed_zero3.json"
TRAIN_MODULE = ROOT / "training" / "judge_sft" / "train.py"


class JudgeSftClusterSmokeTests(unittest.TestCase):
    def test_default_model_is_native_multimodal_qwen38_27b(self) -> None:
        self.assertEqual(DEFAULTS.model_id, "Qwen/Qwen3.8-27B")
        parsed = build_parser().parse_args(
            ["--train-manifest", "train.jsonl", "--output-dir", "out"]
        )
        self.assertEqual(parsed.model_id, "Qwen/Qwen3.8-27B")
        self.assertEqual(parsed.max_steps, -1)
        self.assertFalse(hasattr(parsed, "eval_manifest"))
        self.assertEqual(parsed.lora_rank, 8)
        self.assertEqual(parsed.lora_alpha, 16)
        self.assertEqual(parsed.gradient_accumulation_steps, 16)
        self.assertEqual(
            parsed.lora_target_modules,
            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )

    def test_prepare_sbatch_uses_exact_real_artifacts_and_counts(self) -> None:
        text = PREP_SBATCH.read_text(encoding="utf-8")
        for expected in (
            "qwen38_legacy_two_pass_schema_v2",
            "egolife_rlhf_evidence_v1",
            "six_user_binary_labels_*.jsonl",
            'EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-221}"',
            'EXPECTED_PACKETS="${EXPECTED_PACKETS:-40}"',
            "training.torch_storage_preflight",
            "training.judge_sft.prepare_real_data",
            "--contradiction-policy exclude-task",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("latest_", text)
        self.assertNotIn("eval.jsonl", text)
        self.assertNotIn("--eval-fraction", text)

    def test_runtime_sbatch_probes_real_1800_frame_example(self) -> None:
        text = RUNTIME_SBATCH.read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gres=gpu:1",
            "#SBATCH --constraint=h200",
            'MIN_PIXELS="${MIN_PIXELS:-3136}"',
            'MAX_PIXELS="${MAX_PIXELS:-262144}"',
            "training.torch_storage_preflight",
            "training.judge_sft.select_smoke_example",
            "processor_and_1800_frame_probe",
            "training.judge_sft.runtime_probe",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("smoke_manifest", text)
        self.assertNotIn("VIDEO_LIST", text)
        self.assertNotIn("training.judge_sft.train ", text)
        self.assertNotIn("latest_", text)
        self.assertLess(
            text.index("training.torch_storage_preflight"),
            text.index("training.judge_sft.runtime_probe"),
        )

    def test_one_step_sbatch_has_training_and_reload_gates(self) -> None:
        text = STEP1_SBATCH.read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gres=gpu:2",
            "#SBATCH --constraint=h200",
            "--nproc_per_node=2",
            "--max-steps 1",
            "--gradient-accumulation-steps 1",
            "training.judge_sft.select_smoke_example",
            "deepspeed_zero3.json",
            "training.judge_sft.adapter_reload",
            "training.judge_sft.smoke_validate",
            "--expected-example-id",
            'ENABLE_CUDA_KEEPER="${ENABLE_CUDA_KEEPER:-1}"',
            "shared/cuda_high_duty.py",
            "start_cuda_keeper",
            "--max-prealloc",
            "CUDA_KEEPER_PID",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("VIDEO_LIST", text)
        self.assertNotIn("latest_", text)
        self.assertLess(
            text.index("training.torch_storage_preflight"),
            text.index("training.judge_sft.train"),
        )

    def test_full_train_is_ten_epoch_real_data_run(self) -> None:
        text = TRAIN_SBATCH.read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gres=gpu:2",
            "--nproc_per_node=2",
            'EPOCHS="${EPOCHS:-10}"',
            'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"',
            'LEARNING_RATE="${LEARNING_RATE:-2e-5}"',
            "--train-manifest",
            "checkpoint_inventory.json",
            "expected_epoch_checkpoints",
            "training.judge_sft.adapter_reload",
            'ENABLE_CUDA_KEEPER="${ENABLE_CUDA_KEEPER:-1}"',
            "shared/cuda_high_duty.py",
            "start_cuda_keeper",
            "--max-prealloc",
            "CUDA_KEEPER_PID",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("--eval-manifest", text)
        self.assertNotIn("eval.jsonl", text)
        self.assertNotIn("latest_", text)
        train_source = TRAIN_MODULE.read_text(encoding="utf-8")
        self.assertIn('"save_strategy": "steps" if args.max_steps > 0 else "epoch"', train_source)
        self.assertNotIn("save_total_limit", train_source)
        self.assertNotIn("EarlyStoppingCallback", train_source)

    def test_deepspeed_config_is_zero3_bf16_and_gathers_for_save(self) -> None:
        config = json.loads(DEEPSPEED_CONFIG.read_text(encoding="utf-8"))
        self.assertTrue(config["bf16"]["enabled"])
        self.assertEqual(config["zero_optimization"]["stage"], 3)
        self.assertTrue(
            config["zero_optimization"]["stage3_gather_16bit_weights_on_model_save"]
        )


if __name__ == "__main__":
    unittest.main()
