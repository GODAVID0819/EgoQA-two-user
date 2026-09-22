from __future__ import annotations

import unittest
from pathlib import Path

from training.judge_sft.contracts import DEFAULTS
from training.judge_sft.train import _training_argument_kwargs, build_parser


ROOT = Path(__file__).resolve().parents[3]
PREP_SBATCH = ROOT / "hpc" / "judge_sft" / "prepare_real_manifests.sbatch"
RUNTIME_SBATCH = ROOT / "hpc" / "judge_sft" / "runtime_smoke_qwen38_27b.sbatch"
STEP1_SBATCH = ROOT / "hpc" / "judge_sft" / "train_one_step_qwen38_27b.sbatch"
TRAIN_SBATCH = ROOT / "hpc" / "judge_sft" / "train_real_40_packets_qwen38_27b.sbatch"
HALF5_SBATCH = (
    ROOT / "hpc" / "judge_sft" / "train_half5_ga8_l2_f050_qwen38_27b (1).sbatch"
)
TRAIN_MODULE = ROOT / "training" / "judge_sft" / "train.py"
TRAINER_MODULE = ROOT / "training" / "judge_sft" / "trainer.py"


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
        self.assertEqual(parsed.trainable_decoder_layers, 16)
        self.assertEqual(parsed.gradient_accumulation_steps, 32)
        self.assertEqual(parsed.tensor_parallel_size, 2)
        self.assertFalse(hasattr(parsed, "deepspeed"))
        self.assertFalse(hasattr(parsed, "resume_from_checkpoint"))
        self.assertEqual(parsed.epochs, 10.0)
        self.assertEqual(parsed.image_context_target_fraction, 0.85)
        self.assertEqual(parsed.attn_implementation, "sdpa")
        self.assertEqual(parsed.dataloader_num_workers, 2)
        self.assertEqual(parsed.dataloader_prefetch_factor, 1)
        self.assertEqual(parsed.decoded_image_cache_entries, 2)
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

    def test_warmup_argument_supports_transformers_v4_and_v5(self) -> None:
        parsed = build_parser().parse_args(
            ["--train-manifest", "train.jsonl", "--output-dir", "out"]
        )
        v4 = _training_argument_kwargs(
            parsed,
            {
                "warmup_ratio",
                "evaluation_strategy",
                "parallelism_config",
                "save_only_model",
            },
            parallelism_config="tp2",
        )
        self.assertEqual(v4["warmup_ratio"], 0.1)
        self.assertNotIn("warmup_steps", v4)
        self.assertEqual(v4["evaluation_strategy"], "no")
        self.assertEqual(v4["parallelism_config"], "tp2")
        self.assertTrue(v4["save_only_model"])
        self.assertEqual(v4["dataloader_num_workers"], 2)
        self.assertTrue(v4["dataloader_persistent_workers"])
        self.assertEqual(v4["dataloader_prefetch_factor"], 1)
        self.assertEqual(v4["dataloader_multiprocessing_context"], "fork")

        v5 = _training_argument_kwargs(
            parsed,
            {
                "warmup_steps",
                "eval_strategy",
                "parallelism_config",
                "save_only_model",
            },
            parallelism_config="tp2",
        )
        self.assertEqual(v5["warmup_steps"], 0.1)
        self.assertNotIn("warmup_ratio", v5)
        self.assertEqual(v5["eval_strategy"], "no")

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
            'IMAGE_CONTEXT_TARGET_FRACTION="${IMAGE_CONTEXT_TARGET_FRACTION:-0.85}"',
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
            'MAX_PIXELS="${MAX_PIXELS:-262144}"',
            "--nproc_per_node=2",
            "--max-steps 1",
            '--image-context-target-fraction "${IMAGE_CONTEXT_TARGET_FRACTION}"',
            "--gradient-accumulation-steps 1",
            "training.judge_sft.select_smoke_example",
            '--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"',
            'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"',
            'TRAINABLE_DECODER_LAYERS="${TRAINABLE_DECODER_LAYERS:-16}"',
            '--trainable-decoder-layers "${TRAINABLE_DECODER_LAYERS}"',
            'Version(transformers.__version__) >= Version("5.4.0")',
            'Version(accelerate.__version__) >= Version("1.12.0")',
            'Version(peft.__version__) >= Version("0.19.0")',
            "autocast_adapter_dtype",
            "nvidia-smi topo -m",
            "training.judge_sft.adapter_reload",
            "training.judge_sft.smoke_validate",
            "--expected-example-id",
            'ENABLE_CUDA_KEEPER="${ENABLE_CUDA_KEEPER:-1}"',
            "shared/cuda_high_duty.py",
            "start_cuda_keeper",
            "--max-prealloc",
            "CUDA_KEEPER_PID",
            'ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"',
            'if attn_implementation == "flash_attention_2"',
            "chat_template_preflight",
            "_assert_thinking_disabled",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("VIDEO_LIST", text)
        self.assertNotIn("--deepspeed", text)
        self.assertNotIn("latest_", text)
        self.assertLess(
            text.index("training.torch_storage_preflight"),
            text.index("training.judge_sft.train"),
        )

    def test_half5_launcher_exposes_cpu_preprocessing_controls(self) -> None:
        text = HALF5_SBATCH.read_text(encoding="utf-8")
        for expected in (
            'DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"',
            'DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"',
            'DECODED_IMAGE_CACHE_ENTRIES="${DECODED_IMAGE_CACHE_ENTRIES:-2}"',
            '--dataloader-num-workers "${DATALOADER_NUM_WORKERS}"',
            '--dataloader-prefetch-factor "${DATALOADER_PREFETCH_FACTOR}"',
            '--decoded-image-cache-entries "${DECODED_IMAGE_CACHE_ENTRIES}"',
        ):
            self.assertIn(expected, text)

    def test_full_train_is_ten_epoch_real_data_run(self) -> None:
        text = TRAIN_SBATCH.read_text(encoding="utf-8")
        for expected in (
            "#SBATCH --gres=gpu:2",
            'MAX_PIXELS="${MAX_PIXELS:-262144}"',
            "--nproc_per_node=2",
            'EPOCHS="${EPOCHS:-10}"',
            'IMAGE_CONTEXT_TARGET_FRACTION="${IMAGE_CONTEXT_TARGET_FRACTION:-0.85}"',
            'GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-32}"',
            'LEARNING_RATE="${LEARNING_RATE:-2e-5}"',
            '--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"',
            'TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"',
            'TRAINABLE_DECODER_LAYERS="${TRAINABLE_DECODER_LAYERS:-16}"',
            '--trainable-decoder-layers "${TRAINABLE_DECODER_LAYERS}"',
            'Version(transformers.__version__) >= Version("5.4.0")',
            'Version(accelerate.__version__) >= Version("1.12.0")',
            'Version(peft.__version__) >= Version("0.19.0")',
            "nvidia-smi topo -m",
            "--train-manifest",
            "checkpoint_inventory.json",
            "expected_epoch_checkpoints",
            "training.judge_sft.adapter_reload",
            'ENABLE_CUDA_KEEPER="${ENABLE_CUDA_KEEPER:-1}"',
            "shared/cuda_high_duty.py",
            "start_cuda_keeper",
            "--max-prealloc",
            "CUDA_KEEPER_PID",
            'ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"',
            'if attn_implementation == "flash_attention_2"',
            "chat_template_preflight",
            "_assert_thinking_disabled",
            "processor_and_1800_frame_probe",
            "training.judge_sft.runtime_probe",
        ):
            self.assertIn(expected, text)
        self.assertNotIn("--eval-manifest", text)
        self.assertNotIn("--deepspeed", text)
        self.assertNotIn("RESUME_FROM_CHECKPOINT", text)
        self.assertNotIn("eval.jsonl", text)
        self.assertNotIn("latest_", text)
        self.assertLess(
            text.index("training.judge_sft.runtime_probe"),
            text.index("training.judge_sft.train"),
        )
        train_source = TRAIN_MODULE.read_text(encoding="utf-8")
        self.assertIn('"save_strategy": "steps" if args.max_steps > 0 else "epoch"', train_source)
        self.assertIn("judge_visual_budget_preflight=", train_source)
        self.assertIn("DistributedConfig(tp_size=args.tensor_parallel_size)", train_source)
        self.assertIn("autocast_adapter_dtype=False", train_source)
        self.assertIn("layers_to_transform=trainable_layer_indices", train_source)
        self.assertIn("audit_lora_tensor_parallel_materialization", train_source)
        self.assertIn("ensure_tensor_parallel_metadata", train_source)
        self.assertIn("synchronize_lora_initialization", train_source)
        self.assertIn("materialize_lora_tensor_parallelism", train_source)
        self.assertIn("module._tp_info = TpInfo", train_source)
        self.assertLess(
            train_source.index('tp_plan_audit["materialization"]'),
            train_source.index('tp_plan_audit["metadata"]'),
        )
        self.assertLess(
            train_source.index("model = get_peft_model"),
            train_source.index('tp_plan_audit["lora_initialization"]'),
        )
        self.assertLess(
            train_source.index('tp_plan_audit["lora_initialization"]'),
            train_source.index('tp_plan_audit["lora_materialization"]'),
        )
        seed_position = train_source.index("set_seed(args.seed)")
        peft_position = train_source.index("model = get_peft_model")
        self.assertLess(seed_position, peft_position)
        self.assertIn("torch.distributed.broadcast(", train_source)
        self.assertIn("torch.distributed.all_gather(", train_source)
        self.assertIn("exact_post_broadcast_equality", train_source)
        self.assertIn('layers_pattern="layers"', train_source)
        self.assertIn("disable_input_require_grads", train_source)
        self.assertNotIn("enable_input_require_grads", train_source)
        self.assertIn('"save_only_model"] = True', train_source)
        self.assertIn('"foreach": False', train_source)
        self.assertIn('"fused": False', train_source)
        self.assertNotIn("save_total_limit", train_source)
        self.assertNotIn("EarlyStoppingCallback", train_source)

        trainer_source = TRAINER_MODULE.read_text(encoding="utf-8")
        self.assertIn("def save_model(", trainer_source)
        self.assertIn("is_main_process=is_writer", trainer_source)
        self.assertIn("state_dict=adapter_state_dict", trainer_source)
        self.assertIn("torch.distributed.barrier()", trainer_source)
        self.assertIn(
            "all ranks participate in DTensor.full_tensor() gathers",
            trainer_source,
        )

    def test_training_source_uses_pure_tp_and_independent_images(self) -> None:
        train_source = TRAIN_MODULE.read_text(encoding="utf-8")
        self.assertIn("pure tensor parallelism", train_source)
        self.assertIn("independent Qwen image item", train_source)
        self.assertNotIn('kwargs["deepspeed"]', train_source)


if __name__ == "__main__":
    unittest.main()
