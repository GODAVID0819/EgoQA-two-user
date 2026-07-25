from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
HPC = ROOT / "hpc" / "grpo_v3" / "text_only_a_density"
SCRATCH_VARS = (
    "HOME", "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "MODELSCOPE_CACHE",
    "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT",
    "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE", "TMPDIR", "TMP", "TEMP",
)


class DensitySlurmTests(unittest.TestCase):
    def text(self) -> str:
        return (HPC / "run10.sbatch").read_text(encoding="utf-8")

    def test_single_job_is_scratch_first_and_has_no_forbidden_services(self) -> None:
        text = self.text()
        for variable in SCRATCH_VARS:
            self.assertIn(f"export {variable}=", text)
        self.assertIn("MODEL_LOAD_BOUNDARY=1", text)
        self.assertIn("training.torch_storage_preflight", text)
        self.assertLess(text.index("training.torch_storage_preflight"), text.index("MODEL_LOAD_BOUNDARY=1"))
        lowered = text.lower()
        for forbidden in ("reviewer", "judge", "use_vllm true", "video_paths", "image_paths", "endpoint"):
            self.assertNotIn(forbidden, lowered)

    def test_single_10_step_job_locks_minimal_parameters(self) -> None:
        text = self.text()
        for fragment in (
            "--max_steps 10",
            "--model ${POLICY_MODEL}",
            "--tuner_type lora",
            "--torch_dtype bfloat16",
            "--target_modules q_proj v_proj",
            "--lora_rank 8",
            "--lora_alpha 16",
            "--freeze_vit true",
            "--freeze_aligner true",
            "--num_generations 4",
            "--per_device_train_batch_size 4",
            "--gradient_accumulation_steps 1",
            "--gradient_checkpointing true",
            "--max_completion_length 64",
            "--learning_rate 1e-4",
            "--lr_scheduler_type constant",
            "--beta 0.0",
            "--temperature 0.7",
            "--top_p 1.0",
            "--seed 42",
            "--data_seed 42",
            "--dataset_shuffle false",
            "--use_vllm false",
            "egoqa_text_only_a_density",
            "quick_convergence_summary.json",
        ):
            self.assertIn(fragment, text)
        self.assertNotIn("--resume_from_checkpoint", text)

    def test_ms_swift_grpo_import_dependencies_are_preflighted_before_training(self) -> None:
        text = self.text()
        self.assertIn("egoqa-ms-swift-v4.2.2-vllm024", text)
        self.assertIn("import vllm", text)
        self.assertIn("GRPOTrainer", text)
        self.assertLess(text.index("import vllm"), text.index('"${SWIFT}" rlhf'))

    def test_quick10_uses_smaller_l40s_resource_by_default(self) -> None:
        text = self.text()
        self.assertIn("#SBATCH --cpus-per-task=4", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("#SBATCH --constraint=l40s", text)
        self.assertIn("#SBATCH --mem=32G", text)
        self.assertIn("#SBATCH --time=00:30:00", text)

    def test_fixed_rollout_probe_job_is_deterministic_and_small(self) -> None:
        text = (HPC / "fixed_rollout_probe.sbatch").read_text(encoding="utf-8")
        for variable in SCRATCH_VARS:
            self.assertIn(f"export {variable}=", text)
        self.assertIn("#SBATCH --job-name=egoqa-ad-fixed-probe", text)
        self.assertIn("#SBATCH --cpus-per-task=4", text)
        self.assertIn("#SBATCH --gres=gpu:1", text)
        self.assertIn("#SBATCH --constraint=l40s", text)
        self.assertIn("#SBATCH --mem=32G", text)
        self.assertIn("#SBATCH --time=00:30:00", text)
        self.assertIn("training.grpo_v3.experiments.text_only_a_density.fixed_rollout_probe", text)
        self.assertIn("--steps 10", text)
        self.assertIn("fixed_rollout_summary.json", text)
        lowered = text.lower()
        for forbidden in ("reviewer", "judge", "video_paths", "image_paths", "swift rlhf"):
            self.assertNotIn(forbidden, lowered)


if __name__ == "__main__":
    unittest.main()
