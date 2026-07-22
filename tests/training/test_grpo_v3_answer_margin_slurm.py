from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NAMES = ("scorer_probe", "calibration", "smoke1", "smoke5", "probe40", "fixed_eval")
SCRATCH_VARS = (
    "HOME", "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "MODELSCOPE_CACHE",
    "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT",
    "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE", "TMPDIR", "TMP", "TEMP",
)


class AnswerMarginSlurmTests(unittest.TestCase):
    def text(self, name: str) -> str:
        text = (ROOT / "hpc" / f"grpo_v3_answer_margin_{name}.sbatch").read_text(encoding="utf-8")
        if name in {"smoke5", "probe40"}:
            text += "\n" + (ROOT / "hpc" / "grpo_v3_answer_margin_smoke1.sbatch").read_text(encoding="utf-8")
        return text

    def test_all_jobs_are_scratch_first_and_preflight_before_model(self) -> None:
        for name in NAMES:
            text = self.text(name)
            with self.subTest(name=name):
                self.assertIn("JOB_SCRATCH_ROOT", text)
                for variable in SCRATCH_VARS:
                    self.assertIn(f"export {variable}=", text)
                self.assertIn("VLLM_NO_USAGE_STATS=1", text)
                self.assertNotIn("FLASHINFER_WORKSPACE_DIR", text)
                self.assertLess(text.index("training.torch_storage_preflight"), text.index("MODEL_LOAD_BOUNDARY"))

    def test_gpu_shape_and_scorer_isolation(self) -> None:
        probe = self.text("scorer_probe")
        self.assertIn("#SBATCH --gres=gpu:h100:1", probe)
        self.assertNotIn('"${SWIFT}" rlhf', probe)
        for name in NAMES[1:]:
            text = self.text(name)
            with self.subTest(name=name):
                self.assertIn("#SBATCH --gres=gpu:h100:2", text)
                self.assertIn("CUDA_VISIBLE_DEVICES=1", text)
                self.assertIn("--device cuda:0", text)
                self.assertIn("CUDA_VISIBLE_DEVICES=0", text)

    def test_training_jobs_lock_condition_and_hard_gates(self) -> None:
        for name, steps, previous in (
            ("smoke1", 1, "calibration"),
            ("smoke5", 5, "smoke1"),
            ("probe40", 40, "smoke5"),
        ):
            text = self.text(name)
            with self.subTest(name=name):
                for fragment in (
                    f"--max_steps {steps}", "--temperature 0.5", "--num_generations 4",
                    "--learning_rate 1e-5", "--lr_scheduler_type constant", "--beta 0.0",
                    "--top_p 1.0", "--target_modules q_proj v_proj", "--lora_rank 8",
                    "--lora_alpha 16", "--freeze_vit true", "--freeze_aligner true",
                    "egoqa_combined_video_answer_margin", "gate2_14119442", "checkpoint-1",
                    f"latest_answer_margin_{previous}_output.txt", '"passed"',
                ):
                    self.assertIn(fragment, text)

    def test_fixed_evidence_and_status_contracts(self) -> None:
        for name in NAMES:
            self.assertIn("EGOLIFE2U_DAY2_11350000_A1_A5", self.text(name))
        fixed = self.text("fixed_eval")
        for fragment in ("32", "64", "fixed_eval_results.jsonl", "fixed_eval_summary.json", "not_converged", "invalid"):
            self.assertIn(fragment, fixed)

    def test_scorer_jobs_use_explicit_ffmpeg_runtime_and_torchcodec_preflight(self) -> None:
        for name in NAMES:
            text = self.text(name)
            with self.subTest(name=name):
                self.assertIn("FFMPEG_ENV", text)
                self.assertIn('PATH="${FFMPEG_ENV}/bin:${PATH}"', text)
                self.assertIn('LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"', text)
                self.assertIn("from torchcodec.decoders import VideoDecoder", text)
                self.assertLess(text.index("from torchcodec.decoders import VideoDecoder"), text.rindex("MODEL_LOAD_BOUNDARY"))


if __name__ == "__main__":
    unittest.main()
