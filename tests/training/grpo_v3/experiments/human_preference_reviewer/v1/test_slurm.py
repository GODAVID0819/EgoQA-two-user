from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
HPC = ROOT / "hpc/grpo_v3/human_preference_reviewer/v1"
STAGE0_HPC = ROOT / "hpc/grpo_v3/human_preference_reviewer/stage0"


class ReviewerV1SlurmTests(unittest.TestCase):
    def test_stage0_jobs_are_isolated_and_explicitly_disable_lora(self) -> None:
        for name in ("smoke1", "overfit_probe"):
            text = (STAGE0_HPC / f"{name}.sbatch").read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertIn("--stage stage0", text)
                self.assertIn("human_preference_reviewer/stage0/common.sh", text)
                self.assertNotIn("--stage stage2", text)
        smoke = (STAGE0_HPC / "smoke1.sbatch").read_text(encoding="utf-8")
        overfit = (STAGE0_HPC / "overfit_probe.sbatch").read_text(encoding="utf-8")
        self.assertIn("--max-steps 1", smoke)
        self.assertIn("split_2_1_1.json", overfit)
        self.assertIn("--train-evidence-count 2", overfit)

    def test_all_jobs_are_single_h100_scratch_first_and_jobid_scoped(self) -> None:
        for name in ("structure_probe", "smoke1", "overfit_probe", "train", "evaluate"):
            text = (HPC / f"{name}.sbatch").read_text(encoding="utf-8")
            with self.subTest(name=name):
                self.assertIn("#SBATCH --account=torch_pr_674_tandon_advanced", text)
                self.assertIn("#SBATCH --gres=gpu:h100:1", text)
                self.assertIn("SLURM_JOB_ID", text)
                self.assertIn("source \"${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/common.sh\"", text)
                self.assertNotIn("latest", text.lower())
                self.assertNotIn("overall_utility", text)
                self.assertNotIn("bradley", text.lower())
                self.assertNotIn("grpo_reward", text)

    def test_common_preflight_closes_home_and_cache_writes_before_model(self) -> None:
        text = (HPC / "common.sh").read_text(encoding="utf-8")
        for variable in (
            "HOME", "XDG_CACHE_HOME", "HF_HOME", "HF_DATASETS_CACHE", "MODELSCOPE_CACHE",
            "TORCH_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "VLLM_CACHE_ROOT",
            "CUDA_CACHE_PATH", "FLASHINFER_WORKSPACE_BASE", "TMPDIR", "TMP", "TEMP",
        ):
            self.assertIn(f"export {variable}=", text)
        self.assertIn("training.torch_storage_preflight", text)
        self.assertIn("torchcodec.decoders import VideoDecoder", text)
        self.assertIn("LD_LIBRARY_PATH", text)
        self.assertIn("storage_preflight.json", text)


if __name__ == "__main__":
    unittest.main()
