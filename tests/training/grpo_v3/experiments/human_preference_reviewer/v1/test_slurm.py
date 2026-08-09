from __future__ import annotations

import unittest
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
HPC = ROOT / "hpc/grpo_v3/human_preference_reviewer/v1"
STAGE0_HPC = ROOT / "hpc/grpo_v3/human_preference_reviewer/stage0"


class ReviewerV1SlurmTests(unittest.TestCase):
    def test_runtime_defaults_target_reviewer_repo_and_new_annotation_csv(self) -> None:
        common = (HPC / "common.sh").read_text(encoding="utf-8")
        self.assertIn("/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1", common)
        self.assertIn("rlhf_candidate_scores_merged_70_packets.csv", common)
        self.assertNotIn("EgoQA-two-user-grpo-clean", common)

        for directory in (HPC, STAGE0_HPC):
            for path in directory.glob("*.sbatch"):
                with self.subTest(path=path):
                    text = path.read_text(encoding="utf-8")
                    self.assertIn("/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1", text)
                    self.assertNotIn("EgoQA-two-user-grpo-clean", text)

    def test_formal_train_and_evaluation_use_sixty_ten_validation_contract(self) -> None:
        train = (HPC / "train.sbatch").read_text(encoding="utf-8")
        evaluate = (HPC / "evaluate.sbatch").read_text(encoding="utf-8")

        for text in (train, evaluate):
            self.assertIn("split_60_10.json", text)
            self.assertNotIn("split_40_10_10.json", text)
        self.assertIn("--train-evidence-count 60", train)
        self.assertIn("--validation-evidence-count 10", train)
        self.assertIn("--locked-test-evidence-count 0", train)
        self.assertIn('EVAL_SPLIT="validation"', evaluate)
        self.assertNotIn('EVAL_SPLIT="${EVAL_SPLIT:-validation}"', evaluate)

    def test_media_job_materializes_exact_current_csv_contract(self) -> None:
        path = HPC / "prepare_media.sbatch"
        self.assertTrue(path.is_file())
        text = path.read_text(encoding="utf-8")
        self.assertIn("rlhf_candidate_scores_merged_70_packets.csv", text)
        self.assertIn('assert len(required) == 140', text)
        self.assertIn('assert len(media_map) == 140', text)
        self.assertIn("snapshot_download", text)
        self.assertIn("audit media-map", text)
        self.assertNotIn("EgoQA-two-user-grpo-clean", text)

    def test_minimal_shared_runtime_dependencies_are_present(self) -> None:
        from training.torch_storage_preflight import validate_storage_environment

        self.assertTrue((ROOT / "hpc/shared/env_qwen3vl.sh").is_file())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            result = validate_storage_environment(
                allowed_root=root,
                environ={"HOME": str(root / "home")},
                required_variables=("HOME",),
            )
        self.assertEqual(result["status"], "passed")
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
        self.assertIn("split_4_1_1.json", overfit)
        self.assertIn("--train-evidence-count 4", overfit)
        self.assertIn("--max-steps 480", overfit)
        self.assertIn("--epochs 20", overfit)
        self.assertIn('controlled_overfit_gate', overfit)
        self.assertIn('gate["passed"]', overfit)

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
