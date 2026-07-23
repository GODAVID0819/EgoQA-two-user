from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
SBATCH = ROOT / "hpc/grpo_v3/archived/formality/fixed_eval.sbatch"


class FormalityFixedEvalSlurmTests(unittest.TestCase):
    def test_job_is_scratch_first_formality_only_endpoint_evaluation(self) -> None:
        text = SBATCH.read_text(encoding="utf-8")
        required = (
            "#SBATCH --gres=gpu:2",
            "#SBATCH --constraint=l40s",
            "FORMALITY_PROBE_DIR",
            "formality_probe_14377903",
            "JOB_SCRATCH_ROOT",
            "HOME",
            "XDG_CACHE_HOME",
            "HF_HOME",
            "HF_DATASETS_CACHE",
            "MODELSCOPE_CACHE",
            "TORCH_HOME",
            "TRITON_CACHE_DIR",
            "TORCHINDUCTOR_CACHE_DIR",
            "VLLM_CACHE_ROOT",
            "CUDA_CACHE_PATH",
            "FLASHINFER_WORKSPACE_BASE",
            "TMPDIR",
            "TMP",
            "TEMP",
            "VLLM_NO_USAGE_STATS=1",
            "training.torch_storage_preflight",
            "training.grpo_v3.experiments.archived.formality.fixed_eval",
            "fixed_eval_results.jsonl",
            "fixed_eval_summary.json",
            "run_manifest.json",
            "latest_formality_fixed_eval_output.txt",
            "qa_formality_confidence_v1",
            "--temperature 0.7",
        )
        for fragment in required:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, text)
        for forbidden in (
            '"${SWIFT}" rlhf',
            "--max_steps",
            "egoqa_repo_native_judge",
            "groundedness",
            "answerability",
            "FLASHINFER_WORKSPACE_DIR",
            "experiment_conclusion\"] == \"improved\"",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)

    def test_storage_and_reviewer_preflight_precede_policy_evaluation(self) -> None:
        text = SBATCH.read_text(encoding="utf-8")
        storage_at = text.index("training.torch_storage_preflight")
        reviewer_at = text.index('"${VLLM}" serve')
        models_at = text.index('${REVIEW_BASE_URL}/models')
        chat_at = text.index('${REVIEW_BASE_URL}/chat/completions')
        eval_at = text.index("training.grpo_v3.experiments.archived.formality.fixed_eval")
        self.assertLess(storage_at, reviewer_at)
        self.assertLess(reviewer_at, models_at)
        self.assertLess(models_at, chat_at)
        self.assertLess(chat_at, eval_at)

    def test_completion_asserts_exact_32_row_two_by_sixteen_contract(self) -> None:
        text = SBATCH.read_text(encoding="utf-8")
        for fragment in (
            "len(rows) == 32",
            'counts == {0: 16, 40: 16}',
            'manifest["run_status"] == "passed"',
            'summary["experiment_conclusion"] in',
            '"improved", "not_improved", "inconclusive"',
        ):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
