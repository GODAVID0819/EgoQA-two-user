from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "hpc" / "grpo_v3_formality_smoke.sbatch"
PROBE = ROOT / "hpc" / "grpo_v3_formality_probe.sbatch"


class FormalitySlurmTests(unittest.TestCase):
    def test_common_contract_is_formality_only_native_video_lora(self) -> None:
        for path in (SMOKE, PROBE):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                for fragment in (
                    "#SBATCH --gres=gpu:2",
                    "#SBATCH --constraint=l40s",
                    "--reward_funcs egoqa_qa_formality_confidence",
                    "--num_generations 4",
                    "--temperature 0.7",
                    "--top_p 1.0",
                    "--learning_rate 1e-5",
                    "--lr_scheduler_type constant",
                    "--beta 0.0",
                    "--freeze_vit true",
                    "--freeze_aligner true",
                    "--use_vllm false",
                    "POLICY_INPUT=\"native_video\"",
                    "qa_formality_confidence_v1",
                    "gate2_result.json",
                    "run_manifest.json",
                    "training.grpo_v3_formality_artifacts validate",
                    "training.grpo_v3_formality_artifacts summarize",
                ):
                    self.assertIn(fragment, text)
                for forbidden in (
                    "ground_answer_gap_v1",
                    "--reward_funcs egoqa_repo_native_judge",
                    "EGOQA_GROUNDEDNESS_AUDIT_SUMMARY",
                    "--allowed-local-media-path",
                ):
                    self.assertNotIn(forbidden, text)

    def test_smoke_is_one_step_and_probe_is_forty_steps(self) -> None:
        smoke = SMOKE.read_text(encoding="utf-8")
        probe = PROBE.read_text(encoding="utf-8")
        self.assertIn("--max_steps 1", smoke)
        self.assertNotIn("--max_steps 40", smoke)
        self.assertIn("--max_steps 40", probe)
        self.assertNotIn("latest_formality_smoke_output.txt", probe)
        self.assertIn("training.grpo_v3_formality_convergence", probe)
        self.assertIn("--expected-steps 40", probe)

    def test_failure_results_are_summarized_before_exit(self) -> None:
        for path in (SMOKE, PROBE):
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                validate_at = text.index("training.grpo_v3_formality_artifacts validate")
                summarize_at = text.index("training.grpo_v3_formality_artifacts summarize")
                exit_at = text.rindex("exit")
                self.assertLess(validate_at, summarize_at)
                self.assertLess(summarize_at, exit_at)


if __name__ == "__main__":
    unittest.main()
