from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
DOC_ROOT = ROOT / "training/grpo_v3/experiments/human_preference_reviewer"
RUNBOOK = DOC_ROOT / "TORCH_RUNBOOK_V1_CN.md"
STAGE0_RUNBOOK = DOC_ROOT / "TORCH_RUNBOOK_STAGE0_CN.md"
README = DOC_ROOT / "README_CN.md"
DESIGN = DOC_ROOT / "REVIEWER_STAGED_DESIGN_CN.md"


class ReviewerV1RunbookTests(unittest.TestCase):
    def test_stage0_runbook_is_single_head_only_and_uses_current_data(self) -> None:
        text = STAGE0_RUNBOOK.read_text(encoding="utf-8")
        for required in (
            "Evidence Quality",
            "--stage stage0",
            "Structure",
            "Smoke",
            "Overfit",
            "controlled_overfit_gate",
            "同一固定 probe set",
            "rlhf_candidate_scores_merged_70_packets.csv",
            "32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7",
            "/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1",
        ):
            self.assertIn(required, text)
        for forbidden in (
            "evaluate.sbatch",
            "train.sbatch",
            "--stage stage2",
            "rlhf_candidate_scores_day5_7_full_100_HM.csv",
        ):
            self.assertNotIn(forbidden, text)

    def test_collaborator_docs_explain_current_contract_and_all_stages(self) -> None:
        readme = README.read_text(encoding="utf-8")
        design = DESIGN.read_text(encoding="utf-8")
        for required in (
            "Stage 0", "Stage 1", "Stage 2", "Stage 3", "34", "35", "q_proj", "v_proj",
            "420", "70", "60", "10", "locked test",
        ):
            self.assertIn(required, readme + design)
        self.assertIn("60/10/0", readme)
        self.assertIn("rlhf_candidate_scores_merged_70_packets.csv", readme)

    def test_v1_runbook_contains_current_data_split_media_and_validation_gates(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        for required in (
            "docs/TORCH_EXPERIMENT_META_RULES_CN.md",
            "sftp xl6775@",
            "rlhf_candidate_scores_merged_70_packets.csv",
            "32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7",
            "420", "70", "60/10/0", "split_60_10.json", "140",
            "零 GPU", "Structure Probe", "Smoke", "Overfit Probe", "Validation",
            "sacct", "storage_preflight.json", "parameter_audit.json",
            "本轮能证明", "本轮不能证明",
        ):
            self.assertIn(required, text)
        for forbidden in (
            "rlhf_candidate_scores_day5_7_full_100_HM.csv",
            "split_40_10_10.json",
            "40/10/10",
            "EVAL_SPLIT=locked_test",
            "EgoQA-two-user-grpo-clean",
        ):
            self.assertNotIn(forbidden, text)

    def test_copyable_shell_blocks_preserve_the_ssh_session(self) -> None:
        for path in (RUNBOOK, STAGE0_RUNBOOK):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path):
                self.assertNotIn("logout", text)
                self.assertNotIn("|| exit", text)
                self.assertNotIn("exit 1", text)
                self.assertNotIn("set -euo pipefail\n", text)
                self.assertNotIn("bye\n", text)
                self.assertIn('cd "${PROJECT_ROOT}"', text)


if __name__ == "__main__":
    unittest.main()
