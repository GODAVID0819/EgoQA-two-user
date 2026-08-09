from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[6]
RUNBOOK = ROOT / "training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_V1_CN.md"
STAGE0_RUNBOOK = ROOT / "training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_STAGE0_CN.md"
README = ROOT / "training/grpo_v3/experiments/human_preference_reviewer/README_CN.md"


class ReviewerV1RunbookTests(unittest.TestCase):
    def test_stage0_runbook_is_single_head_only_and_copyable(self) -> None:
        text = STAGE0_RUNBOOK.read_text(encoding="utf-8")
        for required in (
            "Evidence Quality", "--stage stage0", "Structure", "Smoke", "Overfit",
            "sacct", "storage_preflight.json", "training_result.json", "本阶段不能证明",
            "feature/multimodal-reviewer-training",
            '"${PYTHON}" -m unittest discover',
            "split_4_1_1.json", "pre_train_metrics", "post_train_metrics",
            "controlled_overfit_gate", "同一固定 probe set",
        ):
            self.assertIn(required, text)
        for forbidden in ("evaluate.sbatch", "train.sbatch", "--stage stage2", "Bradley"):
            self.assertNotIn(forbidden, text)
        self.assertNotIn("put -r", text)
        self.assertNotIn("exit ", text)
        self.assertNotIn("logout", text)
        self.assertNotIn("set -euo pipefail", text)
        self.assertIn('cd "${PROJECT_ROOT}"', text)
        self.assertIn("-s tests/training/grpo_v3/experiments/human_preference_reviewer/v1", text)
        self.assertNotIn('-t "${ROOT}"', text)
        self.assertNotIn('-t "${PROJECT_ROOT}"', text)
        self.assertGreaterEqual(text.count("--export=ALL"), 3)
        self.assertGreaterEqual(text.count('--chdir="${PROJECT_ROOT}"'), 3)
        self.assertGreaterEqual(text.count('--output="${PROJECT_ROOT}/logs/'), 3)

    def test_collaborator_readme_explains_complete_framework_and_stages(self) -> None:
        text = README.read_text(encoding="utf-8")
        for required in ("Stage 0", "Stage 1", "Stage 2", "Stage 3", "34", "35", "q_proj", "v_proj"):
            self.assertIn(required, text)
    def test_runbook_contains_copyable_gates_and_evidence_boundaries(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        for required in (
            "docs/TORCH_EXPERIMENT_META_RULES_CN.md", "sftp xl6775@", "lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user",
            "F3E006B3A488A3ACA86C8F3B1862392EF3576A73BA78EA202E40F7754DB730AC",
            "零 GPU", "Structure Probe", "Smoke", "Overfit Probe", "40/10/10", "Locked Test",
            "sacct", "scontrol show job -dd", "storage_preflight.json", "parameter_audit.json",
            "本次能证明", "本次不能证明",
        ):
            self.assertIn(required, text)
        self.assertNotIn("huggingface-cli", text)
        self.assertNotIn("latest_*", text)
        self.assertNotIn("set -euo pipefail\n", text)

    def test_v1_runbook_routes_everything_to_reviewer_repository(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        reviewer_root = "/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1"
        self.assertIn(f"PROJECT_ROOT={reviewer_root}", text)
        self.assertIn('cd "${PROJECT_ROOT}"', text)
        self.assertIn("-s tests/training/grpo_v3/experiments/human_preference_reviewer/v1", text)
        self.assertNotIn("/scratch/xl6775/projects/EgoQA-two-user-grpo-clean", text)
        self.assertNotIn('-t "${ROOT}"', text)
        self.assertNotIn('-t "${PROJECT_ROOT}"', text)
        self.assertNotIn("bye", text)
        self.assertNotIn("logout", text)
        self.assertIn("export PROJECT_ROOT OUTPUT_ROOT DATA_DIR", text)
        self.assertGreaterEqual(text.count("--export=ALL"), 6)
        self.assertGreaterEqual(text.count('--chdir="${PROJECT_ROOT}"'), 6)
        self.assertGreaterEqual(text.count('--output="${PROJECT_ROOT}/logs/'), 6)


if __name__ == "__main__":
    unittest.main()
