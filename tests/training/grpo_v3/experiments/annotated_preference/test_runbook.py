from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[5]
RUNBOOK = ROOT / "training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md"


class TorchRunbookTest(unittest.TestCase):
    def test_runbook_locks_pareto_dpo_gate_contract_and_safe_login_shell(self) -> None:
        text = RUNBOOK.read_text(encoding="utf-8")
        required = (
            "feature/annotated-pareto-dpo",
            "32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7",
            "60/10/0",
            "/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1",
            "gate0_data.sbatch",
            "structure_probe.sbatch",
            "smoke1.sbatch",
            "overfit_probe.sbatch",
            "train.sbatch",
            "evaluate.sbatch",
            "sbatch --parsable",
            "squeue -j",
            "sacct -j",
            "${JOBID}",
            "compact_qa_v1",
            "expanded schema",
            "Gate 6",
            "自由生成未验证",
        )
        for item in required:
            self.assertIn(item, text)
        for forbidden in ("latest_", "exit", "logout", "exec", "|| exit 1", "set -e", "set -euo pipefail"):
            self.assertNotIn(forbidden, text)
