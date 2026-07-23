from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


class GrpoV3StructureTests(unittest.TestCase):
    def test_shared_package_replaces_flat_shared_modules(self) -> None:
        expected = (
            ROOT / "training/grpo_v3/shared/adapter_reload.py",
            ROOT / "training/grpo_v3/shared/contract.py",
            ROOT / "training/grpo_v3/shared/data.py",
            ROOT / "training/grpo_v3/shared/json_format.py",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
        for name in (
            "grpo_v3_adapter_reload.py",
            "grpo_v3_contract.py",
            "grpo_v3_data.py",
            "grpo_v3_json_format.py",
        ):
            with self.subTest(old=name):
                self.assertFalse((ROOT / "training" / name).exists())

    def test_baseline_and_runtime_are_not_flat(self) -> None:
        expected = (
            ROOT / "training/grpo_v3/runtime/reward_plugin.py",
            ROOT / "training/grpo_v3/baseline/preflight.py",
            ROOT / "training/grpo_v3/baseline/repo_reward.py",
            ROOT / "hpc/grpo_v3/baseline/gate0.sbatch",
            ROOT / "hpc/grpo_v3/baseline/gate4.sbatch",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_answer_margin_has_one_experiment_home(self) -> None:
        expected = (
            ROOT / "training/grpo_v3/experiments/answer_margin/domain.py",
            ROOT / "training/grpo_v3/experiments/answer_margin/reward.py",
            ROOT / "training/grpo_v3/experiments/answer_margin/scorer.py",
            ROOT / "hpc/grpo_v3/answer_margin/calibration.sbatch",
            ROOT / "hpc/grpo_v3/answer_margin/fixed_eval.sbatch",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())

    def test_answer_margin_is_the_active_temperature_point(self) -> None:
        roots = (
            ROOT / "training/grpo_v3/experiments/answer_margin",
            ROOT / "hpc/grpo_v3/answer_margin",
        )
        text = "\n".join(
            path.read_text(encoding="utf-8")
            for base in roots
            if base.exists()
            for path in base.rglob("*")
            if path.is_file() and path.suffix in {".py", ".sbatch"}
        )
        self.assertIn("0.5", text)
        self.assertNotIn("temperature=0.3", text)
        self.assertNotIn("temperature=0.7", text)

    def test_formality_exists_only_as_archived_experiment(self) -> None:
        archive = ROOT / "training/grpo_v3/experiments/archived/formality"
        self.assertTrue((archive / "reward.py").is_file())
        self.assertTrue((archive / "reward_plugin.py").is_file())
        self.assertFalse(any((ROOT / "training").glob("grpo_v3_formality_*.py")))
        self.assertFalse(any((ROOT / "hpc").glob("grpo_v3_formality_*.sbatch")))

    def test_evaluation_tools_have_one_package(self) -> None:
        expected = (
            ROOT / "training/grpo_v3/evaluation/greedy_compare.py",
            ROOT / "training/grpo_v3/evaluation/greedy_eval.py",
            ROOT / "training/grpo_v3/evaluation/groundedness_audit.py",
        )
        for path in expected:
            with self.subTest(path=path):
                self.assertTrue(path.is_file())
        self.assertFalse(any((ROOT / "training").glob("grpo_v3_greedy_*.py")))
        self.assertFalse((ROOT / "training/grpo_v3_groundedness_audit.py").exists())

    def test_active_code_does_not_depend_on_archived_experiments(self) -> None:
        roots = (
            ROOT / "training/grpo_v3/runtime",
            ROOT / "training/grpo_v3/baseline",
            ROOT / "training/grpo_v3/experiments/answer_margin",
            ROOT / "hpc/grpo_v3/answer_margin",
            ROOT / "hpc/grpo_v3/baseline",
        )
        offenders = []
        for base in roots:
            for path in base.rglob("*"):
                if path.is_file() and path.suffix in {".py", ".sbatch"}:
                    if "experiments.archived" in path.read_text(encoding="utf-8"):
                        offenders.append(path.relative_to(ROOT).as_posix())
        self.assertEqual(offenders, [])

if __name__ == "__main__":
    unittest.main()
