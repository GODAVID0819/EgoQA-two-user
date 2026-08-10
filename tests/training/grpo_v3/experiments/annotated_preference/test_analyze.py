from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.annotated_preference.analyze import analyze_paths, main


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_inputs(root: Path) -> tuple[Path, Path, Path]:
    trainer_state = _write_json(root / "trainer_state.json", {
        "global_step": 3,
        "log_history": [
            {"loss": 1.2, "grad_norm": 0.7, "rewards/margins": 0.1,
             "rewards/accuracies": 0.6, "rewards/logps/chosen": -2.0,
             "rewards/logps/rejected": -2.1},
            {"loss": 0.8, "grad_norm": 0.5, "rewards/margins": 0.4,
             "rewards/accuracies": 0.9, "rewards/logps/chosen": -1.1,
             "rewards/logps/rejected": -1.5, "eval_loss": 0.7,
             "eval_rewards/margins": 0.3, "eval_rewards/accuracies": 0.85,
             "eval_pair_count": 6},
        ],
    })
    audit = _write_json(root / "parameter_audit.json", {
        "lora_delta_nonzero": True,
        "non_lora_delta_zero": True,
        "checkpoint_exists": True,
    })
    manifest = _write_json(root / "dataset_manifest.json", {
        "counts": {"validation_pair_count": 6},
    })
    return trainer_state, audit, manifest


class AnalyzeTests(unittest.TestCase):
    def test_modes_pass_for_happy_path_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = _valid_inputs(Path(directory))
            for mode in ("smoke", "overfit", "train", "validation"):
                with self.subTest(mode=mode):
                    result = analyze_paths(*paths, mode=mode)
                    self.assertEqual("passed", result["status"])
                    self.assertEqual([], result["reasons"])
                    self.assertGreater(result["final_reward_margin"], 0)
                    self.assertGreater(result["final_pair_accuracy"], 0.5)
                    self.assertTrue(result["finite_metrics"])

    def test_failures_record_reasons(self) -> None:
        cases = ("missing", "nan", "missing_eval", "lora_zero", "non_lora_nonzero", "checkpoint_missing")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for case in cases:
                with self.subTest(case=case):
                    trainer_state, audit, manifest = _valid_inputs(root)
                    if case == "missing":
                        trainer_state = root / "missing.json"
                    elif case == "nan":
                        state = json.loads(trainer_state.read_text(encoding="utf-8"))
                        state["log_history"][0]["loss"] = math.nan
                        trainer_state = _write_json(trainer_state, state)
                    elif case == "missing_eval":
                        state = json.loads(trainer_state.read_text(encoding="utf-8"))
                        state["log_history"][-1].pop("eval_loss")
                        state["log_history"][-1].pop("eval_rewards/margins")
                        state["log_history"][-1].pop("eval_rewards/accuracies")
                        trainer_state = _write_json(trainer_state, state)
                    elif case == "lora_zero":
                        audit = _write_json(audit, {**json.loads(audit.read_text()), "lora_delta_nonzero": False})
                    elif case == "non_lora_nonzero":
                        audit = _write_json(audit, {**json.loads(audit.read_text()), "non_lora_delta_zero": False})
                    elif case == "checkpoint_missing":
                        audit = _write_json(audit, {**json.loads(audit.read_text()), "checkpoint_exists": False})
                    mode = "train" if case == "missing_eval" else "smoke"
                    result = analyze_paths(trainer_state, audit, manifest, mode=mode)
                    self.assertEqual("failed", result["status"])
                    self.assertTrue(result["reasons"])

    def test_cli_writes_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trainer_state, audit, manifest = _valid_inputs(root)
            output = root / "dpo_gate_result.json"
            code = main([
                "--trainer-state", str(trainer_state), "--parameter-audit", str(audit),
                "--dataset-manifest", str(manifest), "--mode", "validation", "--output", str(output),
            ])
            self.assertEqual(0, code)
            self.assertEqual("passed", json.loads(output.read_text(encoding="utf-8"))["status"])


if __name__ == "__main__":
    unittest.main()
