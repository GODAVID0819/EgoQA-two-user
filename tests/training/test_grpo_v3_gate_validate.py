from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_gate_validate import validate_gate_artifacts


class GateArtifactValidationTests(unittest.TestCase):
    def _artifacts(
        self,
        root: Path,
        rewards: list[float | None],
        *,
        format_statuses: list[str] | None = None,
        global_step: int = 1,
    ) -> None:
        (root / "trainer_state.json").write_text(json.dumps({"global_step": global_step}), encoding="utf-8")
        adapter = root / "adapter"
        adapter.mkdir()
        (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"weights")
        processor = root / "processor"
        processor.mkdir()
        (processor / "preprocessor_config.json").write_text("{}", encoding="utf-8")
        (root / "adapter_reload.json").write_text(
            json.dumps({
                "status": "passed",
                "adapter_dir": str(adapter.resolve()),
                "lora_parameters": 8,
                "processor_dir": str(processor.resolve()),
                "processor_reloaded": True,
            }),
            encoding="utf-8",
        )
        with (root / "reward_trace.jsonl").open("w", encoding="utf-8") as handle:
            for index, value in enumerate(rewards):
                status = (format_statuses or ["raw_valid"] * len(rewards))[index]
                operations = (
                    [{"operation": "insert_missing_member_comma", "position": 17}]
                    if status == "repaired"
                    else []
                )
                handle.write(json.dumps({
                    "candidate_index": index,
                    "reward": value,
                    "record": {
                        "masked": value is None,
                        "format_validation": {
                            "status": status,
                            "repair_operations": operations,
                        },
                    },
                }) + "\n")

    def test_gate1_and_gate2_accept_valid_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(root, [0.0, 1.0, 2.0, 3.0])
            self.assertEqual(validate_gate_artifacts(root, gate=1)["status"], "passed")
            self.assertEqual(validate_gate_artifacts(root, gate=2)["status"], "passed")

    def test_gate2_accepts_unrecoverable_finite_reward_and_reports_format_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(
                root,
                [0.5, -1.9, -3.0, -1.9],
                format_statuses=["raw_valid", "repaired", "unrecoverable", "raw_valid"],
            )
            result = validate_gate_artifacts(root, gate=2)

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["masked_reward_count"], 0)
        self.assertEqual(result["format_raw_valid_count"], 2)
        self.assertEqual(result["format_repaired_count"], 1)
        self.assertEqual(result["format_unrecoverable_count"], 1)
        self.assertEqual(result["format_raw_valid_rate"], 0.5)
        self.assertEqual(result["format_repaired_rate"], 0.25)
        self.assertEqual(result["format_unrecoverable_rate"], 0.25)
        self.assertEqual(result["format_repair_operation_counts"], {"insert_missing_member_comma": 1})
        self.assertEqual(result["format_reward_revision"], "json_three_tier_v1")
        self.assertEqual(result["format_repaired_penalty"], -0.5)
        self.assertEqual(result["format_unrecoverable_reward"], -3.0)

    def test_gate2_rejects_masks_zero_variance_and_missing_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(root, [1.0, 1.0, None, None])
            result = validate_gate_artifacts(root, gate=2)
            self.assertEqual(result["status"], "failed")
            self.assertIn("reward_std_positive", result["failed_checks"])
            self.assertIn("no_masked_rewards", result["failed_checks"])
            self.assertIn("four_finite_rewards", result["failed_checks"])
            (root / "adapter" / "adapter_model.safetensors").unlink()
            result = validate_gate_artifacts(root, gate=2)
            self.assertIn("adapter_complete", result["failed_checks"])
            (root / "processor" / "preprocessor_config.json").unlink()
            result = validate_gate_artifacts(root, gate=2)
            self.assertIn("processor_config_saved", result["failed_checks"])
            reload_path = root / "adapter_reload.json"
            reload_result = json.loads(reload_path.read_text(encoding="utf-8"))
            reload_result["processor_reloaded"] = False
            reload_path.write_text(json.dumps(reload_result), encoding="utf-8")
            result = validate_gate_artifacts(root, gate=2)
            self.assertIn("processor_reload_passed", result["failed_checks"])

    def test_gate3_and_gate4_require_convergence_steps_and_split_integrity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(root, [0.0, 1.0, 2.0, 3.0], global_step=40)
            (root / "convergence_metrics.json").write_text(
                json.dumps({"status": "passed", "failed_checks": []}), encoding="utf-8"
            )
            self.assertEqual(validate_gate_artifacts(root, gate=3)["status"], "passed")
            (root / "split_manifest.json").write_text(
                json.dumps({
                    "train_count": 40,
                    "eval_count": 10,
                    "intersection_count": 0,
                    "train_evidence_ids": [f"T{i}" for i in range(40)],
                    "eval_evidence_ids": [f"V{i}" for i in range(10)],
                }),
                encoding="utf-8",
            )
            self.assertEqual(validate_gate_artifacts(root, gate=4)["status"], "passed")
            split = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
            split["eval_evidence_ids"][0] = "T0"
            (root / "split_manifest.json").write_text(json.dumps(split), encoding="utf-8")
            result = validate_gate_artifacts(root, gate=4)
            self.assertIn("split_ids_disjoint", result["failed_checks"])

    def test_gate3_rejects_failed_convergence_or_too_few_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._artifacts(root, [0.0, 1.0, 2.0, 3.0], global_step=19)
            (root / "convergence_metrics.json").write_text(
                json.dumps({"status": "failed", "failed_checks": ["train_reward_improved"]}), encoding="utf-8"
            )
            result = validate_gate_artifacts(root, gate=3)
        self.assertIn("convergence_metrics_passed", result["failed_checks"])
        self.assertIn("required_global_steps", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
