from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_answer_margin import ANSWER_MARGIN_REWARD_REVISION
from training.grpo_v3_answer_margin_artifacts import expected_counts, validate_training_artifacts
from training.grpo_v3_answer_margin_preflight import FIXED_EVIDENCE_ID


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _build_output(root: Path, mode: str) -> Path:
    steps, row_count, _ = expected_counts(mode)
    output = root / mode
    adapter = output / "swift" / f"checkpoint-{steps}"
    processor = output / "processor"
    adapter.mkdir(parents=True)
    processor.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (processor / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    _write_json(adapter / "trainer_state.json", {"global_step": steps, "log_history": [{"loss": 0.1}]})
    _write_json(output / "storage_preflight.json", {"status": "passed"})
    _write_json(
        output / "environment_audit.json",
        {
            "status": "passed",
            "policy_environment": {"status": "passed", "python": "/env/train/python", "pip_check": "passed"},
            "scorer_environment": {"status": "passed", "python": "/env/scorer/python", "pip_check": "passed"},
        },
    )
    _write_json(output / "scorer_runtime_probe.json", {"status": "passed", "evidence_id": FIXED_EVIDENCE_ID})
    _write_json(
        output / "checkpoint_inventory.json",
        {
            "status": "passed",
            "parent_job": "gate2_14119442",
            "parent_checkpoint": "checkpoint-1",
            "gate2_result": {"status": "passed", "sha256": "1" * 64},
            "run_manifest": {"sha256": "2" * 64},
            "adapter_files": [{"path": "adapter_model.safetensors", "sha256": "3" * 64}],
            "source": "manifest_and_hash_inventory",
        },
    )
    _write_json(
        output / "resolved_config.json",
        {
            "condition_id": "t05",
            "temperature": 0.5,
            "num_generations": 4,
            "reward_revision": ANSWER_MARGIN_REWARD_REVISION,
            "evidence_id": FIXED_EVIDENCE_ID,
            "parent_job": "gate2_14119442",
            "parent_checkpoint": "checkpoint-1",
        },
    )
    _write_json(
        output / "adapter_reload.json",
        {
            "status": "passed",
            "adapter_dir": str(adapter.resolve()),
            "processor_dir": str(processor.resolve()),
            "lora_parameters": 16,
            "processor_reloaded": True,
            "inference_check": {"status": "passed"},
        },
    )
    _write_json(output / "run_manifest.json", {"status": "completed", "reward_revision": ANSWER_MARGIN_REWARD_REVISION, "condition_id": "t05", "checkpoint_inventory": "checkpoint_inventory.json"})
    rows = []
    for group in range(steps):
        for candidate in range(4):
            reward = group * 0.001 + candidate * 0.1 if group < expected_counts(mode)[2] else group * 0.001
            rows.append(
                {
                    "reward": reward,
                    "record": {
                        "reward_revision": ANSWER_MARGIN_REWARD_REVISION,
                        "experiment_condition_id": "t05",
                        "temperature": 0.5,
                        "evidence_id": FIXED_EVIDENCE_ID,
                        "global_step": group,
                        "reward_call_index": group,
                        "candidate_index": candidate,
                        "masked": False,
                        "normalized_reward": reward,
                        "label_scores": {label: {"sequence_logprob": -float(index)} for index, label in enumerate("ABCDE")},
                        "permutation": [0, 1, 2, 3, 4],
                        "inverse_permutation": [0, 1, 2, 3, 4],
                        "format_validation": {"status": "raw_valid"},
                    },
                }
            )
    self_count = len(rows)
    assert self_count == row_count
    (output / "reward_trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return output


class AnswerMarginArtifactTests(unittest.TestCase):
    def test_gate_counts(self) -> None:
        self.assertEqual(expected_counts("smoke1"), (1, 4, 1))
        self.assertEqual(expected_counts("smoke5"), (5, 20, 4))
        self.assertEqual(expected_counts("probe40"), (40, 160, 32))
        with self.assertRaises(ValueError):
            expected_counts("other")

    def test_all_modes_pass_exact_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            results = {
                mode: validate_training_artifacts(_build_output(Path(tmp), mode), mode=mode)
                for mode in ("smoke1", "smoke5", "probe40")
            }
        for mode, result in results.items():
            steps, rows, positive_groups = expected_counts(mode)
            self.assertEqual(result["run_status"], "passed", mode)
            self.assertEqual(result["research_signal_status"], "passed", mode)
            self.assertEqual(result["trace_count"], rows)
            self.assertEqual(result["global_step"], steps)
            self.assertGreaterEqual(result["positive_variance_group_count"], positive_groups)

    def test_research_failure_does_not_become_infrastructure_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = _build_output(Path(tmp), "smoke5")
            rows = [json.loads(line) for line in (output / "reward_trace.jsonl").read_text().splitlines()]
            for row in rows:
                row["reward"] = row["record"]["normalized_reward"] = 0.0
            (output / "reward_trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            result = validate_training_artifacts(output, mode="smoke5")
        self.assertEqual(result["run_status"], "passed")
        self.assertEqual(result["research_signal_status"], "failed")
        self.assertIn("required_positive_variance_groups", result["failed_research_checks"])

    def test_rejects_latest_pointer_wrong_parent_and_missing_environment_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = _build_output(Path(tmp), "smoke1")
            inventory = json.loads((output / "checkpoint_inventory.json").read_text())
            inventory["source"] = "latest_gate2_output.txt"
            inventory["parent_job"] = "gate3_14169924"
            _write_json(output / "checkpoint_inventory.json", inventory)
            (output / "environment_audit.json").unlink()
            result = validate_training_artifacts(output, mode="smoke1")
        self.assertEqual(result["run_status"], "invalid")
        self.assertIn("parent_inventory_frozen", result["failed_integrity_checks"])
        self.assertIn("separate_environment_audits_passed", result["failed_integrity_checks"])

    def test_rejects_nonfinite_mask_bad_group_shape_and_config_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = _build_output(Path(tmp), "smoke1")
            rows = [json.loads(line) for line in (output / "reward_trace.jsonl").read_text().splitlines()]
            rows[0]["reward"] = float("nan")
            (output / "reward_trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaisesRegex(ValueError, "NaN"):
                validate_training_artifacts(output, mode="smoke1")

            output = _build_output(Path(tmp) / "second", "smoke1")
            rows = [json.loads(line) for line in (output / "reward_trace.jsonl").read_text().splitlines()]
            rows[1]["record"]["masked"] = True
            rows[2]["record"]["candidate_index"] = 1
            (output / "reward_trace.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            config = json.loads((output / "resolved_config.json").read_text())
            config["temperature"] = 0.7
            _write_json(output / "resolved_config.json", config)
            result = validate_training_artifacts(output, mode="smoke1")
        self.assertEqual(result["run_status"], "invalid")
        self.assertIn("zero_infrastructure_masks", result["failed_integrity_checks"])
        self.assertIn("exact_four_candidates_per_group", result["failed_integrity_checks"])
        self.assertIn("locked_condition_config", result["failed_integrity_checks"])


if __name__ == "__main__":
    unittest.main()
