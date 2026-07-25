from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_formality_artifacts import (
    summarize_formality_run,
    validate_formality_artifacts,
)


def _trace_row(call_index: int, candidate_index: int, reward: float) -> dict:
    return {
        "reward_kind": "qa_formality_confidence",
        "reward_call_index": call_index,
        "phase": "train",
        "candidate_index": candidate_index,
        "reward": reward,
        "record": {
            "masked": False,
            "judge_called": True,
            "reward_source": "judge_pass_fail_logprob_margin",
            "reward_components": {"qa_formality_confidence": reward},
            "judge_trace": {"qa_formality": {"parsed": {}}},
        },
    }


def _build_output(root: Path, *, mode: str) -> Path:
    output = root / mode
    adapter = output / "swift" / "checkpoint-1"
    processor = output / "processor"
    adapter.mkdir(parents=True)
    processor.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (processor / "processor_config.json").write_text("{}", encoding="utf-8")
    steps = 1 if mode == "smoke" else 40
    (adapter / "trainer_state.json").write_text(
        json.dumps({"global_step": steps, "log_history": [{"grad_norm": 1.0}]}),
        encoding="utf-8",
    )
    (output / "adapter_reload.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "adapter_dir": str(adapter.resolve()),
                "processor_dir": str(processor.resolve()),
                "lora_parameters": 16,
                "processor_reloaded": True,
            }
        ),
        encoding="utf-8",
    )
    (output / "storage_preflight.json").write_text(
        json.dumps(
            {
                "schema_version": "torch_storage_preflight_v1",
                "status": "passed",
                "allowed_root": str((root / "scratch").resolve()),
                "checks": {},
            }
        ),
        encoding="utf-8",
    )
    rows = [
        _trace_row(call_index, candidate, call_index * 0.01 + candidate * 0.02)
        for call_index in range(steps)
        for candidate in range(4)
    ]
    (output / "reward_trace.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    if mode == "probe":
        (output / "convergence_metrics.json").write_text(
            json.dumps({"status": "passed", "failed_checks": []}),
            encoding="utf-8",
        )
    return output


class FormalityArtifactTests(unittest.TestCase):
    def test_smoke_and_probe_require_exact_artifact_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            smoke = validate_formality_artifacts(_build_output(root, mode="smoke"), mode="smoke")
            probe = validate_formality_artifacts(_build_output(root, mode="probe"), mode="probe")

        self.assertEqual(smoke["status"], "passed")
        self.assertEqual(smoke["trace_count"], 4)
        self.assertEqual(smoke["global_step"], 1)
        self.assertEqual(probe["status"], "passed")
        self.assertEqual(probe["trace_count"], 160)
        self.assertEqual(probe["global_step"], 40)

    def test_probe_rejects_failed_convergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = _build_output(Path(tmp), mode="probe")
            (output / "convergence_metrics.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8"
            )
            result = validate_formality_artifacts(output, mode="probe")
        self.assertIn("convergence_metrics_passed", result["failed_checks"])

    def test_validation_rejects_missing_or_failed_storage_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            missing_output = _build_output(root / "missing", mode="smoke")
            (missing_output / "storage_preflight.json").unlink()
            missing = validate_formality_artifacts(missing_output, mode="smoke")

            failed_output = _build_output(root / "failed", mode="smoke")
            (failed_output / "storage_preflight.json").write_text(
                json.dumps({"status": "failed"}), encoding="utf-8"
            )
            failed = validate_formality_artifacts(failed_output, mode="smoke")

        self.assertIn("storage_preflight_passed", missing["failed_checks"])
        self.assertIn("storage_preflight_passed", failed["failed_checks"])

    def test_manifest_records_parent_hashes_and_single_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = _build_output(root, mode="smoke")
            parent = root / "gate2"
            parent.mkdir()
            dataset = root / "train.jsonl"
            dataset.write_text('{"evidence_id":"E1"}\n', encoding="utf-8")
            expected_dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
            result = validate_formality_artifacts(output, mode="smoke")
            (output / "formality_smoke_result.json").write_text(
                json.dumps(result), encoding="utf-8"
            )
            manifest = summarize_formality_run(
                output_dir=output,
                mode="smoke",
                dataset=dataset,
                parent_run=parent,
                policy_model="policy",
                reviewer_model="reviewer",
                job_id="123",
            )

        self.assertEqual(manifest["reward_revision"], "qa_formality_confidence_v1")
        self.assertEqual(manifest["reward_components"], ["qa_formality_confidence"])
        self.assertEqual(manifest["margin_clip"], 32.0)
        self.assertEqual(manifest["parent_run"], str(parent.resolve()))
        self.assertEqual(
            manifest["dataset_sha256"],
            expected_dataset_sha256,
        )
        self.assertFalse(manifest["calls_video_reviewer"])
        self.assertEqual(manifest["storage_preflight"]["status"], "passed")
        self.assertEqual(
            manifest["resolved_config"]["storage_preflight"]["status"],
            "passed",
        )


if __name__ == "__main__":
    unittest.main()
