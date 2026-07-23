from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from training.grpo_v3.baseline.summary import summarize_run


class V3SummaryTests(unittest.TestCase):
    def test_gate3_manifest_records_content_reward_revision_and_audit_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data.jsonl"
            dataset.write_text(json.dumps({"evidence_id": "E1", "videos": ["a.mp4", "b.mp4"]}) + "\n", encoding="utf-8")
            (root / "gate3_result.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            audit = root / "groundedness_audit_summary.json"
            audit.write_text(json.dumps({"approved_for_weight_change": True}), encoding="utf-8")
            parent = root / "gate2"
            parent.mkdir()
            (parent / "run_manifest.json").write_text("{}", encoding="utf-8")
            with patch.dict(
                "os.environ",
                {
                    "EGOQA_CONTENT_REWARD_REVISION": "ground_answer_gap_v1",
                    "EGOQA_GROUNDEDNESS_AUDIT_SUMMARY": str(audit),
                },
            ):
                manifest = summarize_run(
                    output_dir=root, gate=3, dataset=dataset,
                    policy_model="Qwen/Qwen3-VL-2B-Instruct",
                    reviewer_model="Qwen/Qwen3-VL-8B-Instruct",
                    job_id="789", parent_run=parent,
                )
        self.assertEqual(manifest["content_reward_revision"], "ground_answer_gap_v1")
        self.assertEqual(manifest["groundedness_audit_summary"], str(audit.resolve()))
        self.assertEqual(len(manifest["groundedness_audit_summary_sha256"]), 64)

    def test_manifest_separates_controlled_and_formal_reward(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data.jsonl"
            dataset.write_text(
                json.dumps({"evidence_id": "E1", "videos": ["a.mp4", "b.mp4"]}) + "\n",
                encoding="utf-8",
            )
            (root / "gate1_result.json").write_text(
                json.dumps({"status": "passed", "reward_std": 1.0}), encoding="utf-8"
            )
            swift_dir = root / "swift"
            swift_dir.mkdir()
            (swift_dir / "args.json").write_text(
                json.dumps({"lora_rank": 8, "learning_rate": 1e-5}), encoding="utf-8"
            )
            parent = root / "gate0"
            parent.mkdir()
            (parent / "run_manifest.json").write_text(
                json.dumps({"gate_result": {"media_metadata": {"actual_video_count": 2}}}), encoding="utf-8"
            )
            manifest = summarize_run(
                output_dir=root,
                gate=1,
                dataset=dataset,
                policy_model="Qwen/Qwen3-VL-2B-Instruct",
                reviewer_model=None,
                job_id="123",
                parent_run=parent,
                slurm_stdout=root / "gate1.out",
                slurm_stderr=root / "gate1.err",
                reviewer_log=None,
            )
        self.assertEqual(manifest["gate_status"], "passed")
        self.assertEqual(manifest["reward_revision"], "gate1_controlled_non_formal")
        self.assertFalse(manifest["formal_reward_result"])
        self.assertEqual(manifest["evidence_ids"], ["E1"])
        self.assertEqual(manifest["policy_input"], "native_video")
        self.assertEqual(manifest["train_type"], "lora")
        self.assertEqual(manifest["parent_run"], str(parent.resolve()))
        self.assertIn("resolved_config", manifest)
        self.assertEqual(len(manifest["dataset_sha256"]), 64)
        self.assertEqual(manifest["upstream_gate0_media_metadata"]["actual_video_count"], 2)
        self.assertEqual(manifest["logs"]["slurm_stdout"], str((root / "gate1.out").resolve()))
        self.assertEqual(manifest["logs"]["slurm_stderr"], str((root / "gate1.err").resolve()))
        self.assertIsNone(manifest["logs"]["reviewer"])
        self.assertEqual(manifest["swift_configs"][0]["config"]["lora_rank"], 8)
        self.assertEqual(manifest["swift_configs"][0]["config"]["learning_rate"], 1e-5)

    def test_gate2_manifest_records_three_tier_format_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "data.jsonl"
            dataset.write_text(
                json.dumps({"evidence_id": "E1", "videos": ["a.mp4", "b.mp4"]}) + "\n",
                encoding="utf-8",
            )
            (root / "gate2_result.json").write_text(
                json.dumps({"status": "passed", "reward_std": 1.0}), encoding="utf-8"
            )
            traces = [
                (0.5, "raw_valid", []),
                (-1.9, "repaired", [{"operation": "insert_missing_member_comma", "position": 17}]),
                (-3.0, "unrecoverable", []),
                (-1.9, "raw_valid", []),
            ]
            with (root / "reward_trace.jsonl").open("w", encoding="utf-8") as handle:
                for index, (reward, status, operations) in enumerate(traces):
                    handle.write(json.dumps({
                        "candidate_index": index,
                        "reward": reward,
                        "record": {
                            "masked": False,
                            "format_validation": {
                                "status": status,
                                "repair_operations": operations,
                            },
                        },
                    }) + "\n")
            parent = root / "gate1"
            parent.mkdir()
            (parent / "run_manifest.json").write_text("{}", encoding="utf-8")

            manifest = summarize_run(
                output_dir=root,
                gate=2,
                dataset=dataset,
                policy_model="Qwen/Qwen3-VL-2B-Instruct",
                reviewer_model="Qwen/Qwen3-VL-8B-Instruct",
                job_id="456",
                parent_run=parent,
            )

        self.assertEqual(manifest["reward_revision"], "json_three_tier_v1")
        self.assertEqual(manifest["format_reward_revision"], "json_three_tier_v1")
        self.assertEqual(manifest["format_raw_valid_count"], 2)
        self.assertEqual(manifest["format_repaired_count"], 1)
        self.assertEqual(manifest["format_unrecoverable_count"], 1)
        self.assertEqual(manifest["format_repair_operation_counts"], {"insert_missing_member_comma": 1})
        self.assertEqual(manifest["format_repaired_penalty"], -0.5)
        self.assertEqual(manifest["format_unrecoverable_reward"], -3.0)

    def test_gate4_manifest_records_split_eval_and_convergence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train = root / "train.jsonl"
            eval_dataset = root / "eval.jsonl"
            train.write_text(json.dumps({"evidence_id": "T1", "videos": ["a.mp4", "b.mp4"]}) + "\n", encoding="utf-8")
            eval_dataset.write_text(json.dumps({"evidence_id": "V1", "videos": ["c.mp4", "d.mp4"]}) + "\n", encoding="utf-8")
            (root / "gate4_result.json").write_text(json.dumps({"status": "passed"}), encoding="utf-8")
            convergence = {"status": "passed", "train": {"reward_delta": 0.2}, "eval": {"reward_delta": -0.05}}
            (root / "convergence_metrics.json").write_text(json.dumps(convergence), encoding="utf-8")
            split = root / "split_manifest.json"
            split.write_text(json.dumps({
                "train_count": 40,
                "eval_count": 10,
                "train_evidence_ids": ["T1"],
                "eval_evidence_ids": ["V1"],
            }), encoding="utf-8")
            parent = root / "gate3"
            parent.mkdir()
            (parent / "run_manifest.json").write_text("{}", encoding="utf-8")
            manifest = summarize_run(
                output_dir=root,
                gate=4,
                dataset=train,
                eval_dataset=eval_dataset,
                split_manifest=split,
                policy_model="Qwen/Qwen3-VL-2B-Instruct",
                reviewer_model="Qwen/Qwen3-VL-8B-Instruct",
                job_id="789",
                parent_run=parent,
            )
        self.assertTrue(manifest["formal_reward_result"])
        self.assertEqual(manifest["reward_revision"], "json_three_tier_v1")
        self.assertEqual(manifest["eval_evidence_ids"], ["V1"])
        self.assertEqual(len(manifest["eval_dataset_sha256"]), 64)
        self.assertEqual(manifest["split_manifest"]["train_count"], 40)
        self.assertEqual(manifest["convergence_metrics"]["status"], "passed")


if __name__ == "__main__":
    unittest.main()
