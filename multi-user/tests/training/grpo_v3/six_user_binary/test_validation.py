"""验收必须有实际更新、完整候选组和有限数值。"""
import json
from pathlib import Path
import tempfile
import unittest
from test_pipeline import require_module


class ValidationTests(unittest.TestCase):
    def test_checkpoint_alone_cannot_prove_training(self):
        validator = require_module(self, "validate_run")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "swift/checkpoint-1"
            checkpoint.mkdir(parents=True)
            (root / "run_config.json").write_text(json.dumps({"max_steps": 1, "num_generations": 2, "num_generations_eval": 2}))
            (checkpoint / "adapter_model.safetensors").write_bytes(b"fixture")
            (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 1, "log_history": [{"grad_norm": 1., "eval_loss": 0.}]}))
            rows = [{"request_id": f"batch:{i}", "evidence_id": "P", "source_packet_id": "P", "reward": x,
                     "judge_result": {"status": "scored"}} for i, x in enumerate((.2, .8))]
            (root / "reward_trace.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
            result = validator.validate(root, adapter_check=lambda _: True)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["candidate_count"], 2)
            self.assertEqual(result["source_packet_count"], 1)
            rows.pop()
            (root / "reward_trace.jsonl").write_text(json.dumps(rows[0]))
            result = validator.validate(root, adapter_check=lambda _: True)
            self.assertEqual(result["status"], "failed")
            self.assertIn("complete_observed_groups", result["failed_checks"])


if __name__ == "__main__":
    unittest.main()
