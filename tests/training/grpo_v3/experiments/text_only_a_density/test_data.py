from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3.experiments.text_only_a_density.data import (
    build_records,
    validate_dataset,
    write_dataset_bundle,
)


class DensityDataTests(unittest.TestCase):
    def test_frozen_train_and_eval_records_are_text_only(self) -> None:
        train = build_records("train")
        evaluation = build_records("eval")
        self.assertEqual([row["trial_id"] for row in train], [f"train-{i:02d}" for i in range(10)])
        self.assertEqual([row["trial_id"] for row in evaluation], [f"eval-{i:02d}" for i in range(32)])
        for row in train + evaluation:
            self.assertEqual(set(row), {"messages", "trial_id", "phase"})
            self.assertEqual(row["messages"][0]["role"], "user")
            text = json.dumps(row).lower()
            for forbidden in ("video", "image", "answer", "reward", "judge"):
                self.assertNotIn(forbidden, text)

    def test_bundle_hash_and_schema_detect_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_dataset_bundle(root)
            result = validate_dataset(root / "train.jsonl", manifest, phase="train")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["record_count"], 10)
            (root / "train.jsonl").write_text(
                (root / "train.jsonl").read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                validate_dataset(root / "train.jsonl", manifest, phase="train")


if __name__ == "__main__":
    unittest.main()
