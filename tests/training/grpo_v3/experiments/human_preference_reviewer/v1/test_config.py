from __future__ import annotations

import unittest

from training.grpo_v3.experiments.human_preference_reviewer.v1.config import ReviewerV1Config


class ReviewerV1ConfigTests(unittest.TestCase):
    def test_defaults_lock_first_stage_contract(self) -> None:
        config = ReviewerV1Config()

        self.assertEqual(config.model_name_or_path, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(config.num_labels, 3)
        self.assertEqual(config.last_n_shared_blocks, 2)
        self.assertEqual(config.lora_target_modules, ("q_proj", "v_proj"))
        self.assertEqual((config.lora_r, config.lora_alpha), (8, 16))
        self.assertEqual(config.lora_dropout, 0.05)
        self.assertEqual(
            (config.train_evidence_count, config.validation_evidence_count, config.locked_test_evidence_count),
            (40, 10, 10),
        )

    def test_rejects_contract_drift(self) -> None:
        invalid = (
            {"model_name_or_path": ""},
            {"num_labels": 2},
            {"last_n_shared_blocks": 3},
            {"lora_r": 0},
            {"lora_dropout": 1.0},
        )
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ReviewerV1Config(**kwargs)


if __name__ == "__main__":
    unittest.main()
