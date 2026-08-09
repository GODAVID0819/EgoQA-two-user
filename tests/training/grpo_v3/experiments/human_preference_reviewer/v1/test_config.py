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
            (60, 10, 0),
        )

    def test_zero_locked_test_is_valid_but_negative_is_rejected(self) -> None:
        config = ReviewerV1Config(
            train_evidence_count=60,
            validation_evidence_count=10,
            locked_test_evidence_count=0,
        )
        self.assertEqual(config.locked_test_evidence_count, 0)

        with self.assertRaisesRegex(ValueError, "locked test count"):
            ReviewerV1Config(locked_test_evidence_count=-1)

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

    def test_stage0_enables_only_evidence_head_and_disables_lora(self) -> None:
        config = ReviewerV1Config(stage="stage0")

        self.assertEqual(config.active_heads, ("evidence_quality",))
        self.assertFalse(config.lora_enabled)

    def test_stage2_remains_the_complete_default(self) -> None:
        config = ReviewerV1Config()

        self.assertEqual(config.stage, "stage2")
        self.assertEqual(
            config.active_heads,
            ("evidence_quality", "answerability", "qa_formality"),
        )
        self.assertTrue(config.lora_enabled)


if __name__ == "__main__":
    unittest.main()
