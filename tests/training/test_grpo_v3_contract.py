from __future__ import annotations

import unittest

from training.grpo_v3_contract import (
    DEFAULTS,
    GATE3_STEPS,
    GATE4_EVAL_EVIDENCE,
    GATE4_STEPS,
    GATE4_TRAIN_EVIDENCE,
    assert_gate_transition,
    validate_formal_config,
)


class V3ContractTests(unittest.TestCase):
    def test_locked_defaults_match_strategy(self) -> None:
        self.assertEqual(DEFAULTS.policy_model, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(DEFAULTS.reviewer_model, "Qwen/Qwen3-VL-8B-Instruct")
        self.assertEqual(DEFAULTS.framework_version, "4.2.2")
        self.assertEqual(DEFAULTS.train_type, "lora")
        self.assertEqual(DEFAULTS.torch_dtype, "bfloat16")
        self.assertEqual(DEFAULTS.num_generations, 4)
        self.assertFalse(DEFAULTS.use_vllm)
        self.assertTrue(DEFAULTS.freeze_vit)
        self.assertTrue(DEFAULTS.freeze_aligner)
        self.assertEqual(DEFAULTS.lora_rank, 8)
        self.assertEqual(DEFAULTS.lora_alpha, 16)
        self.assertEqual(DEFAULTS.learning_rate, 1e-5)
        self.assertEqual(DEFAULTS.max_completion_length, 1024)

    def test_formal_config_rejects_sampled_frames_and_qlora(self) -> None:
        validate_formal_config({"policy_input": "native_video", "train_type": "lora"})
        with self.assertRaisesRegex(ValueError, "sampled_frames"):
            validate_formal_config({"policy_input": "sampled_frames", "train_type": "lora"})
        with self.assertRaisesRegex(ValueError, "QLoRA"):
            validate_formal_config({"policy_input": "native_video", "train_type": "qlora"})

    def test_gate_transition_is_strictly_sequential(self) -> None:
        assert_gate_transition(target_gate=0, passed_gates=[])
        assert_gate_transition(target_gate=1, passed_gates=[0])
        assert_gate_transition(target_gate=2, passed_gates=[0, 1])
        with self.assertRaisesRegex(ValueError, "Gate 0"):
            assert_gate_transition(target_gate=1, passed_gates=[])
        with self.assertRaisesRegex(ValueError, "Gate 1"):
            assert_gate_transition(target_gate=2, passed_gates=[0])

    def test_gate3_and_gate4_are_strictly_sequential(self) -> None:
        assert_gate_transition(target_gate=3, passed_gates=[0, 1, 2])
        assert_gate_transition(target_gate=4, passed_gates=[0, 1, 2, 3])
        with self.assertRaisesRegex(ValueError, "Gate 2"):
            assert_gate_transition(target_gate=3, passed_gates=[0, 1])
        with self.assertRaisesRegex(ValueError, "Gate 3"):
            assert_gate_transition(target_gate=4, passed_gates=[0, 1, 2])

    def test_gate3_gate4_defaults_match_strategy(self) -> None:
        self.assertEqual(GATE3_STEPS, 20)
        self.assertEqual(GATE4_STEPS, 40)
        self.assertEqual(GATE4_TRAIN_EVIDENCE, 40)
        self.assertEqual(GATE4_EVAL_EVIDENCE, 10)


if __name__ == "__main__":
    unittest.main()
