from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.modeling import (
    OUTPUT_FIELDS,
    head_module_names,
    last_nonpadding_indices_reference,
)


class ReviewerModelContractTests(unittest.TestCase):
    def test_fp32_heads_receive_fp32_pooled_representation(self) -> None:
        source = Path("training/grpo_v3/experiments/human_preference_reviewer/v1/modeling.py").read_text(encoding="utf-8")
        self.assertIn("head_input = pooled.to(dtype=first_head.weight.dtype)", source)
    def test_output_is_three_named_three_class_heads(self) -> None:
        self.assertEqual(
            OUTPUT_FIELDS,
            ("evidence_logits", "answerability_logits", "formality_logits"),
        )

    def test_last_nonpadding_supports_left_and_right_padding(self) -> None:
        self.assertEqual(
            last_nonpadding_indices_reference([[1, 1, 0, 0], [0, 0, 1, 1]]),
            [1, 3],
        )
        with self.assertRaisesRegex(ValueError, "no active token"):
            last_nonpadding_indices_reference([[0, 0]])

    def test_stage0_maps_to_only_the_evidence_head_module(self) -> None:
        self.assertEqual(
            head_module_names(("evidence_quality",)),
            ("evidence_head",),
        )
        with self.assertRaisesRegex(ValueError, "unsupported active head"):
            head_module_names(("overall_utility",))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is verified in the Torch training environment")
class TorchReviewerModelTests(unittest.TestCase):
    def test_stage0_registers_only_evidence_head(self) -> None:
        from torch import nn
        from training.grpo_v3.experiments.human_preference_reviewer.v1.modeling import ReviewerV1

        reviewer = ReviewerV1(nn.Identity(), hidden_size=8, active_heads=("evidence_quality",))

        self.assertTrue(hasattr(reviewer, "evidence_head"))
        self.assertFalse(hasattr(reviewer, "answerability_head"))
        self.assertFalse(hasattr(reviewer, "formality_head"))

    def test_three_heads_are_independent_and_structured(self) -> None:
        import torch
        from torch import nn
        from training.grpo_v3.experiments.human_preference_reviewer.v1.modeling import ReviewerV1

        class Backbone(nn.Module):
            def forward(self, input_ids, attention_mask, **kwargs):
                hidden = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
                return type("Output", (), {"last_hidden_state": hidden})()

        reviewer = ReviewerV1(Backbone(), hidden_size=8)
        output = reviewer(
            input_ids=torch.tensor([[1, 2, 0], [0, 3, 4]]),
            attention_mask=torch.tensor([[1, 1, 0], [0, 1, 1]]),
        )
        self.assertEqual(tuple(output.evidence_logits.shape), (2, 3))
        self.assertEqual(tuple(output.answerability_logits.shape), (2, 3))
        self.assertEqual(tuple(output.formality_logits.shape), (2, 3))
        self.assertIsNot(reviewer.evidence_head, reviewer.answerability_head)
        self.assertFalse(hasattr(output, "overall_utility"))


if __name__ == "__main__":
    unittest.main()
