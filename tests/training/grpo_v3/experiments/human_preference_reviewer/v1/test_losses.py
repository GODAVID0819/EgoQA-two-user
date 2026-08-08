from __future__ import annotations

import importlib.util
import math
import unittest

from training.grpo_v3.experiments.human_preference_reviewer.v1.losses import (
    grade_to_target,
    mean_three_losses_reference,
)


class AbsoluteLossContractTests(unittest.TestCase):
    def test_grade_mapping_is_explicit(self) -> None:
        self.assertEqual([grade_to_target(value) for value in (1, 2, 3)], [0, 1, 2])
        for value in (0, 4, "2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                grade_to_target(value)

    def test_total_is_equal_arithmetic_mean(self) -> None:
        value = mean_three_losses_reference(0.3, 0.6, 1.2)
        self.assertTrue(math.isfinite(value))
        self.assertAlmostEqual(value, 0.7)


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is verified in the Torch training environment")
class TorchAbsoluteLossTests(unittest.TestCase):
    def test_three_cross_entropies_are_finite(self) -> None:
        import torch
        from training.grpo_v3.experiments.human_preference_reviewer.v1.losses import reviewer_losses
        from training.grpo_v3.experiments.human_preference_reviewer.v1.modeling import ReviewerOutput

        logits = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        output = ReviewerOutput(logits, logits.clone(), logits.clone())
        losses = reviewer_losses(output, {
            "evidence_quality": torch.tensor([1]),
            "answerability": torch.tensor([2]),
            "qa_formality": torch.tensor([3]),
        })
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(torch.allclose(
            losses["loss"],
            (losses["evidence_loss"] + losses["answerability_loss"] + losses["formality_loss"]) / 3,
        ))


if __name__ == "__main__":
    unittest.main()
