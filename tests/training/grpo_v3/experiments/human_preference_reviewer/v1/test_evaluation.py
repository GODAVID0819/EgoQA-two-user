from __future__ import annotations

import unittest

from training.grpo_v3.experiments.human_preference_reviewer.v1.evaluation import classification_metrics


class EvaluationTests(unittest.TestCase):
    def test_reports_fixed_three_class_metrics_and_expected_score(self) -> None:
        metrics = classification_metrics(
            labels=[1, 2, 3, 3],
            probabilities=[
                [0.8, 0.1, 0.1],
                [0.1, 0.7, 0.2],
                [0.1, 0.2, 0.7],
                [0.6, 0.2, 0.2],
            ],
        )

        self.assertEqual(metrics["confusion_matrix"], [[1, 0, 0], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["per_level"]["3"]["support"], 2)
        self.assertEqual(metrics["per_level"]["1"]["predicted_count"], 2)
        self.assertAlmostEqual(metrics["expected_scores"][0], 1.3)
        self.assertGreater(metrics["expected_score_mae"], 0.0)
        self.assertFalse(metrics["insufficient_class_support"])

    def test_absent_level_is_explicit_not_hidden(self) -> None:
        metrics = classification_metrics(labels=[1, 1], probabilities=[[1, 0, 0], [1, 0, 0]])
        self.assertTrue(metrics["insufficient_class_support"])
        self.assertEqual(metrics["per_level"]["3"]["support"], 0)
        self.assertEqual(len(metrics["confusion_matrix"]), 3)


if __name__ == "__main__":
    unittest.main()
