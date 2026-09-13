from __future__ import annotations

import math
import unittest

from training.grpo_v3.experiments.text_only_a_density.domain import score_completion


class DensityDomainTests(unittest.TestCase):
    def test_frozen_reward_boundaries(self) -> None:
        cases = {
            "AAAA": (4, 0, 1.0),
            "BBBB": (0, 4, -1.0),
            "ABAB": (2, 2, 0.0),
            "": (0, 0, -1.0),
            "xyz": (0, 0, -1.0),
            "AaBb!": (1, 1, 0.0),
            "```json\n{\"x\":\"AAA\"}\n```": (3, 0, 1.0),
        }
        for completion, expected in cases.items():
            with self.subTest(completion=completion):
                result = score_completion(completion)
                self.assertEqual((result.n_A, result.n_B, result.reward), expected)
                self.assertTrue(math.isfinite(result.reward))
                self.assertGreaterEqual(result.reward, -1)
                self.assertLessEqual(result.reward, 1)

    def test_counts_every_non_uppercase_ab_character_without_normalizing(self) -> None:
        result = score_completion("A B\naabb!")
        self.assertEqual(result.n_valid, 2)
        self.assertEqual(result.non_ab_character_count, 7)


if __name__ == "__main__":
    unittest.main()
