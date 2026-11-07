"""全池评测复用旧预测时，必须完整且不重复地匹配人工标签。"""
import unittest
from types import SimpleNamespace

from training.judge_sft.evaluation_set import merge_predictions, validate_evaluation_set


class EvaluationSetTests(unittest.TestCase):
    def test_additional_160_and_original_40_cover_the_200_once(self):
        gold = {str(i): int(i < 50) for i in range(200)}
        rows = [dict(example_id=k, gold=v, prediction=v if int(k) < 160 else 1-v,
                     pass_probability=float(v if int(k) < 160 else 1-v)) for k, v in gold.items()]
        merged, metrics = merge_predictions(gold, [rows[:160], rows[160:]])
        self.assertEqual(len(merged), 200)
        self.assertEqual(metrics['accuracy'], .8)
        self.assertEqual(metrics['confusion'], dict(tp=50, tn=110, fp=40, fn=0))

    def test_duplicate_or_missing_prediction_is_rejected(self):
        row = dict(example_id='a', gold=1, prediction=1, pass_probability=.9)
        for partitions in [[[row], [row]], [[]]]:
            with self.assertRaises(ValueError):
                merge_predictions({'a': 1}, partitions)

    def test_stale_gold_and_probability_mismatch_are_rejected(self):
        for row in [dict(example_id='a', gold=0, prediction=0, pass_probability=.1),
                    dict(example_id='a', gold=1, prediction=0, pass_probability=.9)]:
            with self.assertRaises(ValueError):
                merge_predictions({'a': 1}, [[row]])

    def test_explicit_subset_count_does_not_change_old_40_default(self):
        examples = [SimpleNamespace(example_id=str(i), target=int(i < 40),
                                   condition_type='cross_view_requirement') for i in range(160)]
        validate_evaluation_set(examples, 160, 40)
        with self.assertRaises(ValueError):
            validate_evaluation_set(examples, 40, 10)


if __name__ == '__main__':
    unittest.main()
