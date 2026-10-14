"""验证四窗口批次与完整数据覆盖，防止重复窗口或验证条件混入。"""
import collections
import csv
import unittest
from pathlib import Path

from pair_data import make_order, validate_conditions

ROOT = Path(__file__).resolve().parent


def load(name):
    with (ROOT/'conditions'/name).open(encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


class FourWindowTests(unittest.TestCase):
    def setUp(self):
        self.train = load('训练条件_600.csv')
        self.val = load('验证条件_120.csv')

    def test_two_complete_visits_and_candidate_groups(self):
        validate_conditions(self.train, self.val)
        order = make_order(self.train, seed=42)
        self.assertEqual(len(order), 600)
        self.assertEqual(order, make_order(list(reversed(self.train)), seed=42))
        counts = collections.Counter()
        for cycle in range(2):
            for start in range(0, len(order), 4):
                batch = order[start:start+4]
                self.assertEqual(len({r['source_packet_id'] for r in batch}), 4)
                candidates = [r for r in batch for _ in range(4)]
                for j in range(0,16,4):
                    self.assertEqual(len({(r['source_packet_id'],r['speaker_index']) for r in candidates[j:j+4]}),1)
                counts.update((r['source_packet_id'],r['speaker_index']) for r in batch)
        self.assertEqual(set(counts.values()), {2})
        self.assertEqual(sum(counts.values()), 1200)

    def test_reject_validation_condition_in_training(self):
        wrong = self.train[:-1]+[self.val[-1]]
        with self.assertRaises(ValueError):
            validate_conditions(wrong, self.val)

    def test_reject_duplicate_and_wrong_speaker_mapping(self):
        with self.assertRaises(ValueError):
            make_order(self.train[:-1]+[self.train[0]], seed=42)
        wrong=[dict(r) for r in self.train]
        wrong[0]['speaker_user']='错误用户'
        with self.assertRaises(ValueError):
            validate_conditions(wrong,self.val)


if __name__=='__main__':
    unittest.main()
