"""已批准三项质量函数与组内距离的行为回归。"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'multi-user'))
from training.grpo_v3.six_user_binary import reward


def request(i, question=None):
    import json
    qa = {'question': question or 'question ' + str(i), 'options': list('abcde'), 'correct': 'A', 'answer': 'a'}
    return {'request_id': str(i), 'source_packet_id': 'RLHF6U_DAY1_11110000_A1_JAKE',
            'asker_index': 0, 'completion': json.dumps(qa)}


def result(i, *, valid=True, scores=None):
    return {'request_id': str(i), 'status': 'scored' if valid else 'invalid_completion',
            'probabilities': scores or {'formality': 1., 'groundedness': .5, 'combined_answerability': .5}}


class Encoder:
    def __init__(self, vectors): self.vectors = np.asarray(vectors, dtype=float); self.calls = 0
    def encode(self, questions):
        self.calls += 1
        if len(questions) != len(self.vectors): raise AssertionError('encoder batch mismatch')
        return self.vectors


class QualityTests(unittest.TestCase):
    def test_three_judge_quality_is_continuous_and_not_old_product(self):
        self.assertEqual(reward.KEYS, ('formality', 'groundedness', 'combined_answerability'))
        self.assertAlmostEqual(reward.aggregate({'formality': .5, 'groundedness': .4, 'combined_answerability': .5}, mode='continuous'), .46)

    def test_scores_reject_legacy_or_nonfinite_values(self):
        for scores in [dict(formality=.5, groundedness=.5, all_six=.5, speaker_only=.2),
                       dict(formality=True, groundedness=.5, combined_answerability=.5),
                       dict(formality=float('nan'), groundedness=.5, combined_answerability=.5)]:
            with self.assertRaises(ValueError): reward.aggregate(scores, mode='continuous')


class DiversityTests(unittest.TestCase):
    def setUp(self):
        path = HERE / 'multi-user/training/grpo_v3/six_user_binary/diversity.py'
        self.assertTrue(path.is_file(), '同窗静态池及G4距离模块尚未实现')
        spec = importlib.util.spec_from_file_location('moda_diversity_test_target', path)
        self.module = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.module)
        self.requests = [request(i) for i in range(4)]
        self.results = [result(i) for i in range(4)]
        self.encoder = Encoder([[1,0],[0,1],[-1,0],[0,-1]])

    def score(self, **kwargs):
        return self.module.score_group(self.requests, self.results, encoder=self.encoder, **kwargs)

    def test_current_G4_has_three_references_and_correct_bonus(self):
        rows = self.score()
        self.assertEqual([r['reference_count'] for r in rows], [3]*4)
        for row in rows:
            self.assertAlmostEqual(row['quality_reward'], .6)
            self.assertAlmostEqual(row['diversity_distance'], .5)
            self.assertAlmostEqual(row['diversity_bonus'], .25)
            self.assertAlmostEqual(row['reward'], .85)

    def test_quality_failed_current_neighbor_still_prevents_duplicate_bonus(self):
        self.encoder = Encoder([[1,0],[1,0],[-1,0],[0,-1]])
        self.results[1] = result(1, scores={'formality': .1, 'groundedness': .2, 'combined_answerability': .1})
        rows = self.score()
        self.assertEqual(rows[0]['diversity_distance'], 0.)
        self.assertFalse(rows[1]['bonus_gate'])
        self.assertEqual(rows[1]['diversity_bonus'], 0.)
        self.assertEqual(rows[0]['current_reference_count'], 3)

    def test_history_same_question_is_reference_and_other_window_is_excluded(self):
        history = {'DAY1_11110000': [{'history_qa_id': 'h0', 'embedding': [1,0]}],
                   'DAY2_11110000': [{'history_qa_id': 'wrong-window', 'embedding': [0,1]}]}
        rows = self.score(history_by_window=history)
        self.assertEqual(rows[0]['nearest_neighbor_id'], 'h0')
        self.assertEqual(rows[0]['diversity_distance'], 0.)
        self.assertEqual(rows[0]['reference_count'], 4)
        self.assertEqual(rows[1]['history_reference_count'], 1)

    def test_invalid_completion_not_neighbor_and_empty_reference_is_zero(self):
        self.results = [result(i, valid=i==0) for i in range(4)]
        self.encoder = Encoder([[1,0]])
        rows = self.score()
        self.assertEqual(rows[0]['reference_count'], 0)
        self.assertEqual(rows[0]['diversity_distance'], 0.)
        self.assertEqual([r['reward'] for r in rows[1:]], [0.]*3)

    def test_response_order_does_not_change_binding(self):
        expected = self.score()
        self.results.reverse()
        self.assertEqual(expected, self.score())

    def test_cross_speaker_or_window_and_duplicate_ids_raise(self):
        self.requests[1]['asker_index'] = 1
        with self.assertRaises(ValueError): self.score()
        self.requests[1]['asker_index'] = 0
        self.results[1]['request_id'] = self.results[0]['request_id']
        with self.assertRaises(ValueError): self.score()

    def test_bad_or_nonunit_embeddings_raise(self):
        for vectors in [[[0,0]]*4, [[float('nan'),0]]*4, [[2,0]]*4]:
            self.encoder = Encoder(vectors)
            with self.assertRaises(ValueError): self.score()

    def test_lambda_zero_does_not_encode_and_returns_exact_quality(self):
        rows = self.score(diversity_lambda=0)
        self.assertEqual(self.encoder.calls, 0)
        for row in rows:
            self.assertAlmostEqual(row['reward'], .6)
            self.assertEqual(row['reward'], row['quality_reward'])

    def test_total_reward_can_exceed_one(self):
        self.results = [result(i, scores={'formality':1.,'groundedness':1.,'combined_answerability':1.}) for i in range(4)]
        self.assertEqual([r['reward'] for r in self.score()], [1.25]*4)


if __name__ == '__main__': unittest.main()
