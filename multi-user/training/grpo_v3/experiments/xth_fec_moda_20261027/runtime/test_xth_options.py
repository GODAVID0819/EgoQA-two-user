"""本实验在 lxr F/E/C+diversity 代码上新增的部分：gated 质量奖励、xth 历史池来源、反馈环 Judge 反馈端点与切分。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'multi-user'))
from training.grpo_v3.six_user_binary import reward, diversity, history_pool


def request(i):
    qa = {'question': 'question ' + str(i), 'options': list('abcde'), 'correct': 'A', 'answer': 'a'}
    return {'request_id': str(i), 'source_packet_id': 'RLHF6U_DAY1_11110000_A1_A2_A3_A4_A5_A6',
            'asker_index': 0, 'completion': json.dumps(qa)}


class Encoder:
    def __init__(self, vectors): self.vectors = np.asarray(vectors, dtype=float)
    def encode(self, questions):
        assert len(questions) == len(self.vectors)
        return self.vectors


class GatedRewardTests(unittest.TestCase):
    def test_gated_formula(self):
        for g, on in ((.75, True), (.5, False), (.25, False), (.5000001, True)):
            p = {'formality': .5, 'groundedness': g, 'combined_answerability': .75}
            self.assertAlmostEqual(reward.aggregate(p, mode='continuous_gated'), .2 * .5 + .4 * g + (.4 * .75 if on else 0))
            if on:
                self.assertAlmostEqual(reward.aggregate(p, mode='continuous_gated'), reward.aggregate(p, mode='continuous'))

    def test_continuous_unchanged(self):
        self.assertAlmostEqual(reward.aggregate({'formality': .5, 'groundedness': .4, 'combined_answerability': .5}, mode='continuous'), .46)
        self.assertEqual(reward.QUALITY_MODES, ('continuous', 'continuous_gated'))

    def test_score_group_uses_mode_and_keeps_bonus_gate(self):
        scores = [{'formality': 1., 'groundedness': .45, 'combined_answerability': .9},
                  {'formality': 1., 'groundedness': .9, 'combined_answerability': .9},
                  {'formality': 1., 'groundedness': .9, 'combined_answerability': .2},
                  {'formality': .1, 'groundedness': .1, 'combined_answerability': .1}]
        results = [{'request_id': str(i), 'status': 'scored', 'probabilities': s} for i, s in enumerate(scores)]
        encoder = Encoder([[1, 0], [0, 1], [-1, 0], [0, -1]])
        for mode in reward.QUALITY_MODES:
            rows = diversity.score_group([request(i) for i in range(4)], results, encoder=encoder, diversity_lambda=.5, mode=mode)
            for row, s in zip(rows, scores):
                self.assertAlmostEqual(row['quality_reward'], reward.aggregate(s, mode=mode))
                self.assertEqual(row['bonus_gate'], all(v >= .5 for v in s.values()))
                self.assertAlmostEqual(row['reward'], row['quality_reward'] + .5 * int(row['bonus_gate']) * row['diversity_distance'])
        gated = diversity.score_group([request(i) for i in range(4)], results, encoder=encoder, mode='continuous_gated')
        self.assertAlmostEqual(gated[0]['quality_reward'], .2 + .4 * .45)
        with self.assertRaises(ValueError):
            diversity.score_group([request(i) for i in range(4)], results, encoder=encoder, mode='binary')

    def test_plugin_and_launch_accept_gated(self):
        from training.grpo_v3.six_user_binary import plugin, launch
        import inspect
        self.assertIn('continuous_gated', plugin.QUALITY_MODES)
        self.assertIn('"continuous_gated"', inspect.getsource(launch))


class XthHistoryTests(unittest.TestCase):
    def record(self, i, window='DAY1_11110000', f=.9, g=.9, a=.9):
        return {'history_qa_id': f'xth:1:{window}:{i}', 'source_packet_id': f'RLHF6U_{window}_A1_A2_A3_A4_A5_A6',
                'question': f'q{i}?', 'probabilities_xth_F_E_A': {'formality': f, 'groundedness': g, 'all_six': a}}

    def test_qualified_is_strict_three_pass(self):
        self.assertTrue(history_pool.reference_qualified(self.record(0)))
        self.assertFalse(history_pool.reference_qualified(self.record(0, g=.5)))
        with self.assertRaises(ValueError):
            history_pool.reference_qualified({**self.record(0), 'probabilities_xth_F_E_A': {'formality': .9, 'groundedness': .9}})

    def test_save_load_round_trip_capacity_8(self):
        records = [self.record(i) for i in range(12)] + [self.record(99, window='DAY2_11110000', a=.3)]
        windows = ['DAY1_11110000', 'DAY2_11110000']

        class Enc:
            def encode(self, questions):
                v = np.random.default_rng(0).normal(size=(len(questions), 384))
                return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
        with tempfile.TemporaryDirectory() as d:
            meta = history_pool.save_xth_history(records, windows, Enc(), Path(d) / 'pool', source_metadata='test')
            self.assertEqual((meta['score_source'], meta['selected_QA'], meta['empty_windows']), (history_pool.XTH_SCORE_SOURCE, 8, 1))
            by_window, loaded = history_pool.load_history(Path(d) / 'pool')
            self.assertEqual(len(by_window['DAY1_11110000']), 8)
            self.assertNotIn('DAY2_11110000', by_window)

    def test_build_from_feedback_loop_output(self):
        sys.path.insert(0, str(HERE))
        from build_xth_history import accepted_records
        qa = lambda q: json.dumps({'question': q, 'options': list('abcde'), 'correct': 'A', 'answer': 'a'})
        att = lambda n, ok: {'attempt': n, 'completion': qa(f'q{n}'), 'all_pass': ok,
                             'probabilities': {'formality': .9 if ok else .2, 'groundedness': .9, 'all_six': .9}}
        items = [{'evidence_id': 'RLHF6U_DAY1_11110000_A1_A2_A3_A4_A5_A6__ASKER_A1', 'sample': 0,
                  'attempts': [att(1, False), att(2, True)], 'accepted_attempt': 2},
                 {'evidence_id': 'RLHF6U_DAY1_11110000_A1_A2_A3_A4_A5_A6__ASKER_A1', 'sample': 1,
                  'attempts': [att(1, False)] * 3, 'accepted_attempt': None}]
        with tempfile.TemporaryDirectory() as d:
            run = Path(d) / 'fbloop_123'
            (run / 'rank0').mkdir(parents=True)
            (run / 'rank0/feedback_loop.json').write_text(json.dumps({'items': items}))
            rows = accepted_records(run)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]['question'], rows[0]['source_packet_id'], rows[0]['history_qa_id']),
                         ('q2', 'RLHF6U_DAY1_11110000_A1_A2_A3_A4_A5_A6', 'xth:123:RLHF6U_DAY1_11110000_A1_A2_A3_A4_A5_A6__ASKER_A1:s0'))


class FeedbackLoopTests(unittest.TestCase):
    def test_keys_and_feedback_text(self):
        from training.grpo_v3.six_user_binary import feedback_loop
        self.assertEqual(feedback_loop.KEYS, ('formality', 'groundedness', 'combined_answerability'))
        explained = {'combined_answerability': {'continuation': '", "reason": "Speaker sees it.", "available_evidence": [], "missing_evidence": []}'}}
        self.assertEqual(feedback_loop.judge_feedback(['combined_answerability'], explained), 'answerability: Speaker sees it.')

    def test_service_explain_routes_requested_keys_and_prefills_reason(self):
        from training.grpo_v3.six_user_binary import service

        class Example:
            def __init__(self, key): self.key = key

        class Predictor:
            calls = []
            def explain_many(self, examples, max_tokens, forced='fail'):
                self.calls.append(([e.key for e in examples], forced))
                if examples[0].key == 'groundedness' and forced == 'fail':
                    return [{'continuation': '",'} for _ in examples]
                return [{'continuation': '", "reason": "x"'} for _ in examples]
        scorer = object.__new__(service.CandidateScorer)
        scorer.predictor = Predictor()
        scorer.health = lambda: {'instance_id': 'i'}
        with patch.object(service, 'build_examples', side_effect=lambda r: {k: Example(k) for k in service.SCORING_KEYS}):
            out = scorer.explain_many([{'request_id': 'a', 'evidence_id': 'e', 'keys': ['groundedness']},
                                       {'request_id': 'b', 'evidence_id': 'e', 'keys': ['formality', 'combined_answerability']}])
        self.assertEqual([set(o['feedback']) for o in out], [{'groundedness'}, {'formality', 'combined_answerability'}])
        self.assertTrue(out[0]['feedback']['groundedness']['reason_prefilled'])
        with self.assertRaises(ValueError):
            scorer.explain_many([{'request_id': 'a', 'evidence_id': 'e', 'keys': ['all_six']}])

    def test_partition(self):
        sys.path.insert(0, str(HERE))
        from eval_feedback_loop import validation_partition
        self.assertEqual(sorted(sum((validation_partition(8, r, 4) for r in range(4)), [])), list(range(8)))
        with self.assertRaises(ValueError):
            validation_partition(6, 0, 4)


if __name__ == '__main__':
    unittest.main()
