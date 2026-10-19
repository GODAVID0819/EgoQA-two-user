"""通过率：每个Judge按P(pass)>0.5，未打分算不通过；基座与adapter须按输入和种子配对。"""
import json
from pathlib import Path
import tempfile
import unittest
from eval_passrate import merge, pass_rates


def row(e, f, g, a, status='scored', seed=1):
    return {'evidence_id': e, 'seed': seed, 'status': status, 'reward': .2 * f + .4 * g + .4 * a,
            'probabilities': {'formality': f, 'groundedness': g, 'all_six': a} if status == 'scored' else {}}


class PassRateTests(unittest.TestCase):
    def test_threshold_strict_and_unscored_fails(self):
        rows = [row('a', .9, .6, .51), row('b', .5, .9, .9), row('c', 0, 0, 0, status='invalid_completion')]
        s = pass_rates(rows)
        self.assertEqual(s['pass_count'], {'formality': 1, 'groundedness': 2, 'all_six': 2, 'all_three': 1})
        self.assertAlmostEqual(s['valid_rate'], 2 / 3)
        self.assertAlmostEqual(s['mean_p_pass_scored_only']['formality'], .7)

    def test_merge_pairs_ranks(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            for rank, ids in enumerate((['a', 'c'], ['b', 'd'])):
                (out / f'rank{rank}').mkdir()
                for label, p in (('baseline', .4), ('policy', .8)):
                    rec = {'status': 'completed', 'adapter': None if label == 'baseline' else 'x',
                           'rows': [row(e, p, p, p) for e in ids]}
                    (out / f'rank{rank}' / f'validation_{label}.json').write_text(json.dumps(rec))
            s = merge(out, 2, 4)
            self.assertEqual(s['baseline']['pass_rate']['all_three'], 0)
            self.assertEqual(s['policy']['pass_rate']['all_three'], 1)
            with self.assertRaises(ValueError):
                merge(out, 2, 5)


if __name__ == '__main__':
    unittest.main()
