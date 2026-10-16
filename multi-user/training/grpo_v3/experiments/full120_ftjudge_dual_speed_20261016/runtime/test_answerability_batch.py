"""合并S/A后候选和条件仍逐一对应，非法候选仍原样标记。"""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[2]/'EgoQA-main-20260928/multi-user'))
spec=importlib.util.spec_from_file_location('training.grpo_v3.six_user_binary.speed_service',HERE/'service.py')
service=importlib.util.module_from_spec(spec);spec.loader.exec_module(service)


class Predictor:
    identity={'model_id':'fixture','frozen':True}
    config={'merge_answerability_batch':True}
    def __init__(self):self.batches=[]
    def predict_many(self,examples):
        self.batches.append([e.example_id for e in examples])
        return [{'pass_probability':e.probability} for e in examples]


def examples(request):
    if request.get('invalid'):raise service.InvalidCompletion('bad JSON')
    n=int(request['request_id'][1:])
    return {key:SimpleNamespace(example_id=request['request_id']+'::'+key,
                probability=(n+offset)/100,task=SimpleNamespace(value='answerability' if key in ('speaker_only','all_six') else key))
            for offset,key in enumerate(('formality','groundedness','speaker_only','all_six'),1)}


class BatchTests(unittest.TestCase):
    def test_s4_and_a4_are_one_eight_request_batch_with_correct_mapping(self):
        p=Predictor();scorer=service.CandidateScorer(p,'fixed')
        requests=[{'request_id':'r'+str(i),'evidence_id':'window'} for i in range(4)]
        with patch.object(service,'build_examples',side_effect=examples):results=scorer.score_many(requests)
        self.assertEqual([len(b) for b in p.batches],[4,4,8])
        for i,row in enumerate(results):
            self.assertEqual(row['request_id'],'r'+str(i))
            self.assertEqual(row['probabilities']['speaker_only'],(i+3)/100)
            self.assertEqual(row['probabilities']['all_six'],(i+4)/100)

    def test_invalid_candidates_are_excluded_without_shifting_other_bindings(self):
        p=Predictor();requests=[{'request_id':'r'+str(i),'evidence_id':'window','invalid':i==1} for i in range(4)]
        with patch.object(service,'build_examples',side_effect=examples):rows=service.CandidateScorer(p,'fixed').score_many(requests)
        self.assertEqual([len(b) for b in p.batches],[3,3,6])
        self.assertEqual(rows[1]['status'],'invalid_completion')
        self.assertEqual(rows[3]['probabilities']['speaker_only'],.06)
        self.assertEqual(rows[3]['probabilities']['all_six'],.07)


if __name__=='__main__':unittest.main()
