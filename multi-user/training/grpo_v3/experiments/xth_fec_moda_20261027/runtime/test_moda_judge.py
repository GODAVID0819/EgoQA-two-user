"""新模板的模型可见内容和三路候选绑定。"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'multi-user'))
from training.grpo_v3.six_user_binary import judge, service

KEYS = ('formality','groundedness','combined_answerability')


def view_for(asker):
    order = [asker]+[i for i in range(6) if i != asker]
    names = ['Jake','Alice','Tasha','Lucia','Katrina','Shure']
    return {'dataset_root': '/scratch/fixture', 'packet_id': 'packet', 'source_packet_id': 'packet',
            'evidence_id': 'packet__ASKER_'+str(asker), 'required_users': [names[i] for i in order],
            'clips':[{'user_index':i,'agent_name':names[i], 'media_role':'speaker' if pos==0 else 'provider'}
                     for pos,i in enumerate(order)]}


def make_request(asker=0):
    qa = {'question': 'What was next to my cup?', 'options': ['book','bag','plate','bowl','hat'],
          'correct':'A','answer':'book','generator_rationale':'HIDDEN_RATIONALE_SENTINEL'}
    return {'request_id':'r0','source_packet_id':'packet','dataset_root':'/scratch/fixture',
            'asker_index':asker,'evidence_id':'packet__ASKER_'+str(asker),'completion':json.dumps(qa)}


class JudgeTests(unittest.TestCase):
    def test_all_six_speaker_orders_and_exactly_one_joint_C_request(self):
        for asker in range(6):
            view = view_for(asker)
            with patch.object(judge,'full_judge_view',return_value=view), patch.object(judge,'normalized_record_to_example',side_effect=lambda r:r):
                rows = judge.build_examples(make_request(asker))
            self.assertEqual(tuple(rows), KEYS)
            self.assertNotIn('frame_packet', rows['formality'])
            for key in ['groundedness','combined_answerability']:
                self.assertEqual(rows[key]['frame_user_indices'][0], asker)
                self.assertEqual(set(rows[key]['frame_user_indices']), set(range(6)))
                self.assertEqual(len(rows[key]['frame_order']),6)
            c = rows['combined_answerability']
            self.assertEqual(c['condition_type'],'combined_all_six_users')
            self.assertIn('BOTH requirements hold',c['prompt'])
            self.assertIn('What was next to my cup?',c['prompt'])
            self.assertNotIn('HIDDEN_RATIONALE_SENTINEL',c['prompt'])
            self.assertNotIn('"correct"',c['prompt'])
            self.assertNotIn('{question}',c['prompt'])

    def test_F_template_is_verbatim_bound_and_E_contains_real_QA(self):
        request = make_request()
        with patch.object(judge,'full_judge_view',return_value=view_for(0)), patch.object(judge,'normalized_record_to_example',side_effect=lambda r:r):
            rows = judge.build_examples(request)
        self.assertEqual(tuple(rows), KEYS)
        f = rows['formality']['prompt'];e=rows['groundedness']['prompt']
        self.assertIn('FAIL when the answer is merely one person', f)
        self.assertNotIn('<DETERMINISTIC_SCHEMA_RESULT_JSON>',f)
        self.assertNotIn('<PARTICIPANT_NAMES_JSON>',f)
        self.assertNotIn('<QUESTION_ANSWER_ITEM_JSON>',f)
        self.assertNotIn('HIDDEN_RATIONALE_SENTINEL', f)
        self.assertIn('"status": "PASS"', f)
        self.assertIn('HIDDEN_RATIONALE_SENTINEL',e)
        self.assertNotIn('{video_packet_brief(packet)}',e)
        self.assertNotIn('{qa_item JSON}',e)

    def test_real_name_leakage_changes_deterministic_F_branch(self):
        request=make_request();qa=json.loads(request['completion']);qa['question']='What did Alice put next to my cup?';request['completion']=json.dumps(qa)
        with patch.object(judge,'full_judge_view',return_value=view_for(0)), patch.object(judge,'normalized_record_to_example',side_effect=lambda r:r):
            rows=judge.build_examples(request)
        self.assertIn('"status": "FAIL"',rows['formality']['prompt'])


class Predictor:
    identity={'model_id':'fixture','frozen':True}
    config={'merge_answerability_batch':False}
    def __init__(self): self.batches=[]
    def predict_many(self,examples):
        self.batches.append([e.example_id for e in examples])
        return [{'pass_probability':e.probability} for e in examples]


class ServiceTests(unittest.TestCase):
    def test_three_G4_stages_and_invalid_candidate_keeps_slot(self):
        self.assertEqual(getattr(service,'SCORING_KEYS',()),KEYS)
        def examples(request):
            if request['request_id']=='r1': raise service.InvalidCompletion('bad JSON')
            return {key:SimpleNamespace(example_id=request['request_id']+'::'+key,
                    probability=(int(request['request_id'][1:])+offset)/10)
                    for offset,key in enumerate(KEYS,1)}
        requests=[{'request_id':'r'+str(i),'evidence_id':'packet'} for i in range(4)]
        predictor=Predictor()
        with patch.object(service,'build_examples',side_effect=examples):
            rows=service.CandidateScorer(predictor,'test-instance').score_many(requests)
        self.assertEqual([len(b) for b in predictor.batches],[3,3,3])
        self.assertEqual(rows[1]['status'],'invalid_completion')
        self.assertEqual(rows[3]['probabilities']['combined_answerability'],.6)


if __name__=='__main__': unittest.main()
