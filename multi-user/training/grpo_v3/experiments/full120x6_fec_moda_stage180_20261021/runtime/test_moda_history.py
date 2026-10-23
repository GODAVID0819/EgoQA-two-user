"""历史池直接使用旧分数，容量上限和CPU句向量池化。"""
from pathlib import Path
import importlib.util
import sys
import unittest
import tempfile
import json

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'multi-user'))


def row(i, window='DAY1_11110000', E=.4, old_reward=.4):
    return {'history_qa_id':'h'+str(i),'source_packet_id':'RLHF6U_'+window+'_A1_JAKE',
            'question':'question '+str(i),'quality_reward_old':old_reward,
            'probabilities_old_F_E_A_S':{'formality':.5,'groundedness':E,'all_six':.5,'speaker_only':.1}}


class HistoryTests(unittest.TestCase):
    def setUp(self):
        path=HERE/'multi-user/training/grpo_v3/six_user_binary/history_pool.py'
        self.assertTrue(path.is_file(),'旧分数固定历史池模块尚未实现')
        spec=importlib.util.spec_from_file_location('moda_history_target',path)
        self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
        self.assertEqual(getattr(self.module,'HISTORY_SCORE_SOURCE',None),'old_training_F_E_A_S')

    def test_capacity_is_ceiling_and_old_score_threshold_is_inclusive(self):
        rows=[row(i) for i in range(14)]+[row(20,'DAY2_11110000',E=.399),row(21,'DAY3_11110000')]
        selected,counts=self.module.select_history(rows,['DAY1_11110000','DAY2_11110000','DAY3_11110000'])
        self.assertEqual(len(selected),9)
        self.assertEqual(counts,{'DAY1_11110000':8,'DAY2_11110000':0,'DAY3_11110000':1})
        self.assertIn('h21',{r['history_qa_id'] for r in selected})

    def test_selection_is_stable_when_score_completion_order_changes(self):
        rows=[row(i) for i in range(15)]
        first=self.module.select_history(rows,['DAY1_11110000'])[0]
        second=self.module.select_history(list(reversed(rows)),['DAY1_11110000'])[0]
        self.assertEqual(first,second)

    def test_old_quality_prefilter_and_no_new_scores_mixed_into_old_pool(self):
        self.assertEqual(self.module.select_history([row(1,old_reward=.399)],['DAY1_11110000'])[0],[])
        wrong=row(2);wrong['probabilities_old_F_E_A_S']={'formality':1.,'groundedness':1.,'combined_answerability':1.}
        with self.assertRaises(ValueError):self.module.select_history([wrong],['DAY1_11110000'])

    def test_duplicate_ID_or_unknown_window_raises(self):
        with self.assertRaises(ValueError):self.module.select_history([row(1),row(1)],['DAY1_11110000'])
        with self.assertRaises(ValueError):self.module.select_history([row(1)],['DAY2_11110000'])

    def test_masked_mean_ignores_padding_and_normalizes(self):
        hidden=np.array([[[3.,4.],[999.,999.]],[[1.,0.],[0.,1.]]],dtype=np.float32)
        result=self.module.mean_normalize(hidden,np.array([[1,0],[1,1]]))
        np.testing.assert_allclose(result,[[.6,.8],[2**-.5,2**-.5]],rtol=1e-6)
        with self.assertRaises(ValueError):self.module.mean_normalize(np.zeros((1,2,3)),np.ones((1,2)))

    def test_saved_pool_round_trip_empty_pool_and_embedding_row_mismatch(self):
        class Encoder:
            def encode(self,questions):
                vectors=np.zeros((len(questions),384),dtype=np.float32)
                if len(questions):vectors[:,0]=1.
                return vectors
        with tempfile.TemporaryDirectory() as temporary:
            directory=Path(temporary)/'nonempty'
            self.module.save_history([row(1)],['DAY1_11110000'],Encoder(),directory,
                source_metadata='fixture')
            loaded,metadata=self.module.load_history(directory)
            self.assertEqual(metadata['selected_QA'],1)
            self.assertEqual(loaded['DAY1_11110000'][0]['history_qa_id'],'h1')
            path=directory/'history_bank.jsonl'
            stored=json.loads(path.read_text());stored['embedding_row']=1;path.write_text(json.dumps(stored)+'\n')
            with self.assertRaises(ValueError):self.module.load_history(directory)
            empty=Path(temporary)/'empty'
            self.module.save_history([],['DAY1_11110000'],Encoder(),empty,
                source_metadata='fixture')
            self.assertEqual(self.module.load_history(empty)[0],{})


if __name__=='__main__':unittest.main()
