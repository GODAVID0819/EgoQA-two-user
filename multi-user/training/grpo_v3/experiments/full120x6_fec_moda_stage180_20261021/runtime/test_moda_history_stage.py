"""直接从旧训练分数建历史参照，不调用Judge重评分。"""
from pathlib import Path
import importlib.util
import json
import unittest
import sys
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'multi-user'),str(HERE)]


class HistoryStageTests(unittest.TestCase):
    def test_old_scores_are_preserved_without_constructing_Judge_requests(self):
        path=HERE/'history_stage.py'
        self.assertTrue(path.is_file())
        spec=importlib.util.spec_from_file_location('history_stage_test_target',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        self.assertTrue(callable(getattr(module,'historical_records',None)),'需要直接使用旧训练分数，禁止历史重评')
        packet='RLHF6U_DAY1_11110000_A1_JAKE'
        qa={'question':'What was next to my cup?','options':list('abcde'),'correct':'A','answer':'a'}
        source=[{'request_id':'old'+str(i),'status':'scored','reward':.5 if i!=1 else .3,
                 'source_packet_id':packet,'asker_index':0,'evidence_id':'e0','completion':json.dumps(qa),
                 'qa':qa,'global_step':i+1,'source_job':'old',
                 'probabilities':{'formality':.5,'groundedness':.4,'all_six':.5,'speaker_only':.1}} for i in range(4)]
        with patch('training.grpo_v3.six_user_binary.service.JudgeClient',side_effect=AssertionError('禁止历史重评分')):
            records=module.historical_records(source)
        self.assertEqual(len(records),4)
        self.assertEqual(records[0]['probabilities_old_F_E_A_S'],source[0]['probabilities'])
        self.assertNotIn('probabilities',records[0])
        self.assertFalse(hasattr(module,'score_partition'))


if __name__=='__main__':unittest.main()
