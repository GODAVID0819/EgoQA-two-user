"""真实ORM接口将总奖励与候选级指标逐槽位返回。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'multi-user'))
from training.grpo_v3.six_user_binary import plugin
from test_moda_reward import Encoder, request


class Client:
    def score_many(self,requests):
        return [{'request_id':r['request_id'],'evidence_id':r['evidence_id'],'status':'scored',
                 'probabilities':{'formality':1.,'groundedness':.5,'combined_answerability':.5},
                 'judge':{'instance_id':'fixture','frozen':True}} for r in requests]


class PluginTests(unittest.TestCase):
    def test_ORM_uses_diversity_and_logs_four_slots_with_quality_parts(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(plugin,'validate_row'):
            trace=Path(temporary)/'reward_trace.jsonl'
            reward=plugin.SixUserBinaryReward(client=Client(),trace_path=trace,reward_mode='continuous',
                encoder=Encoder([[1,0],[0,1],[-1,0],[0,-1]]),history_by_window={},diversity_lambda=.5)
            values=reward([request(i)['completion'] for i in range(4)],dataset_root='/scratch/fixture',
                source_packet_id=request(0)['source_packet_id'],asker_index=0,evidence_id='fixture',
                required_users=['A','B','C','D','E','F'],generator_image_paths=['/scratch/fixture.jpg'])
            self.assertEqual(len(values),4)
            for value in values:self.assertAlmostEqual(value,.85)
            rows=[json.loads(line) for line in trace.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(rows),4)
            self.assertEqual({r['global_step'] for r in rows},{1})
            self.assertTrue(all(r['bonus_gate'] and r['reference_count']==3 for r in rows))
            self.assertEqual({r['reward_revision'] for r in rows},{'fec_moda_static_history_v1'})


if __name__=='__main__':unittest.main()
