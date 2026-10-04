"""压缩说明不能改变媒体、核心QA结构或已有数据。"""
import copy
import unittest
from test_pipeline import require_module


class GenerationProfileTests(unittest.TestCase):
    def test_compact_profile_retains_original_prompt_and_media(self):
        module=require_module(self,'generation')
        row={'messages':[{'role':'user','content':'<image>\nOriginal schema and instructions.'}],
             'images':['/scratch/a.jpg'],'required_users':['a','b','c','d','e','f'],
             'source_packet_id':'P','asker_index':0}
        original=copy.deepcopy(row)
        updated=module.apply_profile(row,'compact_full_qa_v1')
        self.assertEqual(row,original)
        self.assertEqual(updated['images'],row['images'])
        self.assertEqual(updated['required_users'],row['required_users'])
        self.assertTrue(updated['messages'][0]['content'].startswith(row['messages'][0]['content']))
        self.assertIn('Keep every field',updated['messages'][0]['content'])
        self.assertEqual(module.apply_profile(updated,'compact_full_qa_v1'),updated)
        with self.assertRaises(ValueError):module.apply_profile(row,'unknown')

    def test_completion_budget_applies_to_training_and_validation_config(self):
        module=require_module(self,'workflow')
        config=module.training_config({'project_root':'/scratch/p','model':'/scratch/model',
            'train_python':'/scratch/train/bin/python','judge_python':'/scratch/judge/bin/python',
            'max_completion_length':2048,'baseline_after_training':True},job_id='123',phase='formal',max_steps=60)
        self.assertEqual(config['max_completion_length'],2048)
        self.assertTrue(config['baseline_after_training'])


if __name__=='__main__':unittest.main()
