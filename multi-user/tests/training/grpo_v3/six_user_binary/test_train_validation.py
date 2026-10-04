"""两集合、60步与固定验证对比的边界。"""
import json
from pathlib import Path
import tempfile
import unittest
from test_pipeline import require_module


class TrainValidationTests(unittest.TestCase):
    def test_merge_old_test_keeps_validation_and_all_source_windows_disjoint(self):
        module = require_module(self, 'train_validation')
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d)/'old', Path(d)/'new'
            src.mkdir()
            old = {'train': [{'source_packet_id':p,'asker_index':a} for p in ['T1','T2'] for a in range(6)],
                   'validation':[{'source_packet_id':'V','asker_index':a} for a in range(6)],
                   'test':[{'source_packet_id':'T3','asker_index':a} for a in range(6)]}
            for k, rows in old.items():
                (src/(k+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            report = module.prepare(src, dst)
            self.assertEqual(report['rows'], {'train':18,'validation':6})
            self.assertEqual((src/'validation.jsonl').read_bytes(), (dst/'validation.jsonl').read_bytes())
            self.assertFalse((dst/'test.jsonl').exists())
            self.assertTrue((src/'test.jsonl').exists())

    def test_formal_steps_and_validation_frequency_are_explicit(self):
        workflow = require_module(self, 'workflow')
        launch = require_module(self, 'launch')
        c={'project_root':'/scratch/project','prepared_data_root':'/scratch/prepared',
           'train_python':'/scratch/train/bin/python','judge_python':'/scratch/judge/bin/python',
           'model':'/scratch/Qwen3.8-27B','formal_max_steps':60,'eval_steps':20,'save_steps':20,
           'save_total_limit':3,'paired_validation':True,'validation_seed':42}
        run = workflow.training_config(c,job_id='123',phase='formal',max_steps=60)
        self.assertEqual(run['eval_steps'],20)
        self.assertTrue(run['paired_validation'])
        cmd=launch.swift_command(run,'/scratch/out')
        self.assertEqual(cmd[cmd.index('--max_steps')+1],'60')
        self.assertEqual(cmd[cmd.index('--save_total_limit')+1],'3')
        smoke=workflow.training_config(c,job_id='123',phase='smoke',max_steps=1)
        self.assertFalse(smoke.get('paired_validation',False))
        self.assertEqual(smoke.get('eval_steps',1),1)


if __name__ == '__main__': unittest.main()
