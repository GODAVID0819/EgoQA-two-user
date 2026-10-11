"""任何未通过的零GPU项目都不能解锁提交。"""
import json
from pathlib import Path
import tempfile
import unittest
from test_pipeline import require_module


class SubmitAfterCPUTests(unittest.TestCase):
    def test_explicit_new_split_counts_allow_48_and_reject_mismatch(self):
        module=require_module(self,'submit_after_cpu')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'status.json').write_text('{"status":"passed"}')
            for name in module.CHECKS:
                (root/(name+'.json')).write_text(json.dumps({'status':'passed','checks':{
                    'pip_check':{'returncode':0},
                    'host_compilers':{'CC':{'shared_library_loaded':True},'CXX':{'shared_library_loaded':True}},
                    'rows':{'train':48,'validation':6}}}))
            module.verify_checks(root,expected_split_counts={'train':48,'validation':6})
            with self.assertRaises(ValueError):module.verify_checks(root)
            with self.assertRaises(ValueError):
                module.verify_checks(root,expected_split_counts={'train':42,'validation':6})

    def test_submission_contains_training_entry_and_configuration(self):
        module=require_module(self,'submit_after_cpu')
        c={'execution_mode':'direct','project_root':'/scratch/project','model':'/scratch/model',
           'train_python':'/scratch/train/bin/python','judge_python':'/scratch/judge/bin/python',
           'formal_max_steps':60,'walltime_seconds':46800,'walltime_basis':'实测剩余工作量',
           'slurm':{'account':'a','partition':'p','qos':'q','gres':'gpu:2','cpus':16,'mem':'500G'}}
        cmd=module.direct_submit_command(c,'/scratch/task/workflow.json','/scratch/task')
        self.assertTrue(cmd[-1].endswith('/train_direct.sbatch'))
        self.assertIn('--parsable',cmd)
        self.assertIn('--time=13:00:00',cmd)
        self.assertTrue(any('RUN_CONFIG=/scratch/task/workflow.json' in x for x in cmd))
        self.assertFalse(any('hold.sbatch' in x or 'keeper' in x for x in cmd))

    def test_missing_failed_or_incomplete_checks_block_submission(self):
        module=require_module(self,'submit_after_cpu')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            (root/'status.json').write_text('{"status":"running"}')
            with self.assertRaises(ValueError):module.verify_checks(root)
            (root/'status.json').write_text('{"status":"passed"}')
            with self.assertRaises(FileNotFoundError):module.verify_checks(root)
            for name in module.CHECKS:(root/(name+'.json')).write_text('{"status":"failed"}')
            with self.assertRaises(ValueError):module.verify_checks(root)

    def test_requires_real_compiler_load_and_both_dependency_checks(self):
        module=require_module(self,'submit_after_cpu')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'status.json').write_text('{"status":"passed"}')
            for name in module.CHECKS:
                (root/(name+'.json')).write_text(json.dumps({'status':'passed','checks':{
                    'pip_check':{'returncode':0},'host_compilers':{'CC':{'shared_library_loaded':True},'CXX':{'shared_library_loaded':True}},
                    'rows':{'train':18,'validation':6}}}))
            self.assertEqual(len(module.verify_checks(root)),5)
            report=json.loads((root/'judge_environment.json').read_text())
            report['checks']['host_compilers']['CXX']['shared_library_loaded']=False
            (root/'judge_environment.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):module.verify_checks(root)

    def test_search_submission_rejects_another_learning_rates_cli_report(self):
        module=require_module(self,'submit_after_cpu')
        from training.grpo_v3.six_user_binary.workflow import training_config
        c={'execution_mode':'direct','project_root':'/scratch/project','model':'/scratch/Qwen3.8-27B',
           'train_python':'/scratch/train/bin/python','judge_python':'/scratch/judge/bin/python',
           'formal_max_steps':200,'stop_after_steps':100,'learning_rate':3e-6,'eval_steps':50,
           'lr_scheduler_type':'cosine_with_min_lr','lr_scheduler_kwargs':{'min_lr_rate':.1},'warmup_steps':10}
        expected=training_config(c,job_id='0',phase='formal',max_steps=200)
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'status.json').write_text('{"status":"passed"}')
            for name in module.CHECKS:
                (root/(name+'.json')).write_text(json.dumps({'status':'passed','checks':{
                    'pip_check':{'returncode':0},'host_compilers':{'CC':{'shared_library_loaded':True},'CXX':{'shared_library_loaded':True}},
                    'rows':{'train':18,'validation':6}}}))
            with self.assertRaises(ValueError):module.verify_checks(root,expected_run=expected)
            report=json.loads((root/'train_environment.json').read_text())
            report['checks']['typed_cli']={key:expected[key] for key in
                ('max_steps','eval_steps','learning_rate','lr_scheduler_type','lr_scheduler_kwargs','warmup_steps')}
            report['checks']['typed_cli']['callbacks']=['egoqa_stage_stop']
            (root/'train_environment.json').write_text(json.dumps(report))
            module.verify_checks(root,expected_run=expected)
            report['checks']['typed_cli']['learning_rate']=1e-5
            (root/'train_environment.json').write_text(json.dumps(report))
            with self.assertRaises(ValueError):module.verify_checks(root,expected_run=expected)


if __name__=='__main__':unittest.main()
