"""任何未通过的零GPU项目都不能解锁提交。"""
import json
from pathlib import Path
import tempfile
import unittest
from test_pipeline import require_module


class SubmitAfterCPUTests(unittest.TestCase):
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


if __name__=='__main__':unittest.main()
