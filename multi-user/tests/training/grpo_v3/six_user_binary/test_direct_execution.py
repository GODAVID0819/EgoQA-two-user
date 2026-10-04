"""实际执行批处理壳层，检查训练自动启动及失败状态传播。"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from test_pipeline import require_module


class DirectExecutionTests(unittest.TestCase):
    def test_direct_config_has_no_external_allocation_dependency(self):
        workflow = require_module(self, 'workflow')
        c = {'project_root':'/scratch/project','model':'/scratch/model',
             'train_python':'/scratch/train/bin/python','judge_python':'/scratch/judge/bin/python',
             'execution_mode':'direct'}
        config = workflow.training_config(c,job_id='123',phase='formal',max_steps=60)
        self.assertEqual(config.get('execution_mode'),'direct')
        self.assertNotIn('allocation_manifest',config)

    @unittest.skipIf(os.name == 'nt', '实际 Bash 生命周期在 Torch 登录节点验证')
    def test_batch_starts_training_and_propagates_success_and_failure(self):
        script = Path(__file__).resolve().parents[4] / 'hpc/grpo_v3/six_user_binary/train_direct.sbatch'
        self.assertTrue(script.is_file())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'multi-user').mkdir();config=root/'config.json';config.write_text('{}')
            python=root/'fixture_python';trace=root/'calls.jsonl'
            python.write_text('#!'+sys.executable+'\nimport os,sys,json\n'
                'with open(os.environ["TEST_CALLS"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\n'
                'if "training.grpo_v3.six_user_binary.resume" in sys.argv:sys.exit(int(os.environ["TEST_EXIT"]))\n')
            python.chmod(0o700)
            env={**os.environ,'PROJECT_ROOT':str(root),'TRAIN_PYTHON':str(python),
                 'RUN_CONFIG':str(config),'SLURM_JOB_ID':'123','TEST_CALLS':str(trace)}
            for code in (0,7):
                trace.unlink(missing_ok=True)
                result=subprocess.run(['bash',str(script)],env={**env,'TEST_EXIT':str(code)},capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,code,result.stderr)
                calls=[json.loads(s) for s in trace.read_text().splitlines()]
                self.assertEqual(sum('training.grpo_v3.six_user_binary.resume' in call for call in calls),1)
                self.assertFalse(any('cuda.py' in str(call) or 'attach' in str(call) for call in calls))


if __name__=='__main__':unittest.main()
