"""已确认的LoRA覆盖必须真正进入Swift CLI，旧LR组默认不变。"""
import unittest
from test_pipeline import require_module


class LoRAScopeTests(unittest.TestCase):
    def config(self):
        from test_launch import run_config
        return run_config()

    def test_confirmed_modules_are_forwarded_to_cli(self):
        launch=require_module(self,'launch')
        c=self.config();c['lora_target_modules']=['q_proj','v_proj','in_proj_qkv']
        command=launch.swift_command(c,'/scratch/output')
        self.assertEqual(command[command.index('--target_modules')+1:],c['lora_target_modules'])

    def test_old_trials_keep_original_scope(self):
        launch=require_module(self,'launch')
        command=launch.swift_command(self.config(),'/scratch/output')
        self.assertEqual(command[command.index('--target_modules')+1:],['q_proj','v_proj'])

    def test_workflow_preserves_confirmed_scope(self):
        workflow=require_module(self,'workflow')
        c={'project_root':'/scratch/project','train_python':'/scratch/train/bin/python',
           'judge_python':'/scratch/judge/bin/python','model':'/scratch/model',
           'lora_target_modules':['q_proj','v_proj','in_proj_qkv']}
        self.assertEqual(workflow.training_config(c,job_id='1',phase='formal',max_steps=200)
                         .get('lora_target_modules'),c['lora_target_modules'])

    def test_unconfirmed_or_invalid_scope_is_rejected(self):
        launch=require_module(self,'launch')
        for value in ([],['all-linear'],['q_proj','q_proj'],['in_proj_z'],'q_proj'):
            c=self.config();c['lora_target_modules']=value
            with self.assertRaises(ValueError):launch.swift_command(c,'/scratch/output')


if __name__=='__main__':unittest.main()
