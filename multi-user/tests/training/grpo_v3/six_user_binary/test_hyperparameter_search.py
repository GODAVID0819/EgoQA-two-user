"""搜索配置必须真正到达CLI，筛选终点不能重设学习率总长度。"""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from test_pipeline import require_module
from test_launch import run_config


class HyperparameterSearchTests(unittest.TestCase):
    def test_workflow_to_cli_preserves_search_and_stage_settings(self):
        workflow = require_module(self, 'workflow')
        launch = require_module(self, 'launch')
        c = {'project_root': '/scratch/project', 'model': '/scratch/Qwen3.8-27B',
             'train_python': '/scratch/train/bin/python', 'judge_python': '/scratch/judge/bin/python',
             'learning_rate': 3e-6, 'beta': .01, 'temperature': .8,
             'lr_scheduler_type': 'cosine_with_min_lr', 'lr_scheduler_kwargs': {'min_lr_rate': .1},
             'warmup_steps': 10, 'stop_after_steps': 100, 'eval_steps': 50,
             'seed': 42, 'data_seed': 42}
        run = workflow.training_config(c, job_id='123', phase='formal', max_steps=200)
        cmd = launch.swift_command(run, '/scratch/output')
        for key in ('learning_rate', 'beta', 'temperature', 'lr_scheduler_type', 'warmup_steps', 'eval_steps'):
            self.assertEqual(cmd[cmd.index('--' + key) + 1], str(c[key]))
        self.assertEqual(json.loads(cmd[cmd.index('--lr_scheduler_kwargs') + 1]), c['lr_scheduler_kwargs'])
        self.assertEqual(cmd[cmd.index('--max_steps') + 1], '200')
        self.assertEqual(run['stop_after_steps'], 100)
        self.assertEqual(cmd[cmd.index('--callbacks') + 1], 'egoqa_stage_stop')
        self.assertNotIn('stop_after_steps', workflow.training_config(c, job_id='123', phase='smoke', max_steps=1))

    def test_invalid_stage_endpoint_is_rejected_before_launch(self):
        launch = require_module(self, 'launch')
        for value in (0, True, 201):
            c = {**run_config(), 'max_steps': 200, 'stop_after_steps': value}
            with self.subTest(value=value), self.assertRaises(ValueError):
                launch.validate_config(c)

    def test_stage_callback_stops_saves_and_evaluates_without_changing_horizon(self):
        stage = require_module(self, 'stage_stop')
        with patch.dict('os.environ', {'EGOQA_GRPO_STOP_AT_STEP': '100'}):
            callback = stage.StageStopCallback(SimpleNamespace(max_steps=200), None)
        args = SimpleNamespace(max_steps=200)
        control = SimpleNamespace(should_training_stop=False, should_save=False, should_evaluate=False)
        callback.on_step_end(args, SimpleNamespace(global_step=99), control)
        self.assertFalse(control.should_training_stop)
        callback.on_step_end(args, SimpleNamespace(global_step=100), control)
        self.assertTrue(control.should_training_stop)
        self.assertTrue(control.should_save)
        self.assertTrue(control.should_evaluate)
        self.assertEqual(args.max_steps, 200)

    def test_continuation_callback_runs_until_200(self):
        stage = require_module(self, 'stage_stop')
        with patch.dict('os.environ', {'EGOQA_GRPO_STOP_AT_STEP': '200'}):
            callback = stage.StageStopCallback(SimpleNamespace(max_steps=200), None)
        control = SimpleNamespace(should_training_stop=False, should_save=False, should_evaluate=False)
        callback.on_step_end(SimpleNamespace(max_steps=200), SimpleNamespace(global_step=100), control)
        self.assertFalse(control.should_training_stop)

    def test_single_gpu_submission_uses_automatic_partition_and_qos(self):
        submit = require_module(self, 'submit')
        c = {**run_config(), 'execution_mode': 'direct', 'shared_gpu': True,
             'use_vllm': True, 'policy_gpu_indices': [0], 'judge_gpu_indices': [0],
             'slurm': {'account': 'a', 'gres': 'gpu:1', 'constraint': 'h200', 'cpus': 16, 'mem': '500G'},
             'walltime_basis': '实测100步及验证时间'}
        cmd = submit.sbatch_command(c, '/scratch/config.json', '/scratch/task')
        self.assertFalse(any(x.startswith(('--partition=', '--qos=', '--nodelist=')) for x in cmd))
        self.assertIn('--parsable', cmd)
        self.assertIn('--constraint=h200', cmd)

    def test_submitter_accepts_200_step_schedule_with_100_step_stage(self):
        submit = require_module(self, 'submit_after_cpu')
        self.assertTrue(hasattr(submit, 'validate_submission_config'))
        c = {'formal_max_steps': 200, 'stop_after_steps': 100,
             'paired_validation': True, 'execution_mode': 'direct'}
        submit.validate_submission_config(c)
        with self.assertRaises(ValueError):
            submit.validate_submission_config({**c, 'stop_after_steps': 201})

    def test_stage_validation_requires_complete_100_step_checkpoint_on_200_horizon(self):
        validator = require_module(self, 'validate_run')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / 'swift/v0/checkpoint-100'
            checkpoint.mkdir(parents=True)
            (root / 'run_config.json').write_text(json.dumps({'max_steps': 200, 'stop_after_steps': 100,
                'num_generations': 2, 'num_generations_eval': 2}))
            (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 100, 'max_steps': 200,
                'log_history': [{'grad_norm': .1, 'eval_reward': .4}]}))
            for name in ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt',
                         'rng_state.pth', 'args.json', 'training_args.bin'):
                (checkpoint / name).write_bytes(b'fixture')
            rows = [{'request_id': f'batch:{i}', 'evidence_id': 'P', 'source_packet_id': 'P',
                     'reward': reward, 'judge_result': {'status': 'scored'}} for i, reward in enumerate((.2, .8))]
            (root / 'reward_trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
            result = validator.validate(root, adapter_check=lambda _: True)
            self.assertEqual(result['status'], 'passed')
            self.assertEqual(result['schedule_max_steps'], 200)
            self.assertEqual(result['stage_target_steps'], 100)
            (checkpoint / 'optimizer.pt').unlink()
            self.assertEqual(validator.validate(root, adapter_check=lambda _: True)['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
