"""验证单GPU切换顺序及异常时不会让两个大模型同时驻留。"""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import threading
from test_pipeline import require_module
from test_launch import run_config


class SharedGpuTests(unittest.TestCase):
    def window(self, *, fail_wake=False, fail_sleep=False):
        module = require_module(self, 'shared_gpu')
        events = []
        class Client:
            def resource(self, action):
                events.append(action)
                if action == 'wake' and fail_wake:
                    raise RuntimeError('wake failed')
                if action == 'sleep' and fail_sleep:
                    raise RuntimeError('sleep failed')
                return {'sleeping': action == 'sleep'}
        return events, module.judge_window(Client(), lambda: events.append('policy_off'),
                                            lambda: events.append('policy_on'))

    def test_policy_is_released_before_judge_and_restored_last(self):
        events, window = self.window()
        with window:
            events.append('score')
        self.assertEqual(events, ['policy_off', 'wake', 'score', 'sleep', 'policy_on'])

    def test_scoring_error_still_releases_judge_before_restore(self):
        events, window = self.window()
        with self.assertRaisesRegex(ValueError, 'score'):
            with window:
                raise ValueError('score failed')
        self.assertEqual(events, ['policy_off', 'wake', 'sleep', 'policy_on'])

    def test_failed_judge_release_never_reloads_policy(self):
        events, window = self.window(fail_sleep=True)
        with self.assertRaises(RuntimeError):
            with window:
                events.append('score')
        self.assertNotIn('policy_on', events)

    def test_partial_wake_is_cleaned_before_policy_restore(self):
        events, window = self.window(fail_wake=True)
        with self.assertRaisesRegex(RuntimeError, 'wake'):
            with window:
                self.fail('不应进入评分')
        self.assertEqual(events, ['policy_off', 'wake', 'sleep', 'policy_on'])

    def test_shared_mode_requires_explicit_single_gpu_contract(self):
        launch = require_module(self, 'launch')
        c = run_config()
        c.update(shared_gpu=True, policy_gpu_indices=[0], judge_gpu_indices=[0], use_vllm=True)
        launch.validate_config(c)
        c['shared_gpu'] = False
        with self.assertRaises(ValueError):
            launch.validate_config(c)

    def test_shared_judge_enables_real_sleep_mode(self):
        predictor = require_module(self, 'predictor')
        options = predictor.engine_options({'model_id': '/model', 'judge_mode': 'baseline', 'shared_gpu': True})
        self.assertIs(options.get('enable_sleep_mode'), True)

    def test_resource_control_rejects_another_job_instance(self):
        service = require_module(self, 'service')
        class Predictor:
            identity = {}
            def resource(self, action):
                return {'sleeping': action == 'sleep'}
        scorer = service.CandidateScorer(Predictor(), 'this-job')
        self.assertTrue(hasattr(scorer, 'resource'))
        with self.assertRaises(ValueError):
            scorer.resource({'instance_id': 'other-job', 'action': 'wake'})
        self.assertTrue(scorer.resource({'instance_id': 'this-job', 'action': 'sleep'})['sleeping'])

    def test_real_http_resource_roundtrip_and_wrong_instance(self):
        service = require_module(self, 'service')
        class Predictor:
            identity = {}
            def resource(self, action):
                return {'sleeping': action == 'sleep'}
        server = service.make_server(service.CandidateScorer(Predictor(), 'mine'), port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = 'http://127.0.0.1:' + str(server.server_port)
            client = service.JudgeClient(url, expected_instance='mine')
            self.assertFalse(client.resource('wake')['sleeping'])
            self.assertTrue(client.resource('sleep')['sleeping'])
            with self.assertRaises(RuntimeError):
                service.JudgeClient(url, expected_instance='other').resource('wake')
        finally:
            server.shutdown()
            thread.join(5)
            server.server_close()

    def test_training_wrapper_moves_both_models_and_optimizer_in_order(self):
        module = require_module(self, 'shared_gpu')
        events = []
        class Trainer:
            args = SimpleNamespace(offload_model=True, offload_optimizer=True)
            engine = SimpleNamespace(inner_model_executor=SimpleNamespace(is_sleeping=True))
            accelerator = SimpleNamespace(unwrap_model=lambda model: model)
            model, ref_model, optimizer = 'policy', 'reference', object()
            def offload_model(self, model): events.append('off_' + model)
            def load_model(self, model): events.append('on_' + model)
            def offload_optimizer(self): events.append('off_optimizer')
            def load_optimizer(self): events.append('on_optimizer')
            def _score_completions(self, value):
                events.append('score')
                return value
        class Client:
            def resource(self, action):
                events.append(action)
                return {'sleeping': action == 'sleep'}
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None))
        module.install_training_switch(Client(), Trainer)
        with patch.dict('sys.modules', {'torch': fake_torch}):
            self.assertEqual(Trainer()._score_completions(123), 123)
        self.assertEqual(events, ['off_policy', 'off_reference', 'off_optimizer', 'wake', 'score',
                                  'sleep', 'on_policy', 'on_reference', 'on_optimizer'])

    def test_failed_wake_does_not_skip_actual_sleep_cleanup(self):
        predictor = require_module(self, 'predictor')
        judge = object.__new__(predictor.VllmJudge)
        judge.config, judge.sleeping = {'shared_gpu': True}, True
        events = []
        def wake():
            events.append('wake')
            raise RuntimeError('partial wake')
        judge.llm = SimpleNamespace(wake_up=wake, reset_prefix_cache=lambda: events.append('reset'),
                                   sleep=lambda level: events.append('sleep'))
        with self.assertRaises(RuntimeError): judge.resource('wake')
        self.assertIsNone(judge.sleeping)
        self.assertTrue(judge.resource('sleep')['sleeping'])
        self.assertEqual(events, ['wake', 'reset', 'sleep'])

    def test_shared_smoke_reloads_adapter_without_changing_formal_resume(self):
        workflow = require_module(self, 'workflow')
        c = {'project_root': '/scratch/xl6775/project', 'train_python': '/train/python',
             'judge_python': '/judge/python', 'model': '/model/Qwen3.8-27B',
             'shared_gpu': True, 'use_vllm': True, 'resume_from_checkpoint': '/old/checkpoint-36'}
        smoke = workflow.training_config(c, job_id='123', phase='smoke', max_steps=1)
        formal = workflow.training_config(c, job_id='123', phase='formal', max_steps=60)
        self.assertNotIn('resume_from_checkpoint', smoke)
        self.assertTrue(smoke['paired_validation'])
        self.assertTrue(smoke['baseline_after_training'])
        self.assertEqual(formal['resume_from_checkpoint'], '/old/checkpoint-36')
        self.assertEqual(formal['max_steps'], 60)


if __name__ == '__main__':
    unittest.main()
