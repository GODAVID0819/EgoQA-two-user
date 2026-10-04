"""批处理的候选身份、媒体缓存与 rollout 配置回归。"""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from test_pipeline import require_module
from test_launch import run_config


class AccelerationTests(unittest.TestCase):
    def test_predictor_keeps_order_across_bounded_batches(self):
        predictor = require_module(self, 'predictor')
        judge = object.__new__(predictor.VllmJudge)
        judge.config = {'max_num_seqs': 2}
        judge.adapters = {}
        judge.sampling = object()
        judge.pass_id, judge.fail_id = 1,2
        judge.prepare = lambda e: ({'prompt': str(e.index)}, {'index':e.index})
        calls = []
        def generate(payloads, *args, **kwargs):
            calls.append([p['prompt'] for p in payloads])
            return [SimpleNamespace(outputs=[SimpleNamespace(logprobs=[{1: -float(p['prompt']), 2: -3.}])]) for p in payloads]
        judge.llm = SimpleNamespace(generate=generate)
        values = judge.predict_many([SimpleNamespace(index=i, task=SimpleNamespace(value='formality')) for i in range(3)])
        self.assertEqual(calls, [['0','1'],['2']])
        self.assertEqual([v['index'] for v in values], [0,1,2])
        self.assertGreater(values[0]['pass_probability'], values[2]['pass_probability'])

    def test_cleanup_reaps_workers_even_after_launcher_exits(self):
        launch = require_module(self, 'launch')
        import signal
        calls = []
        process = SimpleNamespace(pid=123, poll=lambda: 1, wait=lambda timeout: 1)
        launch.cleanup_processes([process], kill_group=lambda pid, sig: calls.append((pid,sig)))
        self.assertIn((123, signal.SIGTERM), calls)
        self.assertIn((123, getattr(signal, 'SIGKILL', 9)), calls)

    def test_colocated_rollout_retains_group_and_updates_only_lora(self):
        launch = require_module(self, 'launch')
        c = {**run_config(), 'use_vllm': True, 'attn_impl': 'sdpa'}
        cmd = launch.swift_command(c, '/scratch/test/output')
        value = lambda key: cmd[cmd.index('--' + key) + 1]
        self.assertEqual(value('use_vllm'), 'true')
        self.assertEqual(value('vllm_mode'), 'colocate')
        self.assertEqual(value('vllm_enable_lora'), 'true')
        self.assertEqual(value('sleep_level'), '1')
        self.assertEqual(value('offload_model'), 'true')
        self.assertEqual(value('num_generations'), '4')
        import json
        self.assertEqual(json.loads(value('vllm_engine_kwargs'))['additional_config']['gdn_prefill_backend'], 'triton')
        predictor = require_module(self, 'predictor')
        self.assertEqual(predictor.engine_options({'model_id':'/scratch/model','judge_mode':'baseline'})['additional_config']['gdn_prefill_backend'], 'triton')

    def test_batch_scorer_preserves_invalid_positions_and_task_mapping(self):
        service = require_module(self, 'service')
        calls = []
        class Predictor:
            identity = {'model_id': 'fixture'}
            def predict_many(self, examples):
                calls.append([e.tag for e in examples])
                return [{'pass_probability': e.probability} for e in examples]
        def examples(request):
            if request['request_id'] == 'bad':
                raise service.InvalidCompletion('bad json')
            return {key: SimpleNamespace(tag=request['request_id'] + key, probability=p)
                    for key,p in [('formality', .8), ('groundedness', .7), ('speaker_only', .2), ('all_six', .9)]}
        requests = [{'request_id': key, 'evidence_id': 'E'} for key in ('a','bad','b')]
        scorer = service.CandidateScorer(Predictor(), 'fixture')
        with patch.object(service, 'build_examples', side_effect=examples):
            results = scorer.score_many(requests)
        self.assertEqual([r['request_id'] for r in results], ['a','bad','b'])
        self.assertEqual(results[1]['status'], 'invalid_completion')
        self.assertEqual(results[2]['probabilities']['speaker_only'], .2)
        self.assertEqual(calls[0], ['aformality', 'bformality'])
        self.assertEqual(len(calls), 4)

    def test_batch_client_rejects_swapped_or_missing_results(self):
        service = require_module(self, 'service')
        client = service.JudgeClient('http://fixture', expected_instance='current')
        requests = [{'request_id': key, 'evidence_id': 'E'} for key in ('a','b')]
        replies = [{'request_id': r['request_id'], 'evidence_id':'E', 'status':'invalid_completion',
                    'reason':'fixture', 'judge':{'instance_id':'current'}} for r in requests]
        with patch.object(client, '_request', return_value={'results':replies[::-1]}):
            with self.assertRaisesRegex(RuntimeError, 'identity'):
                client.score_many(requests)
        with patch.object(client, '_request', return_value={'results':replies[:1]}):
            with self.assertRaisesRegex(RuntimeError, '数量'):
                client.score_many(requests)

    def test_media_uuid_preserves_order_and_preprocessing_identity(self):
        predictor = require_module(self, 'predictor')
        ids = predictor.media_uuids(('a.jpg','b.jpg'), effective=24576, min_pixels=3136, patch_size=16, namespace='run1')
        same = predictor.media_uuids(('a.jpg','b.jpg'), effective=24576, min_pixels=3136, patch_size=16, namespace='run1')
        changed = predictor.media_uuids(('a.jpg','b.jpg'), effective=262144, min_pixels=3136, patch_size=16, namespace='run1')
        self.assertEqual(ids, same)
        self.assertNotEqual(ids, changed)
        self.assertNotEqual(ids[0], ids[1])


if __name__ == '__main__':
    unittest.main()
