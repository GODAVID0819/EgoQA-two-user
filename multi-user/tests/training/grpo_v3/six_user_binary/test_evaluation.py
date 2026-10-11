"""固定验证对比必须完整配对，负结果也正常保存。"""
import unittest
import json
from pathlib import Path
import tempfile
from test_pipeline import require_module


class EvaluationTests(unittest.TestCase):
    def test_partial_qkv_evaluation_uses_isolated_patched_worker(self):
        module=require_module(self,'evaluation')
        environ={'VLLM_ENABLE_V1_MULTIPROCESSING':'1','OTHER':'keep'}
        module.configure_engine_process({'lora_target_modules':['q_proj','v_proj','in_proj_qkv']},environ)
        self.assertEqual(environ,{'VLLM_ENABLE_V1_MULTIPROCESSING':'1','OTHER':'keep'})
        options=module.configure_engine_kwargs({'lora_target_modules':['q_proj','v_proj','in_proj_qkv']},
                                               {'enable_chunked_prefill':True})
        self.assertTrue(options['enable_chunked_prefill'])
        self.assertEqual(options['worker_extension_cls'],
            'training.grpo_v3.six_user_binary.packed_worker_extension.PartialPackedLoRAWorkerExtension')

    def test_original_qv_evaluation_preserves_process_setting(self):
        module=require_module(self,'evaluation')
        environ={'VLLM_ENABLE_V1_MULTIPROCESSING':'1'}
        module.configure_engine_process({'lora_target_modules':['q_proj','v_proj']},environ)
        self.assertEqual(environ,{'VLLM_ENABLE_V1_MULTIPROCESSING':'1'})
        self.assertEqual(module.configure_engine_kwargs({'lora_target_modules':['q_proj','v_proj']},
                                                       {'enable_chunked_prefill':True}),
                         {'enable_chunked_prefill':True})

    def test_saved_adapter_validation_does_not_require_training_result(self):
        launch=require_module(self,'launch')
        calls=[]
        c={'validation_only_adapter':'/scratch/checkpoint-3','paired_validation':True,
           'baseline_validation_source':'/scratch/validation_baseline.json'}
        self.assertTrue(launch.run_validation_only(c,lambda adapter:calls.append(adapter)))
        self.assertEqual(calls,['/scratch/checkpoint-3'])
        self.assertFalse(launch.run_validation_only({},lambda adapter:self.fail('不应触发评分')))

    def test_validation_only_requires_existing_paired_baseline(self):
        launch=require_module(self,'launch')
        for c in ({'validation_only_adapter':'/scratch/checkpoint-3'},
                  {'validation_only_adapter':'/scratch/checkpoint-3','paired_validation':True}):
            with self.assertRaises(ValueError):launch.run_validation_only(c,lambda adapter:self.fail('不应评分'))

    def test_workflow_forwards_validation_only_adapter(self):
        workflow=require_module(self,'workflow')
        c={'project_root':'/scratch/project','train_python':'/scratch/train/bin/python',
           'judge_python':'/scratch/judge/bin/python','model':'/scratch/model',
           'validation_only_adapter':'/scratch/source/checkpoint-3'}
        run=workflow.training_config(c,job_id='1',phase='formal',max_steps=200)
        self.assertEqual(run.get('validation_only_adapter'),c['validation_only_adapter'])

    def test_deferred_baseline_uses_base_model_then_final_adapter(self):
        launch=require_module(self,'launch')
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)
            (output/'training_result.json').write_text(json.dumps({'checkpoint':'/scratch/checkpoint-60'}))
            calls=[]
            launch.final_validation({'paired_validation':True,'baseline_after_training':True},output,
                                    lambda adapter=None:calls.append(adapter))
            self.assertEqual(calls,[None,'/scratch/checkpoint-60'])

    def test_frozen_judge_can_be_restarted_without_changing_comparison_contract(self):
        module=require_module(self,'evaluation')
        old={'status':'ready','frozen':True,'model_id':'/scratch/Qwen3.8-27B',
             'judge_mode':'baseline','backend':'vllm_raw_verdict_logprobs','instance_id':'old'}
        new={**old,'instance_id':'new'}
        self.assertTrue(module.same_frozen_judge(old,new))
        self.assertFalse(module.same_frozen_judge(old,{**new,'model_id':'/scratch/other'}))
        self.assertFalse(module.same_frozen_judge(old,{**new,'frozen':False}))

    def test_comparison_reports_negative_delta_without_claiming_success(self):
        module=require_module(self,'evaluation')
        baseline=[{'evidence_id':'E','slot':i,'seed':42,'reward':.8,'status':'scored','probabilities':{'formality':.8}} for i in range(4)]
        policy=[{**r,'reward':.6} for r in baseline]
        report=module.compare_rows(baseline,policy)
        self.assertAlmostEqual(report['reward_delta'],-.2)
        self.assertFalse(report['improved'])
        self.assertEqual(report['paired_candidate_count'],4)
        self.assertEqual(report['paired_input_count'],1)

    def test_missing_duplicate_or_wrong_seed_is_not_silently_averaged(self):
        module=require_module(self,'evaluation')
        rows=[{'evidence_id':'E','slot':i,'seed':42,'reward':.4,'status':'scored','probabilities':{}} for i in range(4)]
        for other in (rows[:3],rows+[rows[0]],[{**r,'seed':43} for r in rows]):
            with self.assertRaises(ValueError): module.compare_rows(rows,other)


if __name__=='__main__':unittest.main()
