"""固定验证对比必须完整配对，负结果也正常保存。"""
import unittest
import json
from pathlib import Path
import tempfile
from test_pipeline import require_module


class EvaluationTests(unittest.TestCase):
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
