"""终态验收允许新总奖励上界且不等待已取消的验证集。"""
from pathlib import Path
import json
import sys
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'multi-user'))
from training.grpo_v3.six_user_binary import validate_run


class ValidationTests(unittest.TestCase):
    def test_completed_training_without_eval_and_reward_above_one(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);checkpoint=root/'swift/v0/checkpoint-1';checkpoint.mkdir(parents=True)
            config={'max_steps':1,'stop_after_steps':1,'num_generations':4,'num_generations_eval':4,
                    'world_size':2,'train_conditions':2,'validation_conditions':0,'eval_strategy':'no',
                    'paired_validation':False,'baseline_after_training':False,'learning_rate':1e-5}
            planned=[{'source_packet_id':f'RLHF6U_DAY{rank+1}_11110000_A1_JAKE','asker_index':0} for rank in range(2)]
            planned_path=root/'planned_train.jsonl'
            planned_path.write_text(''.join(json.dumps(r)+'\n' for r in planned))
            config['train_dataset']=str(planned_path)
            (root/'run_config.json').write_text(json.dumps(config))
            (checkpoint/'trainer_state.json').write_text(json.dumps({'global_step':1,'max_steps':1,
                'log_history':[{'loss':.1,'grad_norm':.2,'learning_rate':1e-6}]}))
            for name in ['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt','rng_state_0.pth',
                         'rng_state_1.pth','args.json','training_args.bin']:
                (checkpoint/name).write_bytes(b'fixture')
            rows=[]
            for rank in range(2):
                p=root/('rank'+str(rank));p.mkdir()
                (p/'ddp_sync.jsonl').write_text(json.dumps({'global_step':1,'all_trainable_parameters_equal':True,'finite':True,'checked_tensors':160})+'\n')
                (p/'gradient_sync.jsonl').write_text(json.dumps({'global_step':0,'finite':True,'world_size':2,'tensors':160,'order':'mean_then_clip_then_optimizer'})+'\n')
                for slot,distance in enumerate([.5,.4,.2,0.]):
                    rows.append({'request_id':f'r{rank}:{slot}','score_batch_id':f'r{rank}',
                         'evidence_id':f'e{rank}','source_packet_id':f'RLHF6U_DAY{rank+1}_11110000_A1_JAKE',
                         'asker_index':0,'global_step':1,'reward':1.+.5*distance,'quality_reward':1.,
                         'bonus_gate':True,'diversity_distance':distance,'diversity_bonus':.5*distance,
                         'reference_count':3,'judge_result':{'status':'scored','probabilities':{
                             'formality':1.,'groundedness':1.,'combined_answerability':1.}}})
                # 后续阶段验收还要求逐rank轨迹与合并记录一致。
                (p/'reward_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows[rank*4:(rank+1)*4]))
            (root/'reward_trace.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            result=validate_run.validate(root,adapter_check=lambda path:True)
            self.assertEqual(result['status'],'passed',result)
            self.assertEqual(result['candidate_count'],8)
            self.assertEqual(result['auto_pass_count'],8)
            self.assertEqual(result['auto_pass_denominator'],8)
            self.assertEqual(result['validation_conditions'],0)
            planned[0]['source_packet_id']='RLHF6U_DAY3_11110000_A1_JAKE'
            planned_path.write_text(''.join(json.dumps(r)+'\n' for r in planned))
            wrong_coverage=validate_run.validate(root,adapter_check=lambda path:True)
            self.assertEqual(wrong_coverage['status'],'failed')
            self.assertIn('actual_conditions_match_training_manifest',wrong_coverage['failed_checks'])


if __name__=='__main__':unittest.main()
