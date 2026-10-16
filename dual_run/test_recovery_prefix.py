"""恢复时拒绝错误步数/缺失状态/改变训练条件，保留真实历史评分前缀。"""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace as N


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        from recovery import validated_resume, seed_reward_prefix, check_restored_optimizer
        self.validate=validated_resume;self.seed=seed_reward_prefix
        self.check_optimizer=check_restored_optimizer
        parent=Path(__file__).parent/'cpu_test_tmp';parent.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(dir=parent);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.source=self.root/'train_123'
        self.cp=self.source/'swift/v0/checkpoint-12';self.cp.mkdir(parents=True)
        self.data=self.root/'train.jsonl';self.rows=[{'source_packet_id':'window_'+str(i),'asker_index':i%6} for i in range(600)]
        self.data.write_text(''.join(json.dumps(r)+'\n' for r in self.rows))
        judge=self.root/'judge.json';judge.write_text('{}')
        base={'max_steps':300,'stop_after_steps':300,'num_generations':4,'lora_rank':64,'lora_alpha':128,
              'world_size':2,'train_conditions':600,'learning_rate':1e-5,'train_dataset':str(self.data),'judge_config':str(judge),
              'resume_from_checkpoint':None}
        (self.source/'run_config.json').write_text(json.dumps(base))
        self.c={**base,'resume_from_checkpoint':str(self.cp),'resume_source_output':str(self.source),
                'recovery_from_job':'123','expected_resume_step':12,'initial_fresh_base':True}
        for name in ['adapter_model.safetensors','optimizer.pt','scheduler.pt','rng_state_0.pth','rng_state_1.pth','args.json','training_args.bin']:
            (self.cp/name).write_bytes(b'x')
        (self.cp/'adapter_config.json').write_text(json.dumps({'r':64,'lora_alpha':128}))
        (self.cp/'trainer_state.json').write_text(json.dumps({'global_step':12,'max_steps':300}))
        audit=self.root/'resume_audit.json';audit.write_text(json.dumps({'status':'passed','global_step':12,
            'checkpoint':str(self.cp),'optimizer_states':160,'scheduler':{'_last_lr':[9.99e-6]}}))
        self.c['resume_cpu_audit']=str(audit)
        for rank in [0,1]:
            folder=self.source/('rank'+str(rank));folder.mkdir()
            rows=[]
            # 每步4行：第一轮(4s, 4s+1)、第二轮(4s+2, 4s+3)，rank r取每轮第r行。
            for step in range(13):
                for round_index in range(2):
                    row=self.rows[4*step+2*round_index+rank]
                    rows.extend({**row,'request_id':f'r{rank}s{step}g{round_index}:{i}','reward':.5} for i in range(4))
            (folder/'reward_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_valid_resume_keeps_original_horizon(self):
        value=self.validate(self.c)
        self.assertEqual(value['step'],12)
        self.assertEqual(self.c['max_steps'],300)

    def test_missing_rank_rng_is_rejected(self):
        (self.cp/'rng_state_1.pth').unlink()
        with self.assertRaises(ValueError):self.validate(self.c)

    def test_changed_learning_rate_is_rejected(self):
        with self.assertRaises(ValueError):self.validate({**self.c,'learning_rate':5e-6})

    def test_prefix_uses_only_checkpointed_conditions_and_preserves_ids(self):
        dest=self.root/'new_output';(dest/'rank0').mkdir(parents=True);(dest/'rank1').mkdir()
        report=self.seed(self.c,dest)
        self.assertEqual(report['inherited_candidates'],192)
        for rank in [0,1]:
            rows=[json.loads(x) for x in (dest/('rank'+str(rank))/'reward_trace.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),96)
            self.assertEqual(rows[-1]['source_packet_id'],self.rows[4*11+2+rank]['source_packet_id'])
        with self.assertRaises(FileExistsError):self.seed(self.c,dest)

    def test_resume_bindings_are_first_round_of_the_resumed_step(self):
        from recovery import expected_resume_bindings
        self.assertEqual(expected_resume_bindings(self.c),
                         [(self.rows[48+r]['source_packet_id'],self.rows[48+r]['asker_index']) for r in (0,1)])

    def test_restored_optimizer_step_and_lr_must_be_continuous(self):
        meta=self.validate(self.c)
        trainer=N(state=N(global_step=12),args=N(max_steps=300),lr_scheduler=N(last_epoch=12),
                  optimizer=N(param_groups=[{'lr':9.99e-6}],state={i:{'step':12.} for i in range(160)}))
        self.check_optimizer(trainer,meta)
        trainer.optimizer.param_groups[0]['lr']=0.
        with self.assertRaises(RuntimeError):self.check_optimizer(trainer,meta)


if __name__=='__main__':unittest.main()
