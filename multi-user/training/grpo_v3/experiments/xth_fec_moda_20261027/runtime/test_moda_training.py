"""720条件覆盖、不同窗口配对和无验证CLI。"""
from pathlib import Path
from collections import Counter
import sys
import unittest

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'multi-user')]
import dual_runtime
from training.grpo_v3.six_user_binary import launch
from training.grpo_v3.six_user_binary.stage_stop import StageStopCallback
from types import SimpleNamespace


def config():
    paths={key:'/scratch/fixture/'+key for key in ['project_root','train_python','judge_python','policy_model',
           'judge_config','train_dataset','output_root','scratch_root']}
    return {**paths,'val_dataset':None,'world_size':2,'num_generations':4,'num_generations_eval':4,
            'per_device_train_batch_size':1,'gradient_accumulation_steps':4,'max_steps':360,'stop_after_steps':360,
            'lora_rank':64,'lora_alpha':128,'train_conditions':720,'validation_conditions':0,
            'policy_gpu_indices':[0],'judge_gpu_indices':[0],'shared_gpu':True,'use_vllm':True,
            'walltime_seconds':230400,'max_length':65536,'max_completion_length':2048,'max_pixels':24576,
            'judge_port':12345,'learning_rate':1e-5,'temperature':.85,'top_p':.95,'top_k':40,'beta':.04,
            'reward_mode':'continuous','eval_strategy':'no','paired_validation':False,'baseline_after_training':False,
            'lora_target_modules':['q_proj','v_proj','in_proj_qkv'],'lora_dropout':.05,
            'lr_scheduler_type':'cosine_with_min_lr','warmup_steps':10,'lr_scheduler_kwargs':{'min_lr_rate':.1}}


class TrainingTests(unittest.TestCase):
    def test_terminal_save_does_not_request_removed_validation(self):
        callback=object.__new__(StageStopCallback);callback.stop_step=360
        for strategy in ['no',SimpleNamespace(value='no')]:
            args=SimpleNamespace(max_steps=360,eval_strategy=strategy)
            control=SimpleNamespace(should_training_stop=False,should_save=False,should_evaluate=False)
            callback.on_step_end(args,SimpleNamespace(global_step=360),control)
            self.assertTrue(control.should_training_stop)
            self.assertTrue(control.should_save)
            self.assertFalse(control.should_evaluate)

    def test_approved_360_720_no_validation_config_is_accepted(self):
        try: dual_runtime.validate_config(config())
        except ValueError as exc:self.fail('旧300/600/120约束尚未退出：'+str(exc))

    def test_effective_CLI_has_no_validation_and_fresh_initialization(self):
        try: command=launch.swift_command(config(),'/scratch/fixture/output')
        except ValueError as exc:self.fail('无验证配置仍被旧入口拒绝：'+str(exc))
        self.assertNotIn('--val_dataset',command)
        self.assertNotIn('--resume_from_checkpoint',command)
        self.assertEqual(command[command.index('--eval_strategy')+1],'no')
        self.assertEqual(command[command.index('--max_steps')+1],'360')
        self.assertEqual(command[command.index('--reward_funcs')+1],'egoqa_fec_moda_v1')
        self.assertEqual(command[command.index('--lora_dropout')+1],'0.05')

    def test_different_packet_aliases_of_same_window_cannot_pair(self):
        rows=[]
        for w in [1,2]:
            for speaker in [0,1]:
                rows.append({'source_packet_id':f'RLHF6U_DAY{w}_11110000_ALIAS{speaker}',
                             'asker_index':speaker,'cost':w*10+speaker})
        pairs=dual_runtime.make_length_pairs(rows,lambda row:row['cost'])
        self.assertEqual(Counter(dual_runtime.condition_binding(r) for r in pairs),
                         Counter(dual_runtime.condition_binding(r) for r in rows))
        self.assertTrue(all(dual_runtime.condition_binding(pairs[i])[0]!=dual_runtime.condition_binding(pairs[i+1])[0]
                            for i in range(0,len(pairs),2)))


if __name__=='__main__':unittest.main()
