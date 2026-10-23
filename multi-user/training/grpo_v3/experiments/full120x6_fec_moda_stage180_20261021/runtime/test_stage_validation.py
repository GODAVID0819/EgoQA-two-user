"""半轮验收必须只接受原清单前半，不能把数据集总数当已训练数。"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'multi-user'))
from training.grpo_v3.six_user_binary import validate_run
from training.grpo_v3.six_user_binary.stage_stop import StageStopCallback
from types import SimpleNamespace


class StageValidationTests(unittest.TestCase):
    def test_callback_saves_at_180_while_schedule_remains_360(self):
        callback = object.__new__(StageStopCallback); callback.stop_step = 180
        args = SimpleNamespace(max_steps=360, eval_strategy='no')
        control = SimpleNamespace(should_training_stop=False, should_save=False, should_evaluate=False)
        callback.on_step_end(args, SimpleNamespace(global_step=179), control)
        self.assertFalse(control.should_training_stop)
        callback.on_step_end(args, SimpleNamespace(global_step=180), control)
        self.assertTrue(control.should_training_stop); self.assertTrue(control.should_save)
        self.assertFalse(control.should_evaluate); self.assertEqual(args.max_steps, 360)

    def test_180_updates_cover_first_360_of_720_conditions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / 'swift/v0/checkpoint-180'
            checkpoint.mkdir(parents=True)
            ordered = json.loads((HERE / 'condition_definition.json').read_text(encoding='utf-8'))['train']
            planned = [{'source_packet_id': row['source_packet_id'], 'asker_index': row['speaker_index']}
                       for row in ordered]
            data = root / 'train.jsonl'; data.write_text(''.join(json.dumps(row) + '\n' for row in planned))
            config = dict(max_steps=360, stop_after_steps=180, num_generations=4, world_size=2,
                          train_conditions=720, validation_conditions=0, eval_strategy='no',
                          val_dataset=None, paired_validation=False, baseline_after_training=False,
                          train_dataset=str(data), learning_rate=1e-5)
            (root / 'run_config.json').write_text(json.dumps(config))
            (checkpoint / 'trainer_state.json').write_text(json.dumps(dict(global_step=180, max_steps=360,
                log_history=[{'step': i, 'loss': .1, 'grad_norm': .2} for i in range(1, 181)])))
            for name in ['adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt',
                         'rng_state_0.pth', 'rng_state_1.pth', 'args.json', 'training_args.bin']:
                (checkpoint / name).write_bytes(b'fixture')
            records = []
            for rank in range(2):
                rank_dir = root / f'rank{rank}'; rank_dir.mkdir()
                (rank_dir / 'ddp_sync.jsonl').write_text(json.dumps(dict(global_step=180,
                    all_trainable_parameters_equal=True, finite=True, checked_tensors=160)) + '\n')
                (rank_dir / 'gradient_sync.jsonl').write_text(''.join(json.dumps(dict(global_step=i,
                    finite=True, world_size=2, tensors=160, order='mean_then_clip_then_optimizer')) + '\n'
                    for i in range(180)))
                for update in range(180):
                    condition = planned[update * 2 + rank]
                    for slot, distance in enumerate([.5, .4, .2, 0.]):
                        records.append(dict(condition, request_id=f'{rank}-{update}:{slot}',
                            global_step=update+1,
                            evidence_id=f'e{rank}-{update}', reward=1 + .5 * distance,
                            quality_reward=1., bonus_gate=True, diversity_distance=distance,
                            diversity_bonus=.5 * distance, reference_count=3,
                            judge_result=dict(status='scored', probabilities=dict(formality=1.,
                                groundedness=1., combined_answerability=1.))))
                (rank_dir/'reward_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records[rank*720:(rank+1)*720]))
            trace = root / 'reward_trace.jsonl'
            trace.write_text(''.join(json.dumps(row) + '\n' for row in records))
            result = validate_run.validate(root, adapter_check=lambda path: True)
            self.assertEqual(result['status'], 'passed', result)
            self.assertEqual(result['unique_conditions'], 360)
            self.assertEqual(result['candidate_count'], 1440)
            wrong_order=records[4:8]+records[:4]+records[8:720]
            (root/'rank0/reward_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in wrong_order))
            result=validate_run.validate(root,adapter_check=lambda path:True)
            self.assertIn('rank0_condition_order',result['failed_checks'])
            (root/'rank0/reward_trace.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records[:720]))
            # 用后半条件替换首条，即便数量相同也必须拒绝。
            records[0].update(planned[360])
            trace.write_text(''.join(json.dumps(row) + '\n' for row in records))
            result = validate_run.validate(root, adapter_check=lambda path: True)
            self.assertIn('actual_conditions_match_training_manifest', result['failed_checks'])


if __name__ == '__main__': unittest.main()
