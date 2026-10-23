"""阶段停止与恢复的结构回归；真实Adam/RNG重载另由Torch CPU作业验证。"""
import json
from pathlib import Path
import tempfile
import unittest

from recovery import validated_resume, expected_resume_bindings


class ResumeTests(unittest.TestCase):
    def fixture(self, root):
        source = root / 'train_999'
        checkpoint = source / 'swift' / 'v0' / 'checkpoint-180'
        checkpoint.mkdir(parents=True)
        for name in ['adapter_model.safetensors', 'optimizer.pt', 'scheduler.pt',
                     'rng_state_0.pth', 'rng_state_1.pth', 'args.json', 'training_args.bin']:
            (checkpoint / name).write_bytes(b'fixture')
        (checkpoint / 'trainer_state.json').write_text(json.dumps({'global_step': 180, 'max_steps': 360}))
        (checkpoint / 'adapter_config.json').write_text(json.dumps({'r': 64, 'lora_alpha': 128}))
        judge = root / 'judge.json'; judge.write_text('{}')
        data = root / 'data.jsonl'
        data.write_text(''.join(json.dumps({'source_packet_id': f'window{i}', 'asker_index': i % 6}) + '\n'
                                for i in range(720)))
        old = dict(project_root=str(root), output_root=str(root / 'outputs'), judge_config=str(judge),
                   train_dataset=str(data), max_steps=360, stop_after_steps=180,
                   max_pixels=50176, max_length=131072, lora_rank=64, lora_alpha=128,
                   walltime_seconds=216000, resume_from_checkpoint=None)
        (source / 'run_config.json').write_text(json.dumps(old))
        audit = root / 'reload.json'
        audit.write_text(json.dumps({'status': 'passed', 'global_step': 180, 'checkpoint': str(checkpoint),
                                     'scheduler': {'_last_lr': [5.701891736297396e-6]}, 'optimizer_states': 160}))
        return dict(old, stop_after_steps=360, resume_from_checkpoint=str(checkpoint),
                    resume_source_output=str(source), expected_resume_step=180,
                    recovery_from_job='999', resume_cpu_audit=str(audit))

    def test_half_epoch_can_continue_without_changing_schedule(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory))
            try:
                actual = validated_resume(config)
            except ValueError as error:
                self.fail('合法半轮恢复被拒绝：' + str(error))
            self.assertEqual(actual['step'], 180)
            self.assertEqual(expected_resume_bindings(config), [('window360', 0), ('window361', 1)])

    def test_resume_rejects_pixel_change(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory)); config['max_pixels'] = 24576
            with self.assertRaises(ValueError): validated_resume(config)

    def test_resume_stage_end_cannot_precede_restored_step(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.fixture(Path(directory)); config['stop_after_steps'] = 100
            with self.assertRaises(ValueError): validated_resume(config)


if __name__ == '__main__': unittest.main()
