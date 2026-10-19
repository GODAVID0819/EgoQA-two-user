import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from resume_audit import audit, expected_lr

CONFIG = {'max_steps': 900, 'warmup_steps': 10, 'learning_rate': 1e-5, 'lr_scheduler_type': 'cosine_with_min_lr',
          'lr_scheduler_kwargs': {'min_lr_rate': 0.1}, 'world_size': 4}


class ResumeAuditTest(unittest.TestCase):
    def make(self, root, step=5, lr=None, nan=False):
        cp = Path(root) / 'checkpoint'
        cp.mkdir()
        p = [torch.nn.Parameter(torch.ones(2, 2)), torch.nn.Parameter(torch.ones(2, 2))]
        opt = torch.optim.AdamW(p, lr=CONFIG['learning_rate'])
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: expected_lr(CONFIG, s) / CONFIG['learning_rate'])
        for _ in range(step):
            sum(x.sum() for x in p).backward()
            opt.step()
            sched.step()
        if nan:
            next(iter(opt.state.values()))['exp_avg'][0, 0] = float('nan')
        state = sched.state_dict()
        if lr is not None:
            state['_last_lr'] = [lr]
        save_file({'a.lora_A.weight': torch.ones(2, 2), 'a.lora_B.weight': torch.ones(2, 2)}, str(cp / 'adapter_model.safetensors'))
        torch.save(opt.state_dict(), cp / 'optimizer.pt')
        torch.save(state, cp / 'scheduler.pt')
        for r in range(CONFIG['world_size']):
            torch.save({'python': 0}, cp / f'rng_state_{r}.pth')
        (cp / 'trainer_state.json').write_text(json.dumps({'global_step': step, 'max_steps': 900}))
        return cp

    def test_passes_and_lr_matches_schedule(self):
        with tempfile.TemporaryDirectory() as d:
            result = audit(self.make(d, step=12), CONFIG)
            self.assertEqual((result['status'], result['global_step'], result['optimizer_states']), ('passed', 12, 2))
            self.assertAlmostEqual(result['scheduler']['_last_lr'][0], expected_lr(CONFIG, 12), delta=1e-15)

    def test_rejects_wrong_lr_and_nan(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                audit(self.make(d, lr=7e-6), CONFIG)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                audit(self.make(d, nan=True), CONFIG)

    def test_schedule_endpoints(self):
        self.assertEqual(expected_lr(CONFIG, 0), 0)
        self.assertAlmostEqual(expected_lr(CONFIG, 10), 1e-5)
        self.assertAlmostEqual(expected_lr(CONFIG, 900), 1e-6)


if __name__ == '__main__':
    unittest.main()
