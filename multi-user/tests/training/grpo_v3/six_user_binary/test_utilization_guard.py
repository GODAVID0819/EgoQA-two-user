"""真实启动低利用率必须提前保护，辅助显存不能挤占模型。"""
import unittest
from test_pipeline import require_module


class UtilizationGuardTests(unittest.TestCase):
    def test_short_drop_reacts_before_long_windows_decline(self):
        guard = require_module(self, 'utilization_guard')
        c = guard.Feedback(target=80, cancel=60, reserve_gib=8, max_prealloc_mib=16)
        for t in range(121):
            c.update(now=t, utilization=100, free_mib=90000)
        action = c.update(now=121, utilization=0, free_mib=90000)
        self.assertGreaterEqual(action['windows']['60'], 80)
        self.assertEqual(action['duty'], c.config['max_duty'])
        # 强响应仍受显存安全边界约束，不能挤占模型。
        self.assertEqual(c.update(now=122, utilization=0, free_mib=100)['duty'], 0)

    def test_low_startup_requires_protection_without_waiting_two_hours(self):
        guard = require_module(self, 'utilization_guard')
        c = guard.Feedback(target=80, reserve_gib=8, max_prealloc_mib=16)
        action = c.update(now=1, utilization=0, free_mib=90000)
        self.assertGreater(action['duty'], 0)
        self.assertTrue(action['memory_safe'])
        self.assertLessEqual(action['allocation_mib'], 16)

    def test_busy_work_yields_and_memory_pressure_releases(self):
        guard = require_module(self, 'utilization_guard')
        c = guard.Feedback(target=80, reserve_gib=8, max_prealloc_mib=16)
        for t in range(65):
            c.update(now=t, utilization=100, free_mib=90000)
        self.assertEqual(c.update(now=66, utilization=100, free_mib=90000)['duty'], 0)
        action = c.update(now=67, utilization=0, free_mib=100)
        self.assertFalse(action['memory_safe'])
        self.assertEqual(action['duty'], 0)

    def test_rolling_windows_cannot_hide_startup_deficit(self):
        guard = require_module(self, 'utilization_guard')
        c = guard.Feedback(target=80, reserve_gib=8, max_prealloc_mib=16)
        for t in range(600):
            c.update(now=t, utilization=0, free_mib=90000)
        for t in range(600, 660):
            c.update(now=t, utilization=100, free_mib=90000)
        result = c.update(now=661, utilization=100, free_mib=90000)
        self.assertEqual(result['windows']['60'], 100)
        self.assertLess(result['windows']['cumulative'], 20)
        self.assertGreater(result['duty'], 0)

    def test_bad_threshold_and_memory_configuration_rejected(self):
        guard = require_module(self, 'utilization_guard')
        for config in ({'target': 60, 'cancel': 60}, {'target': 101},
                       {'max_prealloc_mib': 0}, {'reserve_gib': -1}):
            with self.assertRaises(ValueError):
                guard.checked_config(config)

    def test_sampling_delay_is_weighted_by_time(self):
        guard = require_module(self, 'utilization_guard')
        c = guard.Feedback(target=80, reserve_gib=8, max_prealloc_mib=16)
        c.update(now=0, utilization=0, free_mib=90000)
        c.update(now=9, utilization=100, free_mib=90000)
        result = c.update(now=10, utilization=100, free_mib=90000)
        self.assertAlmostEqual(result['windows']['cumulative'], 10)

    def test_handoff_keeps_previous_startup_history(self):
        guard = require_module(self, 'utilization_guard')
        c = guard.Feedback(target=80, reserve_gib=8, max_prealloc_mib=16)
        c.seed([(0,0),(9,100),(10,100)])
        result = c.update(now=11, utilization=100, free_mib=90000)
        self.assertAlmostEqual(result['windows']['cumulative'], 200/11)

    def test_guard_configuration_reaches_runtime(self):
        workflow = require_module(self, 'workflow')
        config = {'project_root': '/scratch/p', 'model': '/scratch/model',
                  'train_python': '/scratch/train/bin/python', 'judge_python': '/scratch/judge/bin/python',
                  'utilization_guard': {'target': 80, 'cancel': 60, 'reserve_gib': 8, 'max_prealloc_mib': 16}}
        self.assertEqual(workflow.training_config(config, job_id='1', phase='formal', max_steps=200)
                         .get('utilization_guard'), config['utilization_guard'])

    def test_auxiliary_work_is_bounded_and_preserves_rng_on_cpu(self):
        try:
            import torch
        except ImportError:
            self.skipTest('本地没有Torch；在现有Torch环境验证真实CPU张量')
        runtime = require_module(self,'utilization_runtime')
        rng = torch.random.get_rng_state().clone()
        operands = runtime.auxiliary_operands(torch,'cpu')
        self.assertLessEqual(sum(t.numel()*t.element_size() for t in operands),2*1024**2)
        for _ in range(5): runtime.auxiliary_step(torch,operands)
        self.assertTrue(torch.equal(torch.random.get_rng_state(),rng))
        self.assertTrue(torch.equal(operands[0],torch.ones_like(operands[0])))
        self.assertTrue(torch.isfinite(operands[1]).all())


if __name__ == '__main__':
    unittest.main()
