"""每步4个不同窗口的长度配组：每条件恰一次、两rank长度均衡、文件顺序即采样顺序。"""
import random
import unittest


class QuadOrderTests(unittest.TestCase):
    def test_quads_cover_each_condition_once_with_four_windows(self):
        from dual_runtime import make_length_quads, check_step_order
        rows=[{'source_packet_id':f'P{w}','asker_index':a} for w in range(120) for a in range(5)]
        rng=random.Random(1);cost={(r['source_packet_id'],r['asker_index']):rng.randint(1000,9000) for r in rows}
        c=lambda r:cost[(r['source_packet_id'],r['asker_index'])]
        ordered=make_length_quads(rows,c,seed=42)
        self.assertEqual(len({(r['source_packet_id'],r['asker_index']) for r in ordered}),600)
        check_step_order(ordered)
        self.assertLess(abs(sum(c(r) for r in ordered[0::2])-sum(c(r) for r in ordered[1::2])),9000)
        self.assertEqual(ordered,make_length_quads(rows,c,seed=42))

    def test_repeated_window_in_a_step_is_rejected(self):
        from dual_runtime import check_step_order
        rows=[{'source_packet_id':p} for p in ('A','B','C','A')]
        with self.assertRaises(ValueError):check_step_order(rows)

    def test_config_gate_is_four_windows_r128_900_steps(self):
        from dual_runtime import validate_config
        c={'world_size':2,'num_generations':4,'per_device_train_batch_size':1,'gradient_accumulation_steps':8,
           'max_steps':900,'lora_rank':128,'lora_alpha':128,'train_conditions':600,'validation_conditions':120,
           'stop_after_steps':2}
        validate_config(c)
        for change in ({'gradient_accumulation_steps':4},{'lora_rank':64},{'max_steps':300},{'stop_after_steps':901}):
            with self.subTest(change=change),self.assertRaises(ValueError):validate_config({**c,**change})


if __name__=='__main__':unittest.main()
