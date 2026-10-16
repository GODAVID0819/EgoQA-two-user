"""用已安装TRL/Accelerate的真实采样器核查：2卡和4卡下rank r在第k步得到文件第step_row(k,j,r)行。"""
import unittest


class SamplerWorldTests(unittest.TestCase):
    def check(self,world):
        from torch.utils.data import BatchSampler
        from accelerate.data_loader import BatchSamplerShard
        from trl.trainer.utils import RepeatSampler
        from dual_runtime import rounds_per_step, step_row
        rows=list(range(600));g=4;steps_per_generation=4;rounds=rounds_per_step(world)
        # 与TRL GRPOTrainer._get_train_sampler一致：每次生成world个提示×G，重复steps_per_generation次。
        sampler=RepeatSampler(rows,mini_repeat_count=g,batch_size=world*g//g,repeat_count=steps_per_generation,shuffle=False,seed=42)
        batches=BatchSampler(sampler,batch_size=steps_per_generation,drop_last=False)
        for rank in range(world):
            local=list(BatchSamplerShard(batches,num_processes=world,process_index=rank,split_batches=False,even_batches=True))
            self.assertEqual(len(local),600//world*steps_per_generation)
            for i in range(0,len(local),steps_per_generation):
                generation=i//steps_per_generation;step,j=divmod(generation,rounds)
                expected=step_row(step,j,rank,world)
                for b in local[i:i+steps_per_generation]:self.assertEqual(b,[expected]*g)

    def test_two_ranks(self):self.check(2)

    def test_four_ranks(self):self.check(4)


if __name__=='__main__':unittest.main()
