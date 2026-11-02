"""使用已安装的真实 TRL/Accelerate 核验四窗批次，缺依赖时明确退出。"""
import argparse
from collections import Counter
import json
from pathlib import Path


def audit(rows):
    from torch.utils.data import BatchSampler
    from accelerate.data_loader import BatchSamplerShard
    from trl.trainer.utils import RepeatSampler
    from trl import GRPOConfig
    args = GRPOConfig(output_dir='sampler_audit_cpu', use_cpu=True, bf16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        num_generations=4, steps_per_generation=8, max_steps=180,
        shuffle_dataset=False, report_to='none', scale_rewards='group', loss_type='grpo')
    if args.generation_batch_size != 8 or args.steps_per_generation != 8:
        raise ValueError('实际单进程CPU参数解析不符')
    ranks = []
    for rank in range(2):
        sampler = RepeatSampler(rows, mini_repeat_count=4, batch_size=4,
                                repeat_count=8, shuffle=False, seed=42)
        batches = BatchSampler(sampler,batch_size=8,drop_last=False)
        shards = list(BatchSamplerShard(batches,num_processes=2,process_index=rank,
                                       split_batches=False,even_batches=True))
        if len(shards) != 1440:
            raise ValueError('真实采样器微批数不符')
        visits = []
        for step in range(180):
            expected = [4*step+rank*2]*4 + [4*step+rank*2+1]*4
            actual = shards[step*8:step*8+8]
            if actual != [expected]*8:
                raise ValueError('真实分片与每卡两组G4不一致')
            visits.extend(expected[::4])
        ranks.append(visits)
    if Counter(ranks[0]+ranks[1]) != Counter(range(720)):
        raise ValueError('原720条件未逐项覆盖')
    return dict(status='passed',backend='actual_TRL_Accelerate_CPU',updates=180,candidate_slots=2880,
                gpu_world_size_not_tested=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        rows=[json.loads(line) for line in a.dataset.read_text().splitlines() if line.strip()]
        if len(rows)!=720:
            raise ValueError('实际输入必须有720行')
        result=audit(rows)
    except ImportError as exc:
        result=dict(status='pending_dependency',error=str(exc))
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))
    raise SystemExit(0 if result['status']=='passed' else 2)

if __name__=='__main__':
    main()
