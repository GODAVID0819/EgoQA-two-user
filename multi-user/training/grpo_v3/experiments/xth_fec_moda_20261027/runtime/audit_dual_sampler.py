"""用已安装TRL/Accelerate的真实纯CPU采样器核查双卡360步，不启动分布式或模型。"""
import argparse
from collections import Counter
from importlib.metadata import version
import inspect
import json
import os
from pathlib import Path


def audit(definition_path, output_dir):
    print('CPU_AUDIT_STAGE start', flush=True)
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('此CPU审计只允许单进程环境，不启动分布式初始化')
    from torch.utils.data import BatchSampler
    print('CPU_AUDIT_STAGE torch_imported', flush=True)
    from accelerate.data_loader import BatchSamplerShard
    print('CPU_AUDIT_STAGE accelerate_imported', flush=True)
    from trl.trainer.utils import RepeatSampler
    from trl import GRPOConfig
    print('CPU_AUDIT_STAGE trl_imported', flush=True)

    definition = json.loads(definition_path.read_text(encoding='utf-8'))
    ordered = definition['train']
    keys = [(row['source_packet_id'], int(row['speaker_index'])) for row in ordered]
    assert len(ordered) == 720 and len(set(keys)) == 720, '训练条件应恰为720个唯一条件'
    assert all(int(row['input_view_count']) == 6 for row in ordered), '六视角不得改变'
    assert all(float(row['duration_seconds']) == 600 for row in ordered), '十分钟窗口不得改变'
    assert set(Counter(key[1] for key in keys).values()) == {120}, '六speaker各120训练条件'
    assert len(set(key[1] for key in keys)) == 6

    # TRL两字段互斥，只显式传steps_per_generation；CPU世界大小如实为1。
    args = GRPOConfig(output_dir=str(output_dir), use_cpu=True, bf16=False,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        num_generations=4, steps_per_generation=4, max_steps=360,
        shuffle_dataset=False, report_to='none', scale_rewards='group', loss_type='grpo')
    assert args.world_size == 1 and args.generation_batch_size == 4
    assert args.steps_per_generation == 4 and args.gradient_accumulation_steps == 4

    def actual_rank_batches(rank):
        sampler = RepeatSampler(ordered, mini_repeat_count=4, batch_size=2,
                                repeat_count=4, shuffle=False, seed=42)
        batch_sampler = BatchSampler(sampler, batch_size=4, drop_last=False)
        sharded = BatchSamplerShard(batch_sampler, num_processes=2,
            process_index=rank, split_batches=False, even_batches=True)
        return list(sampler), list(sharded)

    indexes, rank0 = actual_rank_batches(0)
    indexes1, rank1 = actual_rank_batches(1)
    assert indexes == indexes1 and len(indexes) == 11520
    ranks = [rank0, rank1]
    for batches in ranks:
        assert len(batches) == 1440
        assert all(len(batch) == 4 and len(set(batch)) == 1 for batch in batches), '每rank必须一完整条件G4'
        for start in range(0, len(batches), 4):
            assert batches[start:start+4] == [batches[start]] * 4, '缓存G4按4个微批训练，不重新访问新条件'
    updates = [batches[::4] for batches in ranks]
    assert all(len(batches) == 360 for batches in updates)
    rank_visits = [[batch[0] for batch in batches] for batches in updates]
    assert all(len(set(visits)) == 360 for visits in rank_visits)
    visits = Counter(rank_visits[0] + rank_visits[1])
    assert visits == Counter(range(720)), '全局720条件每个恰访问一次'
    paired = []
    for step, (left, right) in enumerate(zip(*rank_visits), 1):
        assert keys[left][0] != keys[right][0], '每更新必须两个不同窗口'
        paired.append({'optimizer_step': step,
            'rank0_index': left, 'rank1_index': right,
            'rank0_condition': list(keys[left]), 'rank1_condition': list(keys[right])})

    return {'status': 'passed', 'audit_scope': '真实CPU采样器及单进程parser；不是双卡GPU验收',
        'definition_path': str(definition_path.resolve()),
        'versions': {name: version(name) for name in ['torch', 'trl', 'accelerate', 'transformers']},
        'implementation_files': {name: inspect.getfile(value) for name, value in {
            'RepeatSampler': RepeatSampler, 'BatchSamplerShard': BatchSamplerShard,
            'GRPOConfig': GRPOConfig}.items()},
        'cpu_parser': {'world_size': args.world_size,
            'per_device_train_batch_size': args.per_device_train_batch_size,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'num_generations': args.num_generations,
            'steps_per_generation': args.steps_per_generation,
            'generation_batch_size': args.generation_batch_size,
            'max_steps': args.max_steps, 'shuffle_dataset': args.shuffle_dataset},
        'required_gpu_runtime_fields_not_yet_tested': {
            'world_size': 2, 'generation_batch_size': 8, 'steps_per_generation': 4},
        'sampler': {'mini_repeat_count': 4, 'batch_size': 2, 'repeat_count': 4,
            'shuffle': False, 'seed': 42, 'index_count': len(indexes)},
        'shard': {'num_processes': 2, 'split_batches': False, 'batch_size': 4,
            'local_batches_per_rank': [len(x) for x in ranks],
            'optimizer_updates_per_rank': 360, 'unique_conditions_per_rank': 360,
            'global_unique_conditions': 720, 'visits_per_condition': 1,
            'global_training_candidates': 2880, 'same_window_pairs': 0},
        'actual_sampler_indexes': indexes,
        'actual_rank_local_batches': ranks,
        'optimizer_update_pairs': paired,
        'gpu_forward_or_ddp_or_judge_tested': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--definition', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = audit(args.definition, args.output.parent / 'cpu_parser_args')
    except BaseException as exc:
        result = {'status': 'failed', 'error_type': type(exc).__name__, 'error': str(exc),
                  'gpu_forward_or_ddp_or_judge_tested': False}
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('DUAL_SAMPLER_AUDIT_PASSED', json.dumps({
        'cpu_parser': result['cpu_parser'], 'shard': result['shard'],
        'gpu_forward_or_ddp_or_judge_tested': False}, ensure_ascii=True), flush=True)


if __name__ == '__main__':
    main()
