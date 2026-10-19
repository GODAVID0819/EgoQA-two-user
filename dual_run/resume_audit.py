"""断点续跑前在CPU上核验完整断点，产出 recovery.validated_resume 需要的 resume_cpu_audit。

核验：adapter全有限；优化器状态数=LoRA参数数，每个状态step=global_step且动量有限；
scheduler last_epoch=global_step，学习率与原配置的cosine_with_min_lr在该步的值一致；各rank随机状态可加载。
"""
import argparse
import json
import math
from pathlib import Path


def expected_lr(config, step):
    from transformers.optimization import _get_cosine_schedule_with_warmup_lr_lambda
    total = config['max_steps']
    if config['lr_scheduler_type'] != 'cosine_with_min_lr':
        raise ValueError('仅支持cosine_with_min_lr')
    factor = _get_cosine_schedule_with_warmup_lr_lambda(
        step, num_warmup_steps=config['warmup_steps'], num_training_steps=total, num_cycles=0.5,
        min_lr_rate=config['lr_scheduler_kwargs']['min_lr_rate'])
    return config['learning_rate'] * factor


def audit(checkpoint, config):
    import torch
    from safetensors.torch import load_file
    checkpoint = Path(checkpoint).resolve()
    state = json.loads((checkpoint / 'trainer_state.json').read_text())
    step = state['global_step']
    adapter = load_file(str(checkpoint / 'adapter_model.safetensors'))
    lora = [k for k in adapter if 'lora_' in k]
    if not lora or any(not torch.isfinite(adapter[k].float()).all() for k in adapter):
        raise ValueError('adapter为空或含非有限值')
    opt = torch.load(checkpoint / 'optimizer.pt', map_location='cpu', weights_only=False)
    states = opt['state']
    if len(states) != len(lora):
        raise ValueError(f'优化器状态数{len(states)}与LoRA参数数{len(lora)}不符')
    for s in states.values():
        if float(s['step']) != step:
            raise ValueError('优化器状态step与global_step不符')
        if not all(torch.isfinite(s[k]).all() for k in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('优化器动量含非有限值')
    sched = torch.load(checkpoint / 'scheduler.pt', map_location='cpu', weights_only=False)
    if sched['last_epoch'] != step:
        raise ValueError('scheduler last_epoch与global_step不符')
    want = expected_lr(config, step)
    if any(abs(lr - want) > 1e-12 for lr in sched['_last_lr']) or any(
            abs(g['lr'] - want) > 1e-12 for g in opt['param_groups']):
        raise ValueError(f'断点学习率{sched["_last_lr"]}与原调度第{step}步{want}不符')
    for rank in range(config['world_size']):
        torch.load(checkpoint / f'rng_state_{rank}.pth', map_location='cpu', weights_only=False)
    return {'status': 'passed', 'checkpoint': str(checkpoint), 'global_step': step,
            'max_steps': state['max_steps'], 'lora_tensors': len(lora), 'optimizer_states': len(states),
            'scheduler': {'last_epoch': sched['last_epoch'], '_last_lr': sched['_last_lr']},
            'expected_learning_rate': want, 'rng_ranks': config['world_size']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True, help='原训练run_config.json')
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = audit(a.checkpoint, json.loads(a.config.read_text()))
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
