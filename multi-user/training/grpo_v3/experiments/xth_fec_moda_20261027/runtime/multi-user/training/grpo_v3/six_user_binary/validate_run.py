"""检查实际训练产物，单独报告候选数、视频组覆盖和更新证据。"""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
import re
from .reward import aggregate, validate_probabilities
from .diversity import window_id


def finite_updated_adapter(path, config=None):
    import torch
    from safetensors.torch import load_file
    tensors = load_file(str(path), device="cpu")
    finite=bool(tensors) and all(bool(torch.isfinite(value).all()) for value in tensors.values())
    updated=any('lora_B' in name and bool(value.count_nonzero()) for name,value in tensors.items())
    layers={int(match.group(1)) for name in tensors if (match:=re.search(r'\.layers\.(\d+)\.',name))}
    adapter_config=json.loads((path.parent/'adapter_config.json').read_text())
    run_config=config or json.loads((path.parent.parents[2]/'run_config.json').read_text())
    capacity=run_config.get('capacity_measurement') is True and run_config.get('stop_after_steps')==1
    if not finite or (not updated and not capacity) or layers!=set(range(64)) or len(tensors)!=160:
        raise ValueError('重载adapter非有限、未更新或64语言层覆盖不完整')
    if adapter_config['r']!=64 or adapter_config['lora_alpha']!=128 or adapter_config['lora_dropout']!=.05:
        raise ValueError('最终adapter超参与已批准配置不一致')
    if any('visual' in name or 'aligner' in name for name in tensors): raise ValueError('冻结视觉或aligner出现可训练权重')
    state=json.loads((path.parent/'trainer_state.json').read_text())
    scheduler=torch.load(path.parent/'scheduler.pt',map_location='cpu',weights_only=False)
    optimizer=torch.load(path.parent/'optimizer.pt',map_location='cpu',weights_only=False)
    target=int(state['global_step'])
    from transformers.optimization import get_scheduler
    reference_optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))],lr=run_config['learning_rate'])
    reference_scheduler=get_scheduler(run_config['lr_scheduler_type'],reference_optimizer,
        num_warmup_steps=run_config['warmup_steps'],num_training_steps=run_config['max_steps'],
        scheduler_specific_kwargs=run_config.get('lr_scheduler_kwargs',{}))
    for _ in range(target):reference_optimizer.step();reference_scheduler.step()
    expected_lr=reference_scheduler.get_last_lr()[0]
    if not scheduler.get('_last_lr') or not all(math.isclose(v,expected_lr,rel_tol=1e-6,abs_tol=1e-12) for v in scheduler['_last_lr']):
        raise ValueError('阶段scheduler没有延续原调度曲线')
    if scheduler.get('last_epoch')!=target: raise ValueError('scheduler与真实更新数不同步')
    if len(optimizer.get('state',{}))!=160: raise ValueError('optimizer状态不是160个LoRA参数')
    identifiers=[identifier for group in optimizer.get('param_groups',[]) for identifier in group['params']]
    if len(identifiers)!=160 or set(identifiers)!=set(optimizer['state']):
        raise ValueError('Adam参数组与160项状态未逐项对应')
    for value in optimizer['state'].values():
        if not {'step','exp_avg','exp_avg_sq'}<=set(value):raise ValueError('Adam动量/方差/step缺失')
        if int(value['step'])!=target: raise ValueError('Adam步数与全局更新数不同步')
        if value['exp_avg'].shape!=value['exp_avg_sq'].shape:raise ValueError('Adam两项动量形状不一致')
        if any(not bool(torch.isfinite(v).all()) for v in value.values() if torch.is_tensor(v)):
            raise ValueError('optimizer状态含非有限值')
    if Counter(tuple(v['exp_avg'].shape) for v in optimizer['state'].values())!=Counter(tuple(v.shape) for v in tensors.values()):
        raise ValueError('Adam状态形状集合与实际adapter参数不一致')
    restored_groups=[{'params':[torch.nn.Parameter(torch.zeros_like(optimizer['state'][identifier]['exp_avg']))
        for identifier in group['params']]} for group in optimizer['param_groups']]
    restored_optimizer=torch.optim.AdamW(restored_groups,lr=run_config['learning_rate'])
    restored_optimizer.load_state_dict(optimizer)
    if len(restored_optimizer.state)!=160:raise ValueError('实际CPU optimizer.load_state_dict未恢复160项状态')
    import random
    import numpy as np
    rng_keys=[]
    for rank in range(2):
        rng=torch.load(path.parent/f'rng_state_{rank}.pth',map_location='cpu',weights_only=False)
        if not {'python','numpy','cpu','cuda'}<=set(rng):raise ValueError('双rank随机状态字段不全')
        random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy'])
        torch.Generator(device='cpu').set_state(rng['cpu'])
        cuda_states=rng['cuda'] if isinstance(rng['cuda'],(list,tuple)) else [rng['cuda']]
        if not cuda_states or not all(torch.is_tensor(v) and v.numel()>0 for v in cuda_states):
            raise ValueError('CUDA随机状态为空或不可重载')
        rng_keys.append(sorted(rng))
    report={'status':'passed','loaded_on':'CPU','tensor_count':len(tensors),'language_layers':sorted(layers),
            'finite':True,'nonzero_lora_B':updated,'optimizer_state_entries':len(optimizer['state']),
            'optimizer_and_scheduler_step':target,'last_lr':scheduler['_last_lr'],
            'global_step':target,'checkpoint':str(path.parent),'scheduler':scheduler,
            'optimizer_states':len(optimizer['state']),'rng_keys':rng_keys,
            'boundary':'独立重载权重与优化器/调度器工程检查，不是独立答题评估'}
    (path.parent/'reload_cpu_audit.json').write_text(json.dumps(report,indent=2))
    return True


def validate(output, *, adapter_check=finite_updated_adapter):
    output = Path(output)
    config = json.loads((output / "run_config.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (output / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    states = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in (output / "swift").rglob("trainer_state.json")]
    selected = max(states, key=lambda item: int(item[1].get("global_step", 0))) if states else None
    state = selected[1] if selected else {}
    checkpoint = selected[0].parent if selected else output / "missing_checkpoint"
    history = state.get("log_history", [])
    rewards = [r.get("reward") for r in rows]
    finite = all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1.5 for x in rewards)
    groups = defaultdict(list)
    for row in rows:
        key = (row["request_id"].rsplit(":", 1)[0], row["evidence_id"])
        groups[key].append(row["reward"])
    allowed_sizes = {config["num_generations"]}
    nonconstant = sum(len(values) > 1 and max(values) > min(values) for values in groups.values()) if finite else 0
    adapter = checkpoint / "adapter_model.safetensors"
    target = config.get('stop_after_steps', config['max_steps'])
    checks = {
        "nonempty_reward_trace": bool(rows),
        "finite_rewards": finite,
        "unique_request_ids": len({r["request_id"] for r in rows}) == len(rows),
        "complete_observed_groups": bool(groups) and all(len(v) in allowed_sizes for v in groups.values()),
        "observed_group_variance_positive": nonconstant > 0,
        "at_least_two_judged_candidates": sum(r["judge_result"]["status"] == "scored" for r in rows) >= 2,
        "requested_steps_reached": int(state.get("global_step", 0)) == target,
        "finite_nonzero_gradient": any(isinstance(r.get("grad_norm"), (int, float)) and math.isfinite(r["grad_norm"]) and r["grad_norm"] > 0 for r in history),
        "finite_reported_metrics": all(math.isfinite(v) for row in history for v in row.values() if isinstance(v, (int, float))),
        "no_validation_requested": config.get('validation_conditions')==0 and config.get('val_dataset') is None
            and config.get('eval_strategy')=='no' and not config.get('paired_validation') and not config.get('baseline_after_training'),
        "finite_nonzero_lora_b": adapter.is_file() and (adapter_check(adapter,config=config)
            if adapter_check is finite_updated_adapter else adapter_check(adapter)),
    }
    if 'stop_after_steps' in config:
        required = ('adapter_model.safetensors', 'adapter_config.json', 'optimizer.pt', 'scheduler.pt',
                    'rng_state_0.pth', 'rng_state_1.pth', 'args.json', 'training_args.bin')
        checks['complete_stage_checkpoint'] = all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in required)
        checks['schedule_horizon_preserved'] = state.get('max_steps') == config['max_steps']
    checks['exact_training_candidate_count']=len(rows)==target*8
    checks['exact_training_group_count']=len(groups)==target*2
    condition_counts=defaultdict(int)
    passed=0;invalid=0
    components_correct=True
    for row in rows:
        condition_counts[(window_id(row['source_packet_id']),int(row['asker_index']))]+=1
        if row['judge_result']['status']=='invalid_completion':
            invalid+=1
            components_correct=components_correct and row['reward']==0 and not row['bonus_gate']
            continue
        scores=validate_probabilities(row['judge_result']['probabilities'])
        gate=all(value>=.5 for value in scores.values());passed+=int(gate)
        distance=row['diversity_distance']
        components_correct=components_correct and 0<=distance<=1 and row['bonus_gate']==gate
        components_correct=components_correct and 0<=row['reference_count']<=11
        components_correct=components_correct and math.isclose(row['quality_reward'],aggregate(scores,mode='continuous'),abs_tol=1e-10)
        components_correct=components_correct and math.isclose(row['diversity_bonus'],.5*int(gate)*distance,abs_tol=1e-10)
        components_correct=components_correct and math.isclose(row['reward'],row['quality_reward']+row['diversity_bonus'],abs_tol=1e-10)
    checks['reward_components_match_approved_formula']=components_correct
    begin=config.get('expected_resume_step',0) if config.get('resume_from_checkpoint') else 0
    checks['all_conditions_once']=len(condition_counts)==target*2 and set(condition_counts.values())=={4}
    planned=[json.loads(line) for line in Path(config['train_dataset']).read_text().splitlines() if line.strip()]
    expected=Counter((window_id(row['source_packet_id']),int(row['asker_index'])) for row in planned[:target*2])
    checks['actual_conditions_match_training_manifest']=(len(planned)==config['train_conditions']
        and len(expected)==target*2 and Counter(condition_counts)==Counter({key:count*4 for key,count in expected.items()}))
    if config['train_conditions']==720:
        window_counts=defaultdict(int);speaker_counts=defaultdict(int)
        for row in planned:
            window_counts[window_id(row['source_packet_id'])]+=1;speaker_counts[int(row['asker_index'])]+=1
        checks['source_manifest_120_windows_six_speakers']=len(window_counts)==120 and set(window_counts.values())=={6} and dict(speaker_counts)=={i:120 for i in range(6)}
    rank_traces=[]
    for rank in range(2):
        rank_dir=output/('rank'+str(rank))
        trace_path=rank_dir/'reward_trace.jsonl'
        rank_rows=[json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()] if trace_path.exists() else []
        rank_traces.extend(rank_rows)
        checks[f'rank{rank}_condition_order']=len(rank_rows)==target*4 and all(
            (row['source_packet_id'],int(row['asker_index']),row.get('global_step'))==(
                planned[2*(index//4)+rank]['source_packet_id'],int(planned[2*(index//4)+rank]['asker_index']),index//4+1)
            for index,row in enumerate(rank_rows))
        sync_path=rank_dir/'ddp_sync.jsonl';gradient_path=rank_dir/'gradient_sync.jsonl'
        sync=[json.loads(line) for line in sync_path.read_text().splitlines() if line.strip()] if sync_path.exists() else []
        gradients=[json.loads(line) for line in gradient_path.read_text().splitlines() if line.strip()] if gradient_path.exists() else []
        if begin:
            resume=[row for row in gradients if row.get('event')=='resume_optimizer_verified']
            rng_path=rank_dir/'resume_rng.jsonl'
            rng=[json.loads(line) for line in rng_path.read_text().splitlines() if line.strip()] if rng_path.exists() else []
            checks[f'rank{rank}_native_optimizer_resumed']=len(resume)==1 and resume[0].get('global_step')==begin and resume[0].get('scheduler_last_epoch')==begin
            checks[f'rank{rank}_native_RNG_resumed']=bool(rng) and any(row.get('event')=='native_rng_restore_returned' and row.get('global_step')==begin and row.get('checkpoint')==config['resume_from_checkpoint'] for row in rng)
        gradients=[row for row in gradients if row.get('event')!='resume_optimizer_verified']
        checks['rank'+str(rank)+'_all_LoRA_synchronized']=bool(sync) and max(r['global_step'] for r in sync)==target and all(
            r['all_trainable_parameters_equal'] and r['finite'] and r['checked_tensors']==160 for r in sync)
        checks['rank'+str(rank)+'_gradients_mean_before_clip']=len(gradients)==target-begin and [
            r['global_step'] for r in gradients]==list(range(begin,target)) and all(
            r['finite'] and r['world_size']==2 and r['tensors']==160 and r['order']=='mean_then_clip_then_optimizer' for r in gradients)
    checks['merged_trace_matches_rank_traces']=rows==rank_traces
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "failed_checks": [key for key, value in checks.items() if not value], "global_step": state.get("global_step", 0),
        "checkpoint": str(checkpoint), "schedule_max_steps": config['max_steps'], "stage_target_steps": target,
        "stage_begin":begin,"stage_end":target,"stage_added_updates":target-begin,
        "stage_added_conditions":2*(target-begin),"stage_added_candidate_slots":8*(target-begin),
        "candidate_count": len(rows), "observed_group_count": len(groups),
        "unique_conditions":len(condition_counts),"validation_conditions":0,
        "training_mean_reward":sum(rewards)/len(rewards) if rewards else None,
        "quality_reward_mean":sum(r['quality_reward'] for r in rows)/len(rows) if rows else None,
        "diversity_bonus_mean":sum(r['diversity_bonus'] for r in rows)/len(rows) if rows else None,
        "auto_pass_count":passed,"auto_pass_denominator":len(rows),"auto_pass_rate":passed/len(rows) if rows else None,
        "invalid_completion_count":invalid,
        "nonconstant_group_count": nonconstant, "source_packet_count": len({r["source_packet_id"] for r in rows}),
        "candidate_scope": "仅本轮已完成训练更新的原生成槽位，含invalid，不包含历史重评分或旧验证",
        "evidence_boundary": "仅训练工程证据；不证明固定测试集、人类 QA 质量或跨视频泛化"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(args.output)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    (args.output / "training_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
