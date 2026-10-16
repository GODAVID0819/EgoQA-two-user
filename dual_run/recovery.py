"""校验完整断点并继承已训练条件的真实审计前缀，不回放旧QA训练。"""
import json
from pathlib import Path
from itertools import islice


def head_rows(path,count):
    with Path(path).open() as stream:
        return list(islice((json.loads(line) for line in stream if line.strip()),count))


def validated_resume(config):
    if not config.get('resume_from_checkpoint'):
        return None
    checkpoint=Path(config['resume_from_checkpoint']).resolve()
    source=Path(config['resume_source_output']).resolve()
    step=config['expected_resume_step']
    if type(step) is not int or not 0<step<config['max_steps']:
        raise ValueError('恢复步数必须在原调度范围内')
    if checkpoint.parents[2]!=source or source.name!='train_'+str(config['recovery_from_job']):
        raise ValueError('断点和来源JobID/输出目录不匹配')
    required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
              *('rng_state_'+str(r)+'.pth' for r in range(config['world_size'])),'trainer_state.json','args.json','training_args.bin']
    if not all((checkpoint/name).is_file() and (checkpoint/name).stat().st_size for name in required):
        raise ValueError('恢复断点缺少完整模型/优化器/调度器/双rank随机状态')
    state=json.loads((checkpoint/'trainer_state.json').read_text())
    adapter=json.loads((checkpoint/'adapter_config.json').read_text())
    if state['global_step']!=step or state['max_steps']!=config['max_steps']:
        raise ValueError('恢复不能重置步数或改变原调度horizon')
    if adapter['r']!=config['lora_rank'] or adapter['lora_alpha']!=config['lora_alpha']:
        raise ValueError('恢复adapter rank/alpha不符')
    old=json.loads((source/'run_config.json').read_text())
    controls={'project_root','output_root','judge_config','resume_from_checkpoint','resume_source_output',
              'recovery_from_job','expected_resume_step','initial_fresh_base','resume_cpu_audit','walltime_basis','run_attempt'}
    changed=[key for key in set(old)|set(config) if key not in controls and old.get(key)!=config.get(key)]
    if changed:raise ValueError('恢复不允许改变训练设置：'+','.join(sorted(changed)))
    if json.loads(Path(old['judge_config']).read_text())!=json.loads(Path(config['judge_config']).read_text()):
        raise ValueError('恢复不能改变冻结Judge')
    audit_path=config.get('resume_cpu_audit')
    audit=json.loads(Path(audit_path).read_text()) if audit_path else {}
    if (audit.get('status')!='passed' or audit.get('global_step')!=step
            or Path(audit.get('checkpoint','')).resolve()!=checkpoint):
        raise ValueError('缺少对应断点的实际CPU完整状态核验')
    return {'step':step,'horizon':config['max_steps'],'checkpoint':str(checkpoint),
            'source_output':str(source),'source_job':str(config['recovery_from_job']),
            'expected_learning_rate':audit['scheduler']['_last_lr'][0],
            'optimizer_states':audit['optimizer_states']}


def expected_resume_bindings(config):
    meta=validated_resume(config)
    if meta is None:return None
    # 恢复后第一轮生成：第step步的前world_size行（依次给rank0..）。训练文件按epoch循环。
    from dual_runtime import step_row
    rows=head_rows(config['train_dataset'],config['train_conditions']);world=config['world_size']
    picked=[rows[step_row(meta['step'],0,r,world)%len(rows)] for r in range(world)]
    return [(r['source_packet_id'],int(r['asker_index'])) for r in picked]


def seed_reward_prefix(config,output):
    meta=validated_resume(config)
    if meta is None:return None
    from dual_runtime import rounds_per_step, step_row
    world=config['world_size'];rounds=rounds_per_step(world)
    output=Path(output);source=Path(meta['source_output']);g=config['num_generations'];count=meta['step']*rounds*g
    data=head_rows(config['train_dataset'],config['train_conditions'])
    prepared={};ids=set()
    for rank in range(world):
        rows=[json.loads(x) for x in (source/('rank'+str(rank))/'reward_trace.jsonl').read_text().splitlines() if x.strip()]
        if len(rows)<count:raise ValueError('断点来源评分前缀不完整')
        for step in range(meta['step']):
            for round_index in range(rounds):
                expected=data[step_row(step,round_index,rank,world)%len(data)]
                group=rows[(rounds*step+round_index)*g:(rounds*step+round_index+1)*g]
                if any((r['source_packet_id'],int(r['asker_index']))!=(expected['source_packet_id'],int(expected['asker_index'])) for r in group):
                    raise ValueError('恢复评分前缀与实际训练条件顺序不符')
        prefix=rows[:count]
        for row in prefix:
            if row['request_id'] in ids:raise ValueError('恢复评分前缀request_id重复')
            ids.add(row['request_id'])
        target=output/('rank'+str(rank))/'reward_trace.jsonl'
        if target.exists():raise FileExistsError('禁止覆盖/重复追加已有评分前缀')
        prepared[target]=''.join(json.dumps(row)+'\n' for row in prefix)
    for target,text in prepared.items():target.write_text(text)
    result={**meta,'inherited_candidates':len(ids),'remaining_updates':config['max_steps']-meta['step'],
            'note':'仅继承真实审计记录；由原始数据和native skip继续剩余条件，不把旧QA回放训练'}
    (output/'recovery_lineage.json').write_text(json.dumps(result,indent=2))
    return result


def check_restored_optimizer(trainer,meta):
    if trainer.state.global_step!=meta['step'] or trainer.args.max_steps!=meta['horizon']:
        raise RuntimeError('实际恢复步数/horizon错误')
    if trainer.lr_scheduler.last_epoch!=meta['step']:
        raise RuntimeError('实际scheduler被重置')
    rates=[float(g['lr']) for g in trainer.optimizer.param_groups]
    if not rates or any(abs(lr-meta['expected_learning_rate'])>1e-12 for lr in rates):
        raise RuntimeError('恢复学习率不连续')
    steps={float(value['step']) for value in trainer.optimizer.state.values()}
    if len(trainer.optimizer.state)!=meta['optimizer_states'] or steps!={float(meta['step'])}:
        raise RuntimeError('实际优化器状态没有完整恢复')
    return {'global_step':trainer.state.global_step,'scheduler_last_epoch':trainer.lr_scheduler.last_epoch,
            'learning_rates':rates,'optimizer_steps':sorted(steps),'checkpoint':meta['checkpoint']}
