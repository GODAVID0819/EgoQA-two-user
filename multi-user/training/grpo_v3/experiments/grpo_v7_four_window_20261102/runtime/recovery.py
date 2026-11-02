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
    stop=config.get('stop_after_steps',config['max_steps'])
    if type(stop) is not int or not step<stop<=config['max_steps']:
        raise ValueError('阶段终点必须晚于恢复步数且不超过原调度终点')
    if checkpoint.parents[2]!=source or source.name!='train_'+str(config['recovery_from_job']):
        raise ValueError('断点和来源JobID/输出目录不匹配')
    required=['adapter_model.safetensors','adapter_config.json','optimizer.pt','scheduler.pt',
              'rng_state_0.pth','rng_state_1.pth','trainer_state.json','args.json','training_args.bin']
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
              'recovery_from_job','expected_resume_step','initial_fresh_base','resume_cpu_audit','walltime_basis','run_attempt',
              'stop_after_steps','walltime_seconds'}
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
    rows=head_rows(config['train_dataset'],4*meta['step']+4)
    start=4*meta['step']
    return [(rows[start+2*rank]['source_packet_id'],int(rows[start+2*rank]['asker_index'])) for rank in (0,1)]


def seed_reward_prefix(config,output):
    meta=validated_resume(config)
    if meta is None:return None
    from training.grpo_v3.six_user_binary.v7_batch import replay_history
    output=Path(output);source=Path(meta['source_output']);count=meta['step']*8
    planned=head_rows(config['train_dataset'],4*meta['step'])
    prepared={};all_rows=[]
    for rank in (0,1):
        rows=[json.loads(line) for line in (source/f'rank{rank}/reward_trace.jsonl').read_text().splitlines() if line.strip()][:count]
        if len(rows)!=count:raise ValueError('来源更新的真实八槽前缀不完整')
        for index,row in enumerate(rows):
            expected=planned[4*(index//8)+2*rank+(index%8)//4]
            if (row['source_packet_id'],int(row['asker_index']),row['global_step'])!=(expected['source_packet_id'],int(expected['asker_index']),index//8+1):
                raise ValueError('来源前缀与四窗实际采样顺序不一致')
        target=output/f'rank{rank}/reward_trace.jsonl'
        if target.exists():raise FileExistsError('禁止覆盖已有训练轨迹')
        prepared[target]=rows;all_rows.extend(rows)
    if not replay_history(all_rows,meta['step']):raise ValueError('来源奖励历史不能独立回放')
    for target,rows in prepared.items():
        target.write_text(''.join(json.dumps(row)+chr(10) for row in rows))
    report={**meta,'inherited_candidates':len(all_rows),'remaining_updates':config['max_steps']-meta['step'],
            'note':'同一v7实验，继承完整状态和已完成统计，不重新训练或重评分此前候选'}
    (output/'recovery_lineage.json').write_text(json.dumps(report,indent=2))
    return report


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
