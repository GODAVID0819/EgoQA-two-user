"""容量测量终态与同输入耗时校准；超过本次60小时资源边界则停止提交。"""
import json
from pathlib import Path
import subprocess


def terminal_success(job):
    text=subprocess.run(['sacct','-j',str(job),'-nP','--format=JobIDRaw,State,ExitCode'],capture_output=True,text=True,check=True).stdout
    rows={p[0]:p for line in text.splitlines() if (p:=line.split('|'))[0] in {str(job),str(job)+'.batch'}}
    if set(rows)!={str(job),str(job)+'.batch'} or any(r[1]!='COMPLETED' or r[2]!='0:0' for r in rows.values()):
        raise ValueError('前级尚未双层COMPLETED/0:0：'+str(job))
    return text


def verify_capacity_inputs(root,config,output):
    root=Path(root);output=Path(output)
    read=lambda p:json.loads(Path(p).read_text())
    actual=read(output/'run_config.json')
    allowed={'output_root','train_dataset','train_conditions','stop_after_steps','save_steps',
        'walltime_seconds','run_attempt','capacity_measurement'}
    changed=[key for key in set(config)|set(actual) if key not in allowed and config.get(key)!=actual.get(key)]
    if changed:raise ValueError('容量与正式设置不一致：'+','.join(sorted(changed)))
    if actual.get('capacity_measurement') is not True or actual.get('stop_after_steps')!=1:
        raise ValueError('实际输出不是一次隔离容量测量')
    rows=lambda p:[json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]
    formal_rows=rows(config['train_dataset'])
    indices=read(root/'preflight/input_lengths.json')['longest_pair_indices']
    expected=[formal_rows[index] for index in indices]
    if rows(actual['train_dataset'])!=expected:raise ValueError('实际测量数据不等于指定最长pair的完整原始行')
    for rank in range(2):
        trace=rows(output/f'rank{rank}/reward_trace.jsonl')
        row=expected[rank]
        if len(trace)!=4 or any((r['source_packet_id'],int(r['asker_index']))!=(row['source_packet_id'],int(row['asker_index'])) for r in trace):
            raise ValueError('rank实际生成条件不是最长pair对应条件')
    return indices


def budget_report(config,old,new,pair_step,job,*,elapsed_seconds=0):
    ratio=new[1]/old[pair_step];estimate=sum(old[i] for i in range(1,181))*ratio
    with_margin=estimate*1.25+1800
    return {'status':'passed' if with_margin+elapsed_seconds<=config['walltime_seconds'] else 'needs_review',
        'capacity_job_id':job,'source_job_id':'20191258','matched_source_step':pair_step,
        'source_pair_seconds':old[pair_step],'capacity_pair_seconds':new[1],'matched_pair_ratio':ratio,
        'source_first180_seconds':sum(old[i] for i in range(1,181)),
        'estimated_hours':estimate/3600,'with_25pct_and_30min_hours':with_margin/3600,
        'allocation_elapsed_seconds':elapsed_seconds,
        'total_allocation_with_margin_hours':(with_margin+elapsed_seconds)/3600,
        'requested_hours':config['walltime_seconds']/3600,
        'boundary':'同输入pair校准预算；包含已消耗allocation时间，冷启动和输出长度仍有不确定性'}


def accept_capacity(root,config,*,inline_job_id=None):
    root=Path(root);read=lambda p:json.loads(Path(p).read_text())
    elapsed_seconds=0
    if inline_job_id is None:
        receipt=read(root/'submissions/capacity_submission.json');terminal_success(receipt['job_id'])
    else:
        import os
        from combined_allocation import live_allocation
        receipt=read(root/'submissions/combined_capacity_execution.json')
        formal=read(root/'submissions/current_training_submission.json')
        state=read(root/f'preflight/combined_{inline_job_id}.json')
        if (os.environ.get('SLURM_JOB_ID')!=inline_job_id or receipt.get('job_id')!=inline_job_id
                or formal.get('job_id')!=inline_job_id or formal.get('phase')!='combined_capacity_then_stage180'
                or state.get('status')!='capacity_completed' or state.get('capacity_exit_code')!=0):
            raise ValueError('同allocation测量缺少当前作业、独立子进程成功或准确回执')
        elapsed_seconds=live_allocation(inline_job_id)['elapsed_seconds']
    output=Path(receipt['output_dir']);result=read(output/'training_result.json');manifest=read(output/'run_manifest.json')
    indices=verify_capacity_inputs(root,config,output)
    if (result.get('status')!='passed' or result.get('global_step')!=1 or result.get('candidate_count')!=8
            or result.get('unique_conditions')!=2 or result.get('schedule_max_steps')!=360
            or manifest.get('status')!='completed' or manifest.get('cleanup_errors')):
        raise ValueError('完整容量测量未通过')
    reload=read(Path(result['checkpoint'])/'reload_cpu_audit.json')
    if reload.get('optimizer_and_scheduler_step')!=1 or reload.get('optimizer_state_entries')!=160:
        raise ValueError('测量没有完成160参数完整更新、保存和CPU重载')
    if config.get('checkpoint_input_cpu_offload') is True:
        for rank in range(2):
            audits=[json.loads(line) for line in (output/f'rank{rank}/checkpoint_input_offload.jsonl').read_text().splitlines() if line.strip()]
            installed=[a for a in audits if a.get('event')=='installed']
            steps=[a for a in audits if a.get('event')=='step_end' and a.get('global_step')==1]
            stride=config.get('checkpoint_input_cpu_offload_stride',4)
            if (len(installed)!=1 or installed[0].get('selected_layers')!=list(range(stride-1,64,stride)) or not steps
                    or steps[-1].get('offloaded_bytes',0)<=0
                    or steps[-1].get('packed_tensors',0)<=0
                    or steps[-1].get('packed_tensors')!=steps[-1].get('restored_tensors')):
                raise ValueError('检查点输入转存没有真实执行并完成两卡反向恢复')
    source=Path(read(root/'preflight/source_provenance.json')['source_root'])
    old_receipt=read(source/'submissions/current_training_submission.json')
    if old_receipt['job_id']!='20191258':raise ValueError('源计时JobID不匹配')
    def walls(path):
        values={}
        for rank in range(2):
            for line in (path/f'rank{rank}/phase_timing.jsonl').read_text().splitlines():
                row=json.loads(line)
                if row.get('phase')=='optimizer_step_wall':
                    step=int(row['global_step']);values[step]=max(values.get(step,0),float(row['seconds']))
        return values
    old=walls(Path(old_receipt['output_dir']));new=walls(output)
    pair_step=indices[0]//2+1
    if not set(range(1,181))<=set(old) or pair_step not in old or 1 not in new:
        raise ValueError('缺少源180步或相同pair的实际耗时')
    report=budget_report(config,old,new,pair_step,receipt['job_id'],elapsed_seconds=elapsed_seconds)
    (root/'preflight/formal_walltime_basis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    if report['status']!='passed':raise ValueError('测量校准预算超过60h；不自动扩时或减步，等待讨论')
    return receipt,report
