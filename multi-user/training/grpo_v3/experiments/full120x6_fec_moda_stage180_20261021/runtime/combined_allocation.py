"""同一2H200资源作业内：最长pair完整测量、清理，再独立fresh训练180步。"""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,values):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(values,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(path)


def seconds(value):
    days,clock=value.split('-',1) if '-' in value else ('0',value)
    parts=[int(v) for v in clock.split(':')]
    if len(parts)!=3:raise ValueError('无法解析Slurm时间：'+value)
    return int(days)*86400+parts[0]*3600+parts[1]*60+parts[2]


def live_allocation(job):
    text=subprocess.check_output(['scontrol','show','job',str(job),'-o'],text=True)
    fields=dict(re.findall(r'(\w+)=(\S*)',text))
    if (fields.get('JobId')!=job or fields.get('JobState')!='RUNNING'
            or not fields.get('UserId','').startswith('xl6775(')
            or fields.get('TimeLimit')!='2-12:00:00' or fields.get('Features')!='h200'
            or not all(x in fields.get('ReqTRES','').split(',') for x in ['cpu=32','mem=750G','gres/gpu=2'])):
        raise ValueError('当前资源不是本人2H200/32CPU/750G/60h运行作业：'+text)
    return {**fields,'elapsed_seconds':seconds(fields['RunTime']),'time_limit_seconds':216000}


def validate_settings(capacity,formal):
    fixed={'max_steps':360,'max_pixels':50176,'max_length':131072,'num_generations':4,
           'learning_rate':1e-5,'seed':42,'data_seed':42,'warmup_steps':10,
           'lr_scheduler_type':'cosine_with_min_lr','lr_scheduler_kwargs':{'min_lr_rate':0.1},
           'resume_from_checkpoint':None,'initial_fresh_base':True,'validation_conditions':0}
    for c in [capacity,formal]:
        if any(c.get(k)!=v for k,v in fixed.items()):raise ValueError('必须沿用fresh/50176/G4/原360余弦与seed42')
        stride=c.get('checkpoint_input_cpu_offload_stride',4)
        if type(stride) is not int or stride not in (1,4):raise ValueError('不支持未验证的输入转存层间隔')
    if (capacity.get('capacity_measurement') is not True or capacity['stop_after_steps']!=1
            or capacity['train_conditions']!=2 or capacity['walltime_seconds']!=7200
            or formal.get('capacity_measurement') or formal['stop_after_steps']!=180
            or formal['train_conditions']!=720 or formal['walltime_seconds']!=216000
            or capacity['output_root']==formal['output_root']):
        raise ValueError('测量和正式阶段必须完整隔离且总上限60h')
    allowed={'output_root','train_dataset','train_conditions','stop_after_steps','save_steps',
        'walltime_seconds','run_attempt','capacity_measurement'}
    changed=[k for k in set(capacity)|set(formal) if k not in allowed and capacity.get(k)!=formal.get(k)]
    if changed:raise ValueError('测量与正式配置不同：'+','.join(sorted(changed)))


def launch_phase(root,config_path,timeout,log):
    c=read(config_path)
    env={k:v for k,v in os.environ.items() if not k.startswith('EGOQA_')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    with Path(log).open('w',encoding='utf-8') as stream:
        child=subprocess.Popen([c['train_python'],str(root/'dual_launch.py'),'--config',str(config_path)],
            cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
        old_handlers={}
        def interrupted(signum,frame):raise RuntimeError('合并入口收到终止信号'+str(signum))
        for sig in [signal.SIGTERM,signal.SIGINT]:old_handlers[sig]=signal.signal(sig,interrupted)
        try:
            code=child.wait(timeout=timeout)
            if code:raise subprocess.CalledProcessError(code,child.args)
        finally:
            for sig in old_handlers:signal.signal(sig,signal.SIG_IGN)
            if child.poll() is None:
                child.terminate()
                try:child.wait(timeout=90)
                except subprocess.TimeoutExpired:
                    child.kill();child.wait()
            for sig,handler in old_handlers.items():signal.signal(sig,handler)
    return child.pid


def accepted_result(output,step):
    manifest=read(output/'run_manifest.json');result=read(output/'training_result.json')
    if (manifest.get('status')!='completed' or manifest.get('cleanup_errors')
            or result.get('status')!='passed' or result.get('global_step')!=step
            or result.get('schedule_max_steps')!=360 or result.get('candidate_count')!=step*8
            or result.get('unique_conditions')!=step*2):
        raise ValueError('阶段退出后实际结果或清理未通过')
    return result


def await_submission(root,job,timeout=60):
    deadline=time.monotonic()+timeout
    while True:
        receipt=read(Path(root)/'submissions/current_training_submission.json')
        if receipt.get('phase')!='combined_capacity_then_stage180':
            raise ValueError('当前回执不是合并阶段')
        if receipt.get('job_id')==job and receipt.get('status')=='submitted':return receipt
        if receipt.get('status')!='submitting' or receipt.get('job_id') not in (None,job):
            raise ValueError('只执行本次合并提交的准确JobID')
        if time.monotonic()>=deadline:raise TimeoutError('sbatch返回后的准确JobID回执未及时落盘；禁止重投')
        time.sleep(.1)


def run_combined(root,job):
    root=Path(root);cap_path=root/'configs/capacity.json';formal_path=root/'configs/train.json'
    cap=read(cap_path);formal=read(formal_path);validate_settings(cap,formal)
    receipt=await_submission(root,job)
    state_path=root/f'preflight/combined_{job}.json'
    cap_output=Path(cap['output_root'])/('train_'+job)
    formal_output=Path(formal['output_root'])/('train_'+job)
    if state_path.exists() or cap_output.exists() or formal_output.exists():
        raise FileExistsError('已有阶段状态或输出，禁止重复运行/覆盖')
    allocation=live_allocation(job)
    state={'job_id':job,'mode':'combined_capacity_then_stage180','status':'capacity_running',
           'capacity_output':str(cap_output),'formal_output':str(formal_output),
           'allocation_limit_seconds':216000,'fresh_formal':True}
    def record(**values):
        state.update(values,updated_at_utc=datetime.now(timezone.utc).isoformat());save(state_path,state)
        save(root/'preflight/pipeline_state.json',state)
        print('COMBINED_STATE',json.dumps(state,ensure_ascii=False),flush=True)
    record(allocation=allocation)
    save(root/'submissions/combined_capacity_execution.json',
        {'job_id':job,'mode':'inline_capacity','output_dir':str(cap_output),'config':str(cap_path)})
    try:
        pid=launch_phase(root,cap_path,min(7200,216000-allocation['elapsed_seconds']),
            root/f'preflight/combined_{job}_capacity.log')
        accepted_result(cap_output,1)
        record(status='capacity_completed',capacity_exit_code=0,capacity_pid=pid)
        from capacity_acceptance import accept_capacity
        _,budget=accept_capacity(root,formal,inline_job_id=job)
        allocation=live_allocation(job)
        remaining=216000-allocation['elapsed_seconds']
        if remaining<=0:raise ValueError('allocation已无剩余时间')
        record(status='stage180_running',capacity_acceptance='passed',walltime_basis=budget,
            formal_started_at_utc=datetime.now(timezone.utc).isoformat(),
            formal_remaining_seconds=remaining,fresh_initialization={
                'separate_process':True,'resume_from_checkpoint':None,'seed':42,'data_seed':42,
                'optimizer_step':0,'scheduler_step':0,'data_start_condition_index':0})
        pid=launch_phase(root,formal_path,remaining,root/f'preflight/combined_{job}_stage180.log')
        accepted_result(formal_output,180)
        record(status='completed',formal_exit_code=0,formal_pid=pid,formal_global_step=180)
    except BaseException as exc:
        record(status='failed',failed_phase=state['status'],error=repr(exc),traceback=traceback.format_exc())
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();run_combined(args.root,os.environ['SLURM_JOB_ID'])
