"""仅在指定零GPU检查全部通过后提交一次；任一检查失败则不申请GPU。"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import uuid


CHECKS=('train_environment','judge_environment','train_data','train_processor','judge_processor')


def validate_submission_config(c):
    steps = c.get('formal_max_steps')
    stop = c.get('stop_after_steps', steps)
    if (isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0
            or isinstance(stop, bool) or not isinstance(stop, int) or not 0 < stop <= steps
            or not c.get('paired_validation') or c.get('execution_mode') != 'direct'):
        raise ValueError('提交必须明确正整数训练目标、合法阶段终点及固定验证对比')


def direct_submit_command(config, workflow_path, task_dir):
    from .workflow import training_config
    from .submit import sbatch_command
    if config.get('execution_mode') != 'direct':
        raise ValueError('必须提交直接执行训练的作业')
    run=training_config(config,job_id='0',phase='formal',max_steps=config['formal_max_steps'])
    run.update(slurm=config['slurm'],walltime_basis=config['walltime_basis'])
    return sbatch_command(run,str(workflow_path),str(task_dir))


def verify_checks(directory, *, require_checkpoint=False, expected_run=None, expected_split_counts=None):
    directory=Path(directory)
    status=json.loads((directory/'status.json').read_text())
    if status['status']!='passed':raise ValueError('零GPU检查尚未全部通过')
    reports={name:json.loads((directory/(name+'.json')).read_text()) for name in CHECKS}
    if any(r['status']!='passed' for r in reports.values()):raise ValueError('存在未通过的零GPU项目')
    for role in ('train','judge'):
        checks=reports[role+'_environment']['checks']
        if checks['pip_check']['returncode']!=0:raise ValueError('pip check未通过')
        if not all(checks['host_compilers'][key]['shared_library_loaded'] for key in ('CC','CXX')):
            raise ValueError('实际编译/加载未通过')
    expected_counts={'train':18,'validation':6} if expected_split_counts is None else expected_split_counts
    if (not isinstance(expected_counts,dict) or not {'train','validation'} <= set(expected_counts)
            or any(isinstance(n,bool) or not isinstance(n,int) or n<=0 for n in expected_counts.values())):
        raise ValueError('预期划分计数必须包含正整数训练和验证数量')
    if reports['train_data']['checks']['rows']!=expected_counts:
        raise ValueError(f'训练/验证数量不符：实际{reports["train_data"]["checks"]["rows"]}，期望{expected_counts}')
    if expected_run is not None:
        typed = reports['train_environment']['checks'].get('typed_cli', {})
        for key in ('max_steps', 'eval_steps', 'learning_rate', 'lr_scheduler_type', 'lr_scheduler_kwargs', 'warmup_steps'):
            if key in expected_run and typed.get(key) != expected_run[key]:
                raise ValueError(f'当前配置与实际CLI检查不一致：{key} actual={typed.get(key)!r} expected={expected_run[key]!r}')
        if 'stop_after_steps' in expected_run and 'egoqa_stage_stop' not in typed.get('callbacks', []):
            raise ValueError('当前阶段缺少实际CLI回调检查')
    if require_checkpoint:
        resume=json.loads((directory/'train_checkpoint.json').read_text())
        if resume.get('status')!='passed' or not resume.get('checks',{}).get('finite_updated_adapter'):
            raise ValueError('checkpoint恢复检查未通过')
    return reports


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--cpu-job-id')
    args=parser.parse_args()
    if args.cpu_job_id and not args.cpu_job_id.isdigit():raise ValueError('CPU JobID必须是数字')
    c=json.loads(args.config.read_text())
    validate_submission_config(c)
    root=Path(c['project_root'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=root/'submissions'/('grpo'+str(c['formal_max_steps'])+'_'+stamp)
    task.mkdir(parents=True,exist_ok=False)
    with (args.preflight/'submission.claim').open('x') as f:f.write(str(task))
    record={'status':'waiting_for_cpu_validation','task_dir':str(task),'preflight_dir':str(args.preflight),
            'config':c,'cpu_preflight_job_id':args.cpu_job_id,
            'data_provenance_job_id':c.get('data_provenance_job_id'),'created_at_utc':stamp}
    def save():(task/'submission.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    save()
    try:
        deadline=time.time()+7200
        last_scheduler_check=0
        while True:
            state=json.loads((args.preflight/'status.json').read_text())
            if state['status']=='passed':break
            if state['status']=='failed':raise RuntimeError('零GPU检查失败，不提交GPU：'+state.get('error',''))
            if args.cpu_job_id and time.time()-last_scheduler_check>60:
                cpu_state=subprocess.check_output(['sacct','-X','-nP','-j',args.cpu_job_id,'-o','State'],text=True).strip()
                if any(word in cpu_state for word in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')):
                    raise RuntimeError('CPU验证作业结束但检查未通过：'+cpu_state)
                last_scheduler_check=time.time()
            if time.time()>deadline:raise TimeoutError('等待零GPU结果超时，不提交GPU')
            time.sleep(15)
        from .workflow import training_config
        expected_run=training_config(c,job_id='0',phase='formal',max_steps=c['formal_max_steps'])
        verify_checks(args.preflight,require_checkpoint=bool(c.get('resume_from_checkpoint')),expected_run=expected_run,
                      expected_split_counts=c.get('expected_split_counts'))
        if json.loads(args.config.read_text())!=c:raise ValueError('检查期间配置发生变化，不提交GPU')
        if subprocess.check_output(['whoami'],text=True).strip()!='xl6775':raise RuntimeError('用户身份不符')
        assoc=subprocess.check_output(['sacctmgr','-nP','show','assoc','user=xl6775','format=User,Account,Partition,QOS'],text=True)
        if c['slurm']['account'] not in assoc:raise RuntimeError('当前account不可用')
        partition=subprocess.check_output(['scontrol','show','partition'] + ([c['slurm']['partition']] if c['slurm'].get('partition') else []),text=True)
        qos=subprocess.check_output(['sacctmgr','-nP','show','qos'] + ([c['slurm']['qos']] if c['slurm'].get('qos') else []) + ['format=Name,MaxTRESPU,GrpTRES'],text=True)
        if 'State=UP' not in partition or (c['slurm'].get('qos') and c['slurm']['qos'] not in qos):raise RuntimeError('partition或QOS不可用')
        (task/'scheduler_check.txt').write_text(assoc+'\n'+partition+'\n'+qos)
        script=root/'multi-user/hpc/grpo_v3/six_user_binary/train_direct.sbatch'
        subprocess.run(['bash','-n',str(script)],check=True)
        workflow=task/'workflow.json';workflow.write_text(json.dumps(c,indent=2))
        command=direct_submit_command(c,workflow,task)
        result=subprocess.run(command,text=True,capture_output=True)
        (task/'sbatch.stdout').write_text(result.stdout);(task/'sbatch.stderr').write_text(result.stderr)
        result.check_returncode()
        job=result.stdout.strip().split(';')[0]
        if not job.isdigit():raise RuntimeError('提交未返回合法JobID：'+result.stdout)
        record.update(status='submitted',job_id=job,command=command,submitted_epoch=time.time());save()
        print(json.dumps({'job_id':job,'manifest':str(task/'submission.json')}),flush=True)
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',finished_epoch=time.time());save()
        raise


if __name__=='__main__':main()
