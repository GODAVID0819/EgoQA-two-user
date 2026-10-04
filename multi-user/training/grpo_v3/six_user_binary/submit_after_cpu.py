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


def direct_submit_command(config, workflow_path, task_dir):
    from .workflow import training_config
    from .submit import sbatch_command
    if config.get('execution_mode') != 'direct':
        raise ValueError('必须提交直接执行训练的作业')
    run=training_config(config,job_id='0',phase='formal',max_steps=config['formal_max_steps'])
    run.update(slurm=config['slurm'],walltime_basis=config['walltime_basis'])
    return sbatch_command(run,str(workflow_path),str(task_dir))


def verify_checks(directory, *, require_checkpoint=False):
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
    if reports['train_data']['checks']['rows']!={'train':18,'validation':6}:
        raise ValueError('训练/验证数量不符')
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
    if c.get('formal_max_steps')!=60 or not c.get('paired_validation') or c.get('execution_mode')!='direct':
        raise ValueError('本次提交必须为60步及固定验证对比')
    root=Path(c['project_root'])
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=root/'submissions'/('grpo60_'+stamp)
    task.mkdir(parents=True,exist_ok=False)
    with (args.preflight/'submission.claim').open('x') as f:f.write(str(task))
    record={'status':'waiting_for_cpu_validation','task_dir':str(task),'preflight_dir':str(args.preflight),
            'config':c,'cpu_preflight_job_id':args.cpu_job_id,'failed_job_ids':['18719659','18790400'],'cancelled_job_id':'18837514',
            'data_provenance_job_id':'18719659','created_at_utc':stamp}
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
        verify_checks(args.preflight,require_checkpoint=bool(c.get('resume_from_checkpoint')))
        if json.loads(args.config.read_text())!=c:raise ValueError('检查期间配置发生变化，不提交GPU')
        if subprocess.check_output(['whoami'],text=True).strip()!='xl6775':raise RuntimeError('用户身份不符')
        assoc=subprocess.check_output(['sacctmgr','-nP','show','assoc','user=xl6775','format=User,Account,Partition,QOS'],text=True)
        if c['slurm']['account'] not in assoc:raise RuntimeError('当前account不可用')
        partition=subprocess.check_output(['scontrol','show','partition',c['slurm']['partition']],text=True)
        qos=subprocess.check_output(['sacctmgr','-nP','show','qos',c['slurm']['qos'],'format=Name,MaxTRESPU,GrpTRES'],text=True)
        if 'State=UP' not in partition or c['slurm']['qos'] not in qos:raise RuntimeError('partition或QOS不可用')
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
