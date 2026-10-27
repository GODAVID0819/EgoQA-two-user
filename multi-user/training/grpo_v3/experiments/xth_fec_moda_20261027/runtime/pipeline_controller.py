"""在Slurm CPU依赖作业中串联已授权的验证和提交，不重试失败或恢复旧监控。"""
import argparse
from datetime import datetime,timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import traceback
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parent


def record(**values):
    values.update(updated_at_utc=datetime.now(timezone.utc).isoformat(),controller_job_id=os.environ.get('SLURM_JOB_ID'))
    p=ROOT/'preflight/pipeline_state.json';temporary=p.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(values,ensure_ascii=False,indent=2));temporary.replace(p)
    print('PIPELINE_STATE',json.dumps(values,ensure_ascii=False),flush=True)


def check_policy():
    url='https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/'
    class Text(HTMLParser):
        def __init__(self):super().__init__();self.parts=[]
        def handle_data(self,value):self.parts.append(value)
    content=urllib.request.urlopen(url,timeout=45).read().decode('utf-8')
    parsed=Text();parsed.feed(content);text=' '.join(' '.join(parsed.parts).split())
    if re.search(r'gh\*\s+60%\s+75%',text) is None:
        raise RuntimeError('官网当前gh阈值无法核实为60%/75%；不自动改保护或提交GPU')
    result={'url':url,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'gh_cancel':60,'gh_warning':75}
    (ROOT/'preflight'/('policy_'+os.environ['SLURM_JOB_ID']+'.json')).write_text(json.dumps(result,indent=2))


def submit_controller(stage,previous):
    marker=ROOT/'submissions'/('controller_'+stage+'.json')
    if marker.exists():raise RuntimeError('控制器已有回执，禁止重复提交')
    c=json.loads((ROOT/'configs/train.json').read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=ROOT/'submissions'/('controller_'+stage+'_'+stamp);task.mkdir()
    script=task/'controller.sbatch'
    script.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexport CUDA_VISIBLE_DEVICES=""\nexport PYTHONDONTWRITEBYTECODE=1\nexport PYTHONUNBUFFERED=1\n'+
        c['train_python']+' '+str(ROOT/'pipeline_controller.py')+' --stage '+stage+' --previous '+str(previous)+'\n')
    subprocess.run(['bash','-n',str(script)],check=True)
    command=['sbatch','--parsable','--job-name=egoqa-b50176-control','--account='+c['slurm']['account'],
        '--dependency=afterany:'+str(previous),'--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=8G','--time=00:10:00',
        '--output='+str(task/'controller-%j.out'),'--error='+str(task/'controller-%j.err'),'--chdir='+str(ROOT),str(script)]
    receipt={'status':'submitting','stage':stage,'previous_job_id':str(previous),'task_dir':str(task),'command':command}
    def save():
        for path in [marker,task/'submission.json']:
            temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(receipt,indent=2));temporary.replace(path)
    save();result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (task/'sbatch.stdout').write_text(result.stdout);(task/'sbatch.stderr').write_text(result.stderr)
    result.check_returncode();job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('控制器提交结果未知，先恢复回执')
    receipt.update(status='submitted',job_id=job,stdout=str(task/('controller-'+job+'.out')),stderr=str(task/('controller-'+job+'.err')))
    save();print('CONTROLLER_SUBMITTED',json.dumps(receipt),flush=True)
    return receipt


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['after_cpu','after_capacity'],required=True)
    parser.add_argument('--previous',required=True);args=parser.parse_args()
    if not args.previous.isdigit() or not os.environ.get('SLURM_JOB_ID') or os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise ValueError('控制器只能在显式无GPU的本轮Slurm依赖作业执行')
    replacement=ROOT/'submissions/combined_replacement_requested.json'
    if args.stage=='after_capacity' and replacement.exists():
        request=json.loads(replacement.read_text())
        if (request.get('legacy_capacity_job_id')!=args.previous
                or request.get('legacy_controller_job_id')!=os.environ['SLURM_JOB_ID']):
            raise ValueError('合并替换请求不属于当前控制器')
        record(status='superseded_no_submission',previous_job_id=args.previous,
            reason='用户已改为同一allocation测量后fresh180；旧控制器不再提交')
        return
    sys.path[:0]=[str(ROOT/'multi-user'),str(ROOT)]
    from training.torch_storage_preflight import REQUIRED_STORAGE_VARIABLES,validate_storage_environment
    scratch=Path('/scratch/xl6775/job_scratch')/('b50176_controller_'+os.environ['SLURM_JOB_ID'])
    for name in REQUIRED_STORAGE_VARIABLES+('PIP_CACHE_DIR',):os.environ[name]=str(scratch/name.lower())
    if validate_storage_environment(allowed_root=scratch,environ=os.environ)['status']!='passed':raise RuntimeError('CPU控制器scratch封闭失败')
    from capacity_acceptance import terminal_success
    terminal_success(args.previous)
    check_policy()
    c=json.loads((ROOT/'configs/train.json').read_text())
    phase='capacity' if args.stage=='after_cpu' else 'stage180'
    source_marker=ROOT/'submissions'/('cpu_validation_submission.json' if args.stage=='after_cpu' else 'capacity_submission.json')
    if json.loads(source_marker.read_text())['job_id']!=args.previous:raise ValueError('前级JobID不是当前准确回执')
    record(status='submitting_'+phase,previous_job_id=args.previous)
    subprocess.run([c['train_python'],str(ROOT/'submit_training.py'),'--phase',phase],check=True)
    marker=ROOT/'submissions'/('capacity_submission.json' if phase=='capacity' else 'current_training_submission.json')
    receipt=json.loads(marker.read_text())
    if phase=='capacity':
        next_controller=submit_controller('after_capacity',receipt['job_id'])
        record(status='capacity_submitted_waiting',capacity_job_id=receipt['job_id'],next_controller_job_id=next_controller['job_id'])
    else:
        record(status='stage180_submitted',formal_job_id=receipt['job_id'],capacity_job_id=args.previous)


if __name__=='__main__':
    try:main()
    except BaseException as error:
        record(status='needs_attention_no_retry',error=repr(error),traceback=traceback.format_exc())
        raise
