"""按最新授权零GPU预检，直接提交一个1e-5 fresh基座300步正式任务。"""
import datetime
import argparse
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import uuid

ROOT=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-ftjudge-dual-speed-v5-20261016')
OLD=ROOT.parent/'full120-r64-lrpair-20261013'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--replace-zero-runtime-formals',nargs=2)
    args=parser.parse_args()
    assert pwd.getpwuid(os.getuid()).pw_name=='xl6775' and 'torch' in socket.gethostname().lower()
    marker=ROOT/'submissions/current_pipeline_submission.json'
    actual_source=subprocess.check_output(['sacct','-X','-nP','-j','19879021,19879022,19884436,19926433,19938940','-o','JobID,State,ElapsedRaw,ExitCode'],text=True)
    source_rows={row[0]:row for line in actual_source.splitlines() if (row:=line.split('|'))[0] in {'19879021','19879022','19884436','19926433','19938940'}}
    if set(source_rows)!={'19879021','19879022','19884436','19926433','19938940'}:raise ValueError('源任务状态不完整，禁止重复提交')
    if source_rows['19879021'][1].split()[0] not in {'OUT_OF_MEMORY','FAILED','CANCELLED'}:raise ValueError('源1e-5尚非失败终态')
    if source_rows['19879022'][1].split()[0] not in {'CANCELLED','FAILED','OUT_OF_MEMORY','TIMEOUT'}:raise ValueError('尚未获准/完成停止旧5e-6，禁止重复任务')
    if source_rows['19884436'][1].split()[0]!='FAILED':raise ValueError('19884436非已确认失败终态')
    evidence=json.loads((ROOT/'preflight/import_fix_cpu.json').read_text())
    if evidence.get('status')!='passed':raise ValueError('实际Judge包路径CPU回归未通过')
    if source_rows['19926433'][1].split()[0]!='FAILED':raise ValueError('19926433尚非失败终态')
    evidence=json.loads((ROOT/'preflight/gradient_sync_cpu.json').read_text())
    if evidence.get('status')!='passed':raise ValueError('真实双CPU梯度均值和优化器状态回归未通过')
    if source_rows['19938940'][1].split()[0]!='FAILED':raise ValueError('19938940尚非失败终态')
    evidence=json.loads((ROOT/'preflight/guard_resume_cpu.json').read_text())
    if evidence.get('status')!='passed':raise ValueError('Guard竞态与完整状态恢复CPU回归未通过')
    previous=None
    if marker.exists():
        if not args.replace_zero_runtime_formals:
            raise FileExistsError('已有双卡提交结果，必须读回执而非重新投递')
        from dual_runtime import verify_zero_runtime_replacement
        previous=json.loads(marker.read_text())
        actual=subprocess.check_output(['sacct','-X','-nP','-j',','.join(args.replace_zero_runtime_formals),
                                        '-o','JobID,State,ElapsedRaw,ExitCode'],text=True)
        verify_zero_runtime_replacement(previous,args.replace_zero_runtime_formals,actual)
    elif args.replace_zero_runtime_formals:
        raise FileNotFoundError('找不到待替换的当前回执，禁止猜测提交状态')
    audit=json.loads((ROOT/'preflight/length_pair_sampler.json').read_text())
    if audit.get('status')!='passed':raise ValueError('实际CPU双rank采样审计未通过')
    for name in ['answerability_geometry.json','formality_geometry.json','mixed_configuration.json','speed_data_cpu.json','cpu_speed_result.json']:
        evidence=json.loads((ROOT/'preflight'/name).read_text())
        if evidence.get('status')!='passed':raise ValueError('Judge CPU preparation failed: '+name)
    for name in ['lr1e5']:
        config=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        from dual_runtime import validate_config
        validate_config(config)
        judge=json.loads(Path(config['judge_config']).read_text())
        expected={'formality':str(ROOT.parent/'full120-ftjudge-dual-r64-lrpair-20261015/adapters/formality_checkpoint-504'),
                  'groundedness':None,'answerability':str(ROOT.parent/'full120-ftjudge-dual-r64-lrpair-20261015/adapters/answerability_checkpoint-340')}
        if judge.get('judge_mode')!='mixed' or judge.get('adapters')!=expected:
            raise ValueError('Wrong mixed Judge mapping')
    source=json.loads((OLD/'source_workflow.json').read_text())
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=ROOT/'submissions'/('dual_'+stamp);task.mkdir()
    if previous is not None:
        (task/'previous_pipeline_submission.json').write_text(json.dumps(previous,indent=2))
    record={'status':'submitting','root':str(ROOT),'task_dir':str(task),'created_at_utc':stamp,'jobs':{},
            'old_jobs':['19879021','19879022','19884436','19926433','19938940'],'old_source_preserved':True,'data_job_reused':'19754024',
            'world_size':2,'conditions_per_update':2,'G':4,'global_candidates_per_update':8,
            'formal_steps':300,'formal_lrs':[1e-5],'fresh_base':False,'initial_fresh_base':True,'inherited_global_steps':12,'remaining_optimizer_steps':288,'resume_from_checkpoint':config['resume_from_checkpoint'],'training_target_hours':48,
            'formal_allocation_hours':72,'host_memory':config['slurm']['mem'],'speed_version':5,'source_states':actual_source,'runtime_checks_planned':0,'gpu_smoke_user_disabled':True,'judge_mapping':{'formality':504,'answerability':340,'groundedness':'base'},
            'gpu_policy_url':'https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/',
            'gh_cancel_percent':60,'gh_warning_percent':75,'keeper_target':85}
    if previous is not None:
        record.update(previous_task_dir=previous['task_dir'],
                      superseded_zero_runtime_formals=args.replace_zero_runtime_formals,
                      user_cancelled_gpu_check='19877285')
    def save():
        text=json.dumps(record,indent=2)
        for p in [task/'submission.json',marker]:
            tmp=p.with_suffix('.json.tmp');tmp.write_text(text);tmp.replace(p)
    save()
    template='''#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="__ROOT__"
export PYTHONPATH="__ROOT__/multi-user:__ROOT__"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PATH="__FFMPEG_DIR__:${PATH}"
export LD_LIBRARY_PATH="__FFMPEG_LIB__:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
__PYTHON__ __ROOT__/dual_launch.py --config __ROOT__/configs/__NAME__.json
'''
    for name in ['lr1e5']:
        c=json.loads((ROOT/'configs'/(name+'.json')).read_text())
        text=template.replace('__ROOT__',str(ROOT)).replace('__NAME__',name).replace('__PYTHON__',c['train_python'])
        text=text.replace('__FFMPEG_DIR__',str(Path(source['ffmpeg']).parent))
        text=text.replace('__FFMPEG_LIB__',str(Path(source['ffmpeg']).parent.parent/'lib'))
        script=task/(name+'.sbatch');script.write_text(text)
        subprocess.run(['bash','-n',str(script)],check=True)
        resource=c['slurm'];dependency=None
        cmd=['sbatch','--parsable','--job-name=egoqa-full120-ftjudge-'+name,'--account='+resource['account'],
             '--nodes=1','--ntasks=1','--cpus-per-task=32','--mem='+resource['mem'],'--gres=gpu:2','--constraint=h200',
             '--time=3-00:00:00',
             '--output='+str(task/(name+'-%j.out')),'--error='+str(task/(name+'-%j.err')),'--chdir='+str(ROOT)]
        if dependency:cmd+=['--dependency=afterok:'+dependency]
        cmd.append(str(script))
        try:
            q=subprocess.run(cmd,text=True,capture_output=True,timeout=60)
            (task/(name+'.sbatch.stdout')).write_text(q.stdout);(task/(name+'.sbatch.stderr')).write_text(q.stderr)
            q.check_returncode();job=q.stdout.strip().split(';')[0]
            if not job.isdigit():raise ValueError('无准确sbatch JobID')
            record['jobs'][name]={'job_id':job,'dependency':dependency,'command':cmd,
                'config':str(ROOT/'configs'/(name+'.json')),'output_dir':str(Path(c['output_root'])/('train_'+job)),
                'stdout':str(task/(name+'-'+job+'.out')),'stderr':str(task/(name+'-'+job+'.err'))}
            save();print('DUAL_JOB_SUBMITTED',name,job,flush=True)
        except BaseException as exc:
            record.update(status='submission_failed',error=f'{type(exc).__name__}: {exc}');save();raise
    record['status']='submitted';save();print('DUAL_SUBMISSION_RESULT',json.dumps(record),flush=True)

if __name__=='__main__':main()
