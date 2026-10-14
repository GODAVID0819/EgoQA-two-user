"""按已确认的最小新增重叠策略恢复原两组300步实验，保留旧JobID。"""
import datetime
import json
from pathlib import Path
import pwd
import os
import socket
import subprocess
import uuid

ROOT=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-r64-lrpair-20261013')
POLICY='2026-10-14用户确认：允许最小额外重叠，保持120/600/120；重叠连通组留出同一speaker，不补帧，不继承旧QA'


def main():
    assert pwd.getpwuid(os.getuid()).pw_name=='xl6775' and 'torch' in socket.gethostname().lower()
    marker=ROOT/'submissions/overlap_policy_submission.json'
    if marker.exists():raise FileExistsError('已有该数据政策恢复清单，先核查已有JobID，禁止重复投递')
    current=ROOT/'submissions/current_pipeline_submission.json'
    previous=json.loads(current.read_text())
    expected={'prepare':'19722680','runtime_check':'19722681','lr1e5':'19722682','lr5e6':'19722683'}
    assert {k:v['job_id'] for k,v in previous['jobs'].items()}==expected
    q=subprocess.run(['sacct','-X','-nP','-j',','.join(expected.values()),'-o','JobID,State,ElapsedRaw,Reason,ExitCode'],capture_output=True,text=True,check=True,timeout=30)
    history={line.split('|')[0]:line.split('|') for line in q.stdout.splitlines() if line.strip()}
    assert history[expected['prepare']][1]=='FAILED'
    for name in ['runtime_check','lr1e5','lr5e6']:
        row=history[expected[name]];assert row[1].startswith('CANCELLED') and int(row[2])==0,row
    assert not (ROOT/'dataset_ready.json').exists()
    audit=ROOT/'outputs/source_repair_19722680/source_media_audit.jsonl'
    assert audit.is_file() and audit.stat().st_size>0
    c=json.loads((ROOT/'source_workflow.json').read_text());python=c['train_python']
    resources=c['slurm'];assert resources['gres']=='gpu:1' and resources['constraint']=='h200'
    old_task=Path(previous['task_dir']);script=(old_task/'prepare.sbatch').read_text()
    repair_line=f'"{python}" -u "{ROOT}/repair_sources.py" --root "{ROOT}"'
    assert script.count(repair_line)==1
    script=script.replace(repair_line,repair_line+f' --reuse-audit-output "{audit.parent}"')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=ROOT/'submissions'/('overlap_policy_'+stamp);task.mkdir(exist_ok=False)
    (task/'prepare.sbatch').write_text(script)
    for name in ['runtime_check','lr1e5','lr5e6']:
        (task/(name+'.sbatch')).write_text((old_task/(name+'.sbatch')).read_text())
    for p in task.glob('*.sbatch'):subprocess.run(['bash','-n',str(p)],check=True)
    for name in ['repair_sources.py','recover_overlap_pipeline.py']:
        p=ROOT/name;compile(p.read_text(),str(p),'exec')
    assoc=subprocess.check_output(['sacctmgr','-nP','show','assoc','where','user=xl6775','format=User,Account,QOS'],text=True,timeout=30)
    assert resources['account'] in assoc
    record={'status':'submitting','created_at_utc':stamp,'root':str(ROOT),'task_dir':str(task),'jobs':{},
        'previous_submission':str(old_task/'submission.json'),'previous_states':history,'user_approved_policy':POLICY,
        'source_audit_reused':str(audit),'experiment_count':2,'formal_steps_per_experiment':300,
        'formal_lrs':[1e-5,5e-6],'runtime_checks_executed_previously':0,'runtime_checks_planned':1,
        'no_old_job_cancelled_by_this_script':True,
        'gpu_policy_checked_utc':'2026-10-14T00:13Z','gpu_policy_url':'https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/',
        'gh_cancellation_percent':60,'gh_warning_percent':75,'keeper_target_percent':80}
    def save():
        text=json.dumps(record,ensure_ascii=False,indent=2)
        for p in [task/'submission.json',marker,current]:
            tmp=p.with_suffix('.json.tmp');tmp.write_text(text);tmp.replace(p)
    save()
    def submit(name,dependency=None):
        cpu=name=='prepare'
        command=['sbatch','--parsable','--job-name=egoqa-full120-overlap-'+name,'--account='+resources['account'],
            '--nodes=1','--ntasks=1','--cpus-per-task='+('32' if cpu else '16'),
            '--mem='+('96G' if cpu else resources['mem']),
            '--time='+('08:00:00' if cpu else '02:00:00' if name=='runtime_check' else '7-00:00:00'),
            '--output='+str(task/(name+'-%j.out')),'--error='+str(task/(name+'-%j.err')),'--chdir='+str(ROOT)]
        if not cpu:command+=['--gres='+resources['gres'],'--constraint='+resources['constraint']]
        if dependency:command+=['--dependency=afterok:'+dependency]
        command.append(str(task/(name+'.sbatch')))
        result=subprocess.run(command,capture_output=True,text=True,timeout=60)
        (task/(name+'.sbatch.stdout')).write_text(result.stdout);(task/(name+'.sbatch.stderr')).write_text(result.stderr)
        result.check_returncode();job=result.stdout.strip().split(';')[0];assert job.isdigit()
        item={'job_id':job,'dependency':dependency,'command':command,
            'stdout':str(task/(name+'-'+job+'.out')),'stderr':str(task/(name+'-'+job+'.err'))}
        if cpu:item.update(output_dir=str(ROOT/'outputs'/('prepare_'+job)),source_repair_output=str(ROOT/'outputs'/('source_repair_'+job)))
        else:
            conf=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            item.update(config=str(ROOT/'configs'/(name+'.json')),output_dir=str(Path(conf['output_root'])/('train_'+job)))
        record['jobs'][name]=item;save();print('OVERLAP_JOB_SUBMITTED',name,job,flush=True)
        return job
    try:
        data=submit('prepare');runtime=submit('runtime_check',data)
        submit('lr1e5',runtime);submit('lr5e6',runtime)
        record['status']='submitted';save();print('OVERLAP_PIPELINE_RESULT',json.dumps(record,ensure_ascii=False),flush=True)
    except BaseException as exc:
        record.update(status='submission_failed',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
