"""系统取消了未运行的依赖任务后，恢复同一两组实验，保留全部历史JobID。"""
import datetime
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import uuid

ROOT=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-r64-lrpair-20261013')


def main():
    assert pwd.getpwuid(os.getuid()).pw_name=='xl6775' and 'torch' in socket.gethostname().lower()
    current=ROOT/'submissions/current_pipeline_submission.json'
    if current.exists():raise FileExistsError('已有恢复提交清单，先查已有JobID，不重复提交')
    repair=ROOT/'submissions/source_repair_submission.json'
    if repair.exists():raise FileExistsError('发现此前修复回执，先恢复其已有JobID，不重复提交')
    q=subprocess.run(['sacct','-X','-nP','-j','19717659,19717660,19717661,19717662','-o','JobID,State,ElapsedRaw,Reason,ExitCode'],text=True,capture_output=True,check=True,timeout=30)
    history={line.split('|')[0]:line.split('|') for line in q.stdout.splitlines() if line.strip()}
    assert history['19717659'][1]=='FAILED'
    for job in ['19717660','19717661','19717662']:
        assert history[job][1].startswith('CANCELLED') and int(history[job][2])==0,history[job]
    original=json.loads((ROOT/'submissions/pair_submission.json').read_text())
    old_task=Path(original['task_dir']);c=json.loads((ROOT/'source_workflow.json').read_text())
    python=c['train_python'];script=(old_task/'prepare.sbatch').read_text()
    line=f'"{python}" -u "{ROOT}/prepare_full120.py" --root "{ROOT}"'
    assert script.count(line)==1
    script=script.replace(line,f'"{python}" -u "{ROOT}/repair_sources.py" --root "{ROOT}"\n'+line)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=ROOT/'submissions'/('restored_'+stamp);task.mkdir(exist_ok=False)
    (task/'prepare.sbatch').write_text(script)
    for name in ['runtime_check','lr1e5','lr5e6']:
        (task/(name+'.sbatch')).write_text((old_task/(name+'.sbatch')).read_text())
    for p in task.glob('*.sbatch'):subprocess.run(['bash','-n',str(p)],check=True)
    for p in ROOT.glob('*.py'):compile(p.read_text(),str(p),'exec')
    assoc=subprocess.check_output(['sacctmgr','-nP','show','assoc','where','user=xl6775','format=User,Account,QOS'],text=True,timeout=30)
    assert c['slurm']['account'] in assoc
    record={'status':'submitting','created_at_utc':stamp,'root':str(ROOT),'task_dir':str(task),'jobs':{},
        'previous_submission':str(old_task/'submission.json'),'previous_states':history,
        'restoration_basis':'19717659公开源短片段导致准备失败；三个GPU依赖任务由系统取消且均0运行时；用户已批准同日完整窗口替换，维持120/600/120',
        'experiment_count':2,'formal_steps_per_experiment':300,'formal_lrs':[1e-5,5e-6],
        'runtime_checks_executed_previously':0,'runtime_checks_planned':1,'no_old_job_cancelled_by_this_script':True}
    def save():
        text=json.dumps(record,ensure_ascii=False,indent=2);(task/'submission.json').write_text(text);current.write_text(text)
    save()
    def submit(name,dependency=None):
        cpu=name=='prepare';resources=c['slurm']
        command=['sbatch','--parsable','--job-name=egoqa-full120-restored-'+name,'--account='+resources['account'],
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
        item={'job_id':job,'dependency':dependency,'command':command,'stdout':str(task/(name+'-'+job+'.out')),
            'stderr':str(task/(name+'-'+job+'.err'))}
        if cpu:
            item.update(output_dir=str(ROOT/'outputs'/('prepare_'+job)),source_repair_output=str(ROOT/'outputs'/('source_repair_'+job)))
        else:
            conf=json.loads((ROOT/'configs'/(name+'.json')).read_text())
            item.update(config=str(ROOT/'configs'/(name+'.json')),output_dir=str(Path(conf['output_root'])/('train_'+job)))
        record['jobs'][name]=item;save();print('RESTORED_JOB_SUBMITTED',name,job,flush=True)
        return job
    try:
        data=submit('prepare');runtime=submit('runtime_check',data)
        submit('lr1e5',runtime);submit('lr5e6',runtime)
        record['status']='submitted';save();print('RESTORED_PIPELINE_RESULT',json.dumps(record,ensure_ascii=False),flush=True)
    except BaseException as exc:
        record.update(status='submission_failed',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
