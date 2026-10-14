"""重提CPU准备并修改现有未运行的GPU检查依赖，不重复提交正式任务。"""
import datetime
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess

ROOT=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-r64-lrpair-20261013')


def main():
    assert pwd.getpwuid(os.getuid()).pw_name=='xl6775' and 'torch' in socket.gethostname().lower()
    current=ROOT/'submissions/source_repair_submission.json'
    if current.exists():raise FileExistsError('已有修复提交清单，先查已有JobID，不重复提交')
    p=subprocess.check_output(['sacct','-X','-nP','-j','19717659','-o','State,ExitCode'],text=True)
    assert 'FAILED|1:0' in p,p
    for job in ['19717660','19717661','19717662']:
        status=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
        assert 'JobState=PENDING' in status,status
    original=json.loads((ROOT/'submissions/pair_submission.json').read_text())
    source_script=Path(original['task_dir'])/'prepare.sbatch';script=source_script.read_text()
    c=json.loads((ROOT/'source_workflow.json').read_text());python=c['train_python']
    line=f'"{python}" -u "{ROOT}/prepare_full120.py" --root "{ROOT}"'
    assert script.count(line)==1
    script=script.replace(line,f'"{python}" -u "{ROOT}/repair_sources.py" --root "{ROOT}"\n'+line)
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    task=ROOT/'submissions'/('source_repair_'+stamp);task.mkdir(exist_ok=False)
    path=task/'prepare_repaired.sbatch';path.write_text(script);subprocess.run(['bash','-n',str(path)],check=True)
    for name in ['repair_sources.py','media_quality.py','submit_pair.py']:
        p=ROOT/name;compile(p.read_text(),str(p),'exec')
    command=['sbatch','--parsable','--job-name=egoqa-full120-source-repair','--account='+c['slurm']['account'],
        '--nodes=1','--ntasks=1','--cpus-per-task=32','--mem=96G','--time=08:00:00',
        '--output='+str(task/'repair-%j.out'),'--error='+str(task/'repair-%j.err'),'--chdir='+str(ROOT),str(path)]
    record={'status':'submitting','created_at_utc':stamp,'failed_job_id':'19717659','task_dir':str(task),'command':command,
        'user_approved_replacement':True,'formal_job_ids_unchanged':['19717661','19717662'],'runtime_check_job_id_unchanged':'19717660',
        'basis':'复用19717659已下载有效文件；同一8小时CPU数据预算补全审计、同日完整窗替换、完整帧与r64 CPU审计；不补帧、不减少120/600/120'}
    def save():
        text=json.dumps(record,ensure_ascii=False,indent=2);(task/'submission.json').write_text(text);current.write_text(text)
    save()
    result=subprocess.run(command,capture_output=True,text=True)
    (task/'sbatch.stdout').write_text(result.stdout);(task/'sbatch.stderr').write_text(result.stderr);result.check_returncode()
    job=result.stdout.strip().split(';')[0];assert job.isdigit()
    record.update(status='submitted',job_id=job,stdout=str(task/('repair-'+job+'.out')),stderr=str(task/('repair-'+job+'.err')),
        source_repair_output=str(ROOT/'outputs'/('source_repair_'+job)),data_output=str(ROOT/'outputs'/('prepare_'+job)))
    save();print('SOURCE_REPAIR_CPU_SUBMITTED',job,flush=True)
    update=['scontrol','update','JobId=19717660','Dependency=afterok:'+job]
    result=subprocess.run(update,capture_output=True,text=True);record.update(dependency_update=update,dependency_update_exit=result.returncode,
        dependency_update_stdout=result.stdout,dependency_update_stderr=result.stderr);save();result.check_returncode()
    print('SOURCE_REPAIR_SUBMISSION_RESULT',json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
