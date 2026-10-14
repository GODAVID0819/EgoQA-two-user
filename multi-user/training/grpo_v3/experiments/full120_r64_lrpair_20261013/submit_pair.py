"""一次提交共用CPU准备、唯一r64运行检查和两组300步正式任务。"""
import datetime
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import sys

ROOT=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-r64-lrpair-20261013')


def main():
    assert pwd.getpwuid(os.getuid()).pw_name=='xl6775' and 'torch' in socket.gethostname().lower()
    if (ROOT/'submissions/pair_submission.json').exists():
        raise FileExistsError('本轮已有提交清单；先按清单查状态，不重复提交')
    sys.path.insert(0,str(ROOT/'multi-user'))
    from training.grpo_v3.six_user_binary.workflow import training_config
    from training.grpo_v3.six_user_binary.launch import swift_command,validate_config
    from training.grpo_v3.six_user_binary import launch
    source=json.loads((ROOT/'source_workflow.json').read_text())
    workflow=dict(source,project_root=str(ROOT),prepared_data_root=str(ROOT/'data/trainval'),
        formal_max_steps=300,stop_after_steps=300,eval_steps=300,lora_rank=64,lora_alpha=128,
        per_device_train_batch_size=1,gradient_accumulation_steps=16,walltime_seconds=604800,
        learning_rate=1e-5,resume_from_checkpoint=None,paired_validation=True,baseline_after_training=True)
    workflow.pop('test_dataset',None)
    config=training_config(workflow,job_id='0',phase='formal',max_steps=300)
    config.update(dataset_shuffle=False,full120_speaker_holdout=True,speaker_holdout_validation=False,
        lora_dropout=.05,slurm=source['slurm'],expected_split_counts={'train':600,'validation':120},
        walltime_basis='旧19460878平均380.24秒/单窗口更新，四窗口300步线性约126.75小时；终点120条件框架验证约12.1小时，固定基座/适配器各120条件约13.6小时，加启动约1小时，共153.5小时。申请单作业上限168小时；新r64/四窗口实际吞吐尚未测得，估时余量受7天上限限制，保留每两步完整断点。')
    for name,lr in [('lr1e5',1e-5),('lr5e6',5e-6)]:
        c=dict(config,learning_rate=lr,output_root=str(ROOT/'outputs'/('formal_'+name)),run_attempt=name)
        validate_config(c);(ROOT/'configs'/(name+'.json')).write_text(json.dumps(c,ensure_ascii=False,indent=2))
    minimal=dict(config,train_dataset=str(ROOT/'data/trainval/runtime_train.jsonl'),
        val_dataset=str(ROOT/'data/trainval/runtime_validation.jsonl'),output_root=str(ROOT/'outputs/runtime_r64_b4'),
        full120_speaker_holdout=False,minimum_runtime_check=True,stop_after_steps=1,eval_steps=1,warmup_steps=0,
        walltime_seconds=7200,run_attempt='runtime-r64-b4',
        walltime_basis='唯一新边界检查：r64、四完整输入×G4、真实更新、保存与固定1条件adapter重载；训练按4×380秒估约25分钟，Judge/Policy启动及首次编译约60分钟，验证/重载约15分钟，共100分钟，申请120分钟。')
    validate_config(minimal);(ROOT/'configs/runtime_check.json').write_text(json.dumps(minimal,ensure_ascii=False,indent=2))
    def command_check():
        command=launch.swift_command(config,str(ROOT/'preflight/cli'))
        assert command[command.index('--dataset_shuffle')+1]=='false'
        assert command[command.index('--vllm_max_lora_rank')+1]=='64'
    try:command_check()
    except (AssertionError,ValueError):print('PRE_PATCH_CLI_TEST_EXPECTED_FAILURE',flush=True)
    subprocess.run(['python3',str(ROOT/'install_pair_runtime.py'),'--root',str(ROOT)],check=True)
    __import__('importlib').reload(launch)
    command_check();print('POST_PATCH_CLI_TEST_PASSED',flush=True)
    for p in ROOT.glob('*.py'):
        compile(p.read_text(),str(p),'exec')
    for path in [source['train_python'],source['judge_python'],source['model'],source['judge_config'],source['clip_model'],source['ffmpeg'],source['ffprobe']]:
        if not Path(path).exists():raise FileNotFoundError(path)
    assoc=subprocess.check_output(['sacctmgr','-nP','show','assoc','where','user=xl6775','format=User,Account,QOS'],text=True)
    assert source['slurm']['account'] in assoc
    qos=subprocess.check_output(['sacctmgr','-nP','show','qos','format=Name,MaxWall,MaxTRESPU'],text=True)
    assert 'gpu168|7-00:00:00' in qos
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    task=ROOT/'submissions'/('pair_'+stamp);task.mkdir(exist_ok=False)
    (task/'scheduler_check.txt').write_text(assoc+'\n'+qos)
    header='''#!/usr/bin/env bash
set -euo pipefail
export PROJECT_ROOT="__ROOT__"
export PYTHONPATH="__ROOT__/multi-user:__ROOT__"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PATH="__FFMPEG_DIR__:${PATH}"
export LD_LIBRARY_PATH="__FFMPEG_LIB__:${LD_LIBRARY_PATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
JOB_SCRATCH_ROOT="__ROOT__/job_scratch/${SLURM_JOB_ID}"
for variable in HOME XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE MODELSCOPE_CACHE TORCH_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT CUDA_CACHE_PATH FLASHINFER_WORKSPACE_BASE TMPDIR TMP TEMP PIP_CACHE_DIR; do
 export "${variable}=${JOB_SCRATCH_ROOT}/${variable,,}"
done
mkdir -p "__ROOT__/outputs/job_${SLURM_JOB_ID}"
cd "__ROOT__/multi-user"
"__TRAIN_PYTHON__" -m training.torch_storage_preflight --allowed-root "${JOB_SCRATCH_ROOT}" --output "__ROOT__/outputs/job_${SLURM_JOB_ID}/storage_preflight.json"
'''
    for key,value in {'__ROOT__':str(ROOT),'__FFMPEG_DIR__':str(Path(source['ffmpeg']).parent),
        '__FFMPEG_LIB__':str(Path(source['ffmpeg']).parent.parent/'lib'),'__TRAIN_PYTHON__':source['train_python']}.items():header=header.replace(key,value)
    cpu=header+'\nexport CUDA_VISIBLE_DEVICES=""\nexport OMP_NUM_THREADS=32\nexport MKL_NUM_THREADS=32\n'
    cpu+=f'"{source["train_python"]}" -u "{ROOT}/prepare_full120.py" --root "{ROOT}"\n'
    cpu+=f'"{source["train_python"]}" -u "{ROOT}/audit_runtime.py" --root "{ROOT}"\n'
    (task/'prepare.sbatch').write_text(cpu)
    for name in ['runtime_check','lr1e5','lr5e6']:
        script=header+f'\nexport OMP_NUM_THREADS=16\n"{source["train_python"]}" -u -m training.grpo_v3.six_user_binary.launch --config "{ROOT}/configs/{name}.json"\n'
        (task/(name+'.sbatch')).write_text(script)
    for script in task.glob('*.sbatch'):subprocess.run(['bash','-n',str(script)],check=True)
    record={'status':'submitting','created_at_utc':stamp,'root':str(ROOT),'task_dir':str(task),'jobs':{},
        'experiment_count':2,'formal_steps_per_experiment':300,'formal_lrs':[1e-5,5e-6],
        'preparation_gpu_count':0,'runtime_checks':1,'runtime_check_reused_for_both':True,
        'cpu_walltime_basis':'旧3窗口准备实测558秒，108缺失窗口线性约5.58小时；新下载体量和全量审计留约2.42小时余量，申请8小时、32CPU。',
        'boundary':'正式任务通过Slurm afterok依赖等待完整数据、CPU审计和唯一新显存边界检查；提交或排队不代表训练完成。'}
    def save():
        text=json.dumps(record,ensure_ascii=False,indent=2)
        (task/'submission.json').write_text(text);(ROOT/'submissions/pair_submission.json').write_text(text)
    save()
    def submit(name,dependency=None):
        cpu=name=='prepare';resources=source['slurm']
        command=['sbatch','--parsable','--job-name=egoqa-full120-'+name,'--account='+resources['account'],
            '--nodes=1','--ntasks=1','--cpus-per-task='+('32' if cpu else '16'),
            '--mem='+('96G' if cpu else resources['mem']),
            '--time='+('08:00:00' if cpu else '02:00:00' if name=='runtime_check' else '7-00:00:00'),
            '--output='+str(task/(name+'-%j.out')),'--error='+str(task/(name+'-%j.err')),
            '--chdir='+str(ROOT)]
        if not cpu:command+=['--gres='+resources['gres'],'--constraint='+resources['constraint']]
        if dependency:command+=['--dependency=afterok:'+dependency]
        command.append(str(task/(name+'.sbatch')))
        result=subprocess.run(command,capture_output=True,text=True)
        (task/(name+'.sbatch.stdout')).write_text(result.stdout);(task/(name+'.sbatch.stderr')).write_text(result.stderr)
        result.check_returncode();job=result.stdout.strip().split(';')[0];assert job.isdigit()
        entry={'job_id':job,'dependency':dependency,'command':command,'submitted_epoch':__import__('time').time(),
            'stdout':str(task/(name+'-'+job+'.out')),'stderr':str(task/(name+'-'+job+'.err'))}
        if not cpu:
            c=json.loads((ROOT/'configs'/(name+'.json')).read_text());entry.update(config=str(ROOT/'configs'/(name+'.json')),output_dir=str(Path(c['output_root'])/('train_'+job)))
        else:entry['output_dir']=str(ROOT/'outputs'/('prepare_'+job))
        record['jobs'][name]=entry;save();print('JOB_SUBMITTED',name,job,flush=True)
        return job
    try:
        cpu_job=submit('prepare');runtime_job=submit('runtime_check',cpu_job)
        submit('lr1e5',runtime_job);submit('lr5e6',runtime_job)
        record['status']='submitted';save();print('PAIR_SUBMISSION_RESULT',json.dumps(record,ensure_ascii=False),flush=True)
    except BaseException as exc:
        record.update(status='submission_failed',error=f'{type(exc).__name__}: {exc}');save();raise


if __name__=='__main__':main()
