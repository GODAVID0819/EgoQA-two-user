"""CPU验证通过后只提交一组fresh 360步训练，立即保存准确JobID。"""
from pathlib import Path
from datetime import datetime, timezone
import json
import subprocess
import uuid
import argparse

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['capacity','stage180'],required=True)
    phase=parser.parse_args().phase
    config_path=ROOT/'configs'/('capacity.json' if phase=='capacity' else 'train.json');c=json.loads(config_path.read_text())
    marker=ROOT/'submissions'/('capacity_submission.json' if phase=='capacity' else 'current_training_submission.json')
    if marker.exists():raise RuntimeError('已有正式提交回执；先核验，禁止重复提交')
    cpu=json.loads((ROOT/'preflight/cpu_validation_complete.json').read_text())
    geometry=json.loads((ROOT/'preflight/source_groundedness_adapter_geometry.json').read_text())
    pool=json.loads((Path(c['history_pool_directory'])/'pool_metadata.json').read_text())
    if cpu['status']!='passed' or cpu.get('historical_rescoring_performed') is not False:
        raise ValueError('最新CPU检查必须包含直接旧分数历史池，不能使用旧检查记录')
    cpu_receipt=json.loads((ROOT/'submissions/cpu_validation_submission.json').read_text())
    from capacity_acceptance import terminal_success,accept_capacity
    terminal_success(cpu_receipt['job_id'])
    if (cpu['job_id']!=cpu_receipt['job_id'] or cpu.get('actual_conditions')!=720 or cpu.get('deleted_rows')!=0
            or cpu.get('truncated_rows')!=0 or cpu.get('tests_passed',0)<8
            or not cpu.get('phase_resume_fixture_passed') or set(cpu.get('startup_import_roles',{}))!={'train','judge'}):
        raise ValueError('本轮CPU实际输入/阶段恢复/两角色闭包检查缺失')
    if geometry['status']!='passed' or geometry['tensor_count']!=248:raise ValueError('E全部张量几何未核对')
    if pool.get('score_source')!='old_training_F_E_A_S' or pool.get('historical_rescoring_performed') is not False or pool.get('selected_QA')!=251:
        raise ValueError('用户已禁止历史重评分')
    if (c['max_steps']!=360 or c['learning_rate']!=1e-5 or c.get('resume_from_checkpoint') is not None
            or c['train_conditions']!=(2 if phase=='capacity' else 720) or c['validation_conditions']!=0 or c['eval_strategy']!='no'
            or c['stop_after_steps']!=(1 if phase=='capacity' else 180) or c['max_pixels']!=50176 or c['max_length']!=131072
            or c['slurm']['mem']!='750G' or c['slurm']['cpus']!=32):
        raise ValueError('生效配置不符合已确认360步/1e-5/fresh/720全训练')
    from dual_runtime import validate_config
    validate_config(c)
    if phase=='capacity':
        if not c.get('capacity_measurement'):raise ValueError('测量配置必须明确隔离')
    else:
        if c.get('capacity_measurement') or c['walltime_seconds']!=216000:raise ValueError('正式阶段必须180步/原360调度，资源上限60h')
        capacity,budget=accept_capacity(ROOT,c)
    tools=json.loads((ROOT/'preflight/media_tools.json').read_text())
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=ROOT/'submissions'/('train_'+stamp);task.mkdir()
    script=task/'train.sbatch'
    lines=['#!/usr/bin/env bash','set -euo pipefail',
           'export PYTHONPATH="'+str(ROOT/'multi-user')+':'+str(ROOT)+'"',
           'export PYTHONDONTWRITEBYTECODE=1','export PYTHONUNBUFFERED=1',
           'export VLLM_NO_USAGE_STATS=1','export TOKENIZERS_PARALLELISM=false',
           'export PATH="'+str(Path(tools['ffmpeg']).parent)+':${PATH}"',
           'export LD_LIBRARY_PATH="'+tools['ffmpeg_lib']+':${LD_LIBRARY_PATH:-}"',
           'export HF_HUB_OFFLINE=1','export HF_DATASETS_OFFLINE=1',
           c['train_python']+' '+str(ROOT/'dual_launch.py')+' --config '+str(config_path)]
    script.write_text('\n'.join(lines)+'\n');subprocess.run(['bash','-n',str(script)],check=True)
    record={'status':'submitting','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'root':str(ROOT),'task_dir':str(task),'config':str(config_path),'CPU_preflight_job':cpu['job_id'],
            'max_steps':360,'stop_after_steps':c['stop_after_steps'],'phase':phase,'learning_rate':1e-5,'fresh_LoRA':True,'train_conditions':c['train_conditions'],
            'validation_conditions':0,'history_QA':pool['selected_QA'],'historical_rescoring_performed':False,
            'old_job_modified':False,'GPU_count':2}
    def save():
        for path in [task/'submission.json',marker]:
            temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(record,ensure_ascii=False,indent=2));temporary.replace(path)
    save()
    if phase=='stage180':record.update(capacity_job_id=capacity['job_id'],walltime_basis=budget)
    seconds=7200 if phase=='capacity' else c['walltime_seconds'];days,seconds=divmod(seconds,86400);hours,seconds=divmod(seconds,3600);minutes,seconds=divmod(seconds,60)
    walltime=f'{days}-{hours:02d}:{minutes:02d}:{seconds:02d}'
    command=['sbatch','--parsable','--job-name=egoqa-b50176-'+phase,'--account='+c['slurm']['account'],
             '--nodes=1','--ntasks=1','--cpus-per-task=32','--mem=750G','--gres=gpu:2','--constraint=h200',
             '--time='+walltime,'--output='+str(task/'train-%j.out'),'--error='+str(task/'train-%j.err'),
             '--chdir='+str(ROOT),str(script)]
    record['command']=command;save()
    result=subprocess.run(command,capture_output=True,text=True,timeout=60)
    (task/'sbatch.stdout').write_text(result.stdout);(task/'sbatch.stderr').write_text(result.stderr)
    result.check_returncode();job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('提交结果未知，保留回执，禁止原样重投')
    record.update(status='submitted',job_id=job,output_dir=str(Path(c['output_root'])/('train_'+job)),
                  stdout=str(task/('train-'+job+'.out')),stderr=str(task/('train-'+job+'.err')))
    save();print('TRAINING_SUBMITTED',json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
