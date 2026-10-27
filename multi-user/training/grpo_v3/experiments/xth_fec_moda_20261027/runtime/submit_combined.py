"""先准备并核验合并提交包，旧链退出后只提交一次新2H200/60h作业。"""
import argparse
from datetime import datetime,timezone
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import subprocess
import urllib.request
import uuid
from combined_allocation import read,save,validate_settings

ROOT=Path(__file__).resolve().parent
LEGACY_IDS={'20443244','20443249'}


def verify_legacy_terminal(accounting):
    rows={p[0]:p for line in accounting.splitlines() if (p:=line.split('|'))[0] in LEGACY_IDS}
    if set(rows)!=LEGACY_IDS or any(not p[1].startswith('CANCELLED') for p in rows.values()):
        raise ValueError('旧20443244与20443249尚未确认取消，不提交并行训练链')


def check_policy():
    url='https://services.rt.nyu.edu/docs/hpc/submitting_jobs/slurm_submitting_jobs/'
    class Text(HTMLParser):
        def __init__(self):super().__init__();self.parts=[]
        def handle_data(self,value):self.parts.append(value)
    p=Text();p.feed(urllib.request.urlopen(url,timeout=45).read().decode('utf-8'))
    text=' '.join(' '.join(p.parts).split())
    if re.search(r'gh\*\s+60%\s+75%',text) is None:raise ValueError('现场官网GPU阈值变化，停止提交')
    report={'url':url,'checked_at_utc':datetime.now(timezone.utc).isoformat(),'gh_cancel':60,'gh_warning':75}
    save(ROOT/'preflight/combined_gpu_policy.json',report)
    return report


def verify_preflight():
    formal=read(ROOT/'configs/train.json');capacity=read(ROOT/'configs/capacity.json')
    validate_settings(capacity,formal)
    from dual_runtime import validate_config
    validate_config(formal);validate_config(capacity)
    from capacity_acceptance import terminal_success
    cpu=read(ROOT/'preflight/cpu_validation_complete.json')
    receipt=read(ROOT/'submissions/cpu_validation_submission.json')
    terminal_success(receipt['job_id'])
    if (cpu.get('job_id')!=receipt['job_id'] or cpu.get('status')!='passed'
            or cpu.get('actual_conditions')!=720 or cpu.get('deleted_rows')!=0
            or cpu.get('truncated_rows')!=0 or cpu.get('tests_passed',0)<8
            or not cpu.get('phase_resume_fixture_passed')
            or set(cpu.get('startup_import_roles',{}))!={'train','judge'}
            or cpu.get('historical_rescoring_performed') is not False):
        raise ValueError('缺少本轮已通过的CPU实际输入与恢复检查')
    geometry=read(ROOT/'preflight/source_groundedness_adapter_geometry.json')
    pool=read(Path(formal['history_pool_directory'])/'pool_metadata.json')
    if geometry.get('status')!='passed' or geometry.get('tensor_count')!=248:
        raise ValueError('E248张量几何不匹配')
    if (pool.get('score_source')!='old_training_F_E_A_S' or pool.get('selected_QA')!=251
            or pool.get('historical_rescoring_performed') is not False):
        raise ValueError('固定251历史池不匹配')
    rows=lambda path:[json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    indices=read(ROOT/'preflight/input_lengths.json')['longest_pair_indices']
    full=rows(formal['train_dataset'])
    if indices!=[126,127] or len(full)!=720 or rows(capacity['train_dataset'])!=[full[i] for i in indices]:
        raise ValueError('720原顺序/最长真实pair数据不匹配')
    from judge_prompt_assets import validate_runtime_prompts
    validate_runtime_prompts(ROOT)
    if formal.get('checkpoint_input_cpu_offload') is True:
        offload=read(ROOT/'preflight/checkpoint_offload_cpu_result.json')
        offload_receipt=read(ROOT/'preflight/checkpoint_offload_cpu_submission.json')
        terminal_success(offload_receipt['job_id'])
        if (offload.get('job_id')!=offload_receipt['job_id'] or offload.get('status')!='passed'
                or offload.get('tests_run')!=7 or offload.get('CUDA_initialized') is not False
                or offload.get('actual_swift_context_verified') is not True
                or offload.get('all_layers_cpu_storage_verified') is not True
                or offload.get('regression_before_fix_failed_expected') is not True):
            raise ValueError('检查点输入CPU转存缺少当前原生等价性验证')
    return formal,capacity,cpu


def prepare():
    if (ROOT/'submissions/current_training_submission.json').exists():raise ValueError('已有正式回执，不准备重复提交')
    formal,capacity,cpu=verify_preflight();policy=check_policy()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8]
    task=ROOT/'submissions'/('combined_'+stamp);task.mkdir()
    tools=read(ROOT/'preflight/media_tools.json')
    script=task/'combined.sbatch'
    script.write_text('\n'.join(['#!/usr/bin/env bash','set -euo pipefail',
        'export PYTHONPATH="'+str(ROOT/'multi-user')+':'+str(ROOT)+'"',
        'export PYTHONDONTWRITEBYTECODE=1','export PYTHONUNBUFFERED=1',
        'export VLLM_NO_USAGE_STATS=1','export TOKENIZERS_PARALLELISM=false',
        'export PATH="'+str(Path(tools['ffmpeg']).parent)+':${PATH}"',
        'export LD_LIBRARY_PATH="'+tools['ffmpeg_lib']+':${LD_LIBRARY_PATH:-}"',
        'export HF_HUB_OFFLINE=1','export HF_DATASETS_OFFLINE=1',
        formal['train_python']+' '+str(ROOT/'combined_allocation.py')+' --root '+str(ROOT)])+'\n')
    subprocess.run(['bash','-n',str(script)],check=True)
    command=['sbatch','--parsable','--job-name=egoqa-b50176-combined180',
        '--account='+formal['slurm']['account'],'--nodes=1','--ntasks=1',
        '--cpus-per-task=32','--mem=750G','--gres=gpu:2','--constraint=h200','--time=2-12:00:00',
        '--output='+str(task/'combined-%j.out'),'--error='+str(task/'combined-%j.err'),
        '--chdir='+str(ROOT),str(script)]
    checked=subprocess.run([command[0],'--test-only',*command[1:]],capture_output=True,text=True)
    (task/'test_only.stdout').write_text(checked.stdout);(task/'test_only.stderr').write_text(checked.stderr)
    checked.check_returncode()
    record={'status':'prepared_not_submitted','phase':'combined_capacity_then_stage180',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'task_dir':str(task),
        'command':command,'formal_config':formal,'capacity_config':capacity,
        'config':str(ROOT/'configs/train.json'),'capacity_config_path':str(ROOT/'configs/capacity.json'),
        'CPU_preflight_job':cpu['job_id'],'legacy_capacity_job_id':'20443244','legacy_controller_job_id':'20443249',
        'GPU_count':2,'fresh_LoRA':True,'max_pixels':50176,'stop_after_steps':180,'max_steps':360,
        'train_conditions':720,'validation_conditions':0,'history_QA':251,
        'historical_rescoring_performed':False,'allocation_walltime_seconds':216000,
        'capacity_included_in_60h':True,'GPU_policy':policy}
    save(task/'prepared.json',record);save(ROOT/'submissions/prepared_combined_submission.json',record)
    print('COMBINED_PREPARED',json.dumps(record,ensure_ascii=False),flush=True)


def submit():
    marker=ROOT/'submissions/current_training_submission.json'
    if marker.exists():raise ValueError('已有正式回执；禁止重复提交')
    record=read(ROOT/'submissions/prepared_combined_submission.json')
    if record.get('status')!='prepared_not_submitted':raise ValueError('没有可用的准确准备回执')
    text=subprocess.check_output(['sacct','-j','20443244,20443249','-nP','--format=JobIDRaw,State,ExitCode'],text=True)
    verify_legacy_terminal(text)
    formal,capacity,_=verify_preflight()
    if formal!=record['formal_config'] or capacity!=record['capacity_config']:
        raise ValueError('准备后配置发生变化；先检查差异')
    record['GPU_policy']=check_policy();record['status']='submitting'
    record['legacy_terminal_accounting']=text
    record['submitted_at_utc']=datetime.now(timezone.utc).isoformat()
    # 独占创建正式回执，防止并发提交；未知结果必须先恢复，不能自动重投。
    with marker.open('x',encoding='utf-8') as f:json.dump(record,f,ensure_ascii=False,indent=2)
    task=Path(record['task_dir']);save(task/'submission.json',record)
    result=subprocess.run(record['command'],capture_output=True,text=True,timeout=60)
    (task/'sbatch.stdout').write_text(result.stdout);(task/'sbatch.stderr').write_text(result.stderr)
    result.check_returncode();job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise ValueError('sbatch回执未知，保留记录禁止重投')
    record.update(status='submitted',job_id=job,
        output_dir=str(Path(formal['output_root'])/('train_'+job)),
        capacity_output_dir=str(Path(capacity['output_root'])/('train_'+job)),
        stdout=str(task/('combined-'+job+'.out')),stderr=str(task/('combined-'+job+'.err')))
    save(marker,record);save(task/'submission.json',record)
    print('COMBINED_SUBMITTED',json.dumps(record,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['prepare','submit'],required=True)
    args=parser.parse_args();prepare() if args.mode=='prepare' else submit()
