"""从已训练348 adapter准备独立Aria阶段，仅准备配置，不提交作业。"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
FRESH=HERE.parent/'aria3_fec_20261028'/'prepare_run.py'
if (HERE/'aria_fresh_prepare.py').is_file():FRESH=HERE/'aria_fresh_prepare.py'
spec=importlib.util.spec_from_file_location('aria_fresh_prepare',FRESH)
fresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(fresh)
CHECKPOINT='/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120x6-fec-moda-b50176-resume238-20261025/outputs/stage360/train_20857619/swift/v0-20261025-004728/checkpoint-348'

def build_config(run_root,dataset_root,judge_source,*,profile='safe',checkpoint=CHECKPOINT):
    if profile not in {'safe','prefill16k'}:raise ValueError('推理配置仅支持safe或prefill16k')
    c,j=fresh.build_config(run_root,dataset_root,judge_source)
    budget=8192 if profile=='safe' else 16384
    c.update(initialization_mode='adapter_warm_start',initial_fresh_base=False,reset_training_state=True,
        initial_adapter=checkpoint,reference_adapter=checkpoint,initial_checkpoint_step=348,
        initial_checkpoint_job='20857619',resume_from_checkpoint=None,
        run_attempt='aria3_cp348_fec96',inference_profile=profile,policy_vllm_max_num_batched_tokens=budget)
    j.update(max_num_batched_tokens=budget,cpu_prefetch=True)
    c['source_provenance']['initial_policy_checkpoint']=checkpoint
    c['walltime_basis']='仅沿用96步/2H200草案；提交前按新阶段实测吞吐和本轮预算重新核定'
    return c,j

def materialize_runtime(repo_root,destination):
    destination=fresh.materialize_runtime(repo_root,destination)
    shutil.copy2(FRESH,destination/'aria_fresh_prepare.py')
    shutil.copy2(__file__,destination/'prepare_run.py')
    source=json.loads((destination/'SOURCE_COMMIT.json').read_text())
    source['experiment']='aria3_cp348_20261029'
    (destination/'SOURCE_COMMIT.json').write_text(json.dumps(source,indent=2),encoding='utf-8')
    return destination

def configure_run(root,dataset_root,judge_source,*,profile='safe'):
    root=Path(root);targets=[root/'configs/train.json',root/'configs/judge_fec.json']
    if any(p.exists() for p in targets):raise FileExistsError('不覆盖既有实验配置，请物化新的运行目录')
    config,judge=build_config(str(root),str(dataset_root),judge_source,profile=profile)
    (root/'configs').mkdir(exist_ok=True)
    for target,values in zip(targets,[config,judge]):
        with target.open('x',encoding='utf-8') as f:json.dump(values,f,ensure_ascii=False,indent=2)
    return config,judge

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    m=sub.add_parser('materialize');m.add_argument('--repo',type=Path,required=True);m.add_argument('--destination',type=Path,required=True)
    c=sub.add_parser('configure');c.add_argument('--root',type=Path,required=True);c.add_argument('--dataset-root',type=Path,required=True);c.add_argument('--judge-source',type=Path,required=True);c.add_argument('--profile',choices=['safe','prefill16k'],default='safe')
    d=sub.add_parser('data');d.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    if a.command=='materialize':materialize_runtime(a.repo,a.destination)
    elif a.command=='configure':
        configure_run(a.root,a.dataset_root,json.loads(a.judge_source.read_text()),profile=a.profile)
    else:fresh.prepare_data(json.loads(a.config.read_text()))

if __name__=='__main__':main()
