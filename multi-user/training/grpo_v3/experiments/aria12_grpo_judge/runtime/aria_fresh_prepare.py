"""物化独立三人运行目录，并用现有48行构建训练配置和长度配对顺序。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

HERE=Path(__file__).resolve().parent
BASE=(HERE/'config_template.json' if (HERE/'config_template.json').is_file()
      else HERE.parent/'full120x6_fec_moda_stage180_20261021/examples/train_config_20573337.json')


def materialize_runtime(repo_root, destination):
    repo_root,destination=Path(repo_root),Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('不覆盖已有运行目录')
    destination.mkdir(parents=True,exist_ok=True)
    multi=repo_root/'multi-user'
    selected=list(multi.glob('*.py'))+[p for p in (multi/'training').rglob('*.py')
        if 'experiments' not in p.relative_to(multi/'training').parts and '__pycache__' not in p.parts]
    for source in selected:
        target=destination/source.relative_to(repo_root)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
    for source in (HERE/'runtime').rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix in {'.py','.txt','.json'}:
            target=destination/source.relative_to(HERE/'runtime')
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,target)
    (destination/'runtime_deps_train').mkdir(exist_ok=True)
    shutil.copytree(HERE/'slurm',destination/'slurm')
    shutil.copy2(__file__,destination/'prepare_run.py')
    shutil.copy2(BASE,destination/'config_template.json')
    commit=subprocess.check_output(['git','-C',str(repo_root),'rev-parse','HEAD'],text=True).strip()
    (destination/'SOURCE_COMMIT.json').write_text(json.dumps({'commit':commit,'branch':'lxr-ours-grpo',
        'experiment':'aria3_fec_20261028'},indent=2),encoding='utf-8')
    return destination


def build_config(run_root, dataset_root, judge_source):
    root,data=PurePosixPath(run_root),PurePosixPath(dataset_root)
    original=json.loads(BASE.read_text(encoding='utf-8'))
    c={k:v for k,v in original.items() if not k.startswith(('source_','history_'))}
    c.update(project_root=str(root),train_dataset=str(root/'data/train_length_paired.jsonl'),
        source_train_dataset=str(data/'data/train.jsonl'),judge_config=str(root/'configs/judge_fec.json'),
        output_root=str(root/'outputs/stage96'),scratch_root='/scratch/xl6775/ag3',
        max_steps=96,stop_after_steps=96,train_conditions=48,validation_conditions=0,
        expected_split_counts={'train':48,'validation':0},val_dataset=None,eval_strategy='no',
        initial_fresh_base=True,resume_from_checkpoint=None,frozen_vision_cache=False,
        reward_mode='continuous',reward_revision='aria3_fec_quality_v1',diversity_lambda=0,
        history_pool_directory=None,walltime_seconds=57600,training_target_hours=16,
        run_attempt='aria3_fec_quality96',dataset_shuffle=False,
        source_provenance={'packet_job':'21197719','dataset_root':str(data),'conditions':48,
            'cross_device_sync_verified':False,'history_and_semantic_diversity_enabled':False},
        walltime_basis='历史每更新450.9秒×96约12小时，另含启动收尾及余量，用户确认16小时/32H200卡时上限')
    c['slurm']={'account':'torch_pr_674_tandon_advanced','gres':'gpu:2','constraint':'h200',
                'cpus':32,'mem':'750G','walltime':'16:00:00'}
    judge=json.loads(json.dumps(judge_source))
    judge.update(prompt_directory=str(root/'prompts'),shared_gpu=True,
                 reward_revision='aria3_fec_quality_v1')
    return c,judge


def prepare_data(config):
    root=Path(config['project_root'])
    sys.path[:0]=[str(root/'multi-user'),str(root)]
    from training.grpo_v3.six_user_binary.data import validate_row
    from dual_runtime import condition_binding,make_length_pairs,validate_config
    rows=[json.loads(line) for line in Path(config['source_train_dataset']).read_text().splitlines() if line.strip()]
    if len(rows)!=48 or len({condition_binding(r) for r in rows})!=48:
        raise ValueError('源数据必须是48个唯一三人条件')
    for row in rows:validate_row(row)
    ordered=make_length_pairs(rows,lambda row:len(row['images']),seed=42)
    validate_config(config)
    target=Path(config['train_dataset']);target.parent.mkdir(parents=True,exist_ok=True)
    content=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in ordered)
    if target.exists() and target.read_text()!=content:raise FileExistsError('不覆盖不同的已有训练输入')
    target.write_text(content,encoding='utf-8')
    definition={'train':[{'source_packet_id':row['source_packet_id'],'speaker_index':row['asker_index'],
        'input_view_count':3,'duration_seconds':600,'image_count':len(row['images'])} for row in ordered]}
    (root/'condition_definition.json').write_text(json.dumps(definition,indent=2))
    print('ARIA3_TRAIN_DATA_PREPARED',json.dumps({'conditions':48,'pairs':24,'passes':4,
        'candidate_slots':768,'train_dataset':str(target)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    m=sub.add_parser('materialize');m.add_argument('--repo',type=Path,required=True);m.add_argument('--destination',type=Path,required=True)
    c=sub.add_parser('configure');c.add_argument('--root',type=Path,required=True);c.add_argument('--dataset-root',type=Path,required=True);c.add_argument('--judge-source',type=Path,required=True)
    d=sub.add_parser('data');d.add_argument('--config',type=Path,required=True)
    a=p.parse_args()
    if a.command=='materialize':
        materialize_runtime(a.repo,a.destination)
        shutil.copy2(__file__,a.destination/'prepare_run.py')
    elif a.command=='configure':
        config,judge=build_config(str(a.root),str(a.dataset_root),json.loads(a.judge_source.read_text()))
        (a.root/'configs').mkdir(exist_ok=True)
        (a.root/'configs/train.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
        (a.root/'configs/judge_fec.json').write_text(json.dumps(judge,ensure_ascii=False,indent=2))
    else:prepare_data(json.loads(a.config.read_text()))


if __name__=='__main__':main()
