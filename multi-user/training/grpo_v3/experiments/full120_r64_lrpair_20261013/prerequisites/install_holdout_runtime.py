"""只修改独立实验副本；原默认划分与评分行为保持。"""
import argparse
import json
from pathlib import Path
import shutil


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True,type=Path)
    root=parser.parse_args().root.resolve()
    if root.name!='speaker-holdout-design-20261009':raise ValueError('只允许当前独立根')
    package=root/'multi-user/training/grpo_v3/six_user_binary'
    for name in ['speaker_holdout_split.py','holdout_runtime.py']:shutil.copy2(root/name,package/name)
    changes=[]
    def replace(name,old,new):
        path=package/name;text=path.read_text()
        if text.count(old)!=1:raise ValueError(f'源码锚点不唯一：{name} {old[:60]!r}')
        path.write_text(text.replace(old,new));changes.append(name)
    replace('workflow.py','    return result\n\n\n',
        "    for key in ('speaker_holdout_validation', 'test_dataset'):\n        if key in workflow: result[key] = workflow[key]\n    return result\n\n\n")
    replace('data.py','    parser.add_argument("--validate", type=Path, nargs="+")',
        '    parser.add_argument("--speaker-holdout", action="store_true")\n    parser.add_argument("--validate", type=Path, nargs="+")')
    replace('data.py','        validate_splits(splits)\n',
        '        if args.speaker_holdout:\n            from .speaker_holdout_split import validate_holdout\n            if len(resolved) != 3: raise ValueError("speaker留出模式必须检查训练/验证/测试")\n            validate_holdout(dict(zip(("train", "validation", "test"), splits.values())))\n        else:\n            validate_splits(splits)\n')
    replace('resume.py','        validate_splits(splits)\n',
        "        if c.get('speaker_holdout_validation'):\n            from .speaker_holdout_split import validate_holdout\n            validate_holdout(splits)\n        else:\n            validate_splits(splits)\n")
    replace('launch.py','        subprocess.run([c["train_python"], "-m", "training.grpo_v3.six_user_binary.data", "--validate", c["train_dataset"], c["val_dataset"]], cwd=package, env=env, check=True)',
        '        from .holdout_runtime import data_validation_command\n        subprocess.run(data_validation_command(c), cwd=package, env=env, check=True)')
    replace('launch.py',"    evaluate(checked['checkpoint'])\n\n\n",
        "    evaluate(checked['checkpoint'])\n    if c.get('test_dataset'):\n        evaluate(split='test')\n        evaluate(checked['checkpoint'], split='test')\n\n\n")
    replace('launch.py','        def evaluate_policy(adapter=None):\n            label = "policy" if adapter else "baseline"\n            destination = output / ("validation_" + label + ".json")',
        '        def evaluate_policy(adapter=None, *, split="validation"):\n            from .holdout_runtime import evaluation_files\n            files = evaluation_files(c, output, adapter, split, baseline_source)\n            label = files["label"]\n            destination = Path(files["destination"])\n            if split == "test":\n                Path(files["config_path"]).write_text(json.dumps(files["config"], indent=2))')
    replace('launch.py','"--config", str(output / "run_config.json"), "--output", str(destination),',
        '"--config", files["config_path"], "--output", str(destination),')
    replace('launch.py','eval_command += ["--adapter", str(adapter), "--baseline", str(baseline_source)]',
        'eval_command += ["--adapter", str(adapter), "--baseline", files["baseline"]]')
    replace('evaluation.py',"(args.output.parent/'validation_comparison.json').write_text",
        "(args.output.parent/('test_comparison.json' if args.output.stem.startswith('test_') else 'validation_comparison.json')).write_text")
    cpath=root/'preflight/formal200/workflow.json';c=json.loads(cpath.read_text())
    c['expected_split_counts']={'train':40,'validation':8,'test':6};c['save_steps']=2
    c['experiment_scope']='8窗口40训练/8轮换speaker验证；3未见窗口6测试；新LoRA200步完整退火；固定原Judge'
    for key in ['failure_provenance','source_fixed_validation','source_fixed_validation_job_id','required_completed_evaluation_job_id']:
        c.pop(key,None)
    cpath.write_text(json.dumps(c,ensure_ascii=False,indent=2))
    (root/'preflight/formal200/runtime_changes.json').write_text(json.dumps({'files':sorted(set(changes)),
        'default_packet_disjoint_rule_unchanged':True,'test_separate_from_validation':True,'no_reward_or_algorithm_changes':True},ensure_ascii=False,indent=2))
    print('HOLDOUT_RUNTIME_INSTALLED',sorted(set(changes)),flush=True)


if __name__=='__main__':main()
