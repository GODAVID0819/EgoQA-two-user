"""只修改本次私有副本：四窗口顺序、全量留出验证、r64生成端上限。"""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path)
    root=p.parse_args().root.resolve()
    if root.name!='full120-r64-lrpair-20261013':raise ValueError('只允许本轮独立实验根')
    package=root/'multi-user/training/grpo_v3/six_user_binary'
    changes=[]
    def replace(name,old,new):
        path=package/name;text=path.read_text()
        if old in text:
            if text.count(old)!=1:raise ValueError('源码锚点不唯一：'+name)
            path.write_text(text.replace(old,new));changes.append(name)
        elif new not in text:raise ValueError('源码与已核验来源不同：'+name)
    replace('launch.py','"dataset_shuffle": "true",',
        '"dataset_shuffle": str(c.get("dataset_shuffle", True)).lower(),')
    replace('launch.py','vllm_enable_lora="true", sleep_level=1,',
        'vllm_enable_lora="true", vllm_max_lora_rank=c.get("lora_rank", 8), sleep_level=1,')
    replace('holdout_runtime.py','def data_validation_command(c):\n',
        "def data_validation_command(c):\n    if c.get('full120_speaker_holdout') or c.get('minimum_runtime_check'):\n        command=[c['train_python'],str(Path(c['project_root'])/'validate_full120.py'),'--train',c['train_dataset'],'--val',c['val_dataset'],'--definition',str(Path(c['project_root'])/'condition_definition.json')]\n        if c.get('minimum_runtime_check'): command.append('--minimum-runtime-check')\n        return command\n")
    (root/'runtime_changes.json').write_text(json.dumps({'source_job_id':'19460878','changed_files':sorted(set(changes)),
        'changes':['本轮按四不同窗口排列训练条件并关闭打乱','只在本轮full120模式验证600/120固定留出','训练生成端LoRA rank上限与r64同步'],
        'reward_unchanged':True,'grpo_loss_unchanged':True,'generation_media_unchanged':True},ensure_ascii=False,indent=2))
    print('PAIR_RUNTIME_INSTALLED',sorted(set(changes)),flush=True)


if __name__=='__main__':main()
