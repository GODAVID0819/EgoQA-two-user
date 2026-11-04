import json,shutil
from pathlib import Path
from prepare_run import build_config
from dual_runtime import validate_config
root=Path(__file__).resolve().parent
old=Path('/scratch/xl6775/projects/aria3-cp348-fec-grpo-20261029_1726')
v7=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/grpo-v7-four-window-resume32-20261103_1740')
j7=json.loads((v7/'configs/judge_v7.json').read_text())
c,j=build_config(str(root),str(root/'dataset'),json.loads((old/'configs/judge_fec.json').read_text()))
c.update(max_steps=24,stop_after_steps=24,warmup_steps=2,reward_mode='continuous_gated',reward_revision='aria12_gated_A',
 run_attempt='aria12_5min_cp348_A',output_root=str(root/'outputs/stage24'),scratch_root='/scratch/xl6775/ag12',
 walltime_seconds=32400,training_target_hours=9,walltime_basis='24 updates, 9 hours / 18 H200 hours authorized',
 source_train_dataset=str(root/'data/train_length_paired.jsonl'))
c['slurm']['walltime']='09:00:00';c['utilization_guard']['target']=90
j.update(adapters=j7['adapters'],max_lora_rank=128,gpu_memory_utilization=.78,reward_revision=c['reward_revision'])
for path in j['adapters'].values():
 assert (Path(path)/'adapter_model.safetensors').stat().st_size>0
validate_config(c)
shutil.copyfile(old/'preflight/media_tools.json',root/'preflight/media_tools.json')
if (old/'runtime_deps_train').exists() and not (root/'runtime_deps_train').exists():
 (root/'runtime_deps_train').symlink_to(old/'runtime_deps_train',target_is_directory=True)
(root/'configs').mkdir(exist_ok=True)
for n,obj in [('train.json',c),('judge_fec.json',j)]:
 with (root/'configs'/n).open('x') as f:json.dump(obj,f,indent=2)
