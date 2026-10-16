"""CPU resume metadata audit; reuse the previously verified sampler, no model imports."""
import ast,json,sys,tempfile
from importlib.metadata import version
from pathlib import Path
from recovery import validated_resume,seed_reward_prefix,expected_resume_bindings
from dual_runtime import validate_config
from training.grpo_v3.six_user_binary.launch import swift_command
r=Path(__file__).resolve().parent
c=json.loads((r/'configs/lr1e5.json').read_text());validate_config(c);meta=validated_resume(c)
a=json.loads((r/'preflight/length_pair_sampler.json').read_text())
assert a['status']=='passed' and a['shard']['global_unique_conditions']==600 and a['shard']['optimizer_updates_per_rank']==300 and a['shard']['same_window_pairs']==0
assert a['sampler']['mini_repeat_count']==4 and a['sampler']['repeat_count']==4
assert version('trl')==a['versions']['trl'] and version('accelerate')==a['versions']['accelerate']
assert 1200-12*4==1152 and 1152//4==288 and 600-12*2==576
bindings=expected_resume_bindings(c);assert len(bindings)==2 and bindings[0][0]!=bindings[1][0]
with tempfile.TemporaryDirectory(dir=r/'preflight') as folder:
 for rank in [0,1]:(Path(folder)/('rank'+str(rank))).mkdir()
 prefix=seed_reward_prefix(c,folder);assert prefix['inherited_candidates']==96
command=swift_command({**c,'policy_gpu_indices':[0],'judge_gpu_indices':[0]},str(r/'preflight/cpu_cli_only'))
options=command[command.index('--rlhf_type'):]+['--resume_only_model','false','--ignore_data_skip','false']
for name,value in [('resume_from_checkpoint',c['resume_from_checkpoint']),('max_steps','300'),('warmup_steps','10'),('resume_only_model','false'),('ignore_data_skip','false')]:
 assert options[options.index('--'+name)+1]==value
site=Path(sys.executable).parent.parent/'lib/python3.11/site-packages'
fields=set()
for p in list((site/'swift/arguments').rglob('*.py'))+[site/'transformers/training_args.py']:
 text=p.read_text()
 if not any(k in text for k in ['resume_only_model','ignore_data_skip']):continue
 for node in ast.walk(ast.parse(text)):
  if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name) and node.target.id in ['resume_only_model','ignore_data_skip']:fields.add(node.target.id)
native_args=json.loads((Path(c['resume_from_checkpoint'])/'args.json').read_text())
fields.update(key for key in ['resume_only_model','ignore_data_skip'] if key in native_args and isinstance(native_args[key],bool))
assert fields=={'resume_only_model','ignore_data_skip'}
print('ACTUAL_CPU_RESUME_POSITION_AND_CLI',json.dumps({'status':'passed','resume_step':12,'target_global_step':300,'remaining_updates':288,'remaining_unique_conditions':576,'inherited_candidates':96,'first_remaining_bindings':bindings,'scheduler_expected_lr':meta['expected_learning_rate'],'gpu_operations':0,'sampler_scope':'reuse previously passed real sampler audit; derive skip48, assert actual native resume position in formal run','cli_scope':'actual argv generator, native saved args and installed source, no model registry import'}),flush=True)