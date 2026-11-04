import os,json,sys
from pathlib import Path
import cpu_validate as v
root=Path(__file__).resolve().parent
config=json.loads((root/'configs/train.json').read_text())
_,scratch=v.setup(config)
from prepare_aria12 import prepare,read
from warm_start import validate_adapter_files
from dual_runtime import validate_config
validate_config(config)
validate_adapter_files(config['initial_adapter'],config)
rows=prepare(Path('/scratch/xl6775/projects/aria-three-user-packets-20261028_1500/artifacts/dataset'),root,read(root/'schedule.json'))
s=v.command_and_native_sampler(config,rows,root,scratch)
l=v.scheduler_check(config); roles=v.role_imports(config,root)
assert s['global_training_candidates']==192 and s['optimizer_updates_per_rank']==24 and l['scheduler_last_epoch']==24
out={'status':'passed','sampler':s,'schedule':l,'roles':list(roles),'image_counts':[min(len(x['images']) for x in rows),max(len(x['images']) for x in rows)],'job_id':os.environ['SLURM_JOB_ID']}
(root/'preflight/aria12_real_cpu.json').write_text(json.dumps(out,indent=2))
print('ARIA12_REAL_CPU_OK',out['image_counts'],flush=True)
