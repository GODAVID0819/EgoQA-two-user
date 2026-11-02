"""正式GRPO完整状态续跑入口，记录从启动到验收结束的边界。"""
import argparse,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);a=p.parse_args()
c=json.loads(a.config.read_text());root=Path(c['project_root']);job=os.environ['SLURM_JOB_ID']
start=time.time();report=dict(job_id=job,started_epoch=start,phase='launching',config=str(a.config))
path=root/'reports'/('gpu_launcher_'+job+'.json');path.write_text(json.dumps(report,indent=2))
command=[c['train_python'],'-u',str(root/'dual_launch.py'),'--config',str(a.config)]
print('V7_FORMAL_RUN_START',json.dumps(report),flush=True)
result=subprocess.run(command,cwd=root/'multi-user',env=os.environ.copy())
report.update(finished_epoch=time.time(),wall_seconds=time.time()-start,returncode=result.returncode,phase='completed' if result.returncode==0 else 'failed')
path.write_text(json.dumps(report,indent=2));print('V7_FORMAL_RUN_END',json.dumps(report),flush=True)
raise SystemExit(result.returncode)
