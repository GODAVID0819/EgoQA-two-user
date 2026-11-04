"""只读监控本次准确JobID；不提交、不取消、不修改远端作业。"""
import json,os,pathlib,subprocess,sys,time

root=pathlib.Path(__file__).resolve().parent
manifest=root/(sys.argv[1] if len(sys.argv)>1 else 'formal_submission.json')
job=json.loads(manifest.read_text(encoding='utf-8-sig'))['job_id']
script=root/f'monitor_query_{job}.sh'
script.write_text(f'''ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
JOB={job}
printf 'SACCT_STATE '
sacct -X -n -j "$JOB" -o State,ExitCode,Elapsed,NodeList
python3 - "$ROOT" "$JOB" <<'PY'
import json,sys,pathlib
root=pathlib.Path(sys.argv[1]);job=sys.argv[2];out=root/'outputs'/f'formal_{{job}}'
status=out/'run_status.json'
if status.exists():print('MONITOR_STATUS '+status.read_text().replace('\\n',' '))
log=root/'logs'/f'formal_{{job}}.out'
if log.exists():
    lines=log.read_text(errors='replace').splitlines()
    interesting=[l for l in lines if any(k in l for k in ['MODEL_LOADED','PROGRESS','Traceback','Error:','Exception:','COMPLETE '])]
    print('LOG_TAIL '+json.dumps(interesting[-4:]))
gpu=root/'logs'/f'gpu_{{job}}.csv'
if gpu.exists():print('GPU_TAIL '+json.dumps(gpu.read_text(errors='replace').splitlines()[-2:]))
PY
''',encoding='utf-8')
previous=None
for iteration in range(160):
    result=subprocess.run([sys.executable,str(root/'bridge_request.py'),str(script),'--timeout','180'],
        capture_output=True,text=True,encoding='utf-8',errors='replace')
    text=result.stdout
    if result.returncode:
        print('MONITOR_REQUEST_FAILED',result.returncode,text[-500:],result.stderr[-500:],flush=True)
        break
    record={'job_id':job,'snapshot_local':time.strftime('%Y-%m-%dT%H:%M:%S')}
    for line in text.splitlines():
        if line.startswith('SACCT_STATE '):record['slurm']=line[len('SACCT_STATE '):].strip()
        elif line.startswith('MONITOR_STATUS '):record['status']=json.loads(line[len('MONITOR_STATUS '):])
        elif line.startswith('LOG_TAIL '):record['log']=json.loads(line[len('LOG_TAIL '):])
        elif line.startswith('GPU_TAIL '):record['gpu']=json.loads(line[len('GPU_TAIL '):])
    (root/f'formal_{job}_live_status.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    state=record.get('slurm','');status=record.get('status',{});n=status.get('n',0)
    key=(state.split()[0] if state else '',n//32,status.get('status'))
    if key!=previous:
        print(json.dumps(record,ensure_ascii=False),flush=True);previous=key
    if any(value in state for value in ['COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','PREEMPTED']):
        print('MONITOR_TERMINAL',flush=True);break
    time.sleep(90)
