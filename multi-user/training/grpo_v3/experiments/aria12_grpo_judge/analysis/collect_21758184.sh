set -eu
exec 2>&1
cd /scratch/xl6775/projects/aria12-iter1-eclarity-20261104_0125
sacct -nP -j 21758184 -o JobID,State,ExitCode,Elapsed
python3 - <<'PY'
from pathlib import Path
import datetime,json,math
r=Path('outputs/stage24/train_21758184')
if not all((r/f'rank{i}/phase_timing.jsonl').exists() and (r/f'rank{i}/reward_trace.jsonl').exists() for i in (0,1)):
 print('NO_COMPLETED_DATA');raise SystemExit(0)
def rows(p):
 for l in p.read_text().splitlines():
  try:yield json.loads(l)
  except ValueError:pass
steps=[];rr=[];ids=set()
for i in (0,1):
 d=r/f'rank{i}';steps.append({x['global_step'] for x in rows(d/'phase_timing.jsonl') if x['phase']=='optimizer_step_wall'})
 rr.append(list(rows(d/'reward_trace.jsonl')))
points=[]
for k in sorted(steps[0]&steps[1]):
 m=dict.fromkeys('F E C Total'.split(),0.);p=dict.fromkeys(m,0);invalid=skip=scored=0
 for rank in (0,1):
  group=[x for x in rr[rank] if x['global_step']==k];assert sorted(x['slot'] for x in group)==[0,1,2,3]
  for x in group:
   assert x['request_id'] not in ids;ids.add(x['request_id'])
   j=x['judge_result'];c=0.
   if j['status']=='invalid_completion':f=e=0.;invalid+=1
   else:
    assert j['status']=='scored';q=j['probabilities'];f=q['formality'];e=q['groundedness']
    if e>.5:
     c=q['combined_answerability'];assert isinstance(c,(int,float));scored+=1
    else:assert q['combined_answerability'] is None;skip+=1
   vals=(f,e,c,x['reward']);assert math.isclose(x['reward'],.2*f+.4*e+.4*c,abs_tol=1e-9)
   for name,v,hit in zip(m,vals,(f>=.5,e>=.5,c>=.5,f>=.5 and e>.5 and c>=.5)):m[name]+=v/8;p[name]+=int(hit)
 points.append(dict(step=k,candidate_count=8,invalid_count=invalid,c_skipped=skip,c_scored=scored,mean_reward=m,pass_count=p,pass_rate={n:v/8 for n,v in p.items()}))
n=len(points);assert [p['step'] for p in points]==list(range(1,n+1))
now=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S')
print('ARIA3_EIGHT_CURVE_DATA',json.dumps(dict(job_id='21758184',completed_updates=n,candidate_slots=8*n,captured_at_beijing=now,points=points)))
print('TERMINAL_FILES',[str(p.relative_to(r)) for p in r.glob('*.json')])
print('CKPT',[(p.name,(p/'adapter_model.safetensors').stat().st_size if (p/'adapter_model.safetensors').exists() else 0) for p in r.glob('swift/*/checkpoint-*')])
for fn in ('run_manifest.json','terminal_acceptance.json','progress_stage.json','validation_report.json'):
 if (r/fn).exists():
  a=json.loads((r/fn).read_text()); print(fn,{k:v for k,v in a.items() if k in ('status','global_step','phase','error','failed_checks','cleanup_errors')})
PY
