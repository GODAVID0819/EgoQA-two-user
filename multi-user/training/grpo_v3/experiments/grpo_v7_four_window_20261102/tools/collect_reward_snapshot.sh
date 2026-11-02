python3 - <<'PY'
import json,math,subprocess,statistics
from pathlib import Path
from datetime import datetime,timezone,timedelta
formal=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/grpo-v7-four-window-formal-20261102_1732')
meta=json.loads((formal/'reports/formal_submission.json').read_text())
pilot=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/grpo-v7-four-window-base-20261102_1550/outputs/validation3/train_21644657')
out=Path(meta['output'])
def read_rows(path):
    text=path.read_text()
    lines=text.splitlines();rows=[]
    for i,line in enumerate(lines):
        if not line.strip():continue
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i==len(lines)-1 and not text.endswith('\n'):continue
            raise
    return rows
def completed_steps(base):
    both=[]
    for rank in (0,1):
        path=base/f'rank{rank}'
        saves={int(x['global_step']) for x in read_rows(path/'save_completion.jsonl')}
        times={int(x['global_step']) for x in read_rows(path/'phase_timing.jsonl') if x.get('phase')=='optimizer_step_wall'}
        both.append(saves&times)
    return both[0]&both[1]
psteps=completed_steps(pilot)
assert {1,2,3}<=psteps,psteps
fsteps=completed_steps(out)
end=max(fsteps)
assert all(x in fsteps for x in range(4,end+1)),sorted(fsteps)
seen=set();updates=[]
for job,base,steps in [('21644657',pilot,range(1,4)),('21650205',out,range(4,end+1))]:
    rows_by_rank=[read_rows(base/f'rank{rank}/reward_trace.jsonl') for rank in (0,1)]
    groups_by_rank=[read_rows(base/f'rank{rank}/group_metrics.jsonl') for rank in (0,1)]
    for step in steps:
        rows=[];rank_rows=[]
        for rank in (0,1):
            batch=[x for x in rows_by_rank[rank] if x.get('global_step')==step]
            assert len(batch)==8,(job,step,rank,len(batch))
            assert len({x['request_id'] for x in batch})==8
            metric=[x for x in groups_by_rank[rank] if x.get('global_step')==step]
            assert len(metric)==1,(job,step,rank,len(metric))
            assert math.isclose(sum(x['reward'] for x in batch)/8,metric[0]['training_mean_reward'],abs_tol=1e-10)
            rows+=batch;rank_rows.append(batch)
        weighted={k:[] for k in ['F','E','C','speaker','J','R']}
        raw={k:[] for k in ['F','E','C','speaker']}
        lambdas=set();illegal=0;parse_fail=0;hits=0;valid=0
        for x in rows:
            assert x['request_id'] not in seen,(job,step,x['request_id'])
            seen.add(x['request_id'])
            lam=float(x['speaker_lambda']);lambdas.add(lam)
            result=x['judge_result'];R=float(x['reward']);J=float(x['quality_reward']);S=float(x['speaker_adjustment'])
            assert all(math.isfinite(t) for t in (R,J,S,lam))
            if result['status']=='invalid_completion':
                illegal+=1
                assert x['legal_qa'] is False and R==J==S==0
                F=E=C=0.
            else:
                assert result['status']=='scored' and x['legal_qa'] is True
                probs=result['probabilities']
                vals=[float(probs[k]) for k in ['formality','groundedness','combined_answerability']]
                assert all(0<=p<=1 for p in vals)
                for key,value in zip(['F','E','C'],vals):raw[key].append(value)
                F,E,C=.2*vals[0],.4*vals[1],.4*vals[2]
                h=x['speaker_hit']
                if h is None:
                    parse_fail+=1
                    assert S==0
                else:
                    assert h in (0,1)
                    hits+=int(h);valid+=1;raw['speaker'].append(int(h))
                    assert math.isclose(S,-lam*h,abs_tol=1e-10)
            assert math.isclose(F+E+C,J,abs_tol=1e-10),(job,step,'J')
            assert math.isclose(F+E+C+S,R,abs_tol=1e-10),(job,step,'R')
            for k,v in zip(['F','E','C','speaker','J','R'],[F,E,C,S,J,R]):weighted[k].append(v)
        assert len(lambdas)==1,(step,lambdas)
        means={k:statistics.mean(v) for k,v in weighted.items()}
        assert math.isclose(sum(means[k] for k in ['F','E','C','speaker']),means['R'],abs_tol=1e-10)
        updates.append(dict(global_step=step,source_job=job,slots=16,legal_qa=16-illegal,illegal_qa=illegal,
            speaker_valid=valid,speaker_hits=hits,speaker_parse_failures=parse_fail,speaker_lambda=next(iter(lambdas)),
            weighted_mean=means,raw_valid_mean={k:statistics.mean(v) if v else None for k,v in raw.items()}))
assert [x['global_step'] for x in updates]==list(range(1,end+1))
captured=datetime.now(timezone(timedelta(hours=8))).isoformat()
sacct=subprocess.check_output(['sacct','-j','21644657,21650205','-nP','--format=JobIDRaw,State,ExitCode,ElapsedRaw'],text=True)
result=dict(captured_at_beijing=captured,formal_job='21650205',source_job='21644657',planned_global_steps=180,
    completed_global_step=end,source_steps=[1,3],formal_steps=[4,end],sacct=sacct,updates=updates,
    checks=dict(two_ranks_per_update=True,eight_slots_per_rank=True,unique_candidate_ids=True,
        reward_matches_group_metrics=True,per_candidate_and_step_components_sum_to_reward=True,completed_only=True,
        global_step_contiguous=True),
    definitions=dict(total='R=0.2*pF+0.4*pE+0.4*pC-lambda*S; invalid generations have zero reward',
        weighted_means='all 16 original slots per completed update, including actual zero contribution on invalid generations',
        raw_means='F/E/C: scored QA only; speaker: valid blind answers only; missing judge output is not imputed',
        speaker='S=1 for a correct speaker-only blind answer, 0 for wrong; parsing failure gives no speaker penalty',
        rolling_lambda='logged pre-update lambda from the previous 8 completed updates, capped at 0.2'))
print('REWARD_CURVE_SNAPSHOT_JSON '+json.dumps(result,ensure_ascii=False))
PY

