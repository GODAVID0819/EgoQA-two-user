"""在 Torch 登录节点用标准库提取已完成更新的完整 QA，不启动 GPU。"""
from pathlib import Path
from itertools import islice
from datetime import datetime, timezone, timedelta
import collections
import json
import math
import os
import pwd
import re
import socket

assert pwd.getpwuid(os.getuid()).pw_name=='xl6775'
ROOT=Path('/scratch/xl6775/projects/EgoQA-six-user-baseline-grpo-20260929/variants/full120-ftjudge-dual-speed-v5-20261016')
manifest=json.loads((ROOT/'submissions/current_pipeline_submission.json').read_text())
job=manifest['jobs']['lr1e5']
assert str(job['job_id'])=='19964974'
out=Path(job['output_dir'])
config=json.loads((out/'run_config.json').read_text())
planned=[json.loads(line) for line in Path(config['train_dataset']).read_text().splitlines() if line.strip()]
assert len(planned)==600
logs=list((out/'swift').glob('*/logging.jsonl'))
assert len(logs)==1
logging=[]
for line in logs[0].read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    if 'global_step/max_steps' in r and 'loss' in r:
        r['completed_global_step']=int(r['global_step/max_steps'].split('/')[0])
        logging.append(r)
completed=max(r['completed_global_step'] for r in logging)
assert completed==300
snapshot=Path(__file__).resolve().parent/'history_source'
snapshot.mkdir(parents=True,exist_ok=False)

def parse_qa(text):
    cleaned=text.strip()
    if cleaned.startswith('```'):
        match=re.search(r'```(?:json)?\s*(.*?)\s*```',cleaned,re.DOTALL)
        if match:cleaned=match.group(1).strip()
    start,end=cleaned.find('{'),cleaned.rfind('}')
    if start<0 or end<start:raise ValueError('没有完整QA JSON')
    qa=json.loads(cleaned[start:end+1])
    if not isinstance(qa,dict):raise ValueError('QA必须是对象')
    options=qa.get('options')
    if not isinstance(options,list) or len(options)!=5 or any(not isinstance(x,str) or not x.strip() for x in options):raise ValueError('选项结构不合法')
    correct=qa.get('correct')
    if not isinstance(correct,str) or correct not in ('A','B','C','D','E'):raise ValueError('正确选项不合法')
    if qa.get('answer')!=options[ord(correct)-ord('A')]:raise ValueError('答案字段不一致')
    if not isinstance(qa.get('question'),str) or not qa['question'].strip():raise ValueError('问题缺失')
    return qa

rows=[]
seen=set()
trace_paths=[]
status_counts=collections.Counter()
for rank in (0,1):
    trace=out/f'rank{rank}'/'reward_trace.jsonl'
    trace_paths.append(str(trace))
    with trace.open() as stream:
        raw=[json.loads(line) for line in islice(stream,completed*4)]
    assert len(raw)==completed*4
    audits=[json.loads(line) for line in (out/f'rank{rank}'/'group_audit.jsonl').read_text().splitlines() if line.strip()]
    audit_by_step={r['global_step']+1:r for r in audits if r['mode']=='train'}
    for step in range(1,completed+1):
        group=raw[(step-1)*4:step*4]
        assert len({r['score_batch_id'] for r in group})==1
        assert len({r['evidence_id'] for r in group})==1
        expected=planned[2*(step-1)+rank]
        binding=(expected['source_packet_id'],int(expected['asker_index']))
        assert all((r['source_packet_id'],int(r['asker_index']))==binding for r in group)
        if step>12:
            a=audit_by_step[step]['groups'][rank]
            assert len(a)==4 and all((r['source_packet_id'],int(r['asker_index']))==binding for r in a)
        for slot,r in enumerate(group):
            assert r['request_id'] not in seen
            seen.add(r['request_id'])
            j=r['judge_result']
            status=j['status']
            assert status in ('scored','invalid_completion'),status
            status_counts[status]+=1
            probs=j.get('probabilities')
            if status=='scored':
                assert set(probs)=={'formality','groundedness','speaker_only','all_six'}
                assert all(math.isfinite(v) and 0<=v<=1 for v in probs.values())
            qa=None;error=None
            try:qa=parse_qa(r['completion'])
            except (ValueError,TypeError,KeyError) as exc:error=f'{type(exc).__name__}: {exc}'
            assert status!='scored' or qa is not None,(r['request_id'],error)
            rows.append({k:r[k] for k in ('request_id','reward','reward_mode','reward_revision','source_packet_id','asker_index','evidence_id','completion','score_batch_id','score_batch_size')} |
                        {'global_step':step,'rank':rank,'slot':slot,'status':status,'probabilities':probs,
                         'qa':qa,'parse_error':error,'judge':j.get('judge'),'source_trace':str(trace),
                         'source_job':'19938940' if step<=12 else '19964974'})
rows.sort(key=lambda r:(r['global_step'],r['rank'],r['slot']))
assert len(rows)==completed*8
conditions={(r['source_packet_id'],r['asker_index']) for r in rows}
assert len(conditions)==completed*2
small_planned=[{k:r[k] for k in ('source_packet_id','asker_index','evidence_id','required_users') if k in r} for r in planned]
metadata={'snapshot_time_beijing':datetime.now(timezone(timedelta(hours=8))).isoformat(),
          'snapshot_root':str(snapshot),'job_id':'19964974','inherited_job_id':'19938940',
          'completed_global_step':completed,'planned_global_steps':300,
          'candidate_count':len(rows),'condition_count':len(conditions),'planned_conditions':600,
          'status_counts':dict(status_counts),'trace_paths':trace_paths,'logging_path':str(logs[0]),
          'group_alignment':'已逐组匹配真实训练清单；第13步起另匹配实际双rank group_audit；每rank每更新完整G4。',
          'boundary':'只含快照时已完成更新。原四路F/E/A/S，非本轮新E/联合C评分。不混入框架eval或最终验证。',
          'run_config':config,'judge_config':json.loads(Path(config['judge_config']).read_text()),
          'planned_condition_bindings':small_planned,'logging':logging}
(snapshot/'snapshot_metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
with (snapshot/'complete_training_qa.jsonl').open('w') as stream:
    for r in rows:stream.write(json.dumps(r,ensure_ascii=False)+'\n')
print('SNAPSHOT_READY',json.dumps({k:metadata[k] for k in ('snapshot_time_beijing','snapshot_root','completed_global_step','candidate_count','condition_count','status_counts')}))
print('SNAPSHOT_FILES',[(p.name,p.stat().st_size) for p in snapshot.iterdir()])
