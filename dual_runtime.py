"""双卡运行的纯参数、路径和完整G4组约束。"""
from pathlib import Path
import re


def condition_binding(row):
    match=re.match(r'^RLHF6U_(DAY\d+)_(\d{8})_',row['source_packet_id'])
    if not match:raise ValueError('窗口标识无法解析')
    return (match.group(1)+'_'+match.group(2),int(row.get('asker_index',row.get('speaker_index'))))


def validate_config(c):
    expected={'world_size':2,'num_generations':4,'per_device_train_batch_size':1,
              'gradient_accumulation_steps':4,'max_steps':300,'lora_rank':64,
              'lora_alpha':128,'train_conditions':600,'validation_conditions':120}
    if any(c.get(k)!=v for k,v in expected.items()):
        raise ValueError('双卡参数不符合已批准的两窗口G4配置')
    if c.get('resume_from_checkpoint'):
        from recovery import validated_resume
        validated_resume(c)
    if not 0<c.get('stop_after_steps',300)<=300:
        raise ValueError('实际停止步数超出300步')


def rank_environment(output, records, rank):
    if rank not in (0,1) or len(records)!=2 or records[0]['port']==records[1]['port']:
        raise ValueError('rank或独立Judge端口不合法')
    root=Path(output);p=root/('rank'+str(rank));rec=records[rank]
    return {'EGOQA_GUARD_CUDA_DEVICE':str(rank),
            'EGOQA_SIX_USER_JUDGE_URL':'http://127.0.0.1:'+str(rec['port']),
            'EGOQA_SIX_USER_JUDGE_INSTANCE':rec['instance'],
            'EGOQA_GRPO_V3_REWARD_TRACE':str(p/'reward_trace.jsonl'),
            'EGOQA_UTILIZATION_HANDOFF':str(p/'utilization_handoff.signal'),
            'EGOQA_UTILIZATION_LOG':str(p/'utilization_trainer.jsonl'),
            'EGOQA_UTILIZATION_METRICS':str(p/'gpu_metrics.csv'),
            'EGOQA_SHARED_GPU_TRACE':str(p/'gpu_phase_trace.jsonl'),
            'EGOQA_GRPO_STAGE_AUDIT':str(root/'stage_state.json' if rank==0 else p/'stage_state.json')}


def validate_rank_bindings(groups):
    if len(groups)!=2 or any(len(g)!=4 for g in groups):
        raise ValueError('两个rank各须获得完整G4组')
    keys=[]
    for group in groups:
        values={(r['source_packet_id'],int(r['asker_index'])) for r in group}
        if len(values)!=1:raise ValueError('rank内G4组的窗口speaker错位')
        keys.append(next(iter(values)))
    if keys[0][0]==keys[1][0]:raise ValueError('同一步必须为两个不同窗口')
    return [k[0] for k in keys]


def validation_partition(count,rank):
    if count%2 or rank not in (0,1):raise ValueError('验证数量必须可均分两个rank')
    return list(range(rank,count,2))


def frozen_visual_key(inputs):
    """只为同一完整G4中的同序原始图路径启用缓存；其他输入直接回退。"""
    if len(inputs)!=4:return None
    keys=[]
    for row in inputs:
        images=row.get('generator_image_paths') or row.get('images')
        if not isinstance(images,(list,tuple)) or not images or not all(isinstance(p,str) and p for p in images):
            return None
        keys.append(tuple(images))
    return keys[0] if all(key==keys[0] for key in keys) else None


def make_length_pairs(rows,cost,seed=42):
    """每条件恰一次，同步两rank长度相近且不同窗口，再随机化pair顺序。"""
    from collections import Counter
    import random
    if len(rows)%2 or not rows:raise ValueError('条件数必须为正偶数')
    counts=Counter(row['source_packet_id'] for row in rows)
    if max(counts.values())>len(rows)//2:raise ValueError('无法构成不同窗口配对')
    remaining=sorted(rows,key=lambda r:(cost(r),r['source_packet_id'],int(r['asker_index'])))
    pairs=[]
    while remaining:
        left=remaining.pop(0)
        index=next((i for i,r in enumerate(remaining) if r['source_packet_id']!=left['source_packet_id']),None)
        if index is None:
            j=next((i for i,p in enumerate(pairs) if all(r['source_packet_id']!=left['source_packet_id'] for r in p)),None)
            if j is None:raise ValueError('无法修复末尾同窗口配对')
            a,b=pairs.pop(j);pairs.append((a,left));remaining.append(b)
            remaining.sort(key=cost)
        else:pairs.append((left,remaining.pop(index)))
    rng=random.Random(seed);rng.shuffle(pairs)
    totals=[0,0];ordered=[]
    for a,b in pairs:
        if (totals[0]-totals[1])*(cost(a)-cost(b))>0:a,b=b,a
        ordered.extend([a,b]);totals[0]+=cost(a);totals[1]+=cost(b)
    return ordered


def vision_cache_environment(config):
    enabled=config.get('frozen_vision_cache',False)
    limit=config.get('frozen_vision_cache_max_bytes',2*1024**3)
    if type(enabled) is not bool or type(limit) is not int or not 0<limit<=2*1024**3:
        raise ValueError('冻结视觉缓存必须显式布尔开关，且每rank CPU上限不超过2GiB')
    return {'EGOQA_FROZEN_VISION_CACHE':'1' if enabled else '0',
            'EGOQA_FROZEN_VISION_CACHE_MAX_BYTES':str(limit)}


def verify_zero_runtime_replacement(previous, expected_ids, sacct_text):
    """只允许替换本批已确认零运行且失败/取消的两个正式JobID。"""
    current={previous['jobs'][key]['job_id'] for key in ('lr1e5','lr5e6')}
    if len(expected_ids)!=2 or set(expected_ids)!=current:
        raise ValueError('指定替换JobID与当前回执不一致')
    records={row[0]:row for line in sacct_text.splitlines() if (row:=line.split('|'))[0] in current}
    if set(records)!=current:raise ValueError('实际终态记录不完整，禁止重复提交')
    for row in records.values():
        if len(row)<3:raise ValueError('终态记录字段不完整')
        terminal=row[1].split()[0] in {'CANCELLED','FAILED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY','PREEMPTED'}
        if not terminal or row[2]!='0':
            raise ValueError('旧正式任务不是已确认的零运行终态，禁止替换')
