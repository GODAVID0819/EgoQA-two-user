"""双卡运行的纯参数、路径和完整G4组约束。"""
from pathlib import Path
import re


def condition_binding(row):
    match=re.match(r'^RLHF6U_(DAY\d+)_(\d{8})_',row['source_packet_id'])
    if not match:raise ValueError('窗口标识无法解析')
    return (match.group(1)+'_'+match.group(2),int(row.get('asker_index',row.get('speaker_index'))))


def validate_config(c):
    # 每步4个不同窗口：每rank每4微批生成一轮（1条件×G4）。两卡：每rank累积8微批、两轮；四卡：累积4微批、一轮。
    world=c.get('world_size')
    if world not in WORLD_SIZES:raise ValueError('world_size只允许2或4')
    expected={'num_generations':4,'per_device_train_batch_size':1,
              'gradient_accumulation_steps':4*rounds_per_step(world),'max_steps':900,'lora_rank':128,
              'lora_alpha':128,'train_conditions':600,'validation_conditions':120}
    if any(c.get(k)!=v for k,v in expected.items()):
        raise ValueError('多卡参数不符合四窗口G4、r128、900步配置')
    if c.get('resume_from_checkpoint'):
        from recovery import validated_resume
        validated_resume(c)
    if not 0<c.get('stop_after_steps',c['max_steps'])<=c['max_steps']:
        raise ValueError('实际停止步数超出调度总步数')


def rank_environment(output, records, rank):
    if len(records) not in WORLD_SIZES or rank not in range(len(records)) or len({r['port'] for r in records})!=len(records):
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
    if len(groups) not in WORLD_SIZES or any(len(g)!=4 for g in groups):
        raise ValueError('每个rank各须获得完整G4组')
    keys=[]
    for group in groups:
        values={(r['source_packet_id'],int(r['asker_index'])) for r in group}
        if len(values)!=1:raise ValueError('rank内G4组的窗口speaker错位')
        keys.append(next(iter(values)))
    if len({k[0] for k in keys})!=len(keys):raise ValueError('同一轮各rank必须为不同窗口')
    return [k[0] for k in keys]


ROWS_PER_STEP=4
WORLD_SIZES=(2,4)


def rounds_per_step(world_size):
    """文件第4k..4k+3行为第k步；第j轮生成取4k+W·j..4k+W·j+W-1行，第r行给rank r（TRL BatchSamplerShard按批轮转）。"""
    if world_size not in WORLD_SIZES:raise ValueError('world_size只允许2或4')
    return ROWS_PER_STEP//world_size


def step_row(step,round_index,rank,world_size):
    return ROWS_PER_STEP*step+world_size*round_index+rank


def make_length_quads(rows,cost,seed=42):
    """每条件恰一次；每步4个不同窗口、长度相近，两轮生成各一对(rank0,rank1)，再随机化步顺序并均衡rank。

    文件顺序第4k..4k+3行为第k步：第一轮(4k→rank0, 4k+1→rank1)，第二轮(4k+2→rank0, 4k+3→rank1)。"""
    import random
    if len(rows)%ROWS_PER_STEP or not rows:raise ValueError('条件数必须为4的正整数倍')
    remaining=sorted(rows,key=lambda r:(cost(r),r['source_packet_id'],int(r['asker_index'])))
    quads=[]
    while remaining:
        quad=[remaining.pop(0)]
        for i in range(len(remaining)):
            if len(quad)==ROWS_PER_STEP:break
            if all(remaining[i]['source_packet_id']!=q['source_packet_id'] for q in quad):quad.append(remaining[i])
        if len(quad)<ROWS_PER_STEP:raise ValueError('无法构成4个不同窗口的更新')
        for q in quad[1:]:remaining.remove(q)
        quads.append(quad)
    rng=random.Random(seed);rng.shuffle(quads)
    totals=[0,0];ordered=[]
    for quad in quads:
        for a,b in (quad[0:2],quad[2:4]):
            if (totals[0]-totals[1])*(cost(a)-cost(b))>0:a,b=b,a
            ordered.extend([a,b]);totals[0]+=cost(a);totals[1]+=cost(b)
    return ordered


def check_step_order(rows):
    if len(rows)%ROWS_PER_STEP:raise ValueError('训练行数必须为4的整数倍')
    for start in range(0,len(rows),ROWS_PER_STEP):
        if len({r['source_packet_id'] for r in rows[start:start+ROWS_PER_STEP]})!=ROWS_PER_STEP:
            raise ValueError(f'第{start//ROWS_PER_STEP+1}步不是4个不同窗口')


def validation_partition(count,rank,world_size=2):
    if world_size not in WORLD_SIZES or count%world_size or rank not in range(world_size):
        raise ValueError('验证数量必须可均分各rank')
    return list(range(rank,count,world_size))


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
