"""四窗分组和可重放的历史奖励验收。"""
from collections import deque
import math
import re
import json
from pathlib import Path


def condition_key(row):
    match = re.match(r'^RLHF6U_(DAY\d+_\d{8})_', row['source_packet_id'])
    if not match:
        raise ValueError('无法解析真实窗口')
    return match.group(1), int(row['asker_index'])


def validate_four_bindings(groups):
    if len(groups) != 2 or any(len(group) != 8 for group in groups):
        raise ValueError('两卡各须包含八个候选')
    windows = []
    for group in groups:
        for start in (0,4):
            keys = {condition_key(row) for row in group[start:start+4]}
            if len(keys) != 1:
                raise ValueError('G4 组内真实条件不一致')
            windows.append(next(iter(keys))[0])
    if len(set(windows)) != 4:
        raise ValueError('同一次更新必须来自四个不同窗口')
    return windows


def replay_history(rows, steps):
    """从两个rank的原始盲答独立回放，不信任轨迹自报的lambda。"""
    history = deque(maxlen=8)
    seen = set()
    for step in range(1,steps+1):
        batch = [row for row in rows if row['global_step'] == step]
        if len(batch) != 16:
            return False
        hits = sum(pair[0] for pair in history)
        valid = sum(pair[1] for pair in history)
        penalty = min(.2,max(0.,hits/valid-.2)) if valid else 0.
        snapshot = dict(history_hits=hits, history_valid=valid,
                        history_accuracy=hits/valid if valid else None,
                        history_completed_step=step-1, speaker_lambda=penalty)
        new_hits = new_valid = 0
        for row in batch:
            if row['request_id'] in seen:
                return False
            seen.add(row['request_id'])
            result = row['judge_result']
            if any(row.get(key) != value for key,value in snapshot.items()):
                return False
            if result['status'] == 'invalid_completion':
                if row['reward'] != 0 or row.get('speaker_hit') is not None:
                    return False
                continue
            raw = result['speaker']['raw_output'].strip()
            parsed = raw if raw in tuple('ABCDE') else None
            hit = int(parsed == result['speaker']['gold']) if parsed else None
            p = result['probabilities']
            quality = .2*p['formality']+.4*p['groundedness']+.4*p['combined_answerability']
            expected = quality-penalty*hit if hit is not None else quality
            if row.get('speaker_hit') != hit or not math.isclose(row['reward'],expected,abs_tol=1e-10):
                return False
            new_hits += hit or 0
            new_valid += hit is not None
        history.append((new_hits,new_valid))
    return True


def throughput_summary(output, rows):
    """按两rank已完成更新的较慢时长报告稳态吞吐，启动开销另列。"""
    timings=[]
    for rank in (0,1):
        path=Path(output)/f'rank{rank}/phase_timing.jsonl'
        if not path.exists():
            return dict(status='pending_timing')
        items=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        timings.append({row['global_step']:row['seconds'] for row in items
                        if row.get('phase')=='optimizer_step_wall'})
    steps=sorted(set(timings[0]) & set(timings[1]))
    steady_steps=steps[1:]
    seconds=sum(max(timings[0][step],timings[1][step]) for step in steady_steps)
    measured=[row for row in rows if row['global_step'] in steady_steps]
    if seconds<=0 or not measured:
        return dict(status='pending_timing')
    return dict(status='measured',completed_update_count=len(steps),steady_measured_updates=len(steady_steps),
        first_update_seconds=max(timings[0][steps[0]],timings[1][steps[0]]) if steps else None,
        steady_update_seconds=seconds,
        candidate_slots_per_hour=len(measured)*3600/seconds,
        legal_qa_per_hour=sum(row['legal_qa'] for row in measured)*3600/seconds,
        scope='逐更新两rank较慢用时，排除本作业第一个完成更新；作业启动和排队未计入')

