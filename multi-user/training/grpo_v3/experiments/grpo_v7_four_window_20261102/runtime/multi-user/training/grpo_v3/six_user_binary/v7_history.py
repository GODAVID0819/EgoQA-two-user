"""仅在优化器完成更新后推进、两卡一致的 speaker 统计。"""
from collections import deque
import os

WINDOW = 8
INSTANCES = {}


def history_snapshot(counts):
    hits = sum(pair[0] for pair in counts)
    valid = sum(pair[1] for pair in counts)
    probability = hits / valid if valid else None
    penalty = min(.20, max(0., probability - .20)) if valid else 0.
    return dict(history_hits=hits, history_valid=valid, history_accuracy=probability,
                speaker_lambda=penalty)


class SpeakerHistory:
    def __init__(self):
        self.counts = deque(maxlen=WINDOW)
        self.completed_step = 0
        self.pending = None

    def snapshot(self):
        return {**history_snapshot(self.counts), 'history_completed_step': self.completed_step}

    def stage(self, step, hits, request_ids):
        if step != self.completed_step + 1 or self.pending is not None:
            raise RuntimeError('评分必须每次更新恰调用一次，历史只能由完成的更新推进')
        if len(hits) != len(request_ids) or len(set(request_ids)) != len(request_ids):
            raise RuntimeError('暂存候选 ID 不完整或重复')
        self.pending = dict(step=step, hits=sum(x for x in hits if x is not None),
                            valid=sum(x is not None for x in hits), count=len(hits))

    def commit(self, step, gather):
        if self.pending is None or step != self.pending['step']:
            raise RuntimeError('优化器更新和待提交 speaker 统计不对应')
        shard = {**self.pending, 'before': self.snapshot()}
        shards = gather(shard)
        if len(shards) != 2 or any(s['step'] != step or s['count'] != 8 for s in shards):
            raise RuntimeError('两卡必须同一步、各八个真实槽')
        if any(not 0 <= s['hits'] <= s['valid'] <= 8 for s in shards):
            raise RuntimeError('两卡盲答计数非法')
        if any(s.get('before') != shard['before'] for s in shards):
            raise RuntimeError('两个 rank 使用的历史或系数不一致')
        self.counts.append((sum(s['hits'] for s in shards), sum(s['valid'] for s in shards)))
        self.completed_step = step
        self.pending = None
        return self.snapshot()


def restore_history(source_output,step):
    import json
    from pathlib import Path
    from .v7_batch import replay_history
    from .v7_speaker import parse_letter
    rows=[]
    for rank in (0,1):
        part=[json.loads(line) for line in (Path(source_output)/f'rank{rank}/reward_trace.jsonl').read_text().splitlines() if line.strip()][:step*8]
        if len(part)!=step*8:raise ValueError('恢复历史缺少完整rank轨迹')
        rows.extend(part)
    if not replay_history(rows,step):raise ValueError('恢复历史无法从原始盲答重算')
    result=SpeakerHistory()
    for update in range(1,step+1):
        hits=valid=0
        for row in rows:
            if row['global_step']!=update or row['judge_result']['status']!='scored':continue
            speaker=row['judge_result']['speaker'];letter=parse_letter(speaker['raw_output'])
            if letter is not None:valid+=1;hits+=int(letter==speaker['gold'])
        result.counts.append((hits,valid))
    result.completed_step=step
    return result


def current_history():
    key = os.environ['EGOQA_GRPO_V3_REWARD_TRACE']
    if key not in INSTANCES:
        import json
        metadata=json.loads(os.environ.get('EGOQA_RESUME_METADATA','null'))
        INSTANCES[key]=restore_history(metadata['source_output'],metadata['step']) if metadata else SpeakerHistory()
    return INSTANCES[key]


def commit_distributed(step):
    import torch.distributed as dist
    def gather(local):
        shards = [None] * dist.get_world_size()
        dist.all_gather_object(shards, local)
        return shards
    return current_history().commit(step, gather)
