"""逐候选连续奖励、严格身份对应及分阶段 GPU 使用。"""
from __future__ import annotations
import math
try:
    from .v7_speaker import parse_letter, blind_payload, FrozenSpeaker
except ImportError:
    from v7_speaker import parse_letter, blind_payload, FrozenSpeaker


def invalid_parts():
    return dict(reward=0., quality_reward=0., diversity_distance=0., diversity_bonus=0.,
        bonus_gate=False, legal_qa=False, speaker_hit=None, speaker_parse_status='not_run',
        speaker_adjustment=0., reference_count=0, current_reference_count=0,
        history_reference_count=0, nearest_neighbor_id=None, nearest_neighbor_source=None)


def reward_parts(probabilities, raw_output, gold, *, history):
    if set(probabilities) != {'formality', 'groundedness', 'combined_answerability'}:
        raise ValueError('v7 必须使用三项 F/E/C 概率')
    if any(isinstance(p, bool) or not isinstance(p, (int,float)) or not math.isfinite(p) or not 0<=p<=1 for p in probabilities.values()):
        raise ValueError('Judge 概率非法')
    if gold not in tuple('ABCDE'):
        raise ValueError('缺少合法声明答案')
    pred = parse_letter(raw_output)
    hit = None if pred is None else int(pred == gold)
    j = .2*probabilities['formality'] + .4*probabilities['groundedness'] + .4*probabilities['combined_answerability']
    adjustment = -history['speaker_lambda'] * hit if hit is not None else 0.
    return dict(reward=j+adjustment, quality_reward=j, diversity_distance=0., diversity_bonus=0.,
        bonus_gate=all(p>=.5 for p in probabilities.values()), legal_qa=True,
        speaker_hit=hit, speaker_prediction=pred,
        speaker_parse_status='valid' if pred else 'invalid',
        speaker_adjustment=adjustment, **history, reference_count=0, current_reference_count=0,
        history_reference_count=0, nearest_neighbor_id=None, nearest_neighbor_source=None)


def score_group(requests, results, *, encoder=None, history_by_window=None, diversity_lambda=0, history=None):
    if history is None:
        raise ValueError('奖励必须绑定更新前的历史快照')
    if diversity_lambda != 0:
        raise ValueError('v7 禁止历史池或多样性奖励')
    if len(requests) != 8 or len(results) != 8:
        raise RuntimeError('每 rank 一次评分必须包含两个完整 G4')
    if len({r['request_id'] for r in requests})!=8:
        raise RuntimeError('候选 ID 重复')
    groups=[]
    for start in (0,4):
        keys={(r['source_packet_id'],int(r['asker_index'])) for r in requests[start:start+4]}
        if len(keys)!=1:
            raise RuntimeError('每四条必须来自同一窗口与 speaker')
        groups.append(next(iter(keys)))
    if groups[0]==groups[1]:
        raise RuntimeError('两个 G4 不能来自相同条件')
    parts=[]
    for req,res in zip(requests,results):
        if any(req[k] != res.get(k) for k in ('request_id','evidence_id')):
            raise RuntimeError('候选身份或顺序不匹配')
        if res['status']=='invalid_completion':
            parts.append({**invalid_parts(), **history})
        elif res['status']=='scored':
            s=res['speaker']
            parts.append(reward_parts(res['probabilities'],s['raw_output'],s['gold'], history=history))
        else:
            raise RuntimeError('未知评分状态')
        parts[-1]['request_id']=req['request_id']
        parts[-1]['generation_group_index']=(len(parts)-1)//4
    return parts


def summarize(parts):
    valid=[p['speaker_hit'] for p in parts if p['speaker_hit'] is not None]
    legal=sum(p['legal_qa'] for p in parts)
    complete_groups=[parts[i:i+4] for i in range(0,len(parts),4) if len(parts[i:i+4])==4]
    mixed=sum({p['speaker_hit'] for p in group if p['speaker_hit'] is not None}=={0,1} for group in complete_groups)
    variances=[]
    for group in complete_groups:
        average=sum(p['reward'] for p in group)/4
        variances.append(sum((p['reward']-average)**2 for p in group)/4)
    return dict(total_slots=len(parts), legal_qa_count=sum(p['legal_qa'] for p in parts),
        speaker_mixed_group_count=mixed, speaker_mixed_group_rate=mixed/len(complete_groups) if complete_groups else None,
        g4_reward_variances=variances, group_reward_variance=sum(variances)/len(variances) if variances else None,
        zero_advantage_group=bool(variances) and all(v==0 for v in variances),
        zero_advantage_group_count=sum(v==0 for v in variances),
        legal_qa_rate=sum(p['legal_qa'] for p in parts)/len(parts) if parts else None,
        speaker_valid_count=len(valid), speaker_parse_failure_count=sum(p['speaker_parse_status']=='invalid' for p in parts),
        speaker_valid_rate_over_legal_qa=len(valid)/legal if legal else None,
        speaker_hit_count=sum(valid), speaker_hit_rate_over_all_slots=sum(valid)/len(parts) if parts else None,
        speaker_accuracy=sum(valid)/len(valid) if valid else None,
        speaker_accuracy_minus_chance=sum(valid)/len(valid)-.20 if valid else None)


def speaker_window(judge, speaker, payloads):
    if judge.resource('sleep').get('sleeping') is not True:
        raise RuntimeError('Judge 未确认释放显存')
    try:
        return speaker.predict_many(payloads)
    finally:
        # 释放失败直接抛出；绝不能在状态未知时唤醒 Judge。
        speaker.sleep()


def score_candidates(scorer, requests, score_fec):
    from .judge import parse_completion, full_judge_view
    if scorer.predictor.resource('wake').get('sleeping') is not False:
        raise RuntimeError('Judge 未确认唤醒')
    results=score_fec(requests)
    if len(results)!=len(requests):
        raise RuntimeError('F/E/C 返回数量不一致')
    config=scorer.predictor.config['v7']
    bindings=[]
    for req,res in zip(requests,results):
        if any(req[k]!=res.get(k) for k in ('request_id','evidence_id')):
            raise RuntimeError('F/E/C 候选对应错误')
        if res['status']!='scored':
            continue
        qa=parse_completion(req['completion'])
        view=full_judge_view(req['dataset_root'],req['source_packet_id'],req['asker_index'])
        clips=view['clips']
        if not clips or clips[0]['agent_name'] != view['required_users'][0] or clips[0]['media_role']!='speaker':
            # 具体字段由实际 view 校验；不得从生成器声明推断 speaker。
            raise ValueError('speaker 顺序与真实输入不一致')
        frames=[f['path'] for f in clips[0]['frames']]
        bindings.append((res,qa,blind_payload(qa,frames),view['required_users'][0]))
    raw=['']*len(bindings)
    if bindings:
        if not hasattr(scorer,'v7_speaker'):
            scorer.v7_speaker=FrozenSpeaker(config['speaker'])
        raw=speaker_window(scorer.predictor,scorer.v7_speaker,[b[2] for b in bindings])
        if len(raw)!=len(bindings):
            raise RuntimeError('speaker 返回数量不一致')
    for (res,qa,payload,speaker_user),output in zip(bindings,raw):
        res.update(speaker=dict(raw_output=output,gold=qa['correct'],
            input_users=[speaker_user],
            frame_count=len(payload['frames']), model_id=config['speaker']['model_id'],
            prompt=payload['prompt'], batch_metrics=getattr(scorer.v7_speaker,'last_metrics',{})))
    return results
