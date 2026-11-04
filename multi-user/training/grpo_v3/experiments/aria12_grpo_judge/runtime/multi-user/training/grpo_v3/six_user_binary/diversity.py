"""纯F/E/C奖励；保留真实G4绑定，不计算问题多样性。"""
from __future__ import annotations
import math
import re
from .judge import parse_completion
from .reward import aggregate, validate_probabilities


def window_id(source_packet_id):
    match=re.match(r'^ARIA12_(DAY\d+_\d{8})_',source_packet_id)
    if not match: raise ValueError('三人packet窗口标识不合法')
    return match.group(1)


def score_group(requests, results, *, encoder=None, history_by_window=None, diversity_lambda=0, mode='continuous_gated'):
    if mode!='continuous_gated': raise ValueError('本实验只使用continuous质量奖励')
    if type(diversity_lambda) not in (int,float) or not math.isfinite(diversity_lambda) or diversity_lambda!=0:
        raise ValueError('本实验已取消多样性奖励，diversity_lambda必须为0')
    if len(requests)!=4 or len(results)!=4: raise ValueError('每个rank必须有完整G4')
    ids=[r['request_id'] for r in requests];result_ids=[r['request_id'] for r in results]
    if len(set(ids))!=4 or len(set(result_ids))!=4 or set(ids)!=set(result_ids):
        raise ValueError('候选ID重复、缺失或响应错位')
    bindings={(r['source_packet_id'],int(r['asker_index'])) for r in requests}
    if len(bindings)!=1: raise ValueError('当前G4必须来自同一窗口和提问者')
    window_id(requests[0]['source_packet_id'])
    if requests[0]['asker_index'] not in range(12): raise ValueError('三人提问者索引越界')
    by_id={r['request_id']:r for r in results}
    output=[]
    for request in requests:
        result=by_id[request['request_id']]
        if result['status'] not in {'scored','invalid_completion'}:
            raise ValueError('基础设施或Judge失败不能伪装成零奖励')
        scored=result['status']=='scored'
        scores=validate_probabilities(result['probabilities']) if scored else None
        if scored: parse_completion(request['completion'])
        quality=aggregate(scores,mode=mode) if scored else 0.0
        # 仅保留三项通过统计，不用该标记改变质量奖励。
        passed=bool(scored and all(value is not None and value>=.5 for value in scores.values()))
        output.append({'request_id':request['request_id'],'quality_reward':quality,'reward':quality,
            'bonus_gate':passed,'diversity_distance':0.0,'diversity_bonus':0.0,
            'reference_count':0,'current_reference_count':0,'history_reference_count':0,
            'nearest_neighbor_id':None,'nearest_neighbor_source':None})
    return output
