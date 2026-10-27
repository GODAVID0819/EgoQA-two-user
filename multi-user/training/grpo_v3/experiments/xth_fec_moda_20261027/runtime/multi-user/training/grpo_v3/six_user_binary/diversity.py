"""当前真实G4与固定同窗历史参照；历史题不进入GRPO优势分母。"""
from __future__ import annotations

import math
import re

import numpy as np

from training.grpo_v3.six_user_binary.judge import parse_completion
from training.grpo_v3.six_user_binary.reward import QUALITY_MODES, aggregate, validate_probabilities


def window_id(source_packet_id):
    match = re.match(r'^RLHF6U_(DAY\d+_\d{8})_', source_packet_id)
    if not match:
        raise ValueError('无法从真实packet提取同步窗口')
    return match.group(1)


def checked_vectors(values, *, expected_count=None, dimension=None):
    vectors = np.asarray(values, dtype=np.float64)
    if (vectors.ndim != 2 or (expected_count is not None and len(vectors) != expected_count)
            or not vectors.shape[1] or not np.isfinite(vectors).all()
            or (dimension is not None and vectors.shape[1] != dimension)
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1., atol=1e-5, rtol=0)):
        raise ValueError('问题向量必须有限、非零、维度一致且L2归一化')
    return vectors


def score_group(requests, results, *, encoder=None, history_by_window=None, diversity_lambda=.5, mode='continuous'):
    if mode not in QUALITY_MODES:
        raise ValueError(f'质量奖励只能是 {QUALITY_MODES}')
    if isinstance(diversity_lambda, bool) or not math.isfinite(diversity_lambda) or diversity_lambda < 0:
        raise ValueError('距离权重必须是非负有限数')
    if len(requests) != 4 or len(results) != 4:
        raise ValueError('训练奖励只接受当前真实G4，不把历史QA混入生成组')
    ids = [r['request_id'] for r in requests]
    result_ids = [r['request_id'] for r in results]
    if len(set(ids)) != 4 or len(set(result_ids)) != 4 or set(ids) != set(result_ids):
        raise ValueError('候选ID重复、缺失或响应错位')
    bindings = {(r['source_packet_id'], int(r['asker_index'])) for r in requests}
    if len(bindings) != 1:
        raise ValueError('当前G4必须是同一真实窗口、speaker及rollout')
    window = window_id(requests[0]['source_packet_id'])
    by_id = {r['request_id']: r for r in results}
    valid = []
    output = []
    for i, request in enumerate(requests):
        result = by_id[request['request_id']]
        if result['status'] not in {'scored', 'invalid_completion'}:
            raise ValueError('基础设施或Judge失败不能伪装成零奖励')
        scored = result['status'] == 'scored'
        scores = validate_probabilities(result['probabilities']) if scored else None
        quality = aggregate(scores, mode=mode) if scored else 0.
        gate = bool(scored and all(value >= .5 for value in scores.values()))
        output.append({'request_id': request['request_id'], 'quality_reward': quality,
                       'bonus_gate': gate, 'diversity_distance': 0., 'diversity_bonus': 0.,
                       'reward': quality, 'reference_count': 0, 'current_reference_count': 0,
                       'history_reference_count': 0, 'nearest_neighbor_id': None,
                       'nearest_neighbor_source': None})
        if scored:
            valid.append((i, parse_completion(request['completion'])['question']))
    if not valid or diversity_lambda == 0:
        return output
    if encoder is None:
        raise ValueError('非零距离权重必须提供冻结问题编码器')
    vectors = checked_vectors(encoder.encode([question for _, question in valid]), expected_count=len(valid))
    history = (history_by_window or {}).get(window, [])
    if len(history) > 8 or len({row['history_qa_id'] for row in history}) != len(history):
        raise ValueError('同窗历史池最多8题且ID必须唯一')
    history_vectors = checked_vectors([r['embedding'] for r in history], dimension=vectors.shape[1]) if history else None
    for vector_index, (candidate_index, _) in enumerate(valid):
        neighbors = [(requests[other_index]['request_id'], 'current_G4', vectors[j])
                     for j, (other_index, _) in enumerate(valid) if other_index != candidate_index]
        neighbors.extend((r['history_qa_id'], 'history', history_vectors[j]) for j, r in enumerate(history))
        record = output[candidate_index]
        record.update(reference_count=len(neighbors), current_reference_count=len(valid)-1,
                      history_reference_count=len(history))
        if neighbors:
            similarities = np.clip(np.asarray([r[2] for r in neighbors]) @ vectors[vector_index], -1., 1.)
            nearest = int(np.argmax(similarities))
            distance = float((1. - similarities[nearest]) / 2.)
            record.update(diversity_distance=distance, nearest_neighbor_id=neighbors[nearest][0],
                          nearest_neighbor_source=neighbors[nearest][1])
            record['diversity_bonus'] = diversity_lambda * int(record['bonus_gate']) * distance
            record['reward'] = record['quality_reward'] + record['diversity_bonus']
    return output
