"""三轮Judge反馈环评估：沿用 video_qa_loop 的生成→评审→feedback_to_generator→重生成流程（最多 max_attempts 轮）。

与原流水线的对应与差异（均为有意设置）：
- Generator prompt 用GRPO训练时的同一prompt（build_video_generation_prompt + 简洁输出要求）；反馈用原流水线的
  prompts._feedback_block 插在原位置（Evidence packet metadata 之前），与原流水线带反馈的prompt格式一致。
- 通过与否由已训练的三个冻结Judge按训练时的prompt和 verdict-token 概率判定，P(pass)>0.5 为通过（与GRPO奖励/通过率评估一致），
  三项全过才接受；不再跑原流水线的零样本Judge和strict validation。
- 判fail的Judge在 {"verdict":"fail 之后由同一Judge贪心续写 reason/fix(或missing_evidence)，按原流水线格式
  "{check}: {reason} {fix}" 以 " | " 拼接成反馈。QA本身不合法时按原流水线给出 "Generator output was not valid JSON" 类反馈。
"""
from __future__ import annotations
import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import time

KEYS = ('formality', 'groundedness', 'all_six')
CHECK_NAMES = {'formality': 'qa_formality', 'groundedness': 'evidence_groundedness', 'all_six': 'answerability'}
MARKER = '\n\n\nEvidence packet metadata:\n'
# judge.parse_completion 的中文错误 -> 给Generator的英文反馈
INVALID_FEEDBACK = {
    '生成结果不是合法 QA JSON': 'Generator output was not valid JSON.',
    'QA 必须是对象': 'Generator output was not a JSON object.',
    '必须有五个非空选项': 'Generator output was not valid: the item must have exactly five non-empty string options.',
    'correct 必须为 A–E': 'Generator output was not valid: correct must be one of A, B, C, D, E.',
    'answer 与 correct 不一致': 'Generator output was not valid: answer must exactly match the option selected by correct.',
    '缺少问题': 'Generator output was not valid: the question field is missing or empty.',
}


def inject_feedback(content, feedback, previous_generation):
    """在原流水线 feedback_block 的位置插入重试上下文；无反馈时原样返回（第1轮与GRPO训练prompt逐字相同）。"""
    if not feedback and not previous_generation:
        return content
    if content.count(MARKER) != 1:
        raise ValueError('Generator prompt 中找不到唯一的反馈插入位置')
    from training.judge_sft.prepare_real_data import _repo_module
    block = _repo_module('prompts')._feedback_block(feedback, previous_generation=previous_generation)
    return content.replace(MARKER, '\n\n' + block + '\nEvidence packet metadata:\n')


def parse_feedback(continuation):
    """把 {"verdict":"fail + 续写 还原为字段；JSON不完整时退回逐字段提取。"""
    text = '{"verdict":"fail' + continuation
    end = text.rfind('}')
    if end != -1:
        try:
            value = json.loads(text[:end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    fields = {}
    for key in ('reason', 'fix'):
        m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)' % key, text)
        if m:
            fields[key] = m.group(1).replace('\\"', '"')
    m = re.search(r'"missing_evidence"\s*:\s*\[(.*?)(?:\]|$)', text, re.S)
    if m:
        fields['missing_evidence'] = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
    return fields


def feedback_line(key, fields):
    parts = []
    for name in ('reason', 'fix'):
        value = fields.get(name)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())
    missing = fields.get('missing_evidence')
    if isinstance(missing, list) and any(isinstance(x, str) and x.strip() for x in missing):
        parts.append('Missing evidence: ' + '; '.join(x.strip() for x in missing if isinstance(x, str) and x.strip()))
    return f"{CHECK_NAMES[key]}: {' '.join(parts) or 'judge verdict is fail'}".strip()


def judge_feedback(failed, explained):
    """按原流水线 merge 的格式拼接：每个未通过的检查一段，用 ' | ' 连接。"""
    return ' | '.join(feedback_line(k, parse_feedback(explained[k]['continuation'])) for k in KEYS if k in failed)


def invalid_feedback(reason):
    return INVALID_FEEDBACK.get(reason, 'Generator output was not valid: ' + str(reason))


def new_items(rows, samples):
    return [{'evidence_id': r['evidence_id'], 'global_validation_index': int(r['global_validation_index']),
             'sample': s, 'attempts': [], 'accepted_attempt': None}
            for r in rows for s in range(samples)]


def pending(items, attempt):
    """还没接受、且尚未完成第 attempt 轮的题。"""
    return [i for i in items if i['accepted_attempt'] is None and len(i['attempts']) == attempt - 1]


def next_prompt_inputs(item):
    if not item['attempts']:
        return None, None
    last = item['attempts'][-1]
    return last['feedback_out'], last['completion']


def record_attempt(item, attempt, seed, feedback_in, completion, result, explained):
    passed = {k: result['status'] == 'scored' and result['probabilities'][k] > .5 for k in KEYS}
    accepted = all(passed.values())
    if result['status'] != 'scored':
        feedback_out = invalid_feedback(result.get('reason'))
    elif accepted:
        feedback_out = None
    else:
        feedback_out = judge_feedback([k for k in KEYS if not passed[k]], explained)
    item['attempts'].append({'attempt': attempt, 'seed': seed, 'feedback_in': feedback_in, 'completion': completion,
                             'status': result['status'], 'invalid_reason': result.get('reason'),
                             'probabilities': result.get('probabilities', {}), 'pass': passed, 'all_pass': accepted,
                             'judge_feedback_raw': explained, 'feedback_out': feedback_out})
    if accepted:
        item['accepted_attempt'] = attempt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--judge-url', required=True)
    parser.add_argument('--judge-instance', required=True)
    parser.add_argument('--adapter', type=Path)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--max-attempts', type=int, default=3)
    parser.add_argument('--chunk', type=int, default=6)
    parser.add_argument('--feedback-max-tokens', type=int, default=384)
    args = parser.parse_args()
    c = json.loads(args.config.read_text(encoding='utf-8'))
    from .evaluation import configure_engine_process, configure_engine_kwargs, policy_template
    configure_engine_process(c, os.environ)
    if 'in_proj_qkv' in c.get('lora_target_modules', []):
        from .packed_lora_compat import install
        install()
    from .utilization_runtime import enable_training_guard
    enable_training_guard()
    from .data import read_rows
    from .service import JudgeClient
    rows = {r['evidence_id']: r for r in read_rows(c['val_dataset'])}
    if args.adapter and not (args.adapter / 'adapter_model.safetensors').is_file():
        raise FileNotFoundError(args.adapter / 'adapter_model.safetensors')
    client = JudgeClient(args.judge_url, expected_instance=args.judge_instance)
    settings = {k: c[k] for k in ('policy_model', 'max_length', 'max_completion_length', 'max_pixels', 'temperature', 'top_p', 'top_k')}
    settings.update(validation_seed=c.get('validation_seed', 42), samples=args.samples, max_attempts=args.max_attempts,
                    chunk=args.chunk, feedback_max_tokens=args.feedback_max_tokens,
                    pass_criterion='P(pass)>0.5 for each judge; accept iff all three pass')
    if args.output.exists():
        record = json.loads(args.output.read_text(encoding='utf-8'))
        if record['settings'] != settings or record['adapter'] != (str(args.adapter) if args.adapter else None):
            raise ValueError('续跑设置与已有结果不一致')
        record['resumed_epochs'] = record.get('resumed_epochs', []) + [time.time()]
    else:
        record = {'status': 'running', 'settings': settings, 'adapter': str(args.adapter) if args.adapter else None,
                  'started_epoch': time.time(), 'items': new_items(list(rows.values()), args.samples)}
    record['judge'] = client.health()

    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1, allow_nan=False), encoding='utf-8')
        os.replace(tmp, args.output)
    save()
    template = policy_template(c)
    from swift.infer_engine import VllmEngine, InferRequest, RequestConfig
    import torch
    engine = VllmEngine(c['policy_model'], template=template, torch_dtype=torch.bfloat16,
        enable_sleep_mode=bool(c.get('shared_gpu')),
        adapters=[str(args.adapter)] if args.adapter else None, enable_lora=True, max_lora_rank=c.get('lora_rank', 8),
        max_model_len=c['max_length'], max_num_seqs=min(args.chunk, 4),
        gpu_memory_utilization=c.get('vllm_gpu_memory_utilization', .55), tensor_parallel_size=1,
        enable_prefix_caching=True, mm_processor_cache_gb=4, limit_mm_per_prompt={'image': 1800},
        seed=settings['validation_seed'], engine_kwargs=configure_engine_kwargs(c,
            {'enable_chunked_prefill': True, 'max_num_batched_tokens': 8192,
             'additional_config': {'gdn_prefill_backend': 'triton'}}))
    from .shared_gpu import evaluation_judge_window
    for attempt in range(1, args.max_attempts + 1):
        todo = pending(record['items'], attempt)
        for chunk_index, start in enumerate(range(0, len(todo), args.chunk)):
            chunk = todo[start:start + args.chunk]
            base_seed = settings['validation_seed'] + 1_000_000 * attempt + 1000 * chunk_index + chunk[0]['global_validation_index']
            requests, contexts = [], []
            for item in chunk:
                row = rows[item['evidence_id']]
                feedback, previous = next_prompt_inputs(item)
                messages = deepcopy(row['messages'])
                messages[0]['content'] = inject_feedback(messages[0]['content'], feedback, previous)
                requests.append(InferRequest(messages=messages, images=list(row['images'])))
                contexts.append(feedback)
            decoding = RequestConfig(n=1, seed=base_seed, max_tokens=c['max_completion_length'],
                                     temperature=c['temperature'], top_p=c['top_p'], top_k=c['top_k'])
            responses = engine.infer(requests, decoding, use_tqdm=False)
            if len(responses) != len(chunk):
                raise RuntimeError('生成返回数量不一致')
            judge_requests = []
            for i, (item, response) in enumerate(zip(chunk, responses)):
                row = rows[item['evidence_id']]
                judge_requests.append({**row, 'request_id': f"{item['evidence_id']}:s{item['sample']}:a{attempt}",
                                       'completion': response.choices[0].message.content})
            with evaluation_judge_window(client, engine) if c.get('shared_gpu') else nullcontext():
                judged = client.score_many(judge_requests)
                explain = [{**r, 'keys': [k for k in KEYS if j['probabilities'][k] <= .5], 'max_tokens': args.feedback_max_tokens}
                           for r, j in zip(judge_requests, judged) if j['status'] == 'scored']
                explain = [r for r in explain if r['keys']]
                explained = {r['request_id']: e['feedback'] for r, e in zip(explain, client.explain_many(explain))} if explain else {}
            for i, (item, request, result) in enumerate(zip(chunk, judge_requests, judged)):
                record_attempt(item, attempt, base_seed + i, contexts[i], request['completion'], result,
                               explained.get(request['request_id'], {}))
            save()
    record.update(status='completed', finished_epoch=time.time())
    save()


if __name__ == '__main__':
    main()
