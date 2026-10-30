"""六用户 MCQ 的 VQA 评测（Qwen2.5-VL-7B-Instruct / Qwen3-VL-8B-Instruct + vLLM）。

数据：human_review/curated_pass/vqa.json（169 题，人工全 pass）。两种设置只差输入哪些视频，其它完全相同：
  all6       六个人的视频（speaker 在前）
  evidence2  vqa.json 的 evidence_videos：生成器 evidence 里的两个人，优先非 speaker（不足两人时用 speaker 补；1 题只有 speaker）
  speaker1   只给 speaker 自己的视频（题目设计上应答不出）
  speaker_plus1  speaker + evidence_users 里排第一的非 speaker（与答案重合词最多；只有 speaker 被引用的 1 题只给 speaker）
每段视频（10 分钟）均匀取 --frames 帧、缩放到 --size×--size（默认 Qwen2.5-VL 336、Qwen3-VL 320，各为 patch×2 的倍数）；每段视频前标注是谁的第一视角、谁是提问者（问题里的“I”）。
--shuffle-seed 时每题选项按 (seed, 题id) 固定随机重排，正确字母随之更新（去掉原数据里答案集中在 A/B 的位置偏差）。
贪心解码，只要选项字母；解析不到字母记为错。输出 predictions.jsonl 与 summary.json（总体、按来源、按 evidence 选择方式）。
  python vqa_qwen25vl.py --setting all6 --out <dir> [--limit 3]
"""
import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path('/scratch/tx856/multi_user_qa/dual_full120/human_review/curated_pass/vqa.json')
VIDEO_ROOT = DATA.parent.parent  # vqa.json 里的视频路径相对 human_review/


def build_messages(item, videos, frames, size):
    speaker = item['speaker']
    content = [{'type': 'text', 'text':
                f'The following {len(videos)} videos are first-person (egocentric) recordings made at the same time by '
                f'different people in the same household, all covering the same 10-minute period.'}]
    for n, v in enumerate(videos, 1):
        tag = ' This person is the one asking the question.' if v['agent_name'] == speaker else ''
        content.append({'type': 'text', 'text': f"\nVideo {n}: {v['agent_name']}'s first-person view.{tag}"})
        content.append({'type': 'video', 'video': str(VIDEO_ROOT / v['path']), 'nframes': frames,
                        'resized_height': size, 'resized_width': size})
    options = '\n'.join(f'{k}. {t}' for k, t in item['options'].items())
    content.append({'type': 'text', 'text':
                    f'\n\nThe question is asked by {speaker}; "I", "me" and "my" refer to {speaker}.\n'
                    f'Question: {item["question"]}\nOptions:\n{options}\n'
                    'Answer with the letter of the correct option only.'})
    return [{'role': 'user', 'content': content}]


def shuffle_options(item, seed):
    """按 (seed, id) 固定重排五个选项；返回新 item 与原字母顺序（新 A..E 各对应原来哪个字母）。"""
    letters = list(item['options'])
    order = letters[:]
    random.Random(f'{seed}:{item["id"]}').shuffle(order)
    options = {new: item['options'][old] for new, old in zip(letters, order)}
    return {**item, 'options': options, 'correct': letters[order.index(item['correct'])]}, order


def parse_letter(text):
    m = re.search(r'\b([A-E])\b', text.strip().upper())
    return m.group(1) if m else None


def summarize(rows):
    def acc(rs):
        return {'n': len(rs), 'correct': sum(r['is_correct'] for r in rs),
                'accuracy': round(sum(r['is_correct'] for r in rs) / len(rs), 4) if rs else None}
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        groups['source'][r['source']].append(r)
        groups['evidence_users_selection'][r['evidence_users_selection']].append(r)
    return {'overall': acc(rows), 'unparsed': sum(r['pred'] is None for r in rows),
            **{g: {k: acc(v) for k, v in sorted(d.items())} for g, d in groups.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', choices=('all6', 'evidence2', 'speaker1', 'speaker_plus1'), required=True)
    ap.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--frames', type=int, default=64, help='每段视频取帧数（偶数）')
    ap.add_argument('--size', type=int, help='帧缩放边长；默认 Qwen2.5-VL 336（28 的倍数）、Qwen3-VL 320（32 的倍数）')
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--max-model-len', type=int, default=32768)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--shuffle-seed', type=int)
    ap.add_argument('--ids-file', type=Path, help='只跑这些题（JSON id 列表）')
    a = ap.parse_args()
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    patch = 16 if 'Qwen3-VL' in a.model else 14
    a.size = a.size or (320 if patch == 16 else 336)
    if a.size % (2 * patch):
        raise ValueError(f'--size 须为 {2 * patch} 的倍数')
    items = json.loads(DATA.read_text())['items']
    if a.ids_file:
        keep = set(json.loads(a.ids_file.read_text()))
        items = [i for i in items if i['id'] in keep]
    items = items[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.model)
    llm = LLM(a.model, max_model_len=a.max_model_len, limit_mm_per_prompt={'video': 6, 'image': 0},
              gpu_memory_utilization=0.85, seed=0)
    sampling = SamplingParams(temperature=0., max_tokens=16)

    def prepare(item):
        order = None
        if a.shuffle_seed is not None:
            item, order = shuffle_options(item, a.shuffle_seed)
        provider = next((u for u in item['evidence_users'] if u != item['speaker']), None)
        videos = {'all6': item['videos'], 'evidence2': item['evidence_videos'], 'speaker1': item['videos'][:1],
                  'speaker_plus1': [v for v in item['videos'] if v['agent_name'] in (item['speaker'], provider)]}[a.setting]
        messages = build_messages(item, videos, a.frames, a.size)
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True,
                                                          image_patch_size=patch)
        return {**item, 'option_order': order}, videos, {'prompt': prompt, 'multi_modal_data': {'video': video_inputs}, 'mm_processor_kwargs': video_kwargs}

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(4) as pool, (a.out / 'predictions.jsonl').open('w') as f:
        batches = [items[i:i + a.batch] for i in range(0, len(items), a.batch)]
        pending = pool.map(prepare, batches[0]) if batches else []
        for b in range(len(batches)):
            ready = list(pending)
            if b + 1 < len(batches):
                pending = pool.map(prepare, batches[b + 1])  # 下一批解码与本批推理重叠
            outputs = llm.generate([r[2] for r in ready], sampling, use_tqdm=False)
            for (item, videos, inp), out in zip(ready, outputs):
                text = out.outputs[0].text
                pred = parse_letter(text)
                row = {'id': item['id'], 'source': item['source'], 'setting': a.setting,
                       'evidence_users_selection': item['evidence_users_selection'],
                       'videos': [v['agent_name'] for v in videos], 'speaker': item['speaker'],
                       'prompt_tokens': len(out.prompt_token_ids), 'raw': text, 'pred': pred,
                       'correct': item['correct'], 'option_order': item['option_order'], 'is_correct': pred == item['correct']}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            print(f'{len(rows)}/{len(items)} acc={sum(r["is_correct"] for r in rows) / len(rows):.3f} '
                  f'{time.time() - t0:.0f}s', flush=True)
    summary = {'setting': a.setting, 'model': a.model, 'data': str(DATA), 'frames_per_video': a.frames, 'ids_file': a.ids_file and a.ids_file.name,
               'frame_size': a.size, 'shuffle_seed': a.shuffle_seed,
               'gold_letter_counts': dict(sorted(Counter(r['correct'] for r in rows).items())), 'decoding': 'greedy, max_tokens 16, first A-E letter', **summarize(rows),
               'mean_prompt_tokens': round(sum(r['prompt_tokens'] for r in rows) / len(rows))}
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
