"""六用户 MCQ 的 VQA，输入与 GRPO 生成器完全同一套帧、同一分辨率、同一段输入描述 prompt。

帧：prep_frames.py 写的 frames.json（make_row 重建：speaker 300 帧，provider 只留 keep_masks 存活帧）。
分辨率：与 GRPO 相同，每帧先用 swift rescale_image 缩到面积 <= 24576（1408² -> 156²），再交给模型自己的 processor。
图片放在最前、文字在后（与 GRPO 训练行 "<image>\\n"*N + prompt 相同）；文字里没有逐组标签，分组只靠
metadata 里各用户的 frame_count，与生成器所见一致。
prompt：GRPO prompt 里描述输入的原文段落 + 六用户分组说明 + Evidence packet metadata（只保留本设置给出的用户），
  然后是问题、选项，要求先给字母再给理由与依据（Answer: X / Reason: ...）。
设置：all6 / evidence2（vqa.json 的 evidence_videos 两人）/ evidence_providers（只给 evidence_videos 里 speaker 以外的人：
  两个就两个、一个就一个；evidence 只有 speaker 的题跳过并记进 skipped.json）/ speaker1 / text_only（不给任何帧）。
--shuffle-seed 与 vqa_qwen25vl.py 相同的选项重排。贪心解码。
  python vqa_grpo_frames.py --setting speaker1 --model Qwen/Qwen3-VL-8B-Instruct --out <dir> [--limit 2]
"""
import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DATA = Path('/scratch/tx856/multi_user_qa/dual_full120/human_review/curated_pass/vqa.json')
FRAMES = Path(__file__).resolve().parent / 'frames.json'
# GRPO 生成指令段里与“回答”无关的两段（生成前自检、不要用字幕）不放进 VQA prompt
SKIP_PARAGRAPH_PREFIXES = ('Before returning the question',)
NUMBER_WORDS = {1: 'one', 2: 'two', 6: 'six'}
ANSWER_FORMAT = ('Answer in exactly this format and nothing else:\n'
                 'Answer: <one letter A-E>\n'
                 'Reason: <why you chose this option: which user\'s frames you relied on and what directly visible '
                 'evidence supports it; if the frames do not show the answer, say what you based the choice on instead>')


def rescale_image(img, max_pixels):
    """swift.template.vision_utils.rescale_image 的同一实现（GRPO policy_image_cache 用它缩图）。"""
    import torchvision.transforms as T
    width, height = img.width, img.height
    if max_pixels is None or max_pixels <= 0 or width * height <= max_pixels:
        return img
    ratio = width / height
    height_scaled = math.sqrt(max_pixels / ratio)
    width_scaled = height_scaled * ratio
    return T.Resize((int(height_scaled), int(width_scaled)))(img)


def shuffle_options(item, seed):
    letters = list(item['options'])
    order = letters[:]
    random.Random(f'{seed}:{item["id"]}').shuffle(order)
    options = {new: item['options'][old] for new, old in zip(letters, order)}
    return {**item, 'options': options, 'correct': letters[order.index(item['correct'])]}, order


def parse_letter(text):
    m = re.search(r'Answer\s*:\s*\(?([A-E])\b', text, re.I) or re.search(r'\b([A-E])\b', text.strip())
    return m.group(1).upper() if m else None


def parse_reason(text):
    m = re.search(r'Reason\s*:\s*(.*)', text, re.S | re.I)
    return m.group(1).strip() if m else None


def setting_users(item, fr, setting):
    if setting == 'text_only':
        return []
    if setting == 'all6':
        return list(fr['required_users'])
    if setting == 'speaker1':
        return [item['speaker']]
    if setting == 'evidence_providers':
        chosen = {v['agent_name'] for v in item['evidence_videos']} - {item['speaker']}
        return [u for u in fr['required_users'] if u in chosen]
    chosen = {v['agent_name'] for v in item['evidence_videos']}
    return [u for u in fr['required_users'] if u in chosen]


def build_text(item, fr, users, header):
    speaker = item['speaker']
    options = '\n'.join(f'{k}. {t}' for k, t in item['options'].items())
    question = (f'The question is asked by {speaker}; "I", "me" and "my" refer to {speaker}.\n'
                f'Question: {item["question"]}\nOptions:\n{options}\n\n{ANSWER_FORMAT}')
    if not users:
        return ('You are answering one multiple-choice question about egocentric recordings made by six people in '
                'the same household during the same 10-minute interval. No frames are provided; choose the most '
                'likely option from the question and options alone.\n\n' + question)
    meta = fr['metadata']
    clips = [c for c in meta['clips'] if c['user'] in users]
    shown = {'evidence_id': meta['evidence_id'], 'required_users': users,
             'role_contract': {'speaker_user': speaker,
                               'speaker_frames_supplied': speaker in users,
                               'provider_users': [u for u in users if u != speaker]},
             'clips': clips, 'generator_media_contract': meta['generator_media_contract']}
    n = len(users)
    group_line = header['six_user'].replace('The generator receives', 'You receive').replace(
        "from six people's", f"from {NUMBER_WORDS.get(n, str(n))} {'person' if n == 1 else 'people'}'s")
    if n == 1:
        group_line = group_line.replace('Frames are grouped by user and ordered chronologically within each '
                                        "user's group.", 'Frames are ordered chronologically.')
    paragraphs = [p for p in header['paragraphs'] if not p.startswith(SKIP_PARAGRAPH_PREFIXES)]
    return ('You are answering one multiple-choice question from egocentric visual evidence.\n\n'
            + '\n\n'.join(paragraphs) + '\n\n' + group_line + '\n\nEvidence packet metadata:\n'
            + json.dumps(shown, ensure_ascii=False, indent=2) + '\n\n' + question)


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
    ap.add_argument('--setting', choices=('all6', 'evidence2', 'evidence_providers', 'speaker1', 'text_only'), required=True)
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--max-model-len', type=int, default=131072)
    ap.add_argument('--max-tokens', type=int, default=512)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--shuffle-seed', type=int)
    ap.add_argument('--ids-file', type=Path, help='只跑这些题（JSON id 列表）')
    a = ap.parse_args()
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    store = json.loads(FRAMES.read_text())
    max_pixels = store['max_pixels']
    header = {'paragraphs': store['grpo_input_description'], 'six_user': store['grpo_six_user_media_line']}
    items = [i for i in json.loads(DATA.read_text())['items'] if i['id'] in store['items']]  # 无 GRPO packet 的题跳过
    if a.ids_file:
        keep = set(json.loads(a.ids_file.read_text()))
        items = [i for i in items if i['id'] in keep]
    items = items[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    skipped = []
    if a.setting == 'evidence_providers':  # evidence 只有 speaker：没有可给的 provider，记下不跑
        skipped = [{'id': i['id'], 'speaker': i['speaker'], 'evidence_users': i['evidence_users'],
                    'evidence_users_selection': i['evidence_users_selection']}
                   for i in items if not setting_users(i, store['items'][i['id']], a.setting)]
        items = [i for i in items if setting_users(i, store['items'][i['id']], a.setting)]
        (a.out / 'skipped.json').write_text(json.dumps(skipped, ensure_ascii=False, indent=1))
    processor = AutoProcessor.from_pretrained(a.model)
    side = math.isqrt(max_pixels)  # 1408² 方图缩后 156²；dummy profiling 用同尺寸
    max_images = max(sum(f['frame_counts']) for f in store['items'].values())
    llm = LLM(a.model, max_model_len=a.max_model_len, gpu_memory_utilization=0.85, seed=0,
              limit_mm_per_prompt={'image': {'count': max_images, 'width': side, 'height': side}, 'video': 0})
    sampling = SamplingParams(temperature=0., max_tokens=a.max_tokens)

    def prepare(item):
        order = None
        if a.shuffle_seed is not None:
            item, order = shuffle_options(item, a.shuffle_seed)
        fr = store['items'][item['id']]
        users = setting_users(item, fr, a.setting)
        paths = [p for u in users for p in fr['frames'][u]]
        images = [rescale_image(Image.open(p).convert('RGB'), max_pixels) for p in paths]
        content = [{'type': 'image'} for _ in images] + [{'type': 'text', 'text': build_text(item, fr, users, header)}]
        prompt = processor.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False,
                                               add_generation_prompt=True)
        inp = {'prompt': prompt, **({'multi_modal_data': {'image': images}} if images else {})}
        return {**item, 'option_order': order}, users, len(images), inp

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(4) as pool, (a.out / 'predictions.jsonl').open('w') as f:
        batches = [items[i:i + a.batch] for i in range(0, len(items), a.batch)]
        pending = pool.map(prepare, batches[0]) if batches else []
        for b in range(len(batches)):
            ready = list(pending)
            if b + 1 < len(batches):
                pending = pool.map(prepare, batches[b + 1])
            outputs = llm.generate([r[3] for r in ready], sampling, use_tqdm=False)
            for (item, users, n_images, inp), out in zip(ready, outputs):
                text = out.outputs[0].text
                pred = parse_letter(text)
                row = {'id': item['id'], 'source': item['source'], 'setting': a.setting,
                       'evidence_users_selection': item['evidence_users_selection'], 'users': users,
                       'n_images': n_images, 'speaker': item['speaker'], 'prompt_tokens': len(out.prompt_token_ids),
                       'raw': text, 'pred': pred, 'reason': parse_reason(text), 'correct': item['correct'],
                       'option_order': item['option_order'], 'is_correct': pred == item['correct']}
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            print(f'{len(rows)}/{len(items)} acc={sum(r["is_correct"] for r in rows) / len(rows):.3f} '
                  f'{time.time() - t0:.0f}s', flush=True)
    if rows:
        (a.out / 'prompt_example.txt').write_text(ready[-1][3]['prompt'])
    summary = {'setting': a.setting, 'model': a.model, 'frames': 'GRPO make_row frames (frames.json)',
               'max_pixels': max_pixels, 'resolution': store['resolution'], 'shuffle_seed': a.shuffle_seed,
               'ids_file': a.ids_file and a.ids_file.name,
               'decoding': f'greedy, max_tokens {a.max_tokens}, "Answer: X" then reason',
               'n_skipped_speaker_only_evidence': len(skipped),
               **summarize(rows),
               'mean_images': sum(r['n_images'] for r in rows) / max(len(rows), 1),
               'mean_prompt_tokens': sum(r['prompt_tokens'] for r in rows) / max(len(rows), 1)}
    (a.out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary['overall']), 'unparsed', summary['unparsed'], flush=True)


if __name__ == '__main__':
    main()
