"""六用户 MCQ 的完整视频 VQA：每个用户的 10 分钟视频按 1fps 均匀取 600 帧、320×320，一个任务里依次跑多个设置（模型只加载一次）。

与 vqa_qwen25vl.py 相同的 prompt（每段视频前标注是谁的第一视角、谁是提问者）、选项重排、贪心解码、只要字母；区别：
  --data 可换题目 JSON；--settings 一次跑多个；新增 evidence_providers（evidence_videos 里 speaker 以外的人，
  与 grpo_frames/vqa_grpo_frames.py 同名设置一致；evidence 只有 speaker 的题跳过，记进 skipped.json）。
  python vqa_video_1fps.py --data <vqa.json> --ids-file <ids.json> --settings speaker1,evidence_providers,all6 --out <dir>
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from vqa_qwen25vl import VIDEO_ROOT, build_messages, parse_letter, shuffle_options, summarize

SETTINGS = ('speaker1', 'evidence_providers', 'evidence2', 'all6')
BATCH = {'speaker1': 8, 'evidence_providers': 4, 'evidence2': 4, 'all6': 2}


def pick_videos(item, setting):
    if setting == 'all6':
        return item['videos']
    if setting == 'speaker1':
        return item['videos'][:1]
    if setting == 'evidence2':
        return item['evidence_videos']
    return [v for v in item['evidence_videos'] if v['agent_name'] != item['speaker']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--ids-file', type=Path)
    ap.add_argument('--settings', required=True, help='逗号分隔：' + ','.join(SETTINGS))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--fps', type=float, default=1.0)
    ap.add_argument('--size', type=int, default=320)
    ap.add_argument('--max-model-len', type=int, default=262144)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--shuffle-seed', type=int)
    a = ap.parse_args()
    settings = a.settings.split(',')
    assert all(s in SETTINGS for s in settings), settings
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    patch = 16 if 'Qwen3-VL' in a.model else 14
    if a.size % (2 * patch):
        raise ValueError(f'--size 须为 {2 * patch} 的倍数')
    items = json.loads(a.data.read_text())['items']
    if a.ids_file:
        keep = set(json.loads(a.ids_file.read_text()))
        items = [i for i in items if i['id'] in keep]
        assert len(items) == len(keep), '题目文件里缺 ids'
    items = items[:a.limit]
    a.out.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(a.model)
    llm = LLM(a.model, max_model_len=a.max_model_len, limit_mm_per_prompt={'video': 6, 'image': 0},
              gpu_memory_utilization=0.85, seed=0,
              max_num_batched_tokens=65536)  # 编码器缓存 >= 单段视频 30000 token（默认 16384 不够）
    sampling = SamplingParams(temperature=0., max_tokens=16)

    def prepare(job):
        item, setting = job
        order = None
        if a.shuffle_seed is not None:
            item, order = shuffle_options(item, a.shuffle_seed)
        videos = pick_videos(item, setting)
        frames = []
        for v in videos:  # 1fps：帧数 = 视频时长（秒）×fps，取偶数（Qwen 时间 patch=2）
            n = int(v['duration_seconds'] * a.fps)
            frames.append(n - n % 2)
        messages = build_messages(item, videos, frames[0], a.size)
        vids = [c for c in messages[0]['content'] if c['type'] == 'video']
        for c, n in zip(vids, frames):
            c['nframes'] = n
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        _, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True, return_video_metadata=True,
                                                          image_patch_size=patch)
        # qwen_vl_utils 已把每帧缩到 size×size（2*patch 的倍数）；HF 视频 processor 默认按整段总像素上限再缩一次
        # （600 帧×320² 会被压到约 200²），所以关掉二次缩放
        return {**item, 'option_order': order}, videos, frames, {
            'prompt': prompt, 'multi_modal_data': {'video': video_inputs},
            'mm_processor_kwargs': {**video_kwargs, 'do_resize': False}}

    overview = {}
    for setting in settings:
        out = a.out / setting
        out.mkdir(exist_ok=True)
        todo = [i for i in items if pick_videos(i, setting)]
        skipped = [i['id'] for i in items if not pick_videos(i, setting)]
        (out / 'skipped.json').write_text(json.dumps(skipped, indent=2))
        rows, t0, bs = [], time.time(), BATCH[setting]
        batches = [[(i, setting) for i in todo[k:k + bs]] for k in range(0, len(todo), bs)]
        with ThreadPoolExecutor(4) as pool, (out / 'predictions.jsonl').open('w') as f:
            pending = pool.map(prepare, batches[0]) if batches else []
            for b in range(len(batches)):
                ready = list(pending)
                if b + 1 < len(batches):
                    pending = pool.map(prepare, batches[b + 1])
                outputs = llm.generate([r[3] for r in ready], sampling, use_tqdm=False)
                for (item, videos, frames, _), o in zip(ready, outputs):
                    need = sum(n // 2 * (a.size // (2 * patch)) ** 2 for n in frames)  # 视觉 token 下限
                    if len(o.prompt_token_ids) < need:
                        raise RuntimeError(f"{item['id']} prompt {len(o.prompt_token_ids)} tokens < 视觉 token 下限 {need}：帧被缩小或丢帧")
                    text = o.outputs[0].text
                    pred = parse_letter(text)
                    row = {'id': item['id'], 'source': item['source'], 'setting': setting,
                           'evidence_users_selection': item['evidence_users_selection'],
                           'videos': [v['agent_name'] for v in videos], 'frames_per_video': frames, 'speaker': item['speaker'],
                           'prompt_tokens': len(o.prompt_token_ids), 'raw': text, 'pred': pred,
                           'correct': item['correct'], 'option_order': item['option_order'], 'is_correct': pred == item['correct']}
                    rows.append(row)
                    f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                print(f'[{setting}] {len(rows)}/{len(todo)} acc={sum(r["is_correct"] for r in rows) / len(rows):.3f} '
                      f'{time.time() - t0:.0f}s', flush=True)
        summary = {'setting': setting, 'model': a.model, 'data': a.data.name, 'ids_file': a.ids_file and a.ids_file.name,
                   'sampling': f'full review video per user, {a.fps:g} fps uniform (frames = duration*fps, even), '
                               f'{a.size}x{a.size}', 'video_root': str(VIDEO_ROOT.name), 'shuffle_seed': a.shuffle_seed,
                   'n_skipped': len(skipped),
                   'gold_letter_counts': dict(sorted(Counter(r['correct'] for r in rows).items())),
                   'decoding': 'greedy, max_tokens 16, first A-E letter', **summarize(rows),
                   'mean_prompt_tokens': round(sum(r['prompt_tokens'] for r in rows) / len(rows)),
                   'max_prompt_tokens': max(r['prompt_tokens'] for r in rows),
                   'elapsed_seconds': round(time.time() - t0)}
        (out / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        overview[setting] = summary['overall']
        print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    (a.out / 'overview.json').write_text(json.dumps(overview, indent=2))


if __name__ == '__main__':
    main()
