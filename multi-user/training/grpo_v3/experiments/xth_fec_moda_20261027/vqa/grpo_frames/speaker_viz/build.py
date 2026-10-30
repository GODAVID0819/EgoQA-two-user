"""speaker-only VQA 答对题的可视化数据：题目、选项（模型所见的重排顺序）、两个模型的答案与理由、
生成器 evidence，以及与模型输入相同帧、相同分辨率的 speaker 视频。

题目：Qwen3-VL-8B 或 Qwen2.5-VL-7B 在 speaker1（GRPO 帧）设置下答对的题（并集）。
视频：六个用户各一条，帧即 GRPO 生成器的输入（speaker 300 帧每 2 s；provider 只有 keep_mask 存活帧，
  其余时间格画深灰），按 GRPO 的处理缩放：swift rescale_image 到面积
  <= 24576（1408² -> 156²，torchvision 双线性），再双三次放大到 Qwen3 processor 实际输入的 256²；
  按真实采样率 0.5 fps 编码（视频时间 = 窗口内秒数），每帧都是关键帧便于逐帧跳转。
输出 data.json 与 videos6/<evidence_id>/<user>.mp4。
  python build.py
"""
import json
import random
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
GF = HERE.parent
sys.path.insert(0, str(GF))
from vqa_grpo_frames import rescale_image  # noqa: E402  与 VQA 输入同一缩放函数

FFMPEG = '/scratch/tx856/.conda/envs/utopi/bin/ffmpeg'
CURATED = Path('/scratch/tx856/multi_user_qa/dual_full120/human_review/curated_pass')
RUNS = {'qwen3': GF / 'outputs/qwen3vl8b_speaker1_shuf0_21321632',
        'qwen25': GF / 'outputs/qwen25vl7b_speaker1_shuf0_21321633'}
# 同模型、同 prompt 但不给任何帧（text_only）：标注每题是否只凭题面就能答对
TEXT_RUNS = {'qwen3': GF / 'outputs/qwen3vl8b_text_only_shuf0_21326471',
             'qwen25': GF / 'outputs/qwen25vl7b_text_only_shuf0_21326472'}
MODEL_SIDE = 256  # Qwen3-VL / Qwen3.8 processor 对 156² 输入的实际网格 16x16 patch x16 px


def encode(job):
    """slots：300 个时间格（每 2 s），值为该格输入给模型的帧路径或 None（provider 被 keep_mask 丢掉的格）。
    丢掉的格画成深灰并写明未输入，视频时间仍与窗口秒数一一对应，六人可按同一时间同步。"""
    key, slots, out = job
    if out.is_file():
        return key
    from PIL import Image, ImageDraw
    gap = Image.new('RGB', (MODEL_SIDE, MODEL_SIDE), (40, 40, 40))
    ImageDraw.Draw(gap).text((52, 120), 'not in model input', fill=(170, 170, 170))
    gap = gap.tobytes()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp.mp4')
    proc = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                             '-s', f'{MODEL_SIDE}x{MODEL_SIDE}', '-r', '1/2', '-i', '-', '-c:v', 'libx264',
                             '-crf', '12', '-g', '1', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)],
                            stdin=subprocess.PIPE)
    for p in slots:
        if p is None:
            proc.stdin.write(gap)
            continue
        small = rescale_image(Image.open(p).convert('RGB'), 24576)
        proc.stdin.write(small.resize((MODEL_SIDE, MODEL_SIDE), Image.BICUBIC).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f'ffmpeg failed {key}')
    tmp.rename(out)
    return key


def timeline(paths, n_slots=300):
    """帧文件名 frame_<idx>_<sec>s.jpg -> 300 格时间线；speaker 应全满，provider 只有存活帧。"""
    slots = [None] * n_slots
    for p in paths:
        m = re.search(r'frame_(\d+)_(\d+(?:\.\d+)?)s\.jpg$', p)
        k = int(m.group(1))
        assert float(m.group(2)) == 2. * k and slots[k] is None, p
        slots[k] = p
    return slots


def claims_of(u):
    """dataset.json 的 per_user 两种格式：842 题为 claims 列表；fbloop 题为 evidence[{needed_fact, timeframe}]。"""
    if 'claims' in u:
        return u['claims']
    return [f"{e.get('needed_fact', '')}（生成器自报时间：{e.get('timeframe', '?')}）" for e in u.get('evidence', [])]


def shuffled(item, seed=0):
    letters = list(item['options'])
    order = letters[:]
    random.Random(f'{seed}:{item["id"]}').shuffle(order)
    return {n: item['options'][o] for n, o in zip(letters, order)}


def main():
    frames = json.loads((GF / 'frames.json').read_text())['items']
    vqa = {i['id']: i for i in json.loads((CURATED / 'vqa.json').read_text())['items']}
    dataset = {i['id']: i for i in json.loads((CURATED / 'dataset.json').read_text())['items']}
    preds = {k: {r['id']: r for r in map(json.loads, (d / 'predictions.jsonl').open())} for k, d in RUNS.items()}
    text = {k: {r['id']: r for r in map(json.loads, (d / 'predictions.jsonl').open())} for k, d in TEXT_RUNS.items()}
    ids = [i for i in vqa if any(preds[k].get(i, {}).get('is_correct') for k in preds)]
    items, jobs = [], {}
    for i in ids:
        it, fr = vqa[i], frames[i]
        opts = shuffled(it)
        gold = next(k for k, t in opts.items() if t == it['answer'])
        longest = max(opts, key=lambda k: len(opts[k]))
        users = []
        for u in fr['required_users']:
            slots = timeline(fr['frames'][u])
            assert u != it['speaker'] or None not in slots, i  # speaker 300 帧全输入
            rel = f"videos6/{fr['evidence_id']}/{u}.mp4"
            jobs[(fr['evidence_id'], u)] = ((fr['evidence_id'], u), slots, HERE / rel)
            users.append({'user': u, 'role': 'speaker' if u == it['speaker'] else 'provider', 'video': rel,
                          'n_input_frames': sum(p is not None for p in slots),
                          'kept': ''.join('0' if p is None else '1' for p in slots),
                          'in_generator_evidence': u in it['evidence_users']})
        ge = dataset[i].get('generator_evidence') or {}
        items.append({
            'id': i, 'source': it['source'], 'speaker': it['speaker'], 'packet_id': it['packet_id'],
            'day': it['day'], 'start_clock': it['start_clock'], 'question': it['question'], 'options': opts,
            'gold': gold, 'gold_is_longest': gold == longest, 'users': users,
            'n_frames': 300, 'frame_seconds': 2.0, 'total_input_frames': sum(fr['frame_counts']),
            'speaker_in_generator_evidence': it['speaker'] in it['evidence_users'],
            'generator_evidence_users': it['evidence_users'],
            'generator_claims': [{'user': u['user'], 'role': u['role'], 'claims': claims_of(u)}
                                 for u in ge.get('per_user', [])],
            'models': {k: {'pred': preds[k][i]['pred'], 'correct': preds[k][i]['is_correct'],
                           'reason': preds[k][i]['reason']} for k in preds},
            'text_only': {k: {'pred': text[k][i]['pred'], 'correct': text[k][i]['is_correct'],
                              'reason': text[k][i]['reason']} for k in text}})
    with ProcessPoolExecutor(8) as pool:
        for n, eid in enumerate(pool.map(encode, jobs.values(), chunksize=4), 1):
            if n % 50 == 0 or n == len(jobs):
                print('VIDEO', n, len(jobs), eid, flush=True)
    meta = {'resolution': f'1408² -> swift rescale 156² -> bicubic {MODEL_SIDE}² (Qwen3-VL processor input)',
            'fps': 0.5, 'gap': 'provider 被 keep_mask 去掉、未输入给生成器的时间格显示为深灰', 'runs': {k: str(v.name) for k, v in RUNS.items()},
            'text_only_runs': {k: str(v.name) for k, v in TEXT_RUNS.items()}, 'n_items': len(items)}
    (HERE / 'data.json').write_text(json.dumps({'meta': meta, 'items': items}, ensure_ascii=False))
    print('WROTE', len(items), 'items', len(jobs), 'videos')


if __name__ == '__main__':
    main()
