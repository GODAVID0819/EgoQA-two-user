"""为 169 道 VQA 题重建 GRPO 生成器看到的同一套帧，并检查各模型对这些帧的实际分辨率。

帧：lxr_run 的 make_row(dataset_root, packet_id, asker_index)，即 GRPO 训练行的同一构造函数
  （speaker 300 帧全保留，provider 只留 keep_masks 里相对该 asker 存活的帧）。
  与 lxr_run/data/train_length_paired_v2.jsonl 已有的同 evidence_id 行逐项比对帧路径与帧数。
prompt：从 make_row 生成的 GRPO prompt 原文里截取对输入的描述段落和 Evidence packet metadata，供 VQA 复用。
分辨率：GRPO 先用 swift rescale_image 缩到面积 <= max_pixels(24576)，再交给模型 processor；这里对一帧
  用三个 processor（Qwen3.8 策略、Qwen3-VL-8B、Qwen2.5-VL-7B）跑出 image_grid_thw，记录实际像素与 token 数。
  PYTHONPATH=<lxr_run>/multi-user python prep_frames.py --out frames.json
"""
import argparse
import json
import re
from pathlib import Path

from training.grpo_v3.six_user_binary.data import make_row

DATA = Path('/scratch/tx856/multi_user_qa/dual_full120/human_review/curated_pass/vqa.json')
DATASET_ROOT = '/scratch/hm2991/egolife_rlhf_evidence_v1'
TRAIN = Path('/scratch/tx856/multi_user_qa/lxr_run/data/train_length_paired_v2.jsonl')
MAX_PIXELS = 24576  # lxr_run/configs/train_*_px24576.json 的 max_pixels
PROCESSORS = {'policy_qwen3.8': '/scratch/tx856/models/Qwen3.8-27B',
              'Qwen/Qwen3-VL-8B-Instruct': 'Qwen/Qwen3-VL-8B-Instruct',
              'Qwen/Qwen2.5-VL-7B-Instruct': 'Qwen/Qwen2.5-VL-7B-Instruct'}


def prompt_parts(prompt):
    """GRPO prompt 里描述输入的段落（Input 起到 Output contract 前）与 metadata JSON。"""
    head = prompt.split('\n\nOutput contract:')[0]
    paragraphs = head.split('\n\n')[1:]  # 去掉 "You are generating ..." 任务句
    six_user = re.search(r'The generator receives frames from six .*?within each user\'s group\.', prompt).group(0)
    meta = prompt.split('Evidence packet metadata:\n', 1)[1].split('\n\nReturn exactly one valid JSON object')[0]
    return paragraphs, six_user, json.loads(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    items = json.loads(DATA.read_text())['items']
    train = {}
    for line in TRAIN.open():
        r = json.loads(line)
        train[r['evidence_id']] = (r['generator_image_paths'], r['generator_frame_counts'], r['required_users'])
    out, checked, opening, skipped = {}, 0, None, []
    for item in items:
        if not (Path(DATASET_ROOT) / 'packets' / item['packet_id'] / 'packet.json').is_file():
            skipped.append({'id': item['id'], 'packet_id': item['packet_id']})  # 旧代码窗口，GRPO 数据集里无此 packet
            continue
        speaker_dir = item['videos'][0]['agent_dir']  # 'A4_LUCIA' -> asker_index 3（packet users 按 A1..A6）
        asker_index = int(speaker_dir.split('_')[0][1:]) - 1
        row = make_row(DATASET_ROOT, item['packet_id'], asker_index)
        assert row['required_users'][0] == item['speaker'], (item['id'], row['required_users'], item['speaker'])
        assert row['required_users'] == [v['agent_name'] for v in item['videos']], item['id']
        if row['evidence_id'] in train:
            assert train[row['evidence_id']] == (row['generator_image_paths'], row['generator_frame_counts'],
                                                 row['required_users']), row['evidence_id']
            checked += 1
        paragraphs, six_user, meta = prompt_parts(row['messages'][0]['content'].replace('<image>\n', ''))
        opening = opening or paragraphs
        assert paragraphs == opening, '各题 GRPO 输入描述段落应相同'
        paths, groups, k = row['generator_image_paths'], {}, 0
        for user, n in zip(row['required_users'], row['generator_frame_counts']):
            groups[user] = paths[k:k + n]
            k += n
        out[item['id']] = {'evidence_id': row['evidence_id'], 'required_users': row['required_users'],
                           'frame_counts': row['generator_frame_counts'], 'frames': groups, 'metadata': meta}
    print('ITEMS', len(out), 'MATCHED_TRAIN_ROWS', checked, 'SKIPPED', skipped, flush=True)

    from PIL import Image
    from swift.template.vision_utils import rescale_image
    from transformers import AutoProcessor
    frame = next(iter(out.values()))['frames'][items[0]['speaker']][0]
    raw = Image.open(frame).convert('RGB')
    small = rescale_image(raw, MAX_PIXELS)
    resolution = {'raw': list(raw.size), 'after_swift_rescale': list(small.size)}
    for name, path in PROCESSORS.items():
        proc = AutoProcessor.from_pretrained(path)
        grid = proc.image_processor(images=[small], return_tensors='pt')['image_grid_thw'][0].tolist()
        patch = proc.image_processor.patch_size
        merge = proc.image_processor.merge_size
        resolution[name] = {'grid_thw': grid, 'pixels_hw': [grid[1] * patch, grid[2] * patch],
                            'tokens_per_image': grid[0] * grid[1] * grid[2] // merge ** 2}
    print('RESOLUTION', json.dumps(resolution), flush=True)
    a.out.write_text(json.dumps({'max_pixels': MAX_PIXELS, 'resolution': resolution, 'grpo_input_description': opening,
                                 'grpo_six_user_media_line': six_user, 'skipped': skipped, 'items': out}, ensure_ascii=False))
    print('WROTE', a.out)


if __name__ == '__main__':
    main()
