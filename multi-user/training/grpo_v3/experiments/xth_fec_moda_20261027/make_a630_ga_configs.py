"""写 checkpoint-630 与 generalanswer 相关配置，基于 make_xth_configs.py 已写出的配置，只改下面列出的键。
  python make_a630_ga_configs.py <ROOT> <GA_ROOT>
"""
import json
import sys
from pathlib import Path

A630 = '/scratch/tx856/multi_user_qa/dual_full120/adapters/answerability_checkpoint-630'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    root, ga = Path(sys.argv[1]), Path(sys.argv[2])
    load = lambda name: json.loads((root / 'configs' / name).read_text())

    # 判别 judge：只换 answerability adapter
    judge = load('judge_fec.json')
    judge['adapters'] = {**judge['adapters'], 'answerability': A630}
    write(root / 'configs/judge_fec_a630.json', judge)

    # 旧 prompt、24576 像素、C=checkpoint-630（任务 21619020）
    px = load('train_continuous_gated_px24576.json')
    a630 = {**px, 'judge_config': str(root / 'configs/judge_fec_a630.json'),
            'output_root': str(root / 'outputs/continuous_gated_px24576_a630'),
            'run_attempt': 'xth_continuous_gated_px24576_a630_stage180'}
    write(root / 'configs/train_continuous_gated_px24576_a630.json', a630)

    # 反馈环：max_pixels 与 24576 训练一致（任务 21676581、21676583）
    write(root / 'configs/fbloop_px24576.json', {**load('fbloop.json'), 'max_pixels': 24576, 'run_attempt': 'xth_fbloop_lxr_px24576'})

    # generalanswer prompt、50176 像素（224×224）、C=checkpoint-630（任务 21676092）
    write(ga / 'configs/judge_fec_a630.json', {**judge, 'prompt_directory': str(ga / 'prompts')})
    provenance = {**a630['source_provenance'],
                  'data': 'lxr_run/data/train_length_paired.jsonl rebuilt with user prompts_generalanswer.py (sha256 6cefe391...), no brevity profile'}
    write(ga / 'configs/train_continuous_gated_px50176_a630.json', {
        **a630, 'project_root': str(ga), 'judge_config': str(ga / 'configs/judge_fec_a630.json'),
        'train_dataset': str(ga / 'data/train_length_paired_ga.jsonl'),
        'output_root': str(ga / 'outputs/continuous_gated_px50176_a630'), 'history_pool_directory': str(ga / 'history_pool'),
        'max_pixels': 50176, 'run_attempt': 'xth_continuous_gated_px50176_a630_generalanswer_stage180',
        'source_provenance': provenance})


if __name__ == '__main__':
    main()
