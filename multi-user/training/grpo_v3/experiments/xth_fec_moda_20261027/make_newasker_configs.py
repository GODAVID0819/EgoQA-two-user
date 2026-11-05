"""新 asker 的 600 题反馈环配置（任务 21839296 C630、21874738 C340），基于 make_a630_ga_configs.py 写出的 GA 配置，只改下面列出的键。
输入文件由 build_newasker.py 生成（data/fbloop_newasker120_ga.jsonl）。
  python make_newasker_configs.py <GA_ROOT>
"""
import json
import sys
from pathlib import Path

A340 = '/scratch/tx856/multi_user_qa/dual_full120/adapters/answerability_checkpoint-340'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    ga = Path(sys.argv[1])
    load = lambda name: json.loads((ga / 'configs' / name).read_text())
    judge = load('judge_fec_a630.json')
    write(ga / 'configs/judge_fec_a340.json', {**judge, 'adapters': {**judge['adapters'], 'answerability': A340}})
    train = load('train_continuous_gated_px50176_a630.json')
    base = {**train, 'val_dataset': str(ga / 'data/fbloop_newasker120_ga.jsonl'),
            'output_root': str(ga / 'outputs/feedback_loop'), 'validation_seed': 20261104,
            'source_provenance': {**train['source_provenance'],
                                  'fbloop': '120 windows x 1 asker unused by completed feedback loops (logs/build_newasker.py), '
                                            'validation_seed 20261104'}}
    for c in ('a630', 'a340'):
        write(ga / f'configs/fbloop_ga_px50176_{c}_newasker.json',
              {**base, 'judge_config': str(ga / f'configs/judge_fec_{c}.json'),
               'run_attempt': f'xth_fbloop_ga_px50176_{c}_newasker120'})


if __name__ == '__main__':
    main()
