"""CPU构建固定历史池：来源换成我们 step204 生成器三轮反馈环中被接受的题（替代 lxr 无权限读取的旧300步池）。

筛选、每窗上限8、seed42、MiniLM编码与落盘格式均沿用 history_pool.save_history；正式GPU作业只读取该固定池。
  python build_xth_history.py --config configs/train_<mode>.json --runs <fbloop输出目录> ...
"""
import argparse
import json
from pathlib import Path

from training.grpo_v3.six_user_binary.diversity import window_id
from training.grpo_v3.six_user_binary.judge import parse_completion
from training.grpo_v3.six_user_binary.history_pool import MiniLMEncoder, load_history, save_xth_history


def accepted_records(run_dir):
    run_dir = Path(run_dir)
    job = run_dir.name.rsplit('_', 1)[-1]
    records = []
    for path in sorted(run_dir.glob('rank*/feedback_loop.json')):
        for item in json.loads(path.read_text())['items']:
            if not item['accepted_attempt']:
                continue
            final = item['attempts'][item['accepted_attempt'] - 1]
            if not final['all_pass']:
                raise ValueError(f"{item['evidence_id']} 接受轮并非三项通过")
            packet, asker = item['evidence_id'].rsplit('__ASKER_', 1)
            records.append({'history_qa_id': f"xth:{job}:{item['evidence_id']}:s{item['sample']}",
                            'source_packet_id': packet, 'evidence_id': item['evidence_id'],
                            'source_job': job, 'accepted_attempt': item['accepted_attempt'],
                            'question': parse_completion(final['completion'])['question'],
                            'probabilities_xth_F_E_A': final['probabilities']})
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    a = parser.parse_args()
    c = json.loads(a.config.read_text())
    directory = Path(c['history_pool_directory'])
    if directory.exists():
        raise FileExistsError(f'{directory} 已存在，固定池不覆盖')
    records = [r for run in a.runs for r in accepted_records(run)]
    windows = sorted({window_id(json.loads(line)['source_packet_id'])
                      for line in Path(c['train_dataset']).read_text().splitlines() if line.strip()})
    if len(windows) != 120:
        raise ValueError('训练窗口应为120')
    source = {'runs': [str(r) for r in a.runs], 'accepted_QA': len(records)}
    report = save_xth_history(records, windows, MiniLMEncoder(Path(c['project_root']) / 'encoder'), directory,
                              source_metadata=json.dumps(source, ensure_ascii=False))
    load_history(directory)
    print('XTH_HISTORY_POOL_FIXED', json.dumps({k: report[k] for k in ('selected_QA', 'empty_windows', 'selected_count_distribution')}), flush=True)


if __name__ == '__main__':
    main()
