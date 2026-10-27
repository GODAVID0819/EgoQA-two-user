"""在 Torch 上为物化后的运行根写配置：训练两种质量奖励、单步容量smoke、Judge、反馈环。

训练配置从 lxr 正式运行 20573337 的 examples/train_config_20573337.json 出发，只改：本集群路径与环境、reward_mode、
历史池（我们自己的题）、output_root/run_attempt，以及去掉对 lxr 私有目录的来源引用。超参数逐项保持不变。
  python make_xth_configs.py /scratch/tx856/multi_user_qa/lxr_run
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / 'full120x6_fec_moda_stage180_20261021/examples/train_config_20573337.json'
DUAL = Path('/scratch/tx856/multi_user_qa/dual_full120')
PYTHON = '/scratch/tx856/.conda/envs/grpo-swift/bin/python'
MODEL = '/scratch/tx856/models/Qwen3.8-27B'  # 目录名须为 Qwen3.8-27B（predictor 用目录名核对 adapter 基座），文件链接到 HF 快照
DROP = ('acceleration_packages', 'compiler_environment', 'source_train_dataset', 'source_validation_dataset',
        'source_condition_definition', 'history_source', 'walltime_basis', 'new_GPU_smoke_tasks')


def main():
    root = Path(sys.argv[1])
    (root / 'configs').mkdir(exist_ok=True)
    judge = json.loads((DUAL / 'configs/judge_mixed_f504.json').read_text())
    judge.update(model_id=MODEL, prompt_directory=str(root / 'prompts'))
    (root / 'configs/judge_fec.json').write_text(json.dumps(judge, indent=2) + '\n')

    base = {k: v for k, v in json.loads(SOURCE.read_text()).items() if k not in DROP}
    base.update(project_root=str(root), train_python=PYTHON, judge_python=PYTHON, policy_model=MODEL,
                judge_config=str(root / 'configs/judge_fec.json'), train_dataset=str(root / 'data/train_length_paired.jsonl'),
                scratch_root='/scratch/tx856/js', policy_cuda_home='/scratch/tx856/.conda/cuda130_home',
                history_pool_directory=str(root / 'history_pool'), history_score_source='xth_fbloop_step204_F_E_A',
                slurm={'account': 'torch_pr_111_tandon_advanced', 'gres': 'gpu:h200:2', 'cpus': 32, 'mem': '750G'},
                source_provenance={'lxr_config': 'full120x6_fec_moda_stage180_20261021/examples/train_config_20573337.json',
                                   'data': 'dual_full120/data/trainval/{train,validation}.jsonl reordered by lxr condition_definition.json',
                                   'history_pool': 'step204 feedback-loop accepted QA (we cannot read the lxr pool)'})
    for mode in ('continuous', 'continuous_gated'):
        c = {**base, 'reward_mode': mode, 'output_root': str(root / 'outputs' / mode), 'run_attempt': f'xth_{mode}_stage180'}
        (root / f'configs/train_{mode}.json').write_text(json.dumps(c, indent=2, ensure_ascii=False) + '\n')
    # 单步容量smoke：只用最长两条件中的前两条（同 lxr capacity_measurement 语义），fresh，一次完整更新
    smoke = {**base, 'reward_mode': 'continuous_gated', 'capacity_measurement': True, 'stop_after_steps': 1,
             'train_conditions': 2, 'train_dataset': str(root / 'data/train_capacity2.jsonl'),
             'output_root': str(root / 'outputs/smoke'), 'run_attempt': 'xth_capacity_smoke'}
    (root / 'configs/smoke_capacity.json').write_text(json.dumps(smoke, indent=2, ensure_ascii=False) + '\n')
    # 反馈环：生成设置与训练一致（max_pixels 50176、max_length 131072、采样参数），输入为原验证集（每窗口原验证speaker）
    fb = {**base, 'reward_mode': 'continuous', 'val_dataset': str(DUAL / 'data/trainval/validation.jsonl'),
          'validation_seed': 42, 'output_root': str(root / 'outputs/feedback_loop'), 'run_attempt': 'xth_fbloop_lxr'}
    (root / 'configs/fbloop.json').write_text(json.dumps(fb, indent=2, ensure_ascii=False) + '\n')
    print(sorted(p.name for p in (root / 'configs').iterdir()))


if __name__ == '__main__':
    main()
