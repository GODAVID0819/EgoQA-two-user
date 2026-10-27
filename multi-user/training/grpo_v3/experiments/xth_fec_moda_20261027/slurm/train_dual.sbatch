#!/bin/bash
#SBATCH --job-name=xth_lxr_grpo
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=750G
#SBATCH --time=47:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/lxr_run/logs/%x_%j.out
# lxr F/E/C+固定历史池diversity 双卡GRPO（dual_launch.py，与 lxr 正式运行同入口）。需要 RUN_CONFIG（configs/train_<mode>.json）。
# 墙钟：用户4卡>48h限制之外，47h 一段；按 stop_after_steps 分阶段续跑。
set -euo pipefail
: "${RUN_CONFIG:?RUN_CONFIG is required}"
ROOT=/scratch/tx856/multi_user_qa/lxr_run
export PYTHONPATH=$ROOT/runtime_deps_train:$ROOT/multi-user:$ROOT
export PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH=/scratch/hm2991/conda/envs/egoqa-ms-swift-v4.2.2-vllm024/bin:$PATH
cd $ROOT/multi-user
MEMLOG=$ROOT/logs/hostmem_${SLURM_JOB_ID}.log
( while true; do echo "=== $(date -Is) free_G=$(free -g | awk '/Mem:/{print $2" used="$3" avail="$7}')"
    cat /sys/fs/cgroup/$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)/memory.current 2>/dev/null | awk '{printf "cgroup_memory_current_G=%.1f\n",$1/2^30}'
    sleep 300; done ) >> "$MEMLOG" 2>&1 &
echo "source=$(cat $ROOT/SOURCE_COMMIT.json) config=$RUN_CONFIG"
exec /scratch/tx856/.conda/envs/grpo-swift/bin/python -u $ROOT/dual_launch.py --config "$RUN_CONFIG"
