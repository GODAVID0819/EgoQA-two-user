#!/bin/bash
#SBATCH --job-name=eval_passrate
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=600G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/dual_full120/logs/%x_%j.out
# 基座 vs LoRA 在验证输入上各生成1题，三个冻结Judge按P(pass)>0.5算通过率。需要 ADAPTER、COUNT。
set -euo pipefail
: "${RUN_CONFIG:?RUN_CONFIG is required}" "${ADAPTER:?ADAPTER is required}" "${COUNT:=100}"
export PROJECT_ROOT=/scratch/tx856/multi_user_qa/EgoQA-joint/dual_run
export PYTHONPATH=/scratch/tx856/multi_user_qa/EgoQA-joint/dual_run/multi-user:/scratch/tx856/multi_user_qa/EgoQA-joint/dual_run
export PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export PATH=/scratch/hm2991/conda/envs/egoqa-ms-swift-v4.2.2-vllm024/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /scratch/tx856/multi_user_qa/EgoQA-joint/dual_run/multi-user
# 每5分钟记录作业cgroup内存和本用户各进程RSS（4卡smoke峰值1059G、1100G下第12步OOM；集群上限300G/GPU）。
MEMLOG=/scratch/tx856/multi_user_qa/dual_full120/logs/hostmem_${SLURM_JOB_ID}.log
( while true; do echo "=== $(date -Is) free_G=$(free -g | awk '/Mem:/{print $2" used="$3" avail="$7}')"
    cat /sys/fs/cgroup/$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)/memory.current 2>/dev/null | awk '{printf "cgroup_memory_current_G=%.1f\n",$1/2^30}'
    ps -u "$USER" -o pid=,rss=,args= --sort=-rss | head -15 | awk '{printf "%8.1fG %s ",$2/2^20,$1; for(i=3;i<=NF&&i<8;i++)printf "%s ",$i; print ""}'
    sleep 300; done ) >> "$MEMLOG" 2>&1 &
echo "git_commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) config=$RUN_CONFIG"
exec /scratch/tx856/.conda/envs/grpo-swift/bin/python -u /scratch/tx856/multi_user_qa/EgoQA-joint/dual_run/eval_passrate.py --config "$RUN_CONFIG" --adapter "$ADAPTER" --count "$COUNT" --output-root /scratch/tx856/multi_user_qa/dual_full120/outputs/eval_passrate
