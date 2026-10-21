#!/bin/bash
#SBATCH --job-name=eval_fbloop
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1200G
#SBATCH --time=47:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/dual_full120/logs/%x_%j.out
# 三轮Judge反馈环：前COUNT条验证输入×SAMPLES题，每题最多ATTEMPTS轮（生成→三个已训练Judge P(pass)>0.5判定→同一Judge写反馈→重生成）。
# ADAPTER不设为未训练基座；RESUME_DIR设为已有输出目录时续跑。
set -euo pipefail
: "${RUN_CONFIG:?RUN_CONFIG is required}" "${COUNT:=100}" "${SAMPLES:=3}" "${ATTEMPTS:=3}"
export PROJECT_ROOT=/scratch/tx856/multi_user_qa/EgoQA-fbloop/dual_run
export PYTHONPATH=/scratch/tx856/multi_user_qa/EgoQA-fbloop/dual_run/multi-user:/scratch/tx856/multi_user_qa/EgoQA-fbloop/dual_run
export PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false
export PATH=/scratch/hm2991/conda/envs/egoqa-ms-swift-v4.2.2-vllm024/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
cd /scratch/tx856/multi_user_qa/EgoQA-fbloop/dual_run/multi-user
MEMLOG=/scratch/tx856/multi_user_qa/dual_full120/logs/hostmem_${SLURM_JOB_ID}.log
( while true; do echo "=== $(date -Is) free_G=$(free -g | awk '/Mem:/{print $2" used="$3" avail="$7}')"
    cat /sys/fs/cgroup/$(awk -F: '$1=="0"{print $3}' /proc/self/cgroup)/memory.current 2>/dev/null | awk '{printf "cgroup_memory_current_G=%.1f\n",$1/2^30}'
    ps -u "$USER" -o pid=,rss=,args= --sort=-rss | head -15 | awk '{printf "%8.1fG %s ",$2/2^20,$1; for(i=3;i<=NF&&i<8;i++)printf "%s ",$i; print ""}'
    sleep 300; done ) >> "$MEMLOG" 2>&1 &
echo "git_commit=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) config=$RUN_CONFIG adapter=${ADAPTER:-base} count=$COUNT samples=$SAMPLES attempts=$ATTEMPTS resume=${RESUME_DIR:-none}"
ARGS=(--config "$RUN_CONFIG" --count "$COUNT" --samples "$SAMPLES" --max-attempts "$ATTEMPTS" --output-root /scratch/tx856/multi_user_qa/dual_full120/outputs/feedback_loop)
[ -n "${ADAPTER:-}" ] && ARGS+=(--adapter "$ADAPTER")
[ -n "${RESUME_DIR:-}" ] && ARGS+=(--resume-dir "$RESUME_DIR")
exec /scratch/tx856/.conda/envs/grpo-swift/bin/python -u /scratch/tx856/multi_user_qa/EgoQA-fbloop/dual_run/eval_feedback_loop.py "${ARGS[@]}"
