#!/bin/bash
#SBATCH --job-name=xth_lxr_fbloop
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=1200G
#SBATCH --time=47:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/lxr_run/logs/%x_%j.out
# 三轮Judge反馈环（lxr F/E/C Judge prompt，C=both requirements）。需要 RUN_CONFIG、ADAPTER；COUNT/SAMPLES/ATTEMPTS 可调，RESUME_DIR 续跑。
set -euo pipefail
: "${RUN_CONFIG:?RUN_CONFIG is required}" "${ADAPTER:?ADAPTER is required}" "${COUNT:=100}" "${SAMPLES:=3}" "${ATTEMPTS:=3}"
ROOT=/scratch/tx856/multi_user_qa/lxr_run
export PYTHONPATH=$ROOT/runtime_deps_train:$ROOT/multi-user:$ROOT
export PYTHONDONTWRITEBYTECODE=1 VLLM_NO_USAGE_STATS=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PATH=/scratch/hm2991/conda/envs/egoqa-ms-swift-v4.2.2-vllm024/bin:$PATH
cd $ROOT/multi-user
echo "source=$(cat $ROOT/SOURCE_COMMIT.json) config=$RUN_CONFIG adapter=$ADAPTER count=$COUNT samples=$SAMPLES attempts=$ATTEMPTS resume=${RESUME_DIR:-none}"
ARGS=(--config "$RUN_CONFIG" --adapter "$ADAPTER" --count "$COUNT" --samples "$SAMPLES" --max-attempts "$ATTEMPTS" --output-root $ROOT/outputs/feedback_loop)
[ -n "${RESUME_DIR:-}" ] && ARGS+=(--resume-dir "$RESUME_DIR")
exec /scratch/tx856/.conda/envs/grpo-swift/bin/python -u $ROOT/eval_feedback_loop.py "${ARGS[@]}"
