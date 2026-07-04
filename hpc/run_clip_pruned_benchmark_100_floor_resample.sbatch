#!/usr/bin/env bash
#SBATCH --job-name=egolife_bench100_floor
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"

export EGOLIFE2U_OUTPUT_ROOT="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample}"
export EGOLIFE2U_TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-100}"
export EGOLIFE2U_EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${EGOLIFE2U_TARGET_COUNT}}"
export MIN_PRUNED_VIDEO_SECONDS="${MIN_PRUNED_VIDEO_SECONDS:-8}"
export PRUNING_PROTECTION_MODE="${PRUNING_PROTECTION_MODE:-min_seconds}"
export MIN_PRUNED_VIDEO_PERCENT="${MIN_PRUNED_VIDEO_PERCENT:-}"
export RESUME_GENERATION="${RESUME_GENERATION:-0}"

echo "stage=floor_resample_dispatch"
echo "project_root=${PROJECT_ROOT}"
echo "outdir=${EGOLIFE2U_OUTPUT_ROOT}"
echo "target_count=${EGOLIFE2U_TARGET_COUNT}"
echo "eval_packet_count=${EGOLIFE2U_EVAL_PACKET_COUNT}"
echo "min_pruned_video_seconds=${MIN_PRUNED_VIDEO_SECONDS}"
echo "pruning_protection_mode=${PRUNING_PROTECTION_MODE}"
echo "min_pruned_video_percent=${MIN_PRUNED_VIDEO_PERCENT}"
echo "benchmark_contract=resample_100_synchronized_video_pairs_then_floor_prune_and_generate"

SCRIPT_PATH="${PROJECT_ROOT}/egolife_two_user_qa/hpc/run_clip_pruned_benchmark_100.sbatch"
if [[ ! -s "${SCRIPT_PATH}" ]]; then
  SCRIPT_PATH="${PROJECT_ROOT}/hpc/run_clip_pruned_benchmark_100.sbatch"
fi
if [[ ! -s "${SCRIPT_PATH}" ]]; then
  echo "error=benchmark_script_missing path=${SCRIPT_PATH}" >&2
  exit 1
fi

exec bash "${SCRIPT_PATH}"
