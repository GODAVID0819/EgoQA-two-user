#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_legacy
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
BASE_LAUNCHER="${PROJECT_ROOT}/hpc/qa/production/run_six_user_qa_10min_time_aware.sbatch"
test -s "${BASE_LAUNCHER}"

# Reuse only the expensive candidate/media preprocessing from an explicitly named
# prior run. The base launcher validates the setup summary, candidate JSONL, full
# videos, sampled frames, and source segments before generation begins.
export REUSE_PREPROCESSED_FROM="${REUSE_PREPROCESSED_FROM:-}"
if [[ -z "${REUSE_PREPROCESSED_FROM}" ]]; then
  echo "error=missing_reuse_preprocessed_from set REUSE_PREPROCESSED_FROM to the halted run output directory" >&2
  exit 2
fi

# Do not seed accepted/rejected/intermediate decisions from the time-aware run.
# Every reused evidence packet must receive a fresh legacy-judge evaluation.
export RESUME_QA_FROM=""
export REQUIRE_FRESH_EVIDENCE=0

# These remain optional controls for deliberately excluding known packet IDs.
export COHORT_SKIP_SOURCE="${COHORT_SKIP_SOURCE:-}"
export SKIP_EVIDENCE_IDS="${SKIP_EVIDENCE_IDS:-}"

export RUN_MODE="${RUN_MODE:-six_user_qa_10min_legacy_zero_shot}"
export SIX_USER_JUDGE_MODE="legacy-zero-shot"
export JUDGE_VIDEO_FPS="${JUDGE_VIDEO_FPS:-0.75}"
export REQUIRED_GPU_COUNT=2
export QWEN_MEMORY_SAFE_DEVICE_MAP=balanced
export QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT=2
export CUDA_KEEPER_GPUS=0,1
export CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-12.0}"
# The one-packet hardware smoke already passed. This production launcher always
# evaluates the complete frozen cohort, even if the submitting shell retained a
# QA_PACKET_LIMIT value from an earlier smoke command.
export QA_PACKET_LIMIT=""

# The production launcher owns environment activation, storage/runtime preflights,
# CUDA keeper lifecycle, resume handling, and output-contract validation.
source "${BASE_LAUNCHER}"
