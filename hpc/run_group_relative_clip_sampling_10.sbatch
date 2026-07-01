#!/usr/bin/env bash
#SBATCH --job-name=egolife_grpclip10
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/group_relative_clip_10}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-10}"
MAX_GROUPS="${EGOLIFE2U_MAX_GROUPS:-40}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
PAIRS_PER_GROUP="${GRP_PAIRS_PER_GROUP:-1}"
TOPK="${GRP_TOPK:-3}"
MIN_TOPK_SIM="${GRP_MIN_TOPK_SIM:-0.75}"
MIN_MEAN_SIM="${GRP_MIN_MEAN_SIM:-0.25}"
MAX_MEAN_SIM="${GRP_MAX_MEAN_SIM:-0.70}"
HIGH_SIMILARITY_INTERVAL_THRESHOLD="${GRP_HIGH_SIMILARITY_INTERVAL_THRESHOLD:-0.82}"
TEMPORAL_NEIGHBORHOOD_SECONDS="${GRP_TEMPORAL_NEIGHBORHOOD_SECONDS:-0}"
PRESERVE_SHARED_ANCHOR_SECONDS="${GRP_PRESERVE_SHARED_ANCHOR_SECONDS:-4}"
MIN_PRUNED_VIDEO_SECONDS="${GRP_MIN_PRUNED_VIDEO_SECONDS:-8}"
RANDOM_SEED="${GRP_RANDOM_SEED:-42}"
DURATION_SECONDS="${GRP_DURATION_SECONDS:-30}"
SAMPLE_INTERVAL_SECONDS="${GRP_SAMPLE_INTERVAL_SECONDS:-1.0}"
START_SECONDS="${GRP_START_SECONDS:-0}"

export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "outdir=${OUTDIR}"
echo "cache_dir=${CACHE_DIR}"
echo "target_count=${TARGET_COUNT}"
echo "max_groups=${MAX_GROUPS}"
echo "clip_model_id=${CLIP_MODEL_ID}"
echo "pairs_per_group=${PAIRS_PER_GROUP}"
echo "topk=${TOPK}"
echo "min_topk_sim=${MIN_TOPK_SIM}"
echo "min_mean_sim=${MIN_MEAN_SIM}"
echo "max_mean_sim=${MAX_MEAN_SIM}"
echo "high_similarity_interval_threshold=${HIGH_SIMILARITY_INTERVAL_THRESHOLD}"
echo "temporal_neighborhood_seconds=${TEMPORAL_NEIGHBORHOOD_SECONDS}"
echo "preserve_shared_anchor_seconds=${PRESERVE_SHARED_ANCHOR_SECONDS}"
echo "min_pruned_video_seconds=${MIN_PRUNED_VIDEO_SECONDS}"
echo "random_seed=${RANDOM_SEED}"
echo "duration_seconds=${DURATION_SECONDS}"
echo "sample_interval_seconds=${SAMPLE_INTERVAL_SECONDS}"
python --version
nvidia-smi || true

echo "stage=build_manifest"
python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json" \
  --no-overlays

echo "stage=mine_group_relative_clip_candidates"
python egolife_two_user_qa/group_relative_clip_sampling.py \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/group_relative_clip_candidates.jsonl" \
  --output-dir "${OUTDIR}/group_relative_clip_sampling" \
  --review-dir "${OUTDIR}/manual_review/group_relative_clip" \
  --cache-dir "${CACHE_DIR}" \
  --model-id "${CLIP_MODEL_ID}" \
  --target-count "${TARGET_COUNT}" \
  --max-groups "${MAX_GROUPS}" \
  --min-group-size 6 \
  --duration-seconds "${DURATION_SECONDS}" \
  --sample-interval-seconds "${SAMPLE_INTERVAL_SECONDS}" \
  --start-seconds "${START_SECONDS}" \
  --pairs-per-group "${PAIRS_PER_GROUP}" \
  --topk "${TOPK}" \
  --min-topk-sim "${MIN_TOPK_SIM}" \
  --min-mean-sim "${MIN_MEAN_SIM}" \
  --max-mean-sim "${MAX_MEAN_SIM}" \
  --high-similarity-interval-threshold "${HIGH_SIMILARITY_INTERVAL_THRESHOLD}" \
  --temporal-neighborhood-seconds "${TEMPORAL_NEIGHBORHOOD_SECONDS}" \
  --preserve-shared-anchor-seconds "${PRESERVE_SHARED_ANCHOR_SECONDS}" \
  --min-pruned-video-seconds "${MIN_PRUNED_VIDEO_SECONDS}" \
  --random-seed "${RANDOM_SEED}" \
  --download-media

echo "outputs:"
echo "  candidates=${OUTDIR}/group_relative_clip_candidates.jsonl"
echo "  review=${OUTDIR}/manual_review/group_relative_clip"
echo "  summary=${OUTDIR}/group_relative_clip_sampling/group_relative_clip_summary.json"
echo "date_done=$(date --iso-8601=seconds)"
