#!/usr/bin/env bash
#SBATCH --job-name=egolife_kgrid10
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/pruning_k_grid_10}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

PAIR_COUNT="${KGRID_PAIR_COUNT:-10}"
MAX_GROUPS="${KGRID_MAX_GROUPS:-80}"
K_VALUES="${KGRID_K_VALUES:-4,8,12,16,20,24,30}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
HIGH_SIMILARITY_THRESHOLD="${KGRID_HIGH_SIMILARITY_THRESHOLD:-0.82}"
MIN_PRUNED_VIDEO_SECONDS="${KGRID_MIN_PRUNED_VIDEO_SECONDS:-8}"
PRUNING_PROTECTION_MODE="${KGRID_PRUNING_PROTECTION_MODE:-min_seconds}"
RANDOM_SEED="${KGRID_RANDOM_SEED:-20260716}"

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
echo "date_start=$(date --iso-8601=seconds)"
echo "outdir=${OUTDIR}"
echo "cache_dir=${CACHE_DIR}"
echo "pair_count=${PAIR_COUNT}"
echo "max_groups=${MAX_GROUPS}"
echo "k_values=${K_VALUES}"
echo "clip_model_id=${CLIP_MODEL_ID}"
echo "high_similarity_threshold=${HIGH_SIMILARITY_THRESHOLD}"
echo "min_pruned_video_seconds=${MIN_PRUNED_VIDEO_SECONDS}"
echo "pruning_protection_mode=${PRUNING_PROTECTION_MODE}"
echo "random_seed=${RANDOM_SEED}"
python --version
nvidia-smi || true

echo "stage=build_manifest"
python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json" \
  --no-overlays

echo "stage=run_pruning_k_grid"
python -m egolife_two_user_qa run_pruning_k_grid \
  --manifest "${OUTDIR}/manifest.json" \
  --output-dir "${OUTDIR}/grid" \
  --cache-dir "${CACHE_DIR}" \
  --pair-count "${PAIR_COUNT}" \
  --max-groups "${MAX_GROUPS}" \
  --min-group-size 2 \
  --k-values "${K_VALUES}" \
  --model-id "${CLIP_MODEL_ID}" \
  --duration-seconds 30 \
  --sample-interval-seconds 1.0 \
  --start-seconds 0 \
  --high-similarity-interval-threshold "${HIGH_SIMILARITY_THRESHOLD}" \
  --preserve-shared-anchor-seconds 0 \
  --min-pruned-video-seconds "${MIN_PRUNED_VIDEO_SECONDS}" \
  --pruning-protection-mode "${PRUNING_PROTECTION_MODE}" \
  --random-seed "${RANDOM_SEED}" \
  --download-media

echo "outputs:"
echo "  summary=${OUTDIR}/grid/summary.json"
echo "  cohort=${OUTDIR}/grid/cohort.jsonl"
echo "  metrics=${OUTDIR}/grid/grid_metrics.csv"
echo "  assignments=${OUTDIR}/grid/cluster_assignments.csv"
echo "  trigger_pairs=${OUTDIR}/grid/trigger_pairs.csv"
echo "  centroid_frames=${OUTDIR}/grid/centroid_frames.csv"
echo "  review=${OUTDIR}/grid/review.html"
echo "  videos=${OUTDIR}/grid/pairs/*/K_*/*.mp4"
echo "date_done=$(date --iso-8601=seconds)"
