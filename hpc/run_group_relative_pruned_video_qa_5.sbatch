#!/usr/bin/env bash
#SBATCH --job-name=egolife_grpqa5
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|h100|a100
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/group_relative_pruned_video_qa_5}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-5}"
EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${TARGET_COUNT}}"
MAX_GROUPS="${EGOLIFE2U_MAX_GROUPS:-80}"
MAX_ATTEMPTS="${EGOLIFE2U_MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${EGOLIFE2U_MAX_NEW_TOKENS:-2048}"
GENERATION_MODE="${EGOLIFE2U_GENERATION_MODE:-baseline}"
# Archived modes: clip_guided, discovery, and discovery_control.
if [[ "${GENERATION_MODE}" != "baseline" ]]; then
  echo "Archived generation mode requested: ${GENERATION_MODE}. Only baseline is active." >&2
  exit 2
fi
RESUME_GENERATION="${EGOLIFE2U_RESUME_GENERATION:-0}"

CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
VLM_BACKEND="${VLM_BACKEND:-${EGOLIFE2U_BACKEND:-transformers-local}}"
if [[ -z "${VLM_MODEL_ID:-}" ]]; then
  if [[ "${VLM_BACKEND}" == "gemini" ]]; then
    VLM_MODEL_ID="${GEMINI_MODEL_ID:-gemini-3.5-flash}"
  else
    VLM_MODEL_ID="${QWEN_MODEL_ID}"
  fi
fi
VLM_API_KEY="${VLM_API_KEY:-${GEMINI_API_KEY:-}}"
PAIRS_PER_GROUP="${GRP_PAIRS_PER_GROUP:-1}"
TOPK="${GRP_TOPK:-3}"
MIN_TOPK_SIM="${GRP_MIN_TOPK_SIM:-0.75}"
MIN_MEAN_SIM="${GRP_MIN_MEAN_SIM:-0.25}"
MAX_MEAN_SIM="${GRP_MAX_MEAN_SIM:-0.70}"
HIGH_SIMILARITY_INTERVAL_THRESHOLD="${GRP_HIGH_SIMILARITY_INTERVAL_THRESHOLD:-0.82}"
TEMPORAL_NEIGHBORHOOD_SECONDS="${GRP_TEMPORAL_NEIGHBORHOOD_SECONDS:-0}"
PRESERVE_SHARED_ANCHOR_SECONDS="${GRP_PRESERVE_SHARED_ANCHOR_SECONDS:-4}"
MIN_PRUNED_VIDEO_SECONDS="${GRP_MIN_PRUNED_VIDEO_SECONDS:-8}"
PRUNING_PROTECTION_MODE="${GRP_PRUNING_PROTECTION_MODE:-reject}"
MIN_PRUNED_VIDEO_PERCENT="${GRP_MIN_PRUNED_VIDEO_PERCENT:-}"
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
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "outdir=${OUTDIR}"
echo "cache_dir=${CACHE_DIR}"
echo "target_count=${TARGET_COUNT}"
echo "eval_packet_count=${EVAL_PACKET_COUNT}"
echo "max_groups=${MAX_GROUPS}"
echo "generation_mode=${GENERATION_MODE}"
echo "clip_model_id=${CLIP_MODEL_ID}"
echo "qwen_model_id=${QWEN_MODEL_ID}"
echo "vlm_backend=${VLM_BACKEND}"
echo "vlm_model_id=${VLM_MODEL_ID}"
if [[ -n "${VLM_API_KEY}" ]]; then
  echo "vlm_api_key=provided"
fi
echo "random_seed=${RANDOM_SEED}"
echo "duration_seconds=${DURATION_SECONDS}"
echo "sample_interval_seconds=${SAMPLE_INTERVAL_SECONDS}"
echo "high_similarity_interval_threshold=${HIGH_SIMILARITY_INTERVAL_THRESHOLD}"
echo "preserve_shared_anchor_seconds=${PRESERVE_SHARED_ANCHOR_SECONDS}"
echo "min_pruned_video_seconds=${MIN_PRUNED_VIDEO_SECONDS}"
echo "pruning_protection_mode=${PRUNING_PROTECTION_MODE}"
echo "min_pruned_video_percent=${MIN_PRUNED_VIDEO_PERCENT}"
python --version
nvidia-smi || true

echo "stage=build_manifest"
python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json" \
  --no-overlays

echo "stage=mine_group_relative_pruned_video_candidates"
protection_args=(--pruning-protection-mode "${PRUNING_PROTECTION_MODE}")
if [[ -n "${MIN_PRUNED_VIDEO_PERCENT}" ]]; then
  protection_args+=(--min-pruned-video-percent "${MIN_PRUNED_VIDEO_PERCENT}")
fi
python egolife_two_user_qa/group_relative_clip_sampling.py \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/group_relative_pruned_video_candidates.jsonl" \
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
  "${protection_args[@]}" \
  --random-seed "${RANDOM_SEED}" \
  --download-media

resume_args=()
if [[ "${RESUME_GENERATION}" == "1" ]]; then
  resume_args+=(--resume)
fi

api_key_args=()
if [[ -n "${VLM_API_KEY}" ]]; then
  api_key_args+=(--api-key "${VLM_API_KEY}")
fi

QA_DIR="${OUTDIR}/${GENERATION_MODE}"
mkdir -p "${QA_DIR}"

echo "stage=generate_video_qa_loop"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${OUTDIR}/group_relative_pruned_video_candidates.jsonl" \
  --output "${QA_DIR}/qa_mcq.jsonl" \
  --prompts-output "${QA_DIR}/video_first_prompts.jsonl" \
  --rejected-output "${QA_DIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${QA_DIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${EVAL_PACKET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend "${VLM_BACKEND}" \
  --model-id "${VLM_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generation-mode "${GENERATION_MODE}" \
  --fixed-question-type-schedule \
  "${api_key_args[@]}" \
  "${resume_args[@]}"

echo "stage=validate_outputs"
python -m egolife_two_user_qa validate_outputs \
  --qa "${QA_DIR}/qa_mcq.jsonl" \
  --csv-output "${QA_DIR}/qa_mcq.csv" \
  --human-review-output "${QA_DIR}/human_review_sheet.md" \
  --report "${QA_DIR}/generation_report.md" \
  --strict-review

echo "stage=materialize_review_videos"
python -m egolife_two_user_qa materialize_review_videos \
  --qa "${QA_DIR}/qa_mcq.jsonl" \
  --output-dir "${QA_DIR}/review_videos"

echo "outputs:"
echo "  candidates=${OUTDIR}/group_relative_pruned_video_candidates.jsonl"
echo "  sampling_summary=${OUTDIR}/group_relative_clip_sampling/group_relative_clip_summary.json"
echo "  qa=${QA_DIR}/qa_mcq.jsonl"
echo "  rejected=${QA_DIR}/qa_mcq.rejected.jsonl"
echo "  report=${QA_DIR}/generation_report.md"
echo "  review=${QA_DIR}/human_review_sheet.md"
echo "date_done=$(date --iso-8601=seconds)"
