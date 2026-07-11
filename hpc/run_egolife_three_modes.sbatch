#!/usr/bin/env bash
#SBATCH --job-name=egolife_3modes
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|h100
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/three_mode_ablation_30}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

# Prepare and attempt the same evidence packets for all generation modes.
EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${EGOLIFE2U_TARGET_COUNT:-30}}"
EVIDENCE_TARGET_COUNT="${EGOLIFE2U_EVIDENCE_TARGET_COUNT:-${EVAL_PACKET_COUNT}}"
MAX_ATTEMPTS="${EGOLIFE2U_MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${EGOLIFE2U_MAX_NEW_TOKENS:-2048}"
RANDOM_SEED="${EGOLIFE2U_RANDOM_SEED:-42}"

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
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
CLIP_MAX_PACKETS="${CLIP_MAX_PACKETS:-${EVIDENCE_TARGET_COUNT}}"
CLIP_CONTACT_SHEETS="${CLIP_CONTACT_SHEETS:-20}"
REUSE_PREPROCESSED="${EGOLIFE2U_REUSE_PREPROCESSED:-0}"
START_MODE="${EGOLIFE2U_START_MODE:-baseline}"
RESUME_GENERATION="${EGOLIFE2U_RESUME_GENERATION:-0}"

export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "outdir=${OUTDIR}"
echo "eval_packet_count=${EVAL_PACKET_COUNT}"
echo "evidence_target_count=${EVIDENCE_TARGET_COUNT}"
echo "clip_model_id=${CLIP_MODEL_ID}"
echo "vlm_backend=${VLM_BACKEND}"
echo "vlm_model_id=${VLM_MODEL_ID}"
if [[ -n "${VLM_API_KEY}" ]]; then
  echo "vlm_api_key=provided"
fi
echo "reuse_preprocessed=${REUSE_PREPROCESSED}"
echo "start_mode=${START_MODE}"
echo "resume_generation=${RESUME_GENERATION}"
python --version
nvidia-smi || true

if [[ "${REUSE_PREPROCESSED}" == "1" ]]; then
  echo "stage=reuse_preprocessed"
  test -s "${OUTDIR}/evidence_manifest.clip_exclusive.jsonl"
else
  echo "stage=build_manifest"
  python -m egolife_two_user_qa build_manifest \
    --output "${OUTDIR}/manifest.json" \
    --no-overlays

  echo "stage=prepare_shared_evidence"
  python -m egolife_two_user_qa prepare_evidence \
    --manifest "${OUTDIR}/manifest.json" \
    --output "${OUTDIR}/evidence_manifest.jsonl" \
    --cache-dir "${CACHE_DIR}" \
    --output-root "${OUTDIR}" \
    --users-per-case 2 \
    --target-count "${EVIDENCE_TARGET_COUNT}" \
    --frames-per-clip 4 \
    --random-seed "${RANDOM_SEED}" \
    --stratify-by-day

  echo "stage=mine_clip_exclusive_candidates_preserve_order"
  python -m egolife_two_user_qa mine_clip_exclusive_candidates \
    --evidence "${OUTDIR}/evidence_manifest.jsonl" \
    --output "${OUTDIR}/evidence_manifest.clip_exclusive.jsonl" \
    --output-dir "${OUTDIR}/clip_exclusive_mined" \
    --model-id "${CLIP_MODEL_ID}" \
    --max-packets "${CLIP_MAX_PACKETS}" \
    --duration-seconds 12 \
    --sample-interval-seconds 1.5 \
    --clusters-per-user 4 \
    --anchor-threshold 0.75 \
    --top-k 3 \
    --contact-sheet-count "${CLIP_CONTACT_SHEETS}" \
    --preserve-order

  echo "stage=materialize_shared_review_videos"
  python -m egolife_two_user_qa materialize_review_videos \
    --evidence "${OUTDIR}/evidence_manifest.clip_exclusive.jsonl" \
    --output-dir "${OUTDIR}/manual_review/shared_evidence_videos"
fi

run_mode() {
  local mode="$1"
  local mode_dir="${OUTDIR}/${mode}"
  local resume_args=()
  if [[ "${RESUME_GENERATION}" == "1" ]]; then
    resume_args+=(--resume)
  fi
  local api_key_args=()
  if [[ -n "${VLM_API_KEY}" ]]; then
    api_key_args+=(--api-key "${VLM_API_KEY}")
  fi
  mkdir -p "${mode_dir}"

  echo "stage=generate_video_qa_loop mode=${mode}"
  python -m egolife_two_user_qa generate_video_qa_loop \
    --evidence "${OUTDIR}/evidence_manifest.clip_exclusive.jsonl" \
    --output "${mode_dir}/qa_mcq.jsonl" \
    --prompts-output "${mode_dir}/video_first_prompts.jsonl" \
    --rejected-output "${mode_dir}/qa_mcq.rejected.jsonl" \
    --intermediate-output "${mode_dir}/qa_mcq.intermediate.jsonl" \
    --target-count "${EVAL_PACKET_COUNT}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --backend "${VLM_BACKEND}" \
    --model-id "${VLM_MODEL_ID}" \
    --dtype bfloat16 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --generation-mode "${mode}" \
    --fixed-question-type-schedule \
    "${api_key_args[@]}" \
    "${resume_args[@]}"

  echo "stage=validate_outputs mode=${mode}"
  python -m egolife_two_user_qa validate_outputs \
    --qa "${mode_dir}/qa_mcq.jsonl" \
    --csv-output "${mode_dir}/qa_mcq.csv" \
    --human-review-output "${mode_dir}/human_review_sheet.md" \
    --report "${mode_dir}/generation_report.md" \
    --strict-review

  echo "stage=materialize_review_videos mode=${mode}"
  python -m egolife_two_user_qa materialize_review_videos \
    --qa "${mode_dir}/qa_mcq.jsonl" \
    --output-dir "${mode_dir}/review_videos"
}

case "${START_MODE}" in
  baseline)
    run_mode baseline
    run_mode clip_guided
    run_mode discovery
    run_mode discovery_control
    ;;
  clip_guided)
    run_mode clip_guided
    run_mode discovery
    run_mode discovery_control
    ;;
  discovery)
    run_mode discovery
    run_mode discovery_control
    ;;
  discovery_control)
    run_mode discovery_control
    ;;
  *)
    echo "Unknown EGOLIFE2U_START_MODE=${START_MODE}; expected baseline, clip_guided, discovery, or discovery_control" >&2
    exit 2
    ;;
esac

SUMMARY_PATH="${OUTDIR}/three_mode_acceptance_summary.tsv"
printf "mode\teval_packets\taccepted\trejected_or_failed\n" > "${SUMMARY_PATH}"
for mode in baseline clip_guided discovery discovery_control; do
  mode_dir="${OUTDIR}/${mode}"
  if [[ -s "${mode_dir}/qa_mcq.jsonl" ]]; then
    accepted_count="$(wc -l < "${mode_dir}/qa_mcq.jsonl")"
    accepted_count="${accepted_count//[[:space:]]/}"
  else
    accepted_count="0"
  fi
  if [[ -s "${mode_dir}/qa_mcq.rejected.jsonl" ]]; then
    rejected_count="$(wc -l < "${mode_dir}/qa_mcq.rejected.jsonl")"
    rejected_count="${rejected_count//[[:space:]]/}"
  else
    rejected_count="0"
  fi
  eval_count="$((accepted_count + rejected_count))"
  printf "%s\t%s\t%s\t%s\n" "${mode}" "${eval_count}" "${accepted_count}" "${rejected_count}" >> "${SUMMARY_PATH}"
done

echo "date_done=$(date --iso-8601=seconds)"
