#!/usr/bin/env bash
#SBATCH --job-name=egolife_disc_ctrl
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=80G
#SBATCH --time=6:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/discovery_control_5}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

# This control reuses existing evidence rows. Video files are expected under:
#   CACHE_DIR/<agent_dir>/<DAY_N>/<video_file> or CACHE_DIR/<agent_dir>/<DAYN>/<video_file>
# for example:
#   /scratch/$USER/egolife_two_user_qa_cache/A1_JAKE/DAY_1/DAY1_A1_JAKE_11100000.mp4
#   /scratch/$USER/egolife_two_user_qa_cache/A1_JAKE/DAY1/DAY1_A1_JAKE_11100000.mp4
# They do not need materialized packet review-video folders.
EVIDENCE_JSONL="${EGOLIFE2U_EVIDENCE_JSONL:-}"
RESOLVED_EVIDENCE_JSONL="${EGOLIFE2U_RESOLVED_EVIDENCE_JSONL:-${OUTDIR}/evidence_manifest.cache_resolved.jsonl}"
EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${EGOLIFE2U_TARGET_COUNT:-5}}"
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
MAX_ATTEMPTS="${EGOLIFE2U_MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${EGOLIFE2U_MAX_NEW_TOKENS:-2048}"
RESUME_GENERATION="${EGOLIFE2U_RESUME_GENERATION:-0}"
MATERIALIZE_REVIEW_VIDEOS="${EGOLIFE2U_MATERIALIZE_REVIEW_VIDEOS:-0}"

export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ -z "${EVIDENCE_JSONL}" ]]; then
  if [[ -s "${OUTDIR}/evidence_manifest.jsonl" ]]; then
    EVIDENCE_JSONL="${OUTDIR}/evidence_manifest.jsonl"
  elif [[ -s "${OUTDIR}/evidence_manifest.clip_exclusive.jsonl" ]]; then
    EVIDENCE_JSONL="${OUTDIR}/evidence_manifest.clip_exclusive.jsonl"
  else
    echo "Set EGOLIFE2U_EVIDENCE_JSONL to the existing five-row evidence JSONL." >&2
    echo "The cache should contain CACHE_DIR/<agent_dir>/<DAY_N>/<video_file> or CACHE_DIR/<agent_dir>/<DAYN>/<video_file>." >&2
    exit 2
  fi
fi

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "outdir=${OUTDIR}"
echo "evidence_jsonl=${EVIDENCE_JSONL}"
echo "resolved_evidence_jsonl=${RESOLVED_EVIDENCE_JSONL}"
echo "eval_packet_count=${EVAL_PACKET_COUNT}"
echo "cache_dir=${CACHE_DIR}"
echo "vlm_backend=${VLM_BACKEND}"
echo "vlm_model_id=${VLM_MODEL_ID}"
if [[ -n "${VLM_API_KEY}" ]]; then
  echo "vlm_api_key=provided"
fi
echo "resume_generation=${RESUME_GENERATION}"
echo "materialize_review_videos=${MATERIALIZE_REVIEW_VIDEOS}"
python --version
nvidia-smi || true

echo "stage=preflight_cached_evidence"
python -m egolife_two_user_qa hpc_preflight_cached_evidence \
  --evidence "${EVIDENCE_JSONL}" \
  --target-count "${EVAL_PACKET_COUNT}" \
  --cache-dir "${CACHE_DIR}" \
  --resolved-output "${RESOLVED_EVIDENCE_JSONL}"

mode_dir="${OUTDIR}/discovery_control"
resume_args=()
if [[ "${RESUME_GENERATION}" == "1" ]]; then
  resume_args+=(--resume)
fi
api_key_args=()
if [[ -n "${VLM_API_KEY}" ]]; then
  api_key_args+=(--api-key "${VLM_API_KEY}")
fi
mkdir -p "${mode_dir}"

echo "stage=generate_video_qa_loop mode=discovery_control"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${RESOLVED_EVIDENCE_JSONL}" \
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
  --generation-mode discovery_control \
  --fixed-question-type-schedule \
  "${api_key_args[@]}" \
  "${resume_args[@]}"

echo "stage=validate_outputs mode=discovery_control"
python -m egolife_two_user_qa validate_outputs \
  --qa "${mode_dir}/qa_mcq.jsonl" \
  --csv-output "${mode_dir}/qa_mcq.csv" \
  --human-review-output "${mode_dir}/human_review_sheet.md" \
  --report "${mode_dir}/generation_report.md" \
  --strict-review

if [[ "${MATERIALIZE_REVIEW_VIDEOS}" == "1" ]]; then
  echo "stage=materialize_review_videos mode=discovery_control"
  python -m egolife_two_user_qa materialize_review_videos \
    --qa "${mode_dir}/qa_mcq.jsonl" \
    --output-dir "${mode_dir}/review_videos"
else
  echo "stage=skip_materialize_review_videos reason=using_cache_local_video_paths"
fi

SUMMARY_PATH="${OUTDIR}/discovery_control_acceptance_summary.tsv"
accepted_count="0"
rejected_count="0"
if [[ -s "${mode_dir}/qa_mcq.jsonl" ]]; then
  accepted_count="$(wc -l < "${mode_dir}/qa_mcq.jsonl")"
  accepted_count="${accepted_count//[[:space:]]/}"
fi
if [[ -s "${mode_dir}/qa_mcq.rejected.jsonl" ]]; then
  rejected_count="$(wc -l < "${mode_dir}/qa_mcq.rejected.jsonl")"
  rejected_count="${rejected_count//[[:space:]]/}"
fi
eval_count="$((accepted_count + rejected_count))"
printf "mode\teval_packets\taccepted\trejected_or_failed\n" > "${SUMMARY_PATH}"
printf "discovery_control\t%s\t%s\t%s\n" "${eval_count}" "${accepted_count}" "${rejected_count}" >> "${SUMMARY_PATH}"

echo "date_done=$(date --iso-8601=seconds)"
