#!/usr/bin/env bash
#SBATCH --job-name=egolife_10m_safe
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200|h100
#SBATCH --mem=160G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

# Dedicated 10-minute path. It does not change or call the historical 30-second
# launcher. All generator and judge calls share one resident Qwen3.6-27B model,
# while the memory-safe backend serializes complete calls on that model.
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_LONG_OUTPUT_ROOT:-egolife_two_user_qa/outputs/three_mode_ablation_10min_memory_safe}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${EGOLIFE2U_TARGET_COUNT:-30}}"
EVIDENCE_TARGET_COUNT="${EGOLIFE2U_EVIDENCE_TARGET_COUNT:-${EVAL_PACKET_COUNT}}"
MAX_ATTEMPTS="${EGOLIFE2U_MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${EGOLIFE2U_MAX_NEW_TOKENS:-2048}"
RANDOM_SEED="${EGOLIFE2U_RANDOM_SEED:-42}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
# K is fixed for this run: 4 clusters / 30 seconds scales to 80 / 600 seconds.
PRUNING_CLUSTERS_PER_VIDEO=80
PRUNING_SAMPLE_INTERVAL_SECONDS="${EGOLIFE2U_PRUNING_SAMPLE_INTERVAL_SECONDS:-1.0}"
PRUNING_HIGH_SIMILARITY_THRESHOLD="${EGOLIFE2U_PRUNING_HIGH_SIMILARITY_THRESHOLD:-0.82}"
PRUNING_MIN_VIDEO_SECONDS="${EGOLIFE2U_PRUNING_MIN_VIDEO_SECONDS:-8}"
PRUNING_CLIP_BATCH_SIZE="${EGOLIFE2U_PRUNING_CLIP_BATCH_SIZE:-32}"
REUSE_PREPROCESSED="${EGOLIFE2U_REUSE_PREPROCESSED:-0}"
RESUME_GENERATION="${EGOLIFE2U_RESUME_GENERATION:-0}"

RAW_EVIDENCE="${OUTDIR}/evidence_manifest.jsonl"
PRUNED_EVIDENCE="${OUTDIR}/evidence_manifest.pruned_k80.jsonl"

# These variables are read only by the opt-in memory-safe runner.
export QWEN_MEMORY_SAFE_VIDEO_FPS="${QWEN_MEMORY_SAFE_VIDEO_FPS:-1.0}"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="${QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS:-131072}"
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB="${QWEN_MEMORY_SAFE_GPU_RESERVE_GIB:-5}"
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="${QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB:-16}"
PREFLIGHT_MIN_AVAILABLE_RAM_GIB="${QWEN_MEMORY_SAFE_PREFLIGHT_MIN_AVAILABLE_RAM_GIB:-96}"
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="${QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION:-flash_attention_2}"
export QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS="${QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS:-1}"
export QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE="${QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE:-512}"
export QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR="${QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR:-${CACHE_DIR}/qwen_memory_safe_video_cache}"
MAX_IMAGE_PIXELS="${QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS:-131072}"

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
echo "pipeline=ten_minute_memory_safe"
echo "outdir=${OUTDIR}"
echo "eval_packet_count=${EVAL_PACKET_COUNT}"
echo "evidence_target_count=${EVIDENCE_TARGET_COUNT}"
echo "model_id=${QWEN_MODEL_ID}"
echo "backend=transformers-local-memory-safe"
echo "generator_evidence=${PRUNED_EVIDENCE}"
echo "pruning_required=true"
echo "pruning_clusters_per_video=${PRUNING_CLUSTERS_PER_VIDEO}"
echo "pruning_sample_interval_seconds=${PRUNING_SAMPLE_INTERVAL_SECONDS}"
echo "pruning_high_similarity_threshold=${PRUNING_HIGH_SIMILARITY_THRESHOLD}"
echo "pruning_min_video_seconds=${PRUNING_MIN_VIDEO_SECONDS}"
echo "pruning_clip_batch_size=${PRUNING_CLIP_BATCH_SIZE}"
echo "one_resident_model=true"
echo "serialized_model_calls=true"
echo "video_fps=${QWEN_MEMORY_SAFE_VIDEO_FPS}"
echo "max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "max_input_tokens=${QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS}"
echo "gpu_workspace_reserve_gib=${QWEN_MEMORY_SAFE_GPU_RESERVE_GIB}"
echo "min_available_ram_gib=${QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB}"
echo "preflight_min_available_ram_gib=${PREFLIGHT_MIN_AVAILABLE_RAM_GIB}"
echo "attn_implementation=${QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION}"
echo "transcode_local_videos=${QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS}"
echo "transcode_max_edge=${QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE}"
echo "memory_safe_video_cache_dir=${QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR}"
echo "reuse_preprocessed=${REUSE_PREPROCESSED}"
echo "generation_mode=baseline"
echo "resume_generation=${RESUME_GENERATION}"
python --version
nvidia-smi || true

if [[ "${QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS}" != "1" ]]; then
  echo "error=memory_safe_long_video_transcode_must_remain_enabled" >&2
  exit 2
fi

echo "stage=memory_safe_preflight"
python - "${QWEN_MEMORY_SAFE_VIDEO_FPS}" "${MAX_IMAGE_PIXELS}" \
  "${QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS}" "${MAX_NEW_TOKENS}" \
  "${QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION}" "${QWEN_MEMORY_SAFE_GPU_RESERVE_GIB}" \
  "${PREFLIGHT_MIN_AVAILABLE_RAM_GIB}" "${QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE}" <<'PY'
import math
import sys

import torch
from egolife_two_user_qa.qwen3vl_runner import available_host_memory_bytes
from transformers.utils import is_flash_attn_2_available

fps = float(sys.argv[1])
max_pixels = int(sys.argv[2])
max_input_tokens = int(sys.argv[3])
max_new_tokens = int(sys.argv[4])
attn_implementation = sys.argv[5]
gpu_reserve_gib = float(sys.argv[6])
preflight_min_available_ram_gib = float(sys.argv[7])
transcode_max_edge = int(sys.argv[8])

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
if attn_implementation == "flash_attention_2" and not is_flash_attn_2_available():
    raise SystemExit("flash_attention_2 is unavailable in this environment")

# Qwen3.6-27B: patch16, spatial merge2, temporal patch2. This conservative
# estimate is independent of any additional total-pixel cap in the processor.
frames_per_video = math.ceil(600 * fps)
tokens_per_temporal_pair = math.ceil(max_pixels / (16 * 16 * 2 * 2))
visual_tokens = 2 * math.ceil(frames_per_video / 2) * tokens_per_temporal_pair
estimated_input_tokens = visual_tokens + 8192
kv_bytes_per_token = 16 * 2 * 4 * 256 * 2
estimated_kv_gib = (estimated_input_tokens + max_new_tokens) * kv_bytes_per_token / 1024**3
estimated_raw_host_tensor_gib = (
    2 * frames_per_video * transcode_max_edge**2 * 3 * 4 / 1024**3
)
estimated_host_video_working_gib = 3 * estimated_raw_host_tensor_gib
available_ram_bytes = available_host_memory_bytes()
if available_ram_bytes is None:
    raise SystemExit("cannot determine system/cgroup available host RAM")
available_ram_gib = available_ram_bytes / 1024**3
weight_gib = 55_562_855_904 / 1024**3
required_device_gib = weight_gib + estimated_kv_gib + gpu_reserve_gib

print(f"preflight_frames_per_video={frames_per_video}")
print(f"preflight_visual_tokens_pair={visual_tokens}")
print(f"preflight_estimated_input_tokens={estimated_input_tokens}")
print(f"preflight_max_input_tokens={max_input_tokens}")
print(f"preflight_estimated_kv_gib={estimated_kv_gib:.3f}")
print(f"preflight_gpu_workspace_reserve_gib={gpu_reserve_gib:.3f}")
print(f"preflight_required_device_gib={required_device_gib:.3f}")
device_total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print(f"preflight_device_total_gib={device_total_gib:.3f}")
print(f"preflight_estimated_raw_host_tensor_gib={estimated_raw_host_tensor_gib:.3f}")
print(f"preflight_estimated_host_video_working_gib={estimated_host_video_working_gib:.3f}")
print(f"preflight_available_ram_gib={available_ram_gib:.3f}")
print(f"preflight_required_available_ram_gib={preflight_min_available_ram_gib:.3f}")
if estimated_input_tokens > max_input_tokens:
    raise SystemExit(
        "configured video budget exceeds the memory-safe input-token ceiling: "
        f"estimated={estimated_input_tokens} ceiling={max_input_tokens}"
    )
if required_device_gib > device_total_gib:
    raise SystemExit(
        "Qwen3.6-27B weights, estimated KV cache, and GPU workspace reserve do not fit: "
        f"required={required_device_gib:.3f} GiB device={device_total_gib:.3f} GiB"
    )
if available_ram_gib < preflight_min_available_ram_gib:
    raise SystemExit(
        "insufficient available host RAM before model loading: "
        f"available={available_ram_gib:.3f} GiB "
        f"required={preflight_min_available_ram_gib:.3f} GiB"
    )
print("memory_safe_preflight=ok")
PY

if [[ "${REUSE_PREPROCESSED}" == "1" ]]; then
  echo "stage=reuse_unpruned_ten_minute_evidence"
  test -s "${RAW_EVIDENCE}"
else
  echo "stage=build_manifest"
  python -m egolife_two_user_qa build_manifest \
    --output "${OUTDIR}/manifest.json" \
    --no-overlays

  echo "stage=prepare_ten_minute_evidence"
  python -m egolife_two_user_qa prepare_evidence \
    --manifest "${OUTDIR}/manifest.json" \
    --output "${RAW_EVIDENCE}" \
    --cache-dir "${CACHE_DIR}" \
    --output-root "${OUTDIR}" \
    --users-per-case 2 \
    --target-count "${EVIDENCE_TARGET_COUNT}" \
    --frames-per-clip 4 \
    --evidence-duration-seconds 600 \
    --random-seed "${RANDOM_SEED}" \
    --stratify-by-day

fi

# This stage is mandatory, including when the assembled 10-minute evidence is
# reused. The QA loop never receives the unpruned evidence manifest.
echo "stage=prune_two_ten_minute_videos_k80"
python -m egolife_two_user_qa.paired_evidence_pruning \
  --evidence "${RAW_EVIDENCE}" \
  --output "${PRUNED_EVIDENCE}" \
  --output-dir "${OUTDIR}/paired_video_pruning_k80" \
  --cache-dir "${CACHE_DIR}" \
  --model-id "${CLIP_MODEL_ID}" \
  --expected-duration-seconds 600 \
  --sample-interval-seconds "${PRUNING_SAMPLE_INTERVAL_SECONDS}" \
  --clusters-per-video "${PRUNING_CLUSTERS_PER_VIDEO}" \
  --high-similarity-threshold "${PRUNING_HIGH_SIMILARITY_THRESHOLD}" \
  --preserve-shared-anchor-seconds 0 \
  --min-pruned-video-seconds "${PRUNING_MIN_VIDEO_SECONDS}" \
  --pruning-protection-mode min_seconds \
  --clip-batch-size "${PRUNING_CLIP_BATCH_SIZE}"
test -s "${PRUNED_EVIDENCE}"

echo "stage=materialize_shared_review_videos"
python -m egolife_two_user_qa materialize_review_videos \
  --evidence "${PRUNED_EVIDENCE}" \
  --output-dir "${OUTDIR}/manual_review/shared_evidence_videos"

run_mode() {
  local mode="$1"
  local mode_dir="${OUTDIR}/${mode}"
  local resume_args=()
  if [[ "${RESUME_GENERATION}" == "1" ]]; then
    resume_args+=(--resume)
  fi
  mkdir -p "${mode_dir}"

  echo "stage=generate_video_qa_loop mode=${mode}"
  python -m egolife_two_user_qa generate_video_qa_loop \
    --evidence "${PRUNED_EVIDENCE}" \
    --output "${mode_dir}/qa_mcq.jsonl" \
    --prompts-output "${mode_dir}/video_first_prompts.jsonl" \
    --rejected-output "${mode_dir}/qa_mcq.rejected.jsonl" \
    --intermediate-output "${mode_dir}/qa_mcq.intermediate.jsonl" \
    --target-count "${EVAL_PACKET_COUNT}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --backend transformers-local-memory-safe \
    --model-id "${QWEN_MODEL_ID}" \
    --dtype bfloat16 \
    --max-image-pixels "${MAX_IMAGE_PIXELS}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --disable-thinking \
    --generation-mode "${mode}" \
    --fixed-question-type-schedule \
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

run_mode baseline

SUMMARY_PATH="${OUTDIR}/baseline_acceptance_summary.tsv"
printf "mode\teval_packets\taccepted\trejected_or_failed\n" > "${SUMMARY_PATH}"
for mode in baseline; do
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
