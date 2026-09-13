#!/usr/bin/env bash
#SBATCH --job-name=egoqa_gemini_smoke5
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

# CPU-only OpenRouter job: intentionally no --gres or --gpus request.

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "error=missing_openrouter_api_key" >&2
  exit 2
fi

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs/curated_10min_openrouter_smoke}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"

if [[ -s "${HPC_ROOT}/shared/env_qwen3vl.sh" ]]; then
  ENV_SCRIPT="${HPC_ROOT}/shared/env_qwen3vl.sh"
else
  ENV_SCRIPT="${HPC_ROOT}/env_qwen3vl.sh"
fi

EVIDENCE_OWNER="${EVIDENCE_OWNER:-hm2991}"
TEN_MINUTE_ASSET_ROOT="${TEN_MINUTE_ASSET_ROOT:-/scratch/${EVIDENCE_OWNER}/Long-video-understanding-clip/egolife_two_user_qa/multi-user/outputs/six_user_qa_10min/six_user_qa_10min_time_aware_16463998/ten_minute_setup/time_aware_candidates/assets}"
SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-/scratch/${EVIDENCE_OWNER}/egolife_two_user_qa_cache}"

CURATED_QA="${CURATED_QA:-${PACKAGE_ROOT}/qa_curated_5_gemini_smoke.jsonl}"
PREPARER="${PREPARER:-${PACKAGE_ROOT}/prepare_curated_10min_ablation.py}"
DRIVER="${DRIVER:-${PACKAGE_ROOT}/curated_openrouter_ablation.py}"
MODEL_ID="${MODEL_ID:-google/gemini-2.5-flash:nitro}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
REASONING_EFFORT="${REASONING_EFFORT:-none}"

RUN_LABEL="${RUN_LABEL:-curated_gemini25_flash_smoke5_${SLURM_JOB_ID}}"
if [[ ! "${RUN_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error=invalid_run_label value=${RUN_LABEL}" >&2
  exit 2
fi
OUTDIR="${OUTPUT_ROOT}/${RUN_LABEL}"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/${RUN_LABEL}"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"
PYTHON=""
CURRENT_STAGE="bootstrap"
STORAGE_PREFLIGHT_MODULE=""

write_status() {
  local status_value="$1"
  local stage_value="$2"
  local exit_code_value="${3:-0}"
  if [[ -d "${OUTDIR}" ]]; then
    printf '{"status":"%s","stage":"%s","job_id":"%s","exit_code":%s,"timestamp":"%s"}\n' \
      "${status_value}" "${stage_value}" "${SLURM_JOB_ID}" \
      "${exit_code_value}" "$(date --iso-8601=seconds)" \
      > "${OUTDIR}/stage_status.json"
  fi
}

stage_start() {
  CURRENT_STAGE="$1"
  printf 'stage=%s status=started timestamp=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)"
  write_status "running" "${CURRENT_STAGE}" 0
}

on_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'stage=%s status=failed exit_code=%s line=%s command=%q\n' \
    "${CURRENT_STAGE}" "${exit_code}" "${line_number}" "${failed_command}" >&2
  write_status "failed" "${CURRENT_STAGE}" "${exit_code}"
  exit "${exit_code}"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

stage_start "validate_project_layout"
cd "${PROJECT_ROOT}"
test -d "${HPC_ROOT}"
test -s "${ENV_SCRIPT}"
test -d "${PACKAGE_ROOT}"
test -s "${CURATED_QA}"
test -s "${PREPARER}"
test -s "${DRIVER}"
test -s "${PACKAGE_ROOT}/qwen3vl_runner.py"
test -s "${PACKAGE_ROOT}/six_video_qa_tester.py"
test -d "${TEN_MINUTE_ASSET_ROOT}"
test -d "${SOURCE_CACHE_ROOT}"
if [[ -s "${PACKAGE_ROOT}/training/torch_storage_preflight.py" ]]; then
  STORAGE_PREFLIGHT_MODULE="egolife_two_user_qa.training.torch_storage_preflight"
elif [[ -s "${PROJECT_ROOT}/training/torch_storage_preflight.py" ]]; then
  STORAGE_PREFLIGHT_MODULE="training.torch_storage_preflight"
else
  echo "error=torch_storage_preflight_missing" >&2
  exit 2
fi
mkdir -p "${HPC_ROOT}/logs" "${OUTDIR}" "${JOB_SCRATCH_ROOT}" "${PYTHON_ALIAS_ROOT}"

stage_start "activate_environment"
export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-${TRAIN_ENV}}"
export QWEN3VL_PROJECT_ROOT="${PROJECT_ROOT}"
source "${ENV_SCRIPT}"
PYTHON="$(command -v python)"
test -x "${PYTHON}"

stage_start "configure_job_storage"
export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/hf_datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${TMPDIR}" \
  "${PIP_CACHE_DIR}" "${JOB_SCRATCH_ROOT}/openrouter_video_cache"
ln -s "${PACKAGE_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

stage_start "resolve_video_runtime"
if [[ -x "${FFMPEG_ENV}/bin/ffmpeg" && -x "${FFMPEG_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${FFMPEG_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${FFMPEG_ENV}/bin/ffprobe"
  FFMPEG_LIBRARY_DIR="${FFMPEG_ENV}/lib"
elif [[ -x "${TRAIN_ENV}/bin/ffmpeg" && -x "${TRAIN_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${TRAIN_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${TRAIN_ENV}/bin/ffprobe"
  FFMPEG_LIBRARY_DIR="${TRAIN_ENV}/lib"
else
  FFMPEG_BINARY="$(command -v ffmpeg || true)"
  FFPROBE_BINARY="$(command -v ffprobe || true)"
  FFMPEG_LIBRARY_DIR=""
fi
if [[ -z "${FFMPEG_BINARY}" || ! -x "${FFMPEG_BINARY}" || \
      -z "${FFPROBE_BINARY}" || ! -x "${FFPROBE_BINARY}" ]]; then
  echo "error=ffmpeg_runtime_missing" >&2
  exit 2
fi
export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"
if [[ -d "${FFMPEG_LIBRARY_DIR}" ]]; then
  export LD_LIBRARY_PATH="${FFMPEG_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

export OPENROUTER_VIDEO_MAX_EDGE="${OPENROUTER_VIDEO_MAX_EDGE:-512}"
export OPENROUTER_VIDEO_FPS="${OPENROUTER_VIDEO_FPS:-0.5}"
export OPENROUTER_VIDEO_CRF="${OPENROUTER_VIDEO_CRF:-30}"
export OPENROUTER_VIDEO_CACHE_DIR="${JOB_SCRATCH_ROOT}/openrouter_video_cache"
export OPENROUTER_MAX_RETRIES=0

stage_start "storage_preflight"
"${PYTHON}" -P -m "${STORAGE_PREFLIGHT_MODULE}" \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"

stage_start "runtime_preflight"
"${FFMPEG_BINARY}" -version | head -n 1
"${FFPROBE_BINARY}" -version | head -n 1
"${PYTHON}" -P - "${PACKAGE_ROOT}" <<'PY'
import sys
from pathlib import Path

import egolife_two_user_qa
from torchcodec.decoders import VideoDecoder

expected = Path(sys.argv[1]).resolve()
actual = Path(egolife_two_user_qa.__file__).resolve().parent
if actual != expected:
    raise SystemExit(f"wrong package imported: expected={expected} actual={actual}")
print(f"package_root={actual}")
print(f"torchcodec_decoder={VideoDecoder.__module__}.{VideoDecoder.__name__}")
PY

stage_start "write_run_manifest"
export RUN_MANIFEST_PATH="${OUTDIR}/run_manifest.json"
export RUN_OUTPUT_DIR="${OUTDIR}"
export RUN_JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT}"
export RUN_CURATED_QA="${CURATED_QA}"
export RUN_MODEL_ID="${MODEL_ID}"
export RUN_REASONING_EFFORT="${REASONING_EFFORT}"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": "curated_openrouter_smoke_run_v1",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "output_dir": os.environ["RUN_OUTPUT_DIR"],
    "job_scratch_root": os.environ["RUN_JOB_SCRATCH_ROOT"],
    "curated_qa": os.environ["RUN_CURATED_QA"],
    "question_count": 5,
    "logical_model_call_count": 10,
    "condition_order": ["six_users", "required_users"],
    "model_id": os.environ["RUN_MODEL_ID"],
    "cpu_only": True,
    "openrouter_routing_variant": "nitro",
    "reasoning_effort": os.environ["RUN_REASONING_EFFORT"],
    "openrouter_video_max_edge": int(os.environ["OPENROUTER_VIDEO_MAX_EDGE"]),
    "openrouter_video_fps": float(os.environ["OPENROUTER_VIDEO_FPS"]),
    "openrouter_video_crf": int(os.environ["OPENROUTER_VIDEO_CRF"]),
    "automatic_retry_count": int(os.environ["OPENROUTER_MAX_RETRIES"]),
}
Path(os.environ["RUN_MANIFEST_PATH"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

stage_start "prepare_ten_minute_media"
"${PYTHON}" -P "${PREPARER}" \
  --qa "${CURATED_QA}" \
  --ten-minute-asset-root "${TEN_MINUTE_ASSET_ROOT}" \
  --source-cache-root "${SOURCE_CACHE_ROOT}" \
  --materialized-media-root "${OUTDIR}/prepared_media" \
  --evidence-output "${OUTDIR}/evidence_required_pairs.jsonl" \
  --six-view-output "${OUTDIR}/six_view_manifest.jsonl" \
  --provenance-output "${OUTDIR}/media_provenance.jsonl" \
  --summary-output "${OUTDIR}/media_preparation_summary.json" \
  --ffmpeg "${FFMPEG_BINARY}" \
  --ffprobe "${FFPROBE_BINARY}" \
  --expected-count 5

stage_start "validate_then_run_six_and_required"
"${PYTHON}" -P -m egolife_two_user_qa.curated_openrouter_ablation \
  --qa "${CURATED_QA}" \
  --evidence "${OUTDIR}/evidence_required_pairs.jsonl" \
  --six-view-manifest "${OUTDIR}/six_view_manifest.jsonl" \
  --output-dir "${OUTDIR}/gemini25_flash" \
  --expected-count 5 \
  --model-id "${MODEL_ID}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --reasoning-effort "${REASONING_EFFORT}"

CURRENT_STAGE="complete"
write_status "completed" "complete" 0
printf 'status=completed output=%s summary=%s\n' \
  "${OUTDIR}" "${OUTDIR}/gemini25_flash/paired_summary.json"
