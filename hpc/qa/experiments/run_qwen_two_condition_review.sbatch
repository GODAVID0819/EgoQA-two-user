#!/usr/bin/env bash
#SBATCH --job-name=qwen_review_smoke
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=h100_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=96G
#SBATCH --time=00:50:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

REVIEW_MODE="${REVIEW_MODE:-smoke}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-qwen-two-condition-review-20260901}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/xl6775/experiments/qwen_two_condition_review}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/xl6775/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/xl6775/envs/egoqa-ffmpeg-runtime}"
MODEL_DIR="${MODEL_DIR:-/scratch/xl6775/models/Qwen3.8-27B}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/review_inputs/qwen_two_condition_20260901}"
APPROVED_MARKDOWN="${APPROVED_MARKDOWN:-${INPUT_ROOT}/approved_7.md}"
CURATED_JSONL="${CURATED_JSONL:-${INPUT_ROOT}/curated_17.jsonl}"
: "${MEDIA_ROOT:?MEDIA_ROOT must point to a completed media JobID stitched directory}"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/qwen_two_condition_${REVIEW_MODE}_${SLURM_JOB_ID}"
OUTDIR="${OUTPUT_ROOT}/${REVIEW_MODE}_${SLURM_JOB_ID}"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"

if [[ "${REVIEW_MODE}" == "smoke" ]]; then
  QA_LIMIT_ARGS=(--qa-limit 1)
  EXPECTED_SELECTED=1
elif [[ "${REVIEW_MODE}" == "formal" ]]; then
  QA_LIMIT_ARGS=()
  EXPECTED_SELECTED="${EXPECTED_SELECTED:-21}"
else
  echo "STOP: REVIEW_MODE must be smoke or formal" >&2
  exit 2
fi

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/hf_datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${JOB_SCRATCH_ROOT}/tmp"
export TEMP="${JOB_SCRATCH_ROOT}/tmp"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
export CONDA_PKGS_DIRS="${JOB_SCRATCH_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export QWEN_MEMORY_SAFE_VIDEO_FPS="0.25"
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="3136"
export QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="65536"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="131072"
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB="5"
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="32"
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="sdpa"
export QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS="1"
export QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE="512"
export QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR="${JOB_SCRATCH_ROOT}/qwen_video_cache"

mkdir -p \
  "${OUTDIR}" "${PYTHON_ALIAS_ROOT}" "${HOME}" "${XDG_CACHE_HOME}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${TMPDIR}" \
  "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" "${QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR}"

cd "${PROJECT_ROOT}"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${TRAIN_ENV}"
PYTHON="${TRAIN_ENV}/bin/python"
ln -s "${PROJECT_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

test -x "${PYTHON}"
test -x "${FFMPEG_ENV}/bin/ffmpeg"
test -x "${FFMPEG_ENV}/bin/ffprobe"
test -s "${MODEL_DIR}/config.json"
test -s "${APPROVED_MARKDOWN}"
test -s "${CURATED_JSONL}"
test -d "${MEDIA_ROOT}"
python -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"

if python -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then
  cat "${OUTDIR}/pip_check.txt"
else
  cat "${OUTDIR}/pip_check.txt"
  UNEXPECTED_PIP_CHECK="$(
    grep -Ev '^decord [^[:space:]]+ is not supported on this platform$' \
      "${OUTDIR}/pip_check.txt" || true
  )"
  if [[ -n "${UNEXPECTED_PIP_CHECK}" ]]; then
    echo "STOP: pip check found an unexpected dependency error" >&2
    printf '%s\n' "${UNEXPECTED_PIP_CHECK}" >&2
    exit 1
  fi
  echo "WARNING: pip check reported only the known decord platform metadata warning"
fi

python - <<'PY'
import decord
import torch
import transformers
from qwen_vl_utils.vision_process import get_video_reader_backend

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
backend = get_video_reader_backend()
if backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {backend}")
print(f"torch={torch.__version__}")
print(f"transformers={transformers.__version__}")
print(f"decord={decord.__version__}")
print(f"video_backend={backend}")
PY

python -m egolife_two_user_qa.tools.run_qwen_two_condition_review \
  --approved-markdown "${APPROVED_MARKDOWN}" \
  --curated-jsonl "${CURATED_JSONL}" \
  --media-root "${MEDIA_ROOT}" \
  --output-dir "${OUTDIR}" \
  --model-id "${MODEL_DIR}" \
  --max-new-tokens 256 \
  --max-image-pixels 65536 \
  --disable-thinking \
  "${QA_LIMIT_ARGS[@]}"

python - "${OUTDIR}" "${REVIEW_MODE}" "${EXPECTED_SELECTED}" <<'PY'
import json
import os
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
mode = sys.argv[2]
expected_selected = int(sys.argv[3])
summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
predictions = [
    json.loads(line)
    for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
media_ready_count = int(summary.get("media_ready_count") or 0)
expected_predictions = media_ready_count * 2
base_valid = (
    summary.get("gold_count") == expected_selected
    and len(predictions) == expected_predictions
    and summary.get("inference_error_count") == 0
    and media_ready_count > 0
)
smoke_valid = base_valid and summary.get("paired_count") == 1
valid = smoke_valid if mode == "smoke" else base_valid
status = (
    "passed"
    if valid and summary.get("paired_count") == expected_selected
    else "completed_with_missing_media_and_parse_failures"
    if valid and summary.get("missing_media_count") and summary.get("parse_failure_count")
    else "completed_with_missing_media"
    if valid and summary.get("missing_media_count")
    else "completed_with_parse_failures"
    if valid
    else "failed"
)
result = {
    "status": status,
    "job_id": os.environ["SLURM_JOB_ID"],
    "review_mode": mode,
    "output_dir": str(output_dir),
    "expected_selected": expected_selected,
    "media_ready_count": media_ready_count,
    "missing_media_count": summary.get("missing_media_count"),
    "prediction_count": len(predictions),
    "paired_count": summary.get("paired_count"),
    "parse_failure_count": summary.get("parse_failure_count"),
    "inference_error_count": summary.get("inference_error_count"),
    "summary_path": str(output_dir / "summary.json"),
    "report_path": str(output_dir / "report_cn.md"),
}
(output_dir / "job_result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False))
if not valid:
    raise SystemExit("Qwen two-condition review acceptance failed")
PY

echo "QWEN_TWO_CONDITION_REVIEW_FINISHED result=${OUTDIR}/job_result.json"
