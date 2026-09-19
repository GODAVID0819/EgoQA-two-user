#!/usr/bin/env bash
#SBATCH --job-name=judge38-frame-probe
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=480G
#SBATCH --time=04:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/scratch/${USER}/egolife_vllm_runtime/huggingface}"
MODEL_PATH="${MODEL_PATH:-}"
MANIFEST_DIR="${MANIFEST_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/egolife_judge_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/runtime_probe_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/job_scratch/judge38_frame_probe_${SLURM_JOB_ID}}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-262144}"
IMAGE_CONTEXT_TARGET_FRACTION="${IMAGE_CONTEXT_TARGET_FRACTION:-0.85}"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"
export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
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
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON="${TRAIN_ENV}/bin/python"
TRAIN_MANIFEST="${MANIFEST_DIR}/train.jsonl"
CURRENT_STAGE="bootstrap"

write_status() {
  local status="$1"
  local exit_code="${2:-0}"
  if [[ -d "${OUTPUT_DIR}" ]]; then
    printf '{"status":"%s","stage":"%s","job_id":"%s","exit_code":%s}\n' \
      "${status}" "${CURRENT_STAGE}" "${SLURM_JOB_ID}" "${exit_code}" \
      > "${OUTPUT_DIR}/stage_status.json"
  fi
}

on_error() {
  local exit_code=$?
  trap - ERR
  write_status failed "${exit_code}"
  printf 'judge frame runtime probe failed stage=%s exit_code=%s\n' \
    "${CURRENT_STAGE}" "${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

mkdir -p \
  "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${TMPDIR}" \
  "${XDG_CACHE_HOME}/torch/kernels"
cd "${PACKAGE_ROOT}"

[[ -n "${MANIFEST_DIR}" ]] || {
  echo "Set MANIFEST_DIR to the exact output of prepare_real_manifests.sbatch." >&2
  exit 2
}
for required in "${PYTHON}" "${TRAIN_MANIFEST}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

if [[ -z "${MODEL_PATH}" ]]; then
  mapfile -t MODEL_SNAPSHOTS < <(
    find "${MODEL_CACHE_ROOT}/hub/models--Qwen--Qwen3.8-27B/snapshots" \
      -mindepth 1 -maxdepth 1 -type d -print 2>/dev/null | sort
  )
  if [[ "${#MODEL_SNAPSHOTS[@]}" != "1" ]]; then
    echo "Set MODEL_PATH to one exact local Qwen3.8-27B snapshot; found ${#MODEL_SNAPSHOTS[@]}." >&2
    exit 2
  fi
  MODEL_PATH="${MODEL_SNAPSHOTS[0]}"
fi
[[ -d "${MODEL_PATH}" ]] || { echo "Missing model snapshot: ${MODEL_PATH}" >&2; exit 2; }

CURRENT_STAGE="storage_preflight"
"${PYTHON}" -P -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"

CURRENT_STAGE="environment_preflight"
"${PYTHON}" -P -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
  > "${OUTPUT_DIR}/gpu_environment.csv"
"${PYTHON}" -P - <<'PY' > "${OUTPUT_DIR}/python_environment.txt"
import torch
import transformers
from peft import LoraConfig
from qwen_vl_utils import process_vision_info

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
assert torch.cuda.is_bf16_supported(), "BF16 is unavailable"
assert getattr(transformers, "AutoModelForMultimodalLM", None) is not None
print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("gpu=", torch.cuda.get_device_name(0))
print("peft=", LoraConfig.__module__)
print("qwen_vl_utils=", process_vision_info.__module__)
PY

CURRENT_STAGE="select_real_example"
SMOKE_EXAMPLE_ID="$(
  "${PYTHON}" -P -m training.judge_sft.select_smoke_example \
    --manifest "${TRAIN_MANIFEST}" \
    --output "${OUTPUT_DIR}/selected_example.json"
)"
[[ -n "${SMOKE_EXAMPLE_ID}" ]] || { echo "No smoke example selected." >&2; exit 2; }

CURRENT_STAGE="processor_and_1800_frame_probe"
"${PYTHON}" -P -m training.judge_sft.runtime_probe \
  --manifest "${TRAIN_MANIFEST}" \
  --example-id "${SMOKE_EXAMPLE_ID}" \
  --model-id "${MODEL_PATH}" \
  --local-files-only \
  --min-pixels "${MIN_PIXELS}" \
  --max-pixels "${MAX_PIXELS}" \
  --max-input-tokens "${MAX_INPUT_TOKENS}" \
  --image-context-target-fraction "${IMAGE_CONTEXT_TARGET_FRACTION}" \
  --output "${OUTPUT_DIR}/runtime_probe.json"

CURRENT_STAGE="complete"
write_status passed 0
printf 'judge_frame_runtime_probe_status=passed output=%s example_id=%s\n' \
  "${OUTPUT_DIR}" "${SMOKE_EXAMPLE_ID}"
