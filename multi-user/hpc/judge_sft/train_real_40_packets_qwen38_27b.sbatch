#!/usr/bin/env bash
#SBATCH --job-name=judge38-real40
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
HPC_ROOT="${HPC_ROOT:-${PROJECT_ROOT}/hpc}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/scratch/${USER}/egolife_vllm_runtime/huggingface}"
MODEL_PATH="${MODEL_PATH:-}"
MANIFEST_DIR="${MANIFEST_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/egolife_judge_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/train_real40_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/job_scratch/judge38_real40_${SLURM_JOB_ID}}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${PACKAGE_ROOT}/training/judge_sft/deepspeed_zero3.json}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-262144}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-262144}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
EPOCHS="${EPOCHS:-10}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-16}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
ENABLE_CUDA_KEEPER="${ENABLE_CUDA_KEEPER:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda_high_duty.py}"
CUDA_KEEPER_PAUSE_FILE="${CUDA_KEEPER_PAUSE_FILE:-${JOB_SCRATCH_ROOT}/cuda_keeper.pause}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-all}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-15.0}"
CUDA_KEEPER_MAX_PREALLOC_GB="${CUDA_KEEPER_MAX_PREALLOC_GB:-0.25}"
CUDA_KEEPER_EMERGENCY_MIN_FREE_GB="${CUDA_KEEPER_EMERGENCY_MIN_FREE_GB:-2.0}"
CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-1}"
CUDA_KEEPER_WINDOW_SECONDS="${CUDA_KEEPER_WINDOW_SECONDS:-300}"
CUDA_KEEPER_TARGET_MARGIN="${CUDA_KEEPER_TARGET_MARGIN:-40}"
CUDA_KEEPER_IDLE_UTIL_THRESHOLD="${CUDA_KEEPER_IDLE_UTIL_THRESHOLD:-10}"
CUDA_KEEPER_IDLE_CONFIRM_SECONDS="${CUDA_KEEPER_IDLE_CONFIRM_SECONDS:-0.0}"
CUDA_KEEPER_MAX_PULSE_SECONDS="${CUDA_KEEPER_MAX_PULSE_SECONDS:-0.8}"
CUDA_KEEPER_ACTIVE_REQUEST_IDLE_CONFIRM_SECONDS="${CUDA_KEEPER_ACTIVE_REQUEST_IDLE_CONFIRM_SECONDS:-1.0}"
CUDA_KEEPER_ACTIVE_REQUEST_MAX_PULSE_SECONDS="${CUDA_KEEPER_ACTIVE_REQUEST_MAX_PULSE_SECONDS:-0.30}"

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
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONPATH="${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_KEEPER_PAUSE_FILE

PYTHON="${TRAIN_ENV}/bin/python"
TORCHRUN="${TRAIN_ENV}/bin/torchrun"
TRAIN_MANIFEST="${MANIFEST_DIR}/train.jsonl"
CURRENT_STAGE="bootstrap"
GPU_MONITOR_PID=""
CUDA_KEEPER_PID=""
CUDA_KEEPER_LOG="${OUTPUT_DIR}/cuda_keeper.log"

write_status() {
  local status="$1"
  local exit_code="${2:-0}"
  if [[ -d "${OUTPUT_DIR}" ]]; then
    printf '{"status":"%s","stage":"%s","job_id":"%s","exit_code":%s}\n' \
      "${status}" "${CURRENT_STAGE}" "${SLURM_JOB_ID}" "${exit_code}" \
      > "${OUTPUT_DIR}/stage_status.json"
  fi
}

cleanup() {
  rm -f -- "${CUDA_KEEPER_PAUSE_FILE}" 2>/dev/null || true
  if [[ -n "${GPU_MONITOR_PID}" ]] && kill -0 "${GPU_MONITOR_PID}" 2>/dev/null; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}

start_cuda_keeper() {
  [[ "${ENABLE_CUDA_KEEPER}" == "1" ]] || return 0
  CURRENT_STAGE="start_cuda_keeper"
  rm -f -- "${CUDA_KEEPER_PAUSE_FILE}"
  "${PYTHON}" -P -c "import torch, pynvml, psutil"
  PYTHONUNBUFFERED=1 "${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}" \
    --threshold "${CUDA_KEEPER_THRESHOLD}" \
    --gpus "${CUDA_KEEPER_GPUS}" \
    --reserve "${CUDA_KEEPER_RESERVE}" \
    --max-prealloc "${CUDA_KEEPER_MAX_PREALLOC_GB}" \
    --emergency-min-free-gb "${CUDA_KEEPER_EMERGENCY_MIN_FREE_GB}" \
    --start-used-mib "${CUDA_KEEPER_START_USED_MIB}" \
    --window-seconds "${CUDA_KEEPER_WINDOW_SECONDS}" \
    --target-margin "${CUDA_KEEPER_TARGET_MARGIN}" \
    --idle-util-threshold "${CUDA_KEEPER_IDLE_UTIL_THRESHOLD}" \
    --idle-confirm-seconds "${CUDA_KEEPER_IDLE_CONFIRM_SECONDS}" \
    --max-pulse-seconds "${CUDA_KEEPER_MAX_PULSE_SECONDS}" \
    --active-request-idle-confirm-seconds "${CUDA_KEEPER_ACTIVE_REQUEST_IDLE_CONFIRM_SECONDS}" \
    --active-request-max-pulse-seconds "${CUDA_KEEPER_ACTIVE_REQUEST_MAX_PULSE_SECONDS}" \
    --pause-file "${CUDA_KEEPER_PAUSE_FILE}" \
    > "${CUDA_KEEPER_LOG}" 2>&1 &
  CUDA_KEEPER_PID=$!
  sleep 3
  if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    echo "error=cuda_keeper_exited_early log=${CUDA_KEEPER_LOG}" >&2
    sed -n '1,200p' "${CUDA_KEEPER_LOG}" >&2 || true
    exit 2
  fi
  printf 'cuda_keeper=enabled pid=%s start_used_mib=%s log=%s\n' \
    "${CUDA_KEEPER_PID}" "${CUDA_KEEPER_START_USED_MIB}" "${CUDA_KEEPER_LOG}"
}

on_error() {
  local exit_code=$?
  trap - ERR
  write_status failed "${exit_code}"
  printf 'judge real40 training failed stage=%s exit_code=%s\n' \
    "${CURRENT_STAGE}" "${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR
trap cleanup EXIT

mkdir -p "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${TMPDIR}"
cd "${PACKAGE_ROOT}"
if [[ "${ENABLE_CUDA_KEEPER}" != "0" && "${ENABLE_CUDA_KEEPER}" != "1" ]]; then
  echo "error=invalid_enable_cuda_keeper value=${ENABLE_CUDA_KEEPER}" >&2
  exit 2
fi
if [[ "${ENABLE_CUDA_KEEPER}" == "1" ]]; then
  [[ -s "${CUDA_KEEPER_SCRIPT}" ]] || {
    echo "Missing CUDA keeper script: ${CUDA_KEEPER_SCRIPT}" >&2
    exit 2
  }
fi
[[ -n "${MANIFEST_DIR}" ]] || {
  echo "Set MANIFEST_DIR to the exact output of prepare_real_manifests.sbatch." >&2
  exit 2
}
for required in \
  "${PYTHON}" "${TORCHRUN}" "${TRAIN_MANIFEST}" \
  "${MANIFEST_DIR}/data_audit.json" "${DEEPSPEED_CONFIG}"; do
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
cp "${MANIFEST_DIR}/data_audit.json" "${OUTPUT_DIR}/input_data_audit.json"
"${PYTHON}" -P -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
  > "${OUTPUT_DIR}/gpu_environment.csv"
"${PYTHON}" -P - "${ATTN_IMPLEMENTATION}" <<'PY' > "${OUTPUT_DIR}/python_environment.txt"
import sys

import torch
import transformers
import accelerate
import deepspeed
import safetensors
from peft import LoraConfig
from qwen_vl_utils import process_vision_info

assert torch.cuda.is_available(), "CUDA is unavailable"
assert torch.cuda.device_count() == 2, torch.cuda.device_count()
assert all(torch.cuda.get_device_capability(i)[0] >= 9 for i in range(2))
assert torch.cuda.is_bf16_supported(), "BF16 is unavailable"
assert getattr(transformers, "AutoModelForMultimodalLM", None) is not None
attn_implementation = sys.argv[1]
if attn_implementation == "flash_attention_2":
    import flash_attn

    print("flash_attn=", flash_attn.__version__)
else:
    print("attention_implementation=", attn_implementation)
print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("accelerate=", accelerate.__version__)
print("deepspeed=", deepspeed.__version__)
print("safetensors=", safetensors.__version__)
print("gpus=", [torch.cuda.get_device_name(i) for i in range(2)])
print("peft=", LoraConfig.__module__)
print("qwen_vl_utils=", process_vision_info.__module__)
PY

CURRENT_STAGE="chat_template_preflight"
"${PYTHON}" -P - "${MODEL_PATH}" > "${OUTPUT_DIR}/chat_template_preflight.txt" <<'PY'
import sys

from transformers import AutoProcessor

from training.judge_sft.collator import (
    _apply_chat_template,
    _assert_thinking_disabled,
)

processor = AutoProcessor.from_pretrained(
    sys.argv[1], trust_remote_code=True, local_files_only=True
)
messages = [{"role": "user", "content": [{"type": "text", "text": "template probe"}]}]
rendered = _apply_chat_template(processor, messages)
_assert_thinking_disabled(rendered)
print(repr(rendered[-120:]))
PY

start_cuda_keeper

CURRENT_STAGE="train"
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv -l 10 > "${OUTPUT_DIR}/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
TRAIN_COMMAND=(
  "${TORCHRUN}" --standalone --nproc_per_node=2 -m training.judge_sft.train
  --train-manifest "${TRAIN_MANIFEST}"
  --output-dir "${OUTPUT_DIR}/trainer"
  --model-id "${MODEL_PATH}"
  --local-files-only
  --attn-implementation "${ATTN_IMPLEMENTATION}"
  --min-pixels "${MIN_PIXELS}"
  --max-pixels "${MAX_PIXELS}"
  --max-input-tokens "${MAX_INPUT_TOKENS}"
  --epochs "${EPOCHS}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --deepspeed "${DEEPSPEED_CONFIG}"
)
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  [[ -d "${RESUME_FROM_CHECKPOINT}" ]] || {
    echo "Missing resume checkpoint: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 2
  }
  TRAIN_COMMAND+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
"${TRAIN_COMMAND[@]}"

CURRENT_STAGE="checkpoint_inventory"
"${PYTHON}" -P - "${OUTPUT_DIR}/trainer" "${EPOCHS}" \
  > "${OUTPUT_DIR}/checkpoint_inventory.json" <<'PY'
import json
import math
from pathlib import Path
import sys

trainer_dir = Path(sys.argv[1])
expected = math.ceil(float(sys.argv[2]))
checkpoints = sorted(
    (path for path in trainer_dir.glob("checkpoint-*") if path.is_dir()),
    key=lambda path: int(path.name.rsplit("-", 1)[1]),
)
if len(checkpoints) < expected:
    raise SystemExit(
        f"expected at least {expected} epoch checkpoints, found {len(checkpoints)}"
    )
print(
    json.dumps(
        {
            "status": "passed",
            "expected_epoch_checkpoints": expected,
            "checkpoints": [path.name for path in checkpoints],
        },
        indent=2,
    )
)
PY

CURRENT_STAGE="adapter_reload_and_binary_logit_probe"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -P -m training.judge_sft.adapter_reload \
  --model-id "${MODEL_PATH}" \
  --adapter-dir "${OUTPUT_DIR}/trainer/final_adapter" \
  --attn-implementation "${ATTN_IMPLEMENTATION}" \
  --output "${OUTPUT_DIR}/adapter_reload.json"

CURRENT_STAGE="postcondition"
for required in \
  "${OUTPUT_DIR}/trainer/training_contract.json" \
  "${OUTPUT_DIR}/trainer/trainer_state.json" \
  "${OUTPUT_DIR}/trainer/final_adapter/adapter_config.json" \
  "${OUTPUT_DIR}/checkpoint_inventory.json" \
  "${OUTPUT_DIR}/adapter_reload.json"; do
  [[ -f "${required}" ]] || { echo "Missing training artifact: ${required}" >&2; exit 2; }
done

CURRENT_STAGE="complete"
write_status passed 0
printf 'judge_real40_training_status=passed output=%s\n' "${OUTPUT_DIR}"
