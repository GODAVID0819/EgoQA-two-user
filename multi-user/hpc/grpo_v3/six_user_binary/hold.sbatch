#!/usr/bin/env bash
# 保留本次分配，后续通过 srun --jobid 接入；不固定节点。
set -euo pipefail
: "${PROJECT_ROOT:?缺少 PROJECT_ROOT}"
: "${TRAIN_PYTHON:?缺少 TRAIN_PYTHON}"
: "${SLURM_JOB_ID:?必须在 Slurm 中运行}"
JOB_START_EPOCH=$(date +%s)
OUTPUT_DIR="${PROJECT_ROOT}/outputs/allocation_${SLURM_JOB_ID}"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/grpo_allocation_${SLURM_JOB_ID}"
export PYTHONPATH="${PROJECT_ROOT}/multi-user"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
for variable in HOME XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE MODELSCOPE_CACHE TORCH_HOME \
  TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT CUDA_CACHE_PATH \
  FLASHINFER_WORKSPACE_BASE TMPDIR TMP TEMP PIP_CACHE_DIR; do
  export "${variable}=${JOB_SCRATCH_ROOT}/${variable,,}"
done
mkdir -p "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}/multi-user"
"${TRAIN_PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" --output "${OUTPUT_DIR}/storage_preflight.json"
export CUDA_KEEPER_ENABLE=1
export CUDA_KEEPER_SCRIPT="${PROJECT_ROOT}/hpc/shared/cuda.py"
export CUDA_KEEPER_GPUS=all
export CUDA_KEEPER_START_AFTER_SECONDS=7200
export CUDA_KEEPER_RESERVE_GB=8.0
export CUDA_KEEPER_MAX_PREALLOC_GB=0.25
KEEPER_PID=""
cleanup() {
  if [[ -n "${KEEPER_PID}" ]]; then
    kill -TERM "${KEEPER_PID}" 2>/dev/null || true
    wait "${KEEPER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT
"${TRAIN_PYTHON}" -u "${CUDA_KEEPER_SCRIPT}" \
  --gpus "${CUDA_KEEPER_GPUS}" --threshold 75 --reserve "${CUDA_KEEPER_RESERVE_GB}" \
  --max-prealloc "${CUDA_KEEPER_MAX_PREALLOC_GB}" \
  --start-after-seconds "${CUDA_KEEPER_START_AFTER_SECONDS}" \
  --job-start-epoch-seconds "${JOB_START_EPOCH}" \
  > "${OUTPUT_DIR}/cuda_keeper_${SLURM_JOB_ID}.log" 2>&1 &
KEEPER_PID=$!
"${TRAIN_PYTHON}" - "${OUTPUT_DIR}" "${KEEPER_PID}" "${JOB_START_EPOCH}" <<'PY'
import json, os, socket, sys
from pathlib import Path
out = Path(sys.argv[1])
record = {"job_id": os.environ["SLURM_JOB_ID"], "hostname": socket.gethostname(),
          "keeper_pid": int(sys.argv[2]), "job_start_epoch": int(sys.argv[3]),
          "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
          "keeper_script": os.environ["CUDA_KEEPER_SCRIPT"],
          "start_after_seconds": 7200, "reserve_gib": 8.0, "max_prealloc_gib": .25}
(out / "allocation.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
print(json.dumps(record), flush=True)
PY
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv > "${OUTPUT_DIR}/gpu_inventory.csv"
printf 'ALLOCATION_READY job=%s keeper_pid=%s output=%s\n' "${SLURM_JOB_ID}" "${KEEPER_PID}" "${OUTPUT_DIR}"
WARNED=0
while sleep 60; do
  if ! kill -0 "${KEEPER_PID}" 2>/dev/null && [[ "${WARNED}" -eq 0 ]]; then
    printf 'WARNING: keeper exited; allocation remains available; inspect %s\n' "${OUTPUT_DIR}"
    WARNED=1
  fi
done
