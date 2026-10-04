#!/usr/bin/env bash
# 剩余真实编码检查只分配CPU，绝不申请GPU。
set -euo pipefail
: "${PROJECT_ROOT:?缺少 PROJECT_ROOT}"
: "${TRAIN_PYTHON:?缺少 TRAIN_PYTHON}"
: "${WORKFLOW_CONFIG:?缺少 WORKFLOW_CONFIG}"
: "${REUSE_PREFLIGHT:?缺少 REUSE_PREFLIGHT}"
: "${SLURM_JOB_ID:?必须在Slurm中运行}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH="${PROJECT_ROOT}/multi-user"
export PYTHONDONTWRITEBYTECODE=1
export VLLM_NO_USAGE_STATS=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
PREFLIGHT_OUT="${PROJECT_ROOT}/preflight/trainval60_cpu_${SLURM_JOB_ID}"
JOB_SCRATCH_ROOT="${PREFLIGHT_OUT}"
for variable in HOME XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE MODELSCOPE_CACHE TORCH_HOME \
  TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT CUDA_CACHE_PATH \
  FLASHINFER_WORKSPACE_BASE TMPDIR TMP TEMP PIP_CACHE_DIR; do
  export "${variable}=${JOB_SCRATCH_ROOT}/controller/${variable,,}"
done
mkdir -p "${PREFLIGHT_OUT}"
cd "${PROJECT_ROOT}/multi-user"
"${TRAIN_PYTHON}" -m training.torch_storage_preflight --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${PREFLIGHT_OUT}/storage_preflight.json"
"${TRAIN_PYTHON}" -u -m training.grpo_v3.six_user_binary.zero_gpu_runner \
  --config "${WORKFLOW_CONFIG}" --output "${PREFLIGHT_OUT}" --reuse-passed-from "${REUSE_PREFLIGHT}"
