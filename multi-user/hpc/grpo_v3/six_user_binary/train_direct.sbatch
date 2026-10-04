#!/usr/bin/env bash
# 分配资源后直接执行训练；退出码交给 Slurm，不依赖登录节点接入进程。
set -euo pipefail
: "${PROJECT_ROOT:?缺少 PROJECT_ROOT}"
: "${TRAIN_PYTHON:?缺少 TRAIN_PYTHON}"
: "${RUN_CONFIG:?缺少 RUN_CONFIG}"
: "${SLURM_JOB_ID:?必须在 Slurm 中运行}"
test -s "${RUN_CONFIG}"
test -x "${TRAIN_PYTHON}"
JOB_SCRATCH_ROOT="${PROJECT_ROOT}/job_scratch/batch_${SLURM_JOB_ID}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/direct_${SLURM_JOB_ID}"
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
printf 'DIRECT_TRAINING_START job=%s config=%s\n' "${SLURM_JOB_ID}" "${RUN_CONFIG}"
exec "${TRAIN_PYTHON}" -u -m training.grpo_v3.six_user_binary.resume --config "${RUN_CONFIG}"
