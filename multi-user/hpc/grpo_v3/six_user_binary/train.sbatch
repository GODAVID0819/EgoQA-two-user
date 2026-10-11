#!/usr/bin/env bash
# 资源由 submit.py 从已核验配置传入，禁止写死节点。
set -euo pipefail
: "${PROJECT_ROOT:?缺少 PROJECT_ROOT}"
: "${TRAIN_PYTHON:?缺少 TRAIN_PYTHON}"
: "${RUN_CONFIG:?缺少 RUN_CONFIG}"
export EGOQA_JOB_START_EPOCH="$(date +%s)"
export PYTHONPATH="${PROJECT_ROOT}/multi-user"
export PYTHONDONTWRITEBYTECODE=1
DRIVER_PID=""
cleanup() {
  if [[ -n "${DRIVER_PID}" ]]; then
    kill -TERM "${DRIVER_PID}" 2>/dev/null || true
    wait "${DRIVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT TERM INT
cd "${PROJECT_ROOT}/multi-user"
"${TRAIN_PYTHON}" -m training.grpo_v3.six_user_binary.launch --config "${RUN_CONFIG}" &
DRIVER_PID=$!
wait "${DRIVER_PID}"
DRIVER_PID=""
