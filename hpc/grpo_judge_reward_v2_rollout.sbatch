#!/usr/bin/env bash
#SBATCH --job-name=egoqa-judge-reward-v2
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/grpo-judge-reward-v2-%j.out
#SBATCH --error=logs/grpo-judge-reward-v2-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
ENV_ROOT="${ENV_ROOT:-/scratch/${USER}/envs/egoqa-grpo}"
JOB_HOME="${JOB_HOME:-/scratch/${USER}/job_home}"
export HOME="${JOB_HOME}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/${USER}/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/${USER}/torchinductor_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/${USER}/vllm_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/scratch/${USER}/cuda_cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-/scratch/${USER}/flashinfer_cache}"
ALLOWED_MEDIA_ROOT="${ALLOWED_MEDIA_ROOT:-/scratch/${USER}}"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8000/v1}"
REVIEW_MODEL_ID="${REVIEW_MODEL_ID:-${MODEL_ID}}"
if [[ -z "${REVIEW_BASE_URL:-}" ]]; then
  if [[ "${REVIEW_MODEL_ID}" == "${MODEL_ID}" ]]; then
    REVIEW_BASE_URL="${BASE_URL}"
  else
    REVIEW_BASE_URL="http://127.0.0.1:8001/v1"
  fi
fi
LATEST_GRPO_EVIDENCE="${PROJECT_ROOT}/outputs/latest_grpo_evidence.txt"
if [[ -z "${EVIDENCE_JSONL:-}" ]]; then
  if [[ -s "${LATEST_GRPO_EVIDENCE}" ]]; then
    EVIDENCE_JSONL="$(cat "${LATEST_GRPO_EVIDENCE}")"
  else
    EVIDENCE_JSONL="${PROJECT_ROOT}/outputs/evidence_manifest_rebuilt.jsonl"
  fi
fi
TARGET_COUNT="${TARGET_COUNT:-5}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo_judge_reward_v2_${SLURM_JOB_ID}}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-128}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
REVIEW_VLLM_MAX_MODEL_LEN="${REVIEW_VLLM_MAX_MODEL_LEN:-16384}"
REVIEW_VLLM_MAX_NUM_SEQS="${REVIEW_VLLM_MAX_NUM_SEQS:-${VLLM_MAX_NUM_SEQS}}"
REVIEW_VLLM_GPU_MEMORY_UTILIZATION="${REVIEW_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
USE_SRUN_GPU_BINDING="${USE_SRUN_GPU_BINDING:-1}"

cd "${PROJECT_ROOT}"
mkdir -p logs outputs "${OUTPUT_DIR}" "${HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_DIR}"
source "${ENV_ROOT}/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "date_start=$(date --iso-8601=seconds)"
echo "project_root=${PROJECT_ROOT}"
echo "model_id=${MODEL_ID}"
echo "review_model_id=${REVIEW_MODEL_ID}"
echo "base_url=${BASE_URL}"
echo "review_base_url=${REVIEW_BASE_URL}"
echo "vllm_max_num_seqs=${VLLM_MAX_NUM_SEQS}"
echo "review_vllm_max_model_len=${REVIEW_VLLM_MAX_MODEL_LEN}"
echo "review_vllm_max_num_seqs=${REVIEW_VLLM_MAX_NUM_SEQS}"
echo "use_srun_gpu_binding=${USE_SRUN_GPU_BINDING}"
echo "slurm_job_gpus=${SLURM_JOB_GPUS:-}"
echo "slurm_step_gpus=${SLURM_STEP_GPUS:-}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
python --version
nvidia-smi
python - <<'PY' || true
import torch

print("torch_cuda_available=", torch.cuda.is_available())
count = torch.cuda.device_count()
print("torch_cuda_device_count=", count)
for i in range(count):
    name = torch.cuda.get_device_name(i)
    try:
        mem_info = torch.cuda.mem_get_info(i)
    except Exception as exc:
        mem_info = f"unavailable: {type(exc).__name__}: {exc}"
    print("torch_cuda_device", i, name, mem_info)
PY

REVIEW_ARGS=()
SERVER_PIDS=()

run_vllm_on_one_gpu() {
  local log_path="$1"
  shift
  if [[ "${USE_SRUN_GPU_BINDING}" == "1" ]]; then
    srun --exclusive --ntasks=1 --cpus-per-task=8 --gres=gpu:a100:1 "$@" > "${log_path}" 2>&1
  else
    "$@" > "${log_path}" 2>&1
  fi
}

run_vllm_on_one_gpu "logs/judge-reward-v2-vllm-${SLURM_JOB_ID}.log" \
  vllm serve "${MODEL_ID}" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len "${VLLM_MAX_MODEL_LEN}" \
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --allowed-local-media-path "${ALLOWED_MEDIA_ROOT}" &
POLICY_SERVER_PID=$!
SERVER_PIDS+=("${POLICY_SERVER_PID}")

if [[ "${REVIEW_MODEL_ID}" != "${MODEL_ID}" || "${REVIEW_BASE_URL}" != "${BASE_URL}" ]]; then
  if [[ "$(python -c 'import torch; print(torch.cuda.device_count())')" -lt 2 ]]; then
    echo "Separate review model requested, but fewer than 2 CUDA devices are visible." >&2
    exit 2
  fi
  run_vllm_on_one_gpu "logs/judge-reward-v2-review-vllm-${SLURM_JOB_ID}.log" \
    vllm serve "${REVIEW_MODEL_ID}" \
      --host 127.0.0.1 \
      --port 8001 \
      --dtype bfloat16 \
      --max-model-len "${REVIEW_VLLM_MAX_MODEL_LEN}" \
      --max-num-seqs "${REVIEW_VLLM_MAX_NUM_SEQS}" \
      --gpu-memory-utilization "${REVIEW_VLLM_GPU_MEMORY_UTILIZATION}" \
      --allowed-local-media-path "${ALLOWED_MEDIA_ROOT}" &
  REVIEW_SERVER_PID=$!
  SERVER_PIDS+=("${REVIEW_SERVER_PID}")
  REVIEW_ARGS+=(--review-model-id "${REVIEW_MODEL_ID}" --review-base-url "${REVIEW_BASE_URL}")
fi

cleanup() {
  kill "${SERVER_PIDS[@]}" 2>/dev/null || true
}
trap cleanup EXIT

wait_for_server() {
  local url="$1"
  local log_path="$2"
  local pid="$3"
  for _ in $(seq 1 120); do
    if curl --fail --silent "${url}/models" >/dev/null; then
      return 0
    fi
    if ! kill -0 "${pid}" 2>/dev/null; then
      tail -200 "${log_path}" >&2
      return 1
    fi
    sleep 10
  done
  tail -200 "${log_path}" >&2
  return 1
}

wait_for_server "${BASE_URL}" "logs/judge-reward-v2-vllm-${SLURM_JOB_ID}.log" "${POLICY_SERVER_PID}"
if [[ "${#REVIEW_ARGS[@]}" -gt 0 ]]; then
  wait_for_server "${REVIEW_BASE_URL}" "logs/judge-reward-v2-review-vllm-${SLURM_JOB_ID}.log" "${REVIEW_SERVER_PID}"
fi

curl --fail --silent "${BASE_URL}/models" >/dev/null
if [[ "${#REVIEW_ARGS[@]}" -gt 0 ]]; then
  curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null
fi

if [[ -s "${EVIDENCE_JSONL}" ]]; then
  echo "evidence_jsonl=${EVIDENCE_JSONL}"
else
  echo "missing evidence_jsonl=${EVIDENCE_JSONL}" >&2
  exit 2
fi

python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_JSONL}" \
  --output "${OUTPUT_DIR}/qa.jsonl" \
  --intermediate-output "${OUTPUT_DIR}/qa.intermediate.jsonl" \
  --prompts-output "${OUTPUT_DIR}/prompts.jsonl" \
  --rejected-output "${OUTPUT_DIR}/qa.rejected.jsonl" \
  --backend openai-compatible-local \
  --base-url "${BASE_URL}" \
  --model-id "${MODEL_ID}" \
  "${REVIEW_ARGS[@]}" \
  --target-count "${TARGET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --allow-openai-video-input

python -m grpo_judge_reward.cli score \
  --input "${OUTPUT_DIR}/qa.intermediate.jsonl" \
  --input-kind intermediate \
  --output-dir "${OUTPUT_DIR}/judge_reward_v2"

echo "${OUTPUT_DIR}" > "${PROJECT_ROOT}/outputs/latest_grpo_judge_reward_v2_output.txt"
echo "JUDGE_REWARD_V2_OUTPUT=${OUTPUT_DIR}"
