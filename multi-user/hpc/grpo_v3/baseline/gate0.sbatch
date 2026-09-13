#!/usr/bin/env bash
#SBATCH --job-name=egoqa-v3-g0
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/grpo-v3-gate0-%j.out
#SBATCH --error=logs/grpo-v3-gate0-%j.err

set -euo pipefail

REQUIRED_FRAMEWORK="ms-swift==4.2.2"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/gate0_${SLURM_JOB_ID}}"
EVIDENCE_JSONL="${EVIDENCE_JSONL:-${GRPO_V3_ROOT}/selected_packets_pruned.jsonl}"
QUESTION_TYPE="${QUESTION_TYPE:-commonality}"
GENERATION_MODE="${GENERATION_MODE:-baseline}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
POLICY_INPUT="native_video"

JOB_HOME="${JOB_HOME:-/scratch/${USER}/job_home}"
export HOME="${JOB_HOME}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/${USER}/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/${USER}/torchinductor_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/${USER}/vllm_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/scratch/${USER}/cuda_cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-/scratch/${USER}/flashinfer_cache}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FPS="${FPS:-1}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRPO_V3_ROOT}" "${OUTPUT_DIR}" "${HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_DIR}"

GPU_MONITOR_PID=""
cleanup() {
  if [[ -n "${GPU_MONITOR_PID}" ]]; then
    kill "${GPU_MONITOR_PID}" 2>/dev/null || true
    wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
nvidia-smi \
  --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv -l 5 > "${OUTPUT_DIR}/gpu_metrics.csv" &
GPU_MONITOR_PID=$!

PYTHON="${TRAIN_ENV}/bin/python"
SWIFT="${TRAIN_ENV}/bin/swift"
for required in "${PYTHON}" "${SWIFT}" "${POLICY_MODEL}"; do
  [[ -e "${required}" ]] || { echo "缺少必需路径: ${required}" >&2; exit 2; }
done
"${PYTHON}" -c 'import importlib.metadata as m; assert m.version("ms-swift") == "4.2.2"'
"${SWIFT}" rlhf --help > "${OUTPUT_DIR}/swift_rlhf_help.txt"
grep -- '--tuner_type' "${OUTPUT_DIR}/swift_rlhf_help.txt" >/dev/null
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"

[[ "${EVIDENCE_JSONL}" = /* ]] || EVIDENCE_JSONL="${PROJECT_ROOT}/${EVIDENCE_JSONL}"
[[ -s "${EVIDENCE_JSONL}" ]] || {
  echo "缺少或为空的 Gate 0 evidence JSONL: ${EVIDENCE_JSONL}" >&2
  echo "请先把 selected_packets_pruned.jsonl 上传到 ${GRPO_V3_ROOT}/，或显式传入现有 EVIDENCE_JSONL。" >&2
  exit 2
}

DATASET="${OUTPUT_DIR}/train_native_video.jsonl"
"${PYTHON}" -m unittest \
  tests.training.test_grpo_v3_contract \
  tests.training.test_grpo_v3_data \
  tests.training.test_grpo_v3_gate0_probe -v
"${PYTHON}" -m training.grpo_v3.shared.data \
  --evidence "${EVIDENCE_JSONL}" --output "${DATASET}" \
  --question-type "${QUESTION_TYPE}" --generation-mode "${GENERATION_MODE}" \
  --max-prompts "${MAX_PROMPTS}"
"${PYTHON}" -m training.grpo_v3.baseline.preflight \
  --dataset "${DATASET}" --output "${OUTPUT_DIR}/cpu_preflight.json"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m training.grpo_v3.baseline.gate0_probe \
  --model-path "${POLICY_MODEL}" --dataset "${DATASET}" \
  --output "${OUTPUT_DIR}/gate0_result.json"
"${PYTHON}" -m training.grpo_v3.baseline.summary \
  --output-dir "${OUTPUT_DIR}" --gate 0 --dataset "${DATASET}" \
  --policy-model "${POLICY_MODEL}" --job-id "${SLURM_JOB_ID}" \
  --slurm-stdout "${PROJECT_ROOT}/logs/grpo-v3-gate0-${SLURM_JOB_ID}.out" \
  --slurm-stderr "${PROJECT_ROOT}/logs/grpo-v3-gate0-${SLURM_JOB_ID}.err"

echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_gate0_output.txt"
echo "framework=${REQUIRED_FRAMEWORK} policy_input=${POLICY_INPUT} output=${OUTPUT_DIR}"
