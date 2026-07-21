#!/usr/bin/env bash
#SBATCH --job-name=egoqa-grpo-8b
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=logs/grpo-v2-lora-8b-%j.out
#SBATCH --error=logs/grpo-v2-lora-8b-%j.err

set -euo pipefail

SCRIPT_REVISION="2026-07-14-multimodal-batch-preflight-v5"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
MODEL_PATH="${MODEL_PATH:-/scratch/${USER}/models/Qwen3-VL-8B-Instruct}"
REVIEW_MODEL_ID="${REVIEW_MODEL_ID:-${MODEL_PATH}}"
INFERENCE_ENV="${INFERENCE_ENV:-/scratch/${USER}/envs/egoqa-grpo}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-grpo-train}"
JOB_HOME="${JOB_HOME:-/scratch/${USER}/job_home}"
REVIEW_BASE_URL="${REVIEW_BASE_URL:-http://127.0.0.1:8001/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo_v2_lora_8b_${SLURM_JOB_ID}}"
MAX_PROMPTS="${MAX_PROMPTS:-1}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_STEPS="${MAX_STEPS:-1}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
QUESTION_TYPE="${QUESTION_TYPE:-commonality}"
POLICY_MEDIA_MODE="${POLICY_MEDIA_MODE:-sampled_frames}"
MAX_POLICY_FRAMES_PER_CLIP="${MAX_POLICY_FRAMES_PER_CLIP:-4}"
VLLM_STARTUP_TIMEOUT_SECONDS="${VLLM_STARTUP_TIMEOUT_SECONDS:-2400}"
REVIEW_VLLM_MAX_MODEL_LEN="${REVIEW_VLLM_MAX_MODEL_LEN:-32768}"
REVIEW_VLLM_MAX_NUM_SEQS="${REVIEW_VLLM_MAX_NUM_SEQS:-4}"
REVIEW_VLLM_GPU_MEMORY_UTILIZATION="${REVIEW_VLLM_GPU_MEMORY_UTILIZATION:-0.80}"
ALLOWED_MEDIA_ROOT="${ALLOWED_MEDIA_ROOT:-/scratch/${USER}}"

export HOME="${JOB_HOME}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/${USER}/triton_cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/${USER}/torchinductor_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/${USER}/vllm_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/scratch/${USER}/cuda_cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-/scratch/${USER}/flashinfer_cache}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

cd "${PROJECT_ROOT}"
mkdir -p logs outputs "${OUTPUT_DIR}" "${HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_DIR}"
echo "${OUTPUT_DIR}" > "${PROJECT_ROOT}/outputs/latest_grpo_v2_lora_8b_output.txt"

if [[ -z "${EVIDENCE_JSONL:-}" ]]; then
  LATEST_POINTER="${PROJECT_ROOT}/outputs/latest_grpo_evidence.txt"
  if [[ ! -s "${LATEST_POINTER}" ]]; then
    echo "缺少 ${LATEST_POINTER}，请通过 EVIDENCE_JSONL 指定 evidence JSONL" >&2
    exit 2
  fi
  EVIDENCE_JSONL="$(head -n 1 "${LATEST_POINTER}")"
fi
if [[ "${EVIDENCE_JSONL}" != /* ]]; then
  EVIDENCE_JSONL="${PROJECT_ROOT}/${EVIDENCE_JSONL}"
fi

for required in "${MODEL_PATH}" "${EVIDENCE_JSONL}" "${INFERENCE_ENV}/bin/vllm" "${TRAIN_ENV}/bin/python"; do
  if [[ ! -e "${required}" ]]; then
    echo "缺少必需路径: ${required}" >&2
    exit 2
  fi
done

echo "job_id=${SLURM_JOB_ID}"
echo "script_revision=${SCRIPT_REVISION}"
echo "host=$(hostname)"
echo "model_path=${MODEL_PATH}"
echo "evidence_jsonl=${EVIDENCE_JSONL}"
echo "output_dir=${OUTPUT_DIR}"
echo "max_prompts=${MAX_PROMPTS} num_generations=${NUM_GENERATIONS} max_steps=${MAX_STEPS}"
echo "policy_media_mode=${POLICY_MEDIA_MODE} max_policy_frames_per_clip=${MAX_POLICY_FRAMES_PER_CLIP}"
echo "vllm_use_flashinfer_sampler=${VLLM_USE_FLASHINFER_SAMPLER}"
nvidia-smi

# 无 GPU 的登录节点也可先执行这些测试；作业中再复核上传完整性。
"${TRAIN_ENV}/bin/python" -m unittest tests.training.test_grpo_v2_lora -v

REVIEW_LOG="${PROJECT_ROOT}/logs/grpo-v2-review-${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 "${INFERENCE_ENV}/bin/vllm" serve "${REVIEW_MODEL_ID}" \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len "${REVIEW_VLLM_MAX_MODEL_LEN}" \
  --max-num-seqs "${REVIEW_VLLM_MAX_NUM_SEQS}" \
  --gpu-memory-utilization "${REVIEW_VLLM_GPU_MEMORY_UTILIZATION}" \
  --gdn-prefill-backend triton \
  --allowed-local-media-path "${ALLOWED_MEDIA_ROOT}" \
  > "${REVIEW_LOG}" 2>&1 &
REVIEW_PID=$!

cleanup() {
  kill "${REVIEW_PID}" 2>/dev/null || true
}
trap cleanup EXIT

START_SECONDS="${SECONDS}"
DEADLINE=$((SECONDS + VLLM_STARTUP_TIMEOUT_SECONDS))
while (( SECONDS < DEADLINE )); do
  if curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null; then
    echo "reviewer_ready waited_seconds=$((SECONDS - START_SECONDS))"
    break
  fi
  if ! kill -0 "${REVIEW_PID}" 2>/dev/null; then
    echo "reviewer 在 ready 前退出" >&2
    tail -200 "${REVIEW_LOG}" >&2
    exit 3
  fi
  sleep 10
done
if ! curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null; then
  echo "reviewer 启动超时" >&2
  tail -200 "${REVIEW_LOG}" >&2
  exit 3
fi

CUDA_VISIBLE_DEVICES=0 "${TRAIN_ENV}/bin/python" -m training.grpo_v2_lora \
  --evidence "${EVIDENCE_JSONL}" \
  --model-path "${MODEL_PATH}" \
  --review-model-id "${REVIEW_MODEL_ID}" \
  --review-base-url "${REVIEW_BASE_URL}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-prompts "${MAX_PROMPTS}" \
  --num-generations "${NUM_GENERATIONS}" \
  --max-steps "${MAX_STEPS}" \
  --max-completion-length "${MAX_COMPLETION_LENGTH}" \
  --question-type "${QUESTION_TYPE}" \
  --policy-media-mode "${POLICY_MEDIA_MODE}" \
  --max-policy-frames-per-clip "${MAX_POLICY_FRAMES_PER_CLIP}"

for artifact in \
  "${OUTPUT_DIR}/adapter/adapter_config.json" \
  "${OUTPUT_DIR}/adapter/adapter_model.safetensors" \
  "${OUTPUT_DIR}/reward_trace.jsonl" \
  "${OUTPUT_DIR}/train_metrics.json" \
  "${OUTPUT_DIR}/trainer_state.json" \
  "${OUTPUT_DIR}/run_manifest.json"; do
  if [[ ! -s "${artifact}" ]]; then
    echo "训练结束但缺少产物: ${artifact}" >&2
    exit 4
  fi
done

echo "GRPO_V2_LORA_OUTPUT=${OUTPUT_DIR}"
