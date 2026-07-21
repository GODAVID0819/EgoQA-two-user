#!/usr/bin/env bash
#SBATCH --job-name=egoqa-v3-g2
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=logs/grpo-v3-gate2-%j.out
#SBATCH --error=logs/grpo-v3-gate2-%j.err

set -euo pipefail

REQUIRED_FRAMEWORK="ms-swift==4.2.2"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
INFERENCE_ENV="${INFERENCE_ENV:-/scratch/${USER}/envs/egoqa-grpo}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
REVIEW_MODEL="${REVIEW_MODEL:-/scratch/${USER}/models/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/gate2_${SLURM_JOB_ID}}"
REVIEW_BASE_URL="${REVIEW_BASE_URL:-http://127.0.0.1:8001/v1}"
REVIEW_MAX_NUM_SEQS="${REVIEW_MAX_NUM_SEQS:-4}"
POLICY_INPUT="native_video"
EGOQA_FORMAT_REWARD_REVISION="json_three_tier_v1"
export EGOQA_FORMAT_REWARD_REVISION

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
export VLLM_USE_FLASHINFER_SAMPLER=0
export FPS="${FPS:-1}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export EGOQA_GRPO_V3_REWARD_TRACE="${OUTPUT_DIR}/reward_trace.jsonl"
export EGOQA_POLICY_MODEL="${POLICY_MODEL}"
export EGOQA_REVIEW_MODEL="${REVIEW_MODEL}"
export EGOQA_REVIEW_BASE_URL="${REVIEW_BASE_URL}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRPO_V3_ROOT}" "${OUTPUT_DIR}" "${HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_DIR}"
GPU_MONITOR_PID=""
REVIEW_PID=""
cleanup() {
  if [[ -n "${REVIEW_PID}" ]]; then
    kill "${REVIEW_PID}" 2>/dev/null || true
    wait "${REVIEW_PID}" 2>/dev/null || true
  fi
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
VLLM="${INFERENCE_ENV}/bin/vllm"
for required in "${PYTHON}" "${SWIFT}" "${VLLM}" "${POLICY_MODEL}" "${REVIEW_MODEL}"; do
  [[ -e "${required}" ]] || { echo "缺少必需路径: ${required}" >&2; exit 2; }
done
"${PYTHON}" -c 'import importlib.metadata as m; assert m.version("ms-swift") == "4.2.2"'
"${SWIFT}" rlhf --help > "${OUTPUT_DIR}/swift_rlhf_help.txt"
grep -- '--tuner_type' "${OUTPUT_DIR}/swift_rlhf_help.txt" >/dev/null
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"

if [[ -z "${GATE1_DIR:-}" ]]; then
  POINTER="${GRPO_V3_ROOT}/latest_gate1_output.txt"
  [[ -s "${POINTER}" ]] || { echo "缺少 Gate 1 pointer" >&2; exit 2; }
  GATE1_DIR="$(head -n 1 "${POINTER}")"
fi
GATE0_DIR="${GATE0_DIR:-$(head -n 1 "${GRPO_V3_ROOT}/latest_gate0_output.txt")}"
DATASET="${GATE0_DIR}/train_native_video.jsonl"
"${PYTHON}" -c 'import hashlib,json,sys; from pathlib import Path; p=json.load(open(sys.argv[1])); assert p["status"] == "passed"; m=json.load(open(sys.argv[2])); assert Path(m["parent_run"]).resolve() == Path(sys.argv[3]).resolve(); assert hashlib.sha256(Path(sys.argv[4]).read_bytes()).hexdigest() == m["dataset_sha256"]' \
  "${GATE1_DIR}/gate1_result.json" "${GATE1_DIR}/run_manifest.json" "${GATE0_DIR}" "${DATASET}"
"${PYTHON}" -m training.grpo_v3_preflight \
  --dataset "${DATASET}" \
  --check-repo-reward \
  --output "${OUTPUT_DIR}/cpu_preflight.json"
: > "${EGOQA_GRPO_V3_REWARD_TRACE}"

REVIEW_LOG="${PROJECT_ROOT}/logs/grpo-v3-review-${SLURM_JOB_ID}.log"
# vLLM serve：GPU1 上的冻结 Qwen3-VL-8B reviewer。
CUDA_VISIBLE_DEVICES=1 "${VLLM}" serve "${REVIEW_MODEL}" \
  --host 127.0.0.1 --port 8001 --dtype bfloat16 \
  --max-model-len 32768 --max-num-seqs "${REVIEW_MAX_NUM_SEQS}" --gpu-memory-utilization 0.80 \
  --gdn-prefill-backend triton \
  --allowed-local-media-path "/scratch/${USER}" > "${REVIEW_LOG}" 2>&1 &
REVIEW_PID=$!
for _ in $(seq 1 240); do
  curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null && break
  kill -0 "${REVIEW_PID}" 2>/dev/null || { tail -200 "${REVIEW_LOG}" >&2; exit 3; }
  sleep 10
done
curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null || { tail -200 "${REVIEW_LOG}" >&2; exit 3; }

CUDA_VISIBLE_DEVICES=0 "${SWIFT}" rlhf \
  --rlhf_type grpo \
  --model "${POLICY_MODEL}" \
  --dataset "${DATASET}" \
  --external_plugins "${PROJECT_ROOT}/training/grpo_v3_reward_plugin.py" \
  --reward_funcs egoqa_repo_native_judge \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --freeze_vit true \
  --freeze_aligner true \
  --target_modules q_proj v_proj \
  --lora_rank 8 \
  --lora_alpha 16 \
  --use_vllm false \
  --num_generations 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --max_steps 1 \
  --max_length 32768 \
  --max_completion_length 1024 \
  --learning_rate 1e-5 \
  --beta 0.0 \
  --temperature 0.7 \
  --save_steps 1 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --log_completions true \
  --split_dataset_ratio 0 \
  --dataset_num_proc 1 \
  --dataloader_num_workers 0 \
  --report_to none \
  --output_dir "${OUTPUT_DIR}/swift"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m training.grpo_v3_adapter_reload \
  --model-path "${POLICY_MODEL}" --search-root "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/adapter_reload.json"
"${PYTHON}" -m training.grpo_v3_gate_validate --output-dir "${OUTPUT_DIR}" --gate 2
"${PYTHON}" -m training.grpo_v3_summary \
  --output-dir "${OUTPUT_DIR}" --gate 2 --dataset "${DATASET}" \
  --policy-model "${POLICY_MODEL}" --reviewer-model "${REVIEW_MODEL}" \
  --job-id "${SLURM_JOB_ID}" --parent-run "${GATE1_DIR}" \
  --slurm-stdout "${PROJECT_ROOT}/logs/grpo-v3-gate2-${SLURM_JOB_ID}.out" \
  --slurm-stderr "${PROJECT_ROOT}/logs/grpo-v3-gate2-${SLURM_JOB_ID}.err" \
  --reviewer-log "${REVIEW_LOG}"
echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_gate2_output.txt"
echo "framework=${REQUIRED_FRAMEWORK} policy_input=${POLICY_INPUT} reviewer=Qwen3-VL-8B-Instruct format_reward_revision=${EGOQA_FORMAT_REWARD_REVISION} output=${OUTPUT_DIR}"
