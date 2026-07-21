#!/usr/bin/env bash
#SBATCH --job-name=egoqa-v3-greedy
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/grpo-v3-greedy-%j.out
#SBATCH --error=logs/grpo-v3-greedy-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
INFERENCE_ENV="${INFERENCE_ENV:-/scratch/${USER}/envs/egoqa-grpo}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
REVIEW_MODEL="${REVIEW_MODEL:-/scratch/${USER}/models/Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-${GRPO_V3_ROOT}/gate3_v2_data}"
EVAL_DATASET="${EVAL_DATASET:-${DATA_DIR}/gate3_v2_eval_native_video.jsonl}"
ADAPTER_LABEL="${ADAPTER_LABEL:?请设置 ADAPTER_LABEL=gate2、gate3_old 或 gate3_v2}"
ADAPTER_DIR="${ADAPTER_DIR:?请设置待评估 LoRA 的 ADAPTER_DIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/greedy_${ADAPTER_LABEL}_${SLURM_JOB_ID}}"
REVIEW_BASE_URL="${REVIEW_BASE_URL:-http://127.0.0.1:8001/v1}"
REVIEW_MAX_NUM_SEQS="${REVIEW_MAX_NUM_SEQS:-4}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"

[[ "${ADAPTER_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ADAPTER_LABEL 含非法字符" >&2; exit 2; }

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
export TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export FPS="${FPS:-1}" FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}" FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"
export VIDEO_MAX_PIXELS

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT_DIR}" "${HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_DIR}"

PYTHON="${TRAIN_ENV}/bin/python"
VLLM="${INFERENCE_ENV}/bin/vllm"
for required in "${PYTHON}" "${VLLM}" "${POLICY_MODEL}" "${REVIEW_MODEL}" "${ADAPTER_DIR}" "${EVAL_DATASET}"; do
  [[ -e "${required}" ]] || { echo "缺少必需路径: ${required}" >&2; exit 2; }
done
[[ -s "${EVAL_DATASET}" ]] || { echo "固定评估集为空" >&2; exit 2; }
"${PYTHON}" -c 'import importlib.metadata as m; assert m.version("ms-swift") == "4.2.2"'
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"
"${PYTHON}" -m training.grpo_v3_preflight --dataset "${EVAL_DATASET}" --output "${OUTPUT_DIR}/cpu_preflight.json"

GPU_MONITOR_PID=""
REVIEW_PID=""
cleanup() {
  [[ -z "${REVIEW_PID}" ]] || kill "${REVIEW_PID}" 2>/dev/null || true
  [[ -z "${GPU_MONITOR_PID}" ]] || kill "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu --format=csv -l 5 \
  > "${OUTPUT_DIR}/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
REVIEW_LOG="${PROJECT_ROOT}/logs/grpo-v3-greedy-review-${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 "${VLLM}" serve "${REVIEW_MODEL}" --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs "${REVIEW_MAX_NUM_SEQS}" \
  --gpu-memory-utilization 0.80 --gdn-prefill-backend triton \
  --allowed-local-media-path "/scratch/${USER}" > "${REVIEW_LOG}" 2>&1 &
REVIEW_PID=$!
for _ in $(seq 1 240); do
  curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null && break
  kill -0 "${REVIEW_PID}" 2>/dev/null || exit 3
  sleep 10
done
curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m training.grpo_v3_greedy_eval \
  --dataset "${EVAL_DATASET}" --model-path "${POLICY_MODEL}" \
  --adapter-dir "${ADAPTER_DIR}" --adapter-label "${ADAPTER_LABEL}" \
  --review-model "${REVIEW_MODEL}" --review-base-url "${REVIEW_BASE_URL}" \
  --max-image-pixels "${VIDEO_MAX_PIXELS}" --output "${OUTPUT_DIR}/greedy_results.jsonl"

"${PYTHON}" - "${OUTPUT_DIR}" "${REVIEW_LOG}" "${EVAL_DATASET}" "${ADAPTER_DIR}" <<'PY'
import hashlib, json, sys
from pathlib import Path

output = Path(sys.argv[1])
result = output / "greedy_results.jsonl"
rows = [json.loads(line) for line in result.read_text(encoding="utf-8").splitlines() if line.strip()]
assert len(rows) == 8, f"expected 8 eval rows, got {len(rows)}"
assert all(row["decode_config"] == {"mode": "greedy", "do_sample": False} for row in rows)
manifest = {
    "status": "passed",
    "row_count": len(rows),
    "adapter_label": rows[0]["adapter_label"],
    "dataset": str(Path(sys.argv[3]).resolve()),
    "dataset_sha256": hashlib.sha256(Path(sys.argv[3]).read_bytes()).hexdigest(),
    "adapter_dir": str(Path(sys.argv[4]).resolve()),
    "reviewer_log": sys.argv[2],
}
(output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "greedy_eval_status=passed adapter_label=${ADAPTER_LABEL} output=${OUTPUT_DIR}"
