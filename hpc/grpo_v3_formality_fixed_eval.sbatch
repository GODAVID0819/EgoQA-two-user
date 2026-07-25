#!/usr/bin/env bash
#SBATCH --job-name=egoqa-v3-form-eval
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/grpo-v3-formality-fixed-eval-%j.out
#SBATCH --error=logs/grpo-v3-formality-fixed-eval-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
INFERENCE_ENV="${INFERENCE_ENV:-/scratch/${USER}/envs/egoqa-grpo}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
REVIEW_MODEL="${REVIEW_MODEL:-/scratch/${USER}/models/Qwen3-VL-8B-Instruct}"
FORMALITY_PROBE_DIR="${FORMALITY_PROBE_DIR:-${GRPO_V3_ROOT}/formality_probe_14377903}"
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/formality_fixed_eval_${SLURM_JOB_ID}}"
REVIEW_BASE_URL="${REVIEW_BASE_URL:-http://127.0.0.1:8001/v1}"
EGOQA_FORMALITY_REWARD_REVISION="qa_formality_confidence_v1"
export EGOQA_FORMALITY_REWARD_REVISION

JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/formality_fixed_eval_runtime}"
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
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export FPS="${FPS:-1}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRPO_V3_ROOT}" "${OUTPUT_DIR}"
PYTHON="${TRAIN_ENV}/bin/python"
VLLM="${INFERENCE_ENV}/bin/vllm"
for required in "${PYTHON}" "${VLLM}" "${POLICY_MODEL}" "${REVIEW_MODEL}" \
  "${FORMALITY_PROBE_DIR}" "${FORMALITY_PROBE_DIR}/resolved_config.json"; do
  [[ -e "${required}" ]] || { echo "缺少必需路径: ${required}" >&2; exit 2; }
done

"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"

DATASET="$(${PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset"])' \
  "${FORMALITY_PROBE_DIR}/resolved_config.json")"
[[ -s "${DATASET}" ]] || { echo "Probe dataset 不存在或为空: ${DATASET}" >&2; exit 2; }
"${PYTHON}" -m training.grpo_v3_preflight \
  --dataset "${DATASET}" --output "${OUTPUT_DIR}/cpu_preflight.json"

GPU_MONITOR_PID=""
REVIEW_PID=""
cleanup() {
  [[ -z "${REVIEW_PID}" ]] || kill "${REVIEW_PID}" 2>/dev/null || true
  [[ -z "${GPU_MONITOR_PID}" ]] || kill "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv -l 5 > "${OUTPUT_DIR}/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
REVIEW_LOG="${PROJECT_ROOT}/logs/grpo-v3-formality-fixed-eval-review-${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 "${VLLM}" serve "${REVIEW_MODEL}" \
  --host 127.0.0.1 --port 8001 --dtype bfloat16 --max-model-len 32768 \
  --max-num-seqs 4 --gpu-memory-utilization 0.80 --gdn-prefill-backend triton \
  > "${REVIEW_LOG}" 2>&1 &
REVIEW_PID=$!
for _ in $(seq 1 240); do
  curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null && break
  kill -0 "${REVIEW_PID}" 2>/dev/null || exit 3
  sleep 10
done
curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null
curl --fail --silent --show-error \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${REVIEW_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":8,\"temperature\":0}" \
  "${REVIEW_BASE_URL}/chat/completions" > "${OUTPUT_DIR}/reviewer_text_probe.json"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m training.grpo_v3_formality_fixed_eval \
  --probe-dir "${FORMALITY_PROBE_DIR}" \
  --dataset "${DATASET}" \
  --model-path "${POLICY_MODEL}" \
  --review-model "${REVIEW_MODEL}" \
  --review-base-url "${REVIEW_BASE_URL}" \
  --temperature 0.7 \
  --max-image-pixels "${VIDEO_MAX_PIXELS}" \
  --output-dir "${OUTPUT_DIR}"

"${PYTHON}" - "${OUTPUT_DIR}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

output = Path(sys.argv[1])
rows = [
    json.loads(line)
    for line in (output / "fixed_eval_results.jsonl").read_text(encoding="utf-8").splitlines()
    if line.strip()
]
summary = json.loads((output / "fixed_eval_summary.json").read_text(encoding="utf-8"))
manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
storage = json.loads((output / "storage_preflight.json").read_text(encoding="utf-8"))
counts = dict(Counter(int(row["checkpoint_step"]) for row in rows))
assert len(rows) == 32
assert counts == {0: 16, 40: 16}
assert manifest["run_status"] == "passed"
assert summary["experiment_conclusion"] in {
    "improved", "not_improved", "inconclusive"
}
assert manifest["experiment_conclusion"] == summary["experiment_conclusion"]
assert storage["status"] == "passed"
print({
    "run_status": manifest["run_status"],
    "experiment_conclusion": summary["experiment_conclusion"],
    "row_count": len(rows),
    "counts": counts,
})
PY

echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_formality_fixed_eval_output.txt"
echo "formality_fixed_eval_status=passed output=${OUTPUT_DIR}"
