#!/usr/bin/env bash
#SBATCH --job-name=egoqa-score-rm-probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/score-only-reviewer-probe-%j.out
#SBATCH --error=logs/score-only-reviewer-probe-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/score-only-reward-model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/grpo_score_only}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/xl6775/envs/egoqa-ffmpeg-runtime}"
REVIEWER_MODEL="${REVIEWER_MODEL:-/scratch/xl6775/models/Qwen3-VL-8B-Instruct}"
REVIEWER_CHECKPOINT="${REVIEWER_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/ckpt1}"
DATASET="${DATASET:?set DATASET to the native dual-video GRPO JSONL}"
PORT="${PORT:-8766}"
REVIEWER_BASE_URL="http://127.0.0.1:${PORT}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/reviewer_probe_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/score_only_reviewer_probe_${SLURM_JOB_ID}}"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"
export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_BASE}"
export VLLM_NO_USAGE_STATS=1
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
export FPS="${FPS:-1}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-32}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${TMPDIR}" \
  "${VLLM_CACHE_ROOT}" "${FLASHINFER_WORKSPACE_BASE}"
PYTHON="${TRAIN_ENV}/bin/python"

for required in \
  "${PYTHON}" "${FFMPEG_ENV}/bin/ffmpeg" "${REVIEWER_MODEL}/config.json" "${DATASET}" \
  "${REVIEWER_CHECKPOINT}/reviewer_v1_config.json" \
  "${REVIEWER_CHECKPOINT}/lora_adapter.pt" \
  "${REVIEWER_CHECKPOINT}/ordinal_heads.pt" \
  "${REVIEWER_CHECKPOINT}/processor"
do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 2; }
done

"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
  > "${OUTPUT_DIR}/gpu_environment.csv"
"${FFMPEG_ENV}/bin/ffmpeg" -version > "${OUTPUT_DIR}/ffmpeg_environment.txt"

SERVICE_PID=""
cleanup() {
  [[ -z "${SERVICE_PID}" ]] || kill "${SERVICE_PID}" 2>/dev/null || true
}
trap cleanup EXIT

SERVICE_LOG="${OUTPUT_DIR}/reviewer_service.log"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m \
  training.grpo_v3.experiments.human_preference_reviewer.v1.service \
  --checkpoint "${REVIEWER_CHECKPOINT}" --model "${REVIEWER_MODEL}" \
  --device cuda:0 --torch-dtype bfloat16 --port "${PORT}" \
  > "${SERVICE_LOG}" 2>&1 &
SERVICE_PID=$!

for _ in $(seq 1 180); do
  if curl --fail --silent "${REVIEWER_BASE_URL}/health" > "${OUTPUT_DIR}/reviewer_health.json"; then
    break
  fi
  kill -0 "${SERVICE_PID}" 2>/dev/null || {
    tail -n 200 "${SERVICE_LOG}" >&2 || true
    exit 3
  }
  sleep 5
done

"${PYTHON}" - "${DATASET}" "${REVIEWER_BASE_URL}" "${OUTPUT_DIR}/reviewer_probe_result.json" <<'PY'
import json
import sys
from pathlib import Path

from training.grpo_v3.experiments.human_preference_reviewer.v1.grpo_reward import (
    prepare_completion_for_review,
)
from training.grpo_v3.experiments.human_preference_reviewer.v1.service import (
    ScoreOnlyReviewerClient,
)

dataset, base_url, output = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
row = next(
    (json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()),
    None,
)
assert isinstance(row, dict) and row.get("evidence_id") and row.get("packet_json")
packet = json.loads(row["packet_json"])
completion = json.dumps({
    "question": "Which statement best describes the relation between the two synchronized views?",
    "options": [
        "They show the same activity from different views.",
        "Only the first view shows an activity.",
        "Only the second view shows an activity.",
        "They show unrelated activities.",
        "The relation cannot be determined.",
    ],
    "correct": "E",
    "answer": "The relation cannot be determined.",
})
prepared = prepare_completion_for_review(
    completion, packet, evidence_id=str(row["evidence_id"]), candidate_index=0
)
assert prepared["status"] == "eligible"
client = ScoreOnlyReviewerClient(base_url)
health = client.health()
score = client.score(prepared["candidate"])
assert score["review_key"] == prepared["candidate"]["review_key"]
assert score["evidence_id"] == row["evidence_id"]
assert 0.0 <= float(score["reward"]) <= 1.0
result = {
    "status": "passed",
    "evidence_id": row["evidence_id"],
    "video_paths": [prepared["candidate"]["video_a_path"], prepared["candidate"]["video_b_path"]],
    "health": health,
    "score": score,
}
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": result["status"], "evidence_id": result["evidence_id"]}, ensure_ascii=False))
PY

echo "score_only_reviewer_probe_status=passed output=${OUTPUT_DIR}"
