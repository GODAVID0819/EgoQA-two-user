#!/usr/bin/env bash
#SBATCH --job-name=egoqa-score-rm-grpo-s1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=h100
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/score-only-grpo-smoke1-%j.out
#SBATCH --error=logs/score-only-grpo-smoke1-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/score-only-reward-model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/grpo_score_only}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/xl6775/envs/egoqa-ffmpeg-runtime}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/xl6775/models/Qwen3-VL-8B-Instruct}"
REVIEWER_MODEL="${REVIEWER_MODEL:-/scratch/xl6775/models/Qwen3-VL-8B-Instruct}"
REVIEWER_CHECKPOINT="${REVIEWER_CHECKPOINT:-${PROJECT_ROOT}/checkpoints/ckpt1}"
DATASET="${DATASET:?set DATASET to the native dual-video GRPO JSONL}"
MAX_STEPS="${MAX_STEPS:-1}"
EXPECTED_GROUPS="${EXPECTED_GROUPS:-${MAX_STEPS}}"
PORT="${PORT:-8766}"
REVIEWER_BASE_URL="http://127.0.0.1:${PORT}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/smoke1_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/score_only_grpo_${SLURM_JOB_ID}}"

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
export EGOQA_SCORE_ONLY_REVIEWER_BASE_URL="${REVIEWER_BASE_URL}"
export EGOQA_SCORE_ONLY_REVIEWER_TIMEOUT_SECONDS="${EGOQA_SCORE_ONLY_REVIEWER_TIMEOUT_SECONDS:-300}"
export EGOQA_GRPO_V3_REWARD_TRACE="${OUTPUT_DIR}/reward_trace.jsonl"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${TMPDIR}" \
  "${VLLM_CACHE_ROOT}" "${FLASHINFER_WORKSPACE_BASE}"
PYTHON="${TRAIN_ENV}/bin/python"
SWIFT="${TRAIN_ENV}/bin/swift"

for required in \
  "${PYTHON}" "${SWIFT}" "${FFMPEG_ENV}/bin/ffmpeg" \
  "${POLICY_MODEL}/config.json" "${REVIEWER_MODEL}/config.json" "${DATASET}" \
  "${REVIEWER_CHECKPOINT}/reviewer_v1_config.json" \
  "${REVIEWER_CHECKPOINT}/lora_adapter.pt" \
  "${REVIEWER_CHECKPOINT}/ordinal_heads.pt" \
  "${REVIEWER_CHECKPOINT}/processor"
do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 2; }
done

"${PYTHON}" - "${REVIEWER_CHECKPOINT}" "${DATASET}" <<'PY'
import json, sys, torch
from pathlib import Path
from training.grpo_v3.experiments.human_preference_reviewer.v1.checkpoint import load_checkpoint_contract

checkpoint, dataset = map(Path, sys.argv[1:])
contract = load_checkpoint_contract(checkpoint)
assert contract["contract_version"] == "score_only_ordinal_reviewer_v1"
assert contract["active_heads"] == ["evidence_quality", "answerability", "qa_formality"]
lora = torch.load(checkpoint / "lora_adapter.pt", map_location="cpu", weights_only=True)
heads = torch.load(checkpoint / "ordinal_heads.pt", map_location="cpu", weights_only=True)
assert set(heads) == {"evidence_head", "answerability_head", "formality_head"}
assert set(lora) == set(contract["lora_parameter_names"])
first = next((json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()), None)
assert isinstance(first, dict) and "evidence_id" in first and "packet_json" in first
print(f"checkpoint_preflight=passed lora_tensors={len(lora)} head_modules={len(heads)}")
PY

"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"
"${PYTHON}" -c 'import importlib.metadata as m; assert m.version("ms-swift") == "4.2.2"'
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
  > "${OUTPUT_DIR}/gpu_environment.csv"
"${FFMPEG_ENV}/bin/ffmpeg" -version > "${OUTPUT_DIR}/ffmpeg_environment.txt"

SERVICE_PID=""
GPU_MONITOR_PID=""
cleanup() {
  [[ -z "${SERVICE_PID}" ]] || kill "${SERVICE_PID}" 2>/dev/null || true
  [[ -z "${GPU_MONITOR_PID}" ]] || kill "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv -l 5 > "${OUTPUT_DIR}/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
SERVICE_LOG="${OUTPUT_DIR}/reviewer_service.log"
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" -m \
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
curl --fail --silent "${REVIEWER_BASE_URL}/health" > "${OUTPUT_DIR}/reviewer_health.json"
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="ok"; assert r["trainable_parameter_count"]==0; assert all(r["checks"].values())' \
  "${OUTPUT_DIR}/reviewer_health.json"

: > "${EGOQA_GRPO_V3_REWARD_TRACE}"
CUDA_VISIBLE_DEVICES=0 "${SWIFT}" rlhf \
  --rlhf_type grpo \
  --model "${POLICY_MODEL}" \
  --dataset "${DATASET}" \
  --external_plugins "${PROJECT_ROOT}/training/grpo_v3/runtime/score_only_reward_plugin.py" \
  --reward_funcs egoqa_score_only_ordinal_v1 \
  --tuner_type lora --torch_dtype bfloat16 \
  --freeze_vit true --freeze_aligner true \
  --target_modules q_proj v_proj --lora_rank 8 --lora_alpha 16 \
  --use_vllm false --num_generations 4 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
  --gradient_checkpointing true --max_steps "${MAX_STEPS}" \
  --seed 42 --data_seed 42 --dataset_shuffle false \
  --max_length 32768 --max_completion_length 1024 \
  --learning_rate 1e-5 --lr_scheduler_type constant --beta 0.0 \
  --temperature 0.7 --top_p 1.0 \
  --save_steps "${MAX_STEPS}" --save_total_limit 1 --logging_steps 1 \
  --log_completions true \
  --split_dataset_ratio 0 --dataset_num_proc 1 --dataloader_num_workers 0 \
  --report_to none --output_dir "${OUTPUT_DIR}/swift"

set +e
"${PYTHON}" - "${OUTPUT_DIR}" "${EXPECTED_GROUPS}" <<'PY'
import json, math, sys
from pathlib import Path
from safetensors.torch import load_file

output = Path(sys.argv[1])
expected_rows = int(sys.argv[2]) * 4
rows = [json.loads(line) for line in (output / "reward_trace.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
adapters = list(output.rglob("adapter_model.safetensors"))
states = [json.loads(path.read_text(encoding="utf-8")) for path in output.rglob("trainer_state.json")]
history = [row for state in states for row in state.get("log_history", [])]
grad_norms = [float(row["grad_norm"]) for row in history if row.get("grad_norm") is not None]
global_steps = [int(state.get("global_step", 0)) for state in states]
valid_reviewer_rows = [
    row for row in rows
    if row.get("record", {}).get("reward_source") == "score_only_ordinal_reviewer"
]
reward_values = [float(row["reward"]) for row in rows]
reward_mean = sum(reward_values) / len(reward_values) if reward_values else None
reward_std = (
    math.sqrt(sum((value - reward_mean) ** 2 for value in reward_values) / len(reward_values))
    if reward_values else None
)
lora_b_nonzero = False
for path in adapters:
    tensors = load_file(str(path), device="cpu")
    lora_b_nonzero |= any("lora_B" in name and bool(tensor.count_nonzero()) for name, tensor in tensors.items())
checks = {
    "expected_reward_rows": len(rows) == expected_rows,
    "score_only_revision": all(row.get("reward_revision") == "score_only_ordinal_reviewer_v1" for row in rows),
    "finite_rewards": all(math.isfinite(float(row["reward"])) for row in rows),
    "reward_range": all(0.0 <= float(row["reward"]) <= 1.0 for row in rows),
    "three_head_scores_present": all(
        row["record"]["reward_source"] == "deterministic_rejection"
        or set(row["record"]["reviewer_score"]["expected_scores"]) == {"evidence_quality", "answerability", "qa_formality"}
        for row in rows
    ),
    "valid_reviewer_rewards_at_least_two": len(valid_reviewer_rows) >= 2,
    "reward_std_positive": reward_std is not None and reward_std > 0.0,
    "global_step_at_least_one": bool(global_steps) and max(global_steps) >= 1,
    "adapter_model_safetensors": bool(adapters),
    "finite_nonzero_grad_norm": any(math.isfinite(value) and value > 0.0 for value in grad_norms),
    "lora_B_nonzero": lora_b_nonzero,
}
result = {
    "status": "passed" if all(checks.values()) else "failed",
    "checks": checks,
    "failed_checks": [name for name, passed in checks.items() if not passed],
    "reward_count": len(rows),
    "valid_reviewer_reward_count": len(valid_reviewer_rows),
    "reward_mean": reward_mean,
    "reward_std": reward_std,
    "global_step": max(global_steps) if global_steps else 0,
}
(output / "smoke_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(0 if result["status"] == "passed" else 2)
PY
VALIDATION_STATUS=$?
set -e

"${PYTHON}" - "${OUTPUT_DIR}" "${DATASET}" "${POLICY_MODEL}" "${REVIEWER_MODEL}" "${REVIEWER_CHECKPOINT}" "${MAX_STEPS}" "${VALIDATION_STATUS}" <<'PY'
import json, sys
from pathlib import Path
output = Path(sys.argv[1])
manifest = {
    "status": "passed" if int(sys.argv[7]) == 0 else "failed",
    "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID"),
    "reward_revision": "score_only_ordinal_reviewer_v1",
    "reward_func": "egoqa_score_only_ordinal_v1",
    "reward_formula": "0.4*evidence_expected_norm+0.4*answerability_expected_norm+0.2*formality_expected_norm",
    "dataset": sys.argv[2],
    "policy_model": sys.argv[3],
    "reviewer_model": sys.argv[4],
    "reviewer_checkpoint": sys.argv[5],
    "max_steps": int(sys.argv[6]),
    "reward_trace": "reward_trace.jsonl",
    "smoke_result": "smoke_result.json",
}
(output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

if [[ "${VALIDATION_STATUS}" -ne 0 ]]; then
  echo "score_only_grpo_smoke_status=failed output=${OUTPUT_DIR}" >&2
  exit "${VALIDATION_STATUS}"
fi
echo "score_only_grpo_smoke_status=passed output=${OUTPUT_DIR}"
