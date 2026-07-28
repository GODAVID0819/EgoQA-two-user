#!/usr/bin/env bash
#SBATCH --job-name=egoqa-cvr-s1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/grpo-v3-cvr-smoke1-%j.out
#SBATCH --error=logs/grpo-v3-cvr-smoke1-%j.err

set -euo pipefail

MODE="${MODE:-smoke1}"
STEPS="${STEPS:-1}"
EXPECTED_GROUPS="${EXPECTED_GROUPS:-1}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
INFERENCE_ENV="${INFERENCE_ENV:-/scratch/${USER}/envs/egoqa-grpo}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
REVIEW_MODEL="${REVIEW_MODEL:-/scratch/${USER}/models/Qwen3-VL-8B-Instruct}"
REVIEW_BASE_URL="${REVIEW_BASE_URL:-http://127.0.0.1:8001/v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/qa_cross_view_relation_${MODE}_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/qa_cross_view_relation_${MODE}_${SLURM_JOB_ID}}"

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
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export FPS="${FPS:-1}"
export FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}"
export FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}"
export VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export EGOQA_REVIEW_MODEL="${REVIEW_MODEL}"
export EGOQA_REVIEW_BASE_URL="${REVIEW_BASE_URL}"
export EGOQA_REVIEW_MAX_NEW_TOKENS="${EGOQA_REVIEW_MAX_NEW_TOKENS:-2048}"
export EGOQA_REVIEW_TIMEOUT_SECONDS="${EGOQA_REVIEW_TIMEOUT_SECONDS:-900}"
export EGOQA_REWARD_TRACE="${OUTPUT_DIR}/reward_trace.jsonl"
export EGOQA_GRPO_V3_REWARD_TRACE="${EGOQA_REWARD_TRACE}"
export EGOQA_HELDOUT_EVIDENCE_IDS="${EGOQA_HELDOUT_EVIDENCE_IDS:-}"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRPO_V3_ROOT}" "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}"
PYTHON="${TRAIN_ENV}/bin/python"
SWIFT="${TRAIN_ENV}/bin/swift"
VLLM="${INFERENCE_ENV}/bin/vllm"
for required in "${PYTHON}" "${SWIFT}" "${VLLM}" "${POLICY_MODEL}" "${REVIEW_MODEL}" "${FFMPEG_ENV}/bin/ffmpeg"; do
  [[ -e "${required}" ]] || { echo "missing required path: ${required}" >&2; exit 2; }
done

"${PYTHON}" -m training.torch_storage_preflight --allowed-root "${JOB_SCRATCH_ROOT}" --output "${OUTPUT_DIR}/storage_preflight.json"
"${PYTHON}" -c 'import importlib.metadata as m; assert m.version("ms-swift") == "4.2.2"'
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader > "${OUTPUT_DIR}/gpu_environment.csv"
nvidia-smi > "${OUTPUT_DIR}/nvidia_smi.txt"
"${FFMPEG_ENV}/bin/ffmpeg" -version > "${OUTPUT_DIR}/ffmpeg_environment.txt"
"${PYTHON}" - <<'PY' > "${OUTPUT_DIR}/video_decoder_environment.txt"
import torch
from importlib.metadata import version
from torchcodec.decoders import VideoDecoder
print("torch =", torch.__version__)
print("torchcodec =", version("torchcodec"))
print("video_decoder_backend =", VideoDecoder.__module__)
PY

GATE2_DIR="${GATE2_DIR:-$(head -n 1 "${GRPO_V3_ROOT}/latest_gate2_output.txt")}"
GATE0_DIR="${GATE0_DIR:-$(head -n 1 "${GRPO_V3_ROOT}/latest_gate0_output.txt")}"
DATASET="${DATASET:-${GATE0_DIR}/train_native_video.jsonl}"
ADAPTER_DIR="$("${PYTHON}" -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); assert json.load(open(d/"gate2_result.json"))["status"]=="passed"; p=Path(json.load(open(d/"run_manifest.json"))["adapter_dir"]); assert p.is_dir(); print(p)' "${GATE2_DIR}")"
"${PYTHON}" -m training.grpo_v3.baseline.preflight --dataset "${DATASET}" --output "${OUTPUT_DIR}/cpu_preflight.json"
: > "${EGOQA_REWARD_TRACE}"

GPU_MONITOR_PID=""
REVIEW_PID=""
cleanup() {
  [[ -z "${REVIEW_PID}" ]] || kill "${REVIEW_PID}" 2>/dev/null || true
  [[ -z "${GPU_MONITOR_PID}" ]] || kill "${GPU_MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu --format=csv -l 5 > "${OUTPUT_DIR}/gpu_metrics.csv" &
GPU_MONITOR_PID=$!
REVIEW_LOG="${PROJECT_ROOT}/logs/grpo-v3-cvr-review-${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 "${VLLM}" serve "${REVIEW_MODEL}" --host 127.0.0.1 --port 8001 \
  --dtype bfloat16 --max-model-len 32768 --max-num-seqs 4 --gpu-memory-utilization 0.80 \
  --gdn-prefill-backend triton > "${REVIEW_LOG}" 2>&1 &
REVIEW_PID=$!
for _ in $(seq 1 240); do
  curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null && break
  kill -0 "${REVIEW_PID}" 2>/dev/null || exit 3
  sleep 10
done
curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null

CUDA_VISIBLE_DEVICES=0 "${SWIFT}" rlhf --rlhf_type grpo --model "${POLICY_MODEL}" --adapters "${ADAPTER_DIR}" \
  --dataset "${DATASET}" --external_plugins "${PROJECT_ROOT}/training/grpo_v3/runtime/reward_plugin.py" \
  --reward_funcs egoqa_cross_view_relation_v2 --tuner_type lora --torch_dtype bfloat16 \
  --freeze_vit true --freeze_aligner true --target_modules q_proj v_proj --lora_rank 8 --lora_alpha 16 \
  --use_vllm false --num_generations 4 --per_device_train_batch_size 4 --gradient_accumulation_steps 1 \
  --gradient_checkpointing true --max_steps "${STEPS}" --seed 42 --data_seed 42 --dataset_shuffle false \
  --max_length 32768 --max_completion_length 1024 --learning_rate 1e-5 --lr_scheduler_type constant \
  --beta 0.0 --temperature 0.7 --top_p 1.0 --save_steps "${STEPS}" --save_total_limit 1 --logging_steps 1 \
  --log_completions true --split_dataset_ratio 0 --dataset_num_proc 1 --dataloader_num_workers 0 \
  --report_to none --output_dir "${OUTPUT_DIR}/swift"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m training.grpo_v3.shared.adapter_reload \
  --model-path "${POLICY_MODEL}" --search-root "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/adapter_reload.json"

VALIDATION_STATUS=0
if [[ "${EXPECTED_GROUPS}" -gt 1 ]]; then
  set +e
  "${PYTHON}" -m training.grpo_v3.experiments.qa_cross_view_relation.convergence \
    --trace "${EGOQA_REWARD_TRACE}" --expected-groups "${EXPECTED_GROUPS}" \
    --output "${OUTPUT_DIR}/convergence_metrics.json"
  VALIDATION_STATUS=$?
  set -e
else
  set +e
  "${PYTHON}" - "${EGOQA_REWARD_TRACE}" "${OUTPUT_DIR}/smoke_result.json" <<'PY'
import json, math, sys
from pathlib import Path
rows = [json.loads(x) for x in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines() if x.strip()]
reward_rows = [r for r in rows if r.get("reward_kind") == "qa_cross_view_relation" and r.get("failure_stage") in (None, "")]
ok = len(reward_rows) == 4 and all(math.isfinite(float(r["reward"])) for r in reward_rows)
components = [set(((r.get("record") or {}).get("reward_components") or {})) for r in reward_rows]
ok = ok and all(c == {"qa_cross_view_relation"} for c in components)
Path(sys.argv[2]).write_text(json.dumps({"status": "passed" if ok else "failed", "reward_count": len(reward_rows)}, ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(0 if ok else 2)
PY
  VALIDATION_STATUS=$?
  set -e
fi

"${PYTHON}" - "${OUTPUT_DIR}" "${GATE2_DIR}" "${ADAPTER_DIR}" "${DATASET}" "${POLICY_MODEL}" "${REVIEW_MODEL}" "${MODE}" "${STEPS}" "${VALIDATION_STATUS}" <<'PY'
import json, sys
from pathlib import Path
o = Path(sys.argv[1])
gate2 = Path(sys.argv[2])
adapter = Path(sys.argv[3])
dataset = Path(sys.argv[4])
policy = sys.argv[5]
reviewer = sys.argv[6]
mode = sys.argv[7]
steps = int(sys.argv[8])
validation_status = int(sys.argv[9])
manifest = {
    "status": "passed" if validation_status == 0 else "failed",
    "mode": mode,
    "max_steps": steps,
    "reward_revision": "qa_cross_view_relation_v2",
    "reward_func": "egoqa_cross_view_relation_v2",
    "parent_run": str(gate2),
    "parent_adapter": str(adapter),
    "dataset": str(dataset),
    "policy_model": policy,
    "reviewer_model": reviewer,
    "trace": "reward_trace.jsonl",
}
(o / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_qa_cross_view_relation_${MODE}_output.txt"
if [[ "${VALIDATION_STATUS}" -ne 0 ]]; then
  echo "qa_cross_view_relation_${MODE}_status=failed output=${OUTPUT_DIR}" >&2
  exit "${VALIDATION_STATUS}"
fi
echo "qa_cross_view_relation_${MODE}_status=passed output=${OUTPUT_DIR}"
