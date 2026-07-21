#!/usr/bin/env bash
#SBATCH --job-name=egoqa-v3-g4
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/grpo-v3-gate4-%j.out
#SBATCH --error=logs/grpo-v3-gate4-%j.err

set -euo pipefail
REQUIRED_FRAMEWORK="ms-swift==4.2.2"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
INFERENCE_ENV="${INFERENCE_ENV:-/scratch/${USER}/envs/egoqa-grpo}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
REVIEW_MODEL="${REVIEW_MODEL:-/scratch/${USER}/models/Qwen3-VL-8B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${GRPO_V3_ROOT}/gate4_${SLURM_JOB_ID}}"
SPLIT_DIR="${SPLIT_DIR:-${GRPO_V3_ROOT}/gate4_split_seed42}"
TRAIN_DATASET="${SPLIT_DIR}/train_native_video.jsonl"; EVAL_DATASET="${SPLIT_DIR}/eval_native_video.jsonl"; SPLIT_MANIFEST="${SPLIT_DIR}/split_manifest.json"
REVIEW_BASE_URL="${REVIEW_BASE_URL:-http://127.0.0.1:8001/v1}"; REVIEW_MAX_NUM_SEQS="${REVIEW_MAX_NUM_SEQS:-4}"
POLICY_INPUT="native_video"; EGOQA_FORMAT_REWARD_REVISION="json_three_tier_v1"; export EGOQA_FORMAT_REWARD_REVISION

JOB_HOME="${JOB_HOME:-/scratch/${USER}/job_home}"; export HOME="${JOB_HOME}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}" XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/${USER}/triton_cache}" TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/${USER}/torchinductor_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/${USER}/vllm_cache}" CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/scratch/${USER}/cuda_cache}"
export FLASHINFER_WORKSPACE_DIR="${FLASHINFER_WORKSPACE_DIR:-/scratch/${USER}/flashinfer_cache}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}" TOKENIZERS_PARALLELISM=false VLLM_USE_FLASHINFER_SAMPLER=0
export FPS="${FPS:-1}" FPS_MIN_FRAMES="${FPS_MIN_FRAMES:-4}" FPS_MAX_FRAMES="${FPS_MAX_FRAMES:-64}" VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-50176}"
export EGOQA_GRPO_V3_REWARD_TRACE="${OUTPUT_DIR}/reward_trace.jsonl" EGOQA_POLICY_MODEL="${POLICY_MODEL}" EGOQA_REVIEW_MODEL="${REVIEW_MODEL}" EGOQA_REVIEW_BASE_URL="${REVIEW_BASE_URL}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${GRPO_V3_ROOT}" "${OUTPUT_DIR}" "${HOME}" "${HF_HOME}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_DIR}"
PYTHON="${TRAIN_ENV}/bin/python"; SWIFT="${TRAIN_ENV}/bin/swift"; VLLM="${INFERENCE_ENV}/bin/vllm"
for required in "${PYTHON}" "${SWIFT}" "${VLLM}" "${POLICY_MODEL}" "${REVIEW_MODEL}" "${TRAIN_DATASET}" "${EVAL_DATASET}" "${SPLIT_MANIFEST}"; do [[ -e "${required}" ]] || exit 2; done
"${PYTHON}" -c 'import importlib.metadata as m; assert m.version("ms-swift") == "4.2.2"'
"${SWIFT}" rlhf --help > "${OUTPUT_DIR}/swift_rlhf_help.txt"; grep -- '--tuner_type' "${OUTPUT_DIR}/swift_rlhf_help.txt" >/dev/null
"${PYTHON}" -m pip freeze > "${OUTPUT_DIR}/dependencies.txt"

if [[ -z "${GATE3_DIR:-}" ]]; then GATE3_DIR="$(head -n 1 "${GRPO_V3_ROOT}/latest_gate3_output.txt")"; fi
ADAPTER_DIR="$(${PYTHON} -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); r=json.load(open(d/"gate3_result.json")); m=json.load(open(d/"run_manifest.json")); assert r["status"]=="passed"; assert m["reward_revision"]=="json_three_tier_v1"; p=Path(m["adapter_dir"]); assert p.is_dir(); print(p)' "${GATE3_DIR}")"
"${PYTHON}" -c 'import json,sys; m=json.load(open(sys.argv[1])); a=set(m["train_evidence_ids"]); b=set(m["eval_evidence_ids"]); assert m["train_count"]==40 and m["eval_count"]==10 and len(a)==40 and len(b)==10 and not a&b and m["intersection_count"]==0' "${SPLIT_MANIFEST}"
cp "${SPLIT_MANIFEST}" "${OUTPUT_DIR}/split_manifest.json"
EGOQA_EVAL_EVIDENCE_IDS="$(${PYTHON} -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["eval_evidence_ids"]))' "${SPLIT_MANIFEST}")"; export EGOQA_EVAL_EVIDENCE_IDS
"${PYTHON}" -m training.grpo_v3_preflight --dataset "${TRAIN_DATASET}" --check-repo-reward --output "${OUTPUT_DIR}/cpu_preflight_train.json"
"${PYTHON}" -m training.grpo_v3_preflight --dataset "${EVAL_DATASET}" --check-repo-reward --output "${OUTPUT_DIR}/cpu_preflight_eval.json"
: > "${EGOQA_GRPO_V3_REWARD_TRACE}"

GPU_MONITOR_PID=""; REVIEW_PID=""
cleanup(){ [[ -z "${REVIEW_PID}" ]] || kill "${REVIEW_PID}" 2>/dev/null || true; [[ -z "${GPU_MONITOR_PID}" ]] || kill "${GPU_MONITOR_PID}" 2>/dev/null || true; }
trap cleanup EXIT
nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu --format=csv -l 5 > "${OUTPUT_DIR}/gpu_metrics.csv" & GPU_MONITOR_PID=$!
REVIEW_LOG="${PROJECT_ROOT}/logs/grpo-v3-review-${SLURM_JOB_ID}.log"
CUDA_VISIBLE_DEVICES=1 "${VLLM}" serve "${REVIEW_MODEL}" --host 127.0.0.1 --port 8001 --dtype bfloat16 --max-model-len 32768 \
  --max-num-seqs "${REVIEW_MAX_NUM_SEQS}" --gpu-memory-utilization 0.80 --gdn-prefill-backend triton \
  --allowed-local-media-path "/scratch/${USER}" > "${REVIEW_LOG}" 2>&1 & REVIEW_PID=$!
for _ in $(seq 1 240); do curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null && break; kill -0 "${REVIEW_PID}" 2>/dev/null || exit 3; sleep 10; done
curl --fail --silent "${REVIEW_BASE_URL}/models" >/dev/null

CUDA_VISIBLE_DEVICES=0 "${SWIFT}" rlhf --rlhf_type grpo --model "${POLICY_MODEL}" --adapters "${ADAPTER_DIR}" \
  --dataset "${TRAIN_DATASET}" --val_dataset "${EVAL_DATASET}" --external_plugins "${PROJECT_ROOT}/training/grpo_v3_reward_plugin.py" \
  --reward_funcs egoqa_repo_native_judge --tuner_type lora --torch_dtype bfloat16 --freeze_vit true --freeze_aligner true \
  --target_modules q_proj v_proj --lora_rank 8 --lora_alpha 16 --use_vllm false --num_generations 4 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 --gradient_accumulation_steps 1 --gradient_checkpointing true \
  --max_steps 40 --seed 42 --data_seed 42 --dataset_shuffle true --eval_strategy steps --eval_on_start true --eval_steps 40 \
  --max_length 32768 --max_completion_length 1024 --learning_rate 1e-5 --beta 0.0 --temperature 0.7 \
  --save_steps 10 --save_total_limit 1 --logging_steps 1 --log_completions true --split_dataset_ratio 0 \
  --dataset_num_proc 1 --dataloader_num_workers 0 --report_to none --output_dir "${OUTPUT_DIR}/swift"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -m training.grpo_v3_adapter_reload --model-path "${POLICY_MODEL}" --search-root "${OUTPUT_DIR}" --output "${OUTPUT_DIR}/adapter_reload.json"
"${PYTHON}" -m training.grpo_v3_convergence --gate 4 --trace "${EGOQA_GRPO_V3_REWARD_TRACE}" --training-output "${OUTPUT_DIR}" --split-manifest "${SPLIT_MANIFEST}" --output "${OUTPUT_DIR}/convergence_metrics.json"
set +e
"${PYTHON}" -m training.grpo_v3_gate_validate --output-dir "${OUTPUT_DIR}" --gate 4
GATE_VALIDATE_STATUS=$?
set -e
"${PYTHON}" -m training.grpo_v3_summary --output-dir "${OUTPUT_DIR}" --gate 4 --dataset "${TRAIN_DATASET}" --eval-dataset "${EVAL_DATASET}" --split-manifest "${SPLIT_MANIFEST}" \
  --policy-model "${POLICY_MODEL}" --reviewer-model "${REVIEW_MODEL}" --job-id "${SLURM_JOB_ID}" --parent-run "${GATE3_DIR}" \
  --slurm-stdout "${PROJECT_ROOT}/logs/grpo-v3-gate4-${SLURM_JOB_ID}.out" --slurm-stderr "${PROJECT_ROOT}/logs/grpo-v3-gate4-${SLURM_JOB_ID}.err" --reviewer-log "${REVIEW_LOG}"
if (( GATE_VALIDATE_STATUS != 0 )); then
  echo "Gate 4 验收失败；失败 manifest 已保留在 ${OUTPUT_DIR}/run_manifest.json" >&2
  exit "${GATE_VALIDATE_STATUS}"
fi
echo "${OUTPUT_DIR}" > "${GRPO_V3_ROOT}/latest_gate4_output.txt"
echo "framework=${REQUIRED_FRAMEWORK} policy_input=${POLICY_INPUT} output=${OUTPUT_DIR}"
