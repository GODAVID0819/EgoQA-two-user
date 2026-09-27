#!/usr/bin/env bash
#SBATCH --job-name=judge38-eval-quarter
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --constraint=h200
#SBATCH --mem=420G
#SBATCH --time=8:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
ENV_ROOT="${ENV_ROOT:-/scratch/${USER}/conda/envs/qwen38-vllm}"
PYTHON="${ENV_ROOT}/bin/python"

MODEL_PATH="${MODEL_PATH:-/scratch/${USER}/egolife_vllm_runtime/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"

# 100% original normalized manifest.
FULL_MANIFEST="${FULL_MANIFEST:-/scratch/${USER}/egolife_judge_sft/manifests_40_packets_18017425/train.jsonl}"

# Exact 50% manifest used in the current SFT run.
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/scratch/${USER}/egolife_judge_sft/manifests_40_packets_18017425_half_seed17/train.jsonl}"

TRAIN_RUN="${TRAIN_RUN:-/scratch/${USER}/egolife_judge_sft/train_half5_ga8_l2_f050_18168001}"
ADAPTER_DIR="${ADAPTER_DIR:-${TRAIN_RUN}/trainer/checkpoint-56}"
TRAINING_CONTRACT="${TRAINING_CONTRACT:-${TRAIN_RUN}/trainer/training_contract.json}"

EVAL_SEED="${EVAL_SEED:-2718}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/egolife_judge_sft/eval_epoch1_quarter}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/eval56_${SLURM_JOB_ID}}"
EVAL_MANIFEST="${OUTPUT_DIR}/eval_quarter.jsonl"
EVAL_AUDIT="${OUTPUT_DIR}/eval_quarter_audit.json"

JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/job_scratch/judge38_eval_quarter_${SLURM_JOB_ID}}"

mkdir -p "${OUTPUT_DIR}/base" "${OUTPUT_DIR}/lora" "${JOB_SCRATCH_ROOT}" hpc/logs

for required in \
  "${PYTHON}" \
  "${MODEL_PATH}" \
  "${FULL_MANIFEST}" \
  "${TRAIN_MANIFEST}" \
  "${ADAPTER_DIR}/adapter_config.json" \
  "${TRAINING_CONTRACT}" \
  "${PACKAGE_ROOT}/training/judge_sft/prepare_eval_quarter.py" \
  "${PACKAGE_ROOT}/training/judge_sft/evaluate_single_gpu.py" \
  "${PACKAGE_ROOT}/training/judge_sft/merge_dual_eval.py"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"
export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p \
  "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
  "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" "${TMPDIR}"

cd "${PACKAGE_ROOT}"

echo "=== Build eval quarter: 50% of the unused half ==="
"${PYTHON}" -m training.judge_sft.prepare_eval_quarter \
  --full-manifest "${FULL_MANIFEST}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --output-manifest "${EVAL_MANIFEST}" \
  --audit-output "${EVAL_AUDIT}" \
  --seed "${EVAL_SEED}"

echo "=== Eval manifest audit ==="
cat "${EVAL_AUDIT}"

{
  echo "date=$(date -Is)"
  echo "host=$(hostname)"
  echo "model=${MODEL_PATH}"
  echo "adapter=${ADAPTER_DIR}"
  echo "full_manifest=${FULL_MANIFEST}"
  echo "train_manifest=${TRAIN_MANIFEST}"
  echo "eval_manifest=${EVAL_MANIFEST}"
  sha256sum \
    training/judge_sft/prepare_eval_quarter.py \
    training/judge_sft/evaluate_single_gpu.py \
    training/judge_sft/merge_dual_eval.py \
    training/judge_sft/collator.py \
    training/judge_sft/data.py \
    training/judge_sft/loss.py \
    "${EVAL_MANIFEST}" \
    "${ADAPTER_DIR}/adapter_config.json" \
    "${ADAPTER_DIR}"/adapter_model*.safetensors
} > "${OUTPUT_DIR}/audit.txt"

nvidia-smi --query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu \
  --format=csv -l 10 > "${OUTPUT_DIR}/gpu_metrics.csv" &
MON_PID=$!

cleanup() {
  kill "${MON_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== Launch independent inference ==="
echo "GPU0: base"
echo "GPU1: LoRA checkpoint-56"

set +e
CUDA_VISIBLE_DEVICES=0 \
"${PYTHON}" -u -m training.judge_sft.evaluate_single_gpu \
  --mode base \
  --model-id "${MODEL_PATH}" \
  --eval-manifest "${EVAL_MANIFEST}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --training-contract "${TRAINING_CONTRACT}" \
  --output-dir "${OUTPUT_DIR}/base" \
  > "${OUTPUT_DIR}/base.log" 2>&1 &
BASE_PID=$!

CUDA_VISIBLE_DEVICES=1 \
"${PYTHON}" -u -m training.judge_sft.evaluate_single_gpu \
  --mode lora \
  --model-id "${MODEL_PATH}" \
  --adapter-dir "${ADAPTER_DIR}" \
  --eval-manifest "${EVAL_MANIFEST}" \
  --train-manifest "${TRAIN_MANIFEST}" \
  --training-contract "${TRAINING_CONTRACT}" \
  --output-dir "${OUTPUT_DIR}/lora" \
  > "${OUTPUT_DIR}/lora.log" 2>&1 &
LORA_PID=$!

wait "${BASE_PID}"
BASE_RC=$?
wait "${LORA_PID}"
LORA_RC=$?
set -e

echo "base_rc=${BASE_RC}"
echo "lora_rc=${LORA_RC}"

if [[ "${BASE_RC}" != "0" || "${LORA_RC}" != "0" ]]; then
  echo "One evaluator failed." >&2
  echo "===== base tail =====" >&2
  tail -100 "${OUTPUT_DIR}/base.log" >&2 || true
  echo "===== lora tail =====" >&2
  tail -100 "${OUTPUT_DIR}/lora.log" >&2 || true
  exit 1
fi

echo "=== Paired merge ==="
"${PYTHON}" -m training.judge_sft.merge_dual_eval \
  --base "${OUTPUT_DIR}/base/predictions.jsonl" \
  --lora "${OUTPUT_DIR}/lora/predictions.jsonl" \
  --output "${OUTPUT_DIR}/comparison.json"

echo
echo "===== BASE SUMMARY ====="
cat "${OUTPUT_DIR}/base/summary.json"
echo
echo "===== LORA SUMMARY ====="
cat "${OUTPUT_DIR}/lora/summary.json"
echo
echo "===== PAIRED COMPARISON ====="
cat "${OUTPUT_DIR}/comparison.json"
echo
echo "results=${OUTPUT_DIR}"
