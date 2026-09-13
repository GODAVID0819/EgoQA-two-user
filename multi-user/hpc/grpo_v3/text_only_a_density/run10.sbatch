#!/usr/bin/env bash
#SBATCH --job-name=egoqa-ad-quick10
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/grpo-v3-a-density-quick10-%j.out
#SBATCH --error=logs/grpo-v3-a-density-quick10-%j.err
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
POLICY_MODEL="${POLICY_MODEL:-/scratch/${USER}/models/Qwen3-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo_v3/text_only_a_density_quick10_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/text_only_a_density_quick10_${SLURM_JOB_ID}}"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"
export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm-cache-unused"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
PYTHON="${TRAIN_ENV}/bin/python"
SWIFT="${TRAIN_ENV}/bin/swift"

"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"
"${PYTHON}" -m training.grpo_v3.experiments.text_only_a_density.data \
  --output-dir "${OUTPUT_DIR}/data"
"${PYTHON}" - <<'PY' > "${OUTPUT_DIR}/swift_grpo_import_preflight.txt"
import torch
import transformers
import peft
import swift
import vllm
from swift.rlhf_trainers.grpo_trainer import GRPOTrainer

print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("swift", swift.__version__)
print("vllm", vllm.__version__)
print("GRPOTrainer", GRPOTrainer.__name__)
PY
nvidia-smi --query-gpu=index,uuid,name,memory.total,driver_version --format=csv,noheader \
  > "${OUTPUT_DIR}/gpu_environment.csv"
MODEL_LOAD_BOUNDARY=1

export EGOQA_GRPO_V3_REWARD_TRACE="${OUTPUT_DIR}/reward_trace.jsonl"
"${SWIFT}" rlhf \
  --rlhf_type grpo \
  --model ${POLICY_MODEL} \
  --dataset "${OUTPUT_DIR}/data/train.jsonl" \
  --external_plugins "${PROJECT_ROOT}/training/grpo_v3/experiments/text_only_a_density/reward_plugin.py" \
  --reward_funcs egoqa_text_only_a_density \
  --output_dir "${OUTPUT_DIR}" \
  --max_steps 10 \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --target_modules q_proj v_proj \
  --lora_rank 8 \
  --lora_alpha 16 \
  --freeze_vit true \
  --freeze_aligner true \
  --num_generations 4 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing true \
  --max_completion_length 64 \
  --learning_rate 1e-4 \
  --lr_scheduler_type constant \
  --beta 0.0 \
  --temperature 0.7 \
  --top_p 1.0 \
  --seed 42 \
  --data_seed 42 \
  --dataset_shuffle false \
  --use_vllm false \
  --save_steps 10

"${PYTHON}" -m training.grpo_v3.experiments.text_only_a_density.analyze \
  --trace "${OUTPUT_DIR}/reward_trace.jsonl" \
  --output "${OUTPUT_DIR}/quick_convergence_summary.json" \
  --expected-steps 10
echo "${OUTPUT_DIR}" > "${PROJECT_ROOT}/outputs/grpo_v3/latest_text_only_a_density_quick10_output.txt"
