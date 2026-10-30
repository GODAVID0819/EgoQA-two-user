#!/bin/bash
#SBATCH --job-name=vl3_grpo131_formal
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=vl3_formal_%j.out
set -euo pipefail
: "${VQA_ROOT:?填写本次实验scratch根目录}"
: "${VQA_CODE_ROOT:?填写本目录的绝对路径}"
: "${VQA_ENV_BIN:?填写已准备好的GPU环境bin目录}"
: "${VQA_CPU_REPORT:?填写本次完整CPU检查报告}"
cd "$VQA_CODE_ROOT"
source runtime_env.sh
export HF_HUB_OFFLINE=1 CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS=(--model "$VQA_ROOT/model" --cache "$VQA_ROOT/frames_156" --cpu-report "$VQA_CPU_REPORT" --out "$VQA_ROOT/outputs/formal_$SLURM_JOB_ID")
if [ -n "${VQA_RESUME_FROM:-}" ]; then ARGS+=(--resume-from "$VQA_RESUME_FROM"); fi
# GPU设备只在srun步骤内可见；采样器与模型放在同一步骤。本修复未另开GPU复测。
srun bash -c '
set -euo pipefail
GPU_MONITOR_ID=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}}
GPU_MONITOR_ID=${GPU_MONITOR_ID%%,*}
nvidia-smi -i "$GPU_MONITOR_ID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$VQA_ROOT/logs/gpu_$SLURM_JOB_ID.csv" 2>&1 &
GPU_MONITOR_PID=$!
trap '\''kill "$GPU_MONITOR_PID" 2>/dev/null || true'\'' EXIT
python -u evaluate_grpo.py "$@"
' bash "${ARGS[@]}"
