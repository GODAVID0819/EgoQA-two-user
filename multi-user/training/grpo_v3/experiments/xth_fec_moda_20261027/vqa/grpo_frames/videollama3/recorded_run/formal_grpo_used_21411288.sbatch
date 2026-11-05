#!/bin/bash
#SBATCH --job-name=vl3_grpo131_formal
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xl6775/experiments/vqa_videollama3_grpo131_20261030/logs/formal_%j.out
set -euo pipefail
export VQA_ROOT=/scratch/xl6775/experiments/vqa_videollama3_grpo131_20261030
export VQA_ENV_BIN=/scratch/xl6775/envs/videollama3_grpo131_gpu_20261030/bin
export VQA_FFMPEG_BIN=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/bin
export VQA_FFMPEG_LIB=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/lib
cd "$VQA_ROOT/code"
source runtime_env.sh
export HF_HUB_OFFLINE=1 CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CPU_JOB=$(python -c 'import json,os;from pathlib import Path;print(json.loads((Path(os.environ["VQA_ROOT"])/"jobs/cpu_submission.json").read_text())["job_id"])')
GPU_MONITOR_ID=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}
GPU_MONITOR_ID=${GPU_MONITOR_ID%%,*}
nvidia-smi -i "$GPU_MONITOR_ID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$VQA_ROOT/logs/gpu_$SLURM_JOB_ID.csv" 2>&1 &
GPU_MONITOR_PID=$!
trap 'kill "$GPU_MONITOR_PID" 2>/dev/null || true' EXIT
ARGS=(--model "$VQA_ROOT/model" --cache "$VQA_ROOT/frames_156" --cpu-report "$VQA_ROOT/cpu_ready_$CPU_JOB.json" --out "$VQA_ROOT/outputs/formal_$SLURM_JOB_ID")
if [ -n "${VQA_RESUME_FROM:-}" ]; then ARGS+=(--resume-from "$VQA_RESUME_FROM"); fi
srun python -u evaluate_grpo.py "${ARGS[@]}"
