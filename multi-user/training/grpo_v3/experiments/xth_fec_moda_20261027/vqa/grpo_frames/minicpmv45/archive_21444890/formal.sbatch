#!/bin/bash
#SBATCH --job-name=minicpm45_grpo131
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031/logs/formal_%j.out
set -euo pipefail
export VQA_ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
export VQA_ENV_BIN=/scratch/xl6775/envs/minicpmv45_grpo131_20261031/bin
export VQA_FFMPEG_BIN=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/bin
export VQA_FFMPEG_LIB=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/lib
export FRAME_CACHE=/scratch/xl6775/experiments/vqa_videollama3_grpo131_20261030/frames_156
cd "$VQA_ROOT/code"
source runtime_env.sh
export HF_HUB_OFFLINE=1 CUDA_MODULE_LOADING=LAZY
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
srun bash -c '
set -euo pipefail
GPU_MONITOR_ID=${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-0}}}
GPU_MONITOR_ID=${GPU_MONITOR_ID%%,*}
nvidia-smi -i "$GPU_MONITOR_ID" --query-gpu=timestamp,uuid,utilization.gpu,memory.used,memory.total --format=csv -l 5 > "$VQA_ROOT/logs/gpu_$SLURM_JOB_ID.csv" 2>&1 &
GPU_MONITOR_PID=$!
trap '\''kill "$GPU_MONITOR_PID" 2>/dev/null || true'\'' EXIT
python -u evaluate.py
'
