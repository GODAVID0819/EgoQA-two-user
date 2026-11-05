#!/bin/bash
#SBATCH --job-name=vl3_grpo131_videos
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=vl3_cpu_video_%j.out
set -euo pipefail
: "${VQA_ROOT:?填写本次实验scratch根目录}"
: "${VQA_CODE_ROOT:?填写本目录的绝对路径}"
: "${VQA_ENV_BIN:?填写已准备好的CPU环境bin目录}"
cd "$VQA_CODE_ROOT"
source runtime_env.sh
export CUDA_VISIBLE_DEVICES=-1
python -m unittest test_rebuild -v
python rebuild_from_videos.py --cache "$VQA_ROOT/frames_156" --video-cache "$VQA_ROOT/original_video_cache" \
  --workers 8 --report "$VQA_ROOT/rebuild_ready_$SLURM_JOB_ID.json"
python prepare_frames.py --model "$VQA_ROOT/code/upstream" --cache "$VQA_ROOT/frames_156" \
  --workers 8 --report "$VQA_ROOT/cpu_ready_$SLURM_JOB_ID.json"
bash setup_gpu_assets.sh
