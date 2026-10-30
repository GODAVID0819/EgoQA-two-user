#!/bin/bash
#SBATCH --account=torch_pr_111_general
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/vqa_eval/logs/%x_%j.out
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
cd /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames/speaker_viz
python build.py
