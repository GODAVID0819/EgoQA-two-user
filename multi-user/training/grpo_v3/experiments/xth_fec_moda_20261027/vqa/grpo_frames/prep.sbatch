#!/bin/bash
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/vqa_eval/logs/%x_%j.out
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
export HF_HUB_CACHE=/scratch/tx856/multi_user_qa/vqa_eval/hf_cache HF_HUB_OFFLINE=1 TMPDIR=/scratch/tx856/js/vqaprep_${SLURM_JOB_ID}
mkdir -p $TMPDIR
cd /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames
PYTHONPATH=/scratch/tx856/multi_user_qa/lxr_run/multi-user python prep_frames.py --out frames.json
