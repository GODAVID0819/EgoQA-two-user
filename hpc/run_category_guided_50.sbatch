#!/usr/bin/env bash
#SBATCH --job-name=egolife_count50
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

# Deprecated compatibility entry point. The category-guided experiment has
# been retired; this forwards to the current category-free production launcher.
set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
exec bash "${PROJECT_ROOT}/hpc/run_clip_pruned_sampling_neutral_pf_50.sbatch"
