#!/usr/bin/env bash
#SBATCH --job-name=egolife_xgap50
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

export EGOLIFE2U_PREPROCESS_EXPERIMENT_MODE=cross_user_gate
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"

# The sourced launcher owns the shared storage/video/CUDA preflights, starts
# hpc/shared/cuda.py, writes the JobID-derived run manifest, and dispatches the
# cache-only cross-user gate experiment selected above.
source "${PROJECT_ROOT}/hpc/qa/preprocessing/run_temporal_kmeans_grid_50.sbatch"
