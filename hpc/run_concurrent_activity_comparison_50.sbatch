#!/usr/bin/env bash
#SBATCH --job-name=egolife_impl50
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

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
# Compatibility launcher: concurrency is now one of several optional implicit
# directions in the active underrepresented-family run.
export EGOLIFE2U_OUTPUT_ROOT="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/implicit_underrepresented_families_50}"

cd "${PROJECT_ROOT}"
exec bash hpc/run_clip_pruned_sampling_neutral_pf_50.sbatch
