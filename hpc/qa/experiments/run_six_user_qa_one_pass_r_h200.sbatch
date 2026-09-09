#!/usr/bin/env bash
#SBATCH --job-name=egoqa_onepass_ab
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=320G
#SBATCH --time=14:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

RUN_MODE="one_pass_r_3x10_h200"
ONE_PASS_REASONING_MODE="r"
PROJECT_ROOT="/scratch/xl6775/projects/EgoQA-two-user-onepass-nr-r-h200-20260909"
export RUN_MODE ONE_PASS_REASONING_MODE PROJECT_ROOT
source "${PROJECT_ROOT}/hpc/qa/experiments/run_six_user_qa_one_pass_ab_h200_common.sh"
