#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_pilot5
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=160G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

RUN_MODE="six_user_qa_pilot_5"
ACCEPTED_TARGET="5"
EVIDENCE_TARGET="30"
MAX_GROUPS="240"
export RUN_MODE ACCEPTED_TARGET EVIDENCE_TARGET MAX_GROUPS

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-grpo-clean}"
bash "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"
