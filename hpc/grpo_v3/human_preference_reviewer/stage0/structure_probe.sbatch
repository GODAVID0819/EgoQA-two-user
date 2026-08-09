#!/usr/bin/env bash
#SBATCH --job-name=reviewer-s0-structure
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/logs/reviewer-s0-structure-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/logs/reviewer-s0-structure-%j.err
set -euo pipefail
MODE=stage0_structure
export MODE
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1}"
source "${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/stage0/common.sh"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTDIR=${OUTDIR}"
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit structure \
  --model "${MODEL_DIR}" --expected-layer-count 36 --output "${OUTDIR}/structure_probe.json"
test -s "${OUTDIR}/structure_probe.json"
