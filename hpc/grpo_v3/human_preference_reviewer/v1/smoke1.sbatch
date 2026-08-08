#!/usr/bin/env bash
#SBATCH --job-name=reviewer-v1-smoke
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/logs/reviewer-v1-smoke-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/logs/reviewer-v1-smoke-%j.err
set -euo pipefail
MODE=smoke
export MODE
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-grpo-clean}"
source "${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/common.sh"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTDIR=${OUTDIR}"
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.train smoke \
  --csv "${CSV_PATH}" --media-map "${MEDIA_MAP}" --model "${MODEL_DIR}" \
  --output-dir "${OUTDIR}" --max-steps 1 --epochs 1
test -s "${OUTDIR}/training_result.json"
test -s "${OUTDIR}/checkpoint/parameter_audit.json"
