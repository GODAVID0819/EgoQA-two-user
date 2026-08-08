#!/usr/bin/env bash
#SBATCH --job-name=reviewer-v1-train
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/logs/reviewer-v1-train-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/logs/reviewer-v1-train-%j.err
set -euo pipefail
MODE=train
export MODE
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-grpo-clean}"
source "${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/common.sh"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTDIR=${OUTDIR}"
FORMAL_SPLIT="${DATA_DIR}/split_40_10_10.json"
test -s "${FORMAL_SPLIT}"
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.train fit \
  --csv "${CSV_PATH}" --media-map "${MEDIA_MAP}" --model "${MODEL_DIR}" \
  --split-manifest "${FORMAL_SPLIT}" --output-dir "${OUTDIR}" --epochs 3
test -s "${OUTDIR}/training_result.json"
test -s "${OUTDIR}/checkpoint/parameter_audit.json"
