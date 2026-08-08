#!/usr/bin/env bash
#SBATCH --job-name=reviewer-s0-overfit
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/logs/reviewer-s0-overfit-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean/logs/reviewer-s0-overfit-%j.err
set -euo pipefail
MODE=stage0_overfit
export MODE
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-grpo-clean}"
source "${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/stage0/common.sh"
SMALL_SPLIT="${DATA_DIR}/split_2_1_1.json"
test -s "${SMALL_SPLIT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTDIR=${OUTDIR}"
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.train fit \
  --stage stage0 --csv "${CSV_PATH}" --media-map "${MEDIA_MAP}" --model "${MODEL_DIR}" \
  --split-manifest "${SMALL_SPLIT}" --output-dir "${OUTDIR}" --max-steps 24 --epochs 4 \
  --train-evidence-count 2 --validation-evidence-count 1 --locked-test-evidence-count 1
test -s "${OUTDIR}/training_result.json"
test ! -e "${OUTDIR}/checkpoint/lora_adapter.pt"
