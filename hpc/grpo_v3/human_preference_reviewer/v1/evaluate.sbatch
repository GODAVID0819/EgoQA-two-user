#!/usr/bin/env bash
#SBATCH --job-name=reviewer-v1-eval
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/logs/reviewer-v1-eval-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/logs/reviewer-v1-eval-%j.err
set -euo pipefail
MODE=evaluate
export MODE
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1}"
source "${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/common.sh"
echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTDIR=${OUTDIR}"
FORMAL_SPLIT="${DATA_DIR}/split_60_10.json"
CHECKPOINT_DIR="${CHECKPOINT_DIR:?submit with --export=ALL,CHECKPOINT_DIR=/absolute/checkpoint/path}"
EVAL_SPLIT="validation"
test -s "${FORMAL_SPLIT}"
test -s "${CHECKPOINT_DIR}/reviewer_v1_config.json"
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.train evaluate \
  --csv "${CSV_PATH}" --media-map "${MEDIA_MAP}" --model "${MODEL_DIR}" \
  --split-manifest "${FORMAL_SPLIT}" --checkpoint "${CHECKPOINT_DIR}" \
  --split "${EVAL_SPLIT}" --output-dir "${OUTDIR}"
test -s "${OUTDIR}/evaluation_result.json"
