#!/usr/bin/env bash
#SBATCH --job-name=pareto-dpo-gate0
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-gate0-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-gate0-%j.err
set -euo pipefail

MODE=gate0
export MODE
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
source "${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/common.sh"
DPO_DATA_DIR=${OUTDIR}/dataset
export DPO_DATA_DIR

echo "SLURM_JOB_ID=${SLURM_JOB_ID} OUTDIR=${OUTDIR}"
"${PYTHON}" -m unittest discover \
  -s tests/training/grpo_v3/experiments/annotated_preference -p 'test_*.py' -v
"${PYTHON}" -m compileall -q training/grpo_v3/experiments/annotated_preference
"${PYTHON}" -m training.grpo_v3.experiments.annotated_preference.build_dataset build \
  --csv "${CSV_PATH}" \
  --split "${SPLIT_PATH}" \
  --media-map "${MEDIA_MAP}" \
  --output-dir "${DPO_DATA_DIR}"

test -s "${DPO_DATA_DIR}/train_dpo.jsonl"
test -s "${DPO_DATA_DIR}/validation_dpo.jsonl"
test -s "${DPO_DATA_DIR}/train_pair_index.jsonl"
test -s "${DPO_DATA_DIR}/validation_pair_index.jsonl"
test -s "${DPO_DATA_DIR}/overfit_4_dpo.jsonl"
test -s "${DPO_DATA_DIR}/pareto_audit.json"
test -s "${DPO_DATA_DIR}/dataset_manifest.json"
printf '%s\n' "${DPO_DATA_DIR}" > "${OUTDIR}/dpo_data_dir.txt"
