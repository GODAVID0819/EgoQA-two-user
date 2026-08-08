#!/usr/bin/env bash
# Activate the Qwen/CLIP conda environment on HPC.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"
CONDA_SH=""
for candidate in \
  "${CONDA_ROOT}/etc/profile.d/conda.sh" \
  "/share/apps/anaconda3/2025.06/etc/profile.d/conda.sh" \
  "/share/apps/anaconda3/etc/profile.d/conda.sh" \
  "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh"
do
  if [[ -s "${candidate}" ]]; then
    CONDA_SH="${candidate}"
    break
  fi
done
if [[ -z "${CONDA_SH}" ]]; then
  echo "Could not find conda.sh. Set CONDA_ROOT to your conda install root." >&2
  exit 1
fi
echo "conda_sh=${CONDA_SH}"
echo "conda_env=${CONDA_ENV_NAME}"
source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"
