#!/usr/bin/env bash
#SBATCH --job-name=egoqa-ad-fixed-probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --constraint=l40s
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/grpo-v3-a-density-fixed-probe-%j.out
#SBATCH --error=logs/grpo-v3-a-density-fixed-probe-%j.err
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo_v3/text_only_a_density_fixed_probe_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/text_only_a_density_fixed_probe_${SLURM_JOB_ID}}"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"
export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm-cache-unused"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${PROJECT_ROOT}/logs"
cd "${PROJECT_ROOT}"
PYTHON="${TRAIN_ENV}/bin/python"

"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"
"${PYTHON}" -m training.grpo_v3.experiments.text_only_a_density.fixed_rollout_probe \
  --output-dir "${OUTPUT_DIR}" \
  --steps 10 \
  --learning-rate 0.2 \
  --seed 42
test -s "${OUTPUT_DIR}/fixed_rollout_summary.json"
test -s "${OUTPUT_DIR}/fixed_rollout_analysis.json"
echo "${OUTPUT_DIR}" > "${PROJECT_ROOT}/outputs/grpo_v3/latest_text_only_a_density_fixed_probe_output.txt"
