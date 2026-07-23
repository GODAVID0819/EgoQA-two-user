#!/usr/bin/env bash
#SBATCH --job-name=egolife_qa201_300
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=128G
#SBATCH --time=10:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
WORKER="${PROJECT_ROOT}/egolife_two_user_qa/hpc/run_qa_packet_slice_100.sh"
if [[ ! -s "${WORKER}" ]]; then
  WORKER="${PROJECT_ROOT}/hpc/run_qa_packet_slice_100.sh"
fi
if [[ ! -s "${WORKER}" ]]; then
  echo "error=qa_slice_worker_missing path=${WORKER}" >&2
  exit 1
fi

export PACKET_START_INDEX=200
export BATCH_LABEL="packets_201_300"
exec bash "${WORKER}"

