#!/bin/bash
#SBATCH --job-name=aria12_tp4_speed
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=00:25:00
set -euo pipefail
: "${JUDGE_REPO:?}"
JUDGE_TP_SIZE=4 JUDGE_BENCHMARK_STEPS=2 JUDGE_MLP_CHUNK_SIZE=0 \
  bash "$JUDGE_REPO/hpc/judge_sft/train_aria12_answerability.sbatch"
