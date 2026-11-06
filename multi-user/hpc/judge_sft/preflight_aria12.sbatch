#!/bin/bash
#SBATCH --job-name=aria12_c_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:20:00

set -euo pipefail
: "${JUDGE_REPO:?指定已提交版本的multi-user目录}"
JUDGE_STAGE=preflight bash "$JUDGE_REPO/hpc/judge_sft/train_aria12_answerability.sbatch"
