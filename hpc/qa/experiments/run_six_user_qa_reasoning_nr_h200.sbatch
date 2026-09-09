#!/usr/bin/env bash
#SBATCH --job-name=egoqa_reasoning_ab
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

EXPERIMENT_NAME="reasoning_nr_10x3_h200"
REASONING_MODE="nr"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export EXPERIMENT_NAME REASONING_MODE
source "${SCRIPT_DIR}/run_six_user_qa_reasoning_ab_h200_common.sh"
