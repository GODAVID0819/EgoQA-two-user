#!/usr/bin/env bash
#SBATCH --job-name=egoqa-cvr-p120
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --constraint=l40s
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/grpo-v3-cvr-probe120-%j.out
#SBATCH --error=logs/grpo-v3-cvr-probe120-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
GRPO_V3_ROOT="${GRPO_V3_ROOT:-${PROJECT_ROOT}/outputs/grpo_v3}"
SMOKE_DIR="${SMOKE_DIR:-$(head -n 1 "${GRPO_V3_ROOT}/latest_qa_cross_view_relation_smoke1_output.txt")}"
PYTHON="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}/bin/python"
"${PYTHON}" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "passed"' "${SMOKE_DIR}/smoke_result.json"

export MODE=probe120
export STEPS=120
export EXPECTED_GROUPS=120
exec bash "${PROJECT_ROOT}/hpc/grpo_v3/qa_cross_view_relation/smoke1.sbatch"
