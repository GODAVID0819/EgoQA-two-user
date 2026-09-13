#!/usr/bin/env bash
#SBATCH --job-name=egoqa-judge-reward-v2-score
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/grpo-judge-reward-v2-score-%j.out
#SBATCH --error=logs/grpo-judge-reward-v2-score-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
ENV_ROOT="${ENV_ROOT:-/scratch/${USER}/envs/egoqa-grpo}"
JOB_HOME="${JOB_HOME:-/scratch/${USER}/job_home}"
export HOME="${JOB_HOME}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/cache}"
export PYTHONPATH="${PROJECT_ROOT}/archived/grpo_v2:${PROJECT_ROOT}:${PYTHONPATH:-}"

INPUT_JSONL="${INPUT_JSONL:-${PROJECT_ROOT}/outputs/qa_mcq.intermediate.jsonl}"
INPUT_KIND="${INPUT_KIND:-intermediate}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/grpo_judge_reward_v2_score_${SLURM_JOB_ID}}"

mkdir -p "${PROJECT_ROOT}/logs" "${OUTPUT_DIR}" "${HOME}" "${XDG_CACHE_HOME}"
cd "${PROJECT_ROOT}"
source "${ENV_ROOT}/bin/activate"

python -m grpo_judge_reward.cli score \
  --input "${INPUT_JSONL}" \
  --input-kind "${INPUT_KIND}" \
  --output-dir "${OUTPUT_DIR}"

echo "${OUTPUT_DIR}" > "${PROJECT_ROOT}/outputs/latest_grpo_judge_reward_v2_score_output.txt"
echo "JUDGE_REWARD_V2_SCORE_OUTPUT=${OUTPUT_DIR}"
