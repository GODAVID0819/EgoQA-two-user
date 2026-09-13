#!/usr/bin/env bash
#SBATCH --job-name=egoqa-cvr-v3-p120
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3
#SBATCH --constraint=h100
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/grpo-v3-cvr-v3-probe120-%j.out
#SBATCH --error=logs/grpo-v3-cvr-v3-probe120-%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/projects/EgoQA-two-user}"
PYTHON="${TRAIN_ENV:-/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024}/bin/python"
PROBE_DIR="${PROBE_DIR:-}"
[[ -n "${PROBE_DIR}" ]] || { echo "PROBE_DIR is required; run probe20_v3.sbatch first" >&2; exit 2; }
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; assert r["checks"]["convergence_metrics_passed"]; assert r["checks"]["required_global_steps"]; assert r["checks"]["parameter_delta_nonzero"]' "${PROBE_DIR}/training_gate_result.json"

export MODE=probe120
export STEPS=120
export EXPECTED_GROUPS=120
exec bash "${PROJECT_ROOT}/hpc/grpo_v3/qa_cross_view_relation/smoke1_v3.sbatch"
