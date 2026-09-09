#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_packets100
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

WRAPPER_STAGE="production_wrapper_bootstrap"
wrapper_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'stage=%s status=failed timestamp=%s exit_code=%s line=%s command=%q\n' \
    "${WRAPPER_STAGE}" "$(date --iso-8601=seconds)" "${exit_code}" \
    "${line_number}" "${failed_command}" >&2
  exit "${exit_code}"
}
trap 'wrapper_error "$LINENO" "$BASH_COMMAND"' ERR
printf 'stage=%s status=started timestamp=%s job_id=%s host=%s\n' \
  "${WRAPPER_STAGE}" "$(date --iso-8601=seconds)" \
  "${SLURM_JOB_ID:-unknown}" "$(hostname)"

# Mine exactly 100 six-user evidence packets and run the feedback-driven QA
# loop for up to three attempts per packet. QA acceptance may be partial;
# packet materialization and packet-level generation coverage may not.
RUN_MODE="six_user_qa_packets_100"
ACCEPTED_TARGET="100"
EVIDENCE_TARGET="100"
MAX_GROUPS="800"
MAX_ATTEMPTS="3"
ALLOW_PARTIAL="1"
export RUN_MODE ACCEPTED_TARGET EVIDENCE_TARGET MAX_GROUPS MAX_ATTEMPTS ALLOW_PARTIAL

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
OUTPUT_ROOT="${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/six_user_qa"
export PROJECT_ROOT PACKAGE_ROOT OUTPUT_ROOT

WRAPPER_STAGE="handoff_to_six_user_runtime"
printf 'stage=%s status=started timestamp=%s runtime=%s\n' \
  "${WRAPPER_STAGE}" "$(date --iso-8601=seconds)" \
  "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"
bash "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"
WRAPPER_STAGE="production_wrapper_complete"
printf 'stage=%s status=completed timestamp=%s\n' \
  "${WRAPPER_STAGE}" "$(date --iso-8601=seconds)"
