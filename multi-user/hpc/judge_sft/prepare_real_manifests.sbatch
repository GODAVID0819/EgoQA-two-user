#!/usr/bin/env bash
#SBATCH --job-name=judge38-data40
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
DATASET_ROOT="${DATASET_ROOT:-/scratch/${USER}/egolife_rlhf_evidence_v1}"
GENERATION_ROOT="${GENERATION_ROOT:-/scratch/${USER}/egolife_rlhf_qa_generation/qwen38_legacy_two_pass_schema_v2}"
LABEL_DIR="${LABEL_DIR:-/scratch/${USER}/egolife_judge_sft_labels}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/${USER}/egolife_judge_sft}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/manifests_40_packets_${SLURM_JOB_ID}}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/job_scratch/judge38_data40_${SLURM_JOB_ID}}"
EXPECTED_CANDIDATES="${EXPECTED_CANDIDATES:-221}"
EXPECTED_PACKETS="${EXPECTED_PACKETS:-40}"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg"
export HF_HOME="${JOB_SCRATCH_ROOT}/huggingface"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/huggingface/datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

PYTHON="${TRAIN_ENV}/bin/python"
CURRENT_STAGE="bootstrap"

write_status() {
  local status="$1"
  local exit_code="${2:-0}"
  if [[ -d "${OUTPUT_DIR}" ]]; then
    printf '{"status":"%s","stage":"%s","job_id":"%s","exit_code":%s}\n' \
      "${status}" "${CURRENT_STAGE}" "${SLURM_JOB_ID}" "${exit_code}" \
      > "${OUTPUT_DIR}/stage_status.json"
  fi
}

on_error() {
  local exit_code=$?
  trap - ERR
  write_status failed "${exit_code}"
  printf 'judge data preparation failed stage=%s exit_code=%s\n' \
    "${CURRENT_STAGE}" "${exit_code}" >&2
  exit "${exit_code}"
}
trap on_error ERR

mkdir -p "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${TMPDIR}"
cd "${PACKAGE_ROOT}"
for required in "${PYTHON}" "${DATASET_ROOT}/packets" "${GENERATION_ROOT}" "${LABEL_DIR}"; do
  [[ -e "${required}" ]] || { echo "Missing required path: ${required}" >&2; exit 2; }
done
mapfile -t LABEL_FILES < <(
  find "${LABEL_DIR}" -maxdepth 1 -type f \
    -name 'six_user_binary_labels_*.jsonl' -print | sort
)
if [[ "${#LABEL_FILES[@]}" != "2" ]]; then
  echo "Expected exactly two completed label JSONLs in ${LABEL_DIR}; found ${#LABEL_FILES[@]}." >&2
  exit 2
fi

CURRENT_STAGE="storage_preflight"
"${PYTHON}" -P -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"

CURRENT_STAGE="join_labels_generation_and_frames"
"${PYTHON}" -P -m training.judge_sft.prepare_real_data \
  --labels "${LABEL_FILES[@]}" \
  --generation-root "${GENERATION_ROOT}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-candidates "${EXPECTED_CANDIDATES}" \
  --expected-packets "${EXPECTED_PACKETS}" \
  --contradiction-policy exclude-task \
  > "${OUTPUT_DIR}/prepare_stdout.json"

CURRENT_STAGE="postcondition"
for required in train.jsonl data_audit.json prompt_snapshots.json excluded_task_rows.jsonl; do
  [[ -f "${OUTPUT_DIR}/${required}" ]] || {
    echo "Missing preparation artifact: ${OUTPUT_DIR}/${required}" >&2
    exit 2
  }
done
"${PYTHON}" -P -m training.judge_sft.select_smoke_example \
  --manifest "${OUTPUT_DIR}/train.jsonl" \
  --output "${OUTPUT_DIR}/selected_smoke_example.json" \
  > "${OUTPUT_DIR}/selected_smoke_example_id.txt"

CURRENT_STAGE="complete"
write_status passed 0
printf 'judge_data_preparation_status=passed output=%s\n' "${OUTPUT_DIR}"
