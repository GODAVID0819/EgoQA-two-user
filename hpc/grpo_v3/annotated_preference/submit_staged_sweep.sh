#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
TRAIN_ENV=${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/annotated_preference}
PYTHON=${TRAIN_ENV}/bin/python
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/staged_train.sbatch
LEARNING_RATES="3e-5 6e-5 1e-4"
STEPS_PER_EPOCH=66

OVERFIT_RESULT=$("${PYTHON}" - "${OUTPUT_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

candidates = []
for path in Path(sys.argv[1]).glob("overfit_*/dpo_gate_result.json"):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if result.get("status") == "passed" and result.get("mode") == "overfit":
        candidates.append((path.stat().st_mtime_ns, path))
if candidates:
    print(max(candidates)[1])
PY
)

test -x "${PYTHON}"
test -f "${SCRIPT}"
test -s "${OVERFIT_RESULT}"
mkdir -p "${OUTPUT_ROOT}"

SWEEP_TAG=$(date +%Y%m%d_%H%M%S)
SWEEP_DIR=${OUTPUT_ROOT}/staged_sweep_${SWEEP_TAG}
JOBS=${SWEEP_DIR}/jobs.tsv
mkdir -p "${SWEEP_DIR}"
printf 'learning_rate\ttarget_epoch\tinitial_step\ttarget_step\tjob_id\tdependency_job_id\tresume_checkpoint\toutput_dir\tsubmitted_at\n' > "${JOBS}"

for LR in ${LEARNING_RATES}; do
  PREVIOUS_JOB=
  PREVIOUS_CHECKPOINT=
  for TARGET_EPOCH in 1 2 3; do
    INITIAL_STEP=$(( (TARGET_EPOCH - 1) * STEPS_PER_EPOCH ))
    TARGET_GLOBAL_STEP=$(( TARGET_EPOCH * STEPS_PER_EPOCH ))
    LR_TAG=${LR//-/m}
    DEPENDENCY_ARGS=()
    if [ -n "${PREVIOUS_JOB}" ]; then
      DEPENDENCY_ARGS=(--dependency=afterok:${PREVIOUS_JOB})
    fi
    JOB_RAW=$(sbatch --parsable \
      "${DEPENDENCY_ARGS[@]}" \
      --job-name="pareto-${LR_TAG}-e${TARGET_EPOCH}" \
      --export=ALL,OVERFIT_RESULT="${OVERFIT_RESULT}",LEARNING_RATE="${LR}",TARGET_EPOCH="${TARGET_EPOCH}",EXPECTED_INITIAL_STEP="${INITIAL_STEP}",TARGET_GLOBAL_STEP="${TARGET_GLOBAL_STEP}",RESUME_CHECKPOINT="${PREVIOUS_CHECKPOINT}" \
      --chdir="${PROJECT_ROOT}" "${SCRIPT}")
    JOB_ID=${JOB_RAW%%;*}
    [[ "${JOB_ID}" =~ ^[0-9]+$ ]]
    OUTPUT_DIR=${OUTPUT_ROOT}/staged_${JOB_ID}
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${LR}" "${TARGET_EPOCH}" "${INITIAL_STEP}" "${TARGET_GLOBAL_STEP}" \
      "${JOB_ID}" "${PREVIOUS_JOB}" "${PREVIOUS_CHECKPOINT}" "${OUTPUT_DIR}" \
      "$(date -Iseconds)" >> "${JOBS}"
    PREVIOUS_JOB=${JOB_ID}
    PREVIOUS_CHECKPOINT=${OUTPUT_DIR}/checkpoint
  done
done

ROW_COUNT=$(awk -F '\t' 'NR > 1 {count++} END {print count+0}' "${JOBS}")
test "${ROW_COUNT}" -eq 9
printf '%s\n' "${JOBS}" > "${OUTPUT_ROOT}/active_staged_sweep_manifest.txt"
printf 'sweep_tag\tcreated_at\tjobs_manifest\n%s\t%s\t%s\n' \
  "${SWEEP_TAG}" "$(date -Iseconds)" "${JOBS}" > "${SWEEP_DIR}/sweep_manifest.tsv"
echo "STAGED_SWEEP_SUBMISSION_PASSED count=9"
echo "ACTIVE_MANIFEST=${OUTPUT_ROOT}/active_staged_sweep_manifest.txt"
column -t -s $'\t' "${JOBS}"
