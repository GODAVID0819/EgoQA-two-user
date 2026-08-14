#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
TRAIN_ENV=${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/annotated_preference}
PYTHON=${TRAIN_ENV}/bin/python
SCRIPT=${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/staged_train.sbatch
ACTIVE_POINTER=${OUTPUT_ROOT}/active_staged_recovery_manifest.txt

# learning_rate parent_epoch1_job failed_epoch2_job cancelled_epoch3_job
RECOVERY_SPECS=$(cat <<'EOF'
3e-5 15675190 15675191 15675192
6e-5 15675193 15675194 15675195
1e-4 15675196 15675197 15675198
EOF
)

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

PREVIOUS_RECOVERY_MANIFEST=
if [ -s "${ACTIVE_POINTER}" ]; then
  EXISTING_MANIFEST=$(head -n 1 "${ACTIVE_POINTER}")
  if [ -s "${EXISTING_MANIFEST}" ]; then
    EXISTING_JOB_IDS=$(awk -F '\t' 'NR > 1 && $5 ~ /^[0-9]+$/ {print $5}' \
      "${EXISTING_MANIFEST}" | sort -u | paste -sd, -)
    EXISTING_JOB_COUNT=$(awk -F '\t' 'NR > 1 && $5 ~ /^[0-9]+$/ {seen[$5]=1} END {print length(seen)}' \
      "${EXISTING_MANIFEST}")
    TERMINAL_JOB_COUNT=0
    if [ -n "${EXISTING_JOB_IDS}" ]; then
      TERMINAL_JOB_COUNT=$(sacct -n -X -j "${EXISTING_JOB_IDS}" \
        --format=JobIDRaw,State --parsable2 \
        | awk -F '|' '
            $1 ~ /^[0-9]+$/ {
              state=$2
              sub(/[ +].*$/, "", state)
              if (state ~ /^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED|BOOT_FAIL|DEADLINE|REVOKED|SPECIAL_EXIT)$/) {
                seen[$1]=1
              }
            }
            END {print length(seen)}')
    fi
    if [ "${EXISTING_JOB_COUNT}" -eq 0 ] || \
       [ "${TERMINAL_JOB_COUNT}" -ne "${EXISTING_JOB_COUNT}" ]; then
      echo "STOP: previous staged recovery still has non-terminal jobs: ${EXISTING_MANIFEST}"
      false
    fi
    PREVIOUS_RECOVERY_MANIFEST=${EXISTING_MANIFEST}
    echo "PREVIOUS_RECOVERY_MANIFEST=${PREVIOUS_RECOVERY_MANIFEST}"
  fi
fi

READY=1
while read -r LR PARENT_JOB FAILED_E2 CANCELLED_E3; do
  PARENT_CHECKPOINT=${OUTPUT_ROOT}/staged_${PARENT_JOB}/checkpoint
  for name in adapter_config.json optimizer.pt scheduler.pt trainer_state.json; do
    if [ ! -s "${PARENT_CHECKPOINT}/${name}" ]; then
      echo "MISSING: ${PARENT_CHECKPOINT}/${name}"
      READY=0
    fi
  done
  if [ -s "${PARENT_CHECKPOINT}/trainer_state.json" ]; then
    if ! "${PYTHON}" - "${PARENT_CHECKPOINT}/trainer_state.json" "${PARENT_JOB}" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if state.get("global_step") != 66:
    raise SystemExit(
        f"parent job {sys.argv[2]} global_step={state.get('global_step')} expected=66"
    )
PY
    then
      READY=0
    fi
  else
    READY=0
  fi
done <<< "${RECOVERY_SPECS}"

if [ "${READY}" -ne 1 ]; then
  echo "STOP: recovery preflight failed; no jobs were submitted"
  false
fi

RECOVERY_TAG=$(date +%Y%m%d_%H%M%S)
RECOVERY_DIR=${OUTPUT_ROOT}/staged_recovery_${RECOVERY_TAG}
JOBS=${RECOVERY_DIR}/jobs.tsv
mkdir -p "${RECOVERY_DIR}"
printf 'learning_rate\ttarget_epoch\tinitial_step\ttarget_step\tjob_id\tdependency_job_id\tresume_checkpoint\toutput_dir\tsubmitted_at\treplaces_job_id\n' > "${JOBS}"

while read -r LR PARENT_JOB FAILED_E2 CANCELLED_E3; do
  LR_TAG=${LR//-/m}
  PARENT_CHECKPOINT=${OUTPUT_ROOT}/staged_${PARENT_JOB}/checkpoint
  REPLACED_E2=${FAILED_E2}
  REPLACED_E3=${CANCELLED_E3}
  if [ -n "${PREVIOUS_RECOVERY_MANIFEST}" ]; then
    PREVIOUS_E2=$(awk -F '\t' -v lr="${LR}" \
      'NR > 1 && $1 == lr && $2 == 2 {job=$5} END {print job}' \
      "${PREVIOUS_RECOVERY_MANIFEST}")
    PREVIOUS_E3=$(awk -F '\t' -v lr="${LR}" \
      'NR > 1 && $1 == lr && $2 == 3 {job=$5} END {print job}' \
      "${PREVIOUS_RECOVERY_MANIFEST}")
    if [ -n "${PREVIOUS_E2}" ]; then
      REPLACED_E2=${PREVIOUS_E2}
    fi
    if [ -n "${PREVIOUS_E3}" ]; then
      REPLACED_E3=${PREVIOUS_E3}
    fi
  fi

  E2_RAW=$(sbatch --parsable \
    --job-name="pareto-recovery-${LR_TAG}-e2" \
    --export=ALL,OVERFIT_RESULT="${OVERFIT_RESULT}",LEARNING_RATE="${LR}",TARGET_EPOCH=2,EXPECTED_INITIAL_STEP=66,TARGET_GLOBAL_STEP=132,RESUME_CHECKPOINT="${PARENT_CHECKPOINT}" \
    --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  E2_JOB=${E2_RAW%%;*}
  [[ "${E2_JOB}" =~ ^[0-9]+$ ]]
  E2_OUTPUT=${OUTPUT_ROOT}/staged_${E2_JOB}
  printf '%s\t2\t66\t132\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${LR}" "${E2_JOB}" "${PARENT_JOB}" "${PARENT_CHECKPOINT}" \
    "${E2_OUTPUT}" "$(date -Iseconds)" "${REPLACED_E2}" >> "${JOBS}"

  E3_RESUME=${E2_OUTPUT}/checkpoint
  E3_RAW=$(sbatch --parsable \
    --dependency=afterok:${E2_JOB} \
    --job-name="pareto-recovery-${LR_TAG}-e3" \
    --export=ALL,OVERFIT_RESULT="${OVERFIT_RESULT}",LEARNING_RATE="${LR}",TARGET_EPOCH=3,EXPECTED_INITIAL_STEP=132,TARGET_GLOBAL_STEP=198,RESUME_CHECKPOINT="${E3_RESUME}" \
    --chdir="${PROJECT_ROOT}" "${SCRIPT}")
  E3_JOB=${E3_RAW%%;*}
  [[ "${E3_JOB}" =~ ^[0-9]+$ ]]
  E3_OUTPUT=${OUTPUT_ROOT}/staged_${E3_JOB}
  printf '%s\t3\t132\t198\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${LR}" "${E3_JOB}" "${E2_JOB}" "${E3_RESUME}" \
    "${E3_OUTPUT}" "$(date -Iseconds)" "${REPLACED_E3}" >> "${JOBS}"
done <<< "${RECOVERY_SPECS}"

ROW_COUNT=$(awk -F '\t' 'NR > 1 {count++} END {print count+0}' "${JOBS}")
test "${ROW_COUNT}" -eq 6
printf '%s\n' "${JOBS}" > "${ACTIVE_POINTER}"
printf 'recovery_tag\tcreated_at\tjobs_manifest\n%s\t%s\t%s\n' \
  "${RECOVERY_TAG}" "$(date -Iseconds)" "${JOBS}" > "${RECOVERY_DIR}/recovery_manifest.tsv"
echo "RECOVERY_SUBMISSION_PASSED count=6"
echo "ACTIVE_RECOVERY_MANIFEST=${ACTIVE_POINTER}"
column -t -s $'\t' "${JOBS}"
