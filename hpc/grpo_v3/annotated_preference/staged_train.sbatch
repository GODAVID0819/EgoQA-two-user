#!/usr/bin/env bash
#SBATCH --job-name=pareto-dpo-stage
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:50:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-stage-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-stage-%j.err
set -euo pipefail

MODE=staged
LEARNING_RATE="${LEARNING_RATE:?missing LEARNING_RATE}"
TARGET_EPOCH="${TARGET_EPOCH:?missing TARGET_EPOCH}"
TARGET_GLOBAL_STEP="${TARGET_GLOBAL_STEP:?missing TARGET_GLOBAL_STEP}"
EXPECTED_INITIAL_STEP="${EXPECTED_INITIAL_STEP:?missing EXPECTED_INITIAL_STEP}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
OVERFIT_RESULT="${OVERFIT_RESULT:?missing OVERFIT_RESULT}"
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/annotated_preference}
OUTDIR="${OUTPUT_ROOT}/staged_${SLURM_JOB_ID}"
export MODE OUTDIR
source "${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/common.sh"

test -s "${OVERFIT_RESULT}"
test -s "${DPO_DATA_DIR}/train_dpo.jsonl"
test -s "${DPO_DATA_DIR}/validation_dpo.jsonl"
cp "${DPO_DATA_DIR}/dataset_manifest.json" "${OUTDIR}/dataset_manifest.json"

if [ "${EXPECTED_INITIAL_STEP}" -eq 0 ]; then
  test -z "${RESUME_CHECKPOINT}"
else
  test -d "${RESUME_CHECKPOINT}"
  for name in adapter_config.json optimizer.pt scheduler.pt trainer_state.json; do
    test -s "${RESUME_CHECKPOINT}/${name}"
  done
  "${PYTHON}" - "${RESUME_CHECKPOINT}/trainer_state.json" "${EXPECTED_INITIAL_STEP}" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if state.get("global_step") != expected:
    raise SystemExit(f"resume global_step={state.get('global_step')} expected={expected}")
print(f"RESUME_CONTRACT_PASSED global_step={expected}")
PY
fi

SWIFT_OUTPUT=${JOB_SCRATCH_ROOT}/swift_output
CMD=(swift rlhf
  --rlhf_type dpo
  --model "${MODEL_DIR}"
  --dataset "${DPO_DATA_DIR}/train_dpo.jsonl"
  --val_dataset "${DPO_DATA_DIR}/validation_dpo.jsonl"
  --tuner_type lora
  --freeze_vit true
  --freeze_aligner true
  --target_modules q_proj v_proj
  --lora_rank 8
  --lora_alpha 16
  --torch_dtype bfloat16
  --max_length 32768
  --gradient_checkpointing true
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 8
  --num_train_epochs 3
  --max_steps "${TARGET_GLOBAL_STEP}"
  --learning_rate "${LEARNING_RATE}"
  --beta 0.1
  --lr_scheduler_type constant
  --dataset_shuffle false
  --seed 42
  --data_seed 42
  --eval_strategy epoch
  --save_strategy epoch
  --save_total_limit 3
  --save_only_model false
  --logging_steps 1
  --output_dir "${SWIFT_OUTPUT}"
  --report_to none)

if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume_from_checkpoint "${RESUME_CHECKPOINT}")
fi

save_resolved_command "${OUTDIR}/resolved_command.txt" "${CMD[@]}"
env | sort > "${OUTDIR}/environment.txt"
set +e
"${CMD[@]}"
SWIFT_EXIT_CODE=$?
set -e
printf '%s\n' "${SWIFT_EXIT_CODE}" > "${OUTDIR}/swift_return_code.txt"

CHECKPOINT_DIR=
if [ -d "${SWIFT_OUTPUT}" ]; then
  CHECKPOINT_DIR=$(find "${SWIFT_OUTPUT}" -type d -name "checkpoint-${TARGET_GLOBAL_STEP}" -print | sort | tail -n 1)
fi
INVENTORY=${OUTDIR}/checkpoint_inventory.txt
printf 'swift_exit_code=%s\ntarget_global_step=%s\ncheckpoint_dir=%s\n' \
  "${SWIFT_EXIT_CODE}" "${TARGET_GLOBAL_STEP}" "${CHECKPOINT_DIR}" > "${INVENTORY}"
if [ -n "${CHECKPOINT_DIR}" ] && [ -d "${CHECKPOINT_DIR}" ]; then
  find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -printf '%s bytes\t%f\n' \
    | sort >> "${INVENTORY}"
else
  echo "MISSING: checkpoint-${TARGET_GLOBAL_STEP}" | tee -a "${INVENTORY}"
fi

PERSISTED_PARTIAL="${OUTDIR}/checkpoint.partial"
CHECKPOINT_READY=1
if [ -n "${CHECKPOINT_DIR}" ] && [ -d "${CHECKPOINT_DIR}" ]; then
  if [ -e "${PERSISTED_PARTIAL}" ] || [ -e "${OUTDIR}/checkpoint" ]; then
    echo "STOP: persistent checkpoint destination already exists" | tee -a "${INVENTORY}"
    CHECKPOINT_READY=0
  else
    if ! cp -a "${CHECKPOINT_DIR}" "${PERSISTED_PARTIAL}"; then
      echo "STOP: failed to persist partial checkpoint" | tee -a "${INVENTORY}"
      CHECKPOINT_READY=0
    fi
  fi
else
  CHECKPOINT_READY=0
fi

for name in adapter_config.json optimizer.pt scheduler.pt trainer_state.json; do
  if [ ! -s "${PERSISTED_PARTIAL}/${name}" ]; then
    echo "MISSING: ${PERSISTED_PARTIAL}/${name}" | tee -a "${INVENTORY}"
    CHECKPOINT_READY=0
  fi
done

if [ -s "${PERSISTED_PARTIAL}/trainer_state.json" ]; then
  if ! "${PYTHON}" - "${PERSISTED_PARTIAL}/trainer_state.json" "${TARGET_GLOBAL_STEP}" <<'PY'
import json
import sys
from pathlib import Path

state = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected = int(sys.argv[2])
if state.get("global_step") != expected:
    raise SystemExit(f"final global_step={state.get('global_step')} expected={expected}")
print(f"STAGE_STEP_CONTRACT_PASSED global_step={expected}")
PY
  then
    echo "STOP: target global_step contract failed" | tee -a "${INVENTORY}"
    CHECKPOINT_READY=0
  fi
else
  CHECKPOINT_READY=0
fi

if [ "${SWIFT_EXIT_CODE}" -ne 0 ]; then
  echo "STOP: swift returned ${SWIFT_EXIT_CODE}; partial checkpoint preserved" | tee -a "${INVENTORY}"
  CHECKPOINT_READY=0
fi

if [ "${CHECKPOINT_READY}" -ne 1 ]; then
  echo "STOP: staged checkpoint contract failed; inspect ${INVENTORY}"
  false
fi

mv "${PERSISTED_PARTIAL}" "${OUTDIR}/checkpoint"
cp "${OUTDIR}/checkpoint/trainer_state.json" "${OUTDIR}/trainer_state.json"
cp -a "${OUTDIR}/checkpoint" "${OUTDIR}/adapter"
write_parameter_audit "${OUTDIR}/adapter"
"${PYTHON}" -m training.grpo_v3.experiments.annotated_preference.analyze \
  --trainer-state "${OUTDIR}/trainer_state.json" \
  --parameter-audit "${OUTDIR}/parameter_audit.json" \
  --dataset-manifest "${OUTDIR}/dataset_manifest.json" \
  --mode train --output "${OUTDIR}/dpo_gate_result.json"

printf '{"learning_rate":"%s","target_epoch":%s,"initial_step":%s,"target_step":%s,"job_id":"%s"}\n' \
  "${LEARNING_RATE}" "${TARGET_EPOCH}" "${EXPECTED_INITIAL_STEP}" \
  "${TARGET_GLOBAL_STEP}" "${SLURM_JOB_ID}" > "${OUTDIR}/stage_contract.json"

test -s "${OUTDIR}/checkpoint/optimizer.pt"
test -s "${OUTDIR}/checkpoint/trainer_state.json"
test -s "${OUTDIR}/dpo_gate_result.json"
