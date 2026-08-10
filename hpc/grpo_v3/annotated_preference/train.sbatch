#!/usr/bin/env bash
#SBATCH --job-name=pareto-dpo-train
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-train-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-train-%j.err
set -euo pipefail

MODE=train
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-1}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
BETA=${BETA:-0.1}
OVERFIT_RESULT="${OVERFIT_RESULT:?submit with the explicit overfit dpo_gate_result.json path}"
export MODE
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/annotated_preference}
OUTDIR="${OUTPUT_ROOT}/train_${SLURM_JOB_ID}"
export OUTDIR
source "${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/common.sh"

test -s "${OVERFIT_RESULT}"
"${PYTHON}" - "${OVERFIT_RESULT}" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if result.get("status") == "passed":
    print("overfit_gate=passed")
else:
    raise SystemExit("OVERFIT_RESULT status must be passed")
PY
test -s "${DPO_DATA_DIR}/train_dpo.jsonl"
test -s "${DPO_DATA_DIR}/validation_dpo.jsonl"
cp "${DPO_DATA_DIR}/dataset_manifest.json" "${OUTDIR}/dataset_manifest.json"
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
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
  --num_train_epochs "${NUM_TRAIN_EPOCHS}"
  --learning_rate "${LEARNING_RATE}"
  --beta "${BETA}"
  --lr_scheduler_type constant
  --dataset_shuffle false
  --seed 42
  --data_seed 42
  --eval_strategy epoch
  --save_strategy epoch
  --save_total_limit 1
  --logging_steps 1
  --output_dir "${SWIFT_OUTPUT}"
  --report_to none)
save_resolved_command "${OUTDIR}/resolved_command.txt" "${CMD[@]}"
env | sort > "${OUTDIR}/environment.txt"
"${CMD[@]}"
collect_swift_artifacts "${SWIFT_OUTPUT}"
write_parameter_audit "${OUTDIR}/adapter"
"${PYTHON}" -m training.grpo_v3.experiments.annotated_preference.analyze \
  --trainer-state "${OUTDIR}/trainer_state.json" \
  --parameter-audit "${OUTDIR}/parameter_audit.json" \
  --dataset-manifest "${OUTDIR}/dataset_manifest.json" \
  --mode train --output "${OUTDIR}/dpo_gate_result.json"

test -s "${OUTDIR}/resolved_command.txt"
test -s "${OUTDIR}/environment.txt"
test -s "${OUTDIR}/dataset_manifest.json"
test -s "${OUTDIR}/trainer_state.json"
test -s "${OUTDIR}/adapter/adapter_config.json"
test -s "${OUTDIR}/parameter_audit.json"
test -s "${OUTDIR}/dpo_gate_result.json"
