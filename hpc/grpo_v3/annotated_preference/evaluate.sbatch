#!/usr/bin/env bash
#SBATCH --job-name=pareto-dpo-validation
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-validation-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-validation-%j.err
set -euo pipefail

MODE=validation
ADAPTER_DIR="${ADAPTER_DIR:?submit with an explicit train_JOBID/adapter directory}"
TRAIN_JOB_ID="${TRAIN_JOB_ID:?submit with the exact training JobID}"
export MODE
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/annotated_preference}
OUTDIR="${OUTPUT_ROOT}/validation_${SLURM_JOB_ID}"
export OUTDIR
source "${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/common.sh"

test -s "${ADAPTER_DIR}/adapter_config.json"
test -s "${DPO_DATA_DIR}/validation_dpo.jsonl"
cp "${DPO_DATA_DIR}/dataset_manifest.json" "${OUTDIR}/dataset_manifest.json"
printf '%s\n' "${TRAIN_JOB_ID}" > "${OUTDIR}/train_job_id.txt"
SWIFT_OUTPUT=${JOB_SCRATCH_ROOT}/swift_output

CMD=(swift rlhf
  --rlhf_type dpo
  --model "${MODEL_DIR}"
  --adapters "${ADAPTER_DIR}"
  --dataset "${DPO_DATA_DIR}/validation_dpo.jsonl"
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
  --per_device_eval_batch_size 1
  --learning_rate 1e-5
  --beta 0.1
  --dataset_shuffle false
  --seed 42
  --data_seed 42
  --do_train false
  --do_eval true
  --output_dir "${SWIFT_OUTPUT}"
  --report_to none)
save_resolved_command "${OUTDIR}/resolved_command.txt" "${CMD[@]}"
env | sort > "${OUTDIR}/environment.txt"
"${CMD[@]}"

STATE=$(find "${SWIFT_OUTPUT}" -type f -name trainer_state.json -print | sort | tail -n 1)
test -n "${STATE}"
cp "${STATE}" "${OUTDIR}/trainer_state.json"
cp -a "${ADAPTER_DIR}" "${OUTDIR}/adapter"
write_parameter_audit "${OUTDIR}/adapter"
"${PYTHON}" - "${OUTDIR}/trainer_state.json" "${DPO_DATA_DIR}/validation_dpo.jsonl" <<'PY'
import json
import sys
from pathlib import Path

state_path, dataset_path = map(Path, sys.argv[1:])
state = json.loads(state_path.read_text(encoding="utf-8"))
with dataset_path.open("r", encoding="utf-8") as handle:
    state["eval_pair_count"] = sum(1 for line in handle if line.strip())
state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
PY
printf '%s\n' '{"reported_metrics":["fixed_pair_accuracy","chosen_rejected_margin","eval_loss"]}' \
  > "${OUTDIR}/validation_metrics_contract.json"
"${PYTHON}" -m training.grpo_v3.experiments.annotated_preference.analyze \
  --trainer-state "${OUTDIR}/trainer_state.json" \
  --parameter-audit "${OUTDIR}/parameter_audit.json" \
  --dataset-manifest "${OUTDIR}/dataset_manifest.json" \
  --mode validation --output "${OUTDIR}/dpo_gate_result.json"
