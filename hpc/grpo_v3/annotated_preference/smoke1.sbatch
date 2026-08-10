#!/usr/bin/env bash
#SBATCH --job-name=pareto-dpo-smoke1
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-smoke1-%j.out
#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo/logs/pareto-dpo-smoke1-%j.err
set -euo pipefail

MODE=smoke
export MODE
PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
source "${PROJECT_ROOT}/hpc/grpo_v3/annotated_preference/common.sh"
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
  --per_device_train_batch_size 1
  --gradient_accumulation_steps 1
  --learning_rate 1e-5
  --beta 0.1
  --lr_scheduler_type constant
  --max_steps 1
  --dataset_shuffle false
  --seed 42
  --data_seed 42
  --eval_strategy steps
  --eval_steps 1
  --save_strategy steps
  --save_steps 1
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
  --mode smoke --output "${OUTDIR}/dpo_gate_result.json"
