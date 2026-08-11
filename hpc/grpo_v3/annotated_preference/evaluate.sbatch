#!/usr/bin/env bash
#SBATCH --job-name=pareto-dpo-validation
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
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

TRAIN_DIR=$(dirname "${ADAPTER_DIR}")
SOURCE_STATE="${TRAIN_DIR}/trainer_state.json"
SOURCE_GATE="${TRAIN_DIR}/dpo_gate_result.json"
SOURCE_COMMAND="${TRAIN_DIR}/resolved_command.txt"
test -s "${ADAPTER_DIR}/adapter_config.json"
test -s "${SOURCE_STATE}"
test -s "${SOURCE_GATE}"
test -s "${SOURCE_COMMAND}"
test -s "${DPO_DATA_DIR}/validation_dpo.jsonl"
cp "${DPO_DATA_DIR}/dataset_manifest.json" "${OUTDIR}/dataset_manifest.json"
printf '%s\n' "${TRAIN_JOB_ID}" > "${OUTDIR}/train_job_id.txt"
cp "${SOURCE_COMMAND}" "${OUTDIR}/source_train_command.txt"
cp "${SOURCE_STATE}" "${OUTDIR}/trainer_state.json"
cp -a "${ADAPTER_DIR}" "${OUTDIR}/adapter"
write_parameter_audit "${OUTDIR}/adapter"
SOURCE_STATE_SHA=$(sha256sum "${SOURCE_STATE}" | awk '{print $1}')

"${PYTHON}" - "${SOURCE_GATE}" "${SOURCE_COMMAND}" "${DPO_DATA_DIR}" <<'PY'
import json
import shlex
import sys
from pathlib import Path

gate_path, command_path, dataset_dir = map(Path, sys.argv[1:])
gate = json.loads(gate_path.read_text(encoding="utf-8"))
if gate.get("status") != "passed" or gate.get("mode") != "train":
    raise SystemExit("source Gate 4 result must be passed train mode")
tokens = shlex.split(command_path.read_text(encoding="utf-8"))
expected = {
    "--dataset": str(dataset_dir / "train_dpo.jsonl"),
    "--val_dataset": str(dataset_dir / "validation_dpo.jsonl"),
}
for option, value in expected.items():
    try:
        observed = tokens[tokens.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise SystemExit(f"source Gate 4 command is missing {option}") from error
    if observed != value:
        raise SystemExit(f"source Gate 4 command has unexpected {option}: {observed}")
PY

"${PYTHON}" - \
  "${OUTDIR}/trainer_state.json" "${DPO_DATA_DIR}/validation_dpo.jsonl" \
  "${TRAIN_JOB_ID}" "${SOURCE_STATE_SHA}" <<'PY'
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
train_job_id = sys.argv[3]
source_state_sha = sys.argv[4]
state = json.loads(state_path.read_text(encoding="utf-8"))
with dataset_path.open("r", encoding="utf-8") as handle:
    state["eval_pair_count"] = sum(1 for line in handle if line.strip())
state["validation_provenance"] = {
    "evaluation_origin": "gate4_epoch_end",
    "gate5_optimizer_steps": 0,
    "source_train_job_id": train_job_id,
    "source_trainer_state_sha256": source_state_sha,
}
state_path.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")
PY

printf '%s\n' '{"evaluation_origin":"gate4_epoch_end","gate5_optimizer_steps":0,"reported_metrics":["fixed_pair_accuracy","chosen_rejected_margin","eval_loss"]}' \
  > "${OUTDIR}/validation_metrics_contract.json"
CMD=("${PYTHON}" -m training.grpo_v3.experiments.annotated_preference.analyze
  --trainer-state "${OUTDIR}/trainer_state.json"
  --parameter-audit "${OUTDIR}/parameter_audit.json"
  --dataset-manifest "${OUTDIR}/dataset_manifest.json"
  --mode validation --output "${OUTDIR}/dpo_gate_result.json")
save_resolved_command "${OUTDIR}/resolved_command.txt" "${CMD[@]}"
env | sort > "${OUTDIR}/environment.txt"
"${CMD[@]}"

test -s "${OUTDIR}/source_train_command.txt"
test -s "${OUTDIR}/validation_metrics_contract.json"
test -s "${OUTDIR}/dpo_gate_result.json"
