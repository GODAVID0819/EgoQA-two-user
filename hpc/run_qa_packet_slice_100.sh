#!/usr/bin/env bash

set -euo pipefail

: "${PACKET_START_INDEX:?PACKET_START_INDEX must be set to 0, 100, or 200}"
: "${BATCH_LABEL:?BATCH_LABEL must be set}"

case "${PACKET_START_INDEX}" in
  0|100|200) ;;
  *)
    echo "error=invalid_packet_start_index value=${PACKET_START_INDEX}" >&2
    exit 2
    ;;
esac

PACKET_COUNT=100
PACKET_END_INDEX=$((PACKET_START_INDEX + PACKET_COUNT - 1))

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${EGOLIFE2U_PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa}"
if [[ ! -d "${PACKAGE_ROOT}" ]]; then
  PACKAGE_ROOT="${PROJECT_ROOT}"
fi

EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${PACKAGE_ROOT}/outputs/clip_pruned_packets_300_mixed_temporal/evidence_pruned_pairs.jsonl}"
OUTPUT_ROOT="${EGOLIFE2U_OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs/qa_300_three_jobs_current_prompts}"
OUTDIR="${OUTPUT_ROOT}/${BATCH_LABEL}"
SLICE_PATH="${OUTDIR}/evidence_slice.jsonl"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-262144}"
GENERATION_MODE="${GENERATION_MODE:-baseline}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"
SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.9}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-}"

if [[ "${GENERATION_MODE}" != "baseline" ]]; then
  echo "error=only_baseline_generation_mode_is_active requested=${GENERATION_MODE}" >&2
  exit 2
fi

CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PACKAGE_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_PID=""

ENV_SCRIPT="${PROJECT_ROOT}/hpc/env_qwen3vl.sh"
if [[ ! -s "${ENV_SCRIPT}" ]]; then
  ENV_SCRIPT="${PACKAGE_ROOT}/hpc/env_qwen3vl.sh"
fi

export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    echo "stage=stop_cuda_keeper pid=${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}" "/scratch/${USER}/hf_cache"

if [[ ! -s "${ENV_SCRIPT}" ]]; then
  echo "error=environment_script_missing path=${ENV_SCRIPT}" >&2
  exit 1
fi
source "${ENV_SCRIPT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "date_start=$(date --iso-8601=seconds)"
echo "batch_label=${BATCH_LABEL}"
echo "source_packet_indices_zero_based=${PACKET_START_INDEX}-${PACKET_END_INDEX}"
echo "source_packet_numbers_one_based=$((PACKET_START_INDEX + 1))-$((PACKET_END_INDEX + 1))"
echo "packet_count=${PACKET_COUNT}"
echo "evidence_path=${EVIDENCE_PATH}"
echo "slice_path=${SLICE_PATH}"
echo "outdir=${OUTDIR}"
echo "generator_backend=transformers-local"
echo "generator_model_id=${QWEN_MODEL_ID}"
echo "judge_topology=shared_generator_model_with_greedy_pass_fail_judges"
echo "generation_mode=${GENERATION_MODE}"
echo "question_types=neutral"
echo "generator_decode=sampling temperature=${SAMPLING_TEMPERATURE} top_p=${SAMPLING_TOP_P} top_k=${SAMPLING_TOP_K:-unset}"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "disable_thinking=${DISABLE_THINKING}"
echo "resume_generation=${RESUME_GENERATION}"
echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD} gpus=${CUDA_KEEPER_GPUS} reserve_gib=${CUDA_KEEPER_RESERVE}"
python --version
nvidia-smi || true

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi
if [[ ! -s "${CUDA_KEEPER_SCRIPT}" ]]; then
  echo "error=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}" >&2
  exit 1
fi

echo "stage=preflight_cuda_keeper_imports"
if ! python - <<'PY'
import importlib.util

missing = [name for name in ("torch", "pynvml", "psutil") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_cuda_keeper_dependencies=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_imports=ok")
PY
then
  if [[ "${CUDA_KEEPER_AUTO_INSTALL}" == "1" ]]; then
    echo "stage=install_cuda_keeper_dependencies"
    python -m pip install nvidia-ml-py3 psutil
  fi
fi

python - <<'PY'
import importlib.util

missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("cuda keeper dependencies remain missing: " + ",".join(missing))
print("cuda_keeper_required_imports=ok")
PY

echo "stage=start_cuda_keeper"
python "${CUDA_KEEPER_SCRIPT}" \
  --threshold "${CUDA_KEEPER_THRESHOLD}" \
  --gpus "${CUDA_KEEPER_GPUS}" \
  --reserve "${CUDA_KEEPER_RESERVE}" \
  > "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log" 2>&1 &
CUDA_KEEPER_PID=$!
sleep 2
if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
  echo "error=cuda_keeper_exited_early log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log" >&2
  wait "${CUDA_KEEPER_PID}" || true
  CUDA_KEEPER_PID=""
  exit 1
fi
echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"

echo "stage=materialize_exclusive_evidence_slice"
python - \
  "${EVIDENCE_PATH}" \
  "${SLICE_PATH}" \
  "${OUTDIR}/slice_manifest.json" \
  "${PACKET_START_INDEX}" \
  "${PACKET_COUNT}" \
  "${BATCH_LABEL}" <<'PY'
import json
import sys
from pathlib import Path

source_path = Path(sys.argv[1])
slice_path = Path(sys.argv[2])
manifest_path = Path(sys.argv[3])
start = int(sys.argv[4])
count = int(sys.argv[5])
batch_label = sys.argv[6]

with source_path.open("r", encoding="utf-8") as handle:
    source = [json.loads(line) for line in handle if line.strip()]

if len(source) != 300:
    raise SystemExit(f"expected exactly 300 source evidence pairs, found {len(source)}")
source_ids = [str(row.get("evidence_id") or "") for row in source]
if any(not evidence_id for evidence_id in source_ids):
    raise SystemExit("every source evidence row must have an evidence_id")
if len(source_ids) != len(set(source_ids)):
    raise SystemExit("source evidence contains duplicate evidence_id values")

selected = source[start : start + count]
if len(selected) != count:
    raise SystemExit(f"slice {start}:{start + count} produced {len(selected)} rows")

slice_path.parent.mkdir(parents=True, exist_ok=True)
with slice_path.open("w", encoding="utf-8") as handle:
    for row in selected:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

manifest = {
    "batch_label": batch_label,
    "source_evidence_path": str(source_path),
    "source_evidence_count": len(source),
    "source_start_index_zero_based": start,
    "source_end_index_zero_based": start + count - 1,
    "source_start_number_one_based": start + 1,
    "source_end_number_one_based": start + count,
    "slice_count": len(selected),
    "evidence_ids": [row["evidence_id"] for row in selected],
}
manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"source_evidence_count={len(source)}")
print(f"slice_count={len(selected)}")
print(f"slice_first_evidence_id={selected[0]['evidence_id']}")
print(f"slice_last_evidence_id={selected[-1]['evidence_id']}")
PY

resume_args=()
if [[ "${RESUME_GENERATION}" == "1" ]]; then
  resume_args+=(--resume)
fi

thinking_args=()
if [[ "${DISABLE_THINKING}" == "1" ]]; then
  thinking_args+=(--disable-thinking)
fi

sampling_args=(
  --generator-decode-mode sampling
  --generator-temperature "${SAMPLING_TEMPERATURE}"
  --generator-top-p "${SAMPLING_TOP_P}"
)
if [[ -n "${SAMPLING_TOP_K}" ]]; then
  sampling_args+=(--generator-top-k "${SAMPLING_TOP_K}")
fi

echo "stage=generate_video_qa_loop"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${SLICE_PATH}" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${PACKET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local \
  --model-id "${QWEN_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS}" \
  --generation-mode "${GENERATION_MODE}" \
  --fixed-question-type-schedule \
  --question-types neutral \
  "${sampling_args[@]}" \
  "${thinking_args[@]}" \
  "${resume_args[@]}"

echo "stage=verify_complete_slice_coverage"
python - \
  "${SLICE_PATH}" \
  "${OUTDIR}/qa_mcq.jsonl" \
  "${OUTDIR}/qa_mcq.rejected.jsonl" \
  "${OUTDIR}/video_first_prompts.jsonl" \
  "${OUTDIR}/generation_summary.json" \
  "${PACKET_COUNT}" \
  "${PACKET_START_INDEX}" \
  "${BATCH_LABEL}" <<'PY'
import json
import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
accepted_path = Path(sys.argv[2])
rejected_path = Path(sys.argv[3])
prompts_path = Path(sys.argv[4])
summary_path = Path(sys.argv[5])
target = int(sys.argv[6])
source_start = int(sys.argv[7])
batch_label = sys.argv[8]

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

evidence = read_jsonl(evidence_path)
accepted = read_jsonl(accepted_path)
rejected = read_jsonl(rejected_path)
prompts = read_jsonl(prompts_path)
generation_prompts = [row for row in prompts if row.get("stage") == "generation"]

if len(evidence) != target:
    raise SystemExit(f"expected {target} evidence rows, found {len(evidence)}")
final = accepted + rejected
if len(final) != target:
    raise SystemExit(f"expected {target} final accepted/rejected rows, found {len(final)}")

evidence_ids = [str(row.get("evidence_id") or "") for row in evidence]
final_ids = [str(row.get("evidence_id") or "") for row in final]
if len(final_ids) != len(set(final_ids)):
    raise SystemExit("final QA outputs contain duplicate evidence_id values")
if set(final_ids) != set(evidence_ids):
    missing = sorted(set(evidence_ids) - set(final_ids))
    unexpected = sorted(set(final_ids) - set(evidence_ids))
    raise SystemExit(f"final QA coverage mismatch missing={missing[:5]} unexpected={unexpected[:5]}")
if len(generation_prompts) < target:
    raise SystemExit(f"expected at least {target} generation prompts, found {len(generation_prompts)}")

summary = {
    "batch_label": batch_label,
    "source_start_index_zero_based": source_start,
    "source_end_index_zero_based": source_start + target - 1,
    "attempted_evidence_pair_count": len(evidence),
    "accepted_qa_count": len(accepted),
    "rejected_qa_count": len(rejected),
    "generation_prompt_count": len(generation_prompts),
    "complete_evidence_coverage": True,
    "unique_final_evidence_id_count": len(set(final_ids)),
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"attempted_evidence_pair_count={len(evidence)}")
print(f"accepted_qa_count={len(accepted)}")
print(f"rejected_qa_count={len(rejected)}")
print(f"unique_final_evidence_id_count={len(set(final_ids))}")
PY

echo "outputs:"
echo "  evidence_slice=${SLICE_PATH}"
echo "  slice_manifest=${OUTDIR}/slice_manifest.json"
echo "  accepted_qa=${OUTDIR}/qa_mcq.jsonl"
echo "  rejected_qa=${OUTDIR}/qa_mcq.rejected.jsonl"
echo "  intermediate=${OUTDIR}/qa_mcq.intermediate.jsonl"
echo "  prompts=${OUTDIR}/video_first_prompts.jsonl"
echo "  generation_summary=${OUTDIR}/generation_summary.json"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log"
echo "date_done=$(date --iso-8601=seconds)"

