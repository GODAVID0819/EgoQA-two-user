#!/usr/bin/env bash
#SBATCH --job-name=egolife_sampn50
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"

# Reuse the existing CLIP-pruned two-user evidence packets.
EVIDENCE_ROOT="${EGOLIFE2U_EVIDENCE_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${EVIDENCE_ROOT}/evidence_pruned_pairs.jsonl}"

OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_sampling_neutral_pf_50}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-50}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
GENERATION_MODE="${GENERATION_MODE:-baseline}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"

SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.9}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-}"

CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-1}"
CUDA_KEEPER_PID=""

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
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "evidence_root=${EVIDENCE_ROOT}"
echo "evidence_path=${EVIDENCE_PATH}"
echo "outdir=${OUTDIR}"
echo "cache_dir=${CACHE_DIR}"
echo "target_count=${TARGET_COUNT}"
echo "model_id=${QWEN_MODEL_ID}"
echo "generation_mode=${GENERATION_MODE}"
echo "question_types=neutral"
echo "generator_decode_mode=sampling"
echo "review_decode_mode=greedy"
echo "sampling_temperature=${SAMPLING_TEMPERATURE}"
echo "sampling_top_p=${SAMPLING_TOP_P}"
echo "sampling_top_k=${SAMPLING_TOP_K}"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "disable_thinking=${DISABLE_THINKING}"
echo "resume_generation=${RESUME_GENERATION}"
echo "media_routing=generator_uses_existing_pruned_videos;judges_and_answerability_use_full_30s_originals"
echo "judge_contract=historical_status_based_PASS_FAIL_rubrics"
echo "entropy_contract=PASS_FAIL_status_logits_are_external_diagnostics_only"
echo "entropy_effect_on_gates=none"
python --version
nvidia-smi || true

echo "stage=preflight_pass_fail_status_entropy"
python - <<'PY'
import math

from egolife_two_user_qa.qwen3vl_runner import (
    DEFAULT_CHOICE_FIELD,
    DEFAULT_DECISION_CHOICES,
    Qwen3VLTransformersRunner,
)
from egolife_two_user_qa.video_qa_loop import decision_uncertainty_from_choice_logits

if not hasattr(Qwen3VLTransformersRunner, "generate_with_choice_logits"):
    raise SystemExit("Qwen runner lacks generate_with_choice_logits; sync the current code")
if DEFAULT_CHOICE_FIELD != "status" or DEFAULT_DECISION_CHOICES != ("PASS", "FAIL"):
    raise SystemExit(
        "judge entropy must capture checks.<judge>.status logits for literal PASS and FAIL"
    )

binary = decision_uncertainty_from_choice_logits(
    {
        "available": True,
        "choice_logits": {"PASS": 0.0, "FAIL": 0.0},
        "generated_choice": "PASS",
    }
)
if binary.get("available") is not True:
    raise SystemExit(f"PASS/FAIL entropy preflight unavailable: {binary}")
if not math.isclose(float(binary["entropy_nats"]), math.log(2.0), rel_tol=1e-7):
    raise SystemExit(f"uniform PASS/FAIL entropy should equal ln(2): {binary}")
print("pass_fail_status_entropy_preflight=ok")
PY

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  if [[ -s "${CUDA_KEEPER_SCRIPT}" ]]; then
    echo "stage=preflight_cuda_keeper_imports"
    if ! python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml", "psutil") if importlib.util.find_spec(name) is None]
if missing:
    print("missing=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_imports=ok")
PY
    then
      if [[ "${CUDA_KEEPER_AUTO_INSTALL}" == "1" ]]; then
        echo "stage=install_cuda_keeper_deps"
        python -m pip install nvidia-ml-py3 psutil || true
      fi
    fi

    if python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_required_after_install=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_required_imports=ok")
PY
    then
      echo "stage=start_cuda_keeper"
      python "${CUDA_KEEPER_SCRIPT}" \
        --threshold "${CUDA_KEEPER_THRESHOLD}" \
        --gpus "${CUDA_KEEPER_GPUS}" \
        --reserve "${CUDA_KEEPER_RESERVE}" \
        > "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log" 2>&1 &
      CUDA_KEEPER_PID=$!
      sleep 2
      if kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
        echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"
      else
        echo "warning=cuda_keeper_exited_early; continuing without keeper"
        CUDA_KEEPER_PID=""
      fi
    else
      echo "warning=cuda_keeper_imports_missing; continuing without keeper"
    fi
  else
    echo "warning=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}; continuing without keeper"
  fi
fi

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
  --evidence "${EVIDENCE_PATH}" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local \
  --model-id "${QWEN_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generation-mode "${GENERATION_MODE}" \
  --fixed-question-type-schedule \
  --question-types neutral \
  "${sampling_args[@]}" \
  "${thinking_args[@]}" \
  "${resume_args[@]}"

echo "stage=verify_generation"
python - "${EVIDENCE_PATH}" "${OUTDIR}" "${TARGET_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
target = int(sys.argv[3])

def read_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

evidence_count = len(read_rows(evidence_path))
accepted = read_rows(output_dir / "qa_mcq.jsonl")
rejected = read_rows(output_dir / "qa_mcq.rejected.jsonl")
prompt_rows = read_rows(output_dir / "video_first_prompts.jsonl")
generation_prompts = [row for row in prompt_rows if row.get("stage") == "generation"]

for row in generation_prompts:
    mode = (row.get("generator_decode") or {}).get("mode")
    if mode != "sampling":
        raise SystemExit(f"non-sampling generator trace found: {mode}")
    if row.get("question_type") != "neutral":
        raise SystemExit(f"non-neutral generator trace found: {row.get('question_type')}")

print(f"trace_check evidence_count={evidence_count}")
print(f"trace_check generation_prompt_count={len(generation_prompts)}")
print(f"trace_check accepted_count={len(accepted)}")
print(f"trace_check rejected_count={len(rejected)}")

if evidence_count < target:
    raise SystemExit(f"expected at least {target} evidence packets, found {evidence_count}")
if len(generation_prompts) < target:
    raise SystemExit(f"expected at least {target} generation prompts, found {len(generation_prompts)}")
if len(accepted) + len(rejected) < target:
    raise SystemExit(
        f"expected at least {target} final accepted/rejected rows, found {len(accepted) + len(rejected)}"
    )
PY

echo "stage=validate_outputs"
python -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review

echo "outputs:"
echo "  qa=${OUTDIR}/qa_mcq.jsonl"
echo "  intermediate=${OUTDIR}/qa_mcq.intermediate.jsonl"
echo "  rejected=${OUTDIR}/qa_mcq.rejected.jsonl"
echo "  prompts=${OUTDIR}/video_first_prompts.jsonl"
echo "  report=${OUTDIR}/generation_report.md"
echo "date_end=$(date --iso-8601=seconds)"
