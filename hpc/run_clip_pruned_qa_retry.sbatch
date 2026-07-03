#!/usr/bin/env bash
#SBATCH --job-name=egolife_qa_retry
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${OUTDIR}/evidence_pruned_pairs.jsonl}"

EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-auto}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3-VL-8B-Instruct}"
GENERATION_MODE="${GENERATION_MODE:-baseline}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
RESUME_GENERATION="${RESUME_GENERATION:-1}"

CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/egolife_two_user_qa/hpc/cuda.py}"
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
mkdir -p hpc/logs "${OUTDIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_path_missing_or_empty path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if [[ "${EVAL_PACKET_COUNT}" == "auto" ]]; then
  EVAL_PACKET_COUNT="$(python - "${EVIDENCE_PATH}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open("r", encoding="utf-8") as f:
    print(sum(1 for line in f if line.strip()))
PY
)"
fi

QA_DIR="${OUTDIR}/${GENERATION_MODE}"
mkdir -p "${QA_DIR}"

resume_args=()
if [[ "${RESUME_GENERATION}" == "1" ]]; then
  resume_args+=(--resume)
fi

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "outdir=${OUTDIR}"
echo "evidence_path=${EVIDENCE_PATH}"
echo "qa_dir=${QA_DIR}"
echo "eval_packet_count=${EVAL_PACKET_COUNT}"
echo "qwen_model_id=${QWEN_MODEL_ID}"
echo "generation_mode=${GENERATION_MODE}"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "resume_generation=${RESUME_GENERATION}"
echo "media_routing=generator_uses_existing_pruned_videos;judges_and_answerability_use_full_30s_originals"
echo "cuda_keeper_enable=${CUDA_KEEPER_ENABLE}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
echo "cuda_keeper_threshold=${CUDA_KEEPER_THRESHOLD}"
echo "cuda_keeper_gpus=${CUDA_KEEPER_GPUS}"
echo "cuda_keeper_reserve=${CUDA_KEEPER_RESERVE}"
echo "cuda_keeper_auto_install=${CUDA_KEEPER_AUTO_INSTALL}"
python --version
nvidia-smi || true

if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  if [[ -s "${CUDA_KEEPER_SCRIPT}" ]]; then
    echo "stage=preflight_cuda_keeper_imports"
    if ! python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
optional_missing = [name for name in ("psutil",) if importlib.util.find_spec(name) is None]
if missing or optional_missing:
    print("missing_required=" + ",".join(missing))
    print("missing_optional=" + ",".join(optional_missing))
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
        > "${OUTDIR}/cuda_keeper_qa_retry_${SLURM_JOB_ID:-local}.log" 2>&1 &
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

echo "stage=generate_video_qa_loop_resume"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_PATH}" \
  --output "${QA_DIR}/qa_mcq.jsonl" \
  --prompts-output "${QA_DIR}/video_first_prompts.jsonl" \
  --rejected-output "${QA_DIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${QA_DIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${EVAL_PACKET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local \
  --model-id "${QWEN_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generation-mode "${GENERATION_MODE}" \
  --fixed-question-type-schedule \
  "${resume_args[@]}"

echo "stage=verify_generation_traces"
python - <<PY
import json
from pathlib import Path

target = int("${EVAL_PACKET_COUNT}")
evidence_path = Path("${EVIDENCE_PATH}")
prompts_path = Path("${QA_DIR}/video_first_prompts.jsonl")
intermediate_path = Path("${QA_DIR}/qa_mcq.intermediate.jsonl")
accepted_path = Path("${QA_DIR}/qa_mcq.jsonl")
rejected_path = Path("${QA_DIR}/qa_mcq.rejected.jsonl")

def count_lines(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def count_generation_prompts(path):
    if not path.exists():
        return 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") == "generation":
                total += 1
    return total

evidence_count = count_lines(evidence_path)
generation_prompt_count = count_generation_prompts(prompts_path)
intermediate_count = count_lines(intermediate_path)
accepted_count = count_lines(accepted_path)
rejected_count = count_lines(rejected_path)

print(f"trace_check evidence_count={evidence_count}")
print(f"trace_check generation_prompt_count={generation_prompt_count}")
print(f"trace_check intermediate_count={intermediate_count}")
print(f"trace_check accepted_count={accepted_count}")
print(f"trace_check rejected_count={rejected_count}")

if evidence_count < target:
    raise SystemExit(f"expected at least {target} evidence packets, found {evidence_count}")
if generation_prompt_count < target:
    raise SystemExit(f"expected at least {target} generation prompt traces, found {generation_prompt_count}")
if accepted_count + rejected_count < target:
    raise SystemExit(
        f"expected at least {target} final accepted/rejected QA rows, found {accepted_count + rejected_count}"
    )
PY

echo "stage=validate_outputs"
python -m egolife_two_user_qa validate_outputs \
  --qa "${QA_DIR}/qa_mcq.jsonl" \
  --csv-output "${QA_DIR}/qa_mcq.csv" \
  --human-review-output "${QA_DIR}/human_review_sheet.md" \
  --report "${QA_DIR}/generation_report.md" \
  --strict-review

echo "qa_output=${QA_DIR}/qa_mcq.jsonl"
echo "qa_intermediate=${QA_DIR}/qa_mcq.intermediate.jsonl"
echo "qa_rejected=${QA_DIR}/qa_mcq.rejected.jsonl"
echo "qa_prompts=${QA_DIR}/video_first_prompts.jsonl"
echo "report=${QA_DIR}/generation_report.md"
echo "date_end=$(date --iso-8601=seconds)"
