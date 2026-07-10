#!/usr/bin/env bash
#SBATCH --job-name=egolife_decode50
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"

# Reuse existing CLIP-pruned video-pair packets.
EVIDENCE_ROOT="${EGOLIFE2U_EVIDENCE_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${EVIDENCE_ROOT}/evidence_pruned_pairs.jsonl}"

OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_decode_ablation_50}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-50}"
EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${TARGET_COUNT}}"

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
echo "eval_packet_count=${EVAL_PACKET_COUNT}"
echo "qwen_model_id=${QWEN_MODEL_ID}"
echo "generation_mode=${GENERATION_MODE}"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "disable_thinking=${DISABLE_THINKING}"
echo "resume_generation=${RESUME_GENERATION}"
echo "sampling_temperature=${SAMPLING_TEMPERATURE}"
echo "sampling_top_p=${SAMPLING_TOP_P}"
echo "sampling_top_k=${SAMPLING_TOP_K}"
echo "cuda_keeper_enable=${CUDA_KEEPER_ENABLE}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
echo "cuda_keeper_threshold=${CUDA_KEEPER_THRESHOLD}"
echo "cuda_keeper_gpus=${CUDA_KEEPER_GPUS}"
echo "cuda_keeper_reserve=${CUDA_KEEPER_RESERVE}"
echo "cuda_keeper_auto_install=${CUDA_KEEPER_AUTO_INSTALL}"
echo "benchmark_contract=reuse_existing_clip_pruned_packets_compare_greedy_vs_sampling_generator_decode"
echo "media_routing=generator_uses_existing_pruned_videos;judges_and_answerability_use_full_30s_originals"
python --version
nvidia-smi || true

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

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

run_decode_mode() {
  local decode_mode="$1"
  local qa_dir="${OUTDIR}/${GENERATION_MODE}_${decode_mode}"
  mkdir -p "${qa_dir}"

  decode_args=(--generator-decode-mode "${decode_mode}")
  if [[ "${decode_mode}" == "sampling" ]]; then
    decode_args+=(--generator-temperature "${SAMPLING_TEMPERATURE}")
    decode_args+=(--generator-top-p "${SAMPLING_TOP_P}")
    if [[ -n "${SAMPLING_TOP_K}" ]]; then
      decode_args+=(--generator-top-k "${SAMPLING_TOP_K}")
    fi
  fi

  echo "stage=generate_video_qa_loop decode_mode=${decode_mode}"
  python -m egolife_two_user_qa generate_video_qa_loop \
    --evidence "${EVIDENCE_PATH}" \
    --output "${qa_dir}/qa_mcq.jsonl" \
    --prompts-output "${qa_dir}/video_first_prompts.jsonl" \
    --rejected-output "${qa_dir}/qa_mcq.rejected.jsonl" \
    --intermediate-output "${qa_dir}/qa_mcq.intermediate.jsonl" \
    --target-count "${EVAL_PACKET_COUNT}" \
    --max-attempts "${MAX_ATTEMPTS}" \
    --backend transformers-local \
    --model-id "${QWEN_MODEL_ID}" \
    --dtype bfloat16 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --generation-mode "${GENERATION_MODE}" \
    --fixed-question-type-schedule \
    "${decode_args[@]}" \
    "${thinking_args[@]}" \
    "${resume_args[@]}"

  echo "stage=verify_generation_traces decode_mode=${decode_mode}"
  python - "${EVIDENCE_PATH}" "${qa_dir}" "${EVAL_PACKET_COUNT}" "${decode_mode}" <<'PY'
import json
import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
qa_dir = Path(sys.argv[2])
target = int(sys.argv[3])
decode_mode = sys.argv[4]

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
                mode = row.get("generator_decode", {}).get("mode")
                if mode != decode_mode:
                    raise SystemExit(f"decode_mode mismatch in prompt trace: expected {decode_mode}, got {mode}")
    return total

evidence_count = count_lines(evidence_path)
generation_prompt_count = count_generation_prompts(qa_dir / "video_first_prompts.jsonl")
accepted_count = count_lines(qa_dir / "qa_mcq.jsonl")
rejected_count = count_lines(qa_dir / "qa_mcq.rejected.jsonl")

print(f"trace_check decode_mode={decode_mode} evidence_count={evidence_count}")
print(f"trace_check decode_mode={decode_mode} generation_prompt_count={generation_prompt_count}")
print(f"trace_check decode_mode={decode_mode} accepted_count={accepted_count}")
print(f"trace_check decode_mode={decode_mode} rejected_count={rejected_count}")

if evidence_count < target:
    raise SystemExit(f"expected at least {target} evidence packets, found {evidence_count}")
if generation_prompt_count < target:
    raise SystemExit(f"expected at least {target} generation prompt traces, found {generation_prompt_count}")
if accepted_count + rejected_count < target:
    raise SystemExit(
        f"expected at least {target} final accepted/rejected QA rows, found {accepted_count + rejected_count}"
    )
PY

  echo "stage=validate_outputs decode_mode=${decode_mode}"
  python -m egolife_two_user_qa validate_outputs \
    --qa "${qa_dir}/qa_mcq.jsonl" \
    --csv-output "${qa_dir}/qa_mcq.csv" \
    --human-review-output "${qa_dir}/human_review_sheet.md" \
    --report "${qa_dir}/generation_report.md" \
    --strict-review

  echo "qa_output_${decode_mode}=${qa_dir}/qa_mcq.jsonl"
  echo "qa_intermediate_${decode_mode}=${qa_dir}/qa_mcq.intermediate.jsonl"
  echo "qa_rejected_${decode_mode}=${qa_dir}/qa_mcq.rejected.jsonl"
  echo "qa_prompts_${decode_mode}=${qa_dir}/video_first_prompts.jsonl"
  echo "report_${decode_mode}=${qa_dir}/generation_report.md"
}

run_decode_mode greedy
run_decode_mode sampling

echo "outputs:"
echo "  reused_evidence=${EVIDENCE_PATH}"
echo "  source_summary=${EVIDENCE_ROOT}/clip_pruned_pairs/group_relative_clip_summary.json"
echo "  source_paired_videos=${EVIDENCE_ROOT}/clip_pruned_pairs/*/benchmark_video_pairs"
echo "  greedy_qa=${OUTDIR}/${GENERATION_MODE}_greedy/qa_mcq.jsonl"
echo "  sampling_qa=${OUTDIR}/${GENERATION_MODE}_sampling/qa_mcq.jsonl"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log"
echo "date_done=$(date --iso-8601=seconds)"
