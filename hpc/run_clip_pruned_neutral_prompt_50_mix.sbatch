#!/usr/bin/env bash
#SBATCH --job-name=egolife_neutral50_hyb
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

# Reuse the same CLIP-pruned video-pair packets as the 50-pair decode ablation.
EVIDENCE_ROOT="${EGOLIFE2U_EVIDENCE_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${EVIDENCE_ROOT}/evidence_pruned_pairs.jsonl}"

OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_neutral_prompt_50_gemini}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-50}"
EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${TARGET_COUNT}}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
VLM_BACKEND="${VLM_BACKEND:-${EGOLIFE2U_BACKEND:-gemini}}"
if [[ -z "${VLM_MODEL_ID:-}" ]]; then
  if [[ "${VLM_BACKEND}" == "gemini" ]]; then
    VLM_MODEL_ID="${GEMINI_MODEL_ID:-gemini-3.5-flash}"
  else
    VLM_MODEL_ID="${QWEN_MODEL_ID}"
  fi
fi
VLM_API_KEY="${VLM_API_KEY:-${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
JUDGE_BACKEND="${JUDGE_BACKEND:-transformers-local}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-${QWEN_MODEL_ID}}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-${MAX_NEW_TOKENS}}"
GENERATION_MODE="${GENERATION_MODE:-baseline}"
GENERATOR_DECODE_MODE="${GENERATOR_DECODE_MODE:-greedy}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
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
echo "vlm_backend=${VLM_BACKEND}"
echo "vlm_model_id=${VLM_MODEL_ID}"
echo "judge_backend=${JUDGE_BACKEND}"
echo "judge_model_id=${JUDGE_MODEL_ID}"
echo "judge_max_new_tokens=${JUDGE_MAX_NEW_TOKENS}"
if [[ -n "${VLM_API_KEY}" ]]; then
  echo "vlm_api_key=provided"
fi
echo "generation_mode=${GENERATION_MODE}"
echo "question_types=neutral"
echo "generator_decode_mode=${GENERATOR_DECODE_MODE}"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "disable_thinking=${DISABLE_THINKING}"
echo "resume_generation=${RESUME_GENERATION}"
echo "cuda_keeper_enable=${CUDA_KEEPER_ENABLE}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
echo "cuda_keeper_threshold=${CUDA_KEEPER_THRESHOLD}"
echo "cuda_keeper_gpus=${CUDA_KEEPER_GPUS}"
echo "cuda_keeper_reserve=${CUDA_KEEPER_RESERVE}"
echo "cuda_keeper_auto_install=${CUDA_KEEPER_AUTO_INSTALL}"
echo "benchmark_contract=reuse_same_first_50_clip_pruned_packets_neutral_prompt_gemini_generator_qwen_judges"
echo "media_routing=generator_uses_existing_pruned_videos;judges_and_answerability_use_full_30s_originals"
python --version
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia_smi=unavailable"
fi

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if [[ "${VLM_BACKEND}" == "gemini" && -z "${VLM_API_KEY}" ]]; then
  echo "error=gemini_api_key_missing set GEMINI_API_KEY, GOOGLE_API_KEY, or VLM_API_KEY before sbatch" >&2
  exit 2
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

api_key_args=()
if [[ -n "${VLM_API_KEY}" ]]; then
  api_key_args+=(--api-key "${VLM_API_KEY}")
fi

judge_args=(
  --judge-backend "${JUDGE_BACKEND}"
  --judge-model-id "${JUDGE_MODEL_ID}"
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS}"
)

decode_args=(--generator-decode-mode "${GENERATOR_DECODE_MODE}")
if [[ "${GENERATOR_DECODE_MODE}" == "sampling" ]]; then
  decode_args+=(--generator-temperature "${SAMPLING_TEMPERATURE}")
  decode_args+=(--generator-top-p "${SAMPLING_TOP_P}")
  if [[ -n "${SAMPLING_TOP_K}" ]]; then
    decode_args+=(--generator-top-k "${SAMPLING_TOP_K}")
  fi
fi

QA_DIR="${OUTDIR}/${GENERATION_MODE}_neutral_${GENERATOR_DECODE_MODE}"
mkdir -p "${QA_DIR}"

echo "stage=generate_video_qa_loop question_types=neutral decode_mode=${GENERATOR_DECODE_MODE}"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_PATH}" \
  --output "${QA_DIR}/qa_mcq.jsonl" \
  --prompts-output "${QA_DIR}/video_first_prompts.jsonl" \
  --rejected-output "${QA_DIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${QA_DIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${EVAL_PACKET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend "${VLM_BACKEND}" \
  --model-id "${VLM_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generation-mode "${GENERATION_MODE}" \
  --fixed-question-type-schedule \
  --question-types neutral \
  "${decode_args[@]}" \
  "${thinking_args[@]}" \
  "${api_key_args[@]}" \
  "${judge_args[@]}" \
  "${resume_args[@]}"

echo "stage=verify_neutral_generation_traces"
python - "${EVIDENCE_PATH}" "${QA_DIR}" "${EVAL_PACKET_COUNT}" "${GENERATOR_DECODE_MODE}" <<'PY'
import json
import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
qa_dir = Path(sys.argv[2])
target = int(sys.argv[3])
decode_mode = sys.argv[4]

forbidden_prompt_fragments = [
    'question_type must be "neutral"',
    'Target question_type: "neutral"',
    "Create a commonality question",
    "Create a difference question",
    "commonality or difference subtype",
]

def count_lines(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def inspect_generation_prompts(path):
    if not path.exists():
        return 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") != "generation":
                continue
            total += 1
            prompt = row.get("prompt") or ""
            mode = row.get("generator_decode", {}).get("mode")
            if mode != decode_mode:
                raise SystemExit(f"decode_mode mismatch in prompt trace: expected {decode_mode}, got {mode}")
            for fragment in forbidden_prompt_fragments:
                if fragment in prompt:
                    raise SystemExit(f"neutral prompt contains forbidden fragment: {fragment}")
    return total

def inspect_judge_prompts(path):
    counts = {
        "qa_formality_judge": 0,
        "evidence_groundedness_judge": 0,
    }
    required_fragments = {
        "qa_formality_judge": [
            "qa_formality quality_score rubric",
            "direct_name_leakage",
            "quality_score",
            "quality_flag",
            "quality_reason",
        ],
        "evidence_groundedness_judge": [
            "evidence_groundedness quality_score rubric",
            "quality_score",
            "quality_flag",
            "quality_reason",
        ],
    }
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stage = row.get("stage")
            if stage not in counts:
                continue
            counts[stage] += 1
            prompt = row.get("prompt") or ""
            if "quality_reward" in prompt:
                raise SystemExit(f"{stage} prompt still asks for quality_reward")
            for fragment in required_fragments[stage]:
                if fragment not in prompt:
                    raise SystemExit(f"{stage} prompt missing required fragment: {fragment}")
    return counts

def inspect_final_rows(path):
    if not path.exists():
        return 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            question_type = row.get("question_type")
            if question_type != "neutral":
                raise SystemExit(f"final row question_type mismatch: expected neutral, got {question_type}")
            judger = (row.get("review") or {}).get("judger") or {}
            checks = judger.get("checks") if isinstance(judger, dict) else {}
            if isinstance(checks, dict):
                for check_name in ("qa_formality", "evidence_groundedness"):
                    check = checks.get(check_name)
                    if not isinstance(check, dict):
                        continue
                    score = check.get("quality_score")
                    flag = check.get("quality_flag")
                    reason = str(check.get("quality_reason") or "").strip()
                    expected_flags = {
                        1: "1_weak_or_reject",
                        2: "2_acceptable",
                        3: "3_strong",
                    }
                    if score not in expected_flags:
                        raise SystemExit(f"{path.name} row missing valid {check_name}.quality_score")
                    if flag != expected_flags[score]:
                        raise SystemExit(f"{path.name} row has mismatched {check_name}.quality_flag")
                    if not reason:
                        raise SystemExit(f"{path.name} row missing {check_name}.quality_reason")
                    if "quality_reward" in check:
                        raise SystemExit(f"{path.name} row still contains {check_name}.quality_reward")
    return total

evidence_count = count_lines(evidence_path)
prompts_path = qa_dir / "video_first_prompts.jsonl"
generation_prompt_count = inspect_generation_prompts(prompts_path)
judge_prompt_counts = inspect_judge_prompts(prompts_path)
accepted_count = inspect_final_rows(qa_dir / "qa_mcq.jsonl")
rejected_count = inspect_final_rows(qa_dir / "qa_mcq.rejected.jsonl")

print(f"trace_check evidence_count={evidence_count}")
print(f"trace_check generation_prompt_count={generation_prompt_count}")
for stage, count in sorted(judge_prompt_counts.items()):
    print(f"trace_check {stage}_count={count}")
print(f"trace_check accepted_count={accepted_count}")
print(f"trace_check rejected_count={rejected_count}")

if evidence_count < target:
    raise SystemExit(f"expected at least {target} evidence packets, found {evidence_count}")
if generation_prompt_count < target:
    raise SystemExit(f"expected at least {target} generation prompt traces, found {generation_prompt_count}")
if sum(judge_prompt_counts.values()) == 0:
    raise SystemExit("expected judge prompt traces with quality scoring rubrics, found none")
if accepted_count + rejected_count < target:
    raise SystemExit(
        f"expected at least {target} final accepted/rejected question-answer rows, found {accepted_count + rejected_count}"
    )
PY

echo "stage=validate_outputs"
python -m egolife_two_user_qa validate_outputs \
  --qa "${QA_DIR}/qa_mcq.jsonl" \
  --csv-output "${QA_DIR}/qa_mcq.csv" \
  --human-review-output "${QA_DIR}/human_review_sheet.md" \
  --report "${QA_DIR}/generation_report.md" \
  --strict-review

echo "outputs:"
echo "  reused_evidence=${EVIDENCE_PATH}"
echo "  source_summary=${EVIDENCE_ROOT}/clip_pruned_pairs/group_relative_clip_summary.json"
echo "  source_paired_videos=${EVIDENCE_ROOT}/clip_pruned_pairs/*/benchmark_video_pairs"
echo "  qa=${QA_DIR}/qa_mcq.jsonl"
echo "  intermediate=${QA_DIR}/qa_mcq.intermediate.jsonl"
echo "  rejected=${QA_DIR}/qa_mcq.rejected.jsonl"
echo "  prompts=${QA_DIR}/video_first_prompts.jsonl"
echo "  report=${QA_DIR}/generation_report.md"
if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log"
fi
echo "date_done=$(date --iso-8601=seconds)"
