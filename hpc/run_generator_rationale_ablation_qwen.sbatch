#!/usr/bin/env bash
#SBATCH --job-name=egolife_rationale_ablation
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=128G
#SBATCH --time=36:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu
#SBATCH --export=ALL

set -euo pipefail

# Phase 1 judges only attempt 1 from each intermediate packet twice. Phase 2
# automatically runs a fresh 50-packet production pipeline whose judges receive
# generator rationale and use only the binary PASS/FAIL contract.
#
#   sbatch hpc/run_generator_rationale_ablation_qwen.sbatch \
#     path/to/qa_mcq.intermediate.jsonl \
#     path/to/evidence_pruned_pairs.jsonl

if [[ "$#" -ne 2 ]]; then
  echo "usage: sbatch $0 <qa_mcq.intermediate.jsonl> <evidence.jsonl>" >&2
  exit 2
fi

JOB_PROJECT_ROOT="/scratch/${USER}/Long-video-understanding-clip"
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ "$1" = /* ]]; then
  INTERMEDIATE_PATH="$1"
else
  INTERMEDIATE_PATH="${SUBMIT_DIR}/$1"
fi
if [[ "$2" = /* ]]; then
  EVIDENCE_PATH="$2"
else
  EVIDENCE_PATH="${SUBMIT_DIR}/$2"
fi
OUTDIR="${RATIONALE_ABLATION_OUTDIR:-egolife_two_user_qa/outputs/generator_rationale_ablation_${SLURM_JOB_ID:-local}}"
OUTPUT_PATH="${OUTDIR}/paired_pass_fail.jsonl"
PRODUCTION_OUTDIR="${OUTDIR}/production_rationale_binary_50"
PRODUCTION_ACCEPTED_PATH="${PRODUCTION_OUTDIR}/qa_mcq.jsonl"
PRODUCTION_REJECTED_PATH="${PRODUCTION_OUTDIR}/qa_mcq.rejected.jsonl"
PRODUCTION_SUMMARY_PATH="${PRODUCTION_OUTDIR}/acceptance_comparison.json"

MODEL_ID="${RATIONALE_ABLATION_MODEL_ID:-Qwen/Qwen3.6-27B}"
ABLATION_MAX_NEW_TOKENS="${RATIONALE_ABLATION_MAX_NEW_TOKENS:-2048}"
ABLATION_MAX_FORMAT_ATTEMPTS="${RATIONALE_ABLATION_MAX_FORMAT_ATTEMPTS:-3}"
MAX_IMAGE_PIXELS="${RATIONALE_ABLATION_MAX_IMAGE_PIXELS:-262144}"
FPS="${RATIONALE_ABLATION_FPS:-1.0}"
MIN_FREE_GIB="${RATIONALE_ABLATION_MIN_FREE_GIB:-12.0}"
MAX_INPUT_TOKENS="${RATIONALE_ABLATION_MAX_INPUT_TOKENS:-245760}"
SEED="${RATIONALE_ABLATION_SEED:-1729}"
RESUME="${RATIONALE_ABLATION_RESUME:-0}"

PRODUCTION_PACKET_COUNT=50
PRODUCTION_MAX_ATTEMPTS="${RATIONALE_PRODUCTION_MAX_ATTEMPTS:-3}"
PRODUCTION_MAX_NEW_TOKENS="${RATIONALE_PRODUCTION_MAX_NEW_TOKENS:-4096}"
PRODUCTION_TEMPERATURE="${RATIONALE_PRODUCTION_TEMPERATURE:-0.7}"
PRODUCTION_TOP_P="${RATIONALE_PRODUCTION_TOP_P:-0.9}"
PRODUCTION_TOP_K="${RATIONALE_PRODUCTION_TOP_K:-40}"
PRODUCTION_RESUME="${RATIONALE_PRODUCTION_RESUME:-0}"

CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${JOB_PROJECT_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-1}"
CUDA_KEEPER_PID=""

export CONDA_ROOT="/share/apps/anaconda3/2025.06"
export CONDA_ENV_NAME="/scratch/${USER}/conda/envs/qwen3vl-smoke"

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    echo "stage=stop_cuda_keeper pid=${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ ! -d "${JOB_PROJECT_ROOT}" ]]; then
  echo "error=project_root_missing path=${JOB_PROJECT_ROOT}" >&2
  exit 1
fi
cd "${JOB_PROJECT_ROOT}"

mkdir -p hpc/logs "${OUTDIR}" "${PRODUCTION_OUTDIR}" "/scratch/${USER}/hf_cache"
if [[ ! -s "${JOB_PROJECT_ROOT}/hpc/env_qwen3vl.sh" ]]; then
  echo "error=environment_script_missing path=${JOB_PROJECT_ROOT}/hpc/env_qwen3vl.sh" >&2
  exit 1
fi
source "${JOB_PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${JOB_PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="/scratch/${USER}/hf_cache"
export TRANSFORMERS_CACHE="/scratch/${USER}/hf_cache"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "date_start=$(date --iso-8601=seconds)"
echo "intermediate=${INTERMEDIATE_PATH}"
echo "evidence=${EVIDENCE_PATH}"
echo "output=${OUTPUT_PATH}"
echo "production_output_dir=${PRODUCTION_OUTDIR}"
echo "model_id=${MODEL_ID}"
echo "paired_ablation_decision_contract=PASS_FAIL_only"
echo "paired_ablation_scoring=disabled"
echo "paired_ablation_quota=disabled"
echo "paired_ablation_cross_candidate_memory=disabled"
echo "paired_ablation_dense_prompt_trace=disabled"
echo "pair_contract=same_question_same_full_videos_only_generator_rationale_changes"
echo "ablation_attempt_selection=attempt_1_only"
echo "ablation_max_new_tokens=${ABLATION_MAX_NEW_TOKENS}"
echo "ablation_max_format_attempts=${ABLATION_MAX_FORMAT_ATTEMPTS}"
echo "production_packet_count=${PRODUCTION_PACKET_COUNT}"
echo "production_judge_generator_rationale=included"
echo "production_judge_contract=binary_PASS_FAIL_only"
echo "production_point_scoring=legacy_archived_not_active"
echo "production_quality_quota=legacy_archived_not_active"
echo "production_pass_fail_entropy_logits=legacy_archived_not_collected"
echo "production_sampling=temperature_${PRODUCTION_TEMPERATURE}_top_p_${PRODUCTION_TOP_P}_top_k_${PRODUCTION_TOP_K}"
echo "max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "fps=${FPS}"
echo "seed=${SEED}"
echo "resume=${RESUME}"
echo "cuda_keeper_enable=${CUDA_KEEPER_ENABLE}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
echo "cuda_keeper_threshold=${CUDA_KEEPER_THRESHOLD}"
echo "cuda_keeper_gpus=${CUDA_KEEPER_GPUS}"
echo "cuda_keeper_reserve=${CUDA_KEEPER_RESERVE}"
echo "cuda_keeper_auto_install=${CUDA_KEEPER_AUTO_INSTALL}"
python --version
nvidia-smi

if [[ ! -s "${INTERMEDIATE_PATH}" ]]; then
  echo "error=intermediate_missing_or_empty path=${INTERMEDIATE_PATH}" >&2
  exit 1
fi
if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_missing_or_empty path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  if [[ -s "${CUDA_KEEPER_SCRIPT}" ]]; then
    echo "stage=preflight_cuda_keeper_imports"
    if ! python - <<'PY'
import importlib.util

missing = [
    name
    for name in ("torch", "pynvml", "psutil")
    if importlib.util.find_spec(name) is None
]
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

missing = [
    name
    for name in ("torch", "pynvml")
    if importlib.util.find_spec(name) is None
]
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

echo "stage=preflight_questions_and_packet_alignment"
python - "${INTERMEDIATE_PATH}" "${EVIDENCE_PATH}" "${PRODUCTION_PACKET_COUNT}" <<'PY'
import sys

from egolife_two_user_qa.generator_rationale_ablation import (
    load_generated_questions,
    summarize_baseline_packets,
)
from egolife_two_user_qa.io_utils import iter_jsonl

questions = load_generated_questions(sys.argv[1])
if not questions:
    raise SystemExit("intermediate file has no parseable attempt-1 generated questions")
packet_count = int(sys.argv[3])
baseline = summarize_baseline_packets(sys.argv[1], packet_count=packet_count)
evidence_rows = list(iter_jsonl(sys.argv[2]))[:packet_count]
if len(evidence_rows) != packet_count:
    raise SystemExit(f"evidence file has {len(evidence_rows)} packets; expected {packet_count}")
evidence_ids = [row.get("evidence_id") for row in evidence_rows]
if baseline["evidence_ids"] != evidence_ids:
    raise SystemExit("first 50 evidence IDs do not exactly match the baseline intermediate")
missing_rationale = sum(not row["qa"].get("generator_rationale") for row in questions)
missing_media = [row["candidate_id"] for row in questions if not row.get("video_evidence")]
if missing_media:
    raise SystemExit(f"questions missing full-video evidence: {missing_media[:10]}")
print(f"preflight_question_count={len(questions)}")
print(f"preflight_expected_judge_calls={len(questions) * 2}")
print(f"preflight_missing_generator_rationale={missing_rationale}")
print(f"preflight_baseline_acceptance_rate={baseline['acceptance_rate']}")
print("preflight_packet_alignment=ok")
PY

resume_args=()
if [[ "${RESUME}" == "1" ]]; then
  resume_args+=(--resume)
fi

echo "stage=paired_pass_fail_judging"
python -m egolife_two_user_qa.generator_rationale_ablation \
  --intermediate "${INTERMEDIATE_PATH}" \
  --output "${OUTPUT_PATH}" \
  --media-root "${JOB_PROJECT_ROOT}" \
  --model-id "${MODEL_ID}" \
  --max-new-tokens "${ABLATION_MAX_NEW_TOKENS}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS}" \
  --fps "${FPS}" \
  --min-free-gib "${MIN_FREE_GIB}" \
  --max-input-tokens "${MAX_INPUT_TOKENS}" \
  --dtype bfloat16 \
  --device-map auto \
  --disable-thinking \
  --seed "${SEED}" \
  --max-format-attempts "${ABLATION_MAX_FORMAT_ATTEMPTS}" \
  "${resume_args[@]}"

echo "stage=verify_compact_outputs"
python - "${INTERMEDIATE_PATH}" "${OUTPUT_PATH}" <<'PY'
import sys

from egolife_two_user_qa.generator_rationale_ablation import (
    load_generated_questions,
    summarize_results,
)
from egolife_two_user_qa.io_utils import iter_jsonl

expected = load_generated_questions(sys.argv[1])
rows = list(iter_jsonl(sys.argv[2]))
if len(rows) != len(expected):
    raise SystemExit(f"row count mismatch: output={len(rows)} expected={len(expected)}")
for row in rows:
    if (row.get("without_rationale") or {}).get("status") not in {"PASS", "FAIL"}:
        raise SystemExit(f"invalid without-rationale status: {row.get('candidate_id')}")
    if (row.get("with_rationale") or {}).get("status") not in {"PASS", "FAIL"}:
        raise SystemExit(f"invalid with-rationale status: {row.get('candidate_id')}")
    forbidden = {"score", "quality_score", "ranking", "prompt", "raw_output", "history"}
    if forbidden & set(row):
        raise SystemExit(f"dense/scoring field found in {row.get('candidate_id')}")
print("verified_summary=" + __import__("json").dumps(summarize_results(rows), sort_keys=True))
PY

production_resume_args=()
if [[ "${PRODUCTION_RESUME}" == "1" ]]; then
  production_resume_args+=(--resume)
fi

echo "stage=production_rationale_binary_50_packets"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_PATH}" \
  --output "${PRODUCTION_ACCEPTED_PATH}" \
  --rejected-output "${PRODUCTION_REJECTED_PATH}" \
  --target-count "${PRODUCTION_PACKET_COUNT}" \
  --max-attempts "${PRODUCTION_MAX_ATTEMPTS}" \
  --backend transformers-local \
  --model-id "${MODEL_ID}" \
  --max-new-tokens "${PRODUCTION_MAX_NEW_TOKENS}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS}" \
  --dtype bfloat16 \
  --disable-thinking \
  --generation-mode baseline \
  --fixed-question-type-schedule \
  --question-types neutral \
  --generator-decode-mode sampling \
  --generator-temperature "${PRODUCTION_TEMPERATURE}" \
  --generator-top-p "${PRODUCTION_TOP_P}" \
  --generator-top-k "${PRODUCTION_TOP_K}" \
  "${production_resume_args[@]}"

echo "stage=compare_production_acceptance"
python - \
  "${INTERMEDIATE_PATH}" \
  "${PRODUCTION_ACCEPTED_PATH}" \
  "${PRODUCTION_REJECTED_PATH}" \
  "${PRODUCTION_SUMMARY_PATH}" \
  "${PRODUCTION_PACKET_COUNT}" <<'PY'
import json
import sys

from egolife_two_user_qa.generator_rationale_ablation import (
    compact_production_outputs,
    compare_acceptance_summaries,
    summarize_baseline_packets,
    summarize_production_outputs,
)
from egolife_two_user_qa.io_utils import iter_jsonl, write_json

baseline = summarize_baseline_packets(sys.argv[1], packet_count=int(sys.argv[5]))
production = summarize_production_outputs(
    sys.argv[2],
    sys.argv[3],
    packet_count=int(sys.argv[5]),
)
comparison = compare_acceptance_summaries(
    baseline,
    production,
    production_label="production_with_rationale_binary",
)
if comparison["same_evidence_id_set"] is not True:
    raise SystemExit("production output evidence IDs differ from baseline")

judge_trace_count = 0
verified_packet_count = 0
for path in (sys.argv[2], sys.argv[3]):
    for row in iter_jsonl(path):
        verified_packet_count += 1
        traces = row.get("generation_trace") or []
        row_judge_trace_count = 0
        for trace in traces:
            judge = trace.get("judge") if isinstance(trace, dict) else None
            if not isinstance(judge, dict):
                continue
            judge_trace_count += 1
            row_judge_trace_count += 1
            if judge.get("generator_rationale_included") is not True:
                raise SystemExit("production judge trace omitted generator rationale")
            if judge.get("pass_fail_only") is not True:
                raise SystemExit("production judge trace is not binary PASS/FAIL")
            if judge.get("pass_fail_entropy_logits") != "legacy_archived_not_collected":
                raise SystemExit("production judge trace did not archive PASS/FAIL logits")
            if judge.get("point_scoring") != "legacy_archived_not_active":
                raise SystemExit("production judge trace reactivated point scoring")
            if "quality_quota" in judge:
                raise SystemExit("production judge trace retained quality-quota metadata")
            merged = judge.get("merged") or {}
            if "decision_uncertainty_summary" in merged:
                raise SystemExit("production review judges retained legacy entropy diagnostics")
            checks = merged.get("checks") or {}
            for check_name in ("qa_formality", "evidence_groundedness"):
                check = checks.get(check_name) or {}
                forbidden = {
                    "quality_score",
                    "quality_flag",
                    "quality_reason",
                    "quality_quota",
                    "quota_rebuttal",
                    "quality_uncertainty",
                    "decision_uncertainty",
                    "status_matches_effective_status",
                }
                found = forbidden & set(check)
                if found:
                    raise SystemExit(
                        f"production {check_name} retained archived fields: {sorted(found)}"
                    )
        if row_judge_trace_count == 0 and not traces:
            config = row.get("production_judge_config") or {}
            if config.get("generator_rationale_included") is not True:
                raise SystemExit("compacted production row lacks rationale-inclusion verification")
            if config.get("pass_fail_only") is not True:
                raise SystemExit("compacted production row lacks binary-contract verification")
            if "quality_quota" in config:
                raise SystemExit("compacted production row retained archived quota metadata")
            if config.get("pass_fail_entropy_logits") != "legacy_archived_not_collected":
                raise SystemExit("compacted production row lacks entropy-archive verification")
if verified_packet_count != int(sys.argv[5]):
    raise SystemExit(
        f"verified {verified_packet_count} production packets; expected {sys.argv[5]}"
    )

comparison["verified_binary_judge_trace_count"] = judge_trace_count
comparison["compact_output_counts"] = compact_production_outputs(
    sys.argv[2],
    sys.argv[3],
    generator_rationale_included=True,
    pass_fail_only=True,
    pass_fail_entropy_logits="legacy_archived_not_collected",
)
write_json(sys.argv[4], comparison)
print("acceptance_comparison=" + json.dumps(comparison, sort_keys=True))
PY

echo "outputs:"
echo "  paired_pass_fail=${OUTPUT_PATH}"
echo "  production_accepted=${PRODUCTION_ACCEPTED_PATH}"
echo "  production_rejected=${PRODUCTION_REJECTED_PATH}"
echo "  production_acceptance_comparison=${PRODUCTION_SUMMARY_PATH}"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log"
echo "date_end=$(date --iso-8601=seconds)"
