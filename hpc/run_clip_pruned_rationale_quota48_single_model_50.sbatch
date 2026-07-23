#!/usr/bin/env bash
#SBATCH --job-name=egolife_q48_single
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

# ARCHIVED LEGACY EXPERIMENT: the quota-48 scoring contract is no longer part
# of the production judges. The historical implementation remains below for
# reference; use run_clip_pruned_sampling_neutral_pf_50.sbatch instead.
echo "error=archived_scored_quota_judge_experiment_not_a_production_pipeline" >&2
exit 2

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2/evidence_pruned_pairs.jsonl}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_rationale_quota48_single_model_50_${SLURM_JOB_ID:-manual}}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

# Fixed experiment contract.
TARGET_COUNT=50
QUALITY_QUOTA=48
MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-262144}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"

# Sampling is passed only to the generator call. Review calls use the runner's
# default greedy decoding because no sampling arguments are passed to them.
GENERATOR_TEMPERATURE=0.7
GENERATOR_TOP_P=0.9
GENERATOR_TOP_K=40

# The keeper is required for this launcher and is stopped by the cleanup trap.
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
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
echo "evidence_path=${EVIDENCE_PATH}"
echo "outdir=${OUTDIR}"
echo "target_count=${TARGET_COUNT}"
echo "model_id=${MODEL_ID}"
echo "runner_topology=one_transformers_local_runner_shared_by_generator_and_all_judges"
echo "generator_decode=sampling temperature=${GENERATOR_TEMPERATURE} top_p=${GENERATOR_TOP_P} top_k=${GENERATOR_TOP_K}"
echo "judge_decode=greedy"
echo "judge_quality_quota_per_category=${QUALITY_QUOTA}"
echo "judge_categories=qa_formality,evidence_groundedness"
echo "generator_rationale_included=true"
echo "pass_fail_entropy_logits=legacy_archived_not_collected"
echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD} gpus=${CUDA_KEEPER_GPUS} reserve_gib=${CUDA_KEEPER_RESERVE}"
python --version
nvidia-smi

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if [[ ! -s "${CUDA_KEEPER_SCRIPT}" ]]; then
  echo "error=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}" >&2
  exit 1
fi

echo "stage=preflight_evidence"
python - "${EVIDENCE_PATH}" "${TARGET_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target = int(sys.argv[2])
count = 0
ids = set()
with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id:
            raise SystemExit(f"missing evidence_id at line {line_number}")
        if evidence_id in ids:
            raise SystemExit(f"duplicate evidence_id at line {line_number}: {evidence_id}")
        ids.add(evidence_id)
        count += 1
if count < target:
    raise SystemExit(f"expected at least {target} evidence pairs, found {count}")
print(f"evidence_count={count} target_first_pairs={target}")
PY

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
  echo "stage=install_cuda_keeper_dependencies"
  python -m pip install nvidia-ml-py3 psutil
fi

python - <<'PY'
import importlib.util

missing = [name for name in ("torch", "pynvml", "psutil") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("cuda keeper dependencies remain missing: " + ",".join(missing))
print("cuda_keeper_imports_after_install=ok")
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

resume_args=()
if [[ "${RESUME_GENERATION}" == "1" ]]; then
  resume_args+=(--resume)
fi

echo "stage=generate_video_qa_loop"
# Deliberately do not pass --judge-backend, --judge-model-id,
# --judge-max-new-tokens, --judge-base-url, --judge-api-key, or
# --judge-reasoning-effort. Their absence keeps judge_runner identical to the
# one generator runner created by video_qa_loop.py.
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_PATH}" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local \
  --model-id "${MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS}" \
  --disable-thinking \
  --generation-mode baseline \
  --fixed-question-type-schedule \
  --question-types neutral \
  --generator-decode-mode sampling \
  --generator-temperature "${GENERATOR_TEMPERATURE}" \
  --generator-top-p "${GENERATOR_TOP_P}" \
  --generator-top-k "${GENERATOR_TOP_K}" \
  --judge-quality-quota "${QUALITY_QUOTA}" \
  "${resume_args[@]}"

echo "stage=verify_generation"
python - \
  "${EVIDENCE_PATH}" \
  "${OUTDIR}" \
  "${TARGET_COUNT}" \
  "${QUALITY_QUOTA}" \
  "${GENERATOR_TEMPERATURE}" \
  "${GENERATOR_TOP_P}" \
  "${GENERATOR_TOP_K}" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

evidence_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
target = int(sys.argv[3])
quota = int(sys.argv[4])
temperature = float(sys.argv[5])
top_p = float(sys.argv[6])
top_k = int(sys.argv[7])


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


evidence = read_rows(evidence_path)
accepted = read_rows(output_dir / "qa_mcq.jsonl")
rejected = read_rows(output_dir / "qa_mcq.rejected.jsonl")
prompts = read_rows(output_dir / "video_first_prompts.jsonl")
final_rows = [*accepted, *rejected]

expected_ids = [str(row.get("evidence_id") or "") for row in evidence[:target]]
final_ids = [str(row.get("evidence_id") or "") for row in final_rows]
if len(final_rows) != target:
    raise SystemExit(
        f"expected exactly {target} final accepted/rejected evidence pairs, found {len(final_rows)}"
    )
if len(set(final_ids)) != target:
    raise SystemExit("final output contains duplicate or missing evidence IDs")
if set(final_ids) != set(expected_ids):
    missing = sorted(set(expected_ids) - set(final_ids))
    extra = sorted(set(final_ids) - set(expected_ids))
    raise SystemExit(f"final evidence IDs do not match the first {target}: missing={missing} extra={extra}")

generation_prompts = [row for row in prompts if row.get("stage") == "generation"]
judge_prompts = [
    row
    for row in prompts
    if row.get("stage") in {"qa_formality_judge", "evidence_groundedness_judge"}
]
if len(generation_prompts) < target:
    raise SystemExit(f"expected at least {target} generator attempts, found {len(generation_prompts)}")
if not judge_prompts:
    raise SystemExit("no scored judge prompts were recorded")

for row in generation_prompts:
    config = row.get("generator_decode") or {}
    if config.get("mode") != "sampling":
        raise SystemExit(f"generator did not use sampling: {config}")
    if float(config.get("temperature")) != temperature:
        raise SystemExit(f"wrong generator temperature: {config}")
    if float(config.get("top_p")) != top_p:
        raise SystemExit(f"wrong generator top_p: {config}")
    if int(config.get("top_k")) != top_k:
        raise SystemExit(f"wrong generator top_k: {config}")

for row in judge_prompts:
    if row.get("generator_rationale_included") is not True:
        raise SystemExit("a judge prompt omitted generator_rationale")
    if row.get("pass_fail_only") is not False:
        raise SystemExit("a judge prompt used the legacy binary-only contract")
    snapshot = row.get("quality_quota") or {}
    if snapshot.get("quota") != quota:
        raise SystemExit(f"a judge prompt used the wrong per-category quota: {snapshot}")
    prompt = str(row.get("prompt") or "")
    required = f"**The prompt budget for this category is at most {quota} 3-point assignments.**"
    if required not in prompt:
        raise SystemExit("a judge prompt omitted the required visible quota instruction")

three_point_counts = Counter()
scored_checks = 0
for row in final_rows:
    if row.get("model_id") != row.get("review_model_id"):
        raise SystemExit(
            f"generator/reviewer model IDs differ for evidence_id={row.get('evidence_id')}"
        )
    for trace in row.get("generation_trace") or []:
        judge_trace = trace.get("judge") if isinstance(trace, dict) else None
        if not isinstance(judge_trace, dict):
            continue
        if judge_trace.get("pass_fail_entropy_logits") != "legacy_archived_not_collected":
            raise SystemExit("a judge trace retained the legacy PASS/FAIL entropy path")
        merged = judge_trace.get("merged")
        if not isinstance(merged, dict):
            continue
        if "decision_uncertainty_summary" in merged:
            raise SystemExit("a judge trace retained legacy entropy JSON")
        checks = merged.get("checks") or {}
        for category in ("qa_formality", "evidence_groundedness"):
            check = checks.get(category) or {}
            score = check.get("quality_score")
            if score not in {1, 2, 3}:
                raise SystemExit(f"{category} omitted its 1/2/3 score")
            if not str(check.get("quality_reason") or "").strip():
                raise SystemExit(f"{category} omitted its score rationale")
            audit = check.get("quality_quota") or {}
            if audit.get("quota") != quota:
                raise SystemExit(f"{category} omitted its per-category quota audit")
            if audit.get("quota_rebuttal_required") and not str(
                check.get("quota_rebuttal") or ""
            ).strip():
                raise SystemExit(f"{category} omitted its required over-quota rebuttal")
            if score == 3:
                three_point_counts[category] += 1
            scored_checks += 1

if scored_checks == 0:
    raise SystemExit("no scored judge traces were verified")

print(f"verified_evidence_pairs={len(final_rows)}")
print(f"accepted_count={len(accepted)}")
print(f"rejected_count={len(rejected)}")
print(f"generator_attempt_count={len(generation_prompts)}")
print(f"scored_judge_check_count={scored_checks}")
print(f"qa_formality_three_point_count={three_point_counts['qa_formality']}")
print(f"evidence_groundedness_three_point_count={three_point_counts['evidence_groundedness']}")
PY

echo "stage=validate_accepted_outputs"
python -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review

echo "outputs:"
echo "  accepted=${OUTDIR}/qa_mcq.jsonl"
echo "  rejected=${OUTDIR}/qa_mcq.rejected.jsonl"
echo "  intermediate=${OUTDIR}/qa_mcq.intermediate.jsonl"
echo "  prompts=${OUTDIR}/video_first_prompts.jsonl"
echo "  report=${OUTDIR}/generation_report.md"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log"
echo "date_end=$(date --iso-8601=seconds)"
