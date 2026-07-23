#!/usr/bin/env bash
#SBATCH --job-name=egolife_g25_q48
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

# This is an isolated hybrid experiment. It does not call or source the
# production Qwen-generator launcher.
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2/evidence_pruned_pairs.jsonl}"
OUTDIR="${EGOLIFE2U_GEMINI25_TEST_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_gemini25_openrouter_generator_qwen_judge_quota48_50_${SLURM_JOB_ID:-manual}}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT=50
QUALITY_QUOTA=48
GENERATOR_BACKEND="openrouter"
GENERATOR_MODEL_ID="google/gemini-2.5-flash"
OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
# Replace only the placeholder text below. The surrounding shell quotes are
# syntax and do not become part of the key value.
OPENROUTER_API_KEY="PASTE_OPENROUTER_API_KEY_HERE"
export OPENROUTER_API_KEY
JUDGE_BACKEND="transformers-local"
JUDGE_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
GENERATOR_MAX_NEW_TOKENS="${GENERATOR_MAX_NEW_TOKENS:-4096}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-4096}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-262144}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"

# Sampling is sent only to the remote generator. All Qwen judge and
# answerability calls keep the runner's default greedy decoding.
GENERATOR_TEMPERATURE=0.7
GENERATOR_TOP_P=0.9
GENERATOR_TOP_K=40

# The generator receives the existing pruned clips as local-video data URLs.
# Zero preserves their encoded resolution and frame rate. If the gateway rejects
# a large upload, these can be overridden for an explicit compression ablation.
export OPENROUTER_VIDEO_MAX_EDGE="${OPENROUTER_VIDEO_MAX_EDGE:-0}"
export OPENROUTER_VIDEO_FPS="${OPENROUTER_VIDEO_FPS:-0}"
export OPENROUTER_VIDEO_CRF="${OPENROUTER_VIDEO_CRF:-28}"
export OPENROUTER_VIDEO_CACHE_DIR="${OPENROUTER_VIDEO_CACHE_DIR:-${CACHE_DIR}/openrouter_gemini25_video_cache}"
export OPENROUTER_MAX_RETRIES="${OPENROUTER_MAX_RETRIES:-4}"
export OPENROUTER_RETRY_DELAY_SECONDS="${OPENROUTER_RETRY_DELAY_SECONDS:-5}"

# The CUDA keeper is required and shares the single allocated GPU with the one
# local Qwen judge instance.
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
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "${OPENROUTER_VIDEO_CACHE_DIR}" "/scratch/${USER}/hf_cache"

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
echo "generator_backend=${GENERATOR_BACKEND}"
echo "generator_model_id=${GENERATOR_MODEL_ID}"
echo "openrouter_base_url=${OPENROUTER_BASE_URL}"
echo "openrouter_api_key=embedded_in_experiment_sbatch"
echo "judge_backend=${JUDGE_BACKEND}"
echo "judge_model_id=${JUDGE_MODEL_ID}"
echo "runner_topology=remote_gemini_generator_plus_one_local_qwen_runner_for_all_judges"
echo "generator_decode=sampling temperature=${GENERATOR_TEMPERATURE} top_p=${GENERATOR_TOP_P} top_k=${GENERATOR_TOP_K}"
echo "judge_decode=greedy"
echo "judge_quality_quota_per_category=${QUALITY_QUOTA}"
echo "judge_categories=qa_formality,evidence_groundedness"
echo "generator_rationale_included=true"
echo "quota_prompt_format=plain_text_no_markdown_bold_markers"
echo "pass_fail_entropy_logits=legacy_archived_not_collected"
echo "generator_media=pruned_pair_videos_via_openrouter_video_data_urls"
echo "judge_media=full_original_pair_videos_via_local_qwen"
echo "openrouter_video_max_edge=${OPENROUTER_VIDEO_MAX_EDGE}"
echo "openrouter_video_fps=${OPENROUTER_VIDEO_FPS}"
echo "openrouter_video_crf=${OPENROUTER_VIDEO_CRF}"
echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD} gpus=${CUDA_KEEPER_GPUS} reserve_gib=${CUDA_KEEPER_RESERVE}"
python --version
nvidia-smi

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if [[ -z "${OPENROUTER_API_KEY:-}" || "${OPENROUTER_API_KEY}" == "PASTE_OPENROUTER_API_KEY_HERE" ]]; then
  echo "error=openrouter_api_key_missing replace PASTE_OPENROUTER_API_KEY_HERE in this sbatch" >&2
  exit 2
fi

echo "stage=preflight_openrouter_auth"
# Validate the inherited key without printing it or placing it in process
# arguments. This fails before the CUDA keeper starts or Qwen is loaded.
python - "${OPENROUTER_BASE_URL}/key" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.request

key = os.environ.get("OPENROUTER_API_KEY", "")
if not key or not key.strip():
    raise SystemExit("OPENROUTER_API_KEY is empty or whitespace-only inside the Slurm job")
if key != key.strip():
    raise SystemExit(
        "OPENROUTER_API_KEY has leading/trailing whitespace; re-export the key without it"
    )
if len(key) >= 2 and key[0] == key[-1] and key[0] in {"'", '"'}:
    raise SystemExit(
        "quotation marks became part of OPENROUTER_API_KEY; use "
        "export OPENROUTER_API_KEY=\"sk-or-v1-...\""
    )

request = urllib.request.Request(
    sys.argv[1],
    headers={
        "Accept": "application/json",
        "Authorization": f"Bearer {key}",
        "User-Agent": "egolife-two-user-qa-auth-preflight/1.0",
    },
    method="GET",
)
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8"))
except urllib.error.HTTPError as exc:
    detail = exc.read().decode("utf-8", errors="replace")[-1000:]
    raise SystemExit(f"OpenRouter authentication preflight failed: HTTP {exc.code}: {detail}") from exc
except urllib.error.URLError as exc:
    raise SystemExit(f"OpenRouter authentication preflight could not connect: {exc}") from exc

if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
    raise SystemExit("OpenRouter authentication preflight returned an unexpected response")
print(f"openrouter_auth=ok key_length={len(key)}")
PY

if [[ ! -s "${CUDA_KEEPER_SCRIPT}" ]]; then
  echo "error=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}" >&2
  exit 3
fi

echo "stage=preflight_experiment_contract"
python - "${EVIDENCE_PATH}" "${TARGET_COUNT}" "${QUALITY_QUOTA}" <<'PY'
import json
import sys
from pathlib import Path

from egolife_two_user_qa.prompts import quality_quota_prompt

path = Path(sys.argv[1])
target = int(sys.argv[2])
quota = int(sys.argv[3])
rows = []
with path.open("r", encoding="utf-8") as handle:
    for line_number, line in enumerate(handle, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        evidence_id = str(row.get("evidence_id") or "")
        if not evidence_id:
            raise SystemExit(f"missing evidence_id at line {line_number}")
        rows.append(row)
if len(rows) < target:
    raise SystemExit(f"expected at least {target} evidence pairs, found {len(rows)}")
if len({str(row.get('evidence_id')) for row in rows[:target]}) != target:
    raise SystemExit(f"the first {target} evidence rows contain duplicate IDs")

bold_marker = "*" * 2
for previous in (0, quota - 1, quota):
    prompt = quality_quota_prompt(
        previous_three_point_assignments=previous,
        quota=quota,
    )
    required = f"The prompt budget for this category is at most {quota} 3-point assignments."
    if required not in prompt:
        raise SystemExit("plain-text quota prompt omitted its budget sentence")
    if bold_marker in prompt:
        raise SystemExit("quota prompt contains a markdown bold marker")
print(f"evidence_count={len(rows)} target_first_pairs={target}")
print("quota_prompt_plain_text=ok")
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
  exit 4
fi
echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"

resume_args=()
if [[ "${RESUME_GENERATION}" == "1" ]]; then
  resume_args+=(--resume)
fi

echo "stage=generate_video_qa_loop"
# OPENROUTER_API_KEY stays in the environment instead of appearing in the
# process arguments. Omitting --qa-formality-use-generator routes every judge,
# including text-only formality and answerability, to the one local Qwen runner.
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_PATH}" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend "${GENERATOR_BACKEND}" \
  --model-id "${GENERATOR_MODEL_ID}" \
  --base-url "${OPENROUTER_BASE_URL}" \
  --allow-openai-video-input \
  --max-new-tokens "${GENERATOR_MAX_NEW_TOKENS}" \
  --judge-backend "${JUDGE_BACKEND}" \
  --judge-model-id "${JUDGE_MODEL_ID}" \
  --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS}" \
  --dtype bfloat16 \
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
  "${GENERATOR_MODEL_ID}" \
  "${JUDGE_MODEL_ID}" \
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
generator_model = sys.argv[5]
judge_model = sys.argv[6]
temperature = float(sys.argv[7])
top_p = float(sys.argv[8])
top_k = int(sys.argv[9])


def read_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


evidence = read_rows(evidence_path)
accepted = read_rows(output_dir / "qa_mcq.jsonl")
rejected = read_rows(output_dir / "qa_mcq.rejected.jsonl")
prompts = read_rows(output_dir / "video_first_prompts.jsonl")
final_rows = accepted + rejected

expected_ids = [str(row.get("evidence_id") or "") for row in evidence[:target]]
final_ids = [str(row.get("evidence_id") or "") for row in final_rows]
if len(final_rows) != target:
    raise SystemExit(f"expected exactly {target} accepted/rejected pairs, found {len(final_rows)}")
if len(set(final_ids)) != target or set(final_ids) != set(expected_ids):
    raise SystemExit("final outputs do not cover each of the first evidence pairs exactly once")

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
    if not row.get("video_paths"):
        raise SystemExit("a generator attempt did not receive the pruned videos")

bold_marker = "*" * 2
for row in judge_prompts:
    if row.get("model_id") != judge_model:
        raise SystemExit(f"judge prompt routed to the wrong model: {row.get('model_id')}")
    if row.get("generator_rationale_included") is not True:
        raise SystemExit("a judge prompt omitted generator_rationale")
    if row.get("pass_fail_only") is not False:
        raise SystemExit("a judge prompt used the legacy binary-only contract")
    snapshot = row.get("quality_quota") or {}
    if snapshot.get("quota") != quota:
        raise SystemExit(f"a judge prompt used the wrong per-category quota: {snapshot}")
    prompt = str(row.get("prompt") or "")
    required = f"The prompt budget for this category is at most {quota} 3-point assignments."
    if required not in prompt:
        raise SystemExit("a judge prompt omitted the visible quota instruction")
    if f"{bold_marker}{required}{bold_marker}" in prompt:
        raise SystemExit("a judge quota instruction still contains markdown bold markers")

for row in accepted:
    if row.get("model_id") != generator_model:
        raise SystemExit(f"accepted row has wrong generator model: {row.get('model_id')}")
    if row.get("review_model_id") != judge_model:
        raise SystemExit(f"accepted row has wrong reviewer model: {row.get('review_model_id')}")
    review_models = row.get("review_model_ids") or {}
    for category in ("qa_formality", "evidence_groundedness", "answerability"):
        if review_models.get(category) != judge_model:
            raise SystemExit(f"accepted row routed {category} to {review_models.get(category)}")

three_point_counts = Counter()
scored_checks = 0
judged_traces = 0
for row in final_rows:
    for trace in row.get("generation_trace") or []:
        judge_trace = trace.get("judge") if isinstance(trace, dict) else None
        if not isinstance(judge_trace, dict) or not judge_trace.get("merged"):
            continue
        judged_traces += 1
        if judge_trace.get("pass_fail_entropy_logits") != "legacy_archived_not_collected":
            raise SystemExit("a trace retained the legacy PASS/FAIL entropy path")
        if (judge_trace.get("qa_formality") or {}).get("model_id") != judge_model:
            raise SystemExit("qa_formality trace was not produced by Qwen")
        if (judge_trace.get("evidence_groundedness") or {}).get("model_id") != judge_model:
            raise SystemExit("evidence_groundedness trace was not produced by Qwen")
        if judge_trace.get("answerability_model_id") != judge_model:
            raise SystemExit("answerability trace was not produced by Qwen")
        merged = judge_trace["merged"]
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
            if audit.get("quota_rebuttal_required") and not str(check.get("quota_rebuttal") or "").strip():
                raise SystemExit(f"{category} omitted its required over-quota rebuttal")
            if score == 3:
                three_point_counts[category] += 1
            scored_checks += 1

if judged_traces == 0 or scored_checks == 0:
    raise SystemExit("no completed Qwen judge traces were verified")

print(f"verified_evidence_pairs={len(final_rows)}")
print(f"accepted_count={len(accepted)}")
print(f"rejected_count={len(rejected)}")
print(f"generator_attempt_count={len(generation_prompts)}")
print(f"qwen_judged_trace_count={judged_traces}")
print(f"scored_judge_check_count={scored_checks}")
print(f"qa_formality_three_point_count={three_point_counts['qa_formality']}")
print(f"evidence_groundedness_three_point_count={three_point_counts['evidence_groundedness']}")
PY

echo "outputs:"
echo "  qa=${OUTDIR}/qa_mcq.jsonl"
echo "  intermediate=${OUTDIR}/qa_mcq.intermediate.jsonl"
echo "  rejected=${OUTDIR}/qa_mcq.rejected.jsonl"
echo "  prompts=${OUTDIR}/video_first_prompts.jsonl"
echo "date_end=$(date --iso-8601=seconds)"
