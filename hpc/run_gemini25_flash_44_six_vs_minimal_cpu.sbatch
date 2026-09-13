#!/usr/bin/env bash
#SBATCH --job-name=g25f-6v2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --export=ALL

set -euo pipefail

# Slurm runs a private spool copy of this script, so BASH_SOURCE points under
# /opt/slurm rather than at the submitted file. Resolve from the preserved
# submission directory; PROJECT_ROOT remains an explicit override.
REPO_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "${REPO_ROOT}" ]]; then
  echo "Cannot resolve the project root: submit with sbatch from Long-video-understanding-clip or set PROJECT_ROOT." >&2
  exit 1
fi
REPO_ROOT="$(realpath -- "${REPO_ROOT}")"
PACKAGE_DIR="${REPO_ROOT}/egolife_two_user_qa"

if [[ ! -f "${PACKAGE_DIR}/__init__.py" ]]; then
  echo "Submit this job from the Long-video-understanding-clip project root, or set PROJECT_ROOT explicitly." >&2
  echo "Resolved project root: ${REPO_ROOT}" >&2
  echo "Missing Python package: ${PACKAGE_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}"

: "${RUN_DIR:?Set RUN_DIR to six_user_baseline_neutral_reuse_packets_100_16266077}"
: "${OPENROUTER_API_KEY:?Export OPENROUTER_API_KEY before sbatch}"

RUN_DIR="$(realpath -- "${RUN_DIR}")"
ACCEPTED_QA="${RUN_DIR}/qa_mcq.jsonl"
EVIDENCE_JSONL="${RUN_DIR}/six_user_candidates.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/gemini25_flash_44_six_vs_minimal}"
PYTHON_BIN="${PYTHON_BIN:-python}"

for required_file in \
  "${ACCEPTED_QA}" \
  "${EVIDENCE_JSONL}" \
  "${RUN_DIR}/six_user_qa_result.json"; do
  if [[ ! -s "${required_file}" ]]; then
    echo "Missing or empty last-run artifact: ${required_file}" >&2
    exit 1
  fi
done

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

# Moderate CPU transcodes keep six base64 videos practical without reducing
# the visual detail as aggressively as a 480p evaluation would.
export OPENROUTER_VIDEO_MAX_EDGE="${OPENROUTER_VIDEO_MAX_EDGE:-720}"
export OPENROUTER_VIDEO_FPS="${OPENROUTER_VIDEO_FPS:-2}"
export OPENROUTER_VIDEO_CRF="${OPENROUTER_VIDEO_CRF:-23}"
export OPENROUTER_VIDEO_CACHE_DIR="${OPENROUTER_VIDEO_CACHE_DIR:-${OUTPUT_DIR}/video_cache}"

# Preserve the exact 88 logical-call budget. Failed rows are retried only by
# resubmitting this job; --resume removes error rows and retains completed rows.
export OPENROUTER_MAX_RETRIES=0

# Reuse the ffmpeg runtime prepared for the six-user generation jobs. The
# dedicated runtime is preferred; the Qwen environment and inherited PATH are
# fallbacks for clusters where the dedicated environment was moved.
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"
if [[ -x "${FFMPEG_ENV}/bin/ffmpeg" ]]; then
  export PATH="${FFMPEG_ENV}/bin:${PATH}"
  if [[ -d "${FFMPEG_ENV}/lib" ]]; then
    export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
  fi
  FFMPEG_SOURCE="FFMPEG_ENV"
elif [[ -x "${TRAIN_ENV}/bin/ffmpeg" ]]; then
  export PATH="${TRAIN_ENV}/bin:${PATH}"
  FFMPEG_SOURCE="TRAIN_ENV"
elif command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG_SOURCE="inherited_PATH"
else
  echo "ffmpeg is required for the configured OpenRouter video transcodes" >&2
  echo "Checked ${FFMPEG_ENV}/bin/ffmpeg, ${TRAIN_ENV}/bin/ffmpeg, and PATH." >&2
  exit 1
fi
printf 'ffmpeg_source=%s ffmpeg_binary=%s\n' "${FFMPEG_SOURCE}" "$(command -v ffmpeg)"
ffmpeg -version | head -n 1

"${PYTHON_BIN}" -m egolife_two_user_qa.six_vs_minimal_openrouter \
  --qa "${ACCEPTED_QA}" \
  --evidence "${EVIDENCE_JSONL}" \
  --output-dir "${OUTPUT_DIR}" \
  --expected-qa-count 44 \
  --backend openrouter \
  --model-id google/gemini-2.5-flash \
  --turbo \
  --reasoning-effort none \
  --max-new-tokens 64 \
  --resume
