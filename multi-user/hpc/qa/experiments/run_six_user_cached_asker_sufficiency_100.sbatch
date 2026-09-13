#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_askscope100
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

# Experimental rerun of the completed six-user 100-packet job.
#
# Frozen differences from the original production run:
#   1. Reuse the already materialized six_user_candidates.jsonl and all media
#      referenced by it. Do not rebuild the manifest, mine groups, run CLIP, or
#      modify the source run.
#   2. Replace forced A-E answer selection with the video-grounded
#      answerability_evidence_sufficiency_v1 boolean contract.
#   3. Evaluate only the full asker/speaker video. The gate passes exactly when
#      that video scope is judged insufficient to determine one unique option.
#   4. Keep generation, text formality, six-video groundedness, retry count,
#      model, decoding, and media routing aligned with the prior job.

RUN_MODE="six_user_cached_asker_sufficiency_100"
PACKET_COUNT="${PACKET_COUNT:-100}"
ACCEPTED_TARGET="${ACCEPTED_TARGET:-100}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"
ANSWERABILITY_SCHEMA_VERSION="answerability_evidence_sufficiency_v1"
ANSWERABILITY_CONDITION_SCOPE="asker_only"

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
SOURCE_RUN_DIR="${SOURCE_RUN_DIR:-${PACKAGE_ROOT}/outputs/six_user_qa/six_user_qa_packets_100_16160583}"
SOURCE_EVIDENCE="${SOURCE_EVIDENCE:-${SOURCE_RUN_DIR}/six_user_candidates.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs/six_user_qa}"
OUTDIR="${OUTPUT_ROOT}/${RUN_MODE}_${SLURM_JOB_ID}"

TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/scratch/${USER}/hf_cache}"
MODEL_ID="${MODEL_ID:-}"
PYTHON="${TRAIN_ENV}/bin/python"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/${RUN_MODE}_${SLURM_JOB_ID}"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-12.0}"
CUDA_KEEPER_PID=""
CUDA_KEEPER_LOG="${OUTDIR}/cuda_keeper_${SLURM_JOB_ID}.log"
STAGE_STATUS_PATH="${OUTDIR}/stage_status.json"
CURRENT_STAGE="bootstrap"

write_stage_status() {
  local status_value="$1"
  local stage_value="$2"
  local exit_code_value="${3:-0}"
  local signal_value="${4:-}"
  if [[ -d "${OUTDIR}" ]]; then
    printf '{"status":"%s","stage":"%s","timestamp":"%s","job_id":"%s","host":"%s","exit_code":%s,"signal":"%s"}\n' \
      "${status_value}" "${stage_value}" "$(date --iso-8601=seconds)" \
      "${SLURM_JOB_ID:-unknown}" "$(hostname)" "${exit_code_value}" \
      "${signal_value}" > "${STAGE_STATUS_PATH}"
  fi
}

stage_start() {
  CURRENT_STAGE="$1"
  printf 'stage=%s status=started timestamp=%s job_id=%s host=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" \
    "${SLURM_JOB_ID:-unknown}" "$(hostname)"
  write_stage_status "running" "${CURRENT_STAGE}" 0
}

stage_finish() {
  CURRENT_STAGE="$1"
  printf 'stage=%s status=completed timestamp=%s job_id=%s host=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" \
    "${SLURM_JOB_ID:-unknown}" "$(hostname)"
  write_stage_status "completed" "${CURRENT_STAGE}" 0
}

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    printf 'stage=stop_cuda_keeper status=started timestamp=%s pid=%s\n' \
      "$(date --iso-8601=seconds)" "${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}

on_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'stage=%s status=failed timestamp=%s exit_code=%s line=%s command=%q\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" "${exit_code}" \
    "${line_number}" "${failed_command}" >&2
  write_stage_status "failed" "${CURRENT_STAGE}" "${exit_code}"
  exit "${exit_code}"
}

on_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - INT TERM
  printf 'stage=%s status=cancelled timestamp=%s signal=%s exit_code=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" "${signal_name}" \
    "${exit_code}" >&2
  write_stage_status "cancelled" "${CURRENT_STAGE}" "${exit_code}" "${signal_name}"
  exit "${exit_code}"
}

trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap cleanup EXIT

stage_start "validate_static_inputs"
test -d "${PROJECT_ROOT}"
test -d "${PACKAGE_ROOT}"
test -s "${PACKAGE_ROOT}/__main__.py"
test -s "${PACKAGE_ROOT}/prompts.py"
test -s "${PACKAGE_ROOT}/video_qa_loop.py"
test -s "${SOURCE_EVIDENCE}"
test -x "${PYTHON}"
test -s "${CUDA_KEEPER_SCRIPT}"
if [[ "${OUTDIR}" == "${SOURCE_RUN_DIR}" ]]; then
  echo "output directory must differ from the read-only source run" >&2
  exit 2
fi
if [[ "${PACKET_COUNT}" -lt 1 || "${ACCEPTED_TARGET}" -lt 1 || "${MAX_ATTEMPTS}" -lt 1 ]]; then
  echo "PACKET_COUNT, ACCEPTED_TARGET, and MAX_ATTEMPTS must all be positive" >&2
  exit 2
fi
if [[ "${ACCEPTED_TARGET}" -gt "${PACKET_COUNT}" ]]; then
  echo "ACCEPTED_TARGET cannot exceed PACKET_COUNT" >&2
  exit 2
fi

# Resolve the existing Hugging Face snapshot to a concrete read-only model
# directory. This lets every mutable Hugging Face cache remain job-specific.
if [[ -z "${MODEL_ID}" ]]; then
  MODEL_REF="${MODEL_CACHE_ROOT}/hub/models--Qwen--Qwen3.6-27B/refs/main"
  test -s "${MODEL_REF}"
  MODEL_REVISION="$(tr -d '\r\n' < "${MODEL_REF}")"
  MODEL_ID="${MODEL_CACHE_ROOT}/hub/models--Qwen--Qwen3.6-27B/snapshots/${MODEL_REVISION}"
fi
test -s "${MODEL_ID}/config.json"

stage_start "create_job_scoped_paths"
mkdir -p \
  "${OUTDIR}" \
  "${JOB_SCRATCH_ROOT}/home" \
  "${JOB_SCRATCH_ROOT}/xdg_cache" \
  "${JOB_SCRATCH_ROOT}/hf_home" \
  "${JOB_SCRATCH_ROOT}/hf_datasets" \
  "${JOB_SCRATCH_ROOT}/modelscope" \
  "${JOB_SCRATCH_ROOT}/torch" \
  "${JOB_SCRATCH_ROOT}/triton" \
  "${JOB_SCRATCH_ROOT}/torchinductor" \
  "${JOB_SCRATCH_ROOT}/vllm" \
  "${JOB_SCRATCH_ROOT}/cuda" \
  "${JOB_SCRATCH_ROOT}/flashinfer" \
  "${JOB_SCRATCH_ROOT}/tmp" \
  "${JOB_SCRATCH_ROOT}/conda_pkgs" \
  "${JOB_SCRATCH_ROOT}/pip_cache" \
  "${JOB_SCRATCH_ROOT}/qwen_video_cache" \
  "${PYTHON_ALIAS_ROOT}"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf_home"
export HF_HUB_CACHE="${JOB_SCRATCH_ROOT}/hf_home/hub"
export TRANSFORMERS_CACHE="${JOB_SCRATCH_ROOT}/hf_home/transformers"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/hf_datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${JOB_SCRATCH_ROOT}/tmp"
export TEMP="${JOB_SCRATCH_ROOT}/tmp"
export CONDA_PKGS_DIRS="${JOB_SCRATCH_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR="${JOB_SCRATCH_ROOT}/qwen_video_cache"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=decord
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export QWEN_MEMORY_SAFE_VIDEO_FPS=1.0
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS=3136
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS=32768
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB=5
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB=32
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION=sdpa
export QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS=1
export QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE=512

stage_start "activate_and_pin_package"
cd "${PROJECT_ROOT}"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${TRAIN_ENV}"
ln -s "${PACKAGE_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
"${PYTHON}" -P - "${TRAIN_ENV}" "${PACKAGE_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

expected_env = Path(sys.argv[1]).resolve()
expected_package = Path(sys.argv[2]).resolve()
if Path(sys.prefix).resolve() != expected_env:
    raise SystemExit(f"wrong Python prefix: {sys.prefix}")
if Path(os.environ.get("CONDA_PREFIX", "")).resolve() != expected_env:
    raise SystemExit(f"wrong CONDA_PREFIX: {os.environ.get('CONDA_PREFIX')}")

import egolife_two_user_qa
import egolife_two_user_qa.video_qa_loop as video_qa_loop

actual_package = Path(egolife_two_user_qa.__file__).resolve().parent
actual_loop = Path(video_qa_loop.__file__).resolve().parent
if actual_package != expected_package or actual_loop != expected_package:
    raise SystemExit(
        f"wrong package import: expected={expected_package} "
        f"package={actual_package} loop={actual_loop}"
    )
print(f"python={sys.executable}")
print(f"package={actual_package}")
PY

stage_start "resolve_ffmpeg_runtime"
FFMPEG_BINARY=""
FFPROBE_BINARY=""
FFMPEG_SOURCE=""
if [[ -x "${FFMPEG_ENV}/bin/ffmpeg" && -x "${FFMPEG_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${FFMPEG_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${FFMPEG_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="FFMPEG_ENV"
  if [[ -d "${FFMPEG_ENV}/lib" ]]; then
    export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
  fi
elif [[ -x "${TRAIN_ENV}/bin/ffmpeg" && -x "${TRAIN_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${TRAIN_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${TRAIN_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="TRAIN_ENV"
else
  FFMPEG_BINARY="$(command -v ffmpeg || true)"
  FFPROBE_BINARY="$(command -v ffprobe || true)"
  FFMPEG_SOURCE="activated_PATH"
fi
if [[ -z "${FFMPEG_BINARY}" || ! -x "${FFMPEG_BINARY}" || \
      -z "${FFPROBE_BINARY}" || ! -x "${FFPROBE_BINARY}" ]]; then
  printf 'FFmpeg runtime not found. Checked FFMPEG_ENV=%s, TRAIN_ENV=%s, and activated PATH=%s\n' \
    "${FFMPEG_ENV}" "${TRAIN_ENV}" "${PATH}" >&2
  printf 'ffmpeg_candidate=%s ffprobe_candidate=%s\n' \
    "${FFMPEG_BINARY:-missing}" "${FFPROBE_BINARY:-missing}" >&2
  exit 1
fi
export FFMPEG_BINARY FFPROBE_BINARY FFMPEG_SOURCE
export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"
printf 'stage=resolve_ffmpeg_runtime status=ready timestamp=%s source=%s ffmpeg=%s ffprobe=%s\n' \
  "$(date --iso-8601=seconds)" "${FFMPEG_SOURCE}" \
  "${FFMPEG_BINARY}" "${FFPROBE_BINARY}"

stage_start "storage_preflight"
"${PYTHON}" -P -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"

stage_start "dependency_video_and_cuda_preflight"
PIP_CHECK_STATUS="passed"
if "${PYTHON}" -P -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then
  cat "${OUTDIR}/pip_check.txt"
else
  cat "${OUTDIR}/pip_check.txt"
  UNEXPECTED_PIP_CHECK="$(
    grep -Ev '^decord [^[:space:]]+ is not supported on this platform$' \
      "${OUTDIR}/pip_check.txt" || true
  )"
  if [[ -n "${UNEXPECTED_PIP_CHECK}" ]]; then
    printf 'unexpected dependency errors:\n%s\n' "${UNEXPECTED_PIP_CHECK}" >&2
    exit 2
  fi
  PIP_CHECK_STATUS="known_decord_platform_metadata_warning"
fi
"${FFMPEG_BINARY}" -version | head -n 1
"${FFPROBE_BINARY}" -version | head -n 1
"${PYTHON}" -P - <<'PY'
import decord
import torch
from qwen_vl_utils.vision_process import get_video_reader_backend
from torchcodec.decoders import VideoDecoder

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
backend = get_video_reader_backend()
if backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {backend}")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"decord={decord.__version__}")
print(f"video_backend={backend}")
print(f"torchcodec_decoder={VideoDecoder.__module__}")
PY

stage_start "start_cuda_keeper"
"${PYTHON}" -P -c "import torch, pynvml, psutil"
"${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}" \
  --threshold "${CUDA_KEEPER_THRESHOLD}" \
  --gpus "${CUDA_KEEPER_GPUS}" \
  --reserve "${CUDA_KEEPER_RESERVE}" \
  > "${CUDA_KEEPER_LOG}" 2>&1 &
CUDA_KEEPER_PID=$!
sleep 3
if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
  echo "CUDA keeper exited during startup: ${CUDA_KEEPER_LOG}" >&2
  tail -n 100 "${CUDA_KEEPER_LOG}" >&2 || true
  exit 2
fi

stage_start "snapshot_cached_evidence"
cp -- "${SOURCE_EVIDENCE}" "${OUTDIR}/six_user_candidates.jsonl"
if [[ -s "${SOURCE_RUN_DIR}/manifest.json" ]]; then
  cp -- "${SOURCE_RUN_DIR}/manifest.json" "${OUTDIR}/source_manifest.json"
fi
sha256sum "${SOURCE_EVIDENCE}" > "${OUTDIR}/source_evidence.sha256"
"${PYTHON}" -P - "${OUTDIR}/six_user_candidates.jsonl" "${PACKET_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_count = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != expected_count:
    raise SystemExit(f"expected {expected_count} cached packets, found {len(rows)}")

ids = [str(row.get("evidence_id") or "") for row in rows]
if not all(ids) or len(ids) != len(set(ids)):
    raise SystemExit("cached packet evidence IDs must be non-empty and unique")

for row in rows:
    users = list(row.get("required_users") or [])
    clips = list(row.get("clips") or [])
    if row.get("candidate_type") != "six_user_speaker_consensus":
        raise SystemExit(f"{row.get('evidence_id')}: candidate_type mismatch")
    if len(users) != 6 or len(clips) != 6:
        raise SystemExit(f"{row.get('evidence_id')}: expected six ordered users and clips")
    if [str(clip.get("agent_name") or "") for clip in clips] != users:
        raise SystemExit(f"{row.get('evidence_id')}: clip/user order mismatch")
    for index, clip in enumerate(clips):
        full_video = Path(str(clip.get("full_local_video") or ""))
        if not full_video.is_file():
            raise SystemExit(f"{row.get('evidence_id')}: missing full video {full_video}")
        frames = list(clip.get("frames") or [])
        if not frames:
            raise SystemExit(f"{row.get('evidence_id')}: clip {index} has no generator frames")
        missing_frames = [
            str(frame.get("path") or "")
            for frame in frames
            if not Path(str(frame.get("path") or "")).is_file()
        ]
        if missing_frames:
            raise SystemExit(
                f"{row.get('evidence_id')}: clip {index} has missing frames: {missing_frames[:3]}"
            )
print(f"cached_evidence_contract=passed packet_count={len(rows)}")
PY

stage_start "write_job_manifest"
BRANCH_NAME="$(git -C "${PROJECT_ROOT}" branch --show-current 2>/dev/null || true)"
COMMIT_SHA="$(git -C "${PROJECT_ROOT}" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$(git -C "${PROJECT_ROOT}" status --porcelain 2>/dev/null || true)" ]]; then
  DIRTY_STATE="dirty"
else
  DIRTY_STATE="clean"
fi
"${PYTHON}" -P - "${OUTDIR}/job_manifest.json" <<PY
import json
from pathlib import Path

payload = {
    "status": "running",
    "job_id": "${SLURM_JOB_ID}",
    "run_mode": "${RUN_MODE}",
    "branch": "${BRANCH_NAME}",
    "commit_sha": "${COMMIT_SHA}",
    "dirty_state": "${DIRTY_STATE}",
    "project_root": "${PROJECT_ROOT}",
    "package_root": "${PACKAGE_ROOT}",
    "source_run_dir": "${SOURCE_RUN_DIR}",
    "source_evidence": "${SOURCE_EVIDENCE}",
    "source_run_read_only": True,
    "output_dir": "${OUTDIR}",
    "job_scratch_root": "${JOB_SCRATCH_ROOT}",
    "model_id": "${MODEL_ID}",
    "packet_count": int("${PACKET_COUNT}"),
    "accepted_target": int("${ACCEPTED_TARGET}"),
    "max_attempts": int("${MAX_ATTEMPTS}"),
    "allow_partial": bool(int("${ALLOW_PARTIAL}")),
    "generator_media": "cached speaker-all/provider-retained sampled frames",
    "groundedness_media": "all six full original videos",
    "answerability_media": "full asker video only",
    "answerability_schema_version": "${ANSWERABILITY_SCHEMA_VERSION}",
    "answerability_condition_scope": "${ANSWERABILITY_CONDITION_SCOPE}",
    "answerability_uses_declared_correct": False,
    "ffmpeg_source": "${FFMPEG_SOURCE}",
    "ffmpeg_binary": "${FFMPEG_BINARY}",
    "ffprobe_binary": "${FFPROBE_BINARY}",
    "pip_check_status": "${PIP_CHECK_STATUS}",
    "pip_check_path": "${OUTDIR}/pip_check.txt",
}
Path("${OUTDIR}/job_manifest.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

stage_start "generate_with_asker_only_sufficiency"
"${PYTHON}" -P - \
  "${OUTDIR}/six_user_candidates.jsonl" \
  "${OUTDIR}/qa_mcq.jsonl" \
  "${OUTDIR}/video_first_prompts.jsonl" \
  "${OUTDIR}/qa_mcq.rejected.jsonl" \
  "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  "${ACCEPTED_TARGET}" \
  "${MAX_ATTEMPTS}" \
  "${MODEL_ID}" <<'PY'
from __future__ import annotations

import json
import sys
from typing import Any

from egolife_two_user_qa import prompts as prompt_module
from egolife_two_user_qa import video_qa_loop


SCHEMA_VERSION = "answerability_evidence_sufficiency_v1"
CONDITION_SCOPE = "asker_only"
ANSWERABILITY_SCHEMA = {
    "answerable": True,
    "reason": "short evidence-sufficiency explanation grounded only in the provided condition",
    "missing_information": "empty string if answerable; otherwise the exact missing or ambiguous fact",
}


def build_asker_only_condition(required_users: list[str]) -> list[dict[str, Any]]:
    users = list(required_users)
    if len(users) != 6 or not users[0]:
        raise ValueError(
            "the asker-only six-user experiment requires exactly six ordered required_users"
        )
    return [
        {
            "condition_id": f"speaker_only::{users[0]}",
            "condition_type": "speaker_only",
            "users": [users[0]],
        }
    ]


def build_sufficiency_prompt(qa_item: dict[str, Any], condition: dict[str, Any]) -> str:
    options = "\n".join(
        f"{letter}. {option}"
        for letter, option in zip("ABCDE", qa_item.get("options", []))
    )
    strict_json = prompt_module.STRICT_JSON_OUTPUT_CONTRACT
    return f"""You are the answerability judge for a six-user EgoLife multiple-choice question.

Decide, using only the full asker video provided for this condition, whether this video scope is sufficient for a competent viewer to determine one uniquely supported answer. Evaluate evidence sufficiency; do not answer the question and do not report an option letter or option text.

{strict_json}

Condition:
{json.dumps(condition, ensure_ascii=False, indent=2)}

Question:
{qa_item.get("question")}

Options:
{options}

Rules:
- This is an asker-only scope audit. Use only the one full, unpruned asker video attached to this call.
- Set answerable to true only when that asker video contains the question-relevant facts needed to distinguish one option from all alternatives without guessing.
- Set answerable to false when a required external fact, continuation, state, identity link, endpoint, or other answer-bearing detail is absent, unclear, or ambiguous in the asker video. State the exact missing information.
- Judge what this video scope can establish. Do not run a forced-choice trial, do not guess an option, and do not compare any output with the dataset's declared answer key.
- The options define the possible answers but are not evidence. Do not infer support from which option sounds detailed, natural, or likely.
- Do not use omitted provider videos, filenames, common-sense priors, or memory from any other condition.
- The reason must name the concrete visible support when answerable, or the concrete scope gap when unanswerable.
- missing_information must be empty when answerable is true and must concisely name the unresolved fact when answerable is false.

Return exactly one valid JSON object with this exact shape:
{json.dumps(ANSWERABILITY_SCHEMA, ensure_ascii=False, indent=2)}
"""


def asker_only_gate(
    qa_item: dict[str, Any], evaluations: list[dict[str, Any]]
) -> dict[str, Any]:
    required_users = list(qa_item.get("required_users") or [])
    asker = required_users[0] if len(required_users) == 6 else None
    base = {
        "schema_version": SCHEMA_VERSION,
        "condition_scope": CONDITION_SCOPE,
        "speaker_user": asker,
        "answerability_evaluated_condition_count": len(evaluations),
        "uses_declared_correct_answer": False,
    }
    if not asker:
        return {
            "passed": False,
            "reason": "missing ordered six-user asker metadata",
            "failure_label": "asker_metadata_missing",
            **base,
        }
    if len(evaluations) != 1:
        return {
            "passed": False,
            "reason": f"expected exactly one asker-only evaluation, found {len(evaluations)}",
            "failure_label": "asker_condition_count_invalid",
            **base,
        }
    evaluation = evaluations[0]
    expected_condition_id = f"speaker_only::{asker}"
    if (
        evaluation.get("condition_id") != expected_condition_id
        or evaluation.get("condition_type") != "speaker_only"
        or list(evaluation.get("users") or []) != [asker]
    ):
        return {
            "passed": False,
            "reason": "the sole answerability evaluation is not the ordered asker-only condition",
            "failure_label": "asker_condition_invalid",
            **base,
        }
    answerable = evaluation.get("answerable")
    if type(answerable) is not bool:
        return {
            "passed": False,
            "reason": "asker-only answerability did not return a JSON boolean",
            "failure_label": "asker_answerability_unparsed",
            "asker_answerable": None,
            **base,
        }
    if answerable:
        return {
            "passed": False,
            "reason": "the full asker-video scope was judged sufficient to determine one unique option",
            "failure_label": "asker_scope_answerable",
            "asker_answerable": True,
            "asker_evaluation": evaluation,
            **base,
        }
    return {
        "passed": True,
        "reason": "the full asker-video scope was judged insufficient to determine one unique option",
        "failure_label": None,
        "asker_answerable": False,
        "asker_evaluation": evaluation,
        **base,
    }


# Refuse to run against an unrelated package revision. The completed pilot's
# maintained six-user path has exactly speaker-only and all-six conditions.
probe_users = ["asker", "p1", "p2", "p3", "p4", "p5"]
base_conditions = video_qa_loop.build_answerability_conditions(probe_users)
if [row.get("condition_type") for row in base_conditions] != [
    "speaker_only",
    "combined_all_six_users",
]:
    raise SystemExit(
        "unexpected base six-user answerability contract; refusing experiment override: "
        + repr(base_conditions)
    )

original_run_answerability_eval = video_qa_loop.run_answerability_eval
video_qa_loop.build_answerability_conditions = build_asker_only_condition
video_qa_loop.build_answerability_prompt = build_sufficiency_prompt
video_qa_loop.answerability_gate = asker_only_gate
prompt_module.ANSWERABILITY_SCHEMA = ANSWERABILITY_SCHEMA
prompt_module.ANSWERABILITY_SCHEMA_VERSION = SCHEMA_VERSION


def run_asker_only_answerability_eval(**kwargs: Any) -> dict[str, Any]:
    prompt_rows = kwargs["prompt_rows"]
    start_index = len(prompt_rows)
    result = original_run_answerability_eval(**kwargs)
    for row in prompt_rows[start_index:]:
        row["answerability_schema_version"] = SCHEMA_VERSION
        row["answerability_condition_scope"] = CONDITION_SCOPE
    result["schema_version"] = SCHEMA_VERSION
    result["condition_scope"] = CONDITION_SCOPE
    for evaluation in result.get("evaluations") or []:
        evaluation["schema_version"] = SCHEMA_VERSION
        evaluation["condition_scope"] = CONDITION_SCOPE
    return result


video_qa_loop.run_answerability_eval = run_asker_only_answerability_eval

# Contract preflight: the gate must ignore even a deliberately invalid answer
# key, and the rendered output contract must not request an option selection.
probe_qa = {
    "question": "Which option is supported?",
    "options": ["One", "Two", "Three", "Four", "Five"],
    "correct": "DELIBERATELY_NOT_READ",
    "required_users": probe_users,
}
probe_condition = build_asker_only_condition(probe_users)[0]
probe_gate = asker_only_gate(
    probe_qa,
    [
        {
            **probe_condition,
            "answerable": False,
            "reason": "The requested external detail is absent.",
            "missing_information": "the requested external detail",
        }
    ],
)
if probe_gate.get("passed") is not True:
    raise SystemExit(f"asker-only boolean gate preflight failed: {probe_gate!r}")
probe_prompt = build_sufficiency_prompt(probe_qa, probe_condition)
output_contract = probe_prompt[probe_prompt.rfind("Return exactly one valid JSON object") :]
required_phrases = (
    "Evaluate evidence sufficiency",
    "do not answer the question",
    "Do not run a forced-choice trial",
    "do not compare any output with the dataset's declared answer key",
)
if any(phrase not in probe_prompt for phrase in required_phrases):
    raise SystemExit("asker-only sufficiency prompt is missing a frozen contract phrase")
if '"choice"' in output_contract or '"answer_text"' in output_contract:
    raise SystemExit("asker-only sufficiency output contract requests an answer selection")
print("asker_only_answerability_contract=passed")

(
    evidence_path,
    accepted_path,
    prompts_path,
    rejected_path,
    intermediate_path,
    accepted_target,
    max_attempts,
    model_id,
) = sys.argv[1:]

raise SystemExit(
    video_qa_loop.main(
        [
            "--evidence",
            evidence_path,
            "--output",
            accepted_path,
            "--prompts-output",
            prompts_path,
            "--rejected-output",
            rejected_path,
            "--intermediate-output",
            intermediate_path,
            "--target-count",
            accepted_target,
            "--max-attempts",
            max_attempts,
            "--backend",
            "transformers-local-memory-safe",
            "--model-id",
            model_id,
            "--dtype",
            "bfloat16",
            "--max-image-pixels",
            "65536",
            "--max-new-tokens",
            "2048",
            "--disable-thinking",
            "--generation-mode",
            "baseline",
            "--judge-video-source",
            "full",
            "--fixed-question-type-schedule",
            "--question-types",
            "neutral",
        ]
    )
)
PY

stage_start "validate_asker_only_outputs"
"${PYTHON}" -P - \
  "${OUTDIR}" \
  "${PACKET_COUNT}" \
  "${ACCEPTED_TARGET}" \
  "${MAX_ATTEMPTS}" \
  "${ALLOW_PARTIAL}" \
  "${ANSWERABILITY_SCHEMA_VERSION}" <<'PY'
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

(
    output_dir,
    packet_count,
    accepted_target,
    max_attempts,
    allow_partial,
    schema_version,
) = (
    Path(sys.argv[1]),
    int(sys.argv[2]),
    int(sys.argv[3]),
    int(sys.argv[4]),
    bool(int(sys.argv[5])),
    sys.argv[6],
)


def read_jsonl(name):
    path = output_dir / name
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


candidates = read_jsonl("six_user_candidates.jsonl")
accepted = read_jsonl("qa_mcq.jsonl")
rejected = read_jsonl("qa_mcq.rejected.jsonl")
prompts = read_jsonl("video_first_prompts.jsonl")
intermediate = read_jsonl("qa_mcq.intermediate.jsonl")

if len(candidates) != packet_count:
    raise SystemExit(f"expected {packet_count} cached packets, found {len(candidates)}")
candidate_ids = [str(row.get("evidence_id") or "") for row in candidates]
outcome_ids = [str(row.get("evidence_id") or "") for row in accepted + rejected]
if len(outcome_ids) != len(set(outcome_ids)):
    raise SystemExit("accepted/rejected outcomes contain duplicate evidence IDs")
if set(outcome_ids) != set(candidate_ids):
    raise SystemExit("accepted/rejected outcomes do not cover every cached packet exactly once")

answerability_prompts = [row for row in prompts if row.get("stage") == "answerability"]
if not answerability_prompts:
    raise SystemExit("no packet reached the asker-only answerability judge")
prompts_per_attempt = defaultdict(int)
for row in answerability_prompts:
    if row.get("answerability_schema_version") != schema_version:
        raise SystemExit("recorded answerability prompt has the wrong schema version")
    if row.get("answerability_condition_scope") != "asker_only":
        raise SystemExit("recorded answerability prompt has the wrong condition scope")
    if row.get("condition_id", "").split("::", 1)[0] != "speaker_only":
        raise SystemExit("non-asker answerability condition was executed")
    if len(row.get("video_paths") or []) != 1 or row.get("image_paths"):
        raise SystemExit("asker-only answerability must receive exactly one full video")
    if row.get("media_role") != "full":
        raise SystemExit("asker-only answerability did not use full media")
    prompt = str(row.get("prompt") or "")
    output_contract = prompt[prompt.rfind("Return exactly one valid JSON object") :]
    if '"choice"' in output_contract or '"answer_text"' in output_contract:
        raise SystemExit("recorded answerability prompt requests an answer selection")
    prompts_per_attempt[(str(row.get("qa_id") or ""), row.get("attempt"))] += 1
if any(count != 1 for count in prompts_per_attempt.values()):
    raise SystemExit("an attempt executed more than one answerability condition")

verdict_counts = Counter()
invalid_verdict_count = 0
evaluated_attempt_count = 0
for row in intermediate:
    answerability = row.get("answerability")
    if not isinstance(answerability, dict) or not answerability:
        continue
    if answerability.get("schema_version") != schema_version:
        raise SystemExit("intermediate answerability result has the wrong schema version")
    if answerability.get("condition_scope") != "asker_only":
        raise SystemExit("intermediate answerability result has the wrong condition scope")
    evaluations = answerability.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        raise SystemExit("intermediate attempt did not record exactly one asker evaluation")
    evaluation = evaluations[0]
    if any(key in evaluation for key in ("choice", "answer_text", "evidence_used")):
        # A parse failure in the old runner fallback may retain these legacy keys;
        # record it as invalid instead of treating it as a sufficiency verdict.
        if evaluation.get("answerable") is not None:
            raise SystemExit("valid sufficiency evaluation retained legacy selection fields")
    answerable = evaluation.get("answerable")
    if type(answerable) is bool:
        verdict_counts[str(answerable).lower()] += 1
    else:
        invalid_verdict_count += 1
    evaluated_attempt_count += 1

for row in accepted:
    qa_id = str(row.get("qa_id") or "")
    users = list(row.get("required_users") or [])
    if len(users) != 6:
        raise SystemExit(f"{qa_id}: accepted row lacks six ordered users")
    answerability = (row.get("review") or {}).get("answerability") or {}
    if answerability.get("schema_version") != schema_version:
        raise SystemExit(f"{qa_id}: accepted row lacks the sufficiency schema")
    if answerability.get("condition_scope") != "asker_only":
        raise SystemExit(f"{qa_id}: accepted row has the wrong answerability scope")
    evaluations = answerability.get("evaluations")
    if not isinstance(evaluations, list) or len(evaluations) != 1:
        raise SystemExit(f"{qa_id}: accepted row lacks exactly one asker evaluation")
    evaluation = evaluations[0]
    if evaluation.get("condition_id") != f"speaker_only::{users[0]}":
        raise SystemExit(f"{qa_id}: accepted answerability condition does not match asker")
    if evaluation.get("answerable") is not False:
        raise SystemExit(f"{qa_id}: accepted asker scope was not judged unanswerable")
    if any(key in evaluation for key in ("choice", "answer_text", "evidence_used")):
        raise SystemExit(f"{qa_id}: accepted sufficiency result retains selection fields")
    gate = answerability.get("gate") or {}
    required_gate = {
        "passed": True,
        "condition_scope": "asker_only",
        "asker_answerable": False,
        "answerability_evaluated_condition_count": 1,
        "uses_declared_correct_answer": False,
    }
    for key, expected in required_gate.items():
        if gate.get(key) != expected:
            raise SystemExit(f"{qa_id}: invalid gate field {key}={gate.get(key)!r}")
    asker_videos = [
        str(item.get("full_local_video") or "")
        for item in (row.get("video_evidence") or [])
        if str(item.get("user") or "") == users[0]
    ]
    if len(asker_videos) != 1:
        raise SystemExit(f"{qa_id}: accepted row lacks one full asker video")
    condition_videos = ((evaluation.get("condition_media") or {}).get("video_paths") or [])
    if condition_videos != asker_videos:
        raise SystemExit(f"{qa_id}: answerability media is not the full asker video")

if len(accepted) >= accepted_target:
    status = "passed"
elif allow_partial and accepted:
    status = "partial"
else:
    status = "failed"

result = {
    "status": status,
    "experiment": "six_user_cached_asker_answerability_sufficiency",
    "answerability_schema_version": schema_version,
    "answerability_condition_scope": "asker_only",
    "source_packet_count": len(candidates),
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
    "accepted_target": accepted_target,
    "max_attempts": max_attempts,
    "allow_partial": allow_partial,
    "answerability_prompt_count": len(answerability_prompts),
    "evaluated_attempt_count": evaluated_attempt_count,
    "asker_scope_verdict_counts": dict(sorted(verdict_counts.items())),
    "invalid_verdict_count": invalid_verdict_count,
    "answerability_calls_per_evaluated_attempt": 1,
    "answerability_video_count_per_call": 1,
    "answerability_media_role": "full",
    "gate_uses_declared_correct_answer": False,
    "answer_selection_fields_present_in_accepted": False,
}
(output_dir / "six_user_asker_sufficiency_result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
manifest_path = output_dir / "job_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = status
manifest["result_path"] = str(output_dir / "six_user_asker_sufficiency_result.json")
manifest_path.write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(result, ensure_ascii=False, sort_keys=True))
if status == "failed":
    raise SystemExit(
        f"asker-only research gate failed: accepted={len(accepted)} target={accepted_target}"
    )
PY

stage_start "validate_accepted_artifacts"
"${PYTHON}" -P -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review

stage_start "materialize_review_videos"
"${PYTHON}" -P -m egolife_two_user_qa materialize_review_videos \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --output-dir "${OUTDIR}/review_videos"

stage_finish "six_user_cached_asker_sufficiency_complete"
printf 'source_run=%s\noutput_dir=%s\nresult=%s\n' \
  "${SOURCE_RUN_DIR}" "${OUTDIR}" \
  "${OUTDIR}/six_user_asker_sufficiency_result.json"

