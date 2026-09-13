#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_10m
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --mem=320G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

RUN_MODE="${RUN_MODE:-six_user_qa_10min_time_aware}"
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/outputs/six_user_qa_10min}"
OUTDIR="${OUTPUT_BASE}/${RUN_MODE}_${SLURM_JOB_ID}"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/${RUN_MODE}_${SLURM_JOB_ID}"
CACHE_DIR="${JOB_SCRATCH_ROOT}/media_cache"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"

TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
MODEL_ID="${MODEL_ID:-Qwen/Qwen3.6-27B}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
TARGET_COUNT="${TARGET_COUNT:-20}"
SOURCE_WINDOW_COUNT="${SOURCE_WINDOW_COUNT:-40}"
MAX_GROUPS="${MAX_GROUPS:-1000}"
EVIDENCE_RANDOM_SEED="${EVIDENCE_RANDOM_SEED:-20260831}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
JUDGE_VIDEO_FPS="${JUDGE_VIDEO_FPS:-0.75}"
GENERATOR_MAX_IMAGE_PIXELS="${GENERATOR_MAX_IMAGE_PIXELS:-65536}"
SIX_USER_JUDGE_MODE="${SIX_USER_JUDGE_MODE:-time-aware-map-reduce}"
COHORT_SKIP_SOURCE="${COHORT_SKIP_SOURCE:-}"
SKIP_EVIDENCE_IDS="${SKIP_EVIDENCE_IDS:-}"
QA_PACKET_LIMIT="${QA_PACKET_LIMIT:-}"
REQUIRED_GPU_COUNT=2
REUSE_PREPROCESSED_FROM="${REUSE_PREPROCESSED_FROM:-}"
RESUME_QA_FROM="${RESUME_QA_FROM:-}"
if [[ -n "${RESUME_QA_FROM}" && -z "${REUSE_PREPROCESSED_FROM}" ]]; then
  REUSE_PREPROCESSED_FROM="${RESUME_QA_FROM}"
fi
EXCLUDE_PROCESSED_FROM="${EXCLUDE_PROCESSED_FROM:-${OUTPUT_BASE}}"
if [[ -z "${REQUIRE_FRESH_EVIDENCE+x}" ]]; then
  if [[ -n "${REUSE_PREPROCESSED_FROM}" ]]; then
    REQUIRE_FRESH_EVIDENCE=0
  else
    REQUIRE_FRESH_EVIDENCE=1
  fi
fi
QA_RESUME_ARGS=()
QA_SELECTION_ARGS=()
EVIDENCE_FRESHNESS_ARGS=()

CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="0,1"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-12.0}"
CUDA_KEEPER_PID=""
CUDA_KEEPER_LOG="${OUTDIR}/cuda_keeper_${SLURM_JOB_ID}.log"
CURRENT_STAGE="bootstrap"

stage() {
  CURRENT_STAGE="$1"
  printf 'stage=%s status=started timestamp=%s job_id=%s host=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" "${SLURM_JOB_ID}" "$(hostname)"
}

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}

on_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'stage=%s status=failed exit_code=%s line=%s command=%q\n' \
    "${CURRENT_STAGE}" "${exit_code}" "${line_number}" "${failed_command}" >&2
  exit "${exit_code}"
}

trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap cleanup EXIT INT TERM

stage "validate_project_layout"
cd "${PROJECT_ROOT}"
test -d "${HPC_ROOT}"
test -s "${HPC_ROOT}/shared/env_qwen3vl.sh"
test -s "${CUDA_KEEPER_SCRIPT}"
test -d "${PACKAGE_ROOT}"
test -s "${PACKAGE_ROOT}/ten_minute_six_user_setup.py"
test -s "${PACKAGE_ROOT}/group_relative_clip_sampling.py"
case "${SIX_USER_JUDGE_MODE}" in
  time-aware-map-reduce|legacy-zero-shot) ;;
  *)
    echo "error=invalid_six_user_judge_mode value=${SIX_USER_JUDGE_MODE}" >&2
    exit 2
    ;;
esac
if [[ -n "${QA_PACKET_LIMIT}" && ! "${QA_PACKET_LIMIT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error=invalid_qa_packet_limit value=${QA_PACKET_LIMIT}" >&2
  exit 2
fi
if [[ ! "${REQUIRED_GPU_COUNT}" =~ ^[1-9][0-9]*$ ]]; then
  echo "error=invalid_required_gpu_count value=${REQUIRED_GPU_COUNT}" >&2
  exit 2
fi
if [[ "${REQUIRE_FRESH_EVIDENCE}" != "0" && "${REQUIRE_FRESH_EVIDENCE}" != "1" ]]; then
  echo "error=invalid_require_fresh_evidence value=${REQUIRE_FRESH_EVIDENCE}" >&2
  exit 2
fi
if [[ "${REQUIRE_FRESH_EVIDENCE}" == "1" && -n "${REUSE_PREPROCESSED_FROM}" ]]; then
  echo "error=fresh_evidence_cannot_reuse_preprocessed_packets" >&2
  exit 2
fi
if [[ "${REQUIRE_FRESH_EVIDENCE}" == "1" && -z "${EXCLUDE_PROCESSED_FROM}" ]]; then
  echo "error=fresh_evidence_requires_processed_history_root" >&2
  exit 2
fi
if [[ -n "${EXCLUDE_PROCESSED_FROM}" ]]; then
  EVIDENCE_FRESHNESS_ARGS+=(--exclude-processed-from "${EXCLUDE_PROCESSED_FROM}")
fi
mkdir -p "${HPC_ROOT}/logs" "${OUTDIR}" "${JOB_SCRATCH_ROOT}" \
  "${CACHE_DIR}" "${PYTHON_ALIAS_ROOT}"

stage "activate_environment"
export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-${TRAIN_ENV}}"
source "${HPC_ROOT}/shared/env_qwen3vl.sh"
PYTHON="$(command -v python)"
test -x "${PYTHON}"

stage "configure_pytorch_runtime_libraries"
PYTHON_SITE_PACKAGES="$("${PYTHON}" -P -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PYTORCH_CUDNN_LIB="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"
PYTORCH_CORE_LIB="${PYTHON_SITE_PACKAGES}/torch/lib"
TORCH_RUNTIME_LIBRARY_PATH=""
if compgen -G "${PYTORCH_CUDNN_LIB}/libcudnn.so*" >/dev/null; then
  TORCH_RUNTIME_LIBRARY_PATH="${PYTORCH_CUDNN_LIB}"
fi
if [[ -d "${PYTORCH_CORE_LIB}" ]]; then
  TORCH_RUNTIME_LIBRARY_PATH="${TORCH_RUNTIME_LIBRARY_PATH}${TORCH_RUNTIME_LIBRARY_PATH:+:}${PYTORCH_CORE_LIB}"
fi
if [[ -n "${TORCH_RUNTIME_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
printf 'python_site_packages=%s pytorch_runtime_library_path=%s\n' \
  "${PYTHON_SITE_PACKAGES}" "${TORCH_RUNTIME_LIBRARY_PATH:-bundled_rpath_only}"

stage "configure_job_storage"
if [[ -z "${HF_TOKEN:-}" && -s "${LOGIN_HF_TOKEN_PATH}" ]]; then
  export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"
fi
export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf_home"
export HF_HUB_CACHE="${JOB_SCRATCH_ROOT}/hf_hub"
export TRANSFORMERS_CACHE="${JOB_SCRATCH_ROOT}/transformers"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/hf_datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export CONDA_PKGS_DIRS="${JOB_SCRATCH_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export VLLM_NO_USAGE_STATS=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
# Qwen3.6 linear-attention prefill uses a very large depthwise Conv1d at long
# context lengths.  PyTorch can index that convolution safely only through the
# cuDNN V8 frontend (with cuDNN 9.3+).  A login-shell setting inherited through
# sbatch --export=ALL must not silently force the legacy, 32-bit-only path.
unset TORCH_CUDNN_V8_API_DISABLED
export TORCH_CUDNN_V8_API_DEBUG=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${HF_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" \
  "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" \
  "${TMPDIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"
printf 'hf_authentication=%s\n' \
  "$([[ -n "${HF_TOKEN:-}" || -s "${HF_TOKEN_PATH:-/nonexistent}" ]] && printf configured || printf anonymous)"

ln -s "${PACKAGE_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

stage "resolve_video_runtime"
FFMPEG_BINARY=""
FFPROBE_BINARY=""
FFMPEG_SOURCE=""
if [[ -x "${FFMPEG_ENV}/bin/ffmpeg" && -x "${FFMPEG_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${FFMPEG_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${FFMPEG_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="FFMPEG_ENV"
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
export FFMPEG_BINARY FFPROBE_BINARY
export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"
printf 'ffmpeg_source=%s ffmpeg=%s ffprobe=%s\n' \
  "${FFMPEG_SOURCE}" "${FFMPEG_BINARY}" "${FFPROBE_BINARY}"

stage "storage_preflight"
"${PYTHON}" -P -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"

stage "video_runtime_preflight"
export RUN_REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT}"
"${FFMPEG_BINARY}" -version | head -n 1
"${FFPROBE_BINARY}" -version | head -n 1
"${PYTHON}" -P - <<'PY'
import os

import torch
from torchcodec.decoders import VideoDecoder

required_gpu_count = int(os.environ["RUN_REQUIRED_GPU_COUNT"])
visible_gpu_count = torch.cuda.device_count()
if not torch.cuda.is_available() or visible_gpu_count < required_gpu_count:
    raise SystemExit(
        "PyTorch cannot access the required Slurm GPUs: "
        f"required={required_gpu_count} visible={visible_gpu_count}"
    )
print("torch_version=", torch.__version__)
for device_index in range(visible_gpu_count):
    properties = torch.cuda.get_device_properties(device_index)
    print(
        "gpu=",
        f"index={device_index}",
        f"name={properties.name}",
        f"total_memory_gib={properties.total_memory / 1024**3:.3f}",
    )
print("torchcodec_decoder=", VideoDecoder.__module__)
PY

stage "long_context_cudnn_preflight"
"${PYTHON}" -P - <<'PY'
import importlib.util
import os
import sysconfig
from pathlib import Path

import torch

site_packages = Path(sysconfig.get_paths()["purelib"])
cudnn_candidates = [
    str(path)
    for root in (
        site_packages / "nvidia" / "cudnn" / "lib",
        site_packages / "torch" / "lib",
    )
    for path in root.glob("libcudnn.so*")
]
try:
    cudnn_version = torch.backends.cudnn.version()
except RuntimeError as exc:
    raise SystemExit(
        "PyTorch could not load its matching bundled cuDNN. "
        f"error={exc}; candidates={cudnn_candidates}; "
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}"
    ) from exc
v8_disabled = os.environ.get("TORCH_CUDNN_V8_API_DISABLED")
v8_debug = os.environ.get("TORCH_CUDNN_V8_API_DEBUG")
print(
    "long_context_cudnn_preflight",
    f"cudnn_available={torch.backends.cudnn.is_available()}",
    f"cudnn_enabled={torch.backends.cudnn.enabled}",
    f"cudnn_version={cudnn_version}",
    f"v8_disabled={v8_disabled!r}",
    f"v8_debug={v8_debug!r}",
    f"causal_conv1d_available={importlib.util.find_spec('causal_conv1d') is not None}",
    f"fla_available={importlib.util.find_spec('fla') is not None}",
)
if not torch.backends.cudnn.is_available() or not torch.backends.cudnn.enabled:
    raise SystemExit(
        "Qwen3.6 long-context prefill requires an enabled cuDNN runtime"
    )
if cudnn_version is None or cudnn_version < 90300:
    raise SystemExit(
        "Qwen3.6 long-context prefill requires cuDNN 9.3+ for the 64-bit "
        f"depthwise Conv1d path; detected cudnn_version={cudnn_version!r}"
    )
if v8_disabled == "1":
    raise SystemExit(
        "TORCH_CUDNN_V8_API_DISABLED=1 would force the failing 32-bit Conv1d path"
    )
PY

stage "resolve_qa_cohort_exclusions"
declare -a COHORT_SKIP_IDS=()
declare -A SEEN_COHORT_SKIP_IDS=()
append_cohort_skip_id() {
  local evidence_id="$1"
  [[ -n "${evidence_id}" ]] || return 0
  if [[ -z "${SEEN_COHORT_SKIP_IDS[${evidence_id}]+present}" ]]; then
    SEEN_COHORT_SKIP_IDS["${evidence_id}"]=1
    COHORT_SKIP_IDS+=("${evidence_id}")
  fi
}
if [[ -n "${COHORT_SKIP_SOURCE}" ]]; then
  while IFS= read -r evidence_id; do
    append_cohort_skip_id "${evidence_id}"
  done < <("${PYTHON}" -P - "${COHORT_SKIP_SOURCE}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
if source.is_dir():
    source = source / "qa_mcq.rejected.jsonl"
if not source.is_file():
    raise SystemExit(f"cohort skip source is missing: {source}")
evidence_ids = []
for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    if row.get("status") != "skipped_by_user" and row.get("skipped_by_user") is not True:
        continue
    evidence_id = str(row.get("skip_evidence_id") or row.get("evidence_id") or "").strip()
    if not evidence_id:
        raise SystemExit(f"skipped_by_user row {line_number} has no evidence ID")
    if evidence_id not in evidence_ids:
        evidence_ids.append(evidence_id)
for evidence_id in evidence_ids:
    print(evidence_id)
PY
  )
fi
if [[ -n "${SKIP_EVIDENCE_IDS}" ]]; then
  IFS=',' read -r -a EXPLICIT_SKIP_IDS <<< "${SKIP_EVIDENCE_IDS}"
  for evidence_id in "${EXPLICIT_SKIP_IDS[@]}"; do
    evidence_id="${evidence_id#${evidence_id%%[![:space:]]*}}"
    evidence_id="${evidence_id%${evidence_id##*[![:space:]]}}"
    append_cohort_skip_id "${evidence_id}"
  done
fi
for evidence_id in "${COHORT_SKIP_IDS[@]}"; do
  QA_SELECTION_ARGS+=(--skip-evidence-id "${evidence_id}")
done
if [[ -n "${QA_PACKET_LIMIT}" ]]; then
  QA_SELECTION_ARGS+=(--packet-limit "${QA_PACKET_LIMIT}")
fi
printf '%s\n' "${COHORT_SKIP_IDS[@]}" > "${OUTDIR}/cohort_skip_evidence_ids.txt"
RUN_COHORT_SKIP_IDS=""
for evidence_id in "${COHORT_SKIP_IDS[@]}"; do
  RUN_COHORT_SKIP_IDS="${RUN_COHORT_SKIP_IDS}${RUN_COHORT_SKIP_IDS:+,}${evidence_id}"
done
export RUN_COHORT_SKIP_IDS
printf 'six_user_judge_mode=%s cohort_skip_count=%s qa_packet_limit=%s\n' \
  "${SIX_USER_JUDGE_MODE}" "${#COHORT_SKIP_IDS[@]}" "${QA_PACKET_LIMIT:-none}"

stage "start_cuda_keeper"
"${PYTHON}" -P -c "import torch, pynvml, psutil"
export RUN_CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS}"
"${PYTHON}" -P - <<'PY'
import os

import torch

required = int(os.environ["RUN_REQUIRED_GPU_COUNT"])
visible = torch.cuda.device_count()
selected_text = os.environ["RUN_CUDA_KEEPER_GPUS"].strip().lower()
selected = (
    list(range(visible))
    if selected_text == "all"
    else [int(value.strip()) for value in selected_text.split(",") if value.strip()]
)
expected = list(range(required))
if visible < required:
    raise SystemExit(f"required {required} GPUs but only {visible} are visible")
if selected != expected:
    raise SystemExit(
        "CUDA keeper must cover every requested local GPU exactly once: "
        f"expected={expected} selected={selected}"
    )
print(f"cuda_keeper_gpu_contract_passed required={required} selected={selected}")
PY
PYTHONUNBUFFERED=1 "${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}" \
  --threshold "${CUDA_KEEPER_THRESHOLD}" \
  --gpus "${CUDA_KEEPER_GPUS}" \
  --reserve "${CUDA_KEEPER_RESERVE}" \
  > "${CUDA_KEEPER_LOG}" 2>&1 &
CUDA_KEEPER_PID=$!
sleep 3
if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
  echo "error=cuda_keeper_exited_early log=${CUDA_KEEPER_LOG}" >&2
  sed -n '1,200p' "${CUDA_KEEPER_LOG}" >&2 || true
  exit 2
fi
printf 'cuda_keeper_pid=%s cuda_keeper_script=%s\n' \
  "${CUDA_KEEPER_PID}" "${CUDA_KEEPER_SCRIPT}"

stage "configure_judge_video_downsampling"
export FORCE_QWENVL_VIDEO_READER="decord"
export QWEN_MEMORY_SAFE_VIDEO_FPS="${JUDGE_VIDEO_FPS}"
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="3136"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="262144"
export QWEN_MEMORY_SAFE_ADAPTIVE_IMAGE_PIXELS="1"
export QWEN_MEMORY_SAFE_IMAGE_CONTEXT_TARGET_FRACTION="0.85"
export QWEN_MEMORY_SAFE_IMAGE_TEXT_TOKEN_RESERVE="8192"
export QWEN_MEMORY_SAFE_IMAGE_ITEM_TOKEN_OVERHEAD="2"
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB="5"
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="160"
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="sdpa"
export QWEN_MEMORY_SAFE_DEVICE_MAP="balanced"
export QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT="${REQUIRED_GPU_COUNT}"
export QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS="1"
export QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE="512"
export QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR="${JOB_SCRATCH_ROOT}/qwen_video_cache"
mkdir -p "${QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR}"

stage "write_job_manifest"
export RUN_MANIFEST_PATH="${OUTDIR}/run_manifest.json"
export RUN_PROJECT_ROOT="${PROJECT_ROOT}"
export RUN_HPC_ROOT="${HPC_ROOT}"
export RUN_PACKAGE_ROOT="${PACKAGE_ROOT}"
export RUN_OUTPUT_DIR="${OUTDIR}"
export RUN_JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT}"
export RUN_MODEL_ID="${MODEL_ID}"
export RUN_JUDGE_VIDEO_FPS="${JUDGE_VIDEO_FPS}"
export RUN_GENERATOR_MAX_IMAGE_PIXELS="${GENERATOR_MAX_IMAGE_PIXELS}"
export RUN_SIX_USER_JUDGE_MODE="${SIX_USER_JUDGE_MODE}"
export RUN_COHORT_SKIP_SOURCE="${COHORT_SKIP_SOURCE}"
export RUN_QA_PACKET_LIMIT="${QA_PACKET_LIMIT}"
export RUN_REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT}"
export RUN_CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS}"
export RUN_QWEN_DEVICE_MAP="${QWEN_MEMORY_SAFE_DEVICE_MAP}"
export RUN_QWEN_REQUIRED_MODEL_GPU_COUNT="${QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT}"
export RUN_REUSE_PREPROCESSED_FROM="${REUSE_PREPROCESSED_FROM}"
export RUN_RESUME_QA_FROM="${RESUME_QA_FROM}"
export RUN_EXCLUDE_PROCESSED_FROM="${EXCLUDE_PROCESSED_FROM}"
export RUN_REQUIRE_FRESH_EVIDENCE="${REQUIRE_FRESH_EVIDENCE}"
export RUN_EVIDENCE_RANDOM_SEED="${EVIDENCE_RANDOM_SEED}"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "project_root": os.environ["RUN_PROJECT_ROOT"],
    "hpc_root": os.environ["RUN_HPC_ROOT"],
    "package_root": os.environ["RUN_PACKAGE_ROOT"],
    "output_dir": os.environ["RUN_OUTPUT_DIR"],
    "job_scratch_root": os.environ["RUN_JOB_SCRATCH_ROOT"],
    "model_id": os.environ["RUN_MODEL_ID"],
    "pruning_sample_fps": 1.0,
    "pruning_cluster_window_seconds": 30.0,
    "pruning_clusters_per_window": 12,
    "judge_video_fps": float(os.environ["RUN_JUDGE_VIDEO_FPS"]),
    "judge_temporal_coverage": "full_600_seconds_per_user",
    "six_user_judge_mode": os.environ["RUN_SIX_USER_JUDGE_MODE"],
    "judge_review_mode": (
        "per_user_source_segment_map_reduce"
        if os.environ["RUN_SIX_USER_JUDGE_MODE"] == "time-aware-map-reduce"
        else "legacy_direct_zero_shot"
    ),
    "judge_visual_scope": (
        "one_user_per_visual_call"
        if os.environ["RUN_SIX_USER_JUDGE_MODE"] == "time-aware-map-reduce"
        else "speaker_only_then_all_six_full_videos"
    ),
    "generator_media": "complete_surviving_sampled_frames",
    "generator_configured_max_image_pixels": int(
        os.environ["RUN_GENERATOR_MAX_IMAGE_PIXELS"]
    ),
    "generator_adaptive_image_pixels": True,
    "generator_image_context_target_fraction": 0.85,
    "preprocessing_mode": (
        "reuse_previous_run"
        if os.environ["RUN_REUSE_PREPROCESSED_FROM"]
        else "build_new"
    ),
    "reuse_preprocessed_from": (
        os.environ["RUN_REUSE_PREPROCESSED_FROM"] or None
    ),
    "qa_resume_from": os.environ["RUN_RESUME_QA_FROM"] or None,
    "exclude_processed_from": os.environ["RUN_EXCLUDE_PROCESSED_FROM"] or None,
    "require_fresh_evidence": os.environ["RUN_REQUIRE_FRESH_EVIDENCE"] == "1",
    "evidence_random_seed": int(os.environ["RUN_EVIDENCE_RANDOM_SEED"]),
    "cohort_skip_source": os.environ["RUN_COHORT_SKIP_SOURCE"] or None,
    "cohort_skip_evidence_ids": [
        evidence_id
        for evidence_id in os.environ["RUN_COHORT_SKIP_IDS"].split(",")
        if evidence_id
    ],
    "qa_packet_limit": (
        int(os.environ["RUN_QA_PACKET_LIMIT"])
        if os.environ["RUN_QA_PACKET_LIMIT"]
        else None
    ),
    "required_gpu_count": int(os.environ["RUN_REQUIRED_GPU_COUNT"]),
    "cuda_keeper_gpus": os.environ["RUN_CUDA_KEEPER_GPUS"],
    "qwen_device_map": os.environ["RUN_QWEN_DEVICE_MAP"],
    "qwen_required_model_gpu_count": int(
        os.environ["RUN_QWEN_REQUIRED_MODEL_GPU_COUNT"]
    ),
}
Path(os.environ["RUN_MANIFEST_PATH"]).write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

if [[ -n "${REUSE_PREPROCESSED_FROM}" ]]; then
  stage "reuse_preprocessed_candidates"
  REUSE_PREPROCESSED_FROM="${REUSE_PREPROCESSED_FROM%/}"
  CANDIDATES="${REUSE_PREPROCESSED_FROM}/ten_minute_setup/time_aware_candidates/six_user_10min_time_aware.jsonl"
  SETUP_SUMMARY="${REUSE_PREPROCESSED_FROM}/ten_minute_setup/six_user_10min_setup_summary.json"
  test -s "${CANDIDATES}"
  test -s "${SETUP_SUMMARY}"
  printf 'preprocessing_mode=reuse source=%s candidates=%s summary=%s\n' \
    "${REUSE_PREPROCESSED_FROM}" "${CANDIDATES}" "${SETUP_SUMMARY}"
else
  stage "build_manifest"
  "${PYTHON}" -P -m egolife_two_user_qa build_manifest \
    --output "${OUTDIR}/manifest.json" \
    --no-overlays

  stage "prepare_ten_minute_six_user_candidates"
  "${PYTHON}" -P "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" prepare \
    --manifest "${OUTDIR}/manifest.json" \
    --output-root "${OUTDIR}/ten_minute_setup" \
    --cache-dir "${CACHE_DIR}" \
    --target-count "${TARGET_COUNT}" \
    --source-window-count "${SOURCE_WINDOW_COUNT}" \
    --max-groups "${MAX_GROUPS}" \
    --random-seed "${EVIDENCE_RANDOM_SEED}" \
    --model-id "${CLIP_MODEL_ID}" \
    --device cuda \
    --ffmpeg-binary "${FFMPEG_BINARY}" \
    --clip-batch-size 32 \
    --pruning-clusters-per-window 12 \
    --high-similarity-threshold 0.82 \
    --min-pruned-video-percent 40 \
    --generator-frame-budget 3600 \
    --generator-max-image-pixels "${GENERATOR_MAX_IMAGE_PIXELS}" \
    --judge-video-fps "${JUDGE_VIDEO_FPS}" \
    "${EVIDENCE_FRESHNESS_ARGS[@]}"

  CANDIDATES="${OUTDIR}/ten_minute_setup/time_aware_candidates/six_user_10min_time_aware.jsonl"
  SETUP_SUMMARY="${OUTDIR}/ten_minute_setup/six_user_10min_setup_summary.json"
fi

stage "validate_ten_minute_contract"
"${PYTHON}" -P - "${SETUP_SUMMARY}" "${CANDIDATES}" "${TARGET_COUNT}" "${REQUIRE_FRESH_EVIDENCE}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
candidates_path = Path(sys.argv[2])
target = int(sys.argv[3])
require_fresh_evidence = sys.argv[4] == "1"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
plan = summary["context_plan"]
pruning = summary["time_aware_pruning"]
if plan["pruning_sample_fps"] != 1.0:
    raise SystemExit("pruning must remain at 1 FPS")
if pruning["cluster_window_seconds"] != 30.0 or pruning["clusters_per_window"] != 12:
    raise SystemExit("pruning must use K=12 per fixed 30-second block")
if pruning["max_pair_time_difference_seconds"] != 30.0:
    raise SystemExit("cluster comparisons must use the +/-30-second window")
if pruning["speaker_preserved"] is not True or pruning["pruned_side"] != "providers_only":
    raise SystemExit("asker must be preserved and only providers may be pruned")
if plan["judge_temporal_coverage"] != "full_600_seconds_per_user":
    raise SystemExit("judges must retain the complete ten-minute span")
if plan.get("judge_review_mode") not in {None, "per_user_source_segment_map_reduce"}:
    raise SystemExit("judges must use per-user source-segment map/reduce")
if plan.get("judge_source_segments_per_visual_call") not in {None, 20}:
    raise SystemExit("each judge visual call must use 20 ordered source segments")
if plan["generator_aggregate_frame_budget"] != 3600:
    raise SystemExit("generator must retain the complete sampled-frame ceiling")
if plan.get("estimated_generator_input_tokens", 0) > plan["max_input_tokens"]:
    raise SystemExit("adaptive generator image budget exceeds the token ceiling")
fresh_evidence = summary.get("fresh_evidence") or {}
if require_fresh_evidence:
    if fresh_evidence.get("passed") is not True:
        raise SystemExit("fresh-evidence preparation was not verified")
    if int(fresh_evidence.get("source_window_overlap_count", -1)) != 0:
        raise SystemExit("prepared evidence overlaps previously processed source windows")
rows = [
    json.loads(line)
    for line in candidates_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
if len(rows) != target:
    raise SystemExit(f"expected {target} candidates, found {len(rows)}")
generator_frame_count = 0
judge_video_count = 0
judge_source_segment_count = 0
for row in rows:
    clips = row.get("clips") or []
    if len(clips) != 6:
        raise SystemExit("candidate must contain six users")
    if any(clip.get("local_video") for clip in clips):
        raise SystemExit("generator must not receive a video")
    if clips[0].get("is_pruned") is not False:
        raise SystemExit("asker sampled frames must be unpruned")
    if not all(clip.get("force_frame_inputs") is True for clip in clips):
        raise SystemExit("all generator media must be sampled images")
    for clip in clips:
        source_segments = clip.get("source_segments") or []
        if len(source_segments) != 20:
            raise SystemExit(
                "each ten-minute judge input must preserve its 20 source segments"
            )
        for expected_index, segment in enumerate(source_segments):
            if int(segment.get("segment_index", -1)) != expected_index:
                raise SystemExit("judge source segments must remain in recording order")
            source_segment = Path(str(segment.get("local_video") or ""))
            if not source_segment.is_file() or source_segment.stat().st_size <= 0:
                raise SystemExit(f"cached judge source segment is missing: {source_segment}")
            judge_source_segment_count += 1
        frames = clip.get("frames") or []
        if not frames:
            raise SystemExit("every reused clip must contain generator frames")
        for frame in frames:
            frame_path = Path(str(frame.get("path") or ""))
            if not frame_path.is_file() or frame_path.stat().st_size <= 0:
                raise SystemExit(f"reused generator frame is missing: {frame_path}")
            generator_frame_count += 1
        judge_video = Path(
            str(clip.get("full_local_video") or clip.get("original_local_video") or "")
        )
        if not judge_video.is_file() or judge_video.stat().st_size <= 0:
            raise SystemExit(f"reused full judge video is missing: {judge_video}")
        judge_video_count += 1
print(
    f"ten_minute_contract_passed candidates={len(rows)} "
    f"generator_frames={generator_frame_count} judge_videos={judge_video_count} "
    f"judge_source_segments={judge_source_segment_count}"
)
PY

if [[ -n "${RESUME_QA_FROM}" ]]; then
  stage "seed_qa_resume_state"
  RESUME_QA_FROM="${RESUME_QA_FROM%/}"
  if [[ "${RESUME_QA_FROM}" == "${OUTDIR}" ]]; then
    echo "error=resume_source_must_differ_from_new_output source=${RESUME_QA_FROM}" >&2
    exit 2
  fi
  test -d "${RESUME_QA_FROM}"
  test -f "${RESUME_QA_FROM}/qa_mcq.jsonl"
  cp -f "${RESUME_QA_FROM}/qa_mcq.jsonl" "${OUTDIR}/qa_mcq.jsonl"
  if [[ -f "${RESUME_QA_FROM}/qa_mcq.rejected.jsonl" ]]; then
    cp -f "${RESUME_QA_FROM}/qa_mcq.rejected.jsonl" \
      "${OUTDIR}/qa_mcq.rejected.jsonl"
  fi
  if [[ -f "${RESUME_QA_FROM}/qa_mcq.infrastructure_skipped.jsonl" ]]; then
    cp -f "${RESUME_QA_FROM}/qa_mcq.infrastructure_skipped.jsonl" \
      "${OUTDIR}/qa_mcq.infrastructure_skipped.jsonl"
  fi
  QA_RESUME_ARGS=(--resume)
  printf 'qa_resume_source=%s accepted_rows=%s rejected_rows=%s auxiliary_logs=reset\n' \
    "${RESUME_QA_FROM}" \
    "$(wc -l < "${OUTDIR}/qa_mcq.jsonl")" \
    "$(if [[ -f "${OUTDIR}/qa_mcq.rejected.jsonl" ]]; then wc -l < "${OUTDIR}/qa_mcq.rejected.jsonl"; else printf '0'; fi)"
  printf '%s\n' \
    'resume_note=old video_first_prompts and intermediate JSONLs are intentionally not copied'
fi

stage "generate_six_user_qa"
"${PYTHON}" -P -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${CANDIDATES}" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --infrastructure-skipped-output "${OUTDIR}/qa_mcq.infrastructure_skipped.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local-memory-safe \
  --model-id "${MODEL_ID}" \
  --dtype bfloat16 \
  --max-image-pixels "${GENERATOR_MAX_IMAGE_PIXELS}" \
  --max-new-tokens 2048 \
  --disable-thinking \
  --generation-mode baseline \
  --generator-decode-mode sampling \
  --generator-temperature 0.7 \
  --generator-top-p 0.9 \
  --generator-top-k 40 \
  --judge-video-source full \
  --six-user-judge-mode "${SIX_USER_JUDGE_MODE}" \
  --fixed-question-type-schedule \
  --question-types neutral \
  "${QA_SELECTION_ARGS[@]}" \
  "${QA_RESUME_ARGS[@]}"

if [[ "${REQUIRE_FRESH_EVIDENCE}" == "1" ]]; then
  stage "validate_fresh_questions"
  "${PYTHON}" -P "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" \
    validate-fresh-output \
    --qa "${OUTDIR}/qa_mcq.jsonl" \
    --freshness-manifest "${OUTDIR}/ten_minute_setup/freshness_exclusions.json" \
    --output "${OUTDIR}/fresh_question_validation.json"
fi

stage "validate_six_user_judge_contract"
"${PYTHON}" -P - "${OUTDIR}/video_first_prompts.jsonl" "${OUTDIR}/qa_mcq.intermediate.jsonl" "${SIX_USER_JUDGE_MODE}" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

prompts_path = Path(sys.argv[1])
intermediate_path = Path(sys.argv[2])
six_user_judge_mode = sys.argv[3]
prompt_rows = [
    json.loads(line)
    for line in prompts_path.read_text(encoding="utf-8").splitlines()
    if line.strip()
]
by_attempt = defaultdict(list)
for row in prompt_rows:
    if "image_paths" in row or "video_paths" in row or "condition_media" in row:
        raise SystemExit("prompt log exposed an exact media mapping")
    if row.get("qa_id") is not None and row.get("attempt") is not None:
        by_attempt[(row.get("qa_id"), row.get("attempt"))].append(row)

checked_attempts = 0
for key, rows in by_attempt.items():
    stages = [str(row.get("stage") or "") for row in rows]
    if six_user_judge_mode == "time-aware-map-reduce":
        if (
            stages.count("answerability_condition_aggregation") != 2
            or stages.count("evidence_groundedness_aggregation") != 1
        ):
            continue
        expected_counts = {
            "answerability_fact_plan": 1,
            "answerability_user_fact_audit": 6,
            "answerability_condition_aggregation": 2,
            "evidence_segment_observation": 6,
            "evidence_groundedness_aggregation": 1,
        }
        forbidden_stages = {"evidence_groundedness_judge", "answerability"}
    elif six_user_judge_mode == "legacy-zero-shot":
        if (
            stages.count("answerability") != 2
            or stages.count("evidence_groundedness_judge") != 1
        ):
            continue
        expected_counts = {
            "answerability": 2,
            "evidence_groundedness_judge": 1,
        }
        forbidden_stages = {
            "answerability_fact_plan",
            "answerability_user_fact_audit",
            "answerability_condition_aggregation",
            "evidence_segment_observation",
            "evidence_groundedness_aggregation",
        }
    else:
        raise SystemExit(f"unknown six-user judge mode: {six_user_judge_mode}")
    for stage_name, expected_count in expected_counts.items():
        actual_count = stages.count(stage_name)
        if actual_count != expected_count:
            raise SystemExit(
                f"{key} expected {expected_count} {stage_name} rows, found {actual_count}"
            )
    present_forbidden = sorted(forbidden_stages.intersection(stages))
    if present_forbidden:
        raise SystemExit(f"{key} used forbidden stages: {present_forbidden}")
    for row in rows:
        stage_name = row.get("stage")
        video_count = (row.get("media_summary") or {}).get("video_count")
        if six_user_judge_mode == "time-aware-map-reduce":
            if stage_name in {
                "answerability_user_fact_audit",
                "evidence_segment_observation",
            } and video_count != 20:
                raise SystemExit(
                    f"{key} map-stage judge did not receive 20 source segments"
                )
        elif stage_name == "evidence_groundedness_judge" and video_count != 6:
            raise SystemExit(f"{key} legacy groundedness judge did not receive six videos")
        elif stage_name == "answerability":
            condition_type = (row.get("media_summary") or {}).get("condition_type")
            expected_video_count = {
                "speaker_only": 1,
                "combined_all_six_users": 6,
            }.get(condition_type)
            if expected_video_count is None or video_count != expected_video_count:
                raise SystemExit(
                    f"{key} legacy answerability media mismatch for {condition_type}"
                )
        prompt = str(row.get("prompt") or "")
        if "/scratch/" in prompt or "local_video" in prompt or "timestamp_seconds" in prompt:
            raise SystemExit(f"{key} prompt exposed a path or exact temporal mapping")
    checked_attempts += 1

if checked_attempts < 1:
    raise SystemExit(
        f"no completed six-user {six_user_judge_mode} judge attempt was recorded"
    )

for line_number, line in enumerate(
    intermediate_path.read_text(encoding="utf-8").splitlines(),
    start=1,
):
    if not line.strip():
        continue
    if len(line.encode("utf-8")) > 5_000_000:
        raise SystemExit(f"intermediate row {line_number} exceeds the 5 MB ceiling")
    row = json.loads(line)
    serialized = json.dumps(row, ensure_ascii=False)
    if '"raw_output"' in serialized or '"image_paths"' in serialized or '"video_paths"' in serialized:
        raise SystemExit(f"intermediate row {line_number} retained a bulky trace field")

print(
    "six_user_judge_contract_passed",
    f"mode={six_user_judge_mode}",
    f"attempts={checked_attempts}",
    f"prompt_rows={len(prompt_rows)}",
)
PY

stage "validate_outputs"
"${PYTHON}" -P -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review

stage "complete"
printf 'status=completed output_dir=%s setup_summary=%s candidates=%s\n' \
  "${OUTDIR}" "${SETUP_SUMMARY}" "${CANDIDATES}"
