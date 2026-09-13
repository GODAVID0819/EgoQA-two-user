#!/usr/bin/env bash
#SBATCH --job-name=egoqa_lane_replay
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-six_user_qa_vllm_lane_replay_3setup_ab}"
RUN_SINGLE_PROFILE="${RUN_SINGLE_PROFILE:-}"
RUN_ONLY_SETUP="${RUN_ONLY_SETUP:-}"
RUN_4X3_LIMIT_PROBE="${RUN_4X3_LIMIT_PROBE:-1}"
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
REPLAY_DRIVER="${REPLAY_DRIVER:-${HPC_ROOT}/qa/experiments/replay_six_user_vllm_trace.py}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
PYTHON="${TRAIN_ENV}/bin/python"
VLLM_EXECUTABLE="${TRAIN_ENV}/bin/vllm"
SETSID_EXECUTABLE="$(command -v setsid)"
LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"

# Frozen workload from the completed 17218092 probe.  The replay driver uses
# the prompts exactly as persisted and resolves the omitted media mapping from
# the exact pinned ten-packet evidence artifact.
REFERENCE_JOB_ID="17218092"
REFERENCE_RUN_ROOT="${PACKAGE_ROOT}/outputs/six_user_qa_1h200_throughput_probe/six_user_qa_1h200_limit_10x3_0p5_${REFERENCE_JOB_ID}/runs/six_user_qa_1h200_limit_10x3_0p5_fps050_px091728_${REFERENCE_JOB_ID}"
TRACE_PATH="${TRACE_PATH:-${REFERENCE_RUN_ROOT}/video_first_prompts.jsonl}"
EXPECTED_TRACE_SHA256="${EXPECTED_TRACE_SHA256:-7b23efaae01c3fc5c8dc9996480621ab96e84e07193a66b8f6ac31fd49f66844}"
SOURCE_PREPROCESSING_ROOT="${SOURCE_PREPROCESSING_ROOT:-${PACKAGE_ROOT}/outputs/six_user_qa_10min_fps_resolution_ablation/six_user_qa_10min_vllm_fps_resolution_3x10_17109425/shared_preprocessing}"
EVIDENCE_PATH="${EVIDENCE_PATH:-${SOURCE_PREPROCESSING_ROOT}/ten_minute_setup/time_aware_candidates/six_user_10min_time_aware.jsonl}"
EXPECTED_EVIDENCE_SHA256="${EXPECTED_EVIDENCE_SHA256:-7e2d75fdeb3399e3bd614e1af07508bbc93ea75a7d88aa0997634c9940184f49}"
EXPECTED_REQUEST_COUNT="292"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3.8-27B}"
VLLM_REQUIRED_VERSION="${VLLM_REQUIRED_VERSION:-0.28.0}"
FLASHINFER_REQUIRED_VERSION="${FLASHINFER_REQUIRED_VERSION:-0.6.16.post3}"
VLLM_SERVER_HOST="127.0.0.1"
VLLM_SERVER_PORT="${VLLM_SERVER_PORT:-18000}"
VLLM_SERVER_API_COUNT="${VLLM_SERVER_API_COUNT:-3}"
# vLLM 0.28's async renderer uses this pool for tokenization, chat-template
# rendering, and multimodal preprocessing.  Keep one worker per API process:
# the latest 3x2 run showed that increasing this to two added contention and
# request latency without keeping the engine queue populated.
VLLM_RENDERER_NUM_WORKERS="${VLLM_RENDERER_NUM_WORKERS:-1}"
VLLM_MEDIA_LOADING_THREAD_COUNT="${VLLM_MEDIA_LOADING_THREAD_COUNT:-4}"
VLLM_SERVER_OMP_NUM_THREADS="${VLLM_SERVER_OMP_NUM_THREADS:-2}"
VLLM_MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-262144}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-12}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_MAX_IMAGES="${VLLM_MAX_IMAGES:-3600}"
VLLM_MAX_VIDEOS="${VLLM_MAX_VIDEOS:-20}"
VLLM_MIN_IMAGE_PIXELS="${VLLM_MIN_IMAGE_PIXELS:-3136}"
# vLLM 0.28 uses VLLM_MAX_IMAGE_PIXELS as a pre-decode safety ceiling on
# original raster dimensions.  It is deliberately separate from the much
# smaller per-profile Qwen processor max_pixels values below.
VLLM_MEDIA_DECODE_MAX_PIXELS="${VLLM_MEDIA_DECODE_MAX_PIXELS:-16777216}"
VLLM_VIDEO_FPS="${VLLM_VIDEO_FPS:-0.50}"
VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-2048}"
VLLM_SAMPLING_SEED="${VLLM_SAMPLING_SEED:-20260906}"

# These three diagonal profiles preserve approximately the same nominal visual
# token budget while trading temporal coverage against per-frame resolution.
# The frozen request trace keeps generation image count fixed; video FPS is
# applied to the 125 user-audit video requests, while max_pixels applies to all
# multimodal requests.  Per-request effective pixel caps are recorded because
# the context-safety cap may lower the configured ceiling for large image sets.
SETUP_LABELS=("fps025_px185024" "fps050_px091728" "fps075_px060368")
SETUP_FPS=("0.25" "0.50" "0.75")
SETUP_TOTAL_FRAMES=("900" "1800" "2700")
SETUP_MAX_TOKENS_PER_FRAME=("236" "117" "77")
SETUP_MAX_PIXELS=("185024" "91728" "60368")
SETUP_EQUIV_RESOLUTION=("430x430" "303x303" "246x246")

if [[ -n "${RUN_SINGLE_PROFILE}" && -n "${RUN_ONLY_SETUP}" ]]; then
  echo "error=run_single_profile_and_run_only_setup_are_mutually_exclusive" >&2
  exit 2
fi
if [[ "${RUN_4X3_LIMIT_PROBE}" != "0" && "${RUN_4X3_LIMIT_PROBE}" != "1" ]]; then
  echo "error=run_4x3_limit_probe_must_be_zero_or_one" >&2
  exit 2
fi
if [[ -n "${RUN_ONLY_SETUP}" ]]; then
  setup_is_known=0
  for setup_label in "${SETUP_LABELS[@]}"; do
    if [[ "${setup_label}" == "${RUN_ONLY_SETUP}" ]]; then
      setup_is_known=1
      break
    fi
  done
  if [[ "${setup_is_known}" != "1" ]]; then
    echo "error=unknown_run_only_setup setup=${RUN_ONLY_SETUP}" >&2
    exit 2
  fi
fi

CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-18.0}"
CUDA_KEEPER_MAX_PREALLOC_GB="${CUDA_KEEPER_MAX_PREALLOC_GB:-1.0}"
CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-60000}"
CUDA_KEEPER_WINDOW_SECONDS="${CUDA_KEEPER_WINDOW_SECONDS:-300}"
# Target 75% while a model is resident.  The extra headroom amortizes cold
# model-start and between-profile restart time in the sequential A/B/A job so
# the scheduler-visible job average remains above the 60% cancellation gate.
CUDA_KEEPER_TARGET_MARGIN="${CUDA_KEEPER_TARGET_MARGIN:-15}"
CUDA_KEEPER_IDLE_UTIL_THRESHOLD="${CUDA_KEEPER_IDLE_UTIL_THRESHOLD:-10}"
CUDA_KEEPER_IDLE_CONFIRM_SECONDS="${CUDA_KEEPER_IDLE_CONFIRM_SECONDS:-1.0}"
CUDA_KEEPER_MAX_PULSE_SECONDS="${CUDA_KEEPER_MAX_PULSE_SECONDS:-0.2}"

# B must beat the average of two bracketing 2/1 baselines.  If it does not,
# the 4/3 limit test is skipped rather than spending another H200-hour block.
MIN_THROUGHPUT_GAIN_PERCENT="${MIN_THROUGHPUT_GAIN_PERCENT:-10}"
MAX_BASELINE_DRIFT_PERCENT="${MAX_BASELINE_DRIFT_PERCENT:-15}"
MIN_FREE_HBM_MIB="${MIN_FREE_HBM_MIB:-12288}"

OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/outputs/six_user_qa_vllm_lane_replay}"
EXPERIMENT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/j/${SLURM_JOB_ID}}"
PERSISTENT_RUNTIME_CACHE_ROOT="${JOB_SCRATCH_ROOT}/runtime_cache"
PERSISTENT_MODEL_CACHE_ROOT="${PERSISTENT_RUNTIME_CACHE_ROOT}/huggingface"
PERSISTENT_KERNEL_CACHE_ROOT="${PERSISTENT_RUNTIME_CACHE_ROOT}/kernels"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${PERSISTENT_MODEL_CACHE_ROOT}"
export HF_HUB_CACHE="${PERSISTENT_MODEL_CACHE_ROOT}/hub"
export TRANSFORMERS_CACHE="${PERSISTENT_MODEL_CACHE_ROOT}/transformers"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/hf_datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${PERSISTENT_KERNEL_CACHE_ROOT}/torch"
export TRITON_CACHE_DIR="${PERSISTENT_KERNEL_CACHE_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${PERSISTENT_KERNEL_CACHE_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${PERSISTENT_KERNEL_CACHE_ROOT}/vllm"
export CUDA_CACHE_PATH="${PERSISTENT_KERNEL_CACHE_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${PERSISTENT_KERNEL_CACHE_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export CONDA_PKGS_DIRS="${JOB_SCRATCH_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
# FlashInfer's top-k/top-p sampler may JIT-compile during vLLM's startup
# profile and therefore requires ninja.  Keep FlashAttention plus the
# FlashInfer GDN prefill backend, but use vLLM's native sampler so a cold,
# job-specific cache has no sampler build-tool dependency.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_MEDIA_DECODE_MAX_PIXELS
export VLLM_MAX_IMAGE_PIXELS="${VLLM_MEDIA_DECODE_MAX_PIXELS}"
export FORCE_QWENVL_VIDEO_READER=decord
export VLLM_MEDIA_LOADING_THREAD_COUNT
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MALLOC_ARENA_MAX=2
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -z "${HF_TOKEN:-}" && -s "${LOGIN_HF_TOKEN_PATH}" ]]; then
  export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"
fi

mkdir -p \
  "${HPC_ROOT}/logs" \
  "${EXPERIMENT_ROOT}/profiles" \
  "${JOB_SCRATCH_ROOT}" \
  "${PYTHON_ALIAS_ROOT}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${MODELSCOPE_CACHE}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TMPDIR}" \
  "${CONDA_PKGS_DIRS}" \
  "${PIP_CACHE_DIR}"

ln -sfn "${PACKAGE_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"

stage() {
  printf 'lane_replay_stage=%s status=started timestamp=%s job_id=%s host=%s\n' \
    "$1" "$(date --iso-8601=seconds)" "${SLURM_JOB_ID}" "$(hostname)"
}

cd "${PROJECT_ROOT}"
test -x "${PYTHON}"
test -x "${VLLM_EXECUTABLE}"
test -x "${SETSID_EXECUTABLE}"
test -s "${REPLAY_DRIVER}"
test -s "${CUDA_KEEPER_SCRIPT}"
test -s "${TRACE_PATH}"
test -s "${EVIDENCE_PATH}"

stage "storage_preflight"
"${PYTHON}" -P -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${EXPERIMENT_ROOT}/storage_preflight.json"

stage "runtime_preflight"
"${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 1
"${FFMPEG_ENV}/bin/ffprobe" -version | head -n 1
export VLLM_REQUIRED_VERSION FLASHINFER_REQUIRED_VERSION
"${PYTHON}" -P - <<'PY'
import os
import shutil
from importlib.metadata import version

import decord
from packaging.version import Version
import torch
from qwen_vl_utils.vision_process import get_video_reader_backend
from vllm import envs as vllm_envs
from vllm.vllm_flash_attn import FA2_AVAILABLE, FA3_AVAILABLE, is_fa_version_supported

required_vllm = Version(os.environ["VLLM_REQUIRED_VERSION"])
actual_vllm = Version(version("vllm"))
if actual_vllm != required_vllm:
    raise SystemExit(f"expected vLLM {required_vllm}; found {actual_vllm}")
required_flashinfer = Version(os.environ["FLASHINFER_REQUIRED_VERSION"])
for package in ("flashinfer-python", "flashinfer-cubin"):
    actual = Version(version(package))
    if actual != required_flashinfer:
        raise SystemExit(f"expected {package} {required_flashinfer}; found {actual}")
backend = get_video_reader_backend()
if backend != "decord":
    raise SystemExit(f"unexpected video backend: {backend}")
usable_fa = [item for item in (2, 3) if is_fa_version_supported(item)]
if not usable_fa:
    raise SystemExit(
        f"no usable FlashAttention extension: fa2={FA2_AVAILABLE} fa3={FA3_AVAILABLE}"
    )
if os.environ["VLLM_USE_FLASHINFER_SAMPLER"] not in {"0", "1"}:
    raise SystemExit("VLLM_USE_FLASHINFER_SAMPLER must be 0 or 1")
if vllm_envs.VLLM_USE_FLASHINFER_SAMPLER:
    raise SystemExit(
        "this benchmark requires VLLM_USE_FLASHINFER_SAMPLER=0 so a cold "
        "job-specific FlashInfer cache does not require ninja JIT at startup"
    )
expected_decode_max_pixels = int(os.environ["VLLM_MEDIA_DECODE_MAX_PIXELS"])
if expected_decode_max_pixels <= 0:
    raise SystemExit("VLLM_MEDIA_DECODE_MAX_PIXELS must be positive")
if vllm_envs.VLLM_MAX_IMAGE_PIXELS != expected_decode_max_pixels:
    raise SystemExit(
        "vLLM media decode limit mismatch: "
        f"expected={expected_decode_max_pixels} "
        f"actual={vllm_envs.VLLM_MAX_IMAGE_PIXELS}"
    )
if torch.cuda.device_count() != 1:
    raise SystemExit(f"expected exactly one visible GPU; found {torch.cuda.device_count()}")
props = torch.cuda.get_device_properties(0)
if "H200" not in props.name:
    raise SystemExit(f"expected H200; found {props.name}")
try:
    keeper_probe_stream = torch.cuda.Stream(device=0, priority=0)
    keeper_probe_stream.synchronize()
except Exception as exc:
    raise SystemExit(f"CUDA keeper stream preflight failed: {exc}") from exc
print(
    "lane_replay_runtime_preflight_passed",
    f"vllm={actual_vllm}",
    f"flashinfer={required_flashinfer}",
    f"flash_attention_versions={usable_fa}",
    "flashinfer_sampler=false",
    "sampling_backend=vllm_native",
    "gdn_prefill_backend=flashinfer",
    f"ninja={shutil.which('ninja') or 'missing_not_required'}",
    f"media_decode_max_pixels={expected_decode_max_pixels}",
    f"video_backend={backend}",
    f"decord={decord.__version__}",
    f"gpu={props.name}",
    f"gpu_memory_gib={props.total_memory / 1024**3:.3f}",
    "keeper_stream_priority_zero=passed",
)
PY

stage "freeze_inputs_and_code"
actual_trace_sha="$(sha256sum "${TRACE_PATH}" | awk '{print $1}')"
actual_evidence_sha="$(sha256sum "${EVIDENCE_PATH}" | awk '{print $1}')"
if [[ "${actual_trace_sha}" != "${EXPECTED_TRACE_SHA256}" ]]; then
  echo "error=trace_sha256_mismatch expected=${EXPECTED_TRACE_SHA256} actual=${actual_trace_sha}" >&2
  exit 1
fi
if [[ "${actual_evidence_sha}" != "${EXPECTED_EVIDENCE_SHA256}" ]]; then
  echo "error=evidence_sha256_mismatch expected=${EXPECTED_EVIDENCE_SHA256} actual=${actual_evidence_sha}" >&2
  exit 1
fi
cp -p "${TRACE_PATH}" "${EXPERIMENT_ROOT}/frozen_video_first_prompts.jsonl"
cp -p "${REPLAY_DRIVER}" "${EXPERIMENT_ROOT}/replay_six_user_vllm_trace.py"
cp -p "$0" "${EXPERIMENT_ROOT}/submitted.sbatch"
sha256sum \
  "${EXPERIMENT_ROOT}/frozen_video_first_prompts.jsonl" \
  "${EVIDENCE_PATH}" \
  "${REPLAY_DRIVER}" \
  "${CUDA_KEEPER_SCRIPT}" \
  "$0" > "${EXPERIMENT_ROOT}/code_and_input_sha256.txt"

VALIDATION_ARGS=(
  --trace "${TRACE_PATH}"
  --evidence "${EVIDENCE_PATH}"
  --trace-sha256 "${EXPECTED_TRACE_SHA256}"
  --evidence-sha256 "${EXPECTED_EVIDENCE_SHA256}"
  --expected-requests "${EXPECTED_REQUEST_COUNT}"
  --base-url "http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1"
  --model-id "${MODEL_ID}"
  --max-packets-in-flight 2
  --max-review-lanes 1
  --max-tokens "${VLLM_MAX_TOKENS}"
  --min-pixels "${VLLM_MIN_IMAGE_PIXELS}"
  --max-pixels "${SETUP_MAX_PIXELS[1]}"
  --video-fps "${SETUP_FPS[1]}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --allowed-media-root "/scratch/${USER}"
  --seed "${VLLM_SAMPLING_SEED}"
  --requests-output "${EXPERIMENT_ROOT}/validation_requests_unused.jsonl"
  --summary-output "${EXPERIMENT_ROOT}/validation_summary_unused.json"
  --validate-only
)
"${PYTHON}" -P "${REPLAY_DRIVER}" "${VALIDATION_ARGS[@]}"

export EXPERIMENT_ROOT REFERENCE_JOB_ID TRACE_PATH EXPECTED_TRACE_SHA256
export EVIDENCE_PATH EXPECTED_EVIDENCE_SHA256 EXPECTED_REQUEST_COUNT
export MODEL_ID VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
export VLLM_MEDIA_LOADING_THREAD_COUNT VLLM_SERVER_OMP_NUM_THREADS
export VLLM_MM_PROCESSOR_CACHE_GB VLLM_GPU_MEMORY_UTILIZATION
export VLLM_MAX_MODEL_LEN VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
export VLLM_MIN_IMAGE_PIXELS VLLM_MEDIA_DECODE_MAX_PIXELS VLLM_VIDEO_FPS
export VLLM_USE_FLASHINFER_SAMPLER
export MIN_THROUGHPUT_GAIN_PERCENT MAX_BASELINE_DRIFT_PERCENT MIN_FREE_HBM_MIB
export RUN_SINGLE_PROFILE RUN_ONLY_SETUP RUN_4X3_LIMIT_PROBE
export CUDA_KEEPER_THRESHOLD CUDA_KEEPER_GPUS CUDA_KEEPER_RESERVE
export CUDA_KEEPER_MAX_PREALLOC_GB CUDA_KEEPER_START_USED_MIB
export CUDA_KEEPER_WINDOW_SECONDS CUDA_KEEPER_TARGET_MARGIN
export CUDA_KEEPER_IDLE_UTIL_THRESHOLD CUDA_KEEPER_IDLE_CONFIRM_SECONDS
export CUDA_KEEPER_MAX_PULSE_SECONDS
export RUN_SETUP_LABELS="${SETUP_LABELS[*]}"
export RUN_SETUP_FPS="${SETUP_FPS[*]}"
export RUN_SETUP_TOTAL_FRAMES="${SETUP_TOTAL_FRAMES[*]}"
export RUN_SETUP_MAX_TOKENS_PER_FRAME="${SETUP_MAX_TOKENS_PER_FRAME[*]}"
export RUN_SETUP_MAX_PIXELS="${SETUP_MAX_PIXELS[*]}"
export RUN_SETUP_EQUIV_RESOLUTION="${SETUP_EQUIV_RESOLUTION[*]}"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

def values(name, cast=str):
    return [cast(value) for value in os.environ[name].split()]

labels = values("RUN_SETUP_LABELS")
fps_values = values("RUN_SETUP_FPS", float)
total_frames = values("RUN_SETUP_TOTAL_FRAMES", int)
tokens_per_frame = values("RUN_SETUP_MAX_TOKENS_PER_FRAME", int)
max_pixels = values("RUN_SETUP_MAX_PIXELS", int)
resolutions = values("RUN_SETUP_EQUIV_RESOLUTION")
if not (
    len(labels)
    == len(fps_values)
    == len(total_frames)
    == len(tokens_per_frame)
    == len(max_pixels)
    == len(resolutions)
    == 3
):
    raise SystemExit("lane replay requires exactly three aligned media setups")
for fps, frames, tokens, pixels in zip(
    fps_values, total_frames, tokens_per_frame, max_pixels, strict=True
):
    if frames != round(fps * 6 * 600):
        raise SystemExit(
            f"nominal frame count mismatch: fps={fps} frames={frames}"
        )
    if pixels != tokens * 28 * 28:
        raise SystemExit(
            f"pixel/token mismatch: tokens={tokens} pixels={pixels}"
        )

setups = [
    {
        "label": label,
        "video_fps": fps,
        "nominal_total_video_frames_for_six_users": frames,
        "max_tokens_per_frame": tokens,
        "max_pixels_per_frame": pixels,
        "equivalent_resolution": resolution,
    }
    for label, fps, frames, tokens, pixels, resolution in zip(
        labels,
        fps_values,
        total_frames,
        tokens_per_frame,
        max_pixels,
        resolutions,
        strict=True,
    )
]
profile_order = []
for setup in setups:
    prefix = setup["label"]
    profile_order.extend(
        [
            f"{prefix}__baseline_2x1_a",
            f"{prefix}__candidate_3x2",
            f"{prefix}__baseline_2x1_b",
            f"{prefix}__candidate_4x3_if_setup_gate",
        ]
    )

manifest = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "purpose": "frozen 292-request vLLM lane A/B across three FPS/resolution setups",
    "reference_job_id": os.environ["REFERENCE_JOB_ID"],
    "trace": os.environ["TRACE_PATH"],
    "trace_sha256": os.environ["EXPECTED_TRACE_SHA256"],
    "evidence": os.environ["EVIDENCE_PATH"],
    "evidence_sha256": os.environ["EXPECTED_EVIDENCE_SHA256"],
    "request_count": int(os.environ["EXPECTED_REQUEST_COUNT"]),
    "run_mode": (
        "single_profile"
        if os.environ["RUN_SINGLE_PROFILE"]
        else "single_setup_sequential_ab"
        if os.environ["RUN_ONLY_SETUP"]
        else "full_ab"
    ),
    "selected_profile": os.environ["RUN_SINGLE_PROFILE"] or None,
    "selected_setup": os.environ["RUN_ONLY_SETUP"] or None,
    "run_4x3_limit_probe": bool(int(os.environ["RUN_4X3_LIMIT_PROBE"])),
    "setups": setups,
    "profile_order": profile_order,
    "gate": {
        "minimum_candidate_3x2_throughput_gain_percent": float(os.environ["MIN_THROUGHPUT_GAIN_PERCENT"]),
        "maximum_baseline_drift_percent": float(os.environ["MAX_BASELINE_DRIFT_PERCENT"]),
        "minimum_free_hbm_mib": int(os.environ["MIN_FREE_HBM_MIB"]),
    },
    "vllm": {
        "model": os.environ["MODEL_ID"],
        "api_server_count": int(os.environ["VLLM_SERVER_API_COUNT"]),
        "renderer_num_workers": int(os.environ["VLLM_RENDERER_NUM_WORKERS"]),
        "media_loading_thread_count": int(os.environ["VLLM_MEDIA_LOADING_THREAD_COUNT"]),
        "server_omp_num_threads": int(os.environ["VLLM_SERVER_OMP_NUM_THREADS"]),
        "mm_processor_cache_gb_per_process": float(os.environ["VLLM_MM_PROCESSOR_CACHE_GB"]),
        "gpu_memory_utilization": float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]),
        "max_model_len": int(os.environ["VLLM_MAX_MODEL_LEN"]),
        "max_num_seqs": int(os.environ["VLLM_MAX_NUM_SEQS"]),
        "max_num_batched_tokens": int(os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"]),
        "min_pixels": int(os.environ["VLLM_MIN_IMAGE_PIXELS"]),
        "media_decode_max_pixels": int(os.environ["VLLM_MEDIA_DECODE_MAX_PIXELS"]),
        "profile_specific_max_pixels": True,
        "profile_specific_video_fps": True,
        "flashinfer_sampler": bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"])),
        "sampling_backend": "vllm_native",
        "gdn_prefill_backend": "flashinfer",
    },
    "cuda_keeper": {
        "control_signal": "device_total_util_only",
        "threshold_percent": int(os.environ["CUDA_KEEPER_THRESHOLD"]),
        "target_margin_percent": int(os.environ["CUDA_KEEPER_TARGET_MARGIN"]),
        "rolling_window_seconds": int(os.environ["CUDA_KEEPER_WINDOW_SECONDS"]),
        "idle_util_threshold_percent": int(os.environ["CUDA_KEEPER_IDLE_UTIL_THRESHOLD"]),
        "idle_confirm_seconds": float(os.environ["CUDA_KEEPER_IDLE_CONFIRM_SECONDS"]),
        "max_pulse_seconds": float(os.environ["CUDA_KEEPER_MAX_PULSE_SECONDS"]),
        "reserve_gib": float(os.environ["CUDA_KEEPER_RESERVE"]),
        "max_prealloc_gib": float(os.environ["CUDA_KEEPER_MAX_PREALLOC_GB"]),
        "start_used_mib": int(os.environ["CUDA_KEEPER_START_USED_MIB"]),
        "stream_priority": "lowest_available",
    },
    "interpretation": {
        "frozen_prompts_and_media_paths": True,
        "generation_image_count_is_frozen_from_reference_0p5_fps_trace": True,
        "video_fps_is_applied_to_user_audit_video_requests": True,
        "max_pixels_is_applied_to_all_multimodal_requests": True,
        "model_outputs_ignored": True,
        "generation_lane_count": 1,
        "keeper_work_is_not_productive_inference": True,
        "primary_metric": "completed_packet_loops_per_hour over the fixed trace",
        "secondary_metric": "requests_per_minute for low-level service diagnostics",
    },
}
Path(os.environ["EXPERIMENT_ROOT"], "experiment_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

SERVER_PID=""
KEEPER_PID=""
KEEPER_WATCHDOG_PID=""
GLOBAL_CGROUP_MONITOR_PID=""
PROFILE_MONITOR_PIDS=()
ACTIVE_PROFILE_DIR=""
FINALIZING_JOB=0

stop_pid() {
  local pid="$1"
  if [[ -n "${pid}" ]]; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

stop_server() {
  if [[ -z "${SERVER_PID}" ]]; then
    return
  fi
  local shutdown_checks=15
  local release_checks=30
  if [[ "${FINALIZING_JOB}" == "1" ]]; then
    # Slurm's default KillWait is short.  Keep final/cancellation cleanup below
    # that window so the partial summary and terminal status reach storage.
    shutdown_checks=5
    release_checks=3
  fi
  kill -TERM -- "-${SERVER_PID}" 2>/dev/null || true
  for _ in $(seq 1 "${shutdown_checks}"); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      break
    fi
    sleep 2
  done
  if kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
  local released=0
  for _ in $(seq 1 "${release_checks}"); do
    local used_mib
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
    if [[ "${used_mib}" =~ ^[0-9]+$ && "${used_mib}" -lt 4096 ]]; then
      released=1
      break
    fi
    sleep 2
  done
  if [[ "${released}" != "1" ]]; then
    echo "error=gpu_memory_not_released_after_vllm_shutdown" >&2
    return 1
  fi
}

cleanup_profile() {
  set +e
  stop_pid "${KEEPER_WATCHDOG_PID}"
  KEEPER_WATCHDOG_PID=""
  stop_pid "${KEEPER_PID}"
  KEEPER_PID=""
  for pid in "${PROFILE_MONITOR_PIDS[@]}"; do
    stop_pid "${pid}"
  done
  PROFILE_MONITOR_PIDS=()
  stop_server
  set -e
}

finalize_job() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  FINALIZING_JOB=1
  if [[ -n "${ACTIVE_PROFILE_DIR}" \
      && -d "${ACTIVE_PROFILE_DIR}" \
      && ! -s "${ACTIVE_PROFILE_DIR}/profile_summary.json" ]] \
      && declare -F summarize_profile >/dev/null; then
    summarize_profile "${ACTIVE_PROFILE_DIR}" "${exit_code}" || true
  fi
  cleanup_profile
  stop_pid "${GLOBAL_CGROUP_MONITOR_PID}"
  GLOBAL_CGROUP_MONITOR_PID=""
  printf 'lane_replay_job_status=%s timestamp=%s exit_code=%s experiment_root=%s\n' \
    "$([[ "${exit_code}" == "0" ]] && printf completed || printf failed)" \
    "$(date --iso-8601=seconds)" "${exit_code}" "${EXPERIMENT_ROOT}"
  exit "${exit_code}"
}

trap finalize_job EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

monitor_gpu() {
  local destination="$1"
  set +e
  printf '%s\n' 'sample_iso,sample_epoch,gpu_index,gpu_name,sm_util_percent,memory_used_mib,memory_free_mib,power_watts,power_limit_watts,temperature_c' > "${destination}"
  while true; do
    local sample_iso sample_epoch values
    sample_iso="$(date --iso-8601=seconds)"
    sample_epoch="$(date +%s.%N)"
    values="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)"
    if [[ -n "${values}" ]]; then
      printf '%s,%s,%s\n' "${sample_iso}" "${sample_epoch}" "${values}" >> "${destination}"
    fi
    sleep 1
  done
}

monitor_vllm() {
  local url="$1"
  local destination="$2"
  # Replace the background Bash subshell with Python so PROFILE_MONITOR_PIDS
  # identifies the actual scraper.  Killing it now stops the writer instead
  # of leaving an orphan process that pollutes later profiles.
  exec "${PYTHON}" -u - "${url}" "${destination}" <<'PY'
import datetime
import sys
import time
import urllib.request

url, destination = sys.argv[1:]
with open(destination, "a", encoding="utf-8", buffering=1) as handle:
    while True:
        now = datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat()
        try:
            with urllib.request.urlopen(url, timeout=1.5) as response:
                payload = response.read().decode("utf-8", errors="replace")
            for line in payload.splitlines():
                if line.startswith("vllm:"):
                    handle.write(f"{now}\t{line}\n")
        except Exception as exc:
            handle.write(f"# {now}\tscrape_error={type(exc).__name__}:{exc}\n")
        time.sleep(2)
PY
}

monitor_cgroup_memory() {
  local destination="${EXPERIMENT_ROOT}/cgroup_memory_2s.tsv"
  local relative_path cgroup_dir
  relative_path="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)"
  cgroup_dir="/sys/fs/cgroup${relative_path}"
  printf 'sample_iso\tsample_epoch\tmemory_current_bytes\tmemory_peak_bytes\tmemory_max\tswap_current_bytes\n' > "${destination}"
  while true; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "$(date +%s.%N)" \
      "$(cat "${cgroup_dir}/memory.current" 2>/dev/null || printf unavailable)" \
      "$(cat "${cgroup_dir}/memory.peak" 2>/dev/null || printf unavailable)" \
      "$(cat "${cgroup_dir}/memory.max" 2>/dev/null || printf unavailable)" \
      "$(cat "${cgroup_dir}/memory.swap.current" 2>/dev/null || printf unavailable)" \
      >> "${destination}"
    sleep 2
  done
}

watch_cuda_keeper() {
  local keeper_pid="$1"
  local profile_label="$2"
  local keeper_log="$3"
  local parent_pid="$4"
  while true; do
    sleep 5
    if ! kill -0 "${keeper_pid}" 2>/dev/null; then
      echo "error=cuda_keeper_died_during_replay profile=${profile_label}" >&2
      tail -n 200 "${keeper_log}" >&2 || true
      kill -TERM "${parent_pid}" 2>/dev/null || true
      return 1
    fi
  done
}

monitor_cgroup_memory &
GLOBAL_CGROUP_MONITOR_PID=$!

start_profile_resources() {
  local profile_label="$1"
  local profile_fps="$2"
  local profile_max_pixels="$3"
  local initial_used_mib
  initial_used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')"
  if [[ ! "${initial_used_mib}" =~ ^[0-9]+$ || "${initial_used_mib}" -ge 4096 ]]; then
    echo "error=gpu_not_clean_before_profile profile=${profile_label} memory_used_mib=${initial_used_mib}" >&2
    return 1
  fi
  ACTIVE_PROFILE_DIR="${EXPERIMENT_ROOT}/profiles/${profile_label}"
  mkdir -p "${ACTIVE_PROFILE_DIR}"

  monitor_gpu "${ACTIVE_PROFILE_DIR}/gpu_1s.csv" &
  PROFILE_MONITOR_PIDS+=("$!")

  PYTHONUNBUFFERED=1 "${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}" \
    --threshold "${CUDA_KEEPER_THRESHOLD}" \
    --gpus "${CUDA_KEEPER_GPUS}" \
    --reserve "${CUDA_KEEPER_RESERVE}" \
    --max-prealloc "${CUDA_KEEPER_MAX_PREALLOC_GB}" \
    --start-used-mib "${CUDA_KEEPER_START_USED_MIB}" \
    --window-seconds "${CUDA_KEEPER_WINDOW_SECONDS}" \
    --target-margin "${CUDA_KEEPER_TARGET_MARGIN}" \
    --idle-util-threshold "${CUDA_KEEPER_IDLE_UTIL_THRESHOLD}" \
    --idle-confirm-seconds "${CUDA_KEEPER_IDLE_CONFIRM_SECONDS}" \
    --max-pulse-seconds "${CUDA_KEEPER_MAX_PULSE_SECONDS}" \
    > "${ACTIVE_PROFILE_DIR}/cuda_keeper.log" 2>&1 &
  KEEPER_PID=$!

  local runtime_root="${JOB_SCRATCH_ROOT}/v_${profile_label}"
  mkdir -p "${runtime_root}"
  local limit_mm_json mm_processor_json speculative_json
  limit_mm_json="$(printf '{"image":%s,"video":%s,"audio":0}' "${VLLM_MAX_IMAGES}" "${VLLM_MAX_VIDEOS}")"
  mm_processor_json="$(printf '{"min_pixels":%s,"max_pixels":%s,"fps":%s,"cap_pixels_per_frame":true}' "${VLLM_MIN_IMAGE_PIXELS}" "${profile_max_pixels}" "${profile_fps}")"
  speculative_json='{"method":"mtp","num_speculative_tokens":1}'
  local server_command=(
    "${VLLM_EXECUTABLE}" serve "${MODEL_ID}"
    --host "${VLLM_SERVER_HOST}"
    --port "${VLLM_SERVER_PORT}"
    --served-model-name "${MODEL_ID}"
    --trust-remote-code
    --dtype bfloat16
    --tensor-parallel-size 1
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --max-num-seqs "${VLLM_MAX_NUM_SEQS}"
    --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}"
    --enable-chunked-prefill
    --attention-backend FLASH_ATTN
    --gdn-prefill-backend flashinfer
    --speculative-config "${speculative_json}"
    --limit-mm-per-prompt "${limit_mm_json}"
    --mm-processor-kwargs "${mm_processor_json}"
    --mm-processor-cache-gb "${VLLM_MM_PROCESSOR_CACHE_GB}"
    --mm-processor-cache-type shm
    --mm-encoder-tp-mode data
    --mm-encoder-attn-backend FLASH_ATTN
    --skip-mm-profiling
    --allowed-local-media-path "/scratch/${USER}"
    --api-server-count "${VLLM_SERVER_API_COUNT}"
    --renderer-num-workers "${VLLM_RENDERER_NUM_WORKERS}"
    --default-chat-template-kwargs '{"enable_thinking":false}'
    --enable-prefix-caching
    --enable-mfu-metrics
  )
  printf 'profile=%s vllm_server_command=' "${profile_label}"
  printf '%q ' "${server_command[@]}"
  printf '\n'
  OMP_NUM_THREADS="${VLLM_SERVER_OMP_NUM_THREADS}" \
    TMPDIR="${runtime_root}" TMP="${runtime_root}" TEMP="${runtime_root}" \
    PYTHONUNBUFFERED=1 "${SETSID_EXECUTABLE}" "${server_command[@]}" \
    > "${ACTIVE_PROFILE_DIR}/vllm_server.log" 2>&1 &
  SERVER_PID=$!

  local health_url="http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1/models"
  local ready=0
  export PROFILE_HEALTH_URL="${health_url}"
  for _ in $(seq 1 240); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "error=vllm_server_exited_before_ready profile=${profile_label}" >&2
      tail -n 200 "${ACTIVE_PROFILE_DIR}/vllm_server.log" >&2 || true
      return 1
    fi
    if "${PYTHON}" -P - <<'PY'
import json
import os
import urllib.request

try:
    with urllib.request.urlopen(os.environ["PROFILE_HEALTH_URL"], timeout=2) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if payload.get("data"):
        raise SystemExit(0)
except Exception:
    pass
raise SystemExit(1)
PY
    then
      ready=1
      break
    fi
    sleep 5
  done
  if [[ "${ready}" != "1" ]]; then
    echo "error=vllm_server_readiness_timeout profile=${profile_label}" >&2
    return 1
  fi
  local keeper_ready=0
  for _ in $(seq 1 30); do
    if ! kill -0 "${KEEPER_PID}" 2>/dev/null; then
      echo "error=cuda_keeper_exited_before_control_loop profile=${profile_label}" >&2
      tail -n 200 "${ACTIVE_PROFILE_DIR}/cuda_keeper.log" >&2 || true
      return 1
    fi
    if grep -q 'burning:' "${ACTIVE_PROFILE_DIR}/cuda_keeper.log" 2>/dev/null; then
      keeper_ready=1
      break
    fi
    sleep 1
  done
  if [[ "${keeper_ready}" != "1" ]]; then
    echo "error=cuda_keeper_control_loop_timeout profile=${profile_label}" >&2
    tail -n 200 "${ACTIVE_PROFILE_DIR}/cuda_keeper.log" >&2 || true
    return 1
  fi
  watch_cuda_keeper "${KEEPER_PID}" "${profile_label}" \
    "${ACTIVE_PROFILE_DIR}/cuda_keeper.log" "$$" &
  KEEPER_WATCHDOG_PID=$!
  monitor_vllm "http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/metrics" \
    "${ACTIVE_PROFILE_DIR}/vllm_metrics_2s.promlog" &
  PROFILE_MONITOR_PIDS+=("$!")
  printf 'profile=%s vllm_server_ready timestamp=%s pid=%s cuda_keeper_pid=%s cuda_keeper_control_loop=ready\n' \
    "${profile_label}" "$(date --iso-8601=seconds)" "${SERVER_PID}" "${KEEPER_PID}"
}

summarize_profile() {
  local profile_dir="$1"
  local replay_exit_code="$2"
  "${PYTHON}" -P - "${profile_dir}" "${replay_exit_code}" "${MIN_FREE_HBM_MIB}" "${EXPECTED_REQUEST_COUNT}" "${TRACE_PATH}" <<'PY'
import csv
from datetime import datetime, timedelta
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

root = Path(sys.argv[1])
exit_code = int(sys.argv[2])
minimum_free_mib = int(sys.argv[3])
expected_request_count = int(sys.argv[4])
trace_path = Path(sys.argv[5])
driver_path = root / "replay_summary.json"
driver = {}
driver_load_error = None
if driver_path.exists():
    try:
        driver = json.loads(driver_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        driver_load_error = f"{type(exc).__name__}: {exc}"
partial_replay = not bool(driver)
request_path = root / "requests.jsonl"
records = []
malformed_lines = 0
if request_path.exists():
    for raw in request_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            records.append(json.loads(raw))
        except json.JSONDecodeError:
            malformed_lines += 1
timings = []
request_finished_by_ordinal = {}
for record in records:
    try:
        request_started = datetime.fromisoformat(record["started_at"])
        request_finished = request_started + timedelta(
            seconds=float(record["elapsed_seconds"])
        )
        timings.append((request_started, request_finished))
        request_finished_by_ordinal[int(record["trace_ordinal"])] = request_finished
    except (KeyError, TypeError, ValueError):
        pass
if partial_replay:
    if timings:
        partial_started = min(item[0] for item in timings)
        partial_finished = max(item[1] for item in timings)
        partial_wall_seconds = max(
            0.0, (partial_finished - partial_started).total_seconds()
        )
        completed = len(records)
        driver = {
            "status": "partial",
            "started_at": partial_started.isoformat(),
            "finished_at": partial_finished.isoformat(),
            "wall_seconds": round(partial_wall_seconds, 6),
            "expected_request_count": expected_request_count,
            "request_count": completed,
            "failed_request_count": sum(
                record.get("error") is not None for record in records
            ),
            "requests_per_minute": round(
                completed * 60 / partial_wall_seconds, 6
            ) if partial_wall_seconds > 0 else None,
            "malformed_request_lines": malformed_lines,
        }
started = datetime.fromisoformat(driver["started_at"]) if driver.get("started_at") else None
finished = datetime.fromisoformat(driver["finished_at"]) if driver.get("finished_at") else None
summary = {
    "replay_exit_code": exit_code,
    "partial_replay": partial_replay,
    "driver": driver,
}
if driver_load_error:
    summary["driver_summary_load_error"] = driver_load_error

# Keep packet-level loops separate from attempts and low-level HTTP requests.
# Each generation record starts one attempt group in the frozen trace.  One
# packet-level loop may contain several attempts; a completed packet requires
# every expected, error-free request from all of its attempt groups.
successful_ordinals = {
    int(record["trace_ordinal"])
    for record in records
    if record.get("error") is None and record.get("trace_ordinal") is not None
}
trace_groups = []
current_group = None
if trace_path.exists():
    for ordinal, raw in enumerate(
        trace_path.read_text(encoding="utf-8", errors="replace").splitlines()
    ):
        trace_record = json.loads(raw)
        if trace_record.get("stage") == "generation":
            current_group = {
                "packet_id": trace_record.get("evidence_id"),
                "generation_ordinal": ordinal,
                "ordinals": set(),
                "has_judge": False,
            }
            trace_groups.append(current_group)
        if current_group is not None:
            current_group["ordinals"].add(ordinal)
            if trace_record.get("stage") != "generation":
                current_group["has_judge"] = True
wall_seconds = float(driver.get("wall_seconds") or 0.0)
if trace_groups and wall_seconds > 0:
    completed_generation_attempts = sum(
        group["generation_ordinal"] in successful_ordinals
        for group in trace_groups
    )
    reviewed_groups = [group for group in trace_groups if group["has_judge"]]
    complete_generator_judge_attempts = sum(
        group["ordinals"].issubset(successful_ordinals)
        for group in reviewed_groups
    )
    packet_groups = defaultdict(list)
    for group in trace_groups:
        packet_groups[group["packet_id"]].append(group)
    completed_packets = [
        groups
        for groups in packet_groups.values()
        if all(group["ordinals"].issubset(successful_ordinals) for group in groups)
    ]
    packet_wall_seconds = wall_seconds
    packet_duration_basis = "driver_wall_seconds"
    if partial_replay and started and completed_packets:
        completed_packet_ordinals = {
            ordinal
            for groups in completed_packets
            for group in groups
            for ordinal in group["ordinals"]
        }
        completion_times = [
            request_finished_by_ordinal[ordinal]
            for ordinal in completed_packet_ordinals
            if ordinal in request_finished_by_ordinal
        ]
        if completion_times:
            packet_wall_seconds = max(
                0.0, (max(completion_times) - started).total_seconds()
            )
            packet_duration_basis = "through_last_completed_packet"
    summary["attempt_throughput"] = {
        "completed_generation_attempts": completed_generation_attempts,
        "generation_attempts_per_hour": round(
            completed_generation_attempts * 3600 / wall_seconds, 6
        ),
        "complete_generator_judge_attempts": complete_generator_judge_attempts,
        "complete_generator_judge_attempts_per_hour": round(
            complete_generator_judge_attempts * 3600 / wall_seconds, 6
        ),
        "expected_generation_attempts": len(trace_groups),
        "expected_generator_judge_attempts": len(reviewed_groups),
        "duration_basis": (
            "last_completed_request_window"
            if partial_replay
            else "driver_wall_seconds"
        ),
    }
    summary["packet_loop_throughput"] = {
        "completed_packet_loops": len(completed_packets),
        "completed_packet_loops_per_hour": round(
            len(completed_packets) * 3600 / packet_wall_seconds, 6
        ) if packet_wall_seconds > 0 else None,
        "expected_packet_loops": len(packet_groups),
        "wall_seconds": round(packet_wall_seconds, 6),
        "duration_basis": packet_duration_basis,
    }

gpu_rows = []
gpu_path = root / "gpu_1s.csv"
if gpu_path.exists():
    with gpu_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                when = datetime.fromisoformat(row["sample_iso"])
                if started and when < started:
                    continue
                if finished and when > finished:
                    continue
                gpu_rows.append({
                    "util": float(row["sm_util_percent"]),
                    "used": float(row["memory_used_mib"]),
                    "free": float(row["memory_free_mib"]),
                    "power": float(row["power_watts"]),
                })
            except (KeyError, TypeError, ValueError):
                pass
if gpu_rows:
    summary["gpu_during_replay"] = {
        "samples": len(gpu_rows),
        "sm_util_mean_percent_including_keeper": round(statistics.fmean(row["util"] for row in gpu_rows), 4),
        "max_memory_used_mib": max(row["used"] for row in gpu_rows),
        "min_memory_free_mib": min(row["free"] for row in gpu_rows),
        "mean_power_watts": round(statistics.fmean(row["power"] for row in gpu_rows), 4),
    }

metric_samples = defaultdict(lambda: defaultdict(list))
metric_pattern = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")
metric_path = root / "vllm_metrics_2s.promlog"
if metric_path.exists():
    for raw in metric_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("#") or "\t" not in raw:
            continue
        timestamp, metric = raw.split("\t", 1)
        match = metric_pattern.match(metric)
        if not match:
            continue
        try:
            metric_samples[match.group(1)][timestamp].append(float(match.group(2)))
        except ValueError:
            pass

def aggregates(name):
    return [sum(values) for _, values in sorted(metric_samples.get(name, {}).items())]

vllm = {}
for name in ("vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:kv_cache_usage_perc"):
    values = aggregates(name)
    if values:
        vllm[name] = {
            "mean": round(statistics.fmean(values), 6),
            "max": max(values),
            "zero_fraction": round(sum(value == 0 for value in values) / len(values), 6),
        }
for base in (
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:e2e_request_latency_seconds",
):
    sums = aggregates(base + "_sum")
    counts = aggregates(base + "_count")
    if sums and counts and counts[-1] > 0:
        vllm[base] = {
            "total_seconds": round(sums[-1], 6),
            "request_count": counts[-1],
            "mean_seconds": round(sums[-1] / counts[-1], 6),
        }
preemptions = aggregates("vllm:num_preemptions_total")
vllm["num_preemptions"] = preemptions[-1] if preemptions else None
summary["vllm"] = vllm

keeper_path = root / "cuda_keeper.log"
if keeper_path.exists():
    lines = [line for line in keeper_path.read_text(encoding="utf-8", errors="replace").splitlines() if "burning:" in line]
    burning = [line for line in lines if "burning:Y" in line]
    total_utils = []
    rolling_averages = []
    pulse_seconds = []
    reasons = Counter()
    for line in lines:
        match = re.search(r"total:\s*([0-9]+)%", line)
        if match:
            total_utils.append(int(match.group(1)))
        match = re.search(r"avg[0-9]+s:\s*([0-9.]+)%", line)
        if match:
            rolling_averages.append(float(match.group(1)))
        match = re.search(r"pulse:\s*([0-9.]+)s", line)
        if match:
            pulse_seconds.append(float(match.group(1)))
        match = re.search(r"reason:([^\s]+)", line)
        if match:
            reasons[match.group(1)] += 1
    summary["cuda_keeper"] = {
        "control_signal": "device_total_util_only",
        "controller_samples": len(lines),
        "burning_fraction": round(len(burning) / len(lines), 6) if lines else None,
        "mean_observed_total_util_percent": round(statistics.fmean(total_utils), 4) if total_utils else None,
        "final_rolling_average_util_percent": rolling_averages[-1] if rolling_averages else None,
        "total_requested_pulse_seconds": round(sum(pulse_seconds), 3),
        "decision_reason_counts": dict(sorted(reasons.items())),
        "warning": "keeper work is not productive inference",
    }

failed_requests = int(driver.get("failed_request_count", -1))
free_memory = (summary.get("gpu_during_replay") or {}).get("min_memory_free_mib")
no_preemption = vllm.get("num_preemptions") in {0, 0.0}
summary["safety_gate_passed"] = bool(
    exit_code == 0
    and failed_requests == 0
    and no_preemption
    and free_memory is not None
    and free_memory >= minimum_free_mib
)
(root / "profile_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("lane_replay_profile_summary", json.dumps(summary, sort_keys=True))
PY
}

run_profile() {
  local label="$1"
  local packets="$2"
  local reviews="$3"
  local profile_fps="$4"
  local profile_max_pixels="$5"
  stage "profile_${label}"
  printf 'profile=%s status=started timestamp=%s packets_in_flight=%s generation_lanes=1 review_lanes=%s video_fps=%s max_pixels=%s\n' \
    "${label}" "$(date --iso-8601=seconds)" "${packets}" "${reviews}" \
    "${profile_fps}" "${profile_max_pixels}"
  start_profile_resources "${label}" "${profile_fps}" "${profile_max_pixels}"
  local profile_dir="${ACTIVE_PROFILE_DIR}"
  local replay_args=(
    --trace "${TRACE_PATH}"
    --evidence "${EVIDENCE_PATH}"
    --trace-sha256 "${EXPECTED_TRACE_SHA256}"
    --evidence-sha256 "${EXPECTED_EVIDENCE_SHA256}"
    --expected-requests "${EXPECTED_REQUEST_COUNT}"
    --base-url "http://${VLLM_SERVER_HOST}:${VLLM_SERVER_PORT}/v1"
    --model-id "${MODEL_ID}"
    --max-packets-in-flight "${packets}"
    --max-review-lanes "${reviews}"
    --max-tokens "${VLLM_MAX_TOKENS}"
    --timeout 3600
    --min-pixels "${VLLM_MIN_IMAGE_PIXELS}"
    --max-pixels "${profile_max_pixels}"
    --video-fps "${profile_fps}"
    --max-model-len "${VLLM_MAX_MODEL_LEN}"
    --allowed-media-root "/scratch/${USER}"
    --seed "${VLLM_SAMPLING_SEED}"
    --requests-output "${profile_dir}/requests.jsonl"
    --summary-output "${profile_dir}/replay_summary.json"
  )
  set +e
  "${PYTHON}" -P "${REPLAY_DRIVER}" "${replay_args[@]}" > >(
    awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
      | tee "${profile_dir}/replay_timeline.log"
  ) 2> >(
    awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
      | tee "${profile_dir}/replay_stderr_timeline.log" >&2
  )
  local replay_exit_code=$?
  set -e
  cleanup_profile
  summarize_profile "${profile_dir}" "${replay_exit_code}"
  printf 'profile=%s status=%s timestamp=%s replay_exit_code=%s summary=%s\n' \
    "${label}" \
    "$([[ "${replay_exit_code}" == "0" ]] && printf completed || printf failed)" \
    "$(date --iso-8601=seconds)" "${replay_exit_code}" \
    "${profile_dir}/profile_summary.json"
  if [[ "${replay_exit_code}" != "0" ]]; then
    return "${replay_exit_code}"
  fi
}

evaluate_setup_gate() {
  local setup_label="$1"
  local prefix="${EXPERIMENT_ROOT}/profiles/${setup_label}__"
  export CURRENT_SETUP_LABEL="${setup_label}"
  export BASELINE_A_SUMMARY="${prefix}baseline_2x1_a/profile_summary.json"
  export CANDIDATE_B_SUMMARY="${prefix}candidate_3x2/profile_summary.json"
  export BASELINE_B_SUMMARY="${prefix}baseline_2x1_b/profile_summary.json"
  export SETUP_GATE_OUTPUT="${EXPERIMENT_ROOT}/gates/${setup_label}_ab_gate.json"
  "${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

def load(name):
    return json.loads(Path(os.environ[name]).read_text(encoding="utf-8"))

a1 = load("BASELINE_A_SUMMARY")
b = load("CANDIDATE_B_SUMMARY")
a2 = load("BASELINE_B_SUMMARY")
a1_rate = float(a1["driver"]["requests_per_minute"])
a2_rate = float(a2["driver"]["requests_per_minute"])
b_rate = float(b["driver"]["requests_per_minute"])
baseline = (a1_rate + a2_rate) / 2
gain = 100 * (b_rate / baseline - 1)
drift = 100 * abs(a2_rate - a1_rate) / baseline
minimum_gain = float(os.environ["MIN_THROUGHPUT_GAIN_PERCENT"])
maximum_drift = float(os.environ["MAX_BASELINE_DRIFT_PERCENT"])
passed = bool(
    a1.get("safety_gate_passed")
    and a2.get("safety_gate_passed")
    and b.get("safety_gate_passed")
    and gain >= minimum_gain
    and drift <= maximum_drift
)
result = {
    "setup": os.environ["CURRENT_SETUP_LABEL"],
    "passed": passed,
    "baseline_2x1_a_requests_per_minute": a1_rate,
    "baseline_2x1_b_requests_per_minute": a2_rate,
    "baseline_mean_requests_per_minute": baseline,
    "candidate_3x2_requests_per_minute": b_rate,
    "candidate_gain_percent": gain,
    "baseline_drift_percent": drift,
    "minimum_gain_percent": minimum_gain,
    "maximum_baseline_drift_percent": maximum_drift,
    "decision": "run_4x3_limit_probe" if passed else "stop_at_3x2_gate",
}
destination = Path(os.environ["SETUP_GATE_OUTPUT"])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("lane_replay_ab_gate", json.dumps(result, sort_keys=True))
raise SystemExit(0 if passed else 3)
PY
}

mkdir -p "${EXPERIMENT_ROOT}/gates"
if [[ -n "${RUN_SINGLE_PROFILE}" ]]; then
  selected_setup="${RUN_SINGLE_PROFILE%%__*}"
  selected_variant="${RUN_SINGLE_PROFILE#*__}"
  selected_setup_index=-1
  for setup_index in "${!SETUP_LABELS[@]}"; do
    if [[ "${SETUP_LABELS[${setup_index}]}" == "${selected_setup}" ]]; then
      selected_setup_index="${setup_index}"
      break
    fi
  done
  if (( selected_setup_index < 0 )); then
    echo "error=unknown_single_profile_setup profile=${RUN_SINGLE_PROFILE}" >&2
    exit 2
  fi
  case "${selected_variant}" in
    baseline_2x1_a|baseline_2x1_b)
      single_packets=2
      single_reviews=1
      ;;
    candidate_3x2)
      single_packets=3
      single_reviews=2
      ;;
    candidate_4x3)
      single_packets=4
      single_reviews=3
      ;;
    *)
      echo "error=unknown_single_profile_variant profile=${RUN_SINGLE_PROFILE}" >&2
      exit 2
      ;;
  esac
  run_profile "${RUN_SINGLE_PROFILE}" "${single_packets}" "${single_reviews}" \
    "${SETUP_FPS[${selected_setup_index}]}" \
    "${SETUP_MAX_PIXELS[${selected_setup_index}]}"
  printf 'status=completed run_mode=single_profile profile=%s summary=%s\n' \
    "${RUN_SINGLE_PROFILE}" \
    "${EXPERIMENT_ROOT}/profiles/${RUN_SINGLE_PROFILE}/profile_summary.json"
  exit 0
fi

for setup_index in "${!SETUP_LABELS[@]}"; do
  setup_label="${SETUP_LABELS[${setup_index}]}"
  if [[ -n "${RUN_ONLY_SETUP}" && "${setup_label}" != "${RUN_ONLY_SETUP}" ]]; then
    continue
  fi
  setup_fps="${SETUP_FPS[${setup_index}]}"
  setup_max_pixels="${SETUP_MAX_PIXELS[${setup_index}]}"
  profile_prefix="${setup_label}__"

  run_profile "${profile_prefix}baseline_2x1_a" 2 1 \
    "${setup_fps}" "${setup_max_pixels}"
  run_profile "${profile_prefix}candidate_3x2" 3 2 \
    "${setup_fps}" "${setup_max_pixels}"
  run_profile "${profile_prefix}baseline_2x1_b" 2 1 \
    "${setup_fps}" "${setup_max_pixels}"

  stage "evaluate_${setup_label}_3x2_gate"
  set +e
  evaluate_setup_gate "${setup_label}"
  setup_gate_exit_code=$?
  set -e
  if [[ "${setup_gate_exit_code}" == "0" && "${RUN_4X3_LIMIT_PROBE}" == "1" ]]; then
    run_profile "${profile_prefix}candidate_4x3" 4 3 \
      "${setup_fps}" "${setup_max_pixels}"
  else
    if [[ "${RUN_4X3_LIMIT_PROBE}" == "1" ]]; then
      skip_reason=three_by_two_setup_gate_failed
    else
      skip_reason=limit_probe_disabled
    fi
    printf 'profile=%scandidate_4x3 status=skipped reason=%s gate=%s\n' \
      "${profile_prefix}" "${skip_reason}" \
      "${EXPERIMENT_ROOT}/gates/${setup_label}_ab_gate.json"
  fi
done

stage "write_final_comparison"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["EXPERIMENT_ROOT"])
profiles = {}
for path in sorted((root / "profiles").glob("*/profile_summary.json")):
    profiles[path.parent.name] = json.loads(path.read_text(encoding="utf-8"))
gates = {
    path.stem.removesuffix("_ab_gate"): json.loads(path.read_text(encoding="utf-8"))
    for path in sorted((root / "gates").glob("*_ab_gate.json"))
}
expected_gate_count = 1 if os.environ.get("RUN_ONLY_SETUP") else 3
all_selected_gates_passed = bool(
    all(gate.get("passed") for gate in gates.values())
    and len(gates) == expected_gate_count
)
result = {
    "status": "completed",
    "selected_setup": os.environ.get("RUN_ONLY_SETUP") or None,
    "expected_setup_gate_count": expected_gate_count,
    "all_selected_three_by_two_setup_gates_passed": all_selected_gates_passed,
    "all_three_by_two_setup_gates_passed": bool(
        not os.environ.get("RUN_ONLY_SETUP")
        and all_selected_gates_passed
    ),
    "profiles": profiles,
    "setup_gates": gates,
    "interpretation": (
        "Compare 3x2 against its two bracketing 2x1 baselines within each media "
        "setup. Generation image count is frozen from the reference 0.5-FPS "
        "trace; configured FPS changes video decoding, not that frozen image count. "
        "Do not use keeper-inflated total GPU utilization as useful model utilization."
    ),
}
(root / "final_comparison.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("lane_replay_final_comparison", json.dumps(result, sort_keys=True))
PY

stop_pid "${GLOBAL_CGROUP_MONITOR_PID}"
GLOBAL_CGROUP_MONITOR_PID=""
printf 'status=completed experiment_root=%s comparison=%s gates=%s\n' \
  "${EXPERIMENT_ROOT}" "${EXPERIMENT_ROOT}/final_comparison.json" \
  "${EXPERIMENT_ROOT}/gates"
