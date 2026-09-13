#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_vllm_3x10
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --mem=320G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:-six_user_qa_10min_vllm_fps_resolution_3x10}"
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
RUNTIME_SCRIPT="${RUNTIME_SCRIPT:-${HPC_ROOT}/qa/production/run_six_user_qa_10min_sequential_0p5_fresh30.sbatch}"
LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
PYTHON="${TRAIN_ENV}/bin/python"

# Three ten-packet profiles use the async vLLM serving frontend so independent
# judges are rendered concurrently before entering one shared GPU scheduler.
# The 24-hour allocation still leaves conservative room for three model starts.
TARGET_COUNT="${TARGET_COUNT:-10}"
QA_PACKET_LIMIT="${QA_PACKET_LIMIT:-10}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
SOURCE_WINDOW_COUNT="${SOURCE_WINDOW_COUNT:-80}"
EVIDENCE_RANDOM_SEED="${EVIDENCE_RANDOM_SEED:-20260902}"
VLLM_SAMPLING_SEED="${VLLM_SAMPLING_SEED:-20260906}"
REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT:-2}"
MAX_GROUPS="${MAX_GROUPS:-1000}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
VLLM_REQUIRED_VERSION="${VLLM_REQUIRED_VERSION:-0.28.0}"
FLASHINFER_REQUIRED_VERSION="${FLASHINFER_REQUIRED_VERSION:-0.6.16.post3}"
VLLM_SERVER_API_COUNT="${VLLM_SERVER_API_COUNT:-5}"
VLLM_RENDERER_NUM_WORKERS="${VLLM_RENDERER_NUM_WORKERS:-1}"
VLLM_MEDIA_LOADING_THREAD_COUNT="${VLLM_MEDIA_LOADING_THREAD_COUNT:-2}"
VLLM_SERVER_OMP_NUM_THREADS="${VLLM_SERVER_OMP_NUM_THREADS:-2}"

OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/outputs/six_user_qa_10min_fps_resolution_ablation}"
EXPERIMENT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
# Keep this path short enough for vLLM's Unix-domain IPC sockets.
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/j/${SLURM_JOB_ID}}"
VLLM_IPC_RUNTIME_ROOT="${VLLM_IPC_RUNTIME_ROOT:-${JOB_SCRATCH_ROOT}/v}"
PERSISTENT_RUNTIME_CACHE_ROOT="${PERSISTENT_RUNTIME_CACHE_ROOT:-${JOB_SCRATCH_ROOT}/runtime_cache}"
PERSISTENT_MODEL_CACHE_ROOT="${PERSISTENT_MODEL_CACHE_ROOT:-${PERSISTENT_RUNTIME_CACHE_ROOT}/huggingface}"
PERSISTENT_KERNEL_CACHE_ROOT="${PERSISTENT_KERNEL_CACHE_ROOT:-${PERSISTENT_RUNTIME_CACHE_ROOT}/kernels}"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"
MEDIA_CACHE_DIR="${JOB_SCRATCH_ROOT}/media_cache"

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
export TMPDIR="${VLLM_IPC_RUNTIME_ROOT}"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
if [[ -z "${HF_TOKEN:-}" && -s "${LOGIN_HF_TOKEN_PATH}" ]]; then
  export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"
fi

export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export FORCE_QWENVL_VIDEO_READER=decord
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

mkdir -p \
  "${HPC_ROOT}/logs" \
  "${EXPERIMENT_ROOT}" \
  "${JOB_SCRATCH_ROOT}" \
  "${PYTHON_ALIAS_ROOT}" \
  "${MEDIA_CACHE_DIR}" \
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
  "${TMPDIR}"

ln -sfn "${PACKAGE_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"

cd "${PROJECT_ROOT}"
test -x "${PYTHON}"
test -s "${RUNTIME_SCRIPT}"
test -s "${PACKAGE_ROOT}/qwen3vl_runner.py"
test -s "${PACKAGE_ROOT}/video_qa_loop.py"
grep -q "expected_judge_video_fps" "${RUNTIME_SCRIPT}" || {
  echo "error=runtime_missing_variable_fps_contract" >&2
  exit 2
}

"${PYTHON}" -P -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${EXPERIMENT_ROOT}/storage_preflight.json"

"${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 1
export VLLM_REQUIRED_VERSION FLASHINFER_REQUIRED_VERSION
"${PYTHON}" -P - <<'PY'
import os
from importlib.metadata import version

import decord
from packaging.version import Version
from qwen_vl_utils.vision_process import get_video_reader_backend
from vllm.vllm_flash_attn import (
    FA2_AVAILABLE,
    FA3_AVAILABLE,
    is_fa_version_supported,
)

required_vllm = Version(os.environ["VLLM_REQUIRED_VERSION"])
detected_vllm = Version(version("vllm"))
if detected_vllm != required_vllm:
    raise SystemExit(
        f"this ablation requires vLLM=={required_vllm}; detected {detected_vllm}"
    )
required_flashinfer = Version(os.environ["FLASHINFER_REQUIRED_VERSION"])
detected_flashinfer = {
    package: Version(version(package))
    for package in ("flashinfer-python", "flashinfer-cubin")
}
for package, detected in detected_flashinfer.items():
    if detected != required_flashinfer:
        raise SystemExit(
            f"this ablation requires {package}=={required_flashinfer}; "
            f"detected {detected}"
        )
backend = get_video_reader_backend()
if backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {backend}")
usable_fa = [candidate for candidate in (2, 3) if is_fa_version_supported(candidate)]
if not usable_fa:
    raise SystemExit(
        "vLLM has no usable compiled FlashAttention extension: "
        f"fa2_available={FA2_AVAILABLE} fa3_available={FA3_AVAILABLE}"
    )
print(
    "ablation_environment_preflight_passed",
    f"vllm={detected_vllm}",
    f"flashinfer={detected_flashinfer}",
    f"flash_attention_versions={usable_fa}",
    f"video_backend={backend}",
    f"decord={decord.__version__}",
)
PY

PROFILE_LABELS=(
  "fps025_px185024"
  "fps050_px091728"
  "fps075_px060368"
)
PROFILE_FPS=("0.25" "0.50" "0.75")
PROFILE_TOTAL_FRAMES=("900" "1800" "2700")
PROFILE_MAX_TOKENS_PER_FRAME=("236" "117" "77")
PROFILE_MAX_PIXELS=("185024" "91728" "60368")
PROFILE_EQUIV_RESOLUTION=("430x430" "303x303" "246x246")

export EXPERIMENT_NAME EXPERIMENT_ROOT TARGET_COUNT QA_PACKET_LIMIT MAX_ATTEMPTS
export SOURCE_WINDOW_COUNT EVIDENCE_RANDOM_SEED VLLM_SAMPLING_SEED REQUIRED_GPU_COUNT
export VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
export VLLM_MEDIA_LOADING_THREAD_COUNT
export VLLM_SERVER_OMP_NUM_THREADS
export RUN_PROFILE_LABELS="${PROFILE_LABELS[*]}"
export RUN_PROFILE_FPS="${PROFILE_FPS[*]}"
export RUN_PROFILE_TOTAL_FRAMES="${PROFILE_TOTAL_FRAMES[*]}"
export RUN_PROFILE_MAX_TOKENS_PER_FRAME="${PROFILE_MAX_TOKENS_PER_FRAME[*]}"
export RUN_PROFILE_MAX_PIXELS="${PROFILE_MAX_PIXELS[*]}"
export RUN_PROFILE_EQUIV_RESOLUTION="${PROFILE_EQUIV_RESOLUTION[*]}"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

def values(name, cast=str):
    return [cast(value) for value in os.environ[name].split()]

labels = values("RUN_PROFILE_LABELS")
fps = values("RUN_PROFILE_FPS", float)
frames = values("RUN_PROFILE_TOTAL_FRAMES", int)
tokens = values("RUN_PROFILE_MAX_TOKENS_PER_FRAME", int)
pixels = values("RUN_PROFILE_MAX_PIXELS", int)
resolution = values("RUN_PROFILE_EQUIV_RESOLUTION")
for profile_fps, total_frames, max_tokens, max_pixels in zip(
    fps, frames, tokens, pixels, strict=True
):
    expected_frames = round(profile_fps * 6 * 600)
    if total_frames != expected_frames:
        raise SystemExit(
            f"profile frame total mismatch: configured={total_frames} "
            f"expected={expected_frames}"
        )
    expected_pixels = max_tokens * 28 * 28
    if max_pixels != expected_pixels:
        raise SystemExit(
            f"profile pixel/token mismatch: configured={max_pixels} "
            f"expected={expected_pixels}"
        )
payload = {
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "target_count_per_profile": int(os.environ["TARGET_COUNT"]),
    "packet_limit_per_profile": int(os.environ["QA_PACKET_LIMIT"]),
    "max_attempts_per_packet": int(os.environ["MAX_ATTEMPTS"]),
    "source_window_count": int(os.environ["SOURCE_WINDOW_COUNT"]),
    "evidence_random_seed": int(os.environ["EVIDENCE_RANDOM_SEED"]),
    "vllm_sampling_seed": int(os.environ["VLLM_SAMPLING_SEED"]),
    "same_evidence_policy": "prepare_once_then_reuse_exact_candidate_jsonl",
    "execution_order": labels,
    "profiles": [
        {
            "label": label,
            "video_fps": profile_fps,
            "total_frames_six_users": total_frames,
            "max_tokens_per_frame": max_tokens,
            "max_pixels_per_frame": max_pixels,
            "equivalent_resolution": equiv_resolution,
        }
        for label, profile_fps, total_frames, max_tokens, max_pixels, equiv_resolution
        in zip(labels, fps, frames, tokens, pixels, resolution, strict=True)
    ],
    "runtime": {
        "inference_backend": "openai-compatible-local",
        "vllm_frontend": "managed_async_server",
        "vllm_version": os.environ["VLLM_REQUIRED_VERSION"],
        "flashinfer_version": os.environ["FLASHINFER_REQUIRED_VERSION"],
        "attention_backend": "FLASH_ATTN",
        "max_num_seqs": 8,
        "max_num_batched_tokens": 65536,
        "batch_wait_ms": 20,
        "api_server_count": int(os.environ["VLLM_SERVER_API_COUNT"]),
        "renderer_num_workers": int(os.environ["VLLM_RENDERER_NUM_WORKERS"]),
        "media_loading_thread_count": int(
            os.environ["VLLM_MEDIA_LOADING_THREAD_COUNT"]
        ),
        "server_omp_num_threads": int(
            os.environ["VLLM_SERVER_OMP_NUM_THREADS"]
        ),
        "local_media_transport": "file_uri",
        "cap_pixels_per_frame": True,
        "tensor_parallel_size": int(os.environ["REQUIRED_GPU_COUNT"]),
    },
}
Path(os.environ["EXPERIMENT_ROOT"], "experiment_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n",
    encoding="utf-8",
)
PY

# Prepare the evidence cohort exactly once, before any timed inference profile.
# FPS and max-pixel settings do not change the retained source segments or
# sampled-image files; they are applied by the vLLM processor in each profile.
SHARED_PREPROCESSING_DIR="${EXPERIMENT_ROOT}/shared_preprocessing"
SHARED_MANIFEST="${SHARED_PREPROCESSING_DIR}/manifest.json"
EXCLUDE_PROCESSED_FROM="${EXCLUDE_PROCESSED_FROM:-${PACKAGE_ROOT}/outputs/six_user_qa_10min}"
EVIDENCE_FRESHNESS_ARGS=()
if [[ -n "${EXCLUDE_PROCESSED_FROM}" ]]; then
  EVIDENCE_FRESHNESS_ARGS+=(--exclude-processed-from "${EXCLUDE_PROCESSED_FROM}")
fi
mkdir -p "${SHARED_PREPROCESSING_DIR}"
printf 'stage=prepare_shared_evidence status=started timestamp=%s target=%s source_windows=%s\n' \
  "$(date --iso-8601=seconds)" "${TARGET_COUNT}" "${SOURCE_WINDOW_COUNT}"
"${PYTHON}" -P -m egolife_two_user_qa build_manifest \
  --output "${SHARED_MANIFEST}" \
  --no-overlays
"${PYTHON}" -P "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" prepare \
  --manifest "${SHARED_MANIFEST}" \
  --output-root "${SHARED_PREPROCESSING_DIR}/ten_minute_setup" \
  --cache-dir "${MEDIA_CACHE_DIR}" \
  --target-count "${TARGET_COUNT}" \
  --source-window-count "${SOURCE_WINDOW_COUNT}" \
  --max-groups "${MAX_GROUPS}" \
  --random-seed "${EVIDENCE_RANDOM_SEED}" \
  --model-id "${CLIP_MODEL_ID}" \
  --device cuda \
  --ffmpeg-binary "${FFMPEG_ENV}/bin/ffmpeg" \
  --clip-batch-size 32 \
  --pruning-clusters-per-window 12 \
  --high-similarity-threshold 0.82 \
  --min-pruned-video-percent 40 \
  --generator-frame-budget 3600 \
  --generator-max-image-pixels "${PROFILE_MAX_PIXELS[0]}" \
  --judge-video-fps "${PROFILE_FPS[0]}" \
  "${EVIDENCE_FRESHNESS_ARGS[@]}"

SHARED_CANDIDATES="${SHARED_PREPROCESSING_DIR}/ten_minute_setup/time_aware_candidates/six_user_10min_time_aware.jsonl"
test -s "${SHARED_CANDIDATES}"
SHARED_CANDIDATE_SHA256="$(sha256sum "${SHARED_CANDIDATES}" | awk '{print $1}')"
printf '%s  %s\n' "${SHARED_CANDIDATE_SHA256}" "${SHARED_CANDIDATES}" \
  > "${EXPERIMENT_ROOT}/shared_candidates.sha256"
printf 'stage=prepare_shared_evidence status=completed timestamp=%s candidate_sha256=%s candidates=%s\n' \
  "$(date --iso-8601=seconds)" "${SHARED_CANDIDATE_SHA256}" \
  "$(wc -l < "${SHARED_CANDIDATES}")"

TIMING_TSV="${EXPERIMENT_ROOT}/profile_timing.tsv"
printf 'profile\tfps\tmax_pixels\ttotal_frames\tstart_epoch\tend_epoch\telapsed_seconds\tcandidate_sha256\n' \
  > "${TIMING_TSV}"

for profile_index in "${!PROFILE_LABELS[@]}"; do
  profile_label="${PROFILE_LABELS[${profile_index}]}"
  profile_fps="${PROFILE_FPS[${profile_index}]}"
  profile_total_frames="${PROFILE_TOTAL_FRAMES[${profile_index}]}"
  profile_max_tokens="${PROFILE_MAX_TOKENS_PER_FRAME[${profile_index}]}"
  profile_max_pixels="${PROFILE_MAX_PIXELS[${profile_index}]}"
  profile_resolution="${PROFILE_EQUIV_RESOLUTION[${profile_index}]}"
  run_mode="${EXPERIMENT_NAME}_${profile_label}"
  profile_output_dir="${EXPERIMENT_ROOT}/runs/${run_mode}_${SLURM_JOB_ID}"

  reuse_preprocessed_from="${SHARED_PREPROCESSING_DIR}"
  require_fresh_evidence="0"
  current_hash="$(sha256sum "${SHARED_CANDIDATES}" | awk '{print $1}')"
  if [[ "${current_hash}" != "${SHARED_CANDIDATE_SHA256}" ]]; then
    echo "error=shared_candidate_changed_before_profile profile=${profile_label}" >&2
    exit 1
  fi

  printf 'profile=%s status=started timestamp=%s fps=%s max_pixels=%s total_frames=%s max_tokens_per_frame=%s resolution=%s reuse=%s\n' \
    "${profile_label}" "$(date --iso-8601=seconds)" "${profile_fps}" \
    "${profile_max_pixels}" "${profile_total_frames}" "${profile_max_tokens}" \
    "${profile_resolution}" "${reuse_preprocessed_from:-none}"
  profile_start_epoch="$(date +%s)"

  (
    export RUN_MODE="${run_mode}"
    export OUTPUT_BASE="${EXPERIMENT_ROOT}/runs"
    export TARGET_COUNT QA_PACKET_LIMIT MAX_ATTEMPTS SOURCE_WINDOW_COUNT
    export EVIDENCE_RANDOM_SEED REQUIRED_GPU_COUNT
    export VLLM_SAMPLING_SEED
    export REUSE_PREPROCESSED_FROM="${reuse_preprocessed_from}"
    export REQUIRE_FRESH_EVIDENCE="${require_fresh_evidence}"
    export RESUME_QA_FROM=""
    export COHORT_SKIP_SOURCE=""
    export SKIP_EVIDENCE_IDS=""
    export INFERENCE_BACKEND="openai-compatible-local"
    export START_VLLM_SERVER="1"
    export VLLM_SERVER_HOST="127.0.0.1"
    export VLLM_SERVER_PORT="$((18000 + profile_index))"
    export VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
    export VLLM_MEDIA_LOADING_THREAD_COUNT
    export VLLM_SERVER_OMP_NUM_THREADS
    export VLLM_OPENAI_USE_LOCAL_MEDIA_URIS="1"
    export VLLM_REQUIRED_VERSION FLASHINFER_REQUIRED_VERSION
    export VLLM_TENSOR_PARALLEL_SIZE="${REQUIRED_GPU_COUNT}"
    export VLLM_GPU_MEMORY_UTILIZATION="0.86"
    export VLLM_MAX_MODEL_LEN="262144"
    export VLLM_MAX_NUM_SEQS="8"
    export VLLM_BATCH_WAIT_MS="20"
    export VLLM_MAX_NUM_BATCHED_TOKENS="65536"
    export VLLM_MAX_IMAGES="3600"
    export VLLM_MAX_VIDEOS="20"
    export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
    export VLLM_MM_ENCODER_ATTN_BACKEND="FLASH_ATTN"
    export VLLM_GDN_PREFILL_BACKEND="flashinfer"
    export VLLM_MM_ENCODER_TP_MODE="data"
    export VLLM_MIN_IMAGE_PIXELS="3136"
    export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="3136"
    export VLLM_VIDEO_FPS="${profile_fps}"
    export JUDGE_VIDEO_FPS="${profile_fps}"
    export GENERATOR_MAX_IMAGE_PIXELS="${profile_max_pixels}"
    export QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="${profile_max_pixels}"
    export ENABLE_CUDA_KEEPER="0"
    export JOB_SCRATCH_ROOT VLLM_IPC_RUNTIME_ROOT
    export PERSISTENT_RUNTIME_CACHE_ROOT PERSISTENT_MODEL_CACHE_ROOT
    export PERSISTENT_KERNEL_CACHE_ROOT
    bash "${RUNTIME_SCRIPT}"
  )

  current_hash="$(sha256sum "${SHARED_CANDIDATES}" | awk '{print $1}')"
  if [[ "${current_hash}" != "${SHARED_CANDIDATE_SHA256}" ]]; then
    echo "error=shared_candidate_changed_after_profile profile=${profile_label}" >&2
    exit 1
  fi

  profile_end_epoch="$(date +%s)"
  profile_elapsed_seconds="$((profile_end_epoch - profile_start_epoch))"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${profile_label}" "${profile_fps}" "${profile_max_pixels}" \
    "${profile_total_frames}" "${profile_start_epoch}" "${profile_end_epoch}" \
    "${profile_elapsed_seconds}" "${SHARED_CANDIDATE_SHA256}" >> "${TIMING_TSV}"
  printf 'profile=%s status=completed timestamp=%s elapsed_seconds=%s candidate_sha256=%s output_dir=%s\n' \
    "${profile_label}" "$(date --iso-8601=seconds)" \
    "${profile_elapsed_seconds}" "${SHARED_CANDIDATE_SHA256}" \
    "${profile_output_dir}"
done

printf 'status=completed experiment_root=%s timing=%s candidate_sha256=%s\n' \
  "${EXPERIMENT_ROOT}" "${TIMING_TSV}" "${SHARED_CANDIDATE_SHA256}"
