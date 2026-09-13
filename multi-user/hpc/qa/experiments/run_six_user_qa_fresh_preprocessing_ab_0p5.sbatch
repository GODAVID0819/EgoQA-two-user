#!/usr/bin/env bash
#SBATCH --job-name=egoqa_freshprep_ab05
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=400G
#SBATCH --time=8:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

# Run four sequential, non-array profiles in one allocation: the direct
# segment-to-RGB core refactor on a cold cache; low-risk pipeline optimizations
# with legacy MP4/PNG materialization on a separate cold cache; an identical
# streaming warm-cache replay; then
# a production 6/4/5 generator-judge loop reusing the streaming candidates.
# Preprocessing profiles run first so their comparison remains salvageable if
# the later QA phase is interrupted.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-six_user_qa_fresh_preprocessing_ab_0p5}"
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
RUNTIME_SCRIPT="${RUNTIME_SCRIPT:-${HPC_ROOT}/qa/production/run_six_user_qa_10min_sequential_0p5_fresh30.sbatch}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda_high_duty.py}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
PYTHON="${TRAIN_ENV}/bin/python"
SETSID_EXECUTABLE="$(command -v setsid || true)"

PREVIOUS_COHORT_ROOT="${PREVIOUS_COHORT_ROOT:-${PACKAGE_ROOT}/outputs/six_user_qa_10min_fps_resolution_ablation/six_user_qa_10min_vllm_fps_resolution_3x10_17109425/shared_preprocessing}"
FRESHNESS_HISTORY_ROOT="${FRESHNESS_HISTORY_ROOT:-${PREVIOUS_COHORT_ROOT}}"
OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/outputs/six_user_qa_fresh_preprocessing_ab}"
EXPERIMENT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
PROFILE_OUTPUT_BASE="${EXPERIMENT_ROOT}/profiles"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/j/${SLURM_JOB_ID}}"
# Validate persistent caches against the whole Slurm-job scratch tree while
# keeping each A/B profile's transient/media scratch isolated under p0/p1.
STORAGE_PREFLIGHT_ALLOWED_ROOT="${JOB_SCRATCH_ROOT}"
SHARED_RUNTIME_CACHE_ROOT="${JOB_SCRATCH_ROOT}/runtime_cache"
SHARED_MODEL_CACHE_ROOT="${SHARED_RUNTIME_CACHE_ROOT}/huggingface"
SHARED_KERNEL_CACHE_ROOT="${SHARED_RUNTIME_CACHE_ROOT}/kernels"
SHARED_PREPROCESS_CACHE_BASE="${JOB_SCRATCH_ROOT}/preprocess_cache"
STATUS_PATH="${EXPERIMENT_ROOT}/profile_status.tsv"
JOB_GPU_LOG="${EXPERIMENT_ROOT}/gpu_1s.csv"
JOB_GPU_PROCESS_LOG="${EXPERIMENT_ROOT}/gpu_processes_2s.log"
JOB_MEMORY_LOG="${EXPERIMENT_ROOT}/cgroup_memory_2s.tsv"
JOB_PROCESS_LOG="${EXPERIMENT_ROOT}/top_processes_5s.log"

TARGET_COUNT="10"
MAX_ATTEMPTS="3"
SOURCE_WINDOW_COUNT="${SOURCE_WINDOW_COUNT:-80}"
MAX_GROUPS="${MAX_GROUPS:-1000}"
EVIDENCE_RANDOM_SEED="${EVIDENCE_RANDOM_SEED:-20260902}"
VLLM_SAMPLING_SEED="${VLLM_SAMPLING_SEED:-20260906}"
JUDGE_VIDEO_FPS="0.50"
GENERATOR_MAX_IMAGE_PIXELS="91728"

# The first three profiles measure preprocessing.  The fourth is the only
# output-producing QA run and reuses the streaming cold profile's artifacts.
PROFILE_LABELS=(
  "streaming_cold_b128_w3_m3_preprocess_only"
  "lowrisk_cold_b128_w3_m3_preprocess_only"
  "streaming_warm_b128_w3_m3_preprocess_only"
  "streaming_reuse_p6_g4_r5_e2e"
)
PROFILE_CLIP_BATCHES=("128" "128" "128" "128")
PROFILE_VIDEO_WORKERS=("3" "3" "3" "3")
PROFILE_MEDIA_PREPARE_WORKERS=("3" "3" "3" "3")
PROFILE_WINDOW_DOUBLE_BUFFER=("1" "1" "1" "1")
PROFILE_ASSEMBLE_LONG_VIDEO=("0" "1" "0" "0")
PROFILE_DEFER_GAZE_SUMMARY=("1" "1" "1" "1")
PROFILE_DIRECT_SEGMENT_RGB=("1" "0" "1" "1")
PROFILE_CACHE_FAMILY=("streaming" "lowrisk" "streaming" "streaming")
PROFILE_PREPROCESS_ONLY=("1" "1" "1" "0")
PROFILE_PACKETS=("1" "1" "1" "6")
PROFILE_GENERATIONS=("1" "1" "1" "4")
PROFILE_REVIEWS=("1" "1" "1" "5")
PROFILE_BACKENDS=("vllm-local" "vllm-local" "vllm-local" "openai-compatible-local")
PROFILE_START_SERVERS=("0" "0" "0" "1")

VLLM_SERVER_API_COUNT="3"
VLLM_RENDERER_NUM_WORKERS="1"
VLLM_MEDIA_LOADING_THREAD_COUNT="4"
VLLM_SERVER_OMP_NUM_THREADS="2"
VLLM_MM_PROCESSOR_CACHE_GB="8"
VLLM_GPU_MEMORY_UTILIZATION="0.82"
VLLM_MAX_NUM_SEQS="16"
VLLM_MAX_NUM_BATCHED_TOKENS="65536"
VLLM_ENABLE_PREFIX_CACHING="1"
VLLM_ENABLE_MFU_METRICS="1"

CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-70}"
CUDA_KEEPER_GPUS="0"
CUDA_KEEPER_RESERVE="15.0"
CUDA_KEEPER_MAX_PREALLOC_GB="0.25"
CUDA_KEEPER_EMERGENCY_MIN_FREE_GB="2.0"
CUDA_KEEPER_START_USED_MIB="1024"
CUDA_KEEPER_WINDOW_SECONDS="300"
CUDA_KEEPER_TARGET_MARGIN="$((100 - CUDA_KEEPER_THRESHOLD))"
CUDA_KEEPER_IDLE_UTIL_THRESHOLD="10"
CUDA_KEEPER_IDLE_CONFIRM_SECONDS="0.0"
CUDA_KEEPER_MAX_PULSE_SECONDS="0.8"
CUDA_KEEPER_ACTIVE_REQUEST_IDLE_CONFIRM_SECONDS="1.0"
CUDA_KEEPER_ACTIVE_REQUEST_MAX_PULSE_SECONDS="0.30"

MONITOR_PIDS=()
ACTIVE_PROFILE_PID=""
CURRENT_PROFILE="bootstrap"
CURRENT_STAGE="bootstrap"
OVERALL_EXIT_CODE=0

stage() {
  CURRENT_STAGE="$1"
  printf 'fresh_preprocessing_ab_stage=%s status=started timestamp=%s job_id=%s host=%s profile=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" "${SLURM_JOB_ID}" \
    "$(hostname)" "${CURRENT_PROFILE}"
}

on_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'fresh_preprocessing_ab_stage=%s status=failed exit_code=%s line=%s command=%q profile=%s\n' \
    "${CURRENT_STAGE}" "${exit_code}" "${line_number}" "${failed_command}" \
    "${CURRENT_PROFILE}" >&2
  exit "${exit_code}"
}

require_marker() {
  local marker="$1"
  local path="$2"
  local contract="$3"
  if ! grep -Fq -- "${marker}" "${path}"; then
    printf 'error=missing_contract_marker contract=%s marker=%q path=%s\n' \
      "${contract}" "${marker}" "${path}" >&2
    return 2
  fi
}

stop_pid() {
  local pid="${1:-}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
  fi
}

stop_process_group() {
  local leader="${1:-}"
  if [[ -z "${leader}" ]] || ! kill -0 "${leader}" 2>/dev/null; then
    return 0
  fi
  kill -TERM -- "-${leader}" 2>/dev/null || true
  for _ in $(seq 1 20); do
    if ! kill -0 "${leader}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${leader}" 2>/dev/null; then
    kill -KILL -- "-${leader}" 2>/dev/null || true
  fi
  wait "${leader}" 2>/dev/null || true
}

cleanup() {
  trap - EXIT INT TERM
  stop_process_group "${ACTIVE_PROFILE_PID}"
  ACTIVE_PROFILE_PID=""
  for pid in "${MONITOR_PIDS[@]}"; do
    stop_pid "${pid}"
  done
}

handle_signal() {
  local signal_name="$1"
  local exit_code="$2"
  printf '%s\t%s\t%s\t%s\t%s\tinterrupted\t%s\t-\n' \
    "${CURRENT_PROFILE}" "$(date --iso-8601=seconds)" "$(date +%s)" \
    "$(date --iso-8601=seconds)" "$(date +%s)" "${signal_name}" \
    >> "${STATUS_PATH}" 2>/dev/null || true
  cleanup
  exit "${exit_code}"
}

trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap cleanup EXIT

monitor_gpu() {
  set +e
  printf '%s\n' 'sample_iso,sample_epoch,gpu_index,gpu_name,sm_util_percent,memory_used_mib,memory_free_mib,power_watts,power_limit_watts,temperature_c' > "${JOB_GPU_LOG}"
  while true; do
    local values
    values="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)"
    if [[ -n "${values}" ]]; then
      printf '%s,%s,%s\n' "$(date --iso-8601=seconds)" "$(date +%s.%N)" \
        "${values}" >> "${JOB_GPU_LOG}"
    fi
    sleep 1
  done
}

monitor_gpu_processes() {
  set +e
  while true; do
    printf '# timestamp=%s epoch=%s\n' "$(date --iso-8601=seconds)" \
      "$(date +%s.%N)" >> "${JOB_GPU_PROCESS_LOG}"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits >> "${JOB_GPU_PROCESS_LOG}" 2>&1
    sleep 2
  done
}

monitor_cgroup_memory() {
  set +e
  local relative_path cgroup_dir
  relative_path="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)"
  cgroup_dir="/sys/fs/cgroup${relative_path}"
  printf 'sample_iso\tsample_epoch\tmemory_current_bytes\tmemory_peak_bytes\tmemory_max\tswap_current_bytes\n' > "${JOB_MEMORY_LOG}"
  while true; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "$(date +%s.%N)" \
      "$(cat "${cgroup_dir}/memory.current" 2>/dev/null || printf unavailable)" \
      "$(cat "${cgroup_dir}/memory.peak" 2>/dev/null || printf unavailable)" \
      "$(cat "${cgroup_dir}/memory.max" 2>/dev/null || printf unavailable)" \
      "$(cat "${cgroup_dir}/memory.swap.current" 2>/dev/null || printf unavailable)" \
      >> "${JOB_MEMORY_LOG}"
    sleep 2
  done
}

monitor_processes() {
  set +e
  while true; do
    printf '# timestamp=%s epoch=%s\n' "$(date --iso-8601=seconds)" \
      "$(date +%s.%N)" >> "${JOB_PROCESS_LOG}"
    ps -eo pid=,ppid=,rss=,vsz=,%cpu=,%mem=,etimes=,stat=,comm=,args= \
      --sort=-rss | head -n 40 >> "${JOB_PROCESS_LOG}"
    sleep 5
  done
}

stage "validate_layout"
cd "${PROJECT_ROOT}"
test -x "${PYTHON}"
test -x "${SETSID_EXECUTABLE}"
test -x "${FFMPEG_ENV}/bin/ffmpeg"
test -x "${FFMPEG_ENV}/bin/ffprobe"
test -s "${RUNTIME_SCRIPT}"
test -s "${CUDA_KEEPER_SCRIPT}"
test -s "${PACKAGE_ROOT}/ten_minute_six_user_setup.py"
test -s "${PACKAGE_ROOT}/group_relative_clip_sampling.py"
test -s "${PACKAGE_ROOT}/evidence.py"
test -d "${FRESHNESS_HISTORY_ROOT}"
require_marker 'PREPROCESS_ONLY' "${RUNTIME_SCRIPT}" preprocessing_only
require_marker 'preprocessing_timing.json' "${RUNTIME_SCRIPT}" preprocessing_timing
require_marker '--video-sample-workers' "${RUNTIME_SCRIPT}" preprocessing_runtime
require_marker '--video-sample-workers' "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" preprocessing_cli
require_marker 'video_sample_workers' "${PACKAGE_ROOT}/group_relative_clip_sampling.py" parallel_sampling
require_marker '--media-prepare-workers' "${RUNTIME_SCRIPT}" media_prepare_runtime
require_marker '--media-prepare-workers' "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" media_prepare_cli
require_marker '--window-double-buffer' "${RUNTIME_SCRIPT}" window_double_buffer_runtime
require_marker '--direct-segment-rgb' "${RUNTIME_SCRIPT}" direct_rgb_runtime
require_marker '--temporal-kmeans-time-weight' "${RUNTIME_SCRIPT}" temporal_kmeans_runtime
require_marker 'time_gate_applied_before_cosine' "${PACKAGE_ROOT}/group_relative_clip_sampling.py" early_time_gate
require_marker 'sample_source_segments_rgb' "${PACKAGE_ROOT}/group_relative_clip_sampling.py" direct_segment_decoder
require_marker 'clip_embeddings_v2' "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" persistent_embedding_cache
require_marker 'skip_incomplete_groups=True' "${PACKAGE_ROOT}/ten_minute_six_user_setup.py" incomplete_window_filter
require_marker 'parallel_user_media_preparation' "${PACKAGE_ROOT}/evidence.py" parallel_media_prepare
require_marker '--max-generation-lanes' "${RUNTIME_SCRIPT}" qa_runtime
require_marker 'CUDA_KEEPER_PAUSE_FILE' "${RUNTIME_SCRIPT}" keeper_request_signal
require_marker 'cuda_high_duty_keeper' "${CUDA_KEEPER_SCRIPT}" repaired_keeper
if [[ "${VLLM_RENDERER_NUM_WORKERS}" != "1" ]]; then
  echo "error=renderer_num_workers_must_remain_1" >&2
  exit 2
fi

mkdir -p "${HPC_ROOT}/logs" "${EXPERIMENT_ROOT}" "${PROFILE_OUTPUT_BASE}" \
  "${JOB_SCRATCH_ROOT}" "${SHARED_RUNTIME_CACHE_ROOT}" \
  "${SHARED_MODEL_CACHE_ROOT}" "${SHARED_KERNEL_CACHE_ROOT}" \
  "${SHARED_PREPROCESS_CACHE_BASE}"
printf 'profile\tstarted_at\tstart_epoch\tended_at\tend_epoch\tstatus\texit_code\toutput_dir\n' \
  > "${STATUS_PATH}"

export EXPERIMENT_NAME EXPERIMENT_ROOT TARGET_COUNT MAX_ATTEMPTS
export SOURCE_WINDOW_COUNT MAX_GROUPS EVIDENCE_RANDOM_SEED
export VLLM_SAMPLING_SEED JUDGE_VIDEO_FPS GENERATOR_MAX_IMAGE_PIXELS
export FRESHNESS_HISTORY_ROOT CUDA_KEEPER_THRESHOLD
export RUN_PROFILE_LABELS="${PROFILE_LABELS[*]}"
export RUN_PROFILE_CLIP_BATCHES="${PROFILE_CLIP_BATCHES[*]}"
export RUN_PROFILE_VIDEO_WORKERS="${PROFILE_VIDEO_WORKERS[*]}"
export RUN_PROFILE_MEDIA_PREPARE_WORKERS="${PROFILE_MEDIA_PREPARE_WORKERS[*]}"
export RUN_PROFILE_WINDOW_DOUBLE_BUFFER="${PROFILE_WINDOW_DOUBLE_BUFFER[*]}"
export RUN_PROFILE_ASSEMBLE_LONG_VIDEO="${PROFILE_ASSEMBLE_LONG_VIDEO[*]}"
export RUN_PROFILE_DEFER_GAZE_SUMMARY="${PROFILE_DEFER_GAZE_SUMMARY[*]}"
export RUN_PROFILE_DIRECT_SEGMENT_RGB="${PROFILE_DIRECT_SEGMENT_RGB[*]}"
export RUN_PROFILE_CACHE_FAMILY="${PROFILE_CACHE_FAMILY[*]}"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "purpose": "measure low-risk preprocessing optimizations, the streaming RGB core refactor, persistent-cache reuse, and the production 6/4/5 generator-judge loop",
    "execution": "single_sbatch_sequential_non_array",
    "profile_order": os.environ["RUN_PROFILE_LABELS"].split(),
    "order_bias": "streaming runs first/cold while low-risk runs second on a separate persistent cache, so any host or remote cache advantage biases against the streaming speedup; only the explicit streaming-warm profile reuses its preprocessing cache",
    "paired_preprocessing": {
        "same_random_seed": int(os.environ["EVIDENCE_RANDOM_SEED"]),
        "same_freshness_history_root": os.environ["FRESHNESS_HISTORY_ROOT"],
        "separate_media_cache_directories": True,
        "clip_batch_sizes": [int(v) for v in os.environ["RUN_PROFILE_CLIP_BATCHES"].split()],
        "video_sample_workers": [int(v) for v in os.environ["RUN_PROFILE_VIDEO_WORKERS"].split()],
        "media_prepare_workers": [
            int(v) for v in os.environ["RUN_PROFILE_MEDIA_PREPARE_WORKERS"].split()
        ],
        "window_double_buffer": [
            v == "1" for v in os.environ["RUN_PROFILE_WINDOW_DOUBLE_BUFFER"].split()
        ],
        "assemble_long_video": [
            v == "1" for v in os.environ["RUN_PROFILE_ASSEMBLE_LONG_VIDEO"].split()
        ],
        "defer_gaze_summary": [
            v == "1" for v in os.environ["RUN_PROFILE_DEFER_GAZE_SUMMARY"].split()
        ],
        "direct_segment_rgb": [
            v == "1" for v in os.environ["RUN_PROFILE_DIRECT_SEGMENT_RGB"].split()
        ],
        "cache_family": os.environ["RUN_PROFILE_CACHE_FAMILY"].split(),
    },
    "qa_profile": {
        "max_packets_in_flight": 6,
        "max_generation_lanes": 4,
        "max_review_lanes": 5,
        "target_count": int(os.environ["TARGET_COUNT"]),
        "max_attempts": int(os.environ["MAX_ATTEMPTS"]),
        "judge_video_fps": float(os.environ["JUDGE_VIDEO_FPS"]),
        "generator_max_image_pixels": int(os.environ["GENERATOR_MAX_IMAGE_PIXELS"]),
    },
    "cuda_keeper": {
        "threshold_percent": int(os.environ["CUDA_KEEPER_THRESHOLD"]),
        "interpretation": "scheduler keepalive only; exclude burner work from useful preprocessing and QA throughput",
    },
}
Path(os.environ["EXPERIMENT_ROOT"], "experiment_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

stage "start_resource_monitors"
monitor_gpu & MONITOR_PIDS+=("$!")
monitor_gpu_processes & MONITOR_PIDS+=("$!")
monitor_cgroup_memory & MONITOR_PIDS+=("$!")
monitor_processes & MONITOR_PIDS+=("$!")
if command -v vmstat >/dev/null 2>&1; then
  vmstat -t 1 > "${EXPERIMENT_ROOT}/vmstat_1s.log" 2>&1 &
  MONITOR_PIDS+=("$!")
fi
if command -v iostat >/dev/null 2>&1; then
  iostat -dx 1 > "${EXPERIMENT_ROOT}/iostat_1s.log" 2>&1 &
  MONITOR_PIDS+=("$!")
fi

for profile_index in "${!PROFILE_LABELS[@]}"; do
  label="${PROFILE_LABELS[${profile_index}]}"
  clip_batch="${PROFILE_CLIP_BATCHES[${profile_index}]}"
  video_workers="${PROFILE_VIDEO_WORKERS[${profile_index}]}"
  media_prepare_workers="${PROFILE_MEDIA_PREPARE_WORKERS[${profile_index}]}"
  window_double_buffer="${PROFILE_WINDOW_DOUBLE_BUFFER[${profile_index}]}"
  assemble_long_video="${PROFILE_ASSEMBLE_LONG_VIDEO[${profile_index}]}"
  defer_gaze_summary="${PROFILE_DEFER_GAZE_SUMMARY[${profile_index}]}"
  direct_segment_rgb="${PROFILE_DIRECT_SEGMENT_RGB[${profile_index}]}"
  cache_family="${PROFILE_CACHE_FAMILY[${profile_index}]}"
  preprocess_only="${PROFILE_PREPROCESS_ONLY[${profile_index}]}"
  packets="${PROFILE_PACKETS[${profile_index}]}"
  generations="${PROFILE_GENERATIONS[${profile_index}]}"
  reviews="${PROFILE_REVIEWS[${profile_index}]}"
  backend="${PROFILE_BACKENDS[${profile_index}]}"
  start_server="${PROFILE_START_SERVERS[${profile_index}]}"
  reuse_preprocessed_from=""
  require_fresh_evidence="1"
  exclude_processed_from="${FRESHNESS_HISTORY_ROOT}"
  if [[ "${label}" == "streaming_reuse_p6_g4_r5_e2e" ]]; then
    reuse_preprocessed_from="${PROFILE_OUTPUT_BASE}/${EXPERIMENT_NAME}_streaming_cold_b128_w3_m3_preprocess_only_${SLURM_JOB_ID}"
    require_fresh_evidence="0"
    exclude_processed_from=""
  fi
  profile_mode="${EXPERIMENT_NAME}_${label}"
  profile_output="${PROFILE_OUTPUT_BASE}/${profile_mode}_${SLURM_JOB_ID}"
  profile_scratch="${JOB_SCRATCH_ROOT}/p${profile_index}"
  timeline_out="${EXPERIMENT_ROOT}/${label}_stdout_timeline.log"
  timeline_err="${EXPERIMENT_ROOT}/${label}_stderr_timeline.log"
  start_iso="$(date --iso-8601=seconds)"
  start_epoch="$(date +%s)"
  CURRENT_PROFILE="${label}"
  stage "profile_${label}"

  (
    export RUN_MODE="${profile_mode}"
    export OUTPUT_BASE="${PROFILE_OUTPUT_BASE}"
    export JOB_SCRATCH_ROOT="${profile_scratch}"
    export STORAGE_PREFLIGHT_ALLOWED_ROOT
    export VLLM_IPC_RUNTIME_ROOT="${profile_scratch}/v"
    export PERSISTENT_RUNTIME_CACHE_ROOT="${SHARED_RUNTIME_CACHE_ROOT}"
    export PERSISTENT_MODEL_CACHE_ROOT="${SHARED_MODEL_CACHE_ROOT}"
    export PERSISTENT_KERNEL_CACHE_ROOT="${SHARED_KERNEL_CACHE_ROOT}"
    export PERSISTENT_PREPROCESS_CACHE_ROOT="${SHARED_PREPROCESS_CACHE_BASE}/${cache_family}"
    export TARGET_COUNT MAX_ATTEMPTS SOURCE_WINDOW_COUNT MAX_GROUPS
    export EVIDENCE_RANDOM_SEED
    export REUSE_PREPROCESSED_FROM="${reuse_preprocessed_from}"
    export REQUIRE_FRESH_EVIDENCE="${require_fresh_evidence}"
    export EXCLUDE_PROCESSED_FROM="${exclude_processed_from}"
    export PREPROCESS_ONLY="${preprocess_only}"
    export CLIP_BATCH_SIZE="${clip_batch}"
    export PREPROCESSING_VIDEO_SAMPLE_WORKERS="${video_workers}"
    export PREPROCESSING_MEDIA_PREPARE_WORKERS="${media_prepare_workers}"
    export PREPROCESSING_WINDOW_DOUBLE_BUFFER="${window_double_buffer}"
    export PREPROCESSING_ASSEMBLE_LONG_VIDEO="${assemble_long_video}"
    export PREPROCESSING_DEFER_GAZE_SUMMARY="${defer_gaze_summary}"
    export PREPROCESSING_DIRECT_SEGMENT_RGB="${direct_segment_rgb}"
    export TEMPORAL_KMEANS_TIME_WEIGHT="0.1"
    export TEMPORAL_UNIT_SECONDS="30"
    export MAX_PACKETS_IN_FLIGHT="${packets}"
    export MAX_GENERATION_LANES="${generations}"
    export MAX_REVIEW_LANES="${reviews}"
    export INFERENCE_BACKEND="${backend}"
    export START_VLLM_SERVER="${start_server}"
    export VLLM_OPENAI_USE_LOCAL_MEDIA_URIS="${start_server}"
    export REQUIRED_GPU_COUNT="1"
    export JUDGE_VIDEO_FPS GENERATOR_MAX_IMAGE_PIXELS
    export VLLM_SAMPLING_SEED
    export VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
    export VLLM_MEDIA_LOADING_THREAD_COUNT VLLM_SERVER_OMP_NUM_THREADS
    export VLLM_MM_PROCESSOR_CACHE_GB VLLM_GPU_MEMORY_UTILIZATION
    export VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
    export VLLM_ENABLE_PREFIX_CACHING VLLM_ENABLE_MFU_METRICS
    export ENABLE_CUDA_KEEPER="1"
    export CUDA_KEEPER_SCRIPT CUDA_KEEPER_THRESHOLD CUDA_KEEPER_GPUS
    export CUDA_KEEPER_RESERVE CUDA_KEEPER_MAX_PREALLOC_GB
    export CUDA_KEEPER_EMERGENCY_MIN_FREE_GB CUDA_KEEPER_START_USED_MIB
    export CUDA_KEEPER_WINDOW_SECONDS CUDA_KEEPER_TARGET_MARGIN
    export CUDA_KEEPER_IDLE_UTIL_THRESHOLD CUDA_KEEPER_IDLE_CONFIRM_SECONDS
    export CUDA_KEEPER_MAX_PULSE_SECONDS
    export CUDA_KEEPER_ACTIVE_REQUEST_IDLE_CONFIRM_SECONDS
    export CUDA_KEEPER_ACTIVE_REQUEST_MAX_PULSE_SECONDS
    exec "${SETSID_EXECUTABLE}" bash "${RUNTIME_SCRIPT}"
  ) > >(
    awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
      | tee "${timeline_out}"
  ) 2> >(
    awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
      | tee "${timeline_err}" >&2
  ) &
  ACTIVE_PROFILE_PID=$!

  if wait "${ACTIVE_PROFILE_PID}"; then
    profile_exit_code=0
  else
    profile_exit_code=$?
  fi
  ACTIVE_PROFILE_PID=""
  end_iso="$(date --iso-8601=seconds)"
  end_epoch="$(date +%s)"
  profile_status="$([[ "${profile_exit_code}" == "0" ]] && printf completed || printf failed)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${label}" "${start_iso}" "${start_epoch}" "${end_iso}" "${end_epoch}" \
    "${profile_status}" "${profile_exit_code}" "${profile_output}" \
    >> "${STATUS_PATH}"
  printf 'fresh_preprocessing_profile=%s status=%s elapsed_seconds=%s output=%s\n' \
    "${label}" "${profile_status}" "$((end_epoch - start_epoch))" "${profile_output}"
  if [[ "${profile_exit_code}" != "0" ]]; then
    OVERALL_EXIT_CODE="${profile_exit_code}"
    printf 'error=runtime_profile_failed profile=%s exit_code=%s stdout_timeline=%s stderr_timeline=%s\n' \
      "${label}" "${profile_exit_code}" "${timeline_out}" "${timeline_err}" >&2
  fi
done

CURRENT_PROFILE="summarize"
stage "summarize_results"
export STATUS_PATH JOB_GPU_LOG PROFILE_OUTPUT_BASE OVERALL_EXIT_CODE
"${PYTHON}" -P - <<'PY'
import csv
import json
import os
import re
import statistics
from pathlib import Path

root = Path(os.environ["EXPERIMENT_ROOT"])
job_id = os.environ["SLURM_JOB_ID"]
experiment_name = os.environ["EXPERIMENT_NAME"]
with Path(os.environ["STATUS_PATH"]).open(encoding="utf-8", newline="") as handle:
    statuses = list(csv.DictReader(handle, delimiter="\t"))
status_by_label = {row["profile"]: row for row in statuses}

def output_dir(label: str) -> Path:
    return Path(os.environ["PROFILE_OUTPUT_BASE"]) / f"{experiment_name}_{label}_{job_id}"

def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def log_breakdown(label: str) -> dict:
    text = (root / f"{label}_stdout_timeline.log").read_text(
        encoding="utf-8", errors="replace"
    )
    encoder = re.findall(r"ten_minute_prepare status=encoder_loaded .*?seconds=([0-9.]+)", text)
    source = re.findall(r"ten_minute_prepare status=source_analysis_completed .*?seconds=([0-9.]+)", text)
    evidence = re.findall(r"evidence_window status=prepared .*?seconds=([0-9.]+)", text)
    extraction = re.findall(r"sampled_video_frames status=extracted .*?seconds=([0-9.]+)", text)
    direct_rgb = re.findall(r"sampled_video_rgb status=decoded .*?seconds=([0-9.]+)", text)
    frame_cache_hits = re.findall(r"sampled_video_frames status=cache_hit", text)
    direct_rgb_cache_hits = re.findall(r"sampled_video_rgb status=cache_hit", text)
    embedding = re.findall(r"six_user_analysis status=embedding_completed .*?seconds=([0-9.]+)", text)
    embedding_cache_hits = re.findall(
        r"six_user_analysis status=embedding_completed .*?cache_hit=true", text
    )
    manifest_cache_hits = re.findall(r"evidence_manifest status=cache_hit", text)
    manifest_cache_writes = re.findall(r"evidence_manifest status=cache_written", text)
    media_prepare = re.findall(
        r"evidence_window status=parallel_user_media_preparation_completed .*?seconds=([0-9.]+)",
        text,
    )
    incomplete = re.findall(
        r"evidence_window status=skipped .*?reason=insufficient_users", text
    )
    return {
        "encoder_load_seconds": sum(map(float, encoder)),
        "source_analysis_work_seconds": sum(map(float, source)),
        "evidence_window_work_seconds": sum(map(float, evidence)),
        "ffmpeg_sampling_work_seconds": sum(map(float, extraction)),
        "direct_rgb_decode_work_seconds": sum(map(float, direct_rgb)),
        "parallel_user_media_prepare_wall_seconds": sum(map(float, media_prepare)),
        "clip_embedding_wall_seconds": sum(map(float, embedding)),
        "frame_cache_hit_video_count": len(frame_cache_hits) + len(direct_rgb_cache_hits),
        "embedding_cache_hit_window_count": len(embedding_cache_hits),
        "manifest_cache_hit_count": len(manifest_cache_hits),
        "manifest_cache_write_count": len(manifest_cache_writes),
        "incomplete_windows_skipped_before_media": len(incomplete),
        "source_analysis_count": len(source),
        "sampled_video_count": len(extraction) + len(direct_rgb),
    }

profiles = {}
for label in (
    "lowrisk_cold_b128_w3_m3_preprocess_only",
    "streaming_cold_b128_w3_m3_preprocess_only",
    "streaming_warm_b128_w3_m3_preprocess_only",
    "streaming_reuse_p6_g4_r5_e2e",
):
    status = status_by_label.get(label)
    child = output_dir(label)
    timing_path = child / "preprocessing_timing.json"
    if status is None or not timing_path.exists():
        profiles[label] = {
            "status": status.get("status") if status else "not_run",
            "exit_code": int(status["exit_code"]) if status else None,
            "preprocessing_completed": False,
        }
        continue
    timing = json.loads(timing_path.read_text(encoding="utf-8"))
    candidates_path = Path(timing["candidate_jsonl"])
    candidates = jsonl(candidates_path)
    profiles[label] = {
        "status": status["status"],
        "exit_code": int(status["exit_code"]),
        "preprocessing_completed": True,
        "runtime_wall_seconds": int(status["end_epoch"]) - int(status["start_epoch"]),
        "preprocessing": timing,
        "breakdown": log_breakdown(label),
        "candidate_count": len(candidates),
        "source_window_evidence_ids": sorted(
            {str(row.get("source_window_evidence_id") or "") for row in candidates}
        ),
        "candidate_evidence_ids": [str(row.get("evidence_id") or "") for row in candidates],
    }

lowrisk = profiles.get("lowrisk_cold_b128_w3_m3_preprocess_only", {})
streaming_cold = profiles.get("streaming_cold_b128_w3_m3_preprocess_only", {})
streaming_warm = profiles.get("streaming_warm_b128_w3_m3_preprocess_only", {})
qa_profile = profiles.get("streaming_reuse_p6_g4_r5_e2e", {})
comparison = None
if lowrisk.get("preprocessing_completed") and streaming_cold.get("preprocessing_completed"):
    lowrisk_seconds = int(lowrisk["preprocessing"]["elapsed_seconds"])
    streaming_cold_seconds = int(streaming_cold["preprocessing"]["elapsed_seconds"])
    comparison = {
        "lowrisk_cold_preprocessing_seconds": lowrisk_seconds,
        "streaming_cold_preprocessing_seconds": streaming_cold_seconds,
        "streaming_core_speedup_ratio": round(lowrisk_seconds / streaming_cold_seconds, 6),
        "elapsed_reduction_percent": round(
            100 * (lowrisk_seconds - streaming_cold_seconds) / lowrisk_seconds, 6
        ),
        "source_windows_match": (
            lowrisk["source_window_evidence_ids"]
            == streaming_cold["source_window_evidence_ids"]
        ),
        "candidate_evidence_ids_match": (
            lowrisk["candidate_evidence_ids"]
            == streaming_cold["candidate_evidence_ids"]
        ),
        "interpretation": (
            "both cold profiles use identical worker counts, CLIP batch size, source "
            "selection, and separate preprocessing caches; only the media pipeline differs"
        ),
    }
    if streaming_warm.get("preprocessing_completed"):
        streaming_warm_seconds = int(
            streaming_warm["preprocessing"]["elapsed_seconds"]
        )
        comparison.update(
            {
                "streaming_warm_preprocessing_seconds": streaming_warm_seconds,
                "persistent_cache_speedup_ratio": round(
                    streaming_cold_seconds / streaming_warm_seconds, 6
                ),
                "warm_source_windows_match": (
                    streaming_cold["source_window_evidence_ids"]
                    == streaming_warm["source_window_evidence_ids"]
                ),
                "warm_candidate_evidence_ids_match": (
                    streaming_cold["candidate_evidence_ids"]
                    == streaming_warm["candidate_evidence_ids"]
                ),
            }
        )

qa = None
streaming_child = output_dir("streaming_reuse_p6_g4_r5_e2e")
intermediate_path = streaming_child / "qa_mcq.intermediate.jsonl"
if qa_profile.get("status") == "completed" and intermediate_path.exists():
    rows = jsonl(intermediate_path)
    attempts = sum(len(row.get("attempts") or []) for row in rows)
    accepted_path = streaming_child / "qa_mcq.jsonl"
    accepted = len(jsonl(accepted_path))
    timeline = (root / "streaming_reuse_p6_g4_r5_e2e_stdout_timeline.log").read_text(
        encoding="utf-8", errors="replace"
    )
    matches = re.findall(
        r"qa_packet_pipeline_done .*?packets=(\d+) accepted=(\d+) seconds=([0-9.]+)",
        timeline,
    )
    if len(matches) != 1:
        raise SystemExit(
            f"expected one completed streaming QA pipeline marker; found {len(matches)}"
        )
    logged_packets, logged_accepted, pipeline_seconds = matches[0]
    pipeline_seconds = float(pipeline_seconds)
    if int(logged_packets) != len(rows) or int(logged_accepted) != accepted:
        raise SystemExit("streaming QA counts disagree with the pipeline marker")
    qa_runtime_wall = int(qa_profile["runtime_wall_seconds"])
    preprocessing_seconds = int(streaming_cold["preprocessing"]["elapsed_seconds"])
    preprocessing_profile_wall = int(streaming_cold["runtime_wall_seconds"])
    composed_wall = preprocessing_profile_wall + qa_runtime_wall
    useful_pipeline_seconds = preprocessing_seconds + pipeline_seconds
    qa = {
        "completed_packet_loops": len(rows),
        "generation_attempts": attempts,
        "accepted_qamcq": accepted,
        "qa_pipeline_seconds": pipeline_seconds,
        "qa_only_packet_loops_per_hour": round(len(rows) * 3600 / pipeline_seconds, 6),
        "qa_only_generation_attempts_per_hour": round(attempts * 3600 / pipeline_seconds, 6),
        "fresh_preprocessing_seconds": preprocessing_seconds,
        "fresh_preprocessing_profile_wall_seconds": preprocessing_profile_wall,
        "qa_profile_wall_seconds": qa_runtime_wall,
        "composed_end_to_end_wall_seconds": composed_wall,
        "composed_useful_pipeline_seconds": round(useful_pipeline_seconds, 6),
        "end_to_end_packet_loops_per_hour": round(len(rows) * 3600 / composed_wall, 6),
        "end_to_end_generation_attempts_per_hour": round(attempts * 3600 / composed_wall, 6),
        "useful_pipeline_packet_loops_per_hour": round(
            len(rows) * 3600 / useful_pipeline_seconds, 6
        ),
        "non_qa_pipeline_overhead_seconds": round(composed_wall - pipeline_seconds, 6),
        "preprocessing_fraction_of_full_runtime": round(preprocessing_seconds / composed_wall, 6),
        "composition_note": (
            "The streaming-cold preprocessing and QA reuse profiles are sequential phases in one "
            "allocation. Their wall times are added; this conservatively includes two runtime "
            "environment preflights and excludes the intervening low-risk and warm-cache timing arms."
        ),
        "qa_mcq": str(accepted_path),
        "intermediate": str(intermediate_path),
    }

gpu_rows = []
gpu_path = Path(os.environ["JOB_GPU_LOG"])
if gpu_path.exists():
    with gpu_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            try:
                gpu_rows.append(
                    {
                        "epoch": float(row["sample_epoch"]),
                        "util": float(row["sm_util_percent"]),
                        "memory": float(row["memory_used_mib"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                pass

def gpu_summary(rows: list[dict]) -> dict | None:
    if not rows:
        return None
    utils = [row["util"] for row in rows]
    return {
        "samples": len(rows),
        "sm_util_mean_percent": round(statistics.fmean(utils), 6),
        "fraction_below_10_percent": round(sum(v < 10 for v in utils) / len(utils), 6),
        "fraction_at_least_80_percent": round(sum(v >= 80 for v in utils) / len(utils), 6),
        "max_memory_used_mib": max(row["memory"] for row in rows),
    }

gpu = {"job_wide": gpu_summary(gpu_rows), "profiles": {}}
for label, status in status_by_label.items():
    start = float(status["start_epoch"])
    end = float(status["end_epoch"])
    gpu["profiles"][label] = gpu_summary(
        [row for row in gpu_rows if start <= row["epoch"] <= end]
    )

payload = {
    "status": "completed" if int(os.environ["OVERALL_EXIT_CODE"]) == 0 else "partial_failure",
    "profiles": profiles,
    "preprocessing_comparison": comparison,
    "streaming_end_to_end_qa": qa,
    "scheduler_gpu_including_keeper": gpu,
    "warning": "CUDA keeper activity is scheduler keepalive work, not useful preprocessing or QA throughput.",
}
(root / "fresh_preprocessing_comparison.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
print("fresh_preprocessing_comparison", json.dumps(payload, sort_keys=True))
PY

CURRENT_STAGE="complete"
printf 'fresh_preprocessing_ab_status=%s exit_code=%s comparison=%s\n' \
  "$([[ "${OVERALL_EXIT_CODE}" == "0" ]] && printf completed || printf partial_failure)" \
  "${OVERALL_EXIT_CODE}" "${EXPERIMENT_ROOT}/fresh_preprocessing_comparison.json"
exit "${OVERALL_EXIT_CODE}"
