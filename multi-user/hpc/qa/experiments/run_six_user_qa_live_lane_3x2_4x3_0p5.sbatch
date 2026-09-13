#!/usr/bin/env bash
#SBATCH --job-name=egoqa_live_lane_ab
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --partition=h200_tandon
#SBATCH --qos=gpu48
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

# This is a real, output-producing generator/judge comparison.  Both profiles
# consume the same pinned ten-packet evidence JSONL and run all live retries and
# judges.  Unlike the frozen trace replay, each profile writes qa_mcq.jsonl.
EXPERIMENT_NAME="${EXPERIMENT_NAME:-six_user_qa_live_lane_3x2_4x3_0p5}"
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
PROFILE_RUNNER="${PROFILE_RUNNER:-${HPC_ROOT}/qa/experiments/run_six_user_qa_1h200_throughput_probe_0p5.sbatch}"
RUNTIME_SCRIPT="${RUNTIME_SCRIPT:-${HPC_ROOT}/qa/production/run_six_user_qa_10min_sequential_0p5_fresh30.sbatch}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda_high_duty.py}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
PYTHON="${TRAIN_ENV}/bin/python"
SETSID_EXECUTABLE="$(command -v setsid || true)"

SOURCE_PREPROCESSING_ROOT="${SOURCE_PREPROCESSING_ROOT:-${PACKAGE_ROOT}/outputs/six_user_qa_10min_fps_resolution_ablation/six_user_qa_10min_vllm_fps_resolution_3x10_17109425/shared_preprocessing}"
EXPECTED_SOURCE_CANDIDATE_SHA256="${EXPECTED_SOURCE_CANDIDATE_SHA256:-7e2d75fdeb3399e3bd614e1af07508bbc93ea75a7d88aa0997634c9940184f49}"

OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/outputs/six_user_qa_live_lane_ab}"
EXPERIMENT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
PROFILE_OUTPUT_BASE="${EXPERIMENT_ROOT}/profiles"
JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT:-/scratch/${USER}/j/${SLURM_JOB_ID}}"
SHARED_RUNTIME_CACHE_ROOT="${JOB_SCRATCH_ROOT}/runtime_cache"
SHARED_MODEL_CACHE_ROOT="${SHARED_RUNTIME_CACHE_ROOT}/huggingface"
SHARED_KERNEL_CACHE_ROOT="${SHARED_RUNTIME_CACHE_ROOT}/kernels"
ORCHESTRATION_STATUS="${EXPERIMENT_ROOT}/profile_status.tsv"
JOB_GPU_LOG="${EXPERIMENT_ROOT}/gpu_1s.csv"

PROFILE_LABELS=("lane_3x2" "lane_4x3")
PROFILE_PACKETS=("3" "4")
PROFILE_REVIEWS=("2" "3")

# Frozen across both profiles.  The profile runner validates these values and
# the pinned evidence SHA before model loading.
TARGET_COUNT="10"
MAX_ATTEMPTS="3"
JUDGE_VIDEO_FPS="0.50"
GENERATOR_MAX_IMAGE_PIXELS="91728"
VLLM_MIN_IMAGE_PIXELS="3136"
VLLM_SERVER_API_COUNT="${VLLM_SERVER_API_COUNT:-3}"
VLLM_RENDERER_NUM_WORKERS="${VLLM_RENDERER_NUM_WORKERS:-1}"
VLLM_MEDIA_LOADING_THREAD_COUNT="${VLLM_MEDIA_LOADING_THREAD_COUNT:-4}"
VLLM_SERVER_OMP_NUM_THREADS="${VLLM_SERVER_OMP_NUM_THREADS:-2}"
VLLM_MM_PROCESSOR_CACHE_GB="${VLLM_MM_PROCESSOR_CACHE_GB:-2}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-12}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_ENABLE_MFU_METRICS="${VLLM_ENABLE_MFU_METRICS:-1}"

# The profile runner accepts only the keeper's legacy arguments.  This dedicated
# self-contained keeper supplies the repaired HBM-reclaim-aware high-duty logic:
# one low-priority stream, a 300-second policy window, and short idle pulses that
# target 100% device activity without reserving more than 1 GiB.
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-70}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-18.0}"
CUDA_KEEPER_MAX_PREALLOC_GB="${CUDA_KEEPER_MAX_PREALLOC_GB:-1.0}"
CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-60000}"
CUDA_KEEPER_WINDOW_SECONDS="300"
CUDA_KEEPER_TARGET_MARGIN=""
CUDA_KEEPER_IDLE_UTIL_THRESHOLD="10"
CUDA_KEEPER_IDLE_CONFIRM_SECONDS="0.0"
CUDA_KEEPER_MAX_PULSE_SECONDS="0.8"

ACTIVE_PROFILE_PID=""
GPU_MONITOR_PID=""
CURRENT_PROFILE="bootstrap"
CURRENT_STAGE="bootstrap"

stage() {
  CURRENT_STAGE="$1"
  printf 'live_lane_ab_stage=%s status=started timestamp=%s job_id=%s host=%s\n' \
    "$1" "$(date --iso-8601=seconds)" "${SLURM_JOB_ID}" "$(hostname)"
}

on_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'live_lane_ab_stage=%s status=failed exit_code=%s line=%s command=%q profile=%s\n' \
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
  for _ in $(seq 1 15); do
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

write_interrupted_status() {
  local signal_name="$1"
  printf '%s\t%s\tinterrupted\t%s\t%s\n' \
    "${CURRENT_PROFILE}" "$(date --iso-8601=seconds)" "${signal_name}" \
    "${ACTIVE_PROFILE_PID:-none}" >> "${ORCHESTRATION_STATUS}" 2>/dev/null || true
}

handle_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - INT TERM EXIT
  write_interrupted_status "${signal_name}"
  stop_process_group "${ACTIVE_PROFILE_PID}"
  ACTIVE_PROFILE_PID=""
  stop_pid "${GPU_MONITOR_PID}"
  GPU_MONITOR_PID=""
  printf 'live_lane_ab_status=interrupted signal=%s profile=%s output=%s\n' \
    "${signal_name}" "${CURRENT_PROFILE}" "${EXPERIMENT_ROOT}" >&2
  exit "${exit_code}"
}

cleanup() {
  stop_process_group "${ACTIVE_PROFILE_PID}"
  stop_pid "${GPU_MONITOR_PID}"
}

trap 'handle_signal INT 130' INT
trap 'handle_signal TERM 143' TERM
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap cleanup EXIT

monitor_gpu() {
  set +e
  printf '%s\n' \
    'sample_iso,sample_epoch,gpu_index,gpu_name,sm_util_percent,memory_used_mib,memory_free_mib,power_watts,power_limit_watts,temperature_c' \
    > "${JOB_GPU_LOG}"
  while true; do
    local sample_iso sample_epoch values
    sample_iso="$(date --iso-8601=seconds)"
    sample_epoch="$(date +%s.%N)"
    values="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,power.draw,power.limit,temperature.gpu --format=csv,noheader,nounits 2>/dev/null)"
    if [[ -n "${values}" ]]; then
      printf '%s,%s,%s\n' "${sample_iso}" "${sample_epoch}" "${values}" >> "${JOB_GPU_LOG}"
    else
      printf '# %s nvidia_smi_query_failed\n' "${sample_iso}" >> "${JOB_GPU_LOG}"
    fi
    sleep 1
  done
}

stage "validate_layout"
cd "${PROJECT_ROOT}"
test -x "${PYTHON}"
test -x "${SETSID_EXECUTABLE}"
test -s "${PROFILE_RUNNER}"
test -s "${RUNTIME_SCRIPT}"
test -s "${CUDA_KEEPER_SCRIPT}"
test -x "${FFMPEG_ENV}/bin/ffmpeg"
test -x "${FFMPEG_ENV}/bin/ffprobe"
require_marker 'MAX_PACKETS_IN_FLIGHT' "${PROFILE_RUNNER}" profile_runner_env
require_marker 'MAX_REVIEW_LANES' "${PROFILE_RUNNER}" profile_runner_env
require_marker 'bash "${RUNTIME_SCRIPT}"' "${PROFILE_RUNNER}" profile_runner_runtime_dispatch
require_marker '--max-packets-in-flight' "${RUNTIME_SCRIPT}" qa_runtime
require_marker '--max-review-lanes' "${RUNTIME_SCRIPT}" qa_runtime
require_marker 'qa_mcq.jsonl' "${RUNTIME_SCRIPT}" qa_runtime
require_marker 'cuda_high_duty_keeper' "${CUDA_KEEPER_SCRIPT}" repaired_keeper
require_marker '--max-pulse-seconds' "${CUDA_KEEPER_SCRIPT}" repaired_keeper
require_marker 'reclaimable_free_gb' "${CUDA_KEEPER_SCRIPT}" repaired_keeper
require_marker 'lowest_stream_priority' "${CUDA_KEEPER_SCRIPT}" repaired_keeper
if [[ "${VLLM_RENDERER_NUM_WORKERS}" != "1" ]]; then
  echo "error=renderer_num_workers_must_remain_1 value=${VLLM_RENDERER_NUM_WORKERS}" >&2
  exit 2
fi
if ! [[ "${CUDA_KEEPER_THRESHOLD}" =~ ^[0-9]+$ ]] || \
  ((CUDA_KEEPER_THRESHOLD < 1 || CUDA_KEEPER_THRESHOLD > 100)); then
  echo "error=invalid_keeper_threshold value=${CUDA_KEEPER_THRESHOLD}" >&2
  exit 2
fi
CUDA_KEEPER_TARGET_MARGIN="$((100 - CUDA_KEEPER_THRESHOLD))"

mkdir -p "${HPC_ROOT}/logs" "${EXPERIMENT_ROOT}" "${PROFILE_OUTPUT_BASE}" \
  "${JOB_SCRATCH_ROOT}" "${SHARED_RUNTIME_CACHE_ROOT}" \
  "${SHARED_MODEL_CACHE_ROOT}" "${SHARED_KERNEL_CACHE_ROOT}"
printf 'profile\tstarted_at\tstatus\texit_or_signal\tprocess_id\n' > "${ORCHESTRATION_STATUS}"

stage "video_preflight"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
# The invoked profile runner performs the mandatory scratch-variable storage
# preflight before each fresh model start.
require_marker 'training.torch_storage_preflight' "${PROFILE_RUNNER}" storage_preflight
"${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 1
PYTHONPATH="${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}" "${PYTHON}" -P - <<'PY'
from torchcodec.decoders import VideoDecoder

print("torchcodec_video_decoder_preflight_passed", VideoDecoder.__module__)
PY

stage "validate_pinned_evidence"
SOURCE_CANDIDATES="${SOURCE_PREPROCESSING_ROOT%/}/ten_minute_setup/time_aware_candidates/six_user_10min_time_aware.jsonl"
test -s "${SOURCE_CANDIDATES}"
SOURCE_SHA256="$(sha256sum "${SOURCE_CANDIDATES}" | awk '{print $1}')"
if [[ "${SOURCE_SHA256}" != "${EXPECTED_SOURCE_CANDIDATE_SHA256}" ]]; then
  echo "error=source_candidate_sha256_mismatch expected=${EXPECTED_SOURCE_CANDIDATE_SHA256} actual=${SOURCE_SHA256}" >&2
  exit 2
fi
printf '%s  %s\n' "${SOURCE_SHA256}" "${SOURCE_CANDIDATES}" \
  > "${EXPERIMENT_ROOT}/source_candidates.sha256"

export EXPERIMENT_NAME EXPERIMENT_ROOT SOURCE_PREPROCESSING_ROOT
export RUNTIME_SCRIPT
export EXPECTED_SOURCE_CANDIDATE_SHA256 TARGET_COUNT MAX_ATTEMPTS
export JUDGE_VIDEO_FPS GENERATOR_MAX_IMAGE_PIXELS VLLM_MIN_IMAGE_PIXELS
export VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
export VLLM_MEDIA_LOADING_THREAD_COUNT VLLM_SERVER_OMP_NUM_THREADS
export VLLM_MM_PROCESSOR_CACHE_GB VLLM_GPU_MEMORY_UTILIZATION
export VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
export VLLM_ENABLE_PREFIX_CACHING VLLM_ENABLE_MFU_METRICS
export CUDA_KEEPER_SCRIPT CUDA_KEEPER_THRESHOLD CUDA_KEEPER_GPUS
export CUDA_KEEPER_RESERVE CUDA_KEEPER_MAX_PREALLOC_GB
export CUDA_KEEPER_START_USED_MIB
export CUDA_KEEPER_WINDOW_SECONDS CUDA_KEEPER_TARGET_MARGIN
export CUDA_KEEPER_IDLE_UTIL_THRESHOLD CUDA_KEEPER_IDLE_CONFIRM_SECONDS
export CUDA_KEEPER_MAX_PULSE_SECONDS
export RUN_PROFILE_LABELS="${PROFILE_LABELS[*]}"
export RUN_PROFILE_PACKETS="${PROFILE_PACKETS[*]}"
export RUN_PROFILE_REVIEWS="${PROFILE_REVIEWS[*]}"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

labels = os.environ["RUN_PROFILE_LABELS"].split()
packets = [int(value) for value in os.environ["RUN_PROFILE_PACKETS"].split()]
reviews = [int(value) for value in os.environ["RUN_PROFILE_REVIEWS"].split()]
payload = {
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "purpose": "live generator-judge lane comparison on identical preprocessed evidence",
    "execution": "single_sbatch_sequential_non_array_fresh_vllm_per_profile",
    "profiles": [
        {
            "label": label,
            "max_packets_in_flight": packet_count,
            "generation_lanes": 1,
            "max_review_lanes": review_count,
        }
        for label, packet_count, review_count in zip(labels, packets, reviews, strict=True)
    ],
    "workload": {
        "target_count": int(os.environ["TARGET_COUNT"]),
        "max_attempts": int(os.environ["MAX_ATTEMPTS"]),
        "judge_video_fps": float(os.environ["JUDGE_VIDEO_FPS"]),
        "generator_max_image_pixels": int(os.environ["GENERATOR_MAX_IMAGE_PIXELS"]),
        "source_preprocessing_root": os.environ["SOURCE_PREPROCESSING_ROOT"],
        "source_candidate_sha256": os.environ["EXPECTED_SOURCE_CANDIDATE_SHA256"],
        "live_outputs": True,
        "expected_primary_output": "qa_mcq.jsonl per profile",
    },
    "vllm": {
        "api_server_count": int(os.environ["VLLM_SERVER_API_COUNT"]),
        "renderer_num_workers": int(os.environ["VLLM_RENDERER_NUM_WORKERS"]),
        "media_loading_thread_count": int(os.environ["VLLM_MEDIA_LOADING_THREAD_COUNT"]),
        "gpu_memory_utilization": float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]),
        "max_num_seqs": int(os.environ["VLLM_MAX_NUM_SEQS"]),
        "max_num_batched_tokens": int(os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"]),
    },
    "cuda_keeper": {
        "script": os.environ["CUDA_KEEPER_SCRIPT"],
        "threshold_percent": int(os.environ["CUDA_KEEPER_THRESHOLD"]),
        "target_margin_percent": int(os.environ["CUDA_KEEPER_TARGET_MARGIN"]),
        "target_percent": 100,
        "rolling_window_seconds": int(os.environ["CUDA_KEEPER_WINDOW_SECONDS"]),
        "control_sample_period_seconds": 1.0,
        "cuda_streams_per_gpu": 1,
        "stream_priority": "lowest",
        "idle_util_threshold_percent": int(os.environ["CUDA_KEEPER_IDLE_UTIL_THRESHOLD"]),
        "idle_confirm_seconds": float(os.environ["CUDA_KEEPER_IDLE_CONFIRM_SECONDS"]),
        "maximum_pulse_seconds": float(os.environ["CUDA_KEEPER_MAX_PULSE_SECONDS"]),
        "max_prealloc_gib": float(os.environ["CUDA_KEEPER_MAX_PREALLOC_GB"]),
        "reserve_gib": float(os.environ["CUDA_KEEPER_RESERVE"]),
        "owned_by_profile_runner": True,
        "work_is_not_productive_inference": True,
    },
}
Path(os.environ["EXPERIMENT_ROOT"], "experiment_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

stage "start_job_gpu_monitor"
monitor_gpu &
GPU_MONITOR_PID=$!

for profile_index in "${!PROFILE_LABELS[@]}"; do
  profile_label="${PROFILE_LABELS[${profile_index}]}"
  packet_count="${PROFILE_PACKETS[${profile_index}]}"
  review_count="${PROFILE_REVIEWS[${profile_index}]}"
  profile_experiment_name="${EXPERIMENT_NAME}_${profile_label}"
  profile_root="${PROFILE_OUTPUT_BASE}/${profile_experiment_name}_${SLURM_JOB_ID}"
  profile_run_mode="${profile_experiment_name}_fps050_px091728"
  profile_run_output="${profile_root}/runs/${profile_run_mode}_${SLURM_JOB_ID}"
  profile_scratch="${JOB_SCRATCH_ROOT}/p${profile_index}"
  profile_started="$(date --iso-8601=seconds)"
  profile_start_epoch="$(date +%s)"
  CURRENT_PROFILE="${profile_label}"

  stage "profile_${profile_label}"
  printf '%s\t%s\trunning\t-\t-\n' \
    "${profile_label}" "${profile_started}" >> "${ORCHESTRATION_STATUS}"

  (
    export EXPERIMENT_NAME="${profile_experiment_name}"
    export OUTPUT_BASE="${PROFILE_OUTPUT_BASE}"
    export RUNTIME_SCRIPT
    export SOURCE_PREPROCESSING_ROOT EXPECTED_SOURCE_CANDIDATE_SHA256
    export MAX_PACKETS_IN_FLIGHT="${packet_count}"
    export MAX_REVIEW_LANES="${review_count}"
    export VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
    export VLLM_MEDIA_LOADING_THREAD_COUNT VLLM_SERVER_OMP_NUM_THREADS
    export VLLM_MM_PROCESSOR_CACHE_GB VLLM_GPU_MEMORY_UTILIZATION
    export VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
    export VLLM_ENABLE_PREFIX_CACHING VLLM_ENABLE_MFU_METRICS
    export ENABLE_CUDA_KEEPER="1"
    export CUDA_KEEPER_SCRIPT CUDA_KEEPER_THRESHOLD CUDA_KEEPER_GPUS
    export CUDA_KEEPER_RESERVE CUDA_KEEPER_MAX_PREALLOC_GB
    export CUDA_KEEPER_START_USED_MIB
    # Keep the storage-preflight boundary at the whole Slurm job so the two
    # sequential profiles may share model/kernel caches beneath runtime_cache.
    # Only short-lived IPC paths are isolated per profile.
    export JOB_SCRATCH_ROOT
    export VLLM_IPC_RUNTIME_ROOT="${profile_scratch}/v"
    export PERSISTENT_RUNTIME_CACHE_ROOT="${SHARED_RUNTIME_CACHE_ROOT}"
    export PERSISTENT_MODEL_CACHE_ROOT="${SHARED_MODEL_CACHE_ROOT}"
    export PERSISTENT_KERNEL_CACHE_ROOT="${SHARED_KERNEL_CACHE_ROOT}"
    exec "${SETSID_EXECUTABLE}" bash "${PROFILE_RUNNER}"
  ) > >(
    awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
      | tee "${EXPERIMENT_ROOT}/${profile_label}_stdout_timeline.log"
  ) 2> >(
    awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
      | tee "${EXPERIMENT_ROOT}/${profile_label}_stderr_timeline.log" >&2
  ) &
  ACTIVE_PROFILE_PID=$!

  # ERR traps still fire for a failing command even when errexit is toggled in
  # some Bash contexts.  A command used as an if-condition is explicitly
  # exempt, so capture the child status here without losing its diagnostics.
  if wait "${ACTIVE_PROFILE_PID}"; then
    profile_exit_code=0
  else
    profile_exit_code=$?
  fi
  ACTIVE_PROFILE_PID=""

  profile_end_epoch="$(date +%s)"
  printf '%s\t%s\t%s\t%s\t-\n' \
    "${profile_label}" "$(date --iso-8601=seconds)" \
    "$([[ "${profile_exit_code}" == "0" ]] && printf completed || printf failed)" \
    "${profile_exit_code}" >> "${ORCHESTRATION_STATUS}"
  printf 'profile=%s status=%s elapsed_seconds=%s output=%s qa=%s\n' \
    "${profile_label}" \
    "$([[ "${profile_exit_code}" == "0" ]] && printf completed || printf failed)" \
    "$((profile_end_epoch - profile_start_epoch))" "${profile_root}" \
    "${profile_run_output}/qa_mcq.jsonl"

  if [[ "${profile_exit_code}" != "0" ]]; then
    printf 'error=profile_runner_failed profile=%s exit_code=%s stdout_timeline=%s stderr_timeline=%s\n' \
      "${profile_label}" "${profile_exit_code}" \
      "${EXPERIMENT_ROOT}/${profile_label}_stdout_timeline.log" \
      "${EXPERIMENT_ROOT}/${profile_label}_stderr_timeline.log" >&2
    exit "${profile_exit_code}"
  fi
  test -s "${profile_run_output}/video_first_prompts.jsonl"
  # qa_mcq.jsonl may validly be empty when every real candidate is rejected;
  # the child runner's strict validator is the authoritative semantic gate.
  test -f "${profile_run_output}/qa_mcq.jsonl"
done

CURRENT_PROFILE="summarize"
stage "summarize_live_profiles"
export PROFILE_OUTPUT_BASE
"${PYTHON}" -P - <<'PY'
import csv
import json
import os
import re
import statistics
from pathlib import Path

root = Path(os.environ["EXPERIMENT_ROOT"])
profile_base = Path(os.environ["PROFILE_OUTPUT_BASE"])
job_id = os.environ["SLURM_JOB_ID"]
experiment_name = os.environ["EXPERIMENT_NAME"]
profiles = []
for label, packets, reviews in zip(
    os.environ["RUN_PROFILE_LABELS"].split(),
    map(int, os.environ["RUN_PROFILE_PACKETS"].split()),
    map(int, os.environ["RUN_PROFILE_REVIEWS"].split()),
    strict=True,
):
    child_name = f"{experiment_name}_{label}"
    child_root = profile_base / f"{child_name}_{job_id}"
    run_root = child_root / "runs" / f"{child_name}_fps050_px091728_{job_id}"
    prompts_path = run_root / "video_first_prompts.jsonl"
    accepted_path = run_root / "qa_mcq.jsonl"
    rejected_path = run_root / "qa_mcq.rejected.jsonl"
    metrics_path = child_root / "metrics" / "metrics_summary.json"
    prompt_rows = [
        json.loads(line)
        for line in prompts_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    generation_attempts = sum(row.get("stage") == "generation" for row in prompt_rows)
    accepted = sum(bool(line.strip()) for line in accepted_path.read_text(encoding="utf-8").splitlines())
    rejected = (
        sum(bool(line.strip()) for line in rejected_path.read_text(encoding="utf-8").splitlines())
        if rejected_path.exists()
        else 0
    )
    timeline = (root / f"{label}_stdout_timeline.log").read_text(
        encoding="utf-8", errors="replace"
    )
    matches = re.findall(r"qa_packet_pipeline_done .*?packets=(\d+) accepted=(\d+) seconds=([0-9.]+)", timeline)
    if len(matches) != 1:
        raise SystemExit(f"{label} needs exactly one completed live packet pipeline; found {len(matches)}")
    completed_packets, logged_accepted, pipeline_seconds = matches[0]
    completed_packets = int(completed_packets)
    logged_accepted = int(logged_accepted)
    pipeline_seconds = float(pipeline_seconds)
    if logged_accepted != accepted:
        raise SystemExit(
            f"{label} accepted count mismatch: timeline={logged_accepted} file={accepted}"
        )
    keeper_log = run_root / f"cuda_keeper_{job_id}.log"
    keeper_lines = keeper_log.read_text(encoding="utf-8", errors="replace").splitlines()
    keeper_samples = [line for line in keeper_lines if "burning:" in line]
    keeper_burns = [line for line in keeper_samples if "burning:Y" in line]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    gpu_metrics = metrics.get("gpu") or {}
    scheduler_gpu_mean = gpu_metrics.get("sm_util_mean_percent")
    profiles.append(
        {
            "label": label,
            "max_packets_in_flight": packets,
            "generation_lanes": 1,
            "max_review_lanes": reviews,
            "completed_packet_loops": completed_packets,
            "pipeline_seconds": pipeline_seconds,
            "packet_loops_per_hour": round(completed_packets * 3600 / pipeline_seconds, 6),
            "generation_attempts": generation_attempts,
            "generation_attempts_per_hour": round(generation_attempts * 3600 / pipeline_seconds, 6),
            "accepted_qamcq": accepted,
            "accepted_qamcq_per_hour": round(accepted * 3600 / pipeline_seconds, 6),
            "rejected_rows": rejected,
            "qa_mcq": str(accepted_path),
            "prompts": str(prompts_path),
            "keeper_controller_samples": len(keeper_samples),
            "keeper_burning_samples": len(keeper_burns),
            "keeper_burning_fraction": (
                round(len(keeper_burns) / len(keeper_samples), 6)
                if keeper_samples
                else None
            ),
            "scheduler_gpu": gpu_metrics,
            "scheduler_utilization_gate_passed": (
                scheduler_gpu_mean is not None and scheduler_gpu_mean >= 60.0
            ),
            "vllm_metrics": metrics.get("vllm"),
        }
    )

baseline, candidate = profiles
job_gpu_values = []
with (root / "gpu_1s.csv").open(encoding="utf-8", newline="") as handle:
    for row in csv.DictReader(line for line in handle if not line.startswith("#")):
        try:
            job_gpu_values.append(float(row["sm_util_percent"]))
        except (KeyError, TypeError, ValueError):
            pass
result = {
    "status": "completed",
    "evidence_level": "live_generator_judge_pipeline",
    "profiles": profiles,
    "comparison": {
        "baseline": baseline["label"],
        "candidate": candidate["label"],
        "packet_loop_throughput_gain_percent": round(
            (candidate["packet_loops_per_hour"] / baseline["packet_loops_per_hour"] - 1) * 100,
            6,
        ),
        "generation_attempt_throughput_gain_percent": round(
            (
                candidate["generation_attempts_per_hour"]
                / baseline["generation_attempts_per_hour"]
                - 1
            )
            * 100,
            6,
        ),
    },
    "job_wide_scheduler_gpu": {
        "samples": len(job_gpu_values),
        "sm_util_mean_percent": (
            round(statistics.fmean(job_gpu_values), 6) if job_gpu_values else None
        ),
        "utilization_gate_passed": (
            bool(job_gpu_values) and statistics.fmean(job_gpu_values) >= 60.0
        ),
    },
    "warning": "CUDA keeper work is excluded from useful QA throughput interpretation.",
}
(root / "live_profile_comparison.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("live_lane_ab_comparison", json.dumps(result, sort_keys=True))
PY

stage "complete"
CURRENT_PROFILE="complete"
printf 'live_lane_ab_status=completed output=%s comparison=%s\n' \
  "${EXPERIMENT_ROOT}" "${EXPERIMENT_ROOT}/live_profile_comparison.json"
