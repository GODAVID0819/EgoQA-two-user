#!/usr/bin/env bash
#SBATCH --job-name=egoqa_1h200_probe05
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

EXPERIMENT_NAME="${EXPERIMENT_NAME:?missing EXPERIMENT_NAME}"
REASONING_MODE="${REASONING_MODE:?missing REASONING_MODE}"
if [[ "${REASONING_MODE}" != "nr" && "${REASONING_MODE}" != "r" ]]; then
  echo "error=invalid_reasoning_mode value=${REASONING_MODE}" >&2
  exit 2
fi
JOB_START_EPOCH_SECONDS="$(date +%s)"
PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
RUNTIME_SCRIPT="${RUNTIME_SCRIPT:-${PACKAGE_ROOT}/hpc/qa/production/run_six_user_qa_10min_sequential_0p5_fresh30.sbatch}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen38-vllm}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
PYTHON="${TRAIN_ENV}/bin/python"
LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"

# This is one fixed FPS/resolution profile over the exact ten-packet cohort
# used by jobs 17109425 and 17186857.  Every packet retains its complete
# three-attempt generation/review loop.  The default limit-test shape keeps
# one memory-heavy generation lane and permits two packet reviews at once,
# so three packets can actively feed the async vLLM server without allowing
# multiple 1,800-2,000-image generator requests to materialize together.
PROFILE_LABEL="fps050_px091728"
JUDGE_VIDEO_FPS="0.50"
GENERATOR_MAX_IMAGE_PIXELS="91728"
VLLM_MIN_IMAGE_PIXELS="3136"
TARGET_COUNT="10"
MAX_ATTEMPTS="3"
MAX_PACKETS_IN_FLIGHT="3"
MAX_REVIEW_LANES="2"
SOURCE_WINDOW_COUNT="80"
EVIDENCE_RANDOM_SEED="20260902"
VLLM_SAMPLING_SEED="20260906"
REQUIRED_GPU_COUNT="1"

# Push the frontend and vLLM scheduler while retaining HBM headroom for the
# multimodal encoder.  Renderer workers stay at one because vLLM forbids
# renderer_num_workers>1 with the multimodal processor cache enabled; three
# API processes provide three independent async render paths instead.
VLLM_SERVER_API_COUNT="3"
VLLM_RENDERER_NUM_WORKERS="1"
VLLM_MEDIA_LOADING_THREAD_COUNT="4"
VLLM_SERVER_OMP_NUM_THREADS="2"
VLLM_MM_PROCESSOR_CACHE_GB="2"
VLLM_GPU_MEMORY_UTILIZATION="0.82"
VLLM_MAX_NUM_SEQS="12"
VLLM_MAX_NUM_BATCHED_TOKENS="65536"
VLLM_ENABLE_PREFIX_CACHING="1"
VLLM_ENABLE_MFU_METRICS="${VLLM_ENABLE_MFU_METRICS:-1}"
VLLM_REQUIRED_VERSION="${VLLM_REQUIRED_VERSION:-0.28.0}"
FLASHINFER_REQUIRED_VERSION="${FLASHINFER_REQUIRED_VERSION:-0.6.16.post3}"

# cuda.py is a scheduler keeper, not productive inference.  It waits until
# the vLLM allocation is present, keeps at least 18 GiB free, and may reserve
# at most 1 GiB for its matrix-multiply burner.
CUDA_KEEPER_ENABLE="1"
ENABLE_CUDA_KEEPER="${CUDA_KEEPER_ENABLE}"
CUDA_KEEPER_SCRIPT="${PROJECT_ROOT}/hpc/shared/cuda.py"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="0"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-18.0}"
CUDA_KEEPER_MAX_PREALLOC_GB="${CUDA_KEEPER_MAX_PREALLOC_GB:-1.0}"
CUDA_KEEPER_START_AFTER_SECONDS="7200"

SOURCE_PREPROCESSING_ROOT="${SOURCE_PREPROCESSING_ROOT:-${PACKAGE_ROOT}/outputs/six_user_qa_10min_fps_resolution_ablation/six_user_qa_10min_vllm_fps_resolution_3x10_17109425/shared_preprocessing}"
EXPECTED_SOURCE_CANDIDATE_SHA256="${EXPECTED_SOURCE_CANDIDATE_SHA256:-7e2d75fdeb3399e3bd614e1af07508bbc93ea75a7d88aa0997634c9940184f49}"
OUTPUT_BASE="${OUTPUT_BASE:-${PACKAGE_ROOT}/outputs/six_user_qa_1h200_throughput_probe}"
EXPERIMENT_ROOT="${OUTPUT_BASE}/${EXPERIMENT_NAME}_${SLURM_JOB_ID}"
METRICS_DIR="${EXPERIMENT_ROOT}/metrics"
RUN_MODE="${EXPERIMENT_NAME}_${PROFILE_LABEL}"
RUN_OUTPUT_DIR="${EXPERIMENT_ROOT}/runs/${RUN_MODE}_${SLURM_JOB_ID}"

# All caches, compilation output, and IPC paths are job-specific scratch.
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
export FORCE_QWENVL_VIDEO_READER=decord
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PACKAGE_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
if [[ -z "${HF_TOKEN:-}" && -s "${LOGIN_HF_TOKEN_PATH}" ]]; then
  export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"
fi

mkdir -p \
  "${HPC_ROOT}/logs" \
  "${EXPERIMENT_ROOT}/runs" \
  "${METRICS_DIR}" \
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

stage() {
  printf 'probe_stage=%s status=started timestamp=%s job_id=%s host=%s\n' \
    "$1" "$(date --iso-8601=seconds)" "${SLURM_JOB_ID}" "$(hostname)"
}

cd "${PROJECT_ROOT}"
test -x "${PYTHON}"
test -s "${RUNTIME_SCRIPT}"
test -s "${PACKAGE_ROOT}/qwen3vl_runner.py"
test -s "${PACKAGE_ROOT}/video_qa_loop.py"
test -s "${CUDA_KEEPER_SCRIPT}"
CUDA_KEEPER_SHA256="$(sha256sum "${CUDA_KEEPER_SCRIPT}" | awk '{print $1}')"
printf '%s  %s\n' "${CUDA_KEEPER_SHA256}" "${CUDA_KEEPER_SCRIPT}" > "${METRICS_DIR}/cuda_keeper.sha256"
grep -q -- '--enable-chunked-prefill' "${RUNTIME_SCRIPT}"
grep -q -- '--api-server-count' "${RUNTIME_SCRIPT}"
grep -q -- '--renderer-num-workers' "${RUNTIME_SCRIPT}"
grep -q -- '--mm-processor-kwargs' "${RUNTIME_SCRIPT}"
grep -q -- '--enable-mfu-metrics' "${RUNTIME_SCRIPT}"
grep -q -- '--max-review-lanes' "${RUNTIME_SCRIPT}"
grep -Fq -- '--reasoning-mode "${REASONING_MODE}"' "${RUNTIME_SCRIPT}"
grep -Fq -- '--start-after-seconds "${CUDA_KEEPER_START_AFTER_SECONDS}"' "${RUNTIME_SCRIPT}"
if [[ ! "${MAX_PACKETS_IN_FLIGHT}" =~ ^[34]$ || \
      ! "${MAX_REVIEW_LANES}" =~ ^[23]$ || \
      "${MAX_REVIEW_LANES}" -ge "${MAX_PACKETS_IN_FLIGHT}" ]]; then
  echo "error=invalid_limit_test_width packets=${MAX_PACKETS_IN_FLIGHT} review_lanes=${MAX_REVIEW_LANES}" >&2
  exit 2
fi
if [[ "${VLLM_RENDERER_NUM_WORKERS}" -gt "1" && \
      "${VLLM_MM_PROCESSOR_CACHE_GB}" != "0" ]]; then
  echo "error=vllm_multi_renderer_requires_mm_processor_cache_0" >&2
  exit 2
fi

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
from importlib.metadata import version

import decord
import torch
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
    raise SystemExit(f"expected vLLM {required_vllm}; detected {detected_vllm}")
required_flashinfer = Version(os.environ["FLASHINFER_REQUIRED_VERSION"])
for package in ("flashinfer-python", "flashinfer-cubin"):
    detected = Version(version(package))
    if detected != required_flashinfer:
        raise SystemExit(f"expected {package} {required_flashinfer}; detected {detected}")
backend = get_video_reader_backend()
if backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {backend}")
usable_fa = [candidate for candidate in (2, 3) if is_fa_version_supported(candidate)]
if not usable_fa:
    raise SystemExit(
        "no usable compiled FlashAttention extension: "
        f"fa2_available={FA2_AVAILABLE} fa3_available={FA3_AVAILABLE}"
    )
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit(
        f"probe requires exactly one visible GPU; visible={torch.cuda.device_count()}"
    )
properties = torch.cuda.get_device_properties(0)
if "H200" not in properties.name:
    raise SystemExit(f"probe requires H200; detected={properties.name}")
print(
    "probe_runtime_preflight_passed",
    f"vllm={detected_vllm}",
    f"flashinfer={required_flashinfer}",
    f"flash_attention_versions={usable_fa}",
    f"video_backend={backend}",
    f"decord={decord.__version__}",
    f"gpu={properties.name}",
    f"gpu_memory_gib={properties.total_memory / 1024**3:.3f}",
)
PY

stage "validate_exact_ten_packet_cohort"
SOURCE_PREPROCESSING_ROOT="${SOURCE_PREPROCESSING_ROOT%/}"
SOURCE_MANIFEST="${SOURCE_PREPROCESSING_ROOT}/manifest.json"
SOURCE_SUMMARY="${SOURCE_PREPROCESSING_ROOT}/ten_minute_setup/six_user_10min_setup_summary.json"
SOURCE_CANDIDATES="${SOURCE_PREPROCESSING_ROOT}/ten_minute_setup/time_aware_candidates/six_user_10min_time_aware.jsonl"
test -s "${SOURCE_MANIFEST}"
test -s "${SOURCE_SUMMARY}"
test -s "${SOURCE_CANDIDATES}"
SOURCE_SHA256="$(sha256sum "${SOURCE_CANDIDATES}" | awk '{print $1}')"
if [[ "${SOURCE_SHA256}" != "${EXPECTED_SOURCE_CANDIDATE_SHA256}" ]]; then
  echo "error=source_candidate_sha256_mismatch expected=${EXPECTED_SOURCE_CANDIDATE_SHA256} actual=${SOURCE_SHA256}" >&2
  exit 1
fi
export SOURCE_CANDIDATES METRICS_DIR TARGET_COUNT
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

source = Path(os.environ["SOURCE_CANDIDATES"])
rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
target = int(os.environ["TARGET_COUNT"])
if len(rows) != target:
    raise SystemExit(f"probe requires exactly {target} candidates; found {len(rows)}")
evidence_ids = [str(row.get("evidence_id") or "") for row in rows]
if any(not value for value in evidence_ids) or len(set(evidence_ids)) != target:
    raise SystemExit("probe candidates need ten unique, non-empty evidence IDs")
groups = [str(row.get("generation_group_id") or row.get("evidence_id") or "") for row in rows]
if any(not value for value in groups) or len(set(groups)) != target:
    raise SystemExit("three-packet concurrency requires ten unique generation groups")

def source_videos(row):
    clips = row.get("clips") or []
    if len(clips) != 6:
        raise SystemExit(f"{row.get('evidence_id')} does not contain six clips")
    videos = [
        str(clip.get("full_local_video") or clip.get("original_local_video") or "")
        for clip in clips
    ]
    if any(not value for value in videos):
        raise SystemExit(f"{row.get('evidence_id')} has a missing source video")
    unique_videos = sorted(set(videos))
    if len(unique_videos) != 6:
        raise SystemExit(
            f"{row.get('evidence_id')} needs six distinct source videos; "
            f"found {len(unique_videos)}"
        )
    return unique_videos

video_sets = [source_videos(row) for row in rows]
probe = {
    "source_candidate_jsonl": str(source),
    "evidence_ids": evidence_ids,
    "generation_group_ids": groups,
    "distinct_source_video_sets": len({tuple(videos) for videos in video_sets}),
    "source_videos_by_evidence": dict(zip(evidence_ids, video_sets, strict=True)),
    "generator_frame_counts": [
        sum(len(clip.get("frames") or []) for clip in row.get("clips") or [])
        for row in rows
    ],
}
Path(os.environ["METRICS_DIR"], "probe_inputs.json").write_text(
    json.dumps(probe, indent=2) + "\n", encoding="utf-8"
)
print(
    "probe_inputs_validated",
    f"evidence_ids={','.join(evidence_ids)}",
    f"unique_generation_groups={len(set(groups))}",
    f"distinct_source_video_sets={probe['distinct_source_video_sets']}",
    f"generator_frame_counts={probe['generator_frame_counts']}",
)
PY
printf '%s  %s\n' "${SOURCE_SHA256}" "${SOURCE_CANDIDATES}" > "${METRICS_DIR}/source_candidates.sha256"

export EXPERIMENT_NAME REASONING_MODE EXPERIMENT_ROOT PROFILE_LABEL JUDGE_VIDEO_FPS
export GENERATOR_MAX_IMAGE_PIXELS VLLM_MIN_IMAGE_PIXELS TARGET_COUNT
export MAX_ATTEMPTS MAX_PACKETS_IN_FLIGHT MAX_REVIEW_LANES
export SOURCE_PREPROCESSING_ROOT EXPECTED_SOURCE_CANDIDATE_SHA256
export VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
export VLLM_MEDIA_LOADING_THREAD_COUNT VLLM_SERVER_OMP_NUM_THREADS
export VLLM_MM_PROCESSOR_CACHE_GB VLLM_GPU_MEMORY_UTILIZATION
export VLLM_MAX_NUM_SEQS VLLM_MAX_NUM_BATCHED_TOKENS
export VLLM_ENABLE_PREFIX_CACHING VLLM_ENABLE_MFU_METRICS
export CUDA_KEEPER_ENABLE ENABLE_CUDA_KEEPER CUDA_KEEPER_SCRIPT CUDA_KEEPER_THRESHOLD
export CUDA_KEEPER_GPUS CUDA_KEEPER_RESERVE CUDA_KEEPER_MAX_PREALLOC_GB
export CUDA_KEEPER_START_AFTER_SECONDS JOB_START_EPOCH_SECONDS CUDA_KEEPER_SHA256
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "reasoning_mode": os.environ["REASONING_MODE"],
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "purpose": "single-H200 ten-packet three-attempt concurrency limit test",
    "profile": {
        "label": os.environ["PROFILE_LABEL"],
        "judge_video_fps": float(os.environ["JUDGE_VIDEO_FPS"]),
        "min_pixels_per_frame": int(os.environ["VLLM_MIN_IMAGE_PIXELS"]),
        "max_pixels_per_frame": int(os.environ["GENERATOR_MAX_IMAGE_PIXELS"]),
    },
    "workload": {
        "target_count": int(os.environ["TARGET_COUNT"]),
        "max_attempts": int(os.environ["MAX_ATTEMPTS"]),
        "max_packets_in_flight": int(os.environ["MAX_PACKETS_IN_FLIGHT"]),
        "generation_lanes": 1,
        "max_review_lanes": int(os.environ["MAX_REVIEW_LANES"]),
        "evidence_policy": "reuse_job_17109425_exact_ten_candidate_jsonl",
        "source_preprocessing_root": os.environ["SOURCE_PREPROCESSING_ROOT"],
        "source_candidate_sha256": os.environ["EXPECTED_SOURCE_CANDIDATE_SHA256"],
    },
    "vllm": {
        "version": os.environ["VLLM_REQUIRED_VERSION"],
        "flashinfer_version": os.environ["FLASHINFER_REQUIRED_VERSION"],
        "api_server_count": int(os.environ["VLLM_SERVER_API_COUNT"]),
        "renderer_num_workers": int(os.environ["VLLM_RENDERER_NUM_WORKERS"]),
        "media_loading_thread_count": int(os.environ["VLLM_MEDIA_LOADING_THREAD_COUNT"]),
        "server_omp_num_threads": int(os.environ["VLLM_SERVER_OMP_NUM_THREADS"]),
        "mm_processor_cache_gb_per_process": float(os.environ["VLLM_MM_PROCESSOR_CACHE_GB"]),
        "gpu_memory_utilization": float(os.environ["VLLM_GPU_MEMORY_UTILIZATION"]),
        "max_num_seqs": int(os.environ["VLLM_MAX_NUM_SEQS"]),
        "max_num_batched_tokens": int(os.environ["VLLM_MAX_NUM_BATCHED_TOKENS"]),
        "chunked_prefill": True,
        "prefix_caching": os.environ["VLLM_ENABLE_PREFIX_CACHING"] == "1",
        "mfu_metrics": os.environ["VLLM_ENABLE_MFU_METRICS"] == "1",
        "attention_backend": "FLASH_ATTN",
        "gdn_prefill_backend": "flashinfer",
    },
    "cuda_keeper": {
        "enabled": os.environ["ENABLE_CUDA_KEEPER"] == "1",
        "script": os.environ["CUDA_KEEPER_SCRIPT"],
        "sha256": os.environ["CUDA_KEEPER_SHA256"],
        "threshold_percent": int(os.environ["CUDA_KEEPER_THRESHOLD"]),
        "gpus": os.environ["CUDA_KEEPER_GPUS"],
        "reserve_gib": float(os.environ["CUDA_KEEPER_RESERVE"]),
        "max_prealloc_gib": float(os.environ["CUDA_KEEPER_MAX_PREALLOC_GB"]),
        "start_after_seconds": int(os.environ["CUDA_KEEPER_START_AFTER_SECONDS"]),
        "interpretation": "scheduler keepalive only; exclude burner work from useful throughput",
    },
}
Path(os.environ["EXPERIMENT_ROOT"], "probe_manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY

MONITOR_PIDS=()

monitor_gpu() {
  set +e
  local destination="${METRICS_DIR}/gpu_1s.csv"
  printf '%s\n' 'sample_iso,sample_epoch,gpu_index,gpu_name,pstate,sm_util_percent,memory_util_percent,memory_used_mib,memory_free_mib,power_watts,power_limit_watts,sm_clock_mhz,memory_clock_mhz,temperature_c,pcie_generation,pcie_width' > "${destination}"
  while true; do
    local sample_iso sample_epoch values
    sample_iso="$(date --iso-8601=seconds)"
    sample_epoch="$(date +%s.%N)"
    values="$(nvidia-smi --query-gpu=index,name,pstate,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader,nounits 2>/dev/null)"
    if [[ -n "${values}" ]]; then
      printf '%s,%s,%s\n' "${sample_iso}" "${sample_epoch}" "${values}" >> "${destination}"
    else
      printf '# %s nvidia_smi_query_failed\n' "${sample_iso}" >> "${destination}"
    fi
    sleep 1
  done
}

monitor_gpu_processes() {
  set +e
  local destination="${METRICS_DIR}/gpu_processes_2s.log"
  while true; do
    printf '# timestamp=%s epoch=%s\n' "$(date --iso-8601=seconds)" "$(date +%s.%N)" >> "${destination}"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits >> "${destination}" 2>&1
    sleep 2
  done
}

monitor_cgroup_memory() {
  set +e
  local destination="${METRICS_DIR}/cgroup_memory_2s.tsv"
  local relative_path cgroup_dir
  relative_path="$(awk -F: '$1 == "0" {print $3; exit}' /proc/self/cgroup)"
  cgroup_dir="/sys/fs/cgroup${relative_path}"
  printf 'sample_iso\tsample_epoch\tmemory_current_bytes\tmemory_peak_bytes\tmemory_max\tswap_current_bytes\n' > "${destination}"
  while true; do
    local current peak maximum swap
    current="$(cat "${cgroup_dir}/memory.current" 2>/dev/null || printf 'unavailable')"
    peak="$(cat "${cgroup_dir}/memory.peak" 2>/dev/null || printf 'unavailable')"
    maximum="$(cat "${cgroup_dir}/memory.max" 2>/dev/null || printf 'unavailable')"
    swap="$(cat "${cgroup_dir}/memory.swap.current" 2>/dev/null || printf 'unavailable')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$(date --iso-8601=seconds)" "$(date +%s.%N)" \
      "${current}" "${peak}" "${maximum}" "${swap}" >> "${destination}"
    sleep 2
  done
}

monitor_processes() {
  set +e
  local destination="${METRICS_DIR}/top_processes_5s.log"
  while true; do
    printf '# timestamp=%s epoch=%s\n' "$(date --iso-8601=seconds)" "$(date +%s.%N)" >> "${destination}"
    ps -eo pid=,ppid=,rss=,vsz=,%cpu=,%mem=,etimes=,stat=,comm=,args= --sort=-rss | head -n 40 >> "${destination}"
    sleep 5
  done
}

monitor_vllm_metrics() {
  "${PYTHON}" -u - "http://127.0.0.1:18000/metrics" "${METRICS_DIR}/vllm_metrics_2s.promlog" <<'PY'
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

summarize_metrics() {
  export METRICS_DIR RUN_OUTPUT_DIR
  PROBE_EXIT_CODE="$1" "${PYTHON}" -P - <<'PY'
import csv
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path

root = Path(os.environ["METRICS_DIR"])
summary = {"probe_exit_code": int(os.environ["PROBE_EXIT_CODE"])}

gpu_rows = []
gpu_path = root / "gpu_1s.csv"
if gpu_path.exists():
    with gpu_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            try:
                gpu_rows.append({
                    "util": float(row["sm_util_percent"]),
                    "memory": float(row["memory_used_mib"]),
                    "power": float(row["power_watts"]),
                })
            except (KeyError, TypeError, ValueError):
                pass
if gpu_rows:
    util = sorted(row["util"] for row in gpu_rows)
    percentile = lambda fraction: util[min(len(util) - 1, math.ceil(fraction * len(util)) - 1)]
    summary["gpu"] = {
        "samples": len(gpu_rows),
        "sm_util_mean_percent": round(statistics.fmean(util), 3),
        "sm_util_p50_percent": percentile(0.50),
        "sm_util_p95_percent": percentile(0.95),
        "sm_util_max_percent": max(util),
        "fraction_samples_below_10_percent": round(sum(v < 10 for v in util) / len(util), 4),
        "fraction_samples_at_least_80_percent": round(sum(v >= 80 for v in util) / len(util), 4),
        "max_memory_used_mib": max(row["memory"] for row in gpu_rows),
        "mean_power_watts": round(statistics.fmean(row["power"] for row in gpu_rows), 3),
        "max_power_watts": max(row["power"] for row in gpu_rows),
    }

cgroup_path = root / "cgroup_memory_2s.tsv"
peaks = []
if cgroup_path.exists():
    with cgroup_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                peaks.append(int(row["memory_peak_bytes"]))
            except (KeyError, TypeError, ValueError):
                pass
if peaks:
    summary["host_cgroup"] = {
        "memory_peak_bytes": max(peaks),
        "memory_peak_gib": round(max(peaks) / 1024**3, 3),
    }

metric_samples = defaultdict(lambda: defaultdict(list))
metric_path = root / "vllm_metrics_2s.promlog"
metric_pattern = re.compile(r"^([^\s{]+)(?:\{[^}]*\})?\s+([-+0-9.eE]+)$")
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

vllm_summary = {}
for name in (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
):
    values = aggregates(name)
    if values:
        vllm_summary[name] = {
            "samples": len(values),
            "mean": round(statistics.fmean(values), 6),
            "max": max(values),
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
        vllm_summary[base] = {
            "total_seconds": round(sums[-1], 6),
            "request_count": counts[-1],
            "mean_seconds": round(sums[-1] / counts[-1], 6),
        }
for name in (
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:num_preemptions_total",
    "vllm:mm_cache_queries_total",
    "vllm:mm_cache_hits_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:estimated_flops_per_gpu_total",
    "vllm:estimated_read_bytes_per_gpu_total",
    "vllm:estimated_write_bytes_per_gpu_total",
):
    values = aggregates(name)
    if values:
        vllm_summary[name] = values[-1]
if vllm_summary:
    summary["vllm"] = vllm_summary

keeper_path = Path(os.environ["RUN_OUTPUT_DIR"]) / f"cuda_keeper_{os.environ['SLURM_JOB_ID']}.log"
if keeper_path.exists():
    keeper_lines = [
        line
        for line in keeper_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if "burning:" in line
    ]
    burning_lines = [line for line in keeper_lines if "burning:Y" in line]
    main_utils = []
    for line in keeper_lines:
        match = re.search(r"main:\s*([0-9]+)%", line)
        if match:
            main_utils.append(int(match.group(1)))
    summary["cuda_keeper"] = {
        "log": str(keeper_path),
        "controller_samples": len(keeper_lines),
        "burning_samples": len(burning_lines),
        "burning_fraction": (
            round(len(burning_lines) / len(keeper_lines), 4) if keeper_lines else None
        ),
        "mean_reported_main_process_util_percent": (
            round(statistics.fmean(main_utils), 3) if main_utils else None
        ),
        "warning": "keeper utilization is scheduler keepalive work, not QA throughput",
    }

(root / "metrics_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print("probe_metrics_summary", json.dumps(summary, sort_keys=True))
PY
}

finalize_probe() {
  local exit_code=$?
  trap - EXIT INT TERM
  set +e
  for pid in "${MONITOR_PIDS[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
  for pid in "${MONITOR_PIDS[@]}"; do
    wait "${pid}" 2>/dev/null || true
  done
  summarize_metrics "${exit_code}" || true
  printf 'probe_status=%s timestamp=%s exit_code=%s experiment_root=%s run_output_dir=%s metrics_dir=%s\n' \
    "$([[ "${exit_code}" == "0" ]] && printf completed || printf failed)" \
    "$(date --iso-8601=seconds)" "${exit_code}" "${EXPERIMENT_ROOT}" \
    "${RUN_OUTPUT_DIR}" "${METRICS_DIR}"
  exit "${exit_code}"
}

trap finalize_probe EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

stage "start_resource_monitors"
monitor_gpu & MONITOR_PIDS+=("$!")
monitor_gpu_processes & MONITOR_PIDS+=("$!")
monitor_cgroup_memory & MONITOR_PIDS+=("$!")
monitor_processes & MONITOR_PIDS+=("$!")
if command -v vmstat >/dev/null 2>&1; then
  vmstat -t 1 > "${METRICS_DIR}/vmstat_1s.log" 2>&1 &
  MONITOR_PIDS+=("$!")
fi
monitor_vllm_metrics & MONITOR_PIDS+=("$!")

stage "run_ten_packet_limit_test"
PROBE_START_EPOCH="$(date +%s)"
printf 'probe_config timestamp=%s profile=%s fps=%s min_pixels=%s max_pixels=%s packets=%s attempts=%s packets_in_flight=%s generation_lanes=1 review_lanes=%s api_servers=%s renderers=%s media_threads=%s gpu_memory_utilization=%s max_num_seqs=%s max_num_batched_tokens=%s prefix_caching=%s mfu_metrics=%s cuda_keeper=%s keeper_threshold=%s\n' \
  "$(date --iso-8601=seconds)" "${PROFILE_LABEL}" "${JUDGE_VIDEO_FPS}" \
  "${VLLM_MIN_IMAGE_PIXELS}" "${GENERATOR_MAX_IMAGE_PIXELS}" \
  "${TARGET_COUNT}" "${MAX_ATTEMPTS}" "${MAX_PACKETS_IN_FLIGHT}" \
  "${MAX_REVIEW_LANES}" \
  "${VLLM_SERVER_API_COUNT}" "${VLLM_RENDERER_NUM_WORKERS}" \
  "${VLLM_MEDIA_LOADING_THREAD_COUNT}" "${VLLM_GPU_MEMORY_UTILIZATION}" \
  "${VLLM_MAX_NUM_SEQS}" "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
  "${VLLM_ENABLE_PREFIX_CACHING}" "${VLLM_ENABLE_MFU_METRICS}" \
  "${ENABLE_CUDA_KEEPER}" "${CUDA_KEEPER_THRESHOLD}"

set +e
(
  export RUN_MODE
  export REASONING_MODE
  export OUTPUT_BASE="${EXPERIMENT_ROOT}/runs"
  export TARGET_COUNT MAX_ATTEMPTS SOURCE_WINDOW_COUNT
  export MAX_PACKETS_IN_FLIGHT MAX_REVIEW_LANES
  export EVIDENCE_RANDOM_SEED REQUIRED_GPU_COUNT
  export VLLM_SAMPLING_SEED
  export QA_PACKET_LIMIT=""
  export REUSE_PREPROCESSED_FROM="${SOURCE_PREPROCESSING_ROOT}"
  export REQUIRE_FRESH_EVIDENCE="0"
  export RESUME_QA_FROM=""
  export COHORT_SKIP_SOURCE=""
  export SKIP_EVIDENCE_IDS=""
  export INFERENCE_BACKEND="openai-compatible-local"
  export START_VLLM_SERVER="1"
  export VLLM_SERVER_HOST="127.0.0.1"
  export VLLM_SERVER_PORT="18000"
  export VLLM_SERVER_API_COUNT VLLM_RENDERER_NUM_WORKERS
  export VLLM_MEDIA_LOADING_THREAD_COUNT VLLM_SERVER_OMP_NUM_THREADS
  export VLLM_MM_PROCESSOR_CACHE_GB VLLM_ENABLE_PREFIX_CACHING
  export VLLM_ENABLE_MFU_METRICS
  export VLLM_OPENAI_USE_LOCAL_MEDIA_URIS="1"
  export VLLM_REQUIRED_VERSION FLASHINFER_REQUIRED_VERSION
  export VLLM_TENSOR_PARALLEL_SIZE="1"
  export VLLM_GPU_MEMORY_UTILIZATION
  export VLLM_MAX_MODEL_LEN="262144"
  export VLLM_MAX_NUM_SEQS
  export VLLM_BATCH_WAIT_MS="20"
  export VLLM_MAX_NUM_BATCHED_TOKENS
  export VLLM_MAX_IMAGES="3600"
  export VLLM_MAX_VIDEOS="20"
  export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
  export VLLM_MM_ENCODER_ATTN_BACKEND="FLASH_ATTN"
  export VLLM_GDN_PREFILL_BACKEND="flashinfer"
  export VLLM_MM_ENCODER_TP_MODE="data"
  export VLLM_MIN_IMAGE_PIXELS
  export VLLM_VIDEO_FPS="${JUDGE_VIDEO_FPS}"
  export JUDGE_VIDEO_FPS
  export GENERATOR_MAX_IMAGE_PIXELS
  export QWEN_MEMORY_SAFE_VIDEO_FPS="${JUDGE_VIDEO_FPS}"
  export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="${VLLM_MIN_IMAGE_PIXELS}"
  export QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="${GENERATOR_MAX_IMAGE_PIXELS}"
  export CUDA_KEEPER_ENABLE ENABLE_CUDA_KEEPER CUDA_KEEPER_SCRIPT CUDA_KEEPER_THRESHOLD
  export CUDA_KEEPER_GPUS CUDA_KEEPER_RESERVE CUDA_KEEPER_MAX_PREALLOC_GB
  export CUDA_KEEPER_START_AFTER_SECONDS JOB_START_EPOCH_SECONDS
  export JOB_SCRATCH_ROOT VLLM_IPC_RUNTIME_ROOT
  export PERSISTENT_RUNTIME_CACHE_ROOT PERSISTENT_MODEL_CACHE_ROOT
  export PERSISTENT_KERNEL_CACHE_ROOT
  bash "${RUNTIME_SCRIPT}"
) > >(
  awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
    | tee "${METRICS_DIR}/runtime_stdout_timeline.log"
) 2> >(
  awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush() }' \
    | tee "${METRICS_DIR}/runtime_stderr_timeline.log" >&2
)
RUNTIME_EXIT_CODE=$?
set -e

PROBE_END_EPOCH="$(date +%s)"
printf 'probe_runtime_finished timestamp=%s exit_code=%s elapsed_seconds=%s output_dir=%s\n' \
  "$(date --iso-8601=seconds)" "${RUNTIME_EXIT_CODE}" \
  "$((PROBE_END_EPOCH - PROBE_START_EPOCH))" "${RUN_OUTPUT_DIR}"
exit "${RUNTIME_EXIT_CODE}"
