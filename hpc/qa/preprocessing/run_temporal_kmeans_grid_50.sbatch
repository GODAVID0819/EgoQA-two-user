#!/usr/bin/env bash
#SBATCH --job-name=egolife_tkmeans50
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
EXPERIMENT_MODE="${EGOLIFE2U_PREPROCESS_EXPERIMENT_MODE:-temporal_kmeans}"
if [[ "${EXPERIMENT_MODE}" == "cross_user_gate" ]]; then
  OUTPUT_BASE="${EGOLIFE2U_OUTPUT_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/cross_user_temporal_gate_grid_50}"
  JOB_SCRATCH_PREFIX="egolife_cross_user_gate"
elif [[ "${EXPERIMENT_MODE}" == "temporal_kmeans" ]]; then
  OUTPUT_BASE="${EGOLIFE2U_OUTPUT_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/temporal_kmeans_grid_50}"
  JOB_SCRATCH_PREFIX="egolife_temporal_kmeans"
else
  echo "error=unknown_experiment_mode value=${EXPERIMENT_MODE}" >&2
  exit 2
fi
OUTPUT_DIR="${OUTPUT_BASE}/job_${SLURM_JOB_ID}"
JOB_SCRATCH_ROOT="/scratch/${USER}/${JOB_SCRATCH_PREFIX}_${SLURM_JOB_ID}"
VIDEO_CACHE_DIR="${JOB_SCRATCH_ROOT}/video_cache"

PAIR_COUNT="${TEMPORAL_KMEANS_PAIR_COUNT:-50}"
MAX_GROUPS="${TEMPORAL_KMEANS_MAX_GROUPS:-1000}"
DURATIONS="${TEMPORAL_KMEANS_DURATIONS:-30,180,360,600}"
TIME_WEIGHTS="${TEMPORAL_KMEANS_TIME_WEIGHTS:-0,0.1,0.25,0.5,1,2,4}"
SECONDS_PER_CLUSTER="${TEMPORAL_KMEANS_SECONDS_PER_CLUSTER:-2.5}"
TEMPORAL_UNIT_SECONDS="${TEMPORAL_KMEANS_UNIT_SECONDS:-30}"
SIMILARITY_THRESHOLDS="${TEMPORAL_KMEANS_THRESHOLDS:-0.82}"
SAMPLE_INTERVAL_SECONDS="${TEMPORAL_KMEANS_SAMPLE_INTERVAL_SECONDS:-1}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
CLIP_BATCH_SIZE="${TEMPORAL_KMEANS_CLIP_BATCH_SIZE:-32}"
MAX_ITERATIONS="${TEMPORAL_KMEANS_MAX_ITERATIONS:-25}"
RANDOM_SEED="${TEMPORAL_KMEANS_RANDOM_SEED:-20260825}"
TRACE_PAIR_LIMIT="${TEMPORAL_KMEANS_TRACE_PAIR_LIMIT:-3}"
RESUME_PAIRS_DIR="${TEMPORAL_KMEANS_RESUME_PAIRS_DIR:-}"
RESUME_ARGS=()

CROSS_GATE_SOURCE_EXPERIMENT_DIR="${CROSS_GATE_SOURCE_EXPERIMENT_DIR:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/temporal_kmeans_grid_50/job_16370116/experiment}"
CROSS_GATE_RESUME_EXPERIMENT_DIR="${CROSS_GATE_RESUME_EXPERIMENT_DIR:-}"
CROSS_GATE_WITHIN_TIME_WEIGHT="${CROSS_GATE_WITHIN_TIME_WEIGHT:-0.1}"
CROSS_GATE_MODES="${CROSS_GATE_MODES:-center,interval}"
CROSS_GATE_GAPS="${CROSS_GATE_GAPS:-0,5,10,15,30,60,120}"
CROSS_GATE_RESUME_ARGS=()

CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/shared/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-1}"
CUDA_KEEPER_PID=""
CUDA_KEEPER_LOG="${OUTPUT_DIR}/cuda_keeper_${SLURM_JOB_ID}.log"

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    echo "stage=stop_cuda_keeper pid=${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTPUT_DIR}" "${JOB_SCRATCH_ROOT}" "${VIDEO_CACHE_DIR}"
source "${PROJECT_ROOT}/hpc/shared/env_qwen3vl.sh"
PYTHON="$(command -v python)"

# Mandatory job-specific storage boundary. Keep model/runtime caches and
# temporary files inside this job's scratch directory.
export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf_home"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/hf_datasets"
export TRANSFORMERS_CACHE="${JOB_SCRATCH_ROOT}/transformers"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch_home"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${TMPDIR}"
export TEMP="${TMPDIR}"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${HF_DATASETS_CACHE}" \
  "${TRANSFORMERS_CACHE}" "${MODELSCOPE_CACHE}" "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${TMPDIR}"

FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
FFMPEG_BINARY=""
FFPROBE_BINARY=""
FFMPEG_SOURCE=""

# Prefer the dedicated runtime when it exists, but do not require that optional
# environment. The activated Qwen Conda environment or cluster PATH may already
# provide a complete ffmpeg/ffprobe pair.
if [[ -x "${FFMPEG_ENV}/bin/ffmpeg" && -x "${FFMPEG_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${FFMPEG_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${FFMPEG_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="ffmpeg_env"
elif [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/ffmpeg" && -x "${CONDA_PREFIX}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${CONDA_PREFIX}/bin/ffmpeg"
  FFPROBE_BINARY="${CONDA_PREFIX}/bin/ffprobe"
  FFMPEG_SOURCE="active_conda_env"
else
  FFMPEG_BINARY="$(command -v ffmpeg || true)"
  FFPROBE_BINARY="$(command -v ffprobe || true)"
  FFMPEG_SOURCE="activated_path"
fi

if [[ -z "${FFMPEG_BINARY}" || ! -x "${FFMPEG_BINARY}" || -z "${FFPROBE_BINARY}" || ! -x "${FFPROBE_BINARY}" ]]; then
  echo "error=ffmpeg_runtime_not_found" >&2
  echo "checked_ffmpeg_env=${FFMPEG_ENV}" >&2
  echo "checked_conda_prefix=${CONDA_PREFIX:-<unset>}" >&2
  echo "resolved_ffmpeg=${FFMPEG_BINARY:-<missing>}" >&2
  echo "resolved_ffprobe=${FFPROBE_BINARY:-<missing>}" >&2
  echo "hint=install ffmpeg and ffprobe in the activated Conda environment, or export FFMPEG_ENV to a valid runtime prefix" >&2
  exit 1
fi

FFMPEG_BIN_DIR="$(dirname "${FFMPEG_BINARY}")"
FFPROBE_BIN_DIR="$(dirname "${FFPROBE_BINARY}")"
FFMPEG_RUNTIME_PREFIX="$(dirname "${FFMPEG_BIN_DIR}")"
export FFMPEG_BINARY FFPROBE_BINARY FFMPEG_SOURCE
export PATH="${FFMPEG_BIN_DIR}:${FFPROBE_BIN_DIR}:${PATH}"
if [[ -d "${FFMPEG_RUNTIME_PREFIX}/lib" ]]; then
  export LD_LIBRARY_PATH="${FFMPEG_RUNTIME_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
fi

if [[ "${EXPERIMENT_MODE}" == "cross_user_gate" ]]; then
  if [[ ! -d "${CROSS_GATE_SOURCE_EXPERIMENT_DIR}" ]]; then
    echo "error=cross_gate_source_experiment_missing path=${CROSS_GATE_SOURCE_EXPERIMENT_DIR}" >&2
    exit 1
  fi
  if [[ ! -s "${CROSS_GATE_SOURCE_EXPERIMENT_DIR}/cohort.jsonl" || ! -s "${CROSS_GATE_SOURCE_EXPERIMENT_DIR}/summary.json" ]]; then
    echo "error=cross_gate_source_metadata_missing path=${CROSS_GATE_SOURCE_EXPERIMENT_DIR}" >&2
    exit 1
  fi
  RESUME_CACHE_COUNT="$(find "${CROSS_GATE_SOURCE_EXPERIMENT_DIR}/pairs" -mindepth 2 -maxdepth 2 -name embedding_cache.npz -type f | wc -l)"
  if [[ "${RESUME_CACHE_COUNT}" -lt "${PAIR_COUNT}" ]]; then
    echo "error=cross_gate_source_embedding_caches_incomplete expected=${PAIR_COUNT} actual=${RESUME_CACHE_COUNT}" >&2
    exit 1
  fi
  if [[ -n "${CROSS_GATE_RESUME_EXPERIMENT_DIR}" ]]; then
    if [[ ! -s "${CROSS_GATE_RESUME_EXPERIMENT_DIR}/cohort.jsonl" || ! -s "${CROSS_GATE_RESUME_EXPERIMENT_DIR}/grid_metrics.jsonl" ]]; then
      echo "error=cross_gate_resume_checkpoint_missing path=${CROSS_GATE_RESUME_EXPERIMENT_DIR}" >&2
      exit 1
    fi
    CROSS_GATE_RESUME_ARGS=(--resume-experiment-dir "${CROSS_GATE_RESUME_EXPERIMENT_DIR}")
  fi
else
  if [[ -n "${RESUME_PAIRS_DIR}" ]]; then
    if [[ ! -d "${RESUME_PAIRS_DIR}" ]]; then
      echo "error=resume_pairs_dir_missing path=${RESUME_PAIRS_DIR}" >&2
      exit 1
    fi
    RESUME_CACHE_COUNT="$(find "${RESUME_PAIRS_DIR}" -mindepth 2 -maxdepth 2 -name embedding_cache.npz -type f | wc -l)"
    if [[ "${RESUME_CACHE_COUNT}" -lt 1 ]]; then
      echo "error=resume_embedding_caches_missing path=${RESUME_PAIRS_DIR}" >&2
      exit 1
    fi
    RESUME_ARGS=(--resume-pairs-dir "${RESUME_PAIRS_DIR}")
  else
    RESUME_CACHE_COUNT=0
  fi
fi

echo "job_id=${SLURM_JOB_ID}"
echo "experiment_mode=${EXPERIMENT_MODE}"
echo "project_root=${PROJECT_ROOT}"
echo "output_dir=${OUTPUT_DIR}"
echo "job_scratch_root=${JOB_SCRATCH_ROOT}"
echo "pair_count=${PAIR_COUNT}"
echo "durations=${DURATIONS}"
echo "time_weights=${TIME_WEIGHTS}"
echo "seconds_per_cluster=${SECONDS_PER_CLUSTER}"
echo "temporal_unit_seconds=${TEMPORAL_UNIT_SECONDS}"
echo "similarity_thresholds=${SIMILARITY_THRESHOLDS}"
echo "clip_model_id=${CLIP_MODEL_ID}"
echo "random_seed=${RANDOM_SEED}"
echo "resume_pairs_dir=${RESUME_PAIRS_DIR:-<none>}"
echo "resume_cache_count=${RESUME_CACHE_COUNT}"
if [[ "${EXPERIMENT_MODE}" == "cross_user_gate" ]]; then
  echo "cross_gate_source_experiment_dir=${CROSS_GATE_SOURCE_EXPERIMENT_DIR}"
  echo "cross_gate_resume_experiment_dir=${CROSS_GATE_RESUME_EXPERIMENT_DIR:-<none>}"
  echo "cross_gate_within_time_weight=${CROSS_GATE_WITHIN_TIME_WEIGHT}"
  echo "cross_gate_modes=${CROSS_GATE_MODES}"
  echo "cross_gate_gaps=${CROSS_GATE_GAPS}"
fi
echo "ffmpeg_source=${FFMPEG_SOURCE}"
echo "ffmpeg_binary=${FFMPEG_BINARY}"
echo "ffprobe_binary=${FFPROBE_BINARY}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
echo "cuda_keeper_threshold=${CUDA_KEEPER_THRESHOLD}"
echo "cuda_keeper_gpus=${CUDA_KEEPER_GPUS}"
echo "cuda_keeper_reserve_gib=${CUDA_KEEPER_RESERVE}"
"${PYTHON}" --version
nvidia-smi

echo "stage=storage_preflight"
"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTPUT_DIR}/storage_preflight.json"

echo "stage=video_runtime_preflight"
"${FFMPEG_BINARY}" -version
"${FFPROBE_BINARY}" -version
"${PYTHON}" - <<'PY'
from torchcodec.decoders import VideoDecoder
print("video_decoder_backend=", VideoDecoder.__module__)
PY

if [[ ! -s "${CUDA_KEEPER_SCRIPT}" ]]; then
  echo "error=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}" >&2
  exit 1
fi

echo "stage=preflight_cuda_keeper_dependencies"
if ! "${PYTHON}" - <<'PY'
import importlib

failed = []
for name in ("pynvml", "psutil"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        failed.append(name)
        print(f"cuda_keeper_dependency_import_error name={name} error={exc!r}")
if failed:
    print("missing_or_broken_cuda_keeper_dependencies=" + ",".join(failed))
    raise SystemExit(1)
print("cuda_keeper_python_dependencies=ok")
PY
then
  if [[ "${CUDA_KEEPER_AUTO_INSTALL}" == "1" ]]; then
    echo "stage=install_cuda_keeper_dependencies"
    "${PYTHON}" -m pip install nvidia-ml-py3 psutil
  else
    echo "error=cuda_keeper_dependencies_missing auto_install=0" >&2
    exit 1
  fi
fi

"${PYTHON}" - <<'PY'
import os

import psutil
import pynvml
import torch

print("cuda_keeper_required_imports=ok")
print("torch_version=", torch.__version__)
print("cuda_visible_devices=", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("torch_cuda_device_count=", torch.cuda.device_count())
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("cuda keeper preflight failed: PyTorch cannot access the Slurm GPU")

try:
    pynvml.nvmlInit()
    print("nvml_device_count=", pynvml.nvmlDeviceGetCount())
finally:
    try:
        pynvml.nvmlShutdown()
    except Exception:
        pass
print("cuda_keeper_cuda_nvml_preflight=ok")
PY

echo "stage=start_cuda_keeper"
PYTHONUNBUFFERED=1 "${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}" \
  --threshold "${CUDA_KEEPER_THRESHOLD}" \
  --gpus "${CUDA_KEEPER_GPUS}" \
  --reserve "${CUDA_KEEPER_RESERVE}" \
  > "${CUDA_KEEPER_LOG}" 2>&1 &
CUDA_KEEPER_PID=$!
sleep 2
if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
  CUDA_KEEPER_STATUS=0
  if wait "${CUDA_KEEPER_PID}"; then
    CUDA_KEEPER_STATUS=0
  else
    CUDA_KEEPER_STATUS=$?
  fi
  echo "error=cuda_keeper_exited_early status=${CUDA_KEEPER_STATUS} log=${CUDA_KEEPER_LOG}" >&2
  if [[ -s "${CUDA_KEEPER_LOG}" ]]; then
    echo "cuda_keeper_log_begin" >&2
    sed -n '1,200p' "${CUDA_KEEPER_LOG}" >&2
    echo "cuda_keeper_log_end" >&2
  else
    echo "cuda_keeper_log_empty" >&2
  fi
  CUDA_KEEPER_PID=""
  exit 1
fi
echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"

echo "stage=write_run_manifest"
export TEMPORAL_KMEANS_OUTPUT_DIR="${OUTPUT_DIR}"
export TEMPORAL_KMEANS_EXPERIMENT_MODE="${EXPERIMENT_MODE}"
export TEMPORAL_KMEANS_PAIR_COUNT_EFFECTIVE="${PAIR_COUNT}"
export TEMPORAL_KMEANS_DURATIONS_EFFECTIVE="${DURATIONS}"
export TEMPORAL_KMEANS_WEIGHTS_EFFECTIVE="${TIME_WEIGHTS}"
export TEMPORAL_KMEANS_SPC_EFFECTIVE="${SECONDS_PER_CLUSTER}"
export TEMPORAL_KMEANS_THRESHOLDS_EFFECTIVE="${SIMILARITY_THRESHOLDS}"
export TEMPORAL_KMEANS_FFMPEG_BINARY="${FFMPEG_BINARY}"
export TEMPORAL_KMEANS_FFPROBE_BINARY="${FFPROBE_BINARY}"
export TEMPORAL_KMEANS_FFMPEG_SOURCE="${FFMPEG_SOURCE}"
export TEMPORAL_KMEANS_RESUME_PAIRS_EFFECTIVE="${RESUME_PAIRS_DIR}"
export TEMPORAL_KMEANS_RESUME_CACHE_COUNT_EFFECTIVE="${RESUME_CACHE_COUNT}"
export CROSS_GATE_SOURCE_EXPERIMENT_EFFECTIVE="${CROSS_GATE_SOURCE_EXPERIMENT_DIR}"
export CROSS_GATE_RESUME_EXPERIMENT_EFFECTIVE="${CROSS_GATE_RESUME_EXPERIMENT_DIR}"
export CROSS_GATE_WITHIN_TIME_WEIGHT_EFFECTIVE="${CROSS_GATE_WITHIN_TIME_WEIGHT}"
export CROSS_GATE_MODES_EFFECTIVE="${CROSS_GATE_MODES}"
export CROSS_GATE_GAPS_EFFECTIVE="${CROSS_GATE_GAPS}"
"${PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

output = Path(os.environ["TEMPORAL_KMEANS_OUTPUT_DIR"])
manifest = {
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "experiment_mode": os.environ["TEMPORAL_KMEANS_EXPERIMENT_MODE"],
    "output_dir": str(output),
    "pair_count": int(os.environ["TEMPORAL_KMEANS_PAIR_COUNT_EFFECTIVE"]),
    "durations_seconds": os.environ["TEMPORAL_KMEANS_DURATIONS_EFFECTIVE"],
    "time_weights": os.environ["TEMPORAL_KMEANS_WEIGHTS_EFFECTIVE"],
    "seconds_per_cluster": os.environ["TEMPORAL_KMEANS_SPC_EFFECTIVE"],
    "similarity_thresholds": os.environ["TEMPORAL_KMEANS_THRESHOLDS_EFFECTIVE"],
    "ffmpeg_binary": os.environ["TEMPORAL_KMEANS_FFMPEG_BINARY"],
    "ffprobe_binary": os.environ["TEMPORAL_KMEANS_FFPROBE_BINARY"],
    "ffmpeg_source": os.environ["TEMPORAL_KMEANS_FFMPEG_SOURCE"],
    "resume_pairs_dir": os.environ["TEMPORAL_KMEANS_RESUME_PAIRS_EFFECTIVE"] or None,
    "resume_cache_count": int(os.environ["TEMPORAL_KMEANS_RESUME_CACHE_COUNT_EFFECTIVE"]),
    "cross_gate_source_experiment_dir": os.environ["CROSS_GATE_SOURCE_EXPERIMENT_EFFECTIVE"],
    "cross_gate_resume_experiment_dir": os.environ["CROSS_GATE_RESUME_EXPERIMENT_EFFECTIVE"] or None,
    "cross_gate_within_time_weight": float(os.environ["CROSS_GATE_WITHIN_TIME_WEIGHT_EFFECTIVE"]),
    "cross_gate_modes": os.environ["CROSS_GATE_MODES_EFFECTIVE"],
    "cross_gate_gaps_seconds": os.environ["CROSS_GATE_GAPS_EFFECTIVE"],
}
(output / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

if [[ "${EXPERIMENT_MODE}" == "cross_user_gate" ]]; then
  echo "stage=run_cross_user_temporal_gate_grid"
  "${PYTHON}" -m egolife_two_user_qa.cross_user_temporal_gate_grid_sidecar \
    --source-experiment-dir "${CROSS_GATE_SOURCE_EXPERIMENT_DIR}" \
    --output-dir "${OUTPUT_DIR}/experiment" \
    --pair-count "${PAIR_COUNT}" \
    --durations-seconds "${DURATIONS}" \
    --seconds-per-cluster-values "${SECONDS_PER_CLUSTER}" \
    --similarity-thresholds "${SIMILARITY_THRESHOLDS}" \
    --within-time-weight "${CROSS_GATE_WITHIN_TIME_WEIGHT}" \
    --cross-gap-modes "${CROSS_GATE_MODES}" \
    --cross-gap-seconds "${CROSS_GATE_GAPS}" \
    --temporal-unit-seconds "${TEMPORAL_UNIT_SECONDS}" \
    --sample-interval-seconds "${SAMPLE_INTERVAL_SECONDS}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --pruning-protection-mode min_percent \
    --min-pruned-video-percent 20 \
    --trace-pair-limit "${TRACE_PAIR_LIMIT}" \
    "${CROSS_GATE_RESUME_ARGS[@]}"
else
  echo "stage=build_manifest"
  "${PYTHON}" -m egolife_two_user_qa build_manifest \
    --output "${OUTPUT_DIR}/manifest.json" \
    --no-overlays

  echo "stage=run_temporal_kmeans_grid"
  "${PYTHON}" -m egolife_two_user_qa.temporal_kmeans_grid_sidecar \
    --manifest "${OUTPUT_DIR}/manifest.json" \
    --output-dir "${OUTPUT_DIR}/experiment" \
    --cache-dir "${VIDEO_CACHE_DIR}" \
    --pair-count "${PAIR_COUNT}" \
    --max-groups "${MAX_GROUPS}" \
    --min-group-size 2 \
    --durations-seconds "${DURATIONS}" \
    --time-weights "${TIME_WEIGHTS}" \
    --seconds-per-cluster-values "${SECONDS_PER_CLUSTER}" \
    --temporal-unit-seconds "${TEMPORAL_UNIT_SECONDS}" \
    --similarity-thresholds "${SIMILARITY_THRESHOLDS}" \
    --sample-interval-seconds "${SAMPLE_INTERVAL_SECONDS}" \
    --model-id "${CLIP_MODEL_ID}" \
    --clip-batch-size "${CLIP_BATCH_SIZE}" \
    --max-iterations "${MAX_ITERATIONS}" \
    --pruning-protection-mode min_percent \
    --min-pruned-video-percent 20 \
    --random-seed "${RANDOM_SEED}" \
    --ffmpeg-binary "${FFMPEG_BINARY}" \
    --trace-pair-limit "${TRACE_PAIR_LIMIT}" \
    "${RESUME_ARGS[@]}" \
    --download-media
fi

echo "summary=${OUTPUT_DIR}/experiment/summary.json"
echo "cohort=${OUTPUT_DIR}/experiment/cohort.jsonl"
echo "metrics=${OUTPUT_DIR}/experiment/grid_metrics.csv"
echo "aggregates=${OUTPUT_DIR}/experiment/aggregate_metrics.csv"
echo "review=${OUTPUT_DIR}/experiment/summary.html"
echo "progress=${OUTPUT_DIR}/experiment/progress.json"
echo "storage_preflight=${OUTPUT_DIR}/storage_preflight.json"
echo "cuda_keeper_log=${CUDA_KEEPER_LOG}"
echo "date_done=$(date --iso-8601=seconds)"
