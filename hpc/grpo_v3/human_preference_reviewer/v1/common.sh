#!/usr/bin/env bash
set -euo pipefail

NETID=xl6775
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/human_preference_reviewer/v1}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/xl6775/envs/egoqa-ffmpeg-runtime}"
MODEL_DIR="${MODEL_DIR:-/scratch/xl6775/models/Qwen3-VL-8B-Instruct}"
DATA_DIR="${DATA_DIR:-${PROJECT_ROOT}/data_RLHF/reviewer_v1}"
CSV_PATH="${CSV_PATH:-${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv}"
MEDIA_MAP="${MEDIA_MAP:-${DATA_DIR}/media_map.json}"
MODE="${MODE:?MODE must be set by the sbatch entrypoint}"

JOB_SCRATCH_ROOT="/scratch/xl6775/job_scratch/reviewer_v1_${MODE}_${SLURM_JOB_ID}"
OUTDIR="${OUTPUT_ROOT}/${MODE}_${SLURM_JOB_ID}"
export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf"
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
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p "${OUTDIR}" "${JOB_SCRATCH_ROOT}" "${HOME}" "${XDG_CACHE_HOME}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${TMPDIR}"

export CONDA_ENV_NAME="${TRAIN_ENV}"
source "${PROJECT_ROOT}/hpc/shared/env_qwen3vl.sh"
PYTHON="${TRAIN_ENV}/bin/python"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"

test -s "${MODEL_DIR}/config.json"
test -s "${CSV_PATH}"
test -s "${MEDIA_MAP}"
test -x "${FFMPEG_ENV}/bin/ffmpeg"
test -x "${FFMPEG_ENV}/bin/ffprobe"
"${PYTHON}" -c 'from torchcodec.decoders import VideoDecoder; print("VideoDecoder", VideoDecoder.__module__)' \
  > "${OUTDIR}/video_decoder_environment.txt"
"${PYTHON}" -c 'import importlib.metadata as m; print({n:m.version(n) for n in ("torch","transformers","peft","accelerate")})' \
  > "${OUTDIR}/dependencies.txt"
nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv \
  > "${OUTDIR}/gpu_environment.csv"

export NETID PROJECT_ROOT OUTPUT_ROOT TRAIN_ENV FFMPEG_ENV MODEL_DIR DATA_DIR CSV_PATH MEDIA_MAP
export JOB_SCRATCH_ROOT OUTDIR PYTHON
