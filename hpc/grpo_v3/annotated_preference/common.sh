#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo}
TRAIN_ENV=${TRAIN_ENV:-/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024}
FFMPEG_ENV=${FFMPEG_ENV:-/scratch/xl6775/envs/egoqa-ffmpeg-runtime}
MODEL_DIR=${MODEL_DIR:-/scratch/xl6775/models/Qwen3-VL-8B-Instruct}
DATA_DIR=${DATA_DIR:-${PROJECT_ROOT}/data_RLHF/annotated_preference}
OUTPUT_ROOT=${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/annotated_preference}
CSV_PATH=${CSV_PATH:-${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv}
SPLIT_PATH=${SPLIT_PATH:-${DATA_DIR}/split_60_10.json}
MEDIA_MAP=${MEDIA_MAP:-${DATA_DIR}/media_map.json}
DPO_DATA_DIR=${DPO_DATA_DIR:-${DATA_DIR}/dpo}

export FPS=${FPS:-1}
export FPS_MIN_FRAMES=${FPS_MIN_FRAMES:-4}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-64}
export VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-50176}

: "${MODE:?MODE must be set by the sbatch entrypoint}"
: "${SLURM_JOB_ID:?this entrypoint must run under Slurm}"
JOB_SCRATCH_ROOT=/scratch/xl6775/job_scratch/annotated_preference_${MODE}_${SLURM_JOB_ID}
OUTDIR=${OUTDIR:-${OUTPUT_ROOT}/${MODE}_${SLURM_JOB_ID}}

export HOME=${JOB_SCRATCH_ROOT}/home
export XDG_CACHE_HOME=${JOB_SCRATCH_ROOT}/xdg_cache
export HF_HOME=${JOB_SCRATCH_ROOT}/hf
export HF_DATASETS_CACHE=${JOB_SCRATCH_ROOT}/hf_datasets
export MODELSCOPE_CACHE=${JOB_SCRATCH_ROOT}/modelscope
export TORCH_HOME=${JOB_SCRATCH_ROOT}/torch
export TRITON_CACHE_DIR=${JOB_SCRATCH_ROOT}/triton
export TORCHINDUCTOR_CACHE_DIR=${JOB_SCRATCH_ROOT}/torchinductor
export VLLM_CACHE_ROOT=${JOB_SCRATCH_ROOT}/vllm
export CUDA_CACHE_PATH=${JOB_SCRATCH_ROOT}/cuda
export FLASHINFER_WORKSPACE_BASE=${JOB_SCRATCH_ROOT}/flashinfer
export TMPDIR=${JOB_SCRATCH_ROOT}/tmp
export TMP=${JOB_SCRATCH_ROOT}/tmp
export TEMP=${JOB_SCRATCH_ROOT}/tmp
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
export PATH=${FFMPEG_ENV}/bin:${PATH}
export LD_LIBRARY_PATH=${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}

mkdir -p "${OUTDIR}" "${JOB_SCRATCH_ROOT}" "${HOME}" "${XDG_CACHE_HOME}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${TMPDIR}"

export CONDA_ENV_NAME=${TRAIN_ENV}
source "${PROJECT_ROOT}/hpc/shared/env_qwen3vl.sh"
PYTHON=${TRAIN_ENV}/bin/python
SWIFT=${TRAIN_ENV}/bin/swift
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
cd "${PROJECT_ROOT}"

"${PYTHON}" -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"

test -s "${MODEL_DIR}/config.json"
test -s "${CSV_PATH}"
test -s "${SPLIT_PATH}"
test -s "${MEDIA_MAP}"
test -x "${PYTHON}"
test -x "${SWIFT}"
test -x "${FFMPEG_ENV}/bin/ffmpeg"
test -x "${FFMPEG_ENV}/bin/ffprobe"
"${PYTHON}" -c 'from torchcodec.decoders import VideoDecoder; print("VideoDecoder", VideoDecoder.__module__)' \
  > "${OUTDIR}/video_decoder_environment.txt"
"${PYTHON}" -c 'import importlib.metadata as m; print({n:m.version(n) for n in ("ms-swift","torch","transformers","torchcodec")})' \
  > "${OUTDIR}/dependencies.txt"

save_resolved_command() {
  local destination=$1
  shift
  printf '%q ' "$@" > "${destination}"
  printf '\n' >> "${destination}"
}

collect_swift_artifacts() {
  local swift_root=$1
  local state config checkpoint
  state=$(find "${swift_root}" -type f -name trainer_state.json -print | sort | tail -n 1)
  config=$(find "${swift_root}" -type f -name adapter_config.json -print | sort | tail -n 1)
  test -n "${state}"
  test -n "${config}"
  checkpoint=$(dirname "${config}")
  cp "${state}" "${OUTDIR}/trainer_state.json"
  rm -rf "${OUTDIR}/adapter"
  cp -a "${checkpoint}" "${OUTDIR}/adapter"
  test -s "${OUTDIR}/trainer_state.json"
  test -s "${OUTDIR}/adapter/adapter_config.json"
}

write_parameter_audit() {
  local adapter_dir=$1
  "${PYTHON}" - "${adapter_dir}" "${MODEL_DIR}" "${OUTDIR}/parameter_audit.json" <<'PY'
import json
import sys
from pathlib import Path

from peft import PeftConfig
from safetensors.torch import load_file

adapter = Path(sys.argv[1]).resolve()
model = Path(sys.argv[2]).resolve()
output = Path(sys.argv[3])
config_path = adapter / "adapter_config.json"
weights = sorted(adapter.glob("adapter_model*.safetensors"))
state = {}
for path in weights:
    state.update(load_file(str(path), device="cpu"))
keys = sorted(state)
allowed = all("lora_" in key and ("q_proj" in key or "v_proj" in key) for key in keys)
nonzero = any(tensor.count_nonzero().item() > 0 for key, tensor in state.items() if "lora_B" in key)
peft_config = PeftConfig.from_pretrained(str(adapter), local_files_only=True)
result = {
    "checkpoint_exists": config_path.is_file() and bool(weights),
    "checkpoint_reloadable": peft_config is not None,
    "base_model_local": model.is_dir(),
    "lora_parameter_names": keys,
    "lora_delta_nonzero": bool(nonzero),
    "non_lora_delta_zero": bool(keys) and allowed,
    "target_modules_exact": bool(keys) and allowed,
}
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if not all(result[key] for key in (
    "checkpoint_exists", "checkpoint_reloadable", "base_model_local",
    "lora_delta_nonzero", "non_lora_delta_zero", "target_modules_exact",
)):
    raise SystemExit(2)
PY
}

export PROJECT_ROOT TRAIN_ENV FFMPEG_ENV MODEL_DIR DATA_DIR OUTPUT_ROOT CSV_PATH SPLIT_PATH MEDIA_MAP
export DPO_DATA_DIR JOB_SCRATCH_ROOT OUTDIR PYTHON SWIFT
