#!/usr/bin/env bash
#SBATCH --job-name=egoqa_cur17_2models
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --constraint="h100|h200"
#SBATCH --mem=320G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "error=missing_slurm_job_id" >&2
  exit 2
fi

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}}"
HPC_ROOT="${PROJECT_ROOT}/hpc"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs/curated_10min_six_vs_required}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/scratch/${USER}/hf_cache}"
LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"

if [[ -s "${HPC_ROOT}/shared/env_qwen3vl.sh" ]]; then
  ENV_SCRIPT="${HPC_ROOT}/shared/env_qwen3vl.sh"
else
  ENV_SCRIPT="${HPC_ROOT}/env_qwen3vl.sh"
fi
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/shared/cuda.py}"
if [[ ! -s "${CUDA_KEEPER_SCRIPT}" && -s "${HPC_ROOT}/cuda.py" ]]; then
  CUDA_KEEPER_SCRIPT="${HPC_ROOT}/cuda.py"
fi
REQUIRED_GPU_COUNT=2
CUDA_KEEPER_GPUS="0,1"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-12.0}"
CUDA_KEEPER_PID=""
STORAGE_PREFLIGHT_MODULE=""

# These are the exact 10-minute composites produced by job 16463998. The
# preparer reuses them directly and only fills missing legacy windows from the
# already-populated 30-second EgoLife cache; it never downloads videos.
EVIDENCE_OWNER="${EVIDENCE_OWNER:-hm2991}"
TEN_MINUTE_ASSET_ROOT="${TEN_MINUTE_ASSET_ROOT:-/scratch/${EVIDENCE_OWNER}/Long-video-understanding-clip/egolife_two_user_qa/multi-user/outputs/six_user_qa_10min/six_user_qa_10min_time_aware_16463998/ten_minute_setup/time_aware_candidates/assets}"
SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT:-/scratch/${EVIDENCE_OWNER}/egolife_two_user_qa_cache}"

CURATED_QA="${CURATED_QA:-${PACKAGE_ROOT}/qa_curated_17_trace_review_v3.jsonl}"
PREPARER="${PREPARER:-${PACKAGE_ROOT}/prepare_curated_10min_ablation.py}"
SUMMARIZER="${SUMMARIZER:-${PACKAGE_ROOT}/summarize_curated_ablation.py}"

VIDEO_FPS="${VIDEO_FPS:-0.5}"
MIN_VIDEO_PIXELS="${MIN_VIDEO_PIXELS:-3136}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-65536}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-131072}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

RUN_LABEL="${RUN_LABEL:-curated_trace_v3_qwen3vl_8b_qwen36_27b_${SLURM_JOB_ID}}"
if [[ ! "${RUN_LABEL}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "error=invalid_run_label value=${RUN_LABEL}" >&2
  exit 2
fi
OUTDIR="${OUTPUT_ROOT}/${RUN_LABEL}"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/${RUN_LABEL}_${SLURM_JOB_ID}"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"
PYTHON=""
CURRENT_STAGE="bootstrap"
CUDA_KEEPER_LOG="${OUTDIR}/cuda_keeper_${SLURM_JOB_ID}.log"

write_status() {
  local status_value="$1"
  local stage_value="$2"
  local exit_code_value="${3:-0}"
  if [[ -d "${OUTDIR}" ]]; then
    printf '{"status":"%s","stage":"%s","job_id":"%s","exit_code":%s,"timestamp":"%s"}\n' \
      "${status_value}" "${stage_value}" "${SLURM_JOB_ID}" \
      "${exit_code_value}" "$(date --iso-8601=seconds)" \
      > "${OUTDIR}/stage_status.json"
  fi
}

stage_start() {
  CURRENT_STAGE="$1"
  printf 'stage=%s status=started timestamp=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)"
  write_status "running" "${CURRENT_STAGE}" 0
}

on_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'stage=%s status=failed exit_code=%s line=%s command=%q\n' \
    "${CURRENT_STAGE}" "${exit_code}" "${line_number}" "${failed_command}" >&2
  write_status "failed" "${CURRENT_STAGE}" "${exit_code}"
  exit "${exit_code}"
}

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap cleanup EXIT INT TERM

stage_start "validate_project_layout"
cd "${PROJECT_ROOT}"
test -d "${HPC_ROOT}"
test -s "${ENV_SCRIPT}"
test -s "${CUDA_KEEPER_SCRIPT}"
test -d "${PACKAGE_ROOT}"
test -s "${CURATED_QA}"
test -s "${PREPARER}"
test -s "${SUMMARIZER}"
test -s "${PACKAGE_ROOT}/qwen3vl_runner.py"
test -s "${PACKAGE_ROOT}/six_video_qa_tester.py"
if [[ -s "${PACKAGE_ROOT}/training/torch_storage_preflight.py" ]]; then
  STORAGE_PREFLIGHT_MODULE="egolife_two_user_qa.training.torch_storage_preflight"
elif [[ -s "${PROJECT_ROOT}/training/torch_storage_preflight.py" ]]; then
  STORAGE_PREFLIGHT_MODULE="training.torch_storage_preflight"
else
  echo "error=torch_storage_preflight_missing package_root=${PACKAGE_ROOT} project_root=${PROJECT_ROOT}" >&2
  exit 2
fi
test -d "${TEN_MINUTE_ASSET_ROOT}"
test -d "${SOURCE_CACHE_ROOT}"
mkdir -p "${HPC_ROOT}/logs" "${OUTDIR}" "${JOB_SCRATCH_ROOT}" \
  "${PYTHON_ALIAS_ROOT}" "${MODEL_CACHE_ROOT}" "${MODEL_CACHE_ROOT}/hub"

stage_start "activate_environment"
export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-${TRAIN_ENV}}"
export QWEN3VL_PROJECT_ROOT="${PROJECT_ROOT}"
source "${ENV_SCRIPT}"
PYTHON="$(command -v python)"
test -x "${PYTHON}"

stage_start "configure_pytorch_runtime_libraries"
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

stage_start "configure_job_storage"
if [[ -z "${HF_TOKEN:-}" && -s "${LOGIN_HF_TOKEN_PATH}" ]]; then
  export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"
fi
export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf_home"
# Model snapshots remain in the established persistent cache. All mutable
# processor, compiler, dataset, and temporary caches are isolated per job.
export HF_HUB_CACHE="${MODEL_CACHE_ROOT}/hub"
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
unset TORCH_CUDNN_V8_API_DISABLED
export TORCH_CUDNN_V8_API_DEBUG=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p "${HOME}" "${XDG_CACHE_HOME}" "${HF_HOME}" \
  "${TRANSFORMERS_CACHE}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" \
  "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" \
  "${TMPDIR}" "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}" \
  "${JOB_SCRATCH_ROOT}/qwen_video_cache"

ln -s "${PACKAGE_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

stage_start "resolve_video_runtime"
FFMPEG_BINARY=""
FFPROBE_BINARY=""
FFMPEG_SOURCE=""
FFMPEG_LIBRARY_DIR=""
if [[ -x "${FFMPEG_ENV}/bin/ffmpeg" && -x "${FFMPEG_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${FFMPEG_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${FFMPEG_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="FFMPEG_ENV"
  FFMPEG_LIBRARY_DIR="${FFMPEG_ENV}/lib"
elif [[ -x "${TRAIN_ENV}/bin/ffmpeg" && -x "${TRAIN_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${TRAIN_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${TRAIN_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="TRAIN_ENV"
  FFMPEG_LIBRARY_DIR="${TRAIN_ENV}/lib"
else
  FFMPEG_BINARY="$(command -v ffmpeg || true)"
  FFPROBE_BINARY="$(command -v ffprobe || true)"
  FFMPEG_SOURCE="activated_PATH"
fi
if [[ -z "${FFMPEG_BINARY}" || ! -x "${FFMPEG_BINARY}" || \
      -z "${FFPROBE_BINARY}" || ! -x "${FFPROBE_BINARY}" ]]; then
  printf 'FFmpeg runtime not found. Checked FFMPEG_ENV=%s, TRAIN_ENV=%s, and activated PATH=%s\n' \
    "${FFMPEG_ENV}" "${TRAIN_ENV}" "${PATH}" >&2
  exit 1
fi
export FFMPEG_BINARY FFPROBE_BINARY
export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"
if [[ -d "${FFMPEG_LIBRARY_DIR}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_RUNTIME_LIBRARY_PATH}${TORCH_RUNTIME_LIBRARY_PATH:+:}${FFMPEG_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
printf 'ffmpeg_source=%s ffmpeg=%s ffprobe=%s\n' \
  "${FFMPEG_SOURCE}" "${FFMPEG_BINARY}" "${FFPROBE_BINARY}"

export FORCE_QWENVL_VIDEO_READER="decord"
export QWEN_MEMORY_SAFE_VIDEO_FPS="${VIDEO_FPS}"
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="${MIN_VIDEO_PIXELS}"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS}"
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB="5"
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="64"
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="sdpa"
export QWEN_MEMORY_SAFE_DEVICE_MAP="balanced"
export QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT="${REQUIRED_GPU_COUNT}"
export QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS="1"
export QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE="512"
export QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR="${JOB_SCRATCH_ROOT}/qwen_video_cache"

stage_start "storage_preflight"
"${PYTHON}" -P -m "${STORAGE_PREFLIGHT_MODULE}" \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"

stage_start "runtime_preflight"
export RUN_REQUIRED_GPU_COUNT="${REQUIRED_GPU_COUNT}"
"${FFMPEG_BINARY}" -version | head -n 1
"${FFPROBE_BINARY}" -version | head -n 1
"${PYTHON}" -P - "${PACKAGE_ROOT}" <<'PY'
import os
import sys
from pathlib import Path

import decord
import torch
import transformers
import egolife_two_user_qa
from qwen_vl_utils.vision_process import get_video_reader_backend

expected = Path(sys.argv[1]).resolve()
actual = Path(egolife_two_user_qa.__file__).resolve().parent
if actual != expected:
    raise SystemExit(f"wrong package imported: expected={expected} actual={actual}")
required = int(os.environ["RUN_REQUIRED_GPU_COUNT"])
visible = torch.cuda.device_count()
if not torch.cuda.is_available() or visible < required:
    raise SystemExit(f"required {required} CUDA GPUs but only {visible} are visible")
backend = get_video_reader_backend()
if backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {backend}")
print(f"package_root={actual}")
print(f"torch={torch.__version__} transformers={transformers.__version__}")
print(f"video_backend={backend} decord={decord.__version__}")
for index in range(visible):
    properties = torch.cuda.get_device_properties(index)
    name = properties.name
    if "H100" not in name.upper() and "H200" not in name.upper():
        raise SystemExit(f"GPU {index} does not satisfy h100|h200 contract: {name}")
    print(
        "gpu",
        f"index={index}",
        f"name={name}",
        f"total_memory_gib={properties.total_memory / 1024**3:.3f}",
    )
PY

stage_start "long_context_cudnn_preflight"
"${PYTHON}" -P - <<'PY'
import importlib.util
import os
import sysconfig
from pathlib import Path

import torch

site_packages = Path(sysconfig.get_paths()["purelib"])
candidates = [
    str(path)
    for root in (site_packages / "nvidia" / "cudnn" / "lib", site_packages / "torch" / "lib")
    for path in root.glob("libcudnn.so*")
]
try:
    version = torch.backends.cudnn.version()
except RuntimeError as exc:
    raise SystemExit(
        "PyTorch could not load its matching bundled cuDNN: "
        f"error={exc}; candidates={candidates}; "
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}"
    ) from exc
print(
    "long_context_cudnn_preflight",
    f"available={torch.backends.cudnn.is_available()}",
    f"enabled={torch.backends.cudnn.enabled}",
    f"version={version}",
    f"v8_disabled={os.environ.get('TORCH_CUDNN_V8_API_DISABLED')!r}",
    f"causal_conv1d_available={importlib.util.find_spec('causal_conv1d') is not None}",
)
if not torch.backends.cudnn.is_available() or not torch.backends.cudnn.enabled:
    raise SystemExit("Qwen3.6 long-context prefill requires enabled cuDNN")
if version is None or version < 90300:
    raise SystemExit(f"Qwen3.6 requires cuDNN 9.3+; detected {version!r}")
PY

stage_start "model_resolution_preflight"
"${PYTHON}" -P - <<'PY'
from transformers import AutoConfig

for model_id in ("Qwen/Qwen3.6-27B", "Qwen/Qwen3-VL-8B-Instruct"):
    config = AutoConfig.from_pretrained(model_id)
    print(f"model_resolved={model_id} model_type={config.model_type}")
PY

stage_start "start_cuda_keeper"
"${PYTHON}" -P -c "import torch, pynvml, psutil"
export RUN_CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS}"
"${PYTHON}" -P - <<'PY'
import os
import torch

required = int(os.environ["RUN_REQUIRED_GPU_COUNT"])
visible = torch.cuda.device_count()
selected = [
    int(value.strip())
    for value in os.environ["RUN_CUDA_KEEPER_GPUS"].split(",")
    if value.strip()
]
expected = list(range(required))
if visible < required or selected != expected:
    raise SystemExit(
        "CUDA keeper must cover every requested local GPU exactly once: "
        f"visible={visible} expected={expected} selected={selected}"
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
printf 'cuda_keeper_pid=%s cuda_keeper_gpus=%s cuda_keeper_log=%s\n' \
  "${CUDA_KEEPER_PID}" "${CUDA_KEEPER_GPUS}" "${CUDA_KEEPER_LOG}"

stage_start "write_run_manifest"
RUN_GIT_COMMIT="$(git -C "${PACKAGE_ROOT}" rev-parse HEAD 2>/dev/null || printf unknown)"
if git -C "${PACKAGE_ROOT}" diff --quiet --ignore-submodules -- 2>/dev/null; then
  RUN_GIT_DIRTY="false"
else
  RUN_GIT_DIRTY="true"
fi
export RUN_GIT_COMMIT RUN_GIT_DIRTY
export RUN_MANIFEST_PATH="${OUTDIR}/run_manifest.json"
export RUN_PROJECT_ROOT="${PROJECT_ROOT}"
export RUN_HPC_ROOT="${HPC_ROOT}"
export RUN_PACKAGE_ROOT="${PACKAGE_ROOT}"
export RUN_OUTPUT_DIR="${OUTDIR}"
export RUN_JOB_SCRATCH_ROOT="${JOB_SCRATCH_ROOT}"
export RUN_LABEL_VALUE="${RUN_LABEL}"
export RUN_CURATED_QA="${CURATED_QA}"
export RUN_TEN_MINUTE_ASSET_ROOT="${TEN_MINUTE_ASSET_ROOT}"
export RUN_SOURCE_CACHE_ROOT="${SOURCE_CACHE_ROOT}"
export RUN_MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT}"
export RUN_VIDEO_FPS="${VIDEO_FPS}"
export RUN_MIN_VIDEO_PIXELS="${MIN_VIDEO_PIXELS}"
export RUN_MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS}"
export RUN_MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS}"
export RUN_MAX_NEW_TOKENS="${MAX_NEW_TOKENS}"
export RUN_FFMPEG_BINARY="${FFMPEG_BINARY}"
export RUN_FFPROBE_BINARY="${FFPROBE_BINARY}"
export RUN_CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT}"
export RUN_CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE}"
"${PYTHON}" -P - <<'PY'
import json
import os
from pathlib import Path

payload = {
    "schema_version": "curated_10min_six_vs_required_v1",
    "slurm_job_id": os.environ["SLURM_JOB_ID"],
    "run_label": os.environ["RUN_LABEL_VALUE"],
    "project_root": os.environ["RUN_PROJECT_ROOT"],
    "hpc_root": os.environ["RUN_HPC_ROOT"],
    "package_root": os.environ["RUN_PACKAGE_ROOT"],
    "output_dir": os.environ["RUN_OUTPUT_DIR"],
    "job_scratch_root": os.environ["RUN_JOB_SCRATCH_ROOT"],
    "git_commit": os.environ["RUN_GIT_COMMIT"],
    "git_dirty": os.environ["RUN_GIT_DIRTY"] == "true",
    "curated_qa": os.environ["RUN_CURATED_QA"],
    "question_count": 17,
    "condition_order": ["six_users", "required_users"],
    "model_order": ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3.6-27B"],
    "expected_model_calls": 68,
    "ten_minute_asset_root": os.environ["RUN_TEN_MINUTE_ASSET_ROOT"],
    "source_cache_root": os.environ["RUN_SOURCE_CACHE_ROOT"],
    "persistent_model_cache_root": os.environ["RUN_MODEL_CACHE_ROOT"],
    "video_fps": float(os.environ["RUN_VIDEO_FPS"]),
    "min_video_pixels": int(os.environ["RUN_MIN_VIDEO_PIXELS"]),
    "max_image_pixels": int(os.environ["RUN_MAX_IMAGE_PIXELS"]),
    "max_input_tokens": int(os.environ["RUN_MAX_INPUT_TOKENS"]),
    "max_new_tokens": int(os.environ["RUN_MAX_NEW_TOKENS"]),
    "required_gpu_count": int(os.environ["RUN_REQUIRED_GPU_COUNT"]),
    "qwen_device_map": os.environ["QWEN_MEMORY_SAFE_DEVICE_MAP"],
    "qwen_required_model_gpu_count": int(
        os.environ["QWEN_MEMORY_SAFE_REQUIRED_MODEL_GPU_COUNT"]
    ),
    "cuda_keeper_script": os.environ["RUN_CUDA_KEEPER_SCRIPT"],
    "cuda_keeper_gpus": os.environ["RUN_CUDA_KEEPER_GPUS"],
    "cuda_keeper_reserve_gib": float(os.environ["RUN_CUDA_KEEPER_RESERVE"]),
    "ffmpeg": os.environ["RUN_FFMPEG_BINARY"],
    "ffprobe": os.environ["RUN_FFPROBE_BINARY"],
    "video_backend": os.environ["FORCE_QWENVL_VIDEO_READER"],
    "resume_contract": (
        "Reuse an existing output directory by resubmitting with its RUN_LABEL; "
        "job scratch remains unique to the new SLURM_JOB_ID."
    ),
}
Path(os.environ["RUN_MANIFEST_PATH"]).write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

stage_start "prepare_shared_ten_minute_media"
"${PYTHON}" -P "${PREPARER}" \
  --qa "${CURATED_QA}" \
  --ten-minute-asset-root "${TEN_MINUTE_ASSET_ROOT}" \
  --source-cache-root "${SOURCE_CACHE_ROOT}" \
  --materialized-media-root "${OUTDIR}/prepared_media" \
  --evidence-output "${OUTDIR}/evidence_required_pairs.jsonl" \
  --six-view-output "${OUTDIR}/six_view_manifest.jsonl" \
  --provenance-output "${OUTDIR}/media_provenance.jsonl" \
  --summary-output "${OUTDIR}/media_preparation_summary.json" \
  --ffmpeg "${FFMPEG_BINARY}" \
  --ffprobe "${FFPROBE_BINARY}" \
  --expected-count 17

run_model() {
  local model_label="$1"
  local model_id="$2"
  local model_out="${OUTDIR}/${model_label}"
  mkdir -p "${model_out}"

  if "${PYTHON}" -P - \
    "${model_out}/six_users.jsonl" \
    "${model_out}/required_users.jsonl" \
    "${model_id}" <<'PY'
import json
import sys
from pathlib import Path

for raw_path in sys.argv[1:3]:
    path = Path(raw_path)
    if not path.is_file():
        raise SystemExit(1)
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 17:
        raise SystemExit(1)
    if any(row.get("status") != "completed" for row in rows):
        raise SystemExit(1)
    if any(str(row.get("model_id") or "") != sys.argv[3] for row in rows):
        raise SystemExit(1)
print(f"completed_model_outputs={sys.argv[3]} rows_per_condition=17")
PY
  then
    stage_start "${model_label}_resume_existing_completed_outputs"
    "${PYTHON}" -P "${SUMMARIZER}" \
      --six "${model_out}/six_users.jsonl" \
      --required "${model_out}/required_users.jsonl" \
      --output "${model_out}/paired_summary.json" \
      --per-question-output "${model_out}/per_question_comparison.jsonl" \
      --expected-count 17
    return 0
  fi

  stage_start "${model_label}_six_then_required"

  "${PYTHON}" -P - \
    "${model_id}" \
    "${CURATED_QA}" \
    "${OUTDIR}/evidence_required_pairs.jsonl" \
    "${OUTDIR}/six_view_manifest.jsonl" \
    "${model_out}" \
    "${OUTDIR}/anonymous_media" \
    "${MAX_NEW_TOKENS}" \
    "${MAX_IMAGE_PIXELS}" \
    "${VIDEO_FPS}" \
    "${MAX_INPUT_TOKENS}" <<'PY'
import json
import sys
from pathlib import Path

from egolife_two_user_qa.qwen3vl_runner import MEMORY_SAFE_BACKEND, make_runner
from egolife_two_user_qa.six_video_qa_tester import run_six_video_qa_test

(
    model_id,
    qa_path,
    evidence_path,
    six_view_path,
    output_dir,
    alias_dir,
    max_new_tokens,
    max_image_pixels,
    video_fps,
    max_input_tokens,
) = sys.argv[1:]
output_dir = Path(output_dir)

runner = make_runner(
    MEMORY_SAFE_BACKEND,
    model_id=model_id,
    max_new_tokens=int(max_new_tokens),
    max_image_pixels=int(max_image_pixels),
    dtype="bfloat16",
    disable_thinking=True,
)

summaries = {}
# Required experiment order: all six users first, then the minimal required set.
for label, scope in (("six_users", "six"), ("required_users", "pair")):
    summaries[label] = run_six_video_qa_test(
        qa_paths=[qa_path],
        evidence_path=evidence_path,
        six_view_manifest_path=six_view_path if scope == "six" else None,
        output_path=output_dir / f"{label}.jsonl",
        summary_path=output_dir / f"{label}.summary.json",
        prompts_path=output_dir / f"{label}.prompts.jsonl",
        media_alias_dir=alias_dir,
        backend=MEMORY_SAFE_BACKEND,
        model_id=model_id,
        max_new_tokens=int(max_new_tokens),
        max_image_pixels=int(max_image_pixels),
        dtype="bfloat16",
        disable_thinking=True,
        video_fps=float(video_fps),
        max_input_tokens=int(max_input_tokens),
        decoding_mode="greedy",
        resume=True,
        fail_fast=True,
        runner=runner,
        video_input_scope=scope,
        pair_video_source="full",
    )

(output_dir / "condition_summaries.json").write_text(
    json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY

  stage_start "${model_label}_paired_summary"
  "${PYTHON}" -P "${SUMMARIZER}" \
    --six "${model_out}/six_users.jsonl" \
    --required "${model_out}/required_users.jsonl" \
    --output "${model_out}/paired_summary.json" \
    --per-question-output "${model_out}/per_question_comparison.jsonl" \
    --expected-count 17
}

# Both requested models run in the same allocation. The 8B model runs first.
# Each model sees all six users first, then only the curated minimal pair.
run_model "qwen3vl_8b" "Qwen/Qwen3-VL-8B-Instruct"
run_model "qwen36_27b" "Qwen/Qwen3.6-27B"

stage_start "aggregate_results"
"${PYTHON}" -P - "${OUTDIR}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
models = {}
for label in ("qwen3vl_8b", "qwen36_27b"):
    models[label] = json.loads((root / label / "paired_summary.json").read_text(encoding="utf-8"))
payload = {
    "status": "completed",
    "question_count": 17,
    "condition_order": ["six_users", "required_users"],
    "models": models,
}
(root / "aggregate_summary.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(payload, ensure_ascii=False))
PY

CURRENT_STAGE="complete"
write_status "completed" "complete" 0
printf 'status=completed output=%s summary=%s\n' \
  "${OUTDIR}" "${OUTDIR}/aggregate_summary.json"
