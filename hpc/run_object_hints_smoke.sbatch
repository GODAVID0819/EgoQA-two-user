#!/usr/bin/env bash
#SBATCH --job-name=egolife_obj_reid_smoke
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"

# Reuse existing pruning/preprocessing output. This job does not enter QA generation.
EVIDENCE_ROOT="${EGOLIFE2U_EVIDENCE_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${EVIDENCE_ROOT}/evidence_pruned_pairs.jsonl}"

# Smoke-test outputs: extracted frames, object crops, and enriched evidence JSONL.
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/object_hints_reid_smoke}"
ASSET_DIR="${OBJECT_HINT_ASSET_DIR:-${OUTDIR}/assets}"
OBJECT_HINTED_EVIDENCE="${OBJECT_HINTED_EVIDENCE:-${OUTDIR}/object_hinted_evidence.jsonl}"

MAX_PACKETS="${OBJECT_HINT_MAX_PACKETS:-2}"
FRAMES_PER_CLIP="${OBJECT_HINT_FRAMES_PER_CLIP:-0}"
OBJECTS_PER_CLIP="${OBJECT_HINT_OBJECTS_PER_CLIP:-12}"
OBJECTS_PER_PACKET="${OBJECT_HINT_OBJECTS_PER_PACKET:-24}"
OBJECT_DETECTION_WORKERS="${OBJECT_HINT_N_WORKERS:-1}"
FRAME_SAMPLING_MODE="${OBJECT_HINT_FRAME_SAMPLING_MODE:-clip_medoids}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-262144}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"

REID_ENABLE="${OBJECT_REID_ENABLE:-1}"
REID_MODEL_ID="${OBJECT_REID_MODEL_ID:-openai/clip-vit-base-patch32}"
REID_VISUAL_THRESHOLD="${OBJECT_REID_VISUAL_THRESHOLD:-0.8}"
REID_TEXT_THRESHOLD="${OBJECT_REID_TEXT_THRESHOLD:-0.85}"
REID_BATCH_SIZE="${OBJECT_REID_BATCH_SIZE:-32}"
REID_DEVICE="${OBJECT_REID_DEVICE:-cuda}"

CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-1}"
CUDA_KEEPER_PID=""

export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    echo "stage=stop_cuda_keeper pid=${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}" "${ASSET_DIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "evidence_root=${EVIDENCE_ROOT}"
echo "evidence_path=${EVIDENCE_PATH}"
echo "outdir=${OUTDIR}"
echo "asset_dir=${ASSET_DIR}"
echo "object_hinted_evidence=${OBJECT_HINTED_EVIDENCE}"
echo "max_packets=${MAX_PACKETS}"
echo "frames_per_clip=${FRAMES_PER_CLIP}"
echo "objects_per_clip=${OBJECTS_PER_CLIP}"
echo "objects_per_packet=${OBJECTS_PER_PACKET}"
echo "object_detection_workers=${OBJECT_DETECTION_WORKERS}"
echo "frame_sampling_mode=${FRAME_SAMPLING_MODE}"
echo "qwen_model_id=${QWEN_MODEL_ID}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "max_image_pixels=${MAX_IMAGE_PIXELS}"
echo "disable_thinking=${DISABLE_THINKING}"
echo "reid_enable=${REID_ENABLE}"
echo "reid_model_id=${REID_MODEL_ID}"
echo "reid_visual_threshold=${REID_VISUAL_THRESHOLD}"
echo "reid_text_threshold=${REID_TEXT_THRESHOLD}"
echo "reid_batch_size=${REID_BATCH_SIZE}"
echo "reid_device=${REID_DEVICE}"
echo "benchmark_contract=object_detection_and_reid_smoke_only_no_qa_generation"
python --version
nvidia-smi || true

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "error=ffmpeg_missing reason=object_hints_extracts_frames_from_pruned_local_video" >&2
  exit 1
fi

echo "stage=preflight_object_hint_imports"
python - <<'PY'
import importlib.util
required = ["torch", "PIL"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("missing_required=" + ",".join(missing))
    raise SystemExit(1)
optional_missing = [name for name in ("cv2",) if importlib.util.find_spec(name) is None]
if optional_missing:
    print("missing_optional=" + ",".join(optional_missing))
    print("crop_backend=pil_fallback")
print("object_hint_base_imports=ok")
PY

if [[ "${REID_ENABLE}" == "1" ]]; then
  echo "stage=preflight_reid_imports"
  python - <<'PY'
import importlib.util
missing = [name for name in ("transformers",) if importlib.util.find_spec(name) is None]
if missing:
    print("missing_reid_required=" + ",".join(missing))
    print("install_hint=pip install transformers")
    raise SystemExit(1)
print("reid_imports=ok")
PY
fi

if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  if [[ -s "${CUDA_KEEPER_SCRIPT}" ]]; then
    echo "stage=preflight_cuda_keeper_imports"
    if ! python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
optional_missing = [name for name in ("psutil",) if importlib.util.find_spec(name) is None]
if missing or optional_missing:
    print("missing_required=" + ",".join(missing))
    print("missing_optional=" + ",".join(optional_missing))
    raise SystemExit(1)
print("cuda_keeper_imports=ok")
PY
    then
      if [[ "${CUDA_KEEPER_AUTO_INSTALL}" == "1" ]]; then
        echo "stage=install_cuda_keeper_deps"
        python -m pip install nvidia-ml-py3 psutil || true
      fi
    fi

    if python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_required_after_install=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_required_imports=ok")
PY
    then
      echo "stage=start_cuda_keeper"
      python "${CUDA_KEEPER_SCRIPT}" \
        --threshold "${CUDA_KEEPER_THRESHOLD}" \
        --gpus "${CUDA_KEEPER_GPUS}" \
        --reserve "${CUDA_KEEPER_RESERVE}" \
        > "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log" 2>&1 &
      CUDA_KEEPER_PID=$!
      sleep 2
      if kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
        echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"
      else
        echo "warning=cuda_keeper_exited_early; continuing without keeper"
        CUDA_KEEPER_PID=""
      fi
    else
      echo "warning=cuda_keeper_imports_missing; continuing without keeper"
    fi
  else
    echo "warning=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}; continuing without keeper"
  fi
fi

thinking_args=()
if [[ "${DISABLE_THINKING}" == "1" ]]; then
  thinking_args+=(--disable-thinking)
fi

reid_args=()
if [[ "${REID_ENABLE}" != "1" ]]; then
  reid_args+=(--disable-reid)
else
  reid_args+=(
    --reid-model-id "${REID_MODEL_ID}"
    --reid-device "${REID_DEVICE}"
    --reid-visual-threshold "${REID_VISUAL_THRESHOLD}"
    --reid-text-threshold "${REID_TEXT_THRESHOLD}"
    --reid-batch-size "${REID_BATCH_SIZE}"
  )
fi

echo "stage=add_object_hints"
python -m egolife_two_user_qa add_object_hints \
  --evidence "${EVIDENCE_PATH}" \
  --output "${OBJECT_HINTED_EVIDENCE}" \
  --output-dir "${ASSET_DIR}" \
  --backend transformers-local \
  --model-id "${QWEN_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --max-image-pixels "${MAX_IMAGE_PIXELS}" \
  --frame-sampling-mode "${FRAME_SAMPLING_MODE}" \
  --frames-per-clip "${FRAMES_PER_CLIP}" \
  --objects-per-clip "${OBJECTS_PER_CLIP}" \
  --objects-per-packet "${OBJECTS_PER_PACKET}" \
  --n-workers "${OBJECT_DETECTION_WORKERS}" \
  --max-packets "${MAX_PACKETS}" \
  "${thinking_args[@]}" \
  "${reid_args[@]}"

echo "stage=summarize_object_hints"
python - "${OBJECT_HINTED_EVIDENCE}" "${ASSET_DIR}" <<'PY'
import json
import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
asset_dir = Path(sys.argv[2])
rows = [json.loads(line) for line in evidence_path.read_text(encoding="utf-8").splitlines() if line.strip()]
packets = len(rows)
clip_hint_count = 0
key_object_count = 0
reid_object_count = 0
failures = 0
for packet in rows:
    key_object_count += len(packet.get("object_hints", {}).get("key_objects") or [])
    failures += len(packet.get("object_hints", {}).get("failures") or [])
    for clip in packet.get("clips") or []:
        hints = clip.get("object_hints") or {}
        if hints.get("available"):
            clip_hint_count += 1
        reid_object_count += int(hints.get("reid_object_count") or 0)

crop_root = asset_dir / "object_reid_crops"
frame_root = asset_dir / "object_hint_frames"
crop_count = len(list(crop_root.rglob("*.jpg"))) if crop_root.exists() else 0
frame_count = len(list(frame_root.rglob("*.jpg"))) if frame_root.exists() else 0

print(f"summary_packets={packets}")
print(f"summary_clips_with_hints={clip_hint_count}")
print(f"summary_packet_key_objects={key_object_count}")
print(f"summary_reid_object_count={reid_object_count}")
print(f"summary_failures={failures}")
print(f"summary_extracted_frames={frame_count}")
print(f"summary_object_crops={crop_count}")
print(f"crop_root={crop_root}")
print(f"frame_root={frame_root}")
PY

echo "outputs:"
echo "  input_evidence=${EVIDENCE_PATH}"
echo "  object_hinted_evidence=${OBJECT_HINTED_EVIDENCE}"
echo "  extracted_frames=${ASSET_DIR}/object_hint_frames"
echo "  object_crops=${ASSET_DIR}/object_reid_crops"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log"
echo "date_done=$(date --iso-8601=seconds)"
