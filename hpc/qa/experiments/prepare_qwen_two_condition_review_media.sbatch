#!/usr/bin/env bash
#SBATCH --job-name=qwen_review_media
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --partition=cpu_short
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-qwen-two-condition-review-20260901}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/xl6775/experiments/qwen_two_condition_review}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/xl6775/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/xl6775/envs/egoqa-ffmpeg-runtime}"
INPUT_ROOT="${INPUT_ROOT:-${PROJECT_ROOT}/review_inputs/qwen_two_condition_20260901}"
SELECTION_PATH="${SELECTION_PATH:-${INPUT_ROOT}/selection.jsonl}"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/qwen_review_media_${SLURM_JOB_ID}"
MEDIA_RUN_ROOT="${OUTPUT_ROOT}/media_${SLURM_JOB_ID}"
MEDIA_ROOT="${MEDIA_RUN_ROOT}/stitched"
MEDIA_MANIFEST="${MEDIA_RUN_ROOT}/media_manifest.json"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"

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
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
export CONDA_PKGS_DIRS="${JOB_SCRATCH_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"

mkdir -p \
  "${MEDIA_ROOT}" "${PYTHON_ALIAS_ROOT}" "${HOME}" "${XDG_CACHE_HOME}" \
  "${HF_HOME}" "${HF_DATASETS_CACHE}" "${MODELSCOPE_CACHE}" "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" "${FLASHINFER_WORKSPACE_BASE}" "${TMPDIR}" \
  "${CONDA_PKGS_DIRS}" "${PIP_CACHE_DIR}"

cd "${PROJECT_ROOT}"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${TRAIN_ENV}"
PYTHON="${TRAIN_ENV}/bin/python"
ln -s "${PROJECT_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

test -x "${PYTHON}"
test -x "${FFMPEG_ENV}/bin/ffmpeg"
test -x "${FFMPEG_ENV}/bin/ffprobe"
test -s "${SELECTION_PATH}"
python -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${MEDIA_RUN_ROOT}/storage_preflight.json"

if python -m pip check > "${MEDIA_RUN_ROOT}/pip_check.txt" 2>&1; then
  cat "${MEDIA_RUN_ROOT}/pip_check.txt"
else
  cat "${MEDIA_RUN_ROOT}/pip_check.txt"
  UNEXPECTED_PIP_CHECK="$(
    grep -Ev '^decord [^[:space:]]+ is not supported on this platform$' \
      "${MEDIA_RUN_ROOT}/pip_check.txt" || true
  )"
  if [[ -n "${UNEXPECTED_PIP_CHECK}" ]]; then
    echo "STOP: pip check found an unexpected dependency error" >&2
    printf '%s\n' "${UNEXPECTED_PIP_CHECK}" >&2
    exit 1
  fi
  echo "WARNING: pip check reported only the known decord platform metadata warning"
fi

python "${PROJECT_ROOT}/tools/prepare_qwen_review_stitched_media.py" \
  --selection "${SELECTION_PATH}" \
  --media-root "${MEDIA_ROOT}" \
  --work-root "${JOB_SCRATCH_ROOT}/segments" \
  --ffmpeg "${FFMPEG_ENV}/bin/ffmpeg" \
  --ffprobe "${FFMPEG_ENV}/bin/ffprobe" \
  --manifest "${MEDIA_MANIFEST}" \
  --workers "${SLURM_CPUS_PER_TASK}"

python - "${MEDIA_MANIFEST}" "${MEDIA_ROOT}" "${MEDIA_RUN_ROOT}/job_result.json" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
media_root = Path(sys.argv[2])
result_path = Path(sys.argv[3])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
items = manifest.get("items") or []
valid = (
    manifest.get("status") == "passed"
    and len(items) == 60
    and all(590.0 <= float(item["duration_seconds"]) <= 615.0 for item in items)
    and all(Path(item["output_path"]).is_file() for item in items)
)
result = {
    "status": "passed" if valid else "failed",
    "job_id": __import__("os").environ["SLURM_JOB_ID"],
    "media_root": str(media_root),
    "stitched_video_count": len(items),
    "expected_stitched_video_count": 60,
    "media_manifest": str(manifest_path),
}
result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
if not valid:
    raise SystemExit("media preparation result is incomplete")
PY

echo "QWEN_REVIEW_MEDIA_FINISHED result=${MEDIA_RUN_ROOT}/job_result.json"
