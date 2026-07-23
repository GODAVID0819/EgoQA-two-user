#!/usr/bin/env bash
#SBATCH --job-name=egolife_pruneabl30
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/pruning_ablation_30s}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

PAIR_COUNT="${PRUNING_ABLATION_PAIR_COUNT:-10}"
MAX_GROUPS="${PRUNING_ABLATION_MAX_GROUPS:-80}"
FPS_VALUES="${PRUNING_ABLATION_FPS_VALUES:-0.5,1,2,4}"
K_VALUES="${PRUNING_ABLATION_K_VALUES:-4,8,12,16,20,24,30}"
THRESHOLD_VALUES="${PRUNING_ABLATION_THRESHOLD_VALUES:-0.78,0.80,0.82,0.84,0.86,0.88}"
TEMPORAL_POLICIES="${PRUNING_ABLATION_TEMPORAL_POLICIES:-current,gate_1s,gate_2s,gate_5s,gate_2s_mnn,gate_2s_mnn_contiguous}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
MIN_PRUNED_VIDEO_SECONDS="${PRUNING_ABLATION_MIN_VIDEO_SECONDS:-8}"
PRUNING_PROTECTION_MODE="${PRUNING_ABLATION_PROTECTION_MODE:-min_seconds}"
RANDOM_SEED="${PRUNING_ABLATION_RANDOM_SEED:-20260720}"

# The ablation uses the allocated GPU for the shared dense CLIP embedding pass.
# Keep the repository CUDA controller alive for the entire job and fail early
# if it cannot start, so the run never silently proceeds without it.
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
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "/scratch/${USER}/hf_cache"

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "date_start=$(date --iso-8601=seconds)"
echo "outdir=${OUTDIR}"
echo "cache_dir=${CACHE_DIR}"
echo "pair_count=${PAIR_COUNT}"
echo "max_groups=${MAX_GROUPS}"
echo "fps_values=${FPS_VALUES}"
echo "k_values=${K_VALUES}"
echo "threshold_values=${THRESHOLD_VALUES}"
echo "temporal_policies=${TEMPORAL_POLICIES}"
echo "clip_model_id=${CLIP_MODEL_ID}"
echo "min_pruned_video_seconds=${MIN_PRUNED_VIDEO_SECONDS}"
echo "pruning_protection_mode=${PRUNING_PROTECTION_MODE}"
echo "random_seed=${RANDOM_SEED}"
echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD} gpus=${CUDA_KEEPER_GPUS} reserve_gib=${CUDA_KEEPER_RESERVE}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
echo "cuda_keeper_auto_install=${CUDA_KEEPER_AUTO_INSTALL}"
python --version
nvidia-smi || true

if [[ ! -s "${CUDA_KEEPER_SCRIPT}" ]]; then
  echo "error=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}" >&2
  exit 1
fi

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
    echo "stage=install_cuda_keeper_dependencies"
    python -m pip install nvidia-ml-py3 psutil
  fi
fi

python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("cuda keeper dependencies remain missing: " + ",".join(missing))
print("cuda_keeper_required_imports=ok")
PY

echo "stage=start_cuda_keeper"
python "${CUDA_KEEPER_SCRIPT}" \
  --threshold "${CUDA_KEEPER_THRESHOLD}" \
  --gpus "${CUDA_KEEPER_GPUS}" \
  --reserve "${CUDA_KEEPER_RESERVE}" \
  > "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log" 2>&1 &
CUDA_KEEPER_PID=$!
sleep 2
if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
  echo "error=cuda_keeper_exited_early log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log" >&2
  wait "${CUDA_KEEPER_PID}" || true
  CUDA_KEEPER_PID=""
  exit 1
fi
echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"

echo "stage=build_manifest"
python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json" \
  --no-overlays

echo "stage=run_pruning_ablation"
python -m egolife_two_user_qa run_pruning_ablation \
  --manifest "${OUTDIR}/manifest.json" \
  --output-dir "${OUTDIR}/experiment" \
  --cache-dir "${CACHE_DIR}" \
  --pair-count "${PAIR_COUNT}" \
  --max-groups "${MAX_GROUPS}" \
  --min-group-size 2 \
  --duration-seconds 30 \
  --start-seconds 0 \
  --baseline-fps 1 \
  --baseline-k 12 \
  --baseline-threshold 0.82 \
  --baseline-temporal-policy current \
  --fps-values "${FPS_VALUES}" \
  --k-values "${K_VALUES}" \
  --threshold-values "${THRESHOLD_VALUES}" \
  --temporal-policies "${TEMPORAL_POLICIES}" \
  --model-id "${CLIP_MODEL_ID}" \
  --preserve-shared-anchor-seconds 0 \
  --min-pruned-video-seconds "${MIN_PRUNED_VIDEO_SECONDS}" \
  --pruning-protection-mode "${PRUNING_PROTECTION_MODE}" \
  --random-seed "${RANDOM_SEED}" \
  --download-media

echo "outputs:"
echo "  summary=${OUTDIR}/experiment/summary.json"
echo "  cohort=${OUTDIR}/experiment/cohort.jsonl"
echo "  metrics=${OUTDIR}/experiment/ablation_metrics.csv"
echo "  aggregates=${OUTDIR}/experiment/sweep_aggregates.csv"
echo "  assignments=${OUTDIR}/experiment/cluster_assignments.csv"
echo "  triggers=${OUTDIR}/experiment/trigger_pairs.csv"
echo "  centroids=${OUTDIR}/experiment/centroid_frames.csv"
echo "  review=${OUTDIR}/experiment/review.html"
echo "  videos=${OUTDIR}/experiment/pairs/*/{temporal,threshold,sampling,k}/*/*.mp4"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log"
echo "date_done=$(date --iso-8601=seconds)"
