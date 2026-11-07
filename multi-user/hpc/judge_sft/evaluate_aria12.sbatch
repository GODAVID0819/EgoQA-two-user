#!/bin/bash
#SBATCH --job-name=aria12_eval
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
set -euo pipefail
: "${JUDGE_PROJECT:?}"
: "${JUDGE_REPO:?}"
: "${JUDGE_ENV:?}"
: "${JUDGE_MODEL:?}"
: "${JUDGE_EVAL_ADAPTER:?}"
: "${JUDGE_EVAL_LABEL:?}"
job_root="$JUDGE_PROJECT/jobs/$SLURM_JOB_ID"
mkdir -p "$job_root/cache"
cd "$JUDGE_REPO"
job_env=()
for name in HOME XDG_CACHE_HOME HF_HOME HF_DATASETS_CACHE MODELSCOPE_CACHE TORCH_HOME TRITON_CACHE_DIR TORCHINDUCTOR_CACHE_DIR VLLM_CACHE_ROOT CUDA_CACHE_PATH FLASHINFER_WORKSPACE_BASE TMPDIR TMP TEMP; do
  job_env+=("$name=$job_root/cache/$name")
done
env "${job_env[@]}" "$JUDGE_ENV/bin/python" -m training.torch_storage_preflight \
  --allowed-root "$job_root" --output "$job_root/storage_preflight.json"
env "${job_env[@]}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  "$JUDGE_ENV/bin/python" -m training.judge_sft.evaluate_aria12 \
  --model-id "$JUDGE_MODEL" --adapter-dir "$JUDGE_EVAL_ADAPTER" \
  --eval-manifest "$JUDGE_PROJECT/input/test.jsonl" \
  --output-dir "$job_root/evaluation" --label "$JUDGE_EVAL_LABEL"
