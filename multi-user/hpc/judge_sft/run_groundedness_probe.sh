#!/usr/bin/env bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/$USER/Long-video-understanding-clip}"
PACKAGE_ROOT="${PACKAGE_ROOT:-$PROJECT_ROOT/egolife_two_user_qa/multi-user}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/$USER/conda/envs/qwen38-vllm}"
MANIFEST="${MANIFEST:-/scratch/$USER/egolife_judge_sft/manifests_40_packets_18017425_half_seed17/train.jsonl}"
MODEL_PATH="${MODEL_PATH:-/scratch/$USER/egolife_vllm_runtime/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}"
OUT="${OUT:-/scratch/$USER/egolife_judge_sft/groundedness_probe_${SLURM_JOB_ID:-manual}_$(date +%Y%m%d_%H%M%S)}"
MAX_STEPS=1

mkdir -p "$OUT"
cd "$PACKAGE_ROOT"

GPU_MONITOR_PID=""
CPU_MONITOR_PID=""
cleanup() {
  if [[ -n "$GPU_MONITOR_PID" ]]; then kill "$GPU_MONITOR_PID" 2>/dev/null || true; fi
  if [[ -n "$CPU_MONITOR_PID" ]]; then kill "$CPU_MONITOR_PID" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

nvidia-smi \
  --query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
  --format=csv \
  -lms 500 \
  > "$OUT/gpu_timeseries.csv" &
GPU_MONITOR_PID=$!

"$TRAIN_ENV/bin/python" -u training/judge_sft/resource_monitor.py \
  --output "$OUT/cpu_timeseries.csv" \
  --interval 0.5 &
CPU_MONITOR_PID=$!

{
  echo "started_at=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${SLURM_JOB_ID:-}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-}"
  echo "manifest=$MANIFEST"
  echo "model_path=$MODEL_PATH"
  echo "output=$OUT"
  echo "max_steps=$MAX_STEPS"
  echo "git_head=$(git rev-parse HEAD 2>/dev/null || true)"
  sha256sum \
    training/judge_sft/train.py \
    training/judge_sft/trainer.py \
    training/judge_sft/collator.py \
    training/judge_sft/contracts.py \
    "$MANIFEST" 2>/dev/null || true
  echo "---- nvidia-smi ----"
  nvidia-smi
} > "$OUT/audit.txt" 2>&1

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

"$TRAIN_ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
  -m training.judge_sft.train \
  --train-manifest "$MANIFEST" \
  --output-dir "$OUT/trainer" \
  --model-id "$MODEL_PATH" \
  --local-files-only \
  --tensor-parallel-size 2 \
  --task groundedness \
  --max-steps "$MAX_STEPS" \
  --gradient-accumulation-steps 1 \
  --trainable-decoder-layers 24 \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 1 \
  --decoded-image-cache-entries 2 \
  --probe-instrumentation \
  --probe-skip-save \
  2>&1 | tee "$OUT/train.log"

cleanup
trap - EXIT INT TERM

echo "probe_output=$OUT"
echo "rank0_summary=$OUT/trainer/probe_summary_rank0.json"
echo "rank1_summary=$OUT/trainer/probe_summary_rank1.json"
echo "gpu_timeseries=$OUT/gpu_timeseries.csv"
echo "cpu_timeseries=$OUT/cpu_timeseries.csv"
