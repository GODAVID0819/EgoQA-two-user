#!/bin/bash
# Portable (non-SLURM) launcher for the groundedness judge run.
# Same hyperparameters as train_groundedness_025fps_mb1_ga4_noclassw.sbatch.
# Requires 2 x H200 (141 GB). Run from anywhere; set the paths below or export them.
set -euo pipefail

REPO=${REPO:?set REPO to the multi-user directory of this repository}
ENV=${ENV:?set ENV to the qwen38-vllm conda environment prefix}
MODEL=${MODEL:?set MODEL to the local Qwen3.8-27B snapshot directory}
DATA=${DATA:?set DATA to the dataset manifests directory (contains train/ and test/)}
OUT=${OUT:-$REPO/runs/groundedness_025fps_mb1_ga4_last8_r16_lr3e5_cosine_w005_noclassw_$(date +%Y%m%d_%H%M%S)}
TASK=${TASK:-groundedness}     # all | formality | groundedness | answerability
EPOCHS=${EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:--1}     # e.g. MAX_STEPS=1 for a smoke test
TRAIN_MANIFEST=${TRAIN_MANIFEST:-$DATA/train/train.local.jsonl}
TEST_MANIFEST=${TEST_MANIFEST:-$DATA/test/train.local.jsonl}
LINEAR_ATTN=${LINEAR_ATTN:-0}   # 1 = also put LoRA on the 5 linear-attention projections
RESUME=${RESUME:-}             # path to trainer/checkpoint-<step> to resume from

TARGETS=(q_proj k_proj v_proj o_proj gate_proj up_proj down_proj)
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
if [ "$LINEAR_ATTN" = "1" ]; then
  TARGETS+=(in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj)
fi
EXTRA_ARGS=()
if [ -n "$RESUME" ]; then EXTRA_ARGS+=(--resume-from-checkpoint "$RESUME"); fi

mkdir -p "$OUT"
cd "$REPO"

export PATH="$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$OUT/triton_cache}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR"

# Evaluate on the held-out test split after every epoch; set EVAL=0 to skip.
if [ "${EVAL:-1}" = "1" ]; then export JUDGE_EVAL_MANIFEST="$TEST_MANIFEST"; fi

"$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m training.judge_sft.train_025fps_mb1 \
  --train-manifest "$TRAIN_MANIFEST" \
  --output-dir "$OUT/trainer" \
  --model-id "$MODEL" \
  --local-files-only \
  --attn-implementation sdpa \
  --tensor-parallel-size 2 \
  --task "$TASK" \
  --min-pixels 3136 \
  --max-pixels 119808 \
  --max-input-tokens 262144 \
  --image-context-target-fraction 0.85 \
  --lora-rank 16 \
  --lora-alpha 16 \
  --lora-dropout "$LORA_DROPOUT" \
  --trainable-decoder-layers 8 \
  --lora-target-modules "${TARGETS[@]}" \
  --learning-rate 3e-5 \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-epsilon 1e-8 \
  --epochs "$EPOCHS" \
  --max-steps "$MAX_STEPS" \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  --no-class-weights \
  --max-grad-norm 1.0 \
  --seed 17 \
  --gradient-checkpointing \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 1 \
  --decoded-image-cache-entries 2 \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee -a "$OUT/train.log"

echo "RUN=$OUT"
