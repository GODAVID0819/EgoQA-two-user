#!/bin/bash
#SBATCH --job-name=judge_sft_grounded
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/logs/%x_%j.out
set -euo pipefail

# ============================================================
# Paths (adapted from "#!binbash.txt" to tx856's layout)
# ============================================================

# Directory that contains training/judge_sft (symlink -> ./judge_sft)
REPO=/scratch/tx856/multi_user_qa
ENV=/scratch/tx856/.conda/envs/qwen38-vllm

# Model weights and frame packets are read from hm2991's scratch
# (manifests reference /scratch/hm2991/egolife_rlhf_evidence_v1/packets directly).
MODEL=/scratch/hm2991/egolife_vllm_runtime/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0

TRAIN_MANIFEST=$REPO/manifests_842_question_split/train/train.jsonl
TEST_MANIFEST=$REPO/manifests_842_question_split/test/train.jsonl

TASK=${TASK:-groundedness}   # all | formality | groundedness | answerability
EPOCHS=${EPOCHS:-10}
MAX_STEPS=${MAX_STEPS:--1}   # set e.g. MAX_STEPS=1 for a smoke test

STAMP=$(date +%Y%m%d_%H%M%S)
OUT=${OUT:-/scratch/tx856/multi_user_qa/runs/${TASK}_025fps_mb1_ga4_last8_r16_lr3e5_cosine_w005_noclassw_${STAMP}}

mkdir -p "$OUT" /scratch/tx856/multi_user_qa/logs
cd "$REPO"

# ============================================================
# Environment
# ============================================================

export PATH="$ENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1

# Avoid NFS Triton cache
export TRITON_CACHE_DIR=/scratch/tx856/triton_cache
mkdir -p "$TRITON_CACHE_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable automatic held-out evaluation at every epoch.
export JUDGE_EVAL_MANIFEST="$TEST_MANIFEST"

# ============================================================
# Train
# ============================================================
# --class-weight-smoothing 1e9 makes the pass/fail class weights ~1.0000001
# (i.e. no class weighting); the values are recorded in training_contract.json.

"$ENV/bin/torchrun" \
  --standalone \
  --nproc_per_node=2 \
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
  --lora-dropout 0.05 \
  --trainable-decoder-layers 8 \
  --lora-target-modules \
      q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
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
  --class-weight-smoothing 1e9 \
  --max-grad-norm 1.0 \
  --seed 17 \
  --gradient-checkpointing \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 1 \
  --decoded-image-cache-entries 2 \
  2>&1 | tee "$OUT/train.log"

echo "RUN=$OUT"
