#!/bin/bash
#SBATCH --job-name=judge_eval_margin
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/logs/%x_%j.out
set -euo pipefail

# Per-example margins for base + epoch 1/2/3 checkpoints of
# groundedness_025fps_mb1_ga8_last8_r16_lr3e5_20261004_005220, on the
# 168-row groundedness test split, using the 0.25 FPS training collator.

REPO=/scratch/tx856/multi_user_qa
ENV=/scratch/tx856/.conda/envs/qwen38-vllm
MODEL=/scratch/hm2991/egolife_vllm_runtime/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
RUN=/scratch/hm2991/egolife_judge_sft/groundedness_025fps_mb1_ga8_last8_r16_lr3e5_20261004_005220/trainer
OUT=$REPO/eval_margin/out_$(date +%Y%m%d_%H%M%S)

mkdir -p "$OUT" $REPO/logs
cd "$REPO"

export PATH="$ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR=/scratch/tx856/triton_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TRITON_CACHE_DIR"

"$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m training.judge_sft.evaluate_groundedness_multi_025fps \
  --eval-manifest "$REPO/eval_margin/test_groundedness.jsonl" \
  --train-manifest "$REPO/manifests_842_question_split/train/train.jsonl" \
  --training-contract "$REPO/eval_margin/training_contract.json" \
  --adapter-dirs "$REPO/eval_margin/base_zeroB" "$RUN/checkpoint-85" "$RUN/checkpoint-170" "$RUN/checkpoint-255" \
  --adapter-labels base epoch1 epoch2 epoch3 \
  --model-id "$MODEL" \
  2>&1 | tee "$OUT/eval.log"

echo "OUT=$OUT"
