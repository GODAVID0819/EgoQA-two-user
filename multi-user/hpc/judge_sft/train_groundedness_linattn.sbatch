#!/bin/bash
#SBATCH --job-name=judge_linattn
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=1-23:59:00
#SBATCH --output=/scratch/%u/multi_user_qa/logs/%x_%j.out
# Groundedness judge with LoRA on attention, MLP and the 5 linear-attention
# projections (upper 8 layers), 0.25 FPS, GA=4, cosine + 5% warmup, 10 epochs,
# held-out test evaluation after every epoch.
#
# Per-experiment knobs (pass with sbatch --export=ALL,KEY=VALUE,...):
#   LORA_RANK      LoRA rank r                       (default 128)
#   LORA_ALPHA     LoRA alpha; alpha/r is the scale  (default = LORA_RANK)
#   LR             peak learning rate                (default 1e-4)
#   CLASS_WEIGHTS  1 = inverse-frequency PASS/FAIL weights, 0 = plain BCE (default 0)
#   RESUME         full-state checkpoint dir to resume from (optional)
set -euo pipefail

REPO=${REPO:-/scratch/$USER/multi_user_qa/repo_check/multi-user}
ENV=${ENV:-/scratch/$USER/.conda/envs/qwen38-vllm}
MODEL=${MODEL:-/scratch/hm2991/egolife_vllm_runtime/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0}
MANIFESTS=${MANIFESTS:-/scratch/$USER/multi_user_qa/manifests_842_question_split}

LORA_RANK=${LORA_RANK:-128}
LORA_ALPHA=${LORA_ALPHA:-$LORA_RANK}
LR=${LR:-1e-4}
CLASS_WEIGHTS=${CLASS_WEIGHTS:-0}
RESUME=${RESUME:-}

if [ "$CLASS_WEIGHTS" = "1" ]; then CW_TAG=classw; CW_ARGS=(); else CW_TAG=noclassw; CW_ARGS=(--no-class-weights); fi
RESUME_ARGS=()
if [ -n "$RESUME" ]; then RESUME_ARGS=(--resume-from-checkpoint "$RESUME"); fi

RUN_NAME=groundedness_linattn_r${LORA_RANK}_a${LORA_ALPHA}_lr${LR}_${CW_TAG}_ga4_cosine_w005_job${SLURM_JOB_ID}
OUT=${OUT:-/scratch/$USER/multi_user_qa/runs/$RUN_NAME}
JOB_SCRATCH=/scratch/$USER/multi_user_qa/job_scratch/$SLURM_JOB_ID
mkdir -p "$OUT" "$JOB_SCRATCH"/{triton,tmp,hf,torch,xdg}

# Keep every cache and temp directory job-specific.
export TRITON_CACHE_DIR=$JOB_SCRATCH/triton
export TMPDIR=$JOB_SCRATCH/tmp
export HF_HOME=$JOB_SCRATCH/hf
export TORCH_HOME=$JOB_SCRATCH/torch
export XDG_CACHE_HOME=$JOB_SCRATCH/xdg
export HF_HUB_OFFLINE=1
export PATH="$ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export JUDGE_EVAL_MANIFEST=$MANIFESTS/test/train.jsonl

cd "$REPO"
{
  echo "slurm_job_id=$SLURM_JOB_ID"
  echo "node=$(hostname)"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(git status --porcelain -- training/judge_sft | wc -l)"
  echo "lora_rank=$LORA_RANK lora_alpha=$LORA_ALPHA lr=$LR class_weights=$CLASS_WEIGHTS resume=${RESUME:-none}"
} | tee "$OUT/run_info.txt"

"$ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m training.judge_sft.train_025fps_mb1 \
  --train-manifest "$MANIFESTS/train/train.jsonl" \
  --output-dir "$OUT/trainer" \
  --model-id "$MODEL" \
  --local-files-only \
  --attn-implementation sdpa \
  --tensor-parallel-size 2 \
  --task groundedness \
  --min-pixels 3136 \
  --max-pixels 119808 \
  --max-input-tokens 262144 \
  --image-context-target-fraction 0.85 \
  --lora-rank "$LORA_RANK" \
  --lora-alpha "$LORA_ALPHA" \
  --lora-dropout 0.05 \
  --trainable-decoder-layers 8 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
                        in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj \
  --learning-rate "$LR" \
  --weight-decay 0.01 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --adam-epsilon 1e-8 \
  --epochs 10 \
  --gradient-accumulation-steps 4 \
  --warmup-ratio 0.05 \
  --lr-scheduler-type cosine \
  "${CW_ARGS[@]}" \
  --max-grad-norm 1.0 \
  --seed 17 \
  --gradient-checkpointing \
  --dataloader-num-workers 2 \
  --dataloader-prefetch-factor 1 \
  --decoded-image-cache-entries 2 \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "$OUT/train.log"

echo "RUN=$OUT"
