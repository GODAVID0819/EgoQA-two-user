#!/bin/bash
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/vqa_eval/logs/%x_%j.out
# SETTING=all6|evidence2|evidence_providers|speaker1|text_only  MODEL=HF id  MODEL_TAG=输出前缀  [LIMIT=n] [SHUFFLE=seed] [MAX_LEN] [BATCH] [IDS=grpo_frames/subsets/ 下的子集名]
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
JS=/scratch/tx856/js/vqag_${SLURM_JOB_ID}; mkdir -p $JS
export TMPDIR=$JS XDG_CACHE_HOME=$JS/cache VLLM_CACHE_ROOT=$JS/vllm TRITON_CACHE_DIR=$JS/triton
export HF_HUB_CACHE=/scratch/tx856/multi_user_qa/vqa_eval/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
cd /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames
OUT=outputs/${MODEL_TAG}_${SETTING}${SHUFFLE:+_shuf${SHUFFLE}}${IDS:+_${IDS}}${LIMIT:+_smoke${LIMIT}}_${SLURM_JOB_ID}
python vqa_grpo_frames.py --model "$MODEL" --setting "$SETTING" --out "$OUT" ${LIMIT:+--limit $LIMIT} ${SHUFFLE:+--shuffle-seed $SHUFFLE} ${MAX_LEN:+--max-model-len $MAX_LEN} ${BATCH:+--batch $BATCH} ${IDS:+--ids-file /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames/subsets/$IDS.json}
