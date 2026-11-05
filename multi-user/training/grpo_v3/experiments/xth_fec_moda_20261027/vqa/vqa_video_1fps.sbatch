#!/bin/bash
#SBATCH --account=torch_pr_111_general
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/vqa_eval/logs/%x_%j.out
# DATA=题目 vqa JSON  TAG=输出名  [SETTINGS=speaker1,evidence_providers,all6] [IDS=grpo_frames/subsets/ 下子集名] [SHUFFLE=0] [LIMIT=n]
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
JS=/scratch/tx856/js/vqa1fps_${SLURM_JOB_ID}; mkdir -p $JS
export TMPDIR=$JS XDG_CACHE_HOME=$JS/cache VLLM_CACHE_ROOT=$JS/vllm TRITON_CACHE_DIR=$JS/triton
export HF_HUB_CACHE=/scratch/tx856/multi_user_qa/vqa_eval/hf_cache TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
export FORCE_QWENVL_VIDEO_READER=decord DECORD_EOF_RETRY_MAX=65536
cd /scratch/tx856/multi_user_qa/vqa_eval
OUT=outputs/qwen3vl8b_video1fps320_${TAG}${SHUFFLE:+_shuf${SHUFFLE}}${LIMIT:+_smoke${LIMIT}}_${SLURM_JOB_ID}
python vqa_video_1fps.py --model Qwen/Qwen3-VL-8B-Instruct --data "$DATA" --settings "${SETTINGS:-speaker1,evidence_providers,all6}" --out "$OUT" \
  ${IDS:+--ids-file /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames/subsets/$IDS.json} ${SHUFFLE:+--shuffle-seed $SHUFFLE} ${LIMIT:+--limit $LIMIT}
