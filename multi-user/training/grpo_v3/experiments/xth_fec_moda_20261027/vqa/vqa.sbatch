#!/bin/bash
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/vqa_eval/logs/%x_%j.out
# SETTING=all6|evidence2|speaker1|speaker_plus1  [LIMIT=n 做 smoke] [FRAMES=每段视频帧数，默认64] [SHUFFLE=选项重排seed] [MODEL=HF id，默认 Qwen2.5-VL-7B] [MODEL_TAG=输出目录前缀] [MAX_LEN=vLLM max_model_len，默认32768] [BATCH=每批题数，默认8] [IDS=grpo_frames/subsets/ 下的子集名]
set -euo pipefail
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/tx856/.conda/envs/grpo-swift
JS=/scratch/tx856/js/vqa_${SLURM_JOB_ID}; mkdir -p $JS
export TMPDIR=$JS XDG_CACHE_HOME=$JS/cache VLLM_CACHE_ROOT=$JS/vllm TRITON_CACHE_DIR=$JS/triton
export HF_HUB_CACHE=/scratch/tx856/multi_user_qa/vqa_eval/hf_cache TOKENIZERS_PARALLELISM=false
export FORCE_QWENVL_VIDEO_READER=decord DECORD_EOF_RETRY_MAX=65536  # 个别视频尾帧 decord 默认重试 10240 次不够
cd /scratch/tx856/multi_user_qa/vqa_eval
MODEL=${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
export HF_HUB_OFFLINE=1
OUT=outputs/${MODEL_TAG:+${MODEL_TAG}_}${SETTING}${FRAMES:+_f${FRAMES}}${SHUFFLE:+_shuf${SHUFFLE}}${IDS:+_${IDS}}${LIMIT:+_smoke${LIMIT}}_${SLURM_JOB_ID}
python vqa_qwen25vl.py --model "$MODEL" --setting "$SETTING" --out "$OUT" ${LIMIT:+--limit $LIMIT} ${FRAMES:+--frames $FRAMES} ${SHUFFLE:+--shuffle-seed $SHUFFLE} ${MAX_LEN:+--max-model-len $MAX_LEN} ${BATCH:+--batch $BATCH} ${IDS:+--ids-file /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames/subsets/$IDS.json}
