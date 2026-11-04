set -e
ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
STATE=$(sacct -X -n -j 21448178 -o State | head -1 | xargs)
if [ "$STATE" = RUNNING ]; then
nohup srun --jobid=21448178 --overlap --exact --ntasks=1 --cpus-per-task=4 --gres=gpu:h200:1 --cpu-bind=none bash -c '
set -euo pipefail
MAIN_ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
export VQA_ROOT="$MAIN_ROOT/vision_worker"
export VQA_ENV_BIN=/scratch/xl6775/envs/minicpmv45_grpo131_20261031/bin
export VQA_FFMPEG_BIN=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/bin
export VQA_FFMPEG_LIB=/scratch/xl6775/envs/egoqa-ffmpeg-runtime/lib
export FRAME_CACHE=/scratch/xl6775/experiments/vqa_videollama3_grpo131_20261030/frames_156
cd "$MAIN_ROOT/code"
source runtime_env.sh
export HF_HUB_OFFLINE=1 CUDA_MODULE_LOADING=LAZY PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MINICPM_PRIOR_CACHE
python -u preencode_tail.py "$MAIN_ROOT" "$SLURM_JOB_ID"
' > "$ROOT/logs/tail_preencode_21448178.out" 2>&1 < /dev/null &
printf 'TAIL_PREENCODE_LAUNCHED pid=%s job=21448178 new_gpu_allocation=0 same_allocated_h200=1\n' "$!"
else
printf 'TAIL_PREENCODE_SKIPPED terminal_state=%s\n' "$STATE"
fi
