set -e
ROOT=/scratch/xl6775/experiments/vqa_minicpmv45_grpo131_20261031
STATE=$(sacct -X -n -j 21448178 -o State | head -1 | xargs)
test "$STATE" = RUNNING
nohup srun --jobid=21448178 --overlap --exact --ntasks=1 --cpus-per-task=8 --gres=none --cpu-bind=none python3 -u "$ROOT/code/warm_frames.py" 21448178 > "$ROOT/logs/cpu_prefetch_21448178.out" 2>&1 < /dev/null &
PREFETCH_SRUN_PID=$!
printf 'CPU_PREFETCH_LAUNCHED pid=%s job=21448178 gpu=0 existing_cpus=8\n' "$PREFETCH_SRUN_PID"
