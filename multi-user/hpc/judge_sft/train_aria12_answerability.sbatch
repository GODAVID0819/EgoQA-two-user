#!/bin/bash
#SBATCH --job-name=aria12_c_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G

set -euo pipefail

: "${JUDGE_PROJECT:?指定本次Torch实验目录}"
: "${JUDGE_REPO:?指定已提交版本的multi-user目录}"
: "${JUDGE_ENV:?指定验证过的Qwen训练环境}"
: "${JUDGE_MODEL:?指定Qwen3.8-27B基座目录}"

job_root="$JUDGE_PROJECT/jobs/$SLURM_JOB_ID"
job_cache="$job_root/cache"
input_root="$JUDGE_PROJECT/input"
mkdir -p "$job_root"
cd "$JUDGE_REPO"

# Linux套接字地址有长度上限。短别名仍解析到本次作业的scratch目录。
socket_root="/scratch/$USER/jt"
socket_tmp="$socket_root/$SLURM_JOB_ID"
mkdir -p "$job_cache/tmp" "$socket_root"
if [[ -e "$socket_tmp" || -L "$socket_tmp" ]]; then
  test "$(readlink -f "$socket_tmp")" = "$(readlink -f "$job_cache/tmp")"
else
  ln -s "$job_cache/tmp" "$socket_tmp"
fi

job_env=(
  "HOME=$job_cache/home"
  "XDG_CACHE_HOME=$job_cache/xdg"
  "HF_HOME=$job_cache/hf"
  "HF_DATASETS_CACHE=$job_cache/hf_datasets"
  "MODELSCOPE_CACHE=$job_cache/modelscope"
  "TORCH_HOME=$job_cache/torch"
  "TRITON_CACHE_DIR=$job_cache/triton"
  "TORCHINDUCTOR_CACHE_DIR=$job_cache/inductor"
  "VLLM_CACHE_ROOT=$job_cache/vllm"
  "CUDA_CACHE_PATH=$job_cache/cuda"
  "FLASHINFER_WORKSPACE_BASE=$job_cache/flashinfer"
  "TMPDIR=$socket_tmp" "TMP=$socket_tmp" "TEMP=$socket_tmp"
  "HF_HUB_OFFLINE=1" "TOKENIZERS_PARALLELISM=false" "PYTHONUNBUFFERED=1"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
)

env "${job_env[@]}" "$JUDGE_ENV/bin/python" -m training.torch_storage_preflight \
  --allowed-root "$job_root" --output "$job_root/storage_preflight.json"
env "${job_env[@]}" "$JUDGE_ENV/bin/python" -m training.judge_sft.check_worker_transport \
  --output "$job_root/worker_transport_check.json"

git rev-parse HEAD > "$job_root/git_commit.txt"
if [[ "${JUDGE_STAGE:-train}" == "transport" ]]; then
  exit 0
fi
if [[ "${JUDGE_STAGE:-train}" == "preflight" ]]; then
  env "${job_env[@]}" "$JUDGE_ENV/bin/python" -m training.judge_sft.check_aria12_inputs \
    --model "$JUDGE_MODEL" --input "$input_root" --output "$job_root/processor_preflight.json"
  exit 0
fi
: "${JUDGE_PREFLIGHT:?指定当前CPU预检产物}"
test -s "$JUDGE_PREFLIGHT"

nvidia-smi --query-gpu=timestamp,index,uuid,utilization.gpu,memory.used,memory.total \
  --format=csv,nounits -l 2 > "$job_root/gpu.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

env "${job_env[@]}" "$JUDGE_ENV/bin/torchrun" --standalone --nproc_per_node=2 \
  -m training.judge_sft.train_aria12 \
  --train-manifest "$input_root/train.jsonl" --test-manifest "$input_root/test.jsonl" \
  --model-id "$JUDGE_MODEL" --local-files-only --output-dir "$job_root/trainer" \
  --task answerability --epochs 5 --gradient-accumulation-steps 4 \
  --tensor-parallel-size 2 --attn-implementation sdpa \
  --min-pixels 3136 --max-pixels 119808 --max-input-tokens 262144 \
  --lora-rank 128 --lora-alpha 128 --lora-dropout 0.05 --trainable-decoder-layers 16 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
                        in_proj_qkv in_proj_z in_proj_a in_proj_b out_proj \
  --learning-rate 2e-5 --weight-decay 0.01 --warmup-ratio 0.05 --lr-scheduler-type cosine \
  --no-class-weights --gradient-checkpointing --seed 17 \
  --dataloader-num-workers 4 --dataloader-prefetch-factor 1 --decoded-image-cache-entries 3 \
  2>&1 | tee "$job_root/train.log"

echo "RUN=$job_root"
