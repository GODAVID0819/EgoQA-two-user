#!/bin/bash
# 建立 generalanswer 运行根：复制已物化的运行根（不含 outputs/logs/configs），只把 multi-user/prompts.py 换成 prompts_generalanswer.py，
# 再用 rebuild_generator_rows.py 重建 720 条生成器行。不改动原运行根，CPU 作业运行（要读全部帧文件）。
#   sbatch --cpus-per-task=8 --mem=32G --time=02:00:00 --wrap "bash make_ga_root.sh <ROOT> <GA_ROOT>"
set -euo pipefail
ROOT=${1:?ROOT required}; GA=${2:?GA_ROOT required}
HERE=$(cd "$(dirname "$0")" && pwd)
P=/scratch/tx856/.conda/envs/grpo-swift/bin/python
mkdir -p "$GA/configs" "$GA/outputs" "$GA/logs"
rsync -a --exclude /outputs --exclude /logs --exclude /configs --exclude '__pycache__' --exclude '/data/*_v2.jsonl' "$ROOT/" "$GA/"
cp "$HERE/prompts_generalanswer.py" "$GA/multi-user/prompts.py"
echo "{\"multi-user/prompts.py\": \"prompts_generalanswer.py\", \"sha256\": \"$(sha256sum "$HERE/prompts_generalanswer.py" | cut -d' ' -f1)\"}" > "$GA/PROMPT_OVERRIDE.json"
export CUDA_VISIBLE_DEVICES= PYTHONPATH=$GA/runtime_deps_train:$GA/multi-user:$GA PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
cd "$GA"
"$P" rebuild_generator_rows.py --input data/train_length_paired.jsonl --output data/train_length_paired_ga.jsonl
