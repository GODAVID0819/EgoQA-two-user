#!/bin/bash
# 按 lxr README 的“独立根”布局物化运行目录：仓库 multi-user 与 hpc 依赖 + 本实验 runtime/ 覆盖。
# 只同步源码；ROOT 下已有的 configs/ data/ encoder/ history_pool/ runtime_deps_train/ outputs/ 等运行产物不动。
# 有作业在读 ROOT 时不要重跑（会改动正在运行的源码）。
#   bash materialize_root.sh /scratch/tx856/multi_user_qa/lxr_run
set -euo pipefail
ROOT=${1:?ROOT required}
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
if [ -n "$(git -C "$REPO" status --porcelain -- multi-user hpc)" ]; then
  echo "工作区有未提交改动，先提交再物化" >&2; exit 1
fi
mkdir -p "$ROOT"
rsync -a --delete --exclude training/grpo_v3/experiments --exclude __pycache__ "$REPO/multi-user/" "$ROOT/multi-user/"
rsync -a --delete --exclude __pycache__ "$REPO/hpc/" "$ROOT/hpc/"
# runtime 覆盖：顶层模块、prompts、multi-user/training/grpo_v3/six_user_binary、hpc/shared
rsync -a --exclude __pycache__ --exclude 'multi-user/' --exclude 'hpc/' "$HERE/runtime/" "$ROOT/"
rsync -a --delete --exclude __pycache__ "$HERE/runtime/multi-user/training/grpo_v3/six_user_binary/" "$ROOT/multi-user/training/grpo_v3/six_user_binary/"
rsync -a --exclude __pycache__ "$HERE/runtime/hpc/" "$ROOT/hpc/"
echo "{\"commit\": \"$(git -C "$REPO" rev-parse HEAD)\", \"experiment\": \"${HERE#$REPO/}\", \"materialized_at\": \"$(date -Is)\"}" > "$ROOT/SOURCE_COMMIT.json"
cat "$ROOT/SOURCE_COMMIT.json"
