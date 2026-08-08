#!/usr/bin/env bash
set -euo pipefail

# Stage 0 复用 Reviewer v1 的 scratch-first、TorchCodec 和存储预检合同。
source "${PROJECT_ROOT}/hpc/grpo_v3/human_preference_reviewer/v1/common.sh"
export REVIEWER_STAGE=stage0
