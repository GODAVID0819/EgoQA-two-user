#!/bin/bash
#SBATCH --job-name=xth_resume_audit
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/tx856/multi_user_qa/lxr_run/logs/%x_%j.out
# 续训前 CPU 核验断点（validate_run.finite_updated_adapter 写 checkpoint/reload_cpu_audit.json），
# 再用 recovery.validated_resume 核对续训配置。需要 RUN_CONFIG（含 resume_from_checkpoint 等字段）。
set -euo pipefail
: "${RUN_CONFIG:?RUN_CONFIG is required}"
ROOT=/scratch/tx856/multi_user_qa/lxr_run
export PYTHONPATH=$ROOT/runtime_deps_train:$ROOT/multi-user:$ROOT PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
cd $ROOT/multi-user
/scratch/tx856/.conda/envs/grpo-swift/bin/python - "$RUN_CONFIG" <<'PY'
import json, sys
from pathlib import Path
from training.grpo_v3.six_user_binary.validate_run import finite_updated_adapter
from recovery import validated_resume
c = json.loads(Path(sys.argv[1]).read_text())
checkpoint = Path(c['resume_from_checkpoint'])
finite_updated_adapter(checkpoint / 'adapter_model.safetensors')
print('RELOAD_CPU_AUDIT', (checkpoint / 'reload_cpu_audit.json').read_text()[:400])
print('VALIDATED_RESUME', json.dumps(validated_resume(c)))
PY
