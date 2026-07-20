# `qa_formality`-only 最小收敛实验 Torch Runbook

> 本文命令按顺序执行。新的 Torch 作业尚未运行；本地检查或历史 trace 回放不能写成远程训练成功。

## 0. 执行规则

1. Gate A 失败：停止，不提交 GPU 作业。
2. Smoke 失败：停止，不提交 40-step probe。
3. Probe 失败：保留全部产物，不继续调多个超参数。
4. Smoke 和 probe 都从同一个已通过 Gate 2 adapter 独立开始。
5. 不从 smoke、旧失败 Gate 3 或 Gate 3 v2 checkpoint 续训。

## 1. Windows PowerShell：准备远端目录并上传

先在 PowerShell 运行：

```powershell
ssh xl6775@torch-login-b-2 'mkdir -p /scratch/$USER/projects/EgoQA-two-user/training /scratch/$USER/projects/EgoQA-two-user/tests/training /scratch/$USER/projects/EgoQA-two-user/hpc /scratch/$USER/projects/EgoQA-two-user/docs/GRPO/v3/experiments/qa_formality_only_convergence_v1'
sftp xl6775@torch-login-b-2
```

进入交互式 SFTP 后整块粘贴：

```text
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user
cd /scratch/xl6775/projects/EgoQA-two-user
put training/grpo_v3_formality_reward.py training/
put training/grpo_v3_formality_replay.py training/
put training/grpo_v3_formality_convergence.py training/
put training/grpo_v3_formality_artifacts.py training/
put training/grpo_v3_reward_plugin.py training/
put tests/training/test_grpo_v3_formality_reward.py tests/training/
put tests/training/test_grpo_v3_formality_plugin.py tests/training/
put tests/training/test_grpo_v3_formality_replay.py tests/training/
put tests/training/test_grpo_v3_formality_convergence.py tests/training/
put tests/training/test_grpo_v3_formality_artifacts.py tests/training/
put tests/training/test_grpo_v3_formality_slurm.py tests/training/
put hpc/grpo_v3_formality_smoke.sbatch hpc/
put hpc/grpo_v3_formality_probe.sbatch hpc/
put docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/README_CN.md docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/
put docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/EXPERIMENT_DESIGN_CN.md docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/
put docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/TORCH_RUNBOOK_CN.md docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/
put docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/RESULT_INTERPRETATION_CN.md docs/GRPO/v3/experiments/qa_formality_only_convergence_v1/
bye
```

## 2. Torch 登录节点：固定环境

SSH 登录后整块粘贴：

```bash
export PROJECT_ROOT=/scratch/${USER}/projects/EgoQA-two-user
export GRPO_V3_ROOT=${PROJECT_ROOT}/outputs/grpo_v3
export TRAIN_ENV=/scratch/${USER}/envs/egoqa-ms-swift-v4.2.2-vllm024
export INFERENCE_ENV=/scratch/${USER}/envs/egoqa-grpo
export POLICY_MODEL=/scratch/${USER}/models/Qwen3-VL-2B-Instruct
export REVIEW_MODEL=/scratch/${USER}/models/Qwen3-VL-8B-Instruct
export PYTHONPATH=${PROJECT_ROOT}:${PYTHONPATH:-}
export PYTHON=${TRAIN_ENV}/bin/python
export GATE0_DIR=$(head -n 1 "${GRPO_V3_ROOT}/latest_gate0_output.txt")
export DATASET=${GATE0_DIR}/train_native_video.jsonl
cd ${PROJECT_ROOT}
```

检查路径：

```bash
for required in \
  "${PYTHON}" \
  "${TRAIN_ENV}/bin/swift" \
  "${INFERENCE_ENV}/bin/vllm" \
  "${POLICY_MODEL}" \
  "${REVIEW_MODEL}" \
  "${DATASET}" \
  training/grpo_v3_formality_reward.py \
  training/grpo_v3_formality_replay.py \
  training/grpo_v3_formality_convergence.py \
  training/grpo_v3_formality_artifacts.py \
  hpc/grpo_v3_formality_smoke.sbatch \
  hpc/grpo_v3_formality_probe.sbatch; do
  test -e "${required}" || { echo "缺少: ${required}"; exit 2; }
done
"${PYTHON}" -c 'import importlib.metadata as m; print(m.version("ms-swift")); assert m.version("ms-swift")=="4.2.2"'
```

## 3. Torch 登录节点：源码、测试和 Bash 预检

```bash
cd ${PROJECT_ROOT}
"${PYTHON}" -m compileall training tests/training
"${PYTHON}" -m unittest discover -s tests/training -p 'test_grpo_v3_formality_*.py' -v
"${PYTHON}" -m unittest discover -s tests/training -p 'test_grpo_v3_*.py' -v
bash -n hpc/grpo_v3_formality_smoke.sbatch
bash -n hpc/grpo_v3_formality_probe.sbatch
"${PYTHON}" -m training.grpo_v3_preflight --dataset "${DATASET}" --output /tmp/grpo_v3_formality_source_preflight.json
"${PYTHON}" -m json.tool /tmp/grpo_v3_formality_source_preflight.json
```

任何命令失败都先停止。不要提交 GPU 作业。

## 4. Gate A：历史 trace 离线回放

自动定位旧作业 `14194844` 的 trace：

```bash
export OLD_GATE3_TRACE=$(find "${GRPO_V3_ROOT}" -type f -path '*14194844*' -name reward_trace.jsonl -print -quit)
test -n "${OLD_GATE3_TRACE}"
test -s "${OLD_GATE3_TRACE}"
echo "OLD_GATE3_TRACE=${OLD_GATE3_TRACE}"
export FORMALITY_REPLAY_REPORT=${GRPO_V3_ROOT}/formality_replay_14194844.json
"${PYTHON}" -m training.grpo_v3_formality_replay \
  --trace "${OLD_GATE3_TRACE}" \
  --output "${FORMALITY_REPLAY_REPORT}"
"${PYTHON}" -m json.tool "${FORMALITY_REPLAY_REPORT}"
```

硬验收：

```bash
"${PYTHON}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"]=="passed"; assert d["input_row_count"]==80; assert d["finite_reward_count"]==78; assert d["missing_logprob_count"]==2; assert d["complete_group_count"]==18; assert d["positive_std_group_count"]==18; assert d["positive_std_ratio"]>=0.8; assert d["reward_components"]==["qa_formality_confidence"]; print({k:d[k] for k in ("status","input_sha256","finite_reward_count","missing_logprob_count","complete_group_count","positive_std_group_count","positive_std_ratio","reward_min","reward_max")})' "${FORMALITY_REPLAY_REPORT}"
```

只有该断言通过，才进入 smoke。

## 5. 提交 1-step Smoke

保持在同一个 SSH shell 中：

```bash
cd ${PROJECT_ROOT}
export FORMALITY_SMOKE_JOB_ID=$(sbatch --parsable hpc/grpo_v3_formality_smoke.sbatch)
echo "FORMALITY_SMOKE_JOB_ID=${FORMALITY_SMOKE_JOB_ID}"
squeue -j "${FORMALITY_SMOKE_JOB_ID}" -o '%.18i %.30j %.8T %.10M %.6D %R'
```

监控：

```bash
scontrol show job -dd "${FORMALITY_SMOKE_JOB_ID}" | grep -E 'JobState=|WorkDir=|StdOut=|StdErr=|Command='
tail -n 100 "${PROJECT_ROOT}/logs/grpo-v3-formality-smoke-${FORMALITY_SMOKE_JOB_ID}.out"
tail -n 100 "${PROJECT_ROOT}/logs/grpo-v3-formality-smoke-${FORMALITY_SMOKE_JOB_ID}.err"
tail -n 100 "${PROJECT_ROOT}/logs/grpo-v3-formality-smoke-review-${FORMALITY_SMOKE_JOB_ID}.log"
```

如果作业仍是 `PENDING/RUNNING`，等待后重复监控。日志暂时不存在不能直接推断作业失败，先以 `scontrol` 的真实路径为准。

## 6. Smoke 验收

作业结束后：

```bash
export FORMALITY_SMOKE_DIR=$(cat "${GRPO_V3_ROOT}/latest_formality_smoke_output.txt")
test -d "${FORMALITY_SMOKE_DIR}"
for file in formality_smoke_result.json run_manifest.json resolved_config.json adapter_reload.json reward_trace.jsonl; do
  test -s "${FORMALITY_SMOKE_DIR}/${file}" || { echo "缺少 ${file}"; exit 2; }
done
"${PYTHON}" -m json.tool "${FORMALITY_SMOKE_DIR}/formality_smoke_result.json"
"${PYTHON}" -m json.tool "${FORMALITY_SMOKE_DIR}/run_manifest.json"
"${PYTHON}" -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); r=json.load(open(d/"formality_smoke_result.json")); m=json.load(open(d/"run_manifest.json")); a=json.load(open(d/"adapter_reload.json")); assert r["status"]=="passed"; assert r["trace_count"]==4 and r["finite_reward_count"]==4; assert r["reward_std"]>0; assert r["global_step"]==1; assert m["reward_revision"]=="qa_formality_confidence_v1"; assert m["reward_components"]==["qa_formality_confidence"]; assert m["calls_video_reviewer"] is False; assert a["status"]=="passed" and a["processor_reloaded"] is True; print({"result":r,"adapter":a,"manifest_reward":m["reward_revision"]})' "${FORMALITY_SMOKE_DIR}"
```

只有全部断言通过，才提交 probe。

## 7. 提交独立 40-step Probe

Probe 脚本会重新读取同一个 Gate 2 adapter，不读取 smoke adapter：

```bash
cd ${PROJECT_ROOT}
export FORMALITY_PROBE_JOB_ID=$(sbatch --parsable hpc/grpo_v3_formality_probe.sbatch)
echo "FORMALITY_PROBE_JOB_ID=${FORMALITY_PROBE_JOB_ID}"
squeue -j "${FORMALITY_PROBE_JOB_ID}" -o '%.18i %.30j %.8T %.10M %.6D %R'
```

监控：

```bash
scontrol show job -dd "${FORMALITY_PROBE_JOB_ID}" | grep -E 'JobState=|WorkDir=|StdOut=|StdErr=|Command='
tail -n 100 "${PROJECT_ROOT}/logs/grpo-v3-formality-probe-${FORMALITY_PROBE_JOB_ID}.out"
tail -n 100 "${PROJECT_ROOT}/logs/grpo-v3-formality-probe-${FORMALITY_PROBE_JOB_ID}.err"
tail -n 100 "${PROJECT_ROOT}/logs/grpo-v3-formality-probe-review-${FORMALITY_PROBE_JOB_ID}.log"
```

## 8. Probe 验收

作业结束后：

```bash
export FORMALITY_PROBE_DIR=$(cat "${GRPO_V3_ROOT}/latest_formality_probe_output.txt")
test -d "${FORMALITY_PROBE_DIR}"
for file in formality_probe_result.json convergence_metrics.json run_manifest.json resolved_config.json adapter_reload.json reward_trace.jsonl; do
  test -s "${FORMALITY_PROBE_DIR}/${file}" || { echo "缺少 ${file}"; exit 2; }
done
"${PYTHON}" -m json.tool "${FORMALITY_PROBE_DIR}/convergence_metrics.json"
"${PYTHON}" -m json.tool "${FORMALITY_PROBE_DIR}/formality_probe_result.json"
"${PYTHON}" -c 'import json,sys; from pathlib import Path; d=Path(sys.argv[1]); c=json.load(open(d/"convergence_metrics.json")); r=json.load(open(d/"formality_probe_result.json")); m=json.load(open(d/"run_manifest.json")); a=json.load(open(d/"adapter_reload.json")); assert c["status"]=="passed"; assert c["group_count"]==40 and c["finite_reward_count"]==160; assert c["reward_delta"]>0 and c["reward_slope"]>0; assert c["positive_std_ratio"]>=0.8; assert c["late_unjudgeable_rate"]<=c["early_unjudgeable_rate"]; assert r["status"]=="passed" and r["global_step"]==40; assert m["reward_components"]==["qa_formality_confidence"] and m["calls_video_reviewer"] is False; assert a["status"]=="passed" and a["processor_reloaded"] is True; print({"convergence":c,"result":r,"adapter":a})' "${FORMALITY_PROBE_DIR}"
```

## 9. 失败时收集最小证据

不要改参数重提。先运行：

```bash
sacct -j "${FORMALITY_PROBE_JOB_ID}" --format=JobID,JobName%32,State,ExitCode,Elapsed,MaxRSS,AllocTRES%50
wc -l "${FORMALITY_PROBE_DIR}/reward_trace.jsonl" 2>/dev/null || true
for file in cpu_preflight.json formality_probe_result.json convergence_metrics.json adapter_reload.json run_manifest.json resolved_config.json; do
  test -s "${FORMALITY_PROBE_DIR}/${file}" && "${PYTHON}" -m json.tool "${FORMALITY_PROBE_DIR}/${file}"
done
tail -n 200 "${PROJECT_ROOT}/logs/grpo-v3-formality-probe-${FORMALITY_PROBE_JOB_ID}.out"
tail -n 200 "${PROJECT_ROOT}/logs/grpo-v3-formality-probe-${FORMALITY_PROBE_JOB_ID}.err"
tail -n 200 "${PROJECT_ROOT}/logs/grpo-v3-formality-probe-review-${FORMALITY_PROBE_JOB_ID}.log"
```

## 10. Windows PowerShell：下载产物

下面命令会下载本轮所有 formality smoke/probe 目录，避免手工改作业 ID：

```powershell
New-Item -ItemType Directory -Force -Path 'C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user\outputs\grpo_v3\formality_downloads'
sftp xl6775@torch-login-b-2
```

进入 SFTP 后粘贴：

```text
lcd C:/Users/20661/Desktop/Research/AR/multiuser/EgoQA-two-user/outputs/grpo_v3/formality_downloads
cd /scratch/xl6775/projects/EgoQA-two-user/outputs/grpo_v3
get formality_replay_14194844.json
get -r formality_smoke_*
get -r formality_probe_*
bye
```

下载完成后，先依据 [RESULT_INTERPRETATION_CN.md](./RESULT_INTERPRETATION_CN.md) 汇报，不得仅凭 loss、adapter 变化或单个正 reward 宣称收敛。
