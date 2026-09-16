# Reviewer Stage 0 Torch Runbook

本 Runbook 只验证 **Evidence Quality 单 head + 完全冻结 backbone**。它不注入 LoRA，也不训练 Answerability、Formality 或 Overall Utility。
所有训练作业内部都必须显式传入 `--stage stage0`，禁止依赖默认 Stage 2。

## 1. 本阶段能证明什么

- 两段真实视频和完整 QA 能进入 Qwen3-VL；
- multimodal representation 能送入 Evidence Quality binary `[fail, pass]` head；
- Evidence loss、backward、optimizer、checkpoint 链路工作；
- 只有 Evidence head 更新，backbone 保持冻结；
- 小数据上能观察到同一 candidate 的 loss 下降。

本阶段不能证明 Reviewer 已经能在 unseen evidence 上复现人类评分，也不能证明 LoRA 或三任务联合训练有效。

## 2. 登录与变量

```bash
ssh hm2991@login.torch.hpc.nyu.edu
cd /scratch/hm2991/Long-video-understanding-clip/egolife_two_user_qa/score-only-reward-model
export REVIEWER_ROOT=$(pwd -P)
export HPC_ROOT=${REVIEWER_ROOT}
export CODE_ROOT=${REVIEWER_ROOT}
export DATA_DIR=${CODE_ROOT}/data_RLHF/reviewer_v1
export CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv
export MEDIA_MAP=${DATA_DIR}/media_map.json
export MODEL_DIR=/scratch/hm2991/models/Qwen3-VL-8B-Instruct
export OUTPUT_ROOT=${CODE_ROOT}/outputs/human_preference_reviewer/v1
export TRAIN_ENV=/scratch/hm2991/conda/envs/qwen3vl-smoke
export FFMPEG_ENV=${TRAIN_ENV}
export PYTHON=${TRAIN_ENV}/bin/python
export PACKAGE_ROOT=$(cd "${REVIEWER_ROOT}/.." && pwd -P)
export REWARD_OUTPUT_ROOT=${PACKAGE_ROOT}/outputs/reward_candidate_collection_qwen36_27b
export PYTHONPATH="${CODE_ROOT}:${HPC_ROOT}:${PYTHONPATH:-}"
mkdir -p "${HPC_ROOT}/hpc/logs" "${DATA_DIR}" "${OUTPUT_ROOT}"
cd "${CODE_ROOT}"
```

本轮直接从完整的隔离副本运行；`training/` 与 `hpc/` 都保留在 `score-only-reward-model/` 内，`tests/` 不参与远端训练。

```text
local score-only-reward-model/... -> /scratch/hm2991/Long-video-understanding-clip/egolife_two_user_qa/score-only-reward-model/...
```

先在登录节点创建窄目录：

```bash
test -d "${REVIEWER_ROOT}/training/grpo_v3/experiments/human_preference_reviewer/v1"
test -d "${REVIEWER_ROOT}/hpc/grpo_v3/human_preference_reviewer/stage0"
mkdir -p "${REVIEWER_ROOT}/hpc/logs" "${DATA_DIR}" "${OUTPUT_ROOT}"
```

从 Windows 用 SFTP 同步 Stage 0 的窄代码集合和单个 CSV：

```text
sftp hm2991@dtn.torch.hpc.nyu.edu
lcd "C:/Users/haoya/OneDrive/Desktop/Long context video understanding ARVR/Long-video-understanding/egolife_two_user_qa/score-only-reward-model"
cd /scratch/hm2991/Long-video-understanding-clip/egolife_two_user_qa/score-only-reward-model
put "data_RLHF/reviewer_v1/rlhf_candidate_scores_merged_70_packets.csv" data_RLHF/reviewer_v1/rlhf_candidate_scores_merged_70_packets.csv
bye
```

## 3. 零 GPU Gate

```bash
cd "${CODE_ROOT}"
PREPARE_JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/v1/prepare.sbatch)
PREPARE_JOB=${PREPARE_JOB_RAW%%;*}
sacct -j "${PREPARE_JOB}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
test -s "${DATA_DIR}/split_2_1_1.json"
test -s "${MEDIA_MAP}"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/stage0/common.sh"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/stage0/structure_probe.sbatch"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/stage0/smoke1.sbatch"
bash -n "${HPC_ROOT}/hpc/grpo_v3/human_preference_reviewer/stage0/overfit_probe.sbatch"
```

必须确认 `annotation_audit.json`、`split_train_60_validation_10.json`、`split_2_1_1.json` 与 `media_map.json` 存在且非空。

## 4. Structure Gate

```bash
cd "${HPC_ROOT}"
JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/stage0/structure_probe.sbatch)
JOB_ID=${JOB_RAW%%;*}
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
STRUCTURE_DIR=${OUTPUT_ROOT}/stage0_structure_${JOB_ID}
test -s "${STRUCTURE_DIR}/structure_probe.json"
```

要求 shared stack 为 36 层，目标结构包含 blocks 34、35 的 `q_proj` 和 `v_proj`。Stage 0 不注入这些 LoRA target，Structure 只确认后续 Stage 1/2 可用。

## 5. 单步 Smoke Gate

```bash
JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/stage0/smoke1.sbatch)
JOB_ID=${JOB_RAW%%;*}
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
SMOKE_DIR=${OUTPUT_ROOT}/stage0_smoke_${JOB_ID}
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; assert r["stage"]=="stage0"; assert r["active_heads"]==["evidence_quality"]; assert r["head_parameter_delta_nonzero"]; assert not r["lora_parameter_delta_nonzero"]; print(r["parameter_audit"])' "${SMOKE_DIR}/training_result.json"
test -s "${SMOKE_DIR}/checkpoint/binary_heads.pt"
test ! -e "${SMOKE_DIR}/checkpoint/lora_adapter.pt"
test -s "${SMOKE_DIR}/storage_preflight.json"
```

## 6. Overfit Gate

```bash
JOB_RAW=$(sbatch --parsable hpc/grpo_v3/human_preference_reviewer/stage0/overfit_probe.sbatch)
JOB_ID=${JOB_RAW%%;*}
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
OVERFIT_DIR=${OUTPUT_ROOT}/stage0_overfit_${JOB_ID}
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); d=r["repeated_candidate_loss"]; n=sum(x["improved"] for x in d.values()); print({"repeated":len(d),"improved":n,"throughput":r["throughput"]}); assert len(d)>=6 and n/len(d)>=0.75' "${OVERFIT_DIR}/training_result.json"
```

只有 Structure、Smoke、Overfit 三个 Gate 全部通过，才进入 Stage 1。不要在 Stage 0 结果上运行正式 locked test。

## 7. 失败收集

```bash
scontrol show job -dd "${JOB_ID}"
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
test -s "${OUTPUT_ROOT}/stage0_smoke_${JOB_ID}/dependencies.txt" || true
test -s "${OUTPUT_ROOT}/stage0_smoke_${JOB_ID}/storage_preflight.json" || true
```

报告 JobID、`.out/.err`、`dependencies.txt`、`storage_preflight.json`、`parameter_audit.json` 和 `training_result.json`；不要使用 `latest_*` 路径代替 JobID 产物。
