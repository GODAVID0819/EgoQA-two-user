# Reviewer Stage 0 Torch Runbook

本 Runbook 只验证 **Evidence Quality 单 head + 完全冻结 backbone**。它不注入 LoRA，也不训练 Answerability、Formality 或 Overall Utility。
所有训练作业内部都必须显式传入 `--stage stage0`，禁止依赖默认 Stage 2。

## 1. 本阶段能证明什么

- 两段真实视频和完整 QA 能进入 Qwen3-VL；
- multimodal representation 能送入 Evidence Quality 三分类 head；
- Evidence loss、backward、optimizer、checkpoint 链路工作；
- 只有 Evidence head 更新，backbone 保持冻结；
- 小数据上能观察到同一 candidate 的 loss 下降。

本阶段不能证明 Reviewer 已经能在 unseen evidence 上复现人类评分，也不能证明 LoRA 或三任务联合训练有效。

## 2. 登录与变量

```bash
ssh xl6775@greene.hpc.nyu.edu
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean
export PROJECT_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
export DATA_DIR=${PROJECT_ROOT}/data_RLHF/reviewer_v1
export CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_day5_7_full_100_HM.csv
export MEDIA_MAP=${DATA_DIR}/media_map.json
export MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/human_preference_reviewer
export TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
export PYTHON=${TRAIN_ENV}/bin/python
mkdir -p logs "${DATA_DIR}" "${OUTPUT_ROOT}"
```

代码只通过 Git 分支同步，避免递归 SFTP 把目录上传到错误层级：

```bash
git status --short
git fetch origin feature/multimodal-reviewer-training
git switch feature/multimodal-reviewer-training
git pull --ff-only origin feature/multimodal-reviewer-training
```

`git status --short` 非空时先停止，不覆盖合作者的本地修改。

CSV 使用单文件 SFTP 上传到明确远端目录：

```text
sftp xl6775@greene.hpc.nyu.edu
lcd C:/Users/20661/Documents/xwechat_files/wxid_i096w25uhusk22_e748/msg/file/2026-08
cd /scratch/xl6775/projects/EgoQA-two-user-grpo-clean/data_RLHF/reviewer_v1
put "rlhf_candidate_scores_day5_7_full_100_HM (1)(1).csv" rlhf_candidate_scores_day5_7_full_100_HM.csv
```

## 3. 零 GPU Gate

```bash
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" --output "${DATA_DIR}/annotation_audit_stage0.json" \
  --split-output "${DATA_DIR}/split_2_1_1.json" \
  --train-evidence-count 2 --validation-evidence-count 1 --locked-test-evidence-count 1
"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit media-map \
  --csv "${CSV_PATH}" --dataset-root /scratch/xl6775/datasets/EgoLife \
  --output "${MEDIA_MAP}"
"${PYTHON}" -m unittest discover \
  -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 -p 'test_*.py' -v
bash -n hpc/grpo_v3/human_preference_reviewer/stage0/common.sh
bash -n hpc/grpo_v3/human_preference_reviewer/stage0/structure_probe.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/stage0/smoke1.sbatch
bash -n hpc/grpo_v3/human_preference_reviewer/stage0/overfit_probe.sbatch
```

必须确认 `annotation_audit_stage0.json`、`split_2_1_1.json`、`media_map.json` 存在且非空。

## 4. Structure Gate

```bash
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
test -s "${SMOKE_DIR}/checkpoint/classification_heads.pt"
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
