# Reviewer Stage 0 Torch Runbook

本手册服从 `docs/TORCH_EXPERIMENT_META_RULES_CN.md`。

本 Runbook 只验证 **Evidence Quality 单 head + 完全冻结 backbone**。它不注入 LoRA，也不训练 Answerability、Formality 或 Overall Utility。
所有训练作业内部都必须显式传入 `--stage stage0`，禁止依赖默认 Stage 2。

## 1. 本阶段能证明什么

- 两段真实视频和完整 QA 能进入 Qwen3-VL；
- multimodal representation 能送入 Evidence Quality 三分类 head；
- Evidence loss、backward、optimizer、checkpoint 链路工作；
- 只有 Evidence head 更新，backbone 保持冻结；
- 在同一固定 probe set 上统一比较训练前后指标，证明 Evidence head 的平均 CE 明显下降且预测不塌缩为单一等级。

本阶段不能证明 Reviewer 已经能在 unseen evidence 上复现人类评分，也不能证明 LoRA 或三任务联合训练有效。

## 2. 登录与变量

当前 `EgoQA-two-user-grpo-clean` 可能保留 QA/GRPO 的未提交实验修改。不要在该目录中直接切换 Reviewer 分支，也不要把 Reviewer 远端分支 `pull`、`merge` 或 `rebase` 到当前 GRPO 分支。Reviewer 使用独立 Git worktree：

```bash
ssh xl6775@greene.hpc.nyu.edu
```

确认终端提示符已经位于 Torch 登录节点后，再执行下面整段命令；不要在登录前把两段一起粘贴：

```bash
export SOURCE_ROOT=/scratch/xl6775/projects/EgoQA-two-user-grpo-clean
export REVIEWER_ROOT=/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1
export REVIEWER_BRANCH=feature/multimodal-reviewer-training
export REVIEWER_READY=1

if [[ ! -d "${SOURCE_ROOT}" ]]; then
  echo "STOP: SOURCE_ROOT 不存在：${SOURCE_ROOT}"
  REVIEWER_READY=0
else
  cd "${SOURCE_ROOT}"
  git worktree list
  git fetch origin "${REVIEWER_BRANCH}"
fi

if [[ "${REVIEWER_READY}" == 1 && ! -e "${REVIEWER_ROOT}" ]]; then
  if git show-ref --verify --quiet "refs/heads/${REVIEWER_BRANCH}"; then
    git worktree add "${REVIEWER_ROOT}" "${REVIEWER_BRANCH}"
  else
    git worktree add -b "${REVIEWER_BRANCH}" \
      "${REVIEWER_ROOT}" "origin/${REVIEWER_BRANCH}"
  fi
fi

if [[ "${REVIEWER_READY}" == 1 ]]; then
  ACTUAL_ROOT=$(git -C "${REVIEWER_ROOT}" rev-parse --show-toplevel 2>/dev/null)
  ACTUAL_BRANCH=$(git -C "${REVIEWER_ROOT}" branch --show-current 2>/dev/null)
  if [[ "${ACTUAL_ROOT}" != "${REVIEWER_ROOT}" || "${ACTUAL_BRANCH}" != "${REVIEWER_BRANCH}" ]]; then
    echo "STOP: Reviewer worktree 路径或分支不符合预期"
    REVIEWER_READY=0
  else
    LOCAL_EXCLUDE=$(git -C "${REVIEWER_ROOT}" rev-parse --git-path info/exclude)
    mkdir -p "$(dirname "${LOCAL_EXCLUDE}")"
    for PATTERN in "/data_RLHF/" "/outputs/" "/logs/"; do
      if ! grep -Fqx "${PATTERN}" "${LOCAL_EXCLUDE}" 2>/dev/null; then
        printf '%s\n' "${PATTERN}" >> "${LOCAL_EXCLUDE}"
        echo "ADDED_LOCAL_EXCLUDE: ${PATTERN}"
      fi
    done
    if ! git -C "${REVIEWER_ROOT}" diff --quiet \
      || ! git -C "${REVIEWER_ROOT}" diff --cached --quiet; then
      echo "STOP: Reviewer worktree 存在已跟踪或已暂存修改，禁止自动同步"
      git -C "${REVIEWER_ROOT}" status --short --untracked-files=no
      REVIEWER_READY=0
    fi
  fi
fi

if [[ "${REVIEWER_READY}" == 1 ]]; then
  cd "${REVIEWER_ROOT}"
  if git fetch origin "${REVIEWER_BRANCH}" && \
     git merge --ff-only "origin/${REVIEWER_BRANCH}"; then
    LOCAL_HEAD=$(git rev-parse HEAD)
    REMOTE_HEAD=$(git rev-parse "origin/${REVIEWER_BRANCH}")
    echo "local : ${LOCAL_HEAD}"
    echo "remote: ${REMOTE_HEAD}"
    if [[ "${LOCAL_HEAD}" != "${REMOTE_HEAD}" ]]; then
      echo "STOP: Reviewer 本地 HEAD 与远端分支不一致"
      REVIEWER_READY=0
    fi
  else
    echo "STOP: Reviewer 分支 fetch 或 fast-forward 失败"
    REVIEWER_READY=0
  fi
fi

if [[ "${REVIEWER_READY}" == 1 ]]; then
  export PROJECT_ROOT=${REVIEWER_ROOT}
  export DATA_DIR=${PROJECT_ROOT}/data_RLHF/reviewer_v1
  export CSV_PATH=${DATA_DIR}/rlhf_candidate_scores_merged_70_packets.csv
  export MEDIA_MAP=${DATA_DIR}/media_map.json
  export MODEL_DIR=/scratch/xl6775/models/Qwen3-VL-8B-Instruct
  export OUTPUT_ROOT=${PROJECT_ROOT}/outputs/human_preference_reviewer
  export TRAIN_ENV=/scratch/xl6775/envs/egoqa-ms-swift-v4.2.2-vllm024
  export PYTHON=${TRAIN_ENV}/bin/python
  mkdir -p logs "${DATA_DIR}" "${OUTPUT_ROOT}"
else
  echo "STOP: 未设置 Reviewer 训练变量，请先处理上面的 worktree 问题。"
fi
```

验收条件：当前目录为 `/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1`，当前分支为 `feature/multimodal-reviewer-training`，`LOCAL_HEAD` 与 `REMOTE_HEAD` 完全一致，且已跟踪与已暂存代码均无修改。`data_RLHF/`、`outputs/`、`logs/` 由 repository-local exclude 管理，不阻止同步；原 `EgoQA-two-user-grpo-clean` 目录及其本地修改不会被移动、stash 或覆盖。

如果命令报告本地 Reviewer 分支已被其他 worktree 使用，先运行 `git worktree list` 找到并复用已有目录；不要删除目录，也不要使用 `git worktree add --force`。

CSV 使用单文件 SFTP 上传到明确远端目录：

```text
sftp xl6775@greene.hpc.nyu.edu
lcd C:/Users/20661/Documents/xwechat_files/wxid_i096w25uhusk22_e748/msg/file/2026-08
cd /scratch/xl6775/projects/EgoQA-two-user-reviewer-v1/data_RLHF/reviewer_v1
put "rlhf_candidate_scores_merged_70_packets.csv" rlhf_candidate_scores_merged_70_packets.csv
```

上传后应得到 SHA-256：

```text
32679019FD7C665A0632E9885405BDF13C77B51386EFC56E7B29B24192210CD7
```

## 3. 零 GPU Gate

```bash
cd "${PROJECT_ROOT}"

"${PYTHON}" -m training.grpo_v3.experiments.human_preference_reviewer.v1.audit annotation-csv \
  --csv "${CSV_PATH}" --output "${DATA_DIR}/annotation_audit_stage0.json" \
  --split-output "${DATA_DIR}/split_4_1_1.json" \
  --train-evidence-count 4 --validation-evidence-count 1 --locked-test-evidence-count 1
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

`unittest discover` 必须从 `${PROJECT_ROOT}` 使用上面的相对 `-s` 路径运行。不要额外指定 top-level directory：当前 `tests/` 目录不是 Python package，否则会触发 `Start directory is not importable`。

必须确认 `annotation_audit_stage0.json`、`split_4_1_1.json`、`media_map.json` 存在且非空。`split_4_1_1.json` 的 train label support 必须覆盖 Evidence Quality 的 1、2、3 三个等级。

## 4. Structure Gate

```bash
JOB_RAW=$(sbatch \
  --parsable \
  --export=ALL \
  --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-s0-structure-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-s0-structure-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/stage0/structure_probe.sbatch)
JOB_ID=${JOB_RAW%%;*}
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
STRUCTURE_DIR=${OUTPUT_ROOT}/stage0_structure_${JOB_ID}
test -s "${STRUCTURE_DIR}/structure_probe.json"
```

要求 shared stack 为 36 层，目标结构包含 blocks 34、35 的 `q_proj` 和 `v_proj`。Stage 0 不注入这些 LoRA target，Structure 只确认后续 Stage 1/2 可用。

## 5. 单步 Smoke Gate

```bash
JOB_RAW=$(sbatch \
  --parsable \
  --export=ALL \
  --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-s0-smoke-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-s0-smoke-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/stage0/smoke1.sbatch)
JOB_ID=${JOB_RAW%%;*}
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
SMOKE_DIR=${OUTPUT_ROOT}/stage0_smoke_${JOB_ID}
"${PYTHON}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["status"]=="passed"; assert r["stage"]=="stage0"; assert r["active_heads"]==["evidence_quality"]; assert r["head_parameter_delta_nonzero"]; assert not r["lora_parameter_delta_nonzero"]; print(r["parameter_audit"])' "${SMOKE_DIR}/training_result.json"
test -s "${SMOKE_DIR}/checkpoint/classification_heads.pt"
test ! -e "${SMOKE_DIR}/checkpoint/lora_adapter.pt"
test -s "${SMOKE_DIR}/storage_preflight.json"
```

## 6. Overfit Gate

这个 Gate 在同一固定 probe set 上比较训练前和训练后结果，只证明单 head 可被训练，不代表 validation、locked test 或 unseen evidence 泛化能力。

```bash
JOB_RAW=$(sbatch \
  --parsable \
  --export=ALL \
  --chdir="${PROJECT_ROOT}" \
  --output="${PROJECT_ROOT}/logs/reviewer-s0-overfit-%j.out" \
  --error="${PROJECT_ROOT}/logs/reviewer-s0-overfit-%j.err" \
  hpc/grpo_v3/human_preference_reviewer/stage0/overfit_probe.sbatch)
JOB_ID=${JOB_RAW%%;*}
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS
OVERFIT_DIR=${OUTPUT_ROOT}/stage0_overfit_${JOB_ID}
"${PYTHON}" - "${OVERFIT_DIR}/training_result.json" <<'PY'
import json
import sys
from pathlib import Path

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
gate = result["controlled_overfit_gate"]
print("probe_evidence_ids=", result["probe_evidence_ids"])
print("probe_label_support=", result["probe_label_support"])
print("pre_train_metrics=", result["pre_train_metrics"]["evidence_quality"])
print("post_train_metrics=", result["post_train_metrics"]["evidence_quality"])
print("controlled_overfit_gate=", gate)
assert gate["passed"], gate
PY
```

Gate 同时要求：平均 CE 相对下降至少 30%、至少 80% candidate 的 loss 下降、accuracy 至少提升 20 个百分点，并且训练后预测至少覆盖两个等级。`repeated_candidate_loss` 仅作为辅助诊断，不再决定通过。

只有 Structure、Smoke、Overfit 三个 Gate 全部通过，才进入 Stage 1。不要在 Stage 0 结果上运行正式 locked test。

## 7. 失败收集

```bash
scontrol show job -dd "${JOB_ID}"
sacct -j "${JOB_ID}" -o JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,MaxVMSize
test -s "${OUTPUT_ROOT}/stage0_smoke_${JOB_ID}/dependencies.txt" || true
test -s "${OUTPUT_ROOT}/stage0_smoke_${JOB_ID}/storage_preflight.json" || true
```

报告 JobID、`.out/.err`、`dependencies.txt`、`storage_preflight.json`、`parameter_audit.json` 和 `training_result.json`；不要使用 `latest_*` 路径代替 JobID 产物。
