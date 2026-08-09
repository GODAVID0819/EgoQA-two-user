# Reviewer 60/10 Evidence Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 Reviewer v1 直接读取新的 70-packet CSV，并以 evidence_id 严格构建 60 train / 10 validation / 0 locked-test 的正式合同。

**Architecture:** 保留现有三分类 heads、LoRA 和训练路径，只扩展 CSV 冗余字段兼容性与 split manifest 对零 locked-test 的表达。Slurm 入口、媒体准备和中文 Runbook 使用新 CSV、`split_60_10.json` 与独立 Reviewer 根目录。

**Tech Stack:** Python 3.11、`unittest`、PyTorch/Qwen3-VL 运行合同、Bash/Slurm、Markdown。

---

### Task 1: CSV 与 60/10 split 数据合同

**Files:**
- Modify: `tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_data.py`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/v1/data.py`

- [ ] **Step 1: 写失败测试**

新增测试覆盖：评分完整但没有 `fea_total_score` 时可解析；存在错误总分时仍拒绝；`build_split_manifest(..., 60, 10, 0)` 生成空 locked test；train/validation 非空且互斥；请求负 locked count 拒绝。

- [ ] **Step 2: 验证 RED**

Run:

```bash
python -m unittest tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_data -v
```

Expected: 缺失总分和 zero locked-test 测试失败。

- [ ] **Step 3: 最小实现**

将 `fea_total_score` 改为“若存在则校验”；train/validation count 要求正整数、locked count 允许零；manifest validation 根据 expected count 允许空 locked-test，并跳过空 split 的 class-support 检查。

- [ ] **Step 4: 验证 GREEN**

Run 同 Step 2，Expected: PASS。

### Task 2: 默认配置、audit 与空 split 评估保护

**Files:**
- Modify: `tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_config.py`
- Modify: `tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_audit.py`
- Modify: `tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_train.py`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/v1/config.py`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/v1/audit.py`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/v1/train.py`

- [ ] **Step 1: 写失败测试**

断言默认计数为 `(60, 10, 0)`；audit required evidence 为 70 并对两个非空 split 强制三类 support；选择空 locked-test 时抛出包含 split 名称的错误。

- [ ] **Step 2: 验证 RED**

```bash
python -m unittest \
  tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_config \
  tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_audit \
  tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_train -v
```

- [ ] **Step 3: 最小实现**

更新 config/audit CLI 默认值；full support 使用所有非空 split 的最小计数判断；`select_evidence` 对空请求明确失败，不回退到 validation。

- [ ] **Step 4: 验证 GREEN**

Run 同 Step 2，Expected: PASS。

### Task 3: Slurm 入口与独立仓库默认路径

**Files:**
- Modify: `tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_slurm.py`
- Modify: `hpc/grpo_v3/human_preference_reviewer/v1/common.sh`
- Modify: `hpc/grpo_v3/human_preference_reviewer/v1/train.sbatch`
- Modify: `hpc/grpo_v3/human_preference_reviewer/v1/evaluate.sbatch`
- Modify: `hpc/grpo_v3/human_preference_reviewer/v1/structure_probe.sbatch`
- Modify: `hpc/grpo_v3/human_preference_reviewer/v1/smoke1.sbatch`
- Modify: `hpc/grpo_v3/human_preference_reviewer/v1/overfit_probe.sbatch`
- Modify: `hpc/grpo_v3/human_preference_reviewer/stage0/*.sbatch`

- [ ] **Step 1: 写失败测试**

断言所有默认 project/log 路径使用 `EgoQA-two-user-reviewer-v1`；默认 CSV 为 `rlhf_candidate_scores_merged_70_packets.csv`；train/evaluate 使用 `split_60_10.json`；train 显式传入 60/10/0；evaluate 只允许 validation。

- [ ] **Step 2: 验证 RED**

```bash
python -m unittest tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_slurm -v
```

- [ ] **Step 3: 最小实现**

替换脚本默认路径和文件名，保留 Runbook 的 sbatch override 作为双层保护；evaluate 对非 validation 的 `EVAL_SPLIT` 在加载模型前失败。

- [ ] **Step 4: 验证 GREEN**

Run 同 Step 2，Expected: PASS。

### Task 4: 中文文档与 70-packet Torch Runbook

**Files:**
- Modify: `tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_runbook.py`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/README_CN.md`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/REVIEWER_STAGED_DESIGN_CN.md`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_STAGE0_CN.md`
- Modify: `training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_V1_CN.md`

- [ ] **Step 1: 写失败测试**

要求文档包含新 CSV、固定 SHA、420/70/60/10/0、`split_60_10.json`、140 个 required videos、validation-only 边界；禁止旧 CSV、正式 40/10/10 和本轮 locked-test 提交命令。

- [ ] **Step 2: 验证 RED**

```bash
python -m unittest tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_runbook -v
```

- [ ] **Step 3: 最小实现**

更新数据事实、SFTP、零 GPU audit、媒体清单/下载、manifest、训练、validation 和失败收集命令；移除已获批准的 `.sbatch。)` 中文句号；所有交互式命令保持当前 SSH 会话。

- [ ] **Step 4: 验证 GREEN**

Run 同 Step 2，Expected: PASS。

### Task 5: 使用真实 CSV 做本地合同审计并提交

**Files:**
- Read only: `C:/Users/20661/Documents/xwechat_files/wxid_i096w25uhusk22_e748/msg/file/2026-08/rlhf_candidate_scores_merged_70_packets.csv`

- [ ] **Step 1: 运行完整测试**

```bash
python -m unittest discover -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 -p 'test_*.py' -v
```

- [ ] **Step 2: 运行真实 CSV audit**

用 `annotation_audit_report(..., train_count=60, validation_count=10, locked_test_count=0)` 验证 SHA、420 行、70 eligible evidence、60/10/0/0 和 full class support。

- [ ] **Step 3: 静态检查**

运行 Python compile、`git diff --check`、错误路径/旧文件名/locked-test 命令扫描；本机若无法启动 Bash，则将 `bash -n` 明确保留为集群零 GPU Gate。

- [ ] **Step 4: 创建实现提交**

只暂存本计划涉及的源码、测试、Slurm 和文档，不提交 CSV、模型、媒体或输出产物。
