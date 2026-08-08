# Stage 0B 受控单 Head 过拟合验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将 Stage 0 Overfit Gate 改为固定 probe set 上统一的训练前/训练后评估，严格拒绝单类预测塌缩。

**架构：** 扩展 `_evaluate()` 以返回 candidate 级观测，并新增纯 Python Gate 计算函数；训练入口在 optimizer step 前、每个 epoch 后和训练结束后评估同一组 train evidence。Slurm 只扩大诊断集合与训练轮数，模型结构和 Stage 1/2 保持不变。

**技术栈：** Python 3.11、PyTorch、`unittest`、Slurm、JSON。

---

### 任务 1：锁定受控 Gate 合同

**文件：**

- 修改：`tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_train.py`
- 修改：`training/grpo_v3/experiments/human_preference_reviewer/v1/train.py`

- [ ] **步骤 1：编写失败测试**

测试用合成 pre/post metrics 调用 `_controlled_overfit_gate()`，断言显著下降时通过、全预测 3 时失败、candidate IDs 不一致时抛出 `ValueError`。

- [ ] **步骤 2：运行测试并确认 RED**

运行：

```text
python -m unittest tests.training.grpo_v3.experiments.human_preference_reviewer.v1.test_train -v
```

预期：因 `_controlled_overfit_gate` 尚不存在而失败。

- [ ] **步骤 3：实现最小纯函数**

新增：

```python
def _controlled_overfit_gate(
    pre_metrics: Mapping[str, Any],
    post_metrics: Mapping[str, Any],
    *,
    minimum_loss_reduction: float = 0.30,
    minimum_improved_ratio: float = 0.80,
    minimum_accuracy_gain: float = 0.20,
    minimum_prediction_classes: int = 2,
) -> dict[str, Any]:
    ...
```

返回 thresholds、measurements、failures 和 `passed`。

- [ ] **步骤 4：运行测试并确认 GREEN**

运行任务 1 的单测，预期全部通过。

### 任务 2：采集统一 pre/post 和 epoch 指标

**文件：**

- 修改：`tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_train.py`
- 修改：`training/grpo_v3/experiments/human_preference_reviewer/v1/train.py`

- [ ] **步骤 1：编写失败测试**

测试 `_candidate_observation()` 输出 evidence ID、candidate ID、label、prediction、loss 和三类 probabilities，并测试 Gate 比较同一 candidate 集合。

- [ ] **步骤 2：运行并确认 RED**

预期因 candidate 观测 API 尚不存在而失败。

- [ ] **步骤 3：扩展 evaluation 与训练结果**

让 `_evaluate()` 在每个 active field 下增加 `candidate_results` 和 `prediction_counts`。在训练开始前计算 `pre_train_metrics`，每个完整 epoch 后追加 `epoch_probe_metrics`，训练后计算 `post_train_metrics`，并写出：

```python
{
    "probe_evidence_ids": [...],
    "probe_label_support": {...},
    "pre_train_metrics": {...},
    "post_train_metrics": {...},
    "epoch_probe_metrics": [...],
    "per_candidate_pre_post": [...],
    "controlled_overfit_gate": {...},
}
```

Stage 0 `fit` 若 Gate 未通过，仍保存完整诊断 JSON，但 `status` 写为 `failed_controlled_overfit_gate`，进程返回非零，由 Slurm 标记失败。

- [ ] **步骤 4：运行测试并确认 GREEN**

运行 train 与 evaluation 单测，预期通过。

### 任务 3：更新 Stage 0 Slurm 合同

**文件：**

- 修改：`tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_slurm.py`
- 修改：`hpc/grpo_v3/human_preference_reviewer/stage0/overfit_probe.sbatch`

- [ ] **步骤 1：编写失败静态测试**

断言 Overfit sbatch 使用 `split_4_1_1.json`、4 train evidence、20 epochs、480 max steps，并检查 `controlled_overfit_gate.passed`。

- [ ] **步骤 2：运行并确认 RED**

预期旧的 `split_2_1_1.json` 和 24 steps 使测试失败。

- [ ] **步骤 3：更新 sbatch**

改为：

```text
--split-manifest split_4_1_1.json
--max-steps 480
--epochs 20
--train-evidence-count 4
```

训练后用 Python 读取结果并断言 `controlled_overfit_gate.passed`。

- [ ] **步骤 4：运行并确认 GREEN**

运行 `test_slurm.py`，预期通过；再运行 `bash -n` 检查 sbatch。

### 任务 4：更新中文 Runbook 与登录 Shell 安全规则

**文件：**

- 修改：`tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_runbook.py`
- 修改：`training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_STAGE0_CN.md`

- [ ] **步骤 1：编写失败静态测试**

要求 Runbook 包含 `pre_train_metrics`、`post_train_metrics`、`controlled_overfit_gate`、`split_4_1_1.json`，并禁止 `exit`、`logout` 和直接粘贴段落中的 `set -euo pipefail`。

- [ ] **步骤 2：运行并确认 RED**

预期现有 Runbook 因旧 Gate 和 `exit 1` 失败。

- [ ] **步骤 3：更新 Runbook**

保留现有独立 worktree 同步内容，但将失败处理改为只输出 `STOP` 并通过条件块跳过后续命令。重写 Overfit 章节，明确这是假设内同集可学习性 Gate，不是 validation 或 locked-test 泛化评估。

- [ ] **步骤 4：运行并确认 GREEN**

运行 `test_runbook.py`，预期通过。

### 任务 5：完整验证与提交

**文件：**

- 检查：以上全部修改文件

- [ ] **步骤 1：运行 Reviewer 全部单测**

```text
python -m unittest discover -s tests/training/grpo_v3/experiments/human_preference_reviewer/v1 -p test_*.py -v
```

预期：零失败、零错误。

- [ ] **步骤 2：运行静态检查**

执行 Python 编译、三个 Stage 0 shell 的 `bash -n`（若本机无 Bash，明确记录为远端待验证）和 `git diff --check`。

- [ ] **步骤 3：审核范围**

确认 diff 不包含 LoRA、三-head Stage 2、正式 40/10/10 或 locked test 行为修改，也不覆盖用户现有 worktree 文档修改。

- [ ] **步骤 4：提交实现**

只暂存本计划涉及的最小文件并创建一个 Reviewer Stage 0B 实现提交；推送前报告本地验证与远端 H100 待验证边界。
