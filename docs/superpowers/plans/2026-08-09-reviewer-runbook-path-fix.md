# Reviewer Runbook 路径修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 统一 Reviewer Runbook 的独立仓库路径，并确保 targeted tests 使用可运行的 `unittest discover` 命令。

**方案：** 不改变测试包结构，也不修改训练代码。先扩展 Runbook 静态测试，让旧 `grpo-clean` 路由和 `unittest -t` 组合明确失败；随后只修订两份中文 Runbook，并用完整 Reviewer 测试集验证。

**技术栈：** Markdown、Python `unittest`、Git。

---

### 任务 1：建立失败回归测试

**文件：**
- 修改：`tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_runbook.py`

- [ ] 断言 Stage 0 Runbook 使用相对 start directory，且不包含 `-t "${ROOT}"`。
- [ ] 断言 Reviewer v1 Runbook 使用 `/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1`，且不把代码、数据、日志或输出路由到 `EgoQA-two-user-grpo-clean`。
- [ ] 运行 `test_runbook.py`，确认测试因现有旧路径而失败。

### 任务 2：最小修正文档

**文件：**
- 修改：`training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_STAGE0_CN.md`
- 修改：`training/grpo_v3/experiments/human_preference_reviewer/TORCH_RUNBOOK_V1_CN.md`

- [ ] 在 Stage 0 Runbook 明确从 `${PROJECT_ROOT}` 运行测试，不传 `-t`。
- [ ] 将完整 v1 Runbook 的工作根、SFTP、数据、日志和输出路径统一到独立 Reviewer 仓库。
- [ ] 删除会结束交互式 SFTP 会话的命令。
- [ ] 保留 `EgoQA-two-user-grpo-clean` 仅作为“不要改动的旧工作区”说明，不作为任何运行目标。

### 任务 3：验证与提交

**文件：**
- 测试：`tests/training/grpo_v3/experiments/human_preference_reviewer/v1/test_runbook.py`

- [ ] 运行 Runbook 定向测试并确认通过。
- [ ] 运行完整 Reviewer v1 测试集。
- [ ] 搜索残留的错误 `-t` 和旧运行路径。
- [ ] 运行 `git diff --check`，检查最终 diff，然后创建单一修复提交。
