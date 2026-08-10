# 标注 Pareto-DPO Torch 同步 Runbook 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让用户从当前 Windows 分支开始，按文档逐段复制即可完成 Git、SFTP、数据准备、验证和 Gate 0–5 提交。

**Architecture:** Git 负责代码与提交身份，SFTP 只负责单个 CSV，小型 manifest 和媒体映射在 Torch 上确定性生成。新实验使用独立 worktree，所有训练产物按 Slurm JobID 隔离。

**Tech Stack:** Git、PowerShell、OpenSSH SFTP、Bash、Python、Slurm、ms-swift。

---

### Task 1：锁定同步与数据准备合同

**Files:**
- Modify: `tests/training/grpo_v3/experiments/annotated_preference/test_runbook.py`

- [ ] 增加一组最小字符串合同，覆盖 `git push`、`git fetch`、`git worktree add`、精确 `sftp/lcd/cd/put`、上传后 `bash -n`/`git status`、split 生成、140 视频与 media map 验证。
- [ ] 运行 `python -m unittest tests.training.grpo_v3.experiments.annotated_preference.test_runbook -v`，确认因 Runbook 尚未补全而失败。

### Task 2：补全 Runbook 并验证

**Files:**
- Modify: `training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md`

- [ ] 按“Windows 推送 → Torch worktree → SFTP CSV → split → media → Gate 0–5”的顺序写入完整命令。
- [ ] 保留登录 shell 会话安全、CSV SHA、`60/10/0`、JobID 产物和 Gate 6 边界。
- [ ] 运行定向 Runbook 测试、annotated_preference 测试、`compileall` 和 `git diff --check`。
