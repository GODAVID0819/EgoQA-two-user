# 分阶段 Sweep 可观测性与恢复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在分阶段 DPO 训练失败时保留 Swift 返回码、checkpoint 清单和部分 checkpoint，并只重提三个失败的 epoch 2→3 链。

**架构：** `staged_train.sbatch` 将易失 scratch checkpoint 先复制到 JobID 专属的持久 `checkpoint.partial`，显式验证恢复所需文件和目标 step，通过后再原子提升为 `checkpoint`。新增 `submit_staged_recovery.sh` 从三个已完成 epoch 1 checkpoint 提交六个恢复任务，并写入独立 TSV manifest 与活动指针。

**技术栈：** Bash、Slurm、Python `unittest`、Markdown Runbook。

---

### 任务 1：测试先行锁定新合同

**文件：**
- 修改：`tests/training/grpo_v3/experiments/annotated_preference/test_slurm.py`
- 修改：`tests/training/grpo_v3/experiments/annotated_preference/test_runbook.py`

- [ ] 增加失败测试，要求 runner 写出 `swift_exit_code.txt`、`checkpoint_inventory.txt`、`checkpoint.partial`，并显式报告缺失文件。
- [ ] 增加失败测试，要求恢复提交器使用三个 epoch 1 JobID、`afterok`、`sbatch --parsable` 和独立 `jobs.tsv`。
- [ ] 运行两个测试文件，确认因功能尚未实现而失败。

### 任务 2：实现持久诊断与恢复提交器

**文件：**
- 修改：`hpc/grpo_v3/annotated_preference/staged_train.sbatch`
- 新建：`hpc/grpo_v3/annotated_preference/submit_staged_recovery.sh`

- [ ] 捕获 Swift 返回码而不立即丢失 scratch 证据。
- [ ] 将目标 checkpoint 复制为 `checkpoint.partial`，写出文件清单并显式验证四个恢复文件。
- [ ] 验证 `global_step` 后将 partial 提升为正式 checkpoint；Swift 非零或合同失败仍让作业失败。
- [ ] 提交三条 epoch 2→3 恢复链，并自动持久化六个新 JobID。
- [ ] 运行针对性测试，确认通过。

### 任务 3：补充 Runbook 与完整验证

**文件：**
- 修改：`training/grpo_v3/experiments/annotated_preference/TORCH_RUNBOOK_CN.md`

- [ ] 增加 Git 同步、Torch 预检、一次性恢复提交和无需手填 JobID 的监控块。
- [ ] 运行 `bash -n`、相关 `unittest`、`git diff --check`。
- [ ] 推送分支后给出 Torch 快进和恢复提交命令；远端 Job 完成仍作为未验证边界。
