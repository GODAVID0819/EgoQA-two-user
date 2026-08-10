# 标注 Pareto-DPO Torch 同步 Runbook 设计

## 目标

把现有 Torch Runbook 补成从 Windows 本地仓库到 Torch Gate 0–5 的完整、可复制执行链，直接解决远端分支、项目目录、CSV、split、media map 和 DPO 数据均缺失的问题。

## 已批准方案

- 代码：Windows 将 `feature/annotated-pareto-dpo` 推送到 `origin`；Torch 从现有 `/scratch/xl6775/projects/EgoQA-two-user-reviewer-v1` 执行 `fetch`，再建立 `/scratch/xl6775/projects/EgoQA-two-user-annotated-pareto-dpo` 独立 worktree。
- 已被空输出目录占用的新路径先改名为带时间戳的备份，再建立 worktree，不删除原内容。
- 数据：只用 SFTP 上传单个标注 CSV。`split_60_10.json` 在 Torch 上由 Reviewer v1 审计 CLI 按 seed 42 生成；140 个视频与 `media_map.json` 由现有媒体准备作业生成。
- Gate 0 在 split 和 media map 验证通过后生成稳定的 `${DATA_DIR}/dpo`，后续 Gate 只消费该目录。
- 每个 Slurm 阶段必须保存真实 JobID，并用 `squeue`、`sacct`、JobID 输出目录和日志验收；不得使用 `latest_*`。
- 登录 shell 不使用会终止 SSH 会话的命令；失败只打印 `STOP` 或 `MISSING` 并跳过依赖步骤。

## 验收

Runbook 必须包含：本地 Git 推送、远端 worktree 创建、精确 SFTP、上传后检查、CSV SHA、420/70/60/10/0、140 个视频、split/media map 生成与校验、脚本语法检查、Gate 0–5 提交监控、JobID 失败证据收集，以及未执行 Gate 6 的边界。
