# One-pass NR/R 本地 worktree 精简设计

## 目标

将当前 `codex/onepass-nr-r-h200-20260909` worktree 精简为只服务 one-pass NR/R 后续优化的本地工作目录，同时保持当前实验代码、Git 历史和恢复能力。

## 采用方案

使用 Git sparse-checkout 从当前工作目录移除非核心 tracked 文件，不生成大批删除提交，不修改或推送远端分支。当前 Torch 作业继续以远端提交 `38968ca904a6a6177fd760de738c3a6c5dce0dbd` 为准。

## 保留范围

1. Git/worktree 元数据及 `AGENTS.md`。
2. NR、R、smoke 三个 Slurm 入口和公共 H200 wrapper。
3. 六用户 runtime 脚本及 `hpc/shared/cuda.py`。
4. one-pass evidence 构建、结果汇总和审核渲染工具。
5. `video_qa_loop.py`、`qwen3vl_runner.py`、CLI、one-pass 模块及其真实 Python 导入闭包。
6. `tests/test_one_pass_nr_r_h200_contract.py` 与 `tests/test_six_user_one_pass.py`。
7. `requirements/`。
8. 本设计文档和 sparse 路径清单。

## 移除范围

从当前工作目录移除与 one-pass NR/R 无关的历史数据、notebook、旧 benchmark、旧 ablation、人工审核文件、旧 Slurm 入口、无关测试、历史文档、缓存和空临时目录。`qa_mcq.jsonl` 与 `unused-*.jsonl` 不再出现在精简工作目录中，但仍保留在 Git 历史。

## 安全边界

- 不执行 `git reset --hard` 或 `git clean`。
- 不删除 `.git`，不移除 worktree 注册。
- 不修改、推送或同步 Torch 远端 checkout。
- 不释放或取消任何 Slurm 作业。
- sparse-checkout 可逆，恢复完整目录时使用 Git sparse-checkout disable。

## 验收

- sparse 应用后 `git status` 无代码删除记录。
- NR/R profile 可导入。
- 两个聚焦测试通过。
- 三个 `.sbatch` 与公共 wrapper 通过 Bash 语法检查。
- one-pass 工具通过 Python 编译检查。
- 保留文件不存在断裂的本地导入或脚本路径。
