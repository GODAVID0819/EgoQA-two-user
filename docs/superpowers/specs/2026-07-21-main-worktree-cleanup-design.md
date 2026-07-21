# 主 Worktree 保守整理设计

## 1. 目标

在不修改两个附加 worktree、不删除实验产物、不触碰远程仓库的前提下，将主 worktree 中混杂的源码、测试、文档和 Slurm 脚本整理为主题清晰、能够验证、便于回退的提交，并使主 worktree 最终达到干净状态。

主 worktree 路径：

`C:\Users\20661\Desktop\Research\AR\multiuser\EgoQA-two-user`

## 2. 当前边界

本次只处理主 worktree。以下对象保持原样：

- `codex/formality-fixed-eval` 分支及其 worktree；
- `feat/grpo-ready-p1` 分支及其 worktree；
- 执行前已经存在的四个 stash；
- 远程仓库及本地远程跟踪引用。

本次不执行 `push`、`pull`、`fetch`、`git reset --hard`、`git clean` 或强制删除 worktree。

## 3. 生成产物处理

在仓库根目录 `.gitignore` 中加入：

```gitignore
/outputs/
/analysis_outputs/
/tmp/
```

处理约束：

- `outputs/` 和 `tmp/` 中已有内容继续保留在磁盘；
- 当前本地已经删除的 7 个已跟踪 `analysis_outputs/` 文件，在生成产物清理提交中正式从当前版本移除；
- 被移除的已跟踪分析产物仍保留在 Git 历史中，可按提交恢复；
- 后续重新生成的三个目录内容不再进入常规 Git 状态。

## 4. 文件组织原则

保留现有目录职责，不进行与清理目标无关的大规模改名或移动：

- `training/`：训练、验证和分析实现；
- `tests/training/`：对应自动化测试；
- `hpc/`：Slurm 作业入口；
- `requirements/`：实验依赖；
- `docs/GRPO/`：运行说明、实验设计、结果分析和汇报材料；
- `docs/superpowers/`：已经完成的设计与实施计划。

每个主题提交应尽量包含完整闭环：实现、对应测试、Slurm 入口、依赖和必要文档。若文件跨越多个主题，以运行时依赖闭环和可独立验证为优先，不机械拆分。

## 5. 主题提交设计

预计按以下顺序整理；实际文件归属可依据依赖检查微调，但不得把生成产物混入功能提交。

### 5.1 生成产物规则

建议提交信息：`chore: ignore generated experiment artifacts`

内容：

- 更新 `.gitignore`；
- 提交 `analysis_outputs/` 中 7 个已跟踪文件的删除状态；
- 不删除 `outputs/` 和 `tmp/` 的磁盘内容。

### 5.2 GRPO v2 LoRA 工作流

建议提交信息：`feat: add GRPO v2 LoRA training workflow`

内容：

- `training/grpo_v2_*`；
- `tests/training/test_grpo_v2_*`；
- `hpc/grpo_v2_*`；
- `docs/GRPO/v2/`。

### 5.3 原生视频 GRPO v3 与 Gate

建议提交信息：`feat: add native-video GRPO v3 training gates`

内容：

- v3 contract、data、split、preflight、Gate 数据构建和验证；
- ms-swift Gate 0–4 Slurm 脚本；
- 对应测试；
- ms-swift 依赖；
- 原生视频策略和 Torch runbook。

### 5.4 Adapter 与 Greedy Evaluation

建议提交信息：`feat: add GRPO v3 adapter and greedy evaluation`

内容：

- adapter reload；
- greedy evaluation 与 paired comparison；
- 对应测试和 Slurm 脚本；
- Gate 3 结果分析材料。

### 5.5 Formality Convergence 与固定端点评估

建议提交信息：`feat: add GRPO v3 formality convergence evaluation`

内容：

- formality artifact、convergence 和 fixed endpoint evaluation；
- probe、smoke 和 fixed-eval Slurm 脚本；
- 对应测试；
- 实验说明和运行手册。

### 5.6 实验分析与汇报材料

建议提交信息：`docs: add GRPO v3 experiment analyses and reports`

内容：

- 实验设计与结果分析；
- 汇报 Markdown 和演示文件；
- 与已经实现内容对应的历史设计、实施计划。

### 5.7 Torch 通用路径与元数据规则

建议提交信息：`chore: align Torch experiment paths and metadata`

内容：

- 通用 Slurm 路径和环境调整；
- `qwen3vl_runner.py` 的相关改动；
- Torch 实验元数据规则；
- 能覆盖这些改动的测试或静态检查。

## 6. Fixed-eval 重叠策略

`codex/formality-fixed-eval` 的提交 `c41e13d` 涉及 6 个文件。当前主 worktree 中有 2 个文件与该提交完全一致，4 个文件内容不同，表明主 worktree 版本还包含后续修改。

本次不直接 cherry-pick `c41e13d`，避免覆盖主 worktree 中较新的内容。处理方式为：

1. 比较 4 个不同文件的语义差异；
2. 以主 worktree 当前内容作为候选版本；
3. 运行 fixed-eval 对应测试验证候选版本；
4. 将验证后的文件纳入主线 formality 主题提交；
5. 不修改、不删除另一个 worktree 或其分支。

## 7. 安全与失败处理

- 不使用笼统的 `git add .`；每次只暂存当前主题的明确路径；
- 每次提交前检查暂存文件名单和 `git diff --cached --stat`；
- 不使用强制重置、强制清理或强制删除 worktree；
- 不删除现有 stash；
- 不删除三个生成目录的磁盘内容；
- 测试失败时，先区分既有失败、依赖缺失和当前主题回归；
- 若文件无法可靠归类，停止暂存该文件并保留在工作区，不能为追求干净状态而静默删除；
- 若发现另一个 worktree 状态变化，立即停止并核对影响范围。

## 8. 验证标准

完成后必须验证：

1. 主 worktree 的 `git status --short --branch` 不再显示未提交源码、测试、文档或 Slurm 文件；
2. `outputs/`、`analysis_outputs/` 和 `tmp/` 被 Git 正确忽略；
3. `outputs/` 和 `tmp/` 的已有磁盘内容仍然存在；
4. 与改动对应的训练测试通过；
5. Python 文件通过基础编译检查；
6. `git diff --check` 通过；
7. 两个附加 worktree 的路径、分支、提交和未提交状态与执行前一致；
8. 执行前已有的四个 stash 保持不变；
9. 未修改远程仓库和远程跟踪引用。

若测试因本地缺失可选依赖而不能运行，最终报告必须列出准确命令、错误和未完成的验证边界，不能宣称测试通过。
