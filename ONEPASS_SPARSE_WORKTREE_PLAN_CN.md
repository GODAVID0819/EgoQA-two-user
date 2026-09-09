# One-pass NR/R worktree 精简实施计划

> **执行要求：** 使用 `executing-plans` 按任务顺序执行并在验证点停留。本侧线程禁止调用子代理。

**目标：** 使用可逆的 Git sparse-checkout 将当前 worktree 精简为 one-pass NR/R 最小可运行目录，不产生大批 Git 删除，不影响 Torch 远端作业。

**架构：** 将经过静态导入闭包和运行脚本路径审计的保留文件写入版本化清单 `ONEPASS_SPARSE_PATHS.txt`，再以 non-cone sparse-checkout 应用该清单。实现不修改 Python 或 Slurm 行为；完整仓库内容继续存在于共享 Git 对象库和历史提交中。

**技术栈：** Git sparse-checkout、PowerShell、Bash 语法检查、Python `py_compile`、pytest。

---

### 任务 1：创建精确的 sparse 保留清单

**文件：**

- 创建：`ONEPASS_SPARSE_PATHS.txt`
- 保留：`ONEPASS_SPARSE_WORKTREE_DESIGN_CN.md`
- 保留：`ONEPASS_SPARSE_WORKTREE_PLAN_CN.md`

- [ ] **步骤 1：再次确认 worktree 身份和修改面**

运行：

```powershell
git rev-parse --show-toplevel
git branch --show-current
git status --short --branch
```

预期：根目录是 `EgoQA-two-user-onepass-nr-r-h200-20260909`，分支为 `codex/onepass-nr-r-h200-20260909`，除本计划文件外没有意外修改。

- [ ] **步骤 2：写入完整保留清单**

使用 `apply_patch` 创建 `ONEPASS_SPARSE_PATHS.txt`，内容必须准确为：

```text
/.gitattributes
/.gitignore
/AGENTS.md
/ONEPASS_SPARSE_PATHS.txt
/ONEPASS_SPARSE_WORKTREE_DESIGN_CN.md
/ONEPASS_SPARSE_WORKTREE_PLAN_CN.md
/__init__.py
/__main__.py
/candidate_mining.py
/cli.py
/clip_exclusive_mining.py
/clip_gap_demo.py
/evidence.py
/evidence_chunk_review.py
/gaze_projection.py
/generation_ablation.py
/group_relative_clip_sampling.py
/io_utils.py
/manifest.py
/object_hints.py
/object_reid.py
/observations.py
/one_pass_evidence.py
/one_pass_summary.py
/paired_evidence_pruning.py
/prompts.py
/pruning_ablation.py
/pruning_k_grid.py
/qa_generation_schedule.py
/qa_pipeline.py
/qwen3vl_runner.py
/repair_pruned_pair_media.py
/review_media.py
/schema.py
/six_video_qa_tester.py
/six_view_packet_prep.py
/small_video_model_runner.py
/temporal_kmeans_grid_sidecar.py
/video_qa_loop.py
/requirements/
/tools/build_six_user_one_pass_evidence.py
/tools/render_six_user_10min_review.py
/tools/summarize_six_user_one_pass.py
/tests/test_one_pass_nr_r_h200_contract.py
/tests/test_six_user_one_pass.py
/hpc/shared/cuda.py
/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
/hpc/qa/experiments/run_six_user_qa_one_pass_ab_h200_common.sh
/hpc/qa/experiments/run_six_user_qa_one_pass_nr_h200.sbatch
/hpc/qa/experiments/run_six_user_qa_one_pass_r_h200.sbatch
/hpc/qa/experiments/run_six_user_qa_one_pass_smoke_h200.sbatch
```

- [ ] **步骤 3：验证清单覆盖全部显式运行路径**

运行只读检查，确认公共 wrapper、runtime、三个入口、三个工具、keeper、36 个 Python 闭包文件和两个测试均在清单中；预期无 `MISSING` 输出。

- [ ] **步骤 4：提交清单与实施计划**

运行：

```powershell
git add -- ONEPASS_SPARSE_PATHS.txt ONEPASS_SPARSE_WORKTREE_PLAN_CN.md
git diff --cached --check
git commit -m "chore: 定义 one-pass 精简 worktree 路径"
```

预期：只提交清单和计划，不包含 Python、Slurm 或实验产物变化。

### 任务 2：在当前 worktree 应用 sparse-checkout

**文件：**

- 修改 Git worktree 私有配置：由 `git sparse-checkout` 管理
- 不修改 tracked Python、Slurm 和测试文件

- [ ] **步骤 1：记录应用前状态**

运行：

```powershell
git status --short --branch
git rev-parse HEAD
git worktree list --porcelain
```

预期：状态干净；当前 worktree 注册仍存在。记录当前本地 HEAD，远端实验仍保持在 `38968ca904a6a6177fd760de738c3a6c5dce0dbd`，二者不得混写为同一版本。

- [ ] **步骤 2：初始化 non-cone sparse-checkout**

运行：

```powershell
git sparse-checkout init --no-cone
Get-Content -LiteralPath ONEPASS_SPARSE_PATHS.txt | git sparse-checkout set --no-cone --stdin
```

预期：非清单文件从当前 worktree 消失；命令不产生 tracked 删除记录。

- [ ] **步骤 3：验证被移除和被保留的代表路径**

必须确认：

```text
PRESENT video_qa_loop.py
PRESENT one_pass_evidence.py
PRESENT hpc/qa/experiments/run_six_user_qa_one_pass_nr_h200.sbatch
PRESENT hpc/qa/experiments/run_six_user_qa_one_pass_r_h200.sbatch
PRESENT hpc/qa/experiments/run_six_user_qa_one_pass_smoke_h200.sbatch
PRESENT tests/test_one_pass_nr_r_h200_contract.py
ABSENT qa_mcq.jsonl
ABSENT notebooks
ABSENT ablation_manual_review_cluster_frames.updated.csv
ABSENT unused-accepted.jsonl
```

- [ ] **步骤 4：确认 Git 没有把 sparse 移除误记为删除**

运行：

```powershell
git status --short --branch
git diff --name-status
```

预期：无 tracked 文件删除或修改。

### 任务 3：验证精简目录仍可运行 one-pass NR/R

**文件：**

- 测试：`tests/test_one_pass_nr_r_h200_contract.py`
- 测试：`tests/test_six_user_one_pass.py`
- 验证：三个 `.sbatch`、公共 wrapper、三个 Python 工具

- [ ] **步骤 1：运行 Python 导入检查**

从当前 worktree 根目录运行：

```powershell
python -c "import sys,types; from pathlib import Path; root=Path.cwd(); package=types.ModuleType('egolife_two_user_qa'); package.__path__=[str(root)]; sys.modules['egolife_two_user_qa']=package; import egolife_two_user_qa.video_qa_loop as m; print(callable(m.six_user_one_pass_profiles), callable(m.six_user_one_pass_nr_profiles))"
```

预期输出为 `True True`。该命令显式使用与聚焦测试一致的 `egolife_two_user_qa` alias，不依赖当前 worktree 目录名可以作为 Python 包名。

- [ ] **步骤 2：运行聚焦测试**

运行：

```powershell
pytest -q tests/test_one_pass_nr_r_h200_contract.py tests/test_six_user_one_pass.py
```

预期：全部通过；测试数量以实际收集结果为准，不虚构固定数量。

- [ ] **步骤 3：运行 Python 编译检查**

运行：

```powershell
python -m py_compile one_pass_evidence.py one_pass_summary.py tools/build_six_user_one_pass_evidence.py tools/render_six_user_10min_review.py tools/summarize_six_user_one_pass.py
```

预期：退出码为 0。

- [ ] **步骤 4：运行 Bash 语法检查**

使用本机可用的 Bash 对以下文件执行 `bash -n`：

```text
hpc/qa/experiments/run_six_user_qa_one_pass_ab_h200_common.sh
hpc/qa/experiments/run_six_user_qa_one_pass_nr_h200.sbatch
hpc/qa/experiments/run_six_user_qa_one_pass_r_h200.sbatch
hpc/qa/experiments/run_six_user_qa_one_pass_smoke_h200.sbatch
hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
```

预期：所有文件退出码为 0。

- [ ] **步骤 5：最终状态检查**

运行：

```powershell
git status --short --branch
git sparse-checkout list
```

预期：分支仅因本地计划/清单提交领先远端，不存在代码修改、删除或未跟踪缓存。若测试生成缓存，只删除明确的 `__pycache__` 和 `.pytest_cache` 后再次检查。

### 任务 4：记录回滚方法和交付边界

**文件：**

- 不创建额外文件；回滚命令已经保存在本计划中

- [ ] **步骤 1：验证恢复命令的语义但不实际恢复**

恢复完整 worktree 的命令为：

```powershell
git sparse-checkout disable
```

本次只记录命令，不执行恢复。

- [ ] **步骤 2：汇报精简结果**

最终汇报必须包含：本地 HEAD、保留路径数量、被移除的代表文件、测试结果、Git 状态、恢复命令，以及“未修改 Torch 远端、未操作 Slurm Job”的边界。
