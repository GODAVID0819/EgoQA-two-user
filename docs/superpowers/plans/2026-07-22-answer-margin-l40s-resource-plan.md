# Answer-Margin L40S Resource Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 Answer-Margin 六个 Gate 的默认 GPU 改为 L40S 48GB，并提供有证据的高规格 GPU 升级流程。

**Architecture:** `.sbatch` 负责请求 L40S、设置适当时限并在模型加载前记录 GPU；正式实验设计定义资源边界；Runbook 提供查询、提交、OOM 诊断和单变量升级命令。现有 scorer 逻辑不属于本计划。

**Tech Stack:** Slurm、Bash、NVIDIA SMI、Python `unittest`、Markdown。

---

### Task 1: 资源静态契约

**Files:**
- Modify: `tests/training/test_grpo_v3_answer_margin_slurm.py`

- [ ] 将单卡 probe 的期望改为 `gpu:l40s:1`，其余 Gate 改为 `gpu:l40s:2`。
- [ ] 断言所有实际执行体在 `MODEL_LOAD_BOUNDARY` 前写出 `gpu_environment.csv`。
- [ ] 运行 Slurm 测试并确认旧脚本先失败。

### Task 2: Slurm 默认资源

**Files:**
- Modify: `hpc/grpo_v3_answer_margin_scorer_probe.sbatch`
- Modify: `hpc/grpo_v3_answer_margin_calibration.sbatch`
- Modify: `hpc/grpo_v3_answer_margin_smoke1.sbatch`
- Modify: `hpc/grpo_v3_answer_margin_smoke5.sbatch`
- Modify: `hpc/grpo_v3_answer_margin_probe40.sbatch`
- Modify: `hpc/grpo_v3_answer_margin_fixed_eval.sbatch`

- [ ] 将 H100 GRES 改为相同数量的 L40S。
- [ ] 将 walltime 调整为 `02:00/04:00/04:00/08:00/18:00/12:00`。
- [ ] 在模型加载前用 `nvidia-smi` 保存实际 GPU 证据。
- [ ] 运行静态测试并确认通过。

### Task 3: 正式文档和失败升级流程

**Files:**
- Modify: `docs/GRPO/v3/experiments/combined_video_answer_margin_convergence_v1/EXPERIMENT_DESIGN_CN.md`
- Modify: `docs/GRPO/v3/experiments/combined_video_answer_margin_convergence_v1/TORCH_RUNBOOK_CN.md`

- [ ] 把正式默认资源从 H100 改为 L40S 48GB。
- [ ] 增加 `sinfo`、L40S 提交、OOM 证据采集、A100 80GB/H100 升级命令。
- [ ] 明确非 OOM 基础设施错误不得通过升级 GPU 绕过。
- [ ] 执行占位符扫描、`git diff --check` 和工作树审计。
