# `qa_formality`-only 最小收敛实验入口

## 当前状态

- 本地状态：实现、专项测试、完整 GRPO v3 回归和历史 trace 回放完成后，以本目录 runbook 的“本地验证记录”更新为准。
- 远程状态：尚未执行新的 Torch smoke 或 40-step probe，当前不能声称 reward 已提升。
- 正式顺序：Gate A 历史 trace 回放 → 1-step smoke → 40-step probe。
- 严格门槛：前一阶段失败时不得绕过并提交下一阶段。

## 实验只回答什么

本实验只验证：现有原生双视频 ms-swift GRPO 链路，能否优化冻结 8B `qa_formality` judge 给出的连续 PASS 置信度 reward。

即使 probe 通过，也不说明 groundedness、answerability、综合 QA 质量或完整 Gate 3/Gate 4 已通过。

## 文件导航

- [EXPERIMENT_DESIGN_CN.md](./EXPERIMENT_DESIGN_CN.md)：已锁定的 reward、变量、失败边界和验收规格。
- [TORCH_RUNBOOK_CN.md](./TORCH_RUNBOOK_CN.md)：人工从 Windows 上传并在 Torch 逐块复制执行的命令。
- [RESULT_INTERPRETATION_CN.md](./RESULT_INTERPRETATION_CN.md)：结果能说明什么、不能说明什么。

## 实现入口

```text
training/grpo_v3_formality_reward.py
training/grpo_v3_formality_replay.py
training/grpo_v3_formality_convergence.py
training/grpo_v3_formality_artifacts.py
training/grpo_v3_reward_plugin.py
hpc/grpo_v3_formality_smoke.sbatch
hpc/grpo_v3_formality_probe.sbatch
```

训练 reward component 必须始终只有：

```text
qa_formality_confidence
```

连续 reward 为：

\[
r=
\frac{\operatorname{clip}
\left(\log P(\mathrm{PASS})-\log P(\mathrm{FAIL}),-32,32\right)}{32}
\]

