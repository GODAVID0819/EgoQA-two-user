# Combined-Video Answer-Margin Convergence v1

本实验只验证一个问题：固定 `temperature=0.5` 时，双原生视频、Qwen3-VL-2B BF16 LoRA 的 GRPO 能否在 40 个 optimizer step 内提高冻结答题器的五选一 answer margin。

固定条件：`evidence_id=EGOLIFE2U_DAY2_11350000_A1_A5`、每组 4 个生成、学习率 `1e-5` constant、`beta=0`、`top_p=1`、LoRA `q_proj/v_proj`、`r=8`、`alpha=16`，冻结 ViT 与 aligner。所有训练 Gate 都从 `gate2_14119442/checkpoint-1` 重新开始。

执行顺序不可跳过：本地契约 → scorer-only probe → 8×4 calibration → 1-step → 5-step → 40-step → step 0/40 固定评估（32 seeds，64 行）。本地测试通过不等于 Torch 成功。

最终状态只有三类：

- `passed`：运行完整且八项预注册门槛全部通过；
- `not_converged`：64 行证据完整，但至少一项研究门槛未通过；
- `invalid`：产物、数量、环境、父 checkpoint、非有限值或 adapter reload 等完整性失败。

执行命令见 [TORCH_RUNBOOK_CN.md](TORCH_RUNBOOK_CN.md)，结论口径见 [RESULT_INTERPRETATION_CN.md](RESULT_INTERPRETATION_CN.md)。
