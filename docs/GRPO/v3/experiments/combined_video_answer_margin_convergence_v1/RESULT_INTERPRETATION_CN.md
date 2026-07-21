# 结果解释口径

## 三态结论

- `passed`：64/64 固定评估完整，且八项门槛全部通过。只能说明该固定 answer-margin 条件在本次最小验证中收敛，不能外推真实 QA 综合质量、groundedness、information gap 或其他 reward。
- `not_converged`：运行与证据完整，但数值门槛未全部通过。这是有效负结果，保留全部样本，不得删除不利 seed 后改写主结论。
- `invalid`：完整性或基础设施门槛失败，不能讨论收敛方向；修复后从覆盖失败边界的最小 Gate 重跑。

## 八项预注册门槛

1. step 40 mean reward 严格高于 step 0；
2. 配对 bootstrap 95% CI 下界大于 0；
3. step 40 top-1 命中率不低于 step 0；
4. 40-step 中至少 80%（32/40）组具有正 reward 方差；
5. 固定评估严格为 32 seeds × 2 checkpoints = 64/64 行；
6. 核心 QA 提取率下降不超过 5 个百分点；
7. 训练 trace 为 160/160 有限 reward 且零 infrastructure mask；
8. step 40 adapter 与 processor 保存完整并真实 reload 通过。

`run_status=passed` 只表示完整性通过；只有 `experiment_conclusion=passed` 才表示八项研究门槛同时通过。scorer 未 ready、存储预检失败或 GPU 隔离失败时，只能报告基础设施状态，不能写成 GRPO 不收敛。
