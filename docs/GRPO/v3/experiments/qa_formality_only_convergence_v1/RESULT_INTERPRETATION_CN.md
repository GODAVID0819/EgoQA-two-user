# `qa_formality`-only 实验结果解释

## 唯一允许的通过结论

只有 Gate A、1-step smoke 和 40-step probe 均通过时，才可以写：

> 在本次固定单 evidence、40-step、temperature 0.7、constant learning rate 1e-5 的最小过拟合实验中，现有 GRPO v3 链路能够提高由冻结 8B `qa_formality` judge PASS/FAIL logprob margin 定义的连续 reward；LoRA adapter 已完成更新并可重载。

不能把它改写成：

- 完整 repo-native reward 已收敛；
- QA 的 groundedness 或 answerability 已改善；
- reviewer 与人工高度一致；
- 最终 QA 综合质量提高；
- Gate 4 已解锁；
- 更大规模训练一定有效。

## 结果分类

| 现象 | 允许结论 | 下一步 |
|---|---|---|
| reward delta > 0、slope > 0、正方差比例 ≥ 0.8、adapter 可重载 | 链路能学习单一 formality 目标 | 做人工反 reward-hacking 审计，再决定是否设计泛化实验 |
| reward 不提升、正方差充足、adapter 更新 | optimizer 在更新，但没有可检测的 formality 改善 | 不调多个参数；检查采样轨迹和 judge 置信度分布 |
| 大量零方差组 | 当前采样/reward 无法提供稳定相对优势 | 不能据此否定 GRPO；先修复信号可学性 |
| adapter 没有更新或不能重载 | 训练保存/更新链路失败 | 先修基础设施，不讨论 reward 是否可学 |
| reward 提升但不可恢复率上升 | 可能通过退化输出获取 reward | 直接判失败，不得报告收敛 |
| 出现其他 reward component 或 judge trace | 实验隔离失败 | 直接判失败，修复接线后重跑 |
| reviewer 服务、judge JSON 或 logprob 缺失 | 基础设施失败 | 不得转换成低 reward，也不得写成训练不收敛 |

## 反 Reward Hacking 审计

从训练早期和末期各抽取至少 20 个 completion，检查：

- 问题是否模板化重复；
- 问题是否异常缩短；
- 五个选项是否重复或失去互斥性；
- 是否复制 judge prompt 中的示例；
- 是否靠固定高置信措辞欺骗 judge；
- JSON/MCQ 不可恢复率是否上升。

人工审计只可以否决数值通过，不能新增训练 reward，也不能用主观印象覆盖数值失败。

## 汇报模板

```text
实验：GRPO v3 qa_formality_confidence_v1
本地验证：专项测试数量与结果；完整 GRPO v3 测试数量与结果；compileall；source preflight
Gate A：输入 trace SHA-256；有限 reward；缺失 logprob；完整组；正方差组比例
Smoke 作业：作业 ID；result status；4/4 reward；reward std；global_step；adapter reload
Probe 作业：作业 ID；result status；160/160 reward；前 10 组均值；后 10 组均值；delta；slope；正方差比例；不可恢复率；adapter reload
人工审计：是否发现模板化、结构退化或 judge hacking
当前允许结论：严格从本文结果分类中选择
下一步：只写一个具体动作
```

