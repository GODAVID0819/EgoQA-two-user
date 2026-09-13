# Cross-view Relation 下一阶段实施清单

目标：保留 `qa_cross_view_relation_v2` 基线，新增 v3，使明显的文本内部一致性错误和浅层关系无法获得高分，并把训练扩展到多 `evidence_id`、8B generator 与 32B text judge。

实施边界以
[`../designs/2026-07-28-cross-view-relation-next-stage-design.md`](../designs/2026-07-28-cross-view-relation-next-stage-design.md)
为准。本文件只记录执行顺序，不重复设计论证。

- [x] 给 text-only judge 增加八项结构化检查；明确不评价视频 groundedness 和真实 answerability。
- [x] 新增 v3 reward 入口；内部一致性或浅层关系失败时由代码执行硬封顶，v2 语义保持不变。
- [x] 将人工复核的 31 条高分 QA 冻结为 reviewer 回归样本，其中 9 条为 text-only 负例。
- [x] 增加多 `evidence_id` 数据 Gate：训练/held-out 不重叠，并同时覆盖 commonality 与 difference。
- [x] 在 probe 分析中报告模板集中度与 per-evidence 指标。
- [x] 增加 8B generator、32B text judge 的 smoke/probe120 Slurm 入口及必要远程预检。
- [x] 运行专项测试、训练回归、编译和 shell 静态检查。
- [x] 提交并推送当前分支。

测试原则：只测试会改变 reward、judge 合同、数据 Gate 或远程启动合同的行为；不为纯文档、常量搬运和一次性胶水代码添加低价值测试。
