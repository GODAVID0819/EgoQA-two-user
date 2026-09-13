# Cross-view Relation GRPO 下一阶段设计

## 1. 决策摘要

继续使用已经显示出可学习信号的 `qa_cross_view_relation_v2` GRPO 框架，不回退到 `qa_formality`，也不立即把 noisy groundedness/answerability judge 混入 reward。

下一阶段同时升级：

- 数据：从一个 evidence pair 扩展到多个独立 `evidence_id`；
- generator：从 Qwen3-VL-2B 升级到 Qwen3-VL-8B；
- text-only judge：从 Qwen3-VL-8B 升级到 32B 纯文本 Instruct 模型；
- judge contract：逐项输出检查结果，并对阻断性错误施加最终 reward 硬封顶；
- 评估：增加模板集中度、跨 evidence held-out endpoint 和人工 reviewer 回归集。

## 2. 当前证据与边界

job `14833087` 的 probe40 有 40 个 group、160 个候选和可重载 checkpoint，冻结代理奖励的早期/后期窗口呈上升，因此可以视为当前 reward/训练框架的正向收敛证据。

但该 probe 的数据覆盖只有：

```text
distinct evidence_id = 1
evidence_id = EGOLIFE2U_DAY2_11350000_A1_A5
question_type = commonality（40/40）
```

因此它不能证明跨 clip、跨场景或跨 question type 泛化。桌子、花、笔记本和屏幕内容模板的集中，既可能来自 2B generator，也可能来自同一视频对被重复优化。

本设计把当前结论表述为：

> 策略在一个双用户视频对和冻结 cross-view text reward 下能够提高代理奖励；跨 evidence 泛化尚未验证。

## 3. 本次训练目标

目标名称：

> Cross-view Relational QA Writing Quality

定义：

> 在不给 judge 原始视频的条件下，训练策略生成格式有效、自然、问答内部一致、具有明确跨视角信息关系且不退化为浅层活动报告的两用户五选一 QA。

### 3.1 纳入 reward 的维度

- JSON 与五选项结构；
- `correct`、`answer` 和目标选项一致；
- question/answer/options 的语义类型一致；
- 选项互斥、平行、无语义重复；
- 第一人称或共享记忆表达自然、清楚；
- speaker-side context 与 information request 相关；
- 文本表达具体的信息缺口、关系、状态、位置、结果、后续、对比或其他 coherent relation；
- 活动问题具有具体关系，不是泛化的“另一个人在做什么”；
- 生成的 evidence、claims、rationale 与 QA 在文本层面不互相矛盾。

### 3.2 暂不纳入 reward 的维度

- 视频事实是否支持答案；
- 实际 evidence groundedness；
- required user 0 是否真的无法单独回答；
- 两个视角组合后是否真的唯一可回答；
- 真实时间对齐；
- distractor 的视觉真值。

这些维度后续作为独立 judge 组件加入。当前 text-only 收敛不能被表述为真实 groundedness 或 answerability 改善。

## 4. Anchor 的真实含义

当前 reward 中，`anchor_tier` 是明确的加权项：

```text
A = anchor_tier / 2
reward = 0.60 * Q + 0.25 * A + 0.15 * B
```

因此，在其他项相同的情况下，被 judge 判为更接近或达到 strong anchor 的候选会得到更高奖励，anchor 直接贡献 25%。

但它不是文本 embedding 距离，也不应奖励复刻 strong anchor 的杯子、交接或句式。正确含义应是：

- `0`：不优于弱锚点，通常是浅层活动或缺少关系；
- `1`：存在具体信息关系，但结构强度或表达完整性有限；
- `2`：达到强关系标准，文本中有明确、具体、连贯的信息需求和关系结构。

风险在于当前 judge 可能把“像 strong anchor 的措辞”误当成“关系结构强”。下一版 prompt 必须明确禁止对象、场景、句式相似性影响 anchor tier，并要求先完成绝对检查再定 tier。

## 5. 模板化风险

GRPO 会提高高奖励模式的概率，因此在当前设置中模板收缩是实质风险，而不是影响很小：

- 同一个 evidence pair 被重复 40 次；
- question type 全为 commonality；
- 2B generator 表达能力有限；
- judge 对 “I could not see X; what was X?” 模板容易给高分；
- 组内相对奖励会放大稳定拿分的局部策略。

下一阶段必须同时改变数据覆盖和监控指标。仅提高 temperature 不能解决 reward-driven mode concentration。

模板化评估至少包括：

- 完全相同 question 比例；
- 规范化问题前缀和句法骨架的 top-k 占比；
- `I was ... but I could not ...` 等模板比例；
- 对象词、关系类型和疑问词分布；
- 高分候选的模板集中度；
- train evidence 与 held-out evidence 的差异。

## 6. 数据扩展

probe120 不再对同一 evidence pair 重复 120 组。目标配置：

```text
至少 12 个独立 evidence_id
每个 evidence_id 约 10 个 group
总计约 120 group
commonality / difference 均有覆盖
按 evidence_id 划分 train / held-out
```

若当前可用数据不足 12 个，最低接受：

```text
8 个独立 evidence_id
至少 2 个完全 held-out evidence_id
```

禁止把同一 evidence pair 的不同 completion 拆到 train 和 held-out 两侧。

在提交前必须生成 dataset audit，验证：

- evidence_id 数量与频次；
- required user pair；
- question type 分布；
- 每个视频路径存在；
- train/held-out 无 evidence_id 重叠；
- 每组真实调用仍为 4 completions。

## 7. Generator 与 judge

### 7.1 Generator

下一阶段使用：

```text
Qwen3-VL-8B-Instruct
BF16 LoRA
冻结视觉塔和 aligner
q_proj / v_proj
原生视频输入
```

选择 8B 而不是 4B，是因为时间有限且计算资源可申请；8B 更可能减少格式字段错位和固定措辞，同时仍保持 LoRA 训练可管理。

### 7.2 Text-only judge

judge 不看视频，因此优先使用 32B 纯文本 Instruct 模型。部署首先选择 BF16 tensor parallel 2；若已有经过验证的官方量化权重，可在单独 reviewer-only Gate 中评估后替换。

建议资源形状：

```text
GPU 0：8B VL generator / trainer
GPU 1-2：32B text judge，tensor parallel 2
```

模型升级和 prompt 升级同时进入下一版，但必须保留旧 probe40 的 frozen completion，作为上线前的便宜回归集。这里不要求完成完整研究型 A/B，只要求新组合越过明确的 reviewer Gate。

## 8. Judge 合同与硬封顶

每个候选必须先独立输出以下检查：

```text
question_answer_type_match
options_answer_same_question
semantic_option_uniqueness
answer_resolves_question
premise_relevance
text_claim_consistency
natural_first_person_wording
shallow_activity_relation
```

每项包含：

```json
{
  "status": "PASS/FAIL",
  "reason": "one concise text-only reason"
}
```

完成绝对检查后，才允许输出：

```text
cross_view_relation_score
semantic_naturalness_score
internal_consistency_score
anchor_tier
pairwise_preferences
```

硬约束：

- 任一阻断性一致性检查失败：
  - `internal_consistency_score = 0`
  - `anchor_tier <= 1`
  - 最终 reward cap = `0.40`
- `shallow_activity_relation = FAIL`：
  - `cross_view_relation_score = 0`
  - `anchor_tier = 0`
  - 最终 reward cap = `0.40`
- 自然度存在阻断性语法/角色逻辑错误：
  - `semantic_naturalness_score = 0`
  - 最终 reward cap = `0.55`
- 多个 cap 同时触发时取最小值。

cap 触发原因必须写入 reward trace。硬封顶属于代码合同，不依赖 judge 自己记得降低最终分。

## 9. Reviewer 上线 Gate

使用用户修订后的 31 条高分审计表：

- 9 条带 comment：明确负例；
- 22 条空 comment：不能仅因缺少视频 groundedness/answerability 被 text-only judge 判为错误。

上线要求：

- 9 条明确负例全部不再获得高于 `0.9` 的 reward；
- question/answer 类型错位、语义重复选项和浅层活动负例触发预期 cap；
- 22 条非负例不能因为单视角可回答或视觉 evidence 不充分而被错误惩罚；
- JSON/schema repair 成功；
- forward/reverse instability 明显低于 probe40 的 28/40，目标不高于 20%；
- 每条结果保留逐项检查与两次原始输出。

## 10. Groundedness 与 answerability 的未来接入

未来不需要重写 GRPO 主框架。保留以下稳定接口：

```text
completion group
→ deterministic assessment
→ text relation/formality judge
→ optional groundedness judge
→ optional answerability judge
→ component policy / caps
→ final reward
```

在两个视频 judge 的误判率得到人工校准前，不把它们直接并入当前训练 reward。

后续接入顺序：

1. 先在冻结样本上分别测 groundedness 和 answerability 的人工一致率；
2. 修订 prompt/schema；
3. 为每个组件单独记录 coverage、误判率和 abstain；
4. 只把达到接受标准的组件加入 reward；
5. 使用保底约束，防止新增目标抵消已经获得的 formality/relation 质量。

已有 text-only 成果不会天然增加后续难度；它提供更高比例的格式正确候选，使视频 judge 不必浪费容量处理明显畸形 QA。但若当前阶段过度模板化，后续真实质量优化会受到限制，因此本阶段必须同时监控 diversity 和跨 evidence held-out 表现。

## 11. Gate 顺序

```text
Gate 0：多 evidence dataset audit
Gate 1：32B text judge 服务与固定 31 条 reviewer 回归
Gate 2：8B generator 原生视频真实 4-completion preflight
Gate 3：1-step smoke，reward/cap/gradient/checkpoint/reload
Gate 4：小规模多 evidence probe
Gate 5：约 120-group 多 evidence probe
Gate 6：按 evidence_id held-out 固定端点评估
Gate 7：独立 groundedness/answerability/人工审计
```

当前 `probe120.sbatch` 不能原样提交，因为它仍只是把现有 dataset 和 step 数包装成 120，尚未证明多 evidence 覆盖和新模型资源合同。

## 12. 成功表述

本阶段通过时可以表述：

> 8B VL generator 在多 evidence 数据上，通过 32B text-only cross-view relational QA judge 的 GRPO 训练，提高了 held-out text reward，同时保持格式有效性并降低模板集中。

不能表述：

> 生成 QA 已被视频验证为 grounded，或真实需要两个视角才能回答。
