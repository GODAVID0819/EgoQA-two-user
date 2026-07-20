# GRPO v3 多信号人工审计设计

## 目标

将现有只覆盖 groundedness 的 24 条人工审计升级为多信号审计。在不重新调用 reviewer、不改变现有 Gate 3 硬门槛和 reward 权重的前提下，从原始 `reward_trace.jsonl` 补回 answerability、speaker leakage、provider-only answerability、QA formality、shallow activity 与 JSON format 信号，并允许人工逐项复核。

## 设计边界

- 继续审计同一批 12 个 reviewer groundedness PASS 和 12 个 FAIL 案例。
- 复用已经截取的 48 个视频，不重新生成媒体。
- 新增信号先用于诊断和人机一致性分析，不自动改变 reward、Gate 3 或 Gate 4。
- `approved_for_weight_change` 继续只表示既有 groundedness 审计完成后，操作者显式批准既定权重变更。
- evidence provider 单独能够回答符合当前角色契约，不属于 answerability gate 失败。

## 信号分层

### 视频事实层

- `groundedness`：视频是否支持 evidence claims 与正确答案。

### 可回答性层

- `combined_answerability`：两用户视频合并后是否唯一支持正确项。
- `speaker_leakage`：提问者视频单独是否已经足以答对。
- `provider_answerability`：证据提供者视频单独是否足以答对，仅作诊断。

人工 answerability gate 按下式推导：

\[
\text{Human Answerability Gate}
=
\text{Combined PASS}
\land
\text{Speaker NO\_LEAK}
\]

### QA 质量层

- `qa_formality`：问题自然度、五选项结构、答案唯一性与题型一致性。
- `shallow_activity`：问题是否退化为浅层、显然、缺少真实跨用户信息需求的活动询问。

### 结构层

- `format_status`：原始 JSON、保守修复或不可恢复状态。该字段直接展示机器结果，不要求人工通过视频判断。

## 审计案例数据

每条 `groundedness_audit_cases.jsonl` 在保留现有字段的基础上增加：

- reviewer 的 groundedness、formality、shallow activity 状态与理由；
- `combined_correct`、`speaker_only_correct`、`provider_only_correct`；
- answerability gate 结果与理由；
- answerability 各条件的 condition id、用户、选择、证据、无法回答原因和可用的不确定性摘要；
- reward components 与总 reward；
- 明确的 speaker/provider 用户名。

缺失源字段必须保留为未知值，不得默认转换为 PASS、FAIL、可回答或泄漏。

## CSV 模式

机器字段：

- `reviewer_groundedness`
- `reviewer_combined_answerability`
- `reviewer_speaker_leakage`
- `reviewer_provider_answerability`
- `reviewer_qa_formality`
- `reviewer_shallow_activity`

人工字段：

- `human_groundedness`：`PASS`、`FAIL`、`UNCERTAIN`
- `human_combined_answerability`：`PASS`、`FAIL`、`UNCERTAIN`
- `human_speaker_leakage`：`LEAK`、`NO_LEAK`、`UNCERTAIN`
- `human_provider_answerability`：`ANSWERABLE`、`NOT_ANSWERABLE`、`UNCERTAIN`
- `human_qa_formality`：`PASS`、`FAIL`、`UNCERTAIN`
- `human_shallow_activity`：`PASS`、`FAIL`、`UNCERTAIN`

继续保留 `claim_visible`、`answer_supported`、`reviewer_agreement` 和 `notes`，兼容已经开始的 groundedness 审计。

## 导出和升级安全

- `export` 在目标 CSV 已存在时，按 `case_id` 合并已有人工字段。
- 新案例集合必须与已有案例集合完全一致；不一致时中止并报告，避免把旧判断错配到新案例。
- 覆盖已有 CSV 前创建带时间戳的备份。
- 已有未知列也应在合并中保留，避免丢失人工附加信息。
- 指南、案例 JSONL 与 CSV 使用同一组 case id，并在导出后验证一致性。

## 指南结构

每条案例依次展示：QA 与答案、角色、groundedness、三种 answerability 条件、QA formality、shallow activity、format/reward、视频窗口命令和人工填写提示。reviewer 结论必须在人工观看步骤之后提示阅读，减少锚定偏差。

## 汇总模式

汇总模式升级为 `grpo_v3_multisignal_audit_v2`，同时保留原 groundedness 顶层统计字段以兼容既有消费者。新增：

- 各信号 completed、各类别计数、人机一致数与一致率；
- answerability gate 可推导案例数、通过数、失败数和不确定数；
- groundedness 与 combined failure、speaker leakage 的交叉计数；
- 每行无效值和未完成字段的诊断。

既有最少 20 条、reviewer groundedness PASS/FAIL 各至少 8 条的批准前置条件保持不变。新增人工信号可以部分填写并获得诊断汇总，但不得被伪装为已完成。

## 测试与验收

- 单元测试覆盖所有源字段抽取、未知值处理、CSV 字段、旧 CSV 合并、案例集合不一致保护和备份。
- 单元测试覆盖多信号计数、人机一致率、人工 answerability gate 推导与旧顶层字段兼容。
- 使用本地真实 Gate 3 trace 重新导出当前 24 条案例，确认 case id 与原审计包完全一致、48 个视频未被改动。
- 运行相关测试、`compileall` 与 `git diff --check`。
