# GRPO v3 JSON 格式 reward 三层处理设计

## 1. 背景与目标

Gate 2 作业 `14085999` 已完成原生双视频 policy rollout、冻结 8B reviewer 调用和四条 reward trace，但 candidate 2 的完整 JSON 对象在两个字段之间缺少逗号，导致解析失败并触发整组中止。该错误不涉及输入 JSONL、视频、模型环境或 reviewer 服务。

本设计的目标是：保留内容 reward 的主要训练信号，同时让候选自身的 JSON 格式质量成为可学习、可审计的有限 reward，避免单个纯格式错误中止完整 GRPO group。

唯一策略基线仍是 `docs/GRPO/v3/NATIVE_VIDEO_MSSWIFT_GRPO_STRATEGY_CN.md`。实现时同步修订其中的格式失败语义；模型、媒体、LoRA、reviewer 与 Gate 顺序不变。

## 2. 已批准的范围

实现以下行为：

1. 原始输出是严格 JSON：正常内容评分，`format_penalty=0.0`。
2. 原始输出不是严格 JSON，但能以保守、纯语法操作无歧义修复：保存原文、修复后文本和操作列表；正常运行 reviewer；最终 reward 加 `format_penalty=-0.5`。
3. 无法无歧义修复：不调用 reviewer；返回 `format_failure_reward=-3.0`。
4. 所有由候选自身 JSON 格式造成的结果均返回有限 reward，不再触发整组中止。
5. trace、Gate 结果和 manifest 分别记录原始合法、已修复、不可修复的数量、比例及修复操作。
6. reward 继续使用现有联合标量路线，保留所有 component 级审计。

明确不实现：

- 32-completion policy-only 格式探针；
- 保留合法 candidate 的跨轮候选池；
- 自定义 mask-aware GRPO trainer；
- 整组或单 candidate 动态重采样；
- GDPO 或分阶段 reward 训练；
- sampled frames；
- QLoRA 默认路径。

## 3. 语义边界

“格式失败返回有限 reward”只覆盖由 policy completion 自身造成的 JSON/格式问题。

以下情况仍属于基础设施或证据不可用，不得伪装成模型低分：

- 两段完整 reviewer 视频缺失；
- reviewer 服务不可连接、超时或返回不可用结果；
- 必需 review signals 因系统错误缺失；
- 数据与 policy completion 错位；
- 非有限 reward；
- 其他无法归因于候选质量的运行故障。

这些情况继续写入 masked trace 并中止当前 Gate 2 group。这样不会用 `-3.0` 隐藏 reviewer、媒体或训练基础设施故障。

可解析 JSON 进入现有 schema、reviewer 和 `compute_judge_reward()` 流程。schema、groundedness、answerability、leakage 与 formality 的现有联合评分语义不在本次重新设计范围内。

## 4. 三层解析与评分流程

### 4.1 第一层：原始严格 JSON

严格模式要求 completion 去除首尾普通空白后就是一个 JSON object；Markdown code fence 不属于严格 JSON。

处理：

```text
strict parse success
→ format_status=raw_valid
→ format_penalty=0.0
→ 正常 schema/reviewer/content reward
```

trace 记录原始 completion，但不生成修复文本或修复操作。

### 4.2 第二层：保守、纯语法修复

首版只允许不会修改字符串内容或业务字段的操作：

1. 去除完整包裹 JSON object 的 Markdown `json`/普通 code fence；
2. 删除 object/array 闭合符号前、位于字符串之外的多余逗号；
3. 在两个明确相邻的 object members 之间、位于字符串之外补一个缺失逗号。

实现必须使用能识别字符串、转义符、object/array 深度的扫描器；不得用可能修改字符串内部内容的全局正则替换。

约束：

- 每次修复最多执行 3 个操作；
- 每个操作必须包含类型和字符位置；
- 修复只能插入/删除结构标点或移除外层 code fence；
- 字符串 token 的顺序和值必须保持不变；
- 修复后必须由标准 `json.loads()` 成功解析为 object；
- 任一约束不满足即进入第三层。

处理：

```text
strict parse fail
→ conservative repair success
→ format_status=repaired
→ format_penalty=-0.5
→ 正常 schema/reviewer/content reward
→ final_reward=content_reward-0.5
```

修复不改变现有内容 component，只新增：

```json
"format": -0.5
```

最终 reward 不额外裁剪，保留联合标量的相对顺序。

### 4.3 第三层：不可修复格式失败

包括但不限于：

- 字符串未闭合；
- JSON 被截断；
- 缺失大段 object/array；
- 需要猜测字段名、字段值、选项、答案或 evidence；
- 修复超过 3 个操作；
- 修复后仍不能由标准解析器读取；
- 修复会改变任意字符串内容。

处理：

```text
strict parse fail
→ conservative repair fail
→ format_status=unrecoverable
→ 不调用 8B reviewer
→ masked=false
→ eligible_for_grpo=true
→ reward_total=-3.0
```

reward components 只记录：

```json
"format": -3.0
```

该值表示“候选无法被任务系统消费”，不是 reviewer 对内容质量的判定。

## 5. 数据结构与审计字段

每条 reward trace 新增统一的 `format_validation`：

```json
{
  "status": "raw_valid | repaired | unrecoverable",
  "raw_completion": "...",
  "repaired_completion": "... or null",
  "repair_operations": [
    {
      "operation": "strip_markdown_fence | remove_trailing_comma | insert_missing_member_comma",
      "position": 1700
    }
  ],
  "semantic_text_changed": false,
  "format_penalty": 0.0
}
```

不可修复记录还必须保留标准解析器的异常类型、行、列和字符位置。不得只保存摘要而丢失 raw completion。

Gate 2 聚合结果和 run manifest 新增：

```text
format_raw_valid_count
format_repaired_count
format_unrecoverable_count
format_raw_valid_rate
format_repaired_rate
format_unrecoverable_rate
format_repair_operation_counts
format_repaired_penalty=-0.5
format_unrecoverable_reward=-3.0
format_reward_revision=json_three_tier_v1
```

原有 reward components、reviewer trace、answerability 与 GPU/依赖证据继续保留。

## 6. Gate 与通过条件

Gate 0 和 Gate 1 已验证的环境、原生双视频、LoRA、冻结、optimizer、保存与重载契约不受影响，不需要重跑。

Gate 2 必须使用新 reward revision 重新运行。通过条件仍包括：

- 4 条 completion/reward trace；
- 4 个有限 reward；
- `reward_std > 0`；
- `global_step >= 1`；
- adapter 与 processor 真实重载成功；
- 没有基础设施类 masked reward；
- manifest 明确记录 `json_three_tier_v1` 及格式统计。

不可修复 JSON 获得 `-3.0` 不再视为 masked，因此本身不阻止 Gate 2 通过；它必须出现在格式统计和原始 trace 中。

Gate 2 通过前不得进入 Gate 3。Gate 3/4 暂时继续使用联合标量 reward；是否改用 GDPO 或分阶段训练只在获得 component 曲线和冲突证据后重新决策。

## 7. 错误处理

- 修复器正常判定当前语法不在白名单或超过操作上限：归为 `unrecoverable`，记录原因并返回 `-3.0`。
- 修复器出现非预期内部异常：视为实现/基础设施错误，先写诊断再中止；不得把代码缺陷伪装成模型格式低分。
- reviewer/媒体/数据映射故障：继续 masked 并中止 group；不得返回 `-3.0`。
- content reward 为 `None`：沿用现有非格式 mask 语义并中止 group。
- content reward 为 NaN/Inf：立即失败并保存 trace。
- trace 写入必须先于任何主动中止。

## 8. 测试设计

### 8.1 格式解析单元测试

- 原始严格 JSON：不修改，`raw_valid`，penalty 为 0；
- code fence 包裹：修复成功，记录 `strip_markdown_fence`；
- 本次 Gate 2 缺逗号样例：修复成功，记录正确位置；
- 闭合符号前多余逗号：修复成功；
- 字符串内部类似 `}\n\"key\"` 的内容：绝不修改；
- 转义引号和反斜杠：扫描状态正确；
- 截断 JSON：不可修复；
- 超过操作上限：不可修复；
- 修复前后字符串值不同：不可修复。

### 8.2 reward 集成测试

- raw-valid：reviewer mock 被调用，format component 为 0；
- repaired：reviewer mock 使用修复后的对象，最终 reward 精确减 0.5；
- unrecoverable：reviewer mock 不被调用，返回 -3.0 且非 masked；
- 一组包含 unrecoverable completion：插件返回 4 个有限 reward，不中止；
- 缺 reviewer 视频或 content reward 为 `None`：仍中止 group；
- trace 完整保存 raw/repaired/operations。

### 8.3 Gate/manifest 回归测试

- Gate 2 接受有限的 `-3.0` 格式失败 reward；
- Gate 2 拒绝基础设施 masked reward；
- Gate 2 拒绝零 reward 方差、零 optimizer step 或缺失 adapter；
- manifest 格式计数与 reward trace 一致；
- Gate 顺序、模型、原生视频、BF16 LoRA、冻结 reviewer、无 sampled frames、无默认 QLoRA 契约继续通过。

## 9. 文档与 Torch 交付

实现同步更新：

- `docs/GRPO/v3/NATIVE_VIDEO_MSSWIFT_GRPO_STRATEGY_CN.md`；
- `docs/GRPO/v3/MS_SWIFT_NATIVE_VIDEO_TORCH_RUNBOOK_CN.md`；
- Gate 2 的 result/manifest 字段说明；
- SFTP 最小上传清单；
- 登录节点无 GPU 回放本次 malformed completion 的命令；
- Gate 2 重提与结果验收命令。

所有代码和 Torch 命令先在本地完成并验证；不远程提交作业。
