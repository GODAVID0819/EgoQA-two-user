# 六用户十分钟 Reasoning、Evidence 与 Answerability 设计

## 1. 决策摘要

新增一条独立的六用户十分钟 QA 路径，保留当前已经完成本地与三分钟远端验证的三分钟路径，不修改三分钟默认合同。

十分钟路径固定采用以下语义：

- 同步输入窗口为 `600s`，每位用户由二十个连续 `30s` 原始片段组成；
- pruning 继续使用完整窗口 ZIP 时间感知聚类；
- `cross_gap_mode="center"`；
- `max_cross_gap_seconds=30.0`，即本轮确认的 `G=30s`；
- `temporal_unit_seconds=30.0`；
- `seconds_per_cluster=2.5`，因此十分钟完整窗口使用 `K=240`；
- speaker 完整参与 ZIP pair 诊断，但 QA generator 始终使用 speaker 完整原视频；
- 五个 provider 使用各自 speaker-provider pair 生成的剪枝视频；
- groundedness 和 answerability 继续使用完整原始媒体，不使用 provider 剪枝视频替代事实审核；
- generator、evidence 和 answerability 开启 reasoning，单次输出上限为 `8192`；
- formality 关闭 reasoning，单次输出上限为 `2048`；
- 所有 stage 共用同一份驻留 Qwen 模型，不加载第二份模型。

十分钟输入只扩大可观察时间范围，不把 long-horizon 设为每道题的硬门禁。问题仍必须满足真实跨视角信息缺口；只要题目在十分钟窗口内自然、可见、可回答，就不要求它一定跨越窗口早期和晚期。

## 2. 范围与非目标

本设计覆盖：

1. evidence 的高置信度可见性和用户级多数支持；
2. answerability 的高置信度 needed-fact 判断；
3. 六用户 evidence segment 数由固定六段扩展为按窗口长度计算；
4. `G=30s` 在 ZIP cross-gap 参数、CLI、诊断和十分钟 wrapper 中的传播；
5. generator/evidence/answerability 与 formality 的 stage-specific reasoning 和输出上限；
6. reasoning 输出之后的最终 JSON 提取；
7. 独立十分钟运行入口与一次最小 smoke。

本设计不修改：

- 已验证的三分钟 wrapper 默认值；
- 两用户生产路径；
- GRPO、DPO、reward、reviewer 训练、优化器和 checkpoint；
- 当前五选一 QA schema 的问题、选项、correct 和 answer 字段；
- speaker 完整原视频、provider 剪枝视频的 generator 媒体路由；
- judge 使用完整原始媒体的基本原则；
- 自动提交、自动取消或自动扩展 Slurm 作业的权限边界。

## 3. 运行结构

十分钟路径新增独立 wrapper，建议命名为：

`hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch`

它复用当前六用户 candidate mining、QA loop、三类 judge、retry、部分结果持久化和 JobID 派生输出目录，不调用或覆盖三分钟 wrapper。

首次远端运行固定为：

- 一个 generation group；
- 一个 generation slot；
- 每题最多三次 attempt；
- 一个最小 smoke；
- 生成完整 attempt、prompt、媒体、运行 manifest 和人工审核产物。

首次 smoke 通过后是否增加 generation groups 和 slots，由该 Job 的真实耗时、显存、主机内存、输出截断率和人工质量决定，不在本设计中预先扩大。

十分钟正式资源、account、partition、QOS 和 walltime 不从旧两用户十分钟脚本复制。实施阶段必须依据当前 Torch 查询和首次 smoke 实测确定；三分钟的 `96G/4h` 只能作为历史参考，不能自动视为十分钟资源合同。

## 4. ZIP 时间剪枝与 `G=30s`

十分钟仍采用全窗口时间感知聚类：

\[
K=\left\lceil\frac{600}{2.5}\right\rceil=240.
\]

时间感知距离继续使用：

\[
d^2(i,c)=2\left(1-\cos(e_i,e_c)\right)
+0.1\left(\frac{t_i-t_c}{30}\right)^2.
\]

一个 speaker-provider representative pair 只有同时满足下式才进入剪枝：

\[
\cos(e_s,e_p)\ge 0.82
\quad\land\quad
\left|\bar t_s-\bar t_p\right|\le 30\text{ 秒}.
\]

这里的 `G=30s` 是 cross-view representative center gap，不是把十分钟视频切成互不通信的三十秒局部聚类。`PRUNING_BLOCK_SECONDS=30` 继续保留为运行合同和时间单位，但 ZIP 六用户生产路径仍执行一次完整 `600s/K=240` 聚类。

十分钟 candidate 和诊断必须显式记录：

- `duration_seconds=600`；
- `cross_gap_mode=center`；
- `max_cross_gap_seconds=30`；
- `temporal_unit_seconds=30`；
- `seconds_per_cluster=2.5`；
- `cluster_count=240`；
- 每个 speaker-provider pair 的 accepted/rejected gap 计数；
- 每位 provider 的 keep/remove intervals 和 retained percentage；
- speaker QA media 使用完整原视频。

## 5. Evidence 可见性与用户级投票

### 5.1 二十段输入

`evidence_segment_specs()` 不再要求每位用户正好六段，而是要求：

\[
\text{segment count}=\frac{\text{duration seconds}}{30}.
\]

因此：

- 三分钟继续要求六段；
- 十分钟要求二十段；
- 每段 index、time token、原始时间范围和本地媒体必须连续、完整且顺序一致；
- 任一用户缺段时当前 candidate 不进入 evidence judge。

十分钟 evidence 仍保持每位用户一次视觉调用。该调用接收该用户的二十个独立三十秒视频，返回二十个 segment observations 和一个用户级结论。这样总调用数仍为六个用户视觉调用加一次全局聚合，不把每道 QA 扩大成一百二十次视觉调用。

### 5.2 Segment observation

每个 material claim 继续使用：

- `SUPPORTED`；
- `CONTRADICTED`；
- `NOT_VISIBLE`；
- `AMBIGUOUS`。

同时增加 `confidence`：

- `HIGH`；
- `MEDIUM`；
- `LOW`。

判定规则固定为：

- 只有直接、清楚、可定位到具体 segment 和时间范围的精确事实，才允许 `SUPPORTED/HIGH` 或 `CONTRADICTED/HIGH`；
- 物体太小、太暗、遮挡、距离过远、分辨率不足或依赖常识推测时，必须输出 `NOT_VISIBLE`；
- 物体本身可见但两个解释仍同样合理时，允许 `AMBIGUOUS`；
- `MEDIUM/LOW` 不得形成最终用户投票；
- 一个 related scene、相似物体或题目措辞不能升级为视觉证据。

### 5.3 用户级结论

每位用户的 observation 增加：

```json
{
  "user_vote": {
    "visible": true,
    "confidence": "HIGH",
    "supported_option": "A/B/C/D/E/null",
    "supporting_segment_indices": [0],
    "reason": "直接可见事实与选项的对应关系"
  }
}
```

约束如下：

- `supported_option` 是该用户对答案相关物体或状态的高置信度视觉结论，不是根据 gold answer 复制；
- `visible=false` 时，`supported_option=null`；
- `confidence` 不是 `HIGH` 时，`supported_option=null`；
- 同一用户在多个 segment 中重复看到同一事实仍只计一票；
- 同一用户的 segment observations 自相矛盾时，用户级结论必须为不可投票状态，不能选择出现次数最多的 hallucination。

### 5.4 确定性多数聚合

全局 aggregation 不再要求所有用户一致。程序先从六个 `user_vote` 形成：

- `visible_user_count`；
- `correct_support_count`；
- 每个 A–E 选项的高置信度支持数；
- `not_visible_user_count`；
- `ambiguous_user_count`；
- `competing_options_reaching_threshold`。

`NOT_VISIBLE`、`AMBIGUOUS` 和非 HIGH 结论不进入 `visible_user_count`。

声明答案的支持阈值为：

\[
\text{correct support count}\ge 3
\quad\lor\quad
\text{correct support count}>\frac{\text{visible user count}}{2}.
\]

最终 evidence PASS 还必须满足：

1. 题干中的 material premise 至少有一个高置信度直接可见来源；
2. 声明答案达到上述阈值；
3. 没有另一个选项也达到阈值；
4. 没有与题干身份、对象连续性或同步时间关系直接冲突的高置信度证据；
5. `visible_user_count>0`。

如果两个选项都达到阈值、形成平票或 material premise 无可见来源，则 FAIL。低质量视角的 NOT_VISIBLE 不构成反例。

聚合结果必须保存投票计数和最终 reason，不能只保存模型的一段自然语言总结。

## 6. Answerability 高置信度事实充分性

六用户 answerability 继续只执行：

1. `speaker_only`；
2. `combined_all_six_users`。

不增加五个 provider-only gate，也不恢复 A–E forced-choice。

每个 `needed_fact` 增加 `confidence=HIGH/MEDIUM/LOW`。程序只把 `VISIBLE + HIGH` 视为可见事实：

- speaker-only 中所有必要事实均为 `VISIBLE + HIGH`，则 speaker 自己可以回答，当前 QA FAIL；
- speaker-only 至少一个必要事实不是 `VISIBLE + HIGH`，则 speaker-only insufficient；
- combined-all-six 中每个必要事实至少有一个 `VISIBLE + HIGH` 的真实 source user 和时间范围，则 combined sufficient；
- combined 中任一必要事实只有 NOT_VISIBLE、AMBIGUOUS 或非 HIGH 证据，则 combined insufficient；
- speaker-only insufficient 且 combined sufficient 时，answerability PASS。

Evidence 的“至少三人或 visible 多数”规则不直接替代 answerability。Evidence 判断声明答案是否可靠；answerability 判断指定 condition 是否拥有区分答案所需的全部事实。一个 provider 的清晰视角可以补齐 answerability 的某个事实，但声明答案本身仍需经过 evidence 多数规则。

Answerability prompt 必须明确：

- 不得使用题目措辞、选项先验或 generator rationale 作为视觉事实；
- 低置信度不得标记为 VISIBLE；
- speaker 在视频中直接看到答案时必须如实标记，不能为了构造跨视角依赖而写成 insufficient；
- combined condition 可以由不同用户分别补齐不同 needed facts，不要求每个用户独立回答整道题。

## 7. Stage-specific Reasoning 与输出上限

当前 `disable_thinking` 和 `max_new_tokens` 是 runner 级属性，无法在共享模型上让 formality 与其他 stage 使用不同配置。本设计把它们改为每次调用的不可变参数，不通过临时修改 runner 属性实现，避免并行 judge 之间发生配置竞争。

固定 stage profile：

| Stage | Reasoning | `max_new_tokens` |
|---|---|---:|
| generator | 开启 | 8192 |
| evidence segment observation | 开启 | 8192 |
| evidence aggregation | 开启 | 8192 |
| answerability speaker-only | 开启 | 8192 |
| answerability combined-all-six | 开启 | 8192 |
| qa_formality | 关闭 | 2048 |
| schema-only JSON repair | 关闭 | 2048 |

本地 Qwen runner、OpenRouter runner 和其他现有 backend 的 generate interface 必须接受可选的 per-call profile。未传 profile 的旧调用保持当前默认行为，保证两用户和三分钟路径兼容。

`max_new_tokens=8192` 是输出上限，不要求模型使用满 8192。它覆盖 reasoning trace 和最终 JSON。内存安全估算必须把当前调用的实际输出上限计入 KV 预算，不能继续只读取 runner 初始化时的统一上限。

所有 stage 继续复用同一个 runner/model 对象；不同输出上限不得触发第二次 27B 模型加载。

## 8. Reasoning 输出与 JSON 提取

当前 JSON parser 使用第一个 `{` 到最后一个 `}`。Reasoning 开启后，中间推理可能包含括号、示例 JSON 或其他对象，不能再假定第一个对象就是最终答案。

新 parser 必须：

1. 保留完整 raw response 到 trace；
2. 优先识别 reasoning 结束后的最终 JSON；
3. 从后向前寻找能够被 `json.JSONDecoder` 完整解析的最后一个 JSON object；
4. 只接受 object，不接受数组或标量；
5. 对解析出的 object 继续执行现有 schema 校验；
6. 没有完整最终 JSON 时当前调用失败，不从截断片段猜测字段；
7. schema-only repair 只能修复格式，不能改变 evidence、answerability 或 formality 判定。

Prompt 和 attempt trace 至少记录：

- stage profile；
- reasoning 是否开启；
-输出上限；
- 实际输出 token 数；
- 是否达到输出上限；
- 最终 JSON 是否成功解析；
- repair 是否发生。

## 9. 十分钟媒体与内存边界

十分钟 generator 使用一个完整 speaker 和五个 provider-pruned 视频；evidence 与 answerability 使用六个完整原视频。内存安全 backend 继续使用一个驻留模型并串行化实际模型调用。

十分钟路径初始采用：

- `QWEN_MEMORY_SAFE_VIDEO_FPS=0.25`；
- `max_image_pixels=131072`；
- `max_input_tokens=131072`；
- 本地视频转码最大边长 `512`；
- Flash Attention 可用时使用 `flash_attention_2`。

这些值来自当前 27B memory-safe 路径和三分钟运行经验，只作为首次十分钟 smoke 的工程起点，不表示质量最优。运行前必须估算六路十分钟 combined condition 的输入 token 和当前 `8192` 输出预算；超过上限时停止，不静默降低 FPS、像素或 reasoning。

## 10. 错误处理与恢复

- 任一用户缺少二十个连续 segment：当前 candidate 失败；
- evidence observation JSON 不完整：当前 evidence judge 失败，并保存 raw output；
- 用户级 vote 不满足字段合同：当前 evidence judge 失败，不把该用户静默排除；
- `visible_user_count=0`：evidence FAIL；
- 两个选项同时达到阈值：evidence FAIL；
- answerability source user 不属于当前 condition：当前 condition 无效；
- reasoning 输出达到上限且没有完整 JSON：当前调用失败；
- stage profile 未传播到 runner：本地合同测试失败；
- 达到内部 deadline 后不启动新 attempt，已完成 attempt 保留；
- 远端或 Slurm 失败不触发自动取消、覆盖或清理已有作业。

## 11. TDD 与验证合同

实现必须遵循测试先行，至少覆盖以下 RED→GREEN 行为：

### Evidence

1. 每用户六段的三分钟输入继续合法；
2. 每用户二十段的十分钟输入合法；
3. 任一用户缺段或时间不连续时拒绝；
4. 三个 HIGH visible 用户支持 correct、其余 NOT_VISIBLE 时 PASS；
5. visible 用户中二比一支持 correct 时 PASS；
6. 二比二平票时 FAIL；
7. correct 和 competitor 同时达到阈值时 FAIL；
8. MEDIUM/LOW、NOT_VISIBLE 和 AMBIGUOUS 不进入投票；
9. 同一用户多个 segment 重复支持只计一票；
10. 高置信度题干身份或对象连续性冲突仍然阻断。

### Answerability

1. speaker 全部 needed facts 为 VISIBLE/HIGH 时拒绝；
2. speaker 有不可见 fact、combined 全部 facts 可由一个或多个用户补齐时通过；
3. combined 中存在 MEDIUM、LOW、NOT_VISIBLE 或 AMBIGUOUS 的必要 fact 时拒绝；
4. source user 不属于 condition 时拒绝；
5. 六用户仍只创建 speaker-only 和 combined-all-six 两个 condition；
6. 非六用户 forced-choice 路径保持当前行为。

### Reasoning 与 parser

1. generator/evidence/answerability 收到 reasoning 开启和 8192；
2. formality 收到 reasoning 关闭和 2048；
3. 所有 stage 共享同一 runner/model identity；
4. reasoning 中包含普通大括号时仍能提取最后一个有效 JSON；
5. reasoning 中包含示例 JSON 时选择最后一个符合 schema 的对象；
6. 最终 JSON 截断时失败，不返回部分对象；
7. memory-safe KV 估算使用 per-call 输出上限。

### 十分钟与 pruning

1. 十分钟窗口生成二十个连续片段；
2. 十分钟 ZIP 使用全窗口 `K=240`；
3. center gap 边界 `30s` 接受，超过 `30s` 拒绝；
4. 三分钟原有 `G=10s` 默认合同不变；
5. 十分钟 wrapper 显式传入 `600s/G=30s/reasoning profiles`；
6. speaker/full、provider/pruned、judge/full 媒体路由不变；
7. 现有三分钟聚焦测试继续通过。

本地验证证明代码、schema、参数传播、解析和媒体合同；它不能证明十分钟 GPU runtime、吞吐量、reasoning 质量或人工准确率改善。

## 12. 远端执行与研究结论边界

本设计批准和本地实现不自动授权提交 Slurm 作业。完成本地 TDD 后，首次十分钟运行需要用户再次明确授权。

首次远端运行只允许一个最小 smoke，用于验证：

- 六路十分钟媒体能够准备和读取；
- ZIP `K=240/G=30s` 完成；
- 一个共享模型能够按 stage 切换 reasoning/output profile；
- generator、六个 evidence user calls、一次 aggregation 和两个 answerability conditions 能形成完整 JSON；
- 输出没有因 8192 上限截断；
- 产物、耗时、资源和人工审核材料完整保存。

Smoke 通过只能证明工程链路可运行，不能证明多数投票阈值、reasoning 或十分钟上下文提高了 QA 准确率。研究质量必须在新的固定人工标注集上分别报告 formality、evidence 和 answerability 的一致率，并与本轮十七条有效人工标注使用同一口径比较。

## 13. 修改边界与 Git

预计实施范围限于：

- `prompts.py`；
- `evidence_chunk_review.py`；
- `video_qa_loop.py`；
- `qwen3vl_runner.py`；
- `group_relative_clip_sampling.py`；
- `schema.py` 中必要兼容字段；
- 新十分钟 wrapper；
- 对应的聚焦测试和本设计后续实施计划。

不覆盖当前未提交的三分钟 wrapper、三分钟资源测试、`.bak`、`.codex_runtime` 或人工审核工具。实施提交必须只暂存经审查的源码、测试、十分钟 wrapper 和文档，不包含运行产物、缓存或备份。
