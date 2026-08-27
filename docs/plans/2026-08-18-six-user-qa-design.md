# 六视频 speaker-consensus pruning 与多用户 QA 设计

日期：2026-08-20
状态：用户已批准设计，等待实现

## 1. 目标

在现有六用户 QA、judge、answerability 和 Qwen3.6-27B memory-safe 执行层上，将“双 anchor 星型裁剪”改成 speaker-centered 六视频共识裁剪：

\[
1\ \text{speaker}
+ 5\ \text{providers}
\rightarrow 6\ \text{pruned generator videos}
\rightarrow 1\ \text{QA}
\]

同一同步组固定按用户顺序依次遍历六个 speaker。每个 speaker 候选独立执行共识裁剪和最短保留时长检查；无论失败或成功，都记录结果并继续下一位 speaker，因此一个同步组可以产出多个成功候选。

本轮只修改六用户候选 pruning、媒体物化、packet、prompt、自动审核兼容层及对应 Torch 作业文档，不修改 GRPO、DPO、reviewer、optimizer、checkpoint 或训练合同。

## 2. 非目标与证据边界

- 不要求六位用户缺一不可。
- 不要求五个 providers 全部贡献答案。
- 不遍历全部用户子集做 answerability 搜索。
- 不修改二用户活跃路径及其 pair filter 兼容接口。
- 不计算、校验、冻结或保存版本哈希、文件哈希或媒体哈希。
- 不新增 baseline 或额外 Gate。
- 不把 runtime probe 通过表述为 QA 质量结论。
- 不把自动 reviewer 通过表述为人工确认的自然性或可回答性。
- 正式 pilot 最多尝试 40 个候选；得到 1–39 条 accepted QA 时只作为 `partial` 初步定性观察，不支持统计显著性结论。

## 3. 六用户候选与 speaker 尝试

### 3.1 同步用户采样

从同一同步 group 中按现有采样逻辑取六段视频。组内正好六人时全部保留；人数更多时延续现有随机数生成器与排序规则选择六人。

六用户路径不再计算全部 15 条 pair edge，不再使用 `relative_group_scores`、`score_video_pairs`、pair `kept/rejected`、双 anchor 或 additional-provider 角色选择。provider-provider similarity 不参与候选选择或裁剪。

### 3.2 多 speaker 合同

对选中的六位用户按已排序的固定顺序 `1, 2, 3, 4, 5, 6` 遍历 speaker。每次固定一位 speaker，其余五位均为 provider：

1. speaker 顺序不使用随机数，也不随机打乱；
2. 每位用户在同一同步组中最多作为 speaker 尝试一次；
3. 当前 speaker 的共识裁剪或媒体物化失败时，保存失败诊断并继续下一位 speaker；
4. 当前 speaker 成功时，保存一个独立候选并继续下一位 speaker；
5. 一个同步组可产生 0–6 个成功候选；全部 speaker 均失败时拒绝该 group，并保留六次失败原因。

### 3.3 用户顺序合同

每个候选的六用户顺序固定为：

1. `input_users[0]`：speaker；
2. `input_users[1:6]`：五个 providers。

`required_users` 保留为历史兼容字段，值与 `input_users` 相同并继续决定媒体顺序；在六用户模式中，它不表示六位用户都逻辑必要。

## 4. Speaker-consensus pruning

### 4.1 聚类表示

每个视频延续现有实现：

- 按现有固定采样间隔抽帧；
- 使用现有 CLIP frame embedding；
- 使用 deterministic cosine k-means；
- 请求 12 个 cluster；采样帧不足时实际数量为 `min(12, sampled_frame_count)`；
- 每个 cluster 使用现有 medoid 表示，不改用 centroid。

### 4.2 匹配与阈值

对当前 speaker 的每个 cluster，分别在五个 provider 的 cluster 中选择 similarity 最大的唯一 argmax cluster。只计算 speaker-to-provider 的五个 similarity matrix，不计算 provider-provider matrix。

阈值延续旧实现：

\[
\operatorname{high}(s,p)=\mathbf{1}[\operatorname{similarity}(s,p)\ge 0.82]
\]

若五个 provider argmax 中至少四个达到阈值，则创建联合删除事件。

### 4.3 4-of-5 删除范围

- `5-of-5`：删除 speaker cluster 和五个过阈值 provider argmax cluster。
- `4-of-5`：只删除 speaker cluster 和四个过阈值 provider argmax cluster；未过阈值的第五个 provider cluster 不删除。
- `3-of-5` 及以下：不创建删除事件。

每个事件记录五个 argmax、similarity、是否过阈值、过阈值数量和实际删除的用户/cluster。联合事件不要求六个视频每次都同时删除 cluster；`4-of-5` 明确只影响五段视频。

### 4.4 去重、区间与时长保护

- 同一 provider cluster 可被多个 speaker cluster 命中；marked cluster 按用户集合去重，只物理删除一次。
- 诊断保留全部触发来源，不因去重丢失事件 provenance。
- 删除 cluster 的全部 member frames，不只删除 medoid frame。
- member frames 使用现有区间 helper 转换、合并 remove intervals，并计算 keep intervals。
- 不启用 `max_pair_time_difference_seconds` 或其他时间接近限制，仅比较视觉 embedding。
- 若裁剪后任一视频低于现有最短保留时长，则拒绝当前 speaker 候选；不单独恢复某个 cluster，也不静默回退完整视频。

## 5. 媒体物化与路由

### 5.1 Generator 输入

六段视频都经过当前 speaker 候选的 consensus pruning 物化，并按 `[speaker, provider1, ..., provider5]` 输入 generator。即使某个 provider 没有任何 marked cluster，也仍物化其 pruned 输出；该输出内容可与原窗口等价，但角色统一记录为 consensus-pruned。

媒体模式改为 `six_pruned_videos`。每段 clip 至少记录：用户、位置、`media_role`、pruned path、full original path、marked clusters、remove/keep intervals、删除/保留时长和触发事件。

### 5.2 Judge 输入

- groundedness：同一顺序的六段完整原视频；
- answerability `speaker_only`：一段 speaker 完整原视频；
- answerability `combined_all_six_users`：六段完整原视频。

pruning 不改变 judge 的 full-video 路由，也不增加 provider-only 或其他子集调用。

## 6. Answerability 合同

六用户接受条件保持不变：

\[
\operatorname{Pass}
=
\neg\operatorname{Correct}(\{S\})
\land
\operatorname{Correct}(\{S,P_1,P_2,P_3,P_4,P_5\})
\]

- speaker-only 正确：拒绝；
- speaker-only 错误且 all-six 正确：通过；
- 任一条件无法解析：拒绝并按现有重试合同处理；
- all-six 错误：中性失败记录，不自动归因为噪声；
- `answerability_evaluated_condition_count=2`；
- accepted QA 的 `cross_view_gain=1`。

该合同只证明 speaker 需要外部视角且六视频输入可恢复正确答案，不证明五个 providers 都必要。

## 7. Prompt、schema 与 metadata

### 7.1 Prompt

Generator prompt 改为一名 speaker 加五名普通 providers：

- 从 speaker 的第一人称经历或共享场景提出自然问题；
- speaker 自身视频不能确定答案；
- 允许一个或多个 providers 支持答案；
- 不再描述 anchor/additional 角色、pair filter 或“3 pruned + 3 full”；
- 禁止声称所有 providers 都不可替代或只有六视频全集才能回答。

Groundedness judge 继续使用六段完整视频，检查问题动机、答案证据、supporting claims 和事实错误；未使用某个 provider 不是拒绝理由。

### 7.2 活跃六用户字段

六用户 packet 使用：

- `input_users`
- `required_users`
- `speaker_user`
- `provider_users`
- `evidence_provider_user`
- `evidence_provider_users`
- `media_roles`
- `speaker_consensus_pruning`
- `speaker_attempts`
- `generator_media_mode=six_pruned_videos`

活跃六用户输出不再生成：

- `anchor_provider_users`
- `additional_provider_users`
- `selected_anchor_edges`
- `diagnostic_pair_edges`
- `generator_media_mode=three_pruned_three_full_videos`

生成与审核继续记录 `speaker_only_correct`、`speaker_only_choice`、`all_six_correct`、`all_six_choice`、`cross_view_gain`、`answerability_evaluated_condition_count` 和 `supporting_user_claims`。

## 8. 测试驱动策略

### 8.1 纯算法

先写并确认失败的测试，再实现生产代码：

- `5-of-5` 触发，删除 speaker 加五个 provider clusters；
- 恰好 `4-of-5` 触发，只删除 speaker 加四个过阈值 provider clusters；
- `3-of-5` 不触发；
- 阈值边界 `similarity == 0.82` 计为通过；
- 每个 provider 只选 argmax，不删除其他超过阈值但非 argmax 的 cluster；
- 同一 provider cluster 被多个事件命中时只删除一次，诊断保留全部来源；
- 删除所有 cluster members，而非仅 medoid；
- 不计算 provider-provider matrix。

### 8.2 接入与媒体

- 一个同步组严格按 `1, 2, 3, 4, 5, 6` 遍历 speaker，不调用随机 speaker 选择；
- 当前 speaker 任一视频时长保护失败时记录原因并进入下一 speaker；
- 当前 speaker 成功时产出独立候选并继续下一 speaker；
- 单组可产生 0–6 个成功候选，全部失败时保存六次诊断并拒绝；
- 六段视频均产生 pruned generator path；
- 六段 full original path 保持存在；
- generator 顺序为 `1 speaker + 5 providers` 的 6 pruned；
- groundedness 顺序为同一六用户的 6 full；
- 二用户路径保持兼容。

### 8.3 Prompt、answerability 与 Torch 合同

- 六用户活跃 packet 不再包含 anchor/additional/edge 合同；
- Prompt 不再包含 anchor/additional 描述；
- answerability 仍只有 speaker-only 与 all-six 两次；
- speaker-only 错误且 all-six 正确时通过；
- Qwen3.6-27B、memory-safe backend、decord、显式视频像素上下界不被 pruning 修改；
- runtime probe 与 pilot40 审计 `six_pruned_videos` 和 consensus 字段；
- JobID 仍由 `sbatch --parsable` 自动写入时间戳 manifest。

### 8.4 本地验证

- 新增纯算法与六用户接入定向测试；
- 相关 prompt、video QA、runner 和 Torch 合同回归测试；
- Python `compileall`；
- 两个 `.sbatch` 的 `bash -n`；
- `git diff --check`。

本地验证不能证明 H100 runtime、模型能力、QA 质量或人工终点改善。

## 9. Torch 作业保持项

现有 runtime probe 与 pilot40 只同步更新 pruning/packet 审计字段，保留当前用户修改：

- Qwen3.6-27B memory-safe 路径；
- 强制 decord 视频后端；
- 显式视频像素下界与上界；
- job-specific `HOME`、cache、temp 和 scratch；
- 模型加载前 `storage_preflight.json`；
- `${SLURM_JOB_ID}` 派生输出目录；
- `sbatch --parsable` 和时间戳 submission manifest；
- probe 目标 1 条，pilot 最多 40 个候选及显式 `partial`；
- 现有 walltime 与吞吐依据；
- 不保存任何哈希。

设计与计划阶段不上传 Torch、不提交 Slurm、不取消已有任务。用户批准实施计划后，允许直接在 Torch 上完成窄同步与登录节点零 GPU 验证，然后在同一轮连续提交 runtime probe 和 pilot40。pilot40 使用 `afterok` 依赖 probe 的批处理成功状态；该依赖不替代产物验收。两个作业结果出来后再单独验收并决定后续；不再更新 Torch Runbook。

## 10. 产物与验收

至少保存：

- 输入 manifest；
- 六用户 candidates JSONL；
- 固定 speaker 遍历顺序与逐 speaker 成功/失败 trace；
- 每个 speaker cluster 的五个 provider argmax 与 threshold 结果；
- 联合删除事件、按用户去重的 marked clusters、member frames 和 intervals；
- 六段 generator pruned 与六段 judge full 媒体 provenance；
- prompts、intermediate、accepted 和 rejected JSONL；
- 两个 answerability condition traces；
- storage/environment/job manifests；
- CSV、generation report、human review sheet；
- stdout/stderr。

运行时验收只证明六视频 consensus-pruning 链路可执行。自动验收还要求作业正常、产物完整、计数一致、`accepted_count>0`，且 accepted QA 均满足 speaker-only 错误和 all-six 正确。人工终点评估必须查看六段完整视频，区分问题质量、证据不足、VLM 能力和媒体路由错误。

## 11. 工作区与范围保护

- 继续使用 `feature/multi-user-six-video-qa` 当前工作树；
- 保留现有未提交的 27B、decord、pilot40、Runbook 与远程约束修改；
- 不 reset、checkout、stash、clean 或覆盖用户改动；
- 原地更新现有设计、实施计划和 Runbook，不创建重复文档；
- 不 commit、不 push、不创建或修改远端分支；
- 不使用 SHA 或其他哈希作为身份、检查、manifest 或验收字段。
