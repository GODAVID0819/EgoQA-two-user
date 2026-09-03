# 六用户高质量 Generator 单次生成 30 槽实验设计

## 1. 目标与实验问题

本实验用于回答：在完全复用既有 candidate mining 结果的前提下，仅提高 generator 所见视频的时间与空间质量，是否能改善六用户十分钟 QA 的生成质量，并缩短因反复重试造成的总耗时。

实验固定生成 30 个槽位，而不是追求 30 个 accepted QA。每个槽位只调用一次 generator；随后 judge 给出评审结果，但评审失败不再反馈给 generator，也不触发重生成。generator 解析失败和 judge rejected 都保留为正式统计样本并计入固定分母 30。

本实验不重新执行视频下载、CLIP candidate mining、speaker/provider pruning 或 candidate materialization，也不修改 judge 的视频下采样设置。

## 2. 输入产物与边界

### 2.1 主输入

主输入固定来自 Job `16699348`：

`/scratch/xl6775/projects/EgoQA-two-user-six-user-10min-speed-qwen38-fix-20260901/outputs/six_user_qa/six_user_qa_10min_3groups_x20_qwen38_fast_fix_16699348/candidate_assets/`

使用其中三个已完成的 group asset：

- `DAY1_17200000_group_relative_clip.json`
- `DAY3_17000000_group_relative_clip.json`
- `DAY4_21400000_group_relative_clip.json`

每个 asset 的 `speaker_candidates` 包含六个成功 speaker candidate，因此主输入总计为：

\[
3\ \text{groups} \times 6\ \text{speakers}=18\ \text{speaker packets}
\]

直接位于输出根目录的 `six_user_candidates.jsonl` 只有每组一个顶层 speaker packet，不能作为本实验的完整输入。

### 2.2 备份输入

Job `16794616` 的 `DAY4_21400000_group_relative_clip.json` 含六个可用 speaker candidate，仅作为 DAY4 主 asset 无法读取时的人工确认后备份。本实验默认不混用两个 Job 的 asset，也不自动切换来源。

### 2.3 不在范围内

- 不修改 candidate mining、相似度计算或 pruning 语义。
- 不追求达到某个 accepted 数量后继续补题。
- 不把 judge 反馈送回 generator。
- 不改变 Answerability 的两次调用结构、同步 facts 合同或简单版 Evidence 判断语义。
- 不取消、修改或复用现有 Job `16820668` 的输出目录。
- 不用本实验结果直接宣称 benchmark 质量已经提升；本实验只提供同一流程下的生成质量与耗时观测。

## 3. 紧凑 evidence 恢复

新增一个独立、可测试的恢复步骤，逐个读取三个大型 group asset，从 `speaker_candidates` 提取 18 个紧凑 packet。每次只在内存中处理一个 group asset，避免同时加载三个约 0.8–0.9 GB 的 JSON。

紧凑 packet 必须保留：

- `generation_group_id`、`evidence_id`、`speaker_index`、`speaker_user`；
- 六用户稳定顺序及其身份；
- generator 所需的六路完整视频路径；
- judge 所需的视频路径；
- 每条 evidence 的原始视频时间映射；
- pruning 后的必要证据结构；
- 来源 Job、来源 asset 和原 candidate 标识等最小 provenance。

恢复文件不复制体积巨大的相似度矩阵、逐帧诊断、重复下载日志或其他 generator/judge 不消费的字段。恢复时验证每组恰有六个不同 `speaker_index`，speaker 身份与视频路径齐全，且所有引用视频非空并存在；任一 group 不满足时停止构建 30 槽输入，不静默降级为少于六个 speaker。

## 4. 30 槽 speaker 调度

恢复出的 18 个 packet 将预展开成一个固定 30 行的输入 JSONL。运行时按普通顺序读取这 30 行，不使用 `repeat_evidence`，也不依赖“accepted 后补槽”的动态调度。

每组生成 10 槽。组内六个 speaker 先各获得一槽，再由四个 speaker 各获得第二槽。三个 group 使用不同起点，使全局六个 speaker 各生成五槽：

| Group | 10 槽 speaker 顺序 |
|---|---|
| `DAY1_17200000` | Jake, Alice, Tasha, Lucia, Katrina, Shure, Jake, Alice, Tasha, Lucia |
| `DAY3_17000000` | Katrina, Shure, Jake, Alice, Tasha, Lucia, Katrina, Shure, Jake, Alice |
| `DAY4_21400000` | Tasha, Lucia, Katrina, Shure, Jake, Alice, Tasha, Lucia, Katrina, Shure |

因此每个 group 的 speaker 槽数分布为 `2,2,2,2,1,1`，全局分布为：

| Speaker | 固定槽数 |
|---|---:|
| Jake | 5 |
| Alice | 5 |
| Tasha | 5 |
| Lucia | 5 |
| Katrina | 5 |
| Shure | 5 |

每行必须有唯一且稳定的 `generation_slot_id`，并显式绑定 group、speaker packet 和组内槽序号。即使两个槽复用同一 speaker packet，也仍是两个独立 generator 样本。

## 5. 单次生成与评审数据流

每个槽位执行以下单向流程：

\[
\text{预展开 packet}
\rightarrow \text{generator 一次调用}
\rightarrow \text{解析}
\rightarrow \text{judges}
\rightarrow \text{记录最终状态}
\]

具体合同如下：

1. `MAX_ATTEMPTS=1`，不进行 generator retry。
2. generator 输出继续保留问题、A–E 选项、声明答案、证据细节和原始时间证据；删除已经确认无消费者价值的 `why_two_users_needed`。
3. generator 输出无法解析时，记录原始输出、解析错误、槽位身份和耗时；该槽标为 `parse_failed`，不伪造 QA，也不补生成。
4. 合法 QA 进入题面形式、Evidence、speaker-only Answerability、all-six Answerability 和最终 gate。
5. Answerability 保留两次调用，并确保第二次调用消费与第一次完全同步的 `canonical_facts`，避免两次调用事实合同不一致。
6. Evidence 统一走当前简单版单次判断，所有入口使用同一实现和同一结果字段。
7. judge 拒绝只改变槽位最终分类，不触发新 generator 调用。
8. 为获得完整统计，关闭 `FAIL_FAST_REVIEW`；各 judge 可在合同允许时并行，但所有适用 judge 结果都必须落盘。

## 6. 媒体质量与模型配置

generator 与 judge 复用同一个已加载的 Qwen3.8-27B 模型实例，但使用阶段级媒体配置，避免加载两份模型：

| 阶段 | FPS | 最大像素数 | Thinking |
|---|---:|---:|---|
| generator | 0.5 | 131072 | 关闭 |
| judges | 0.25 | 65536 | 关闭 |

阶段级媒体配置必须参与媒体缓存键或使用隔离缓存，防止 generator 的高质量媒体结果被 judge 错用，或 judge 的低质量缓存反向污染 generator。

除视频采样质量外，generator 的模型、prompt 语义、JSON 合同和解码策略保持当前 Qwen3.8 fast-fix 基线；judge 的 prompt、下采样、Answerability 两调用和 Evidence 简单版合同保持不变。

## 7. 输出与统计

最终输出必须能够按 `generation_slot_id` 一一对齐 30 个预展开输入，并至少区分：

- `generated_valid_json`
- `parse_failed`
- `accepted`
- `rejected_by_formality`
- `rejected_by_evidence`
- `rejected_by_answerability`
- 其他明确记录的运行失败

核心分母固定为 30：

\[
\text{valid generation rate}=\frac{N_{\text{valid JSON}}}{30}
\]

\[
\text{acceptance rate}=\frac{N_{\text{accepted}}}{30}
\]

还需按 group 和 speaker 汇总有效生成率、接受率、主要拒绝原因及耗时。accepted 数量是结果，不是终止条件。

## 8. 耗时观测

为判断 one-shot 与高质量 generator 的实际代价，至少记录：

- 三个 asset 的紧凑恢复耗时和 30 行展开耗时；
- 每槽 generator 媒体准备耗时；
- 每槽 generator 调用耗时与解析耗时；
- 每个 judge 调用耗时；
- 每槽从开始生成到最终 gate 完成的总耗时；
- 30 槽作业 wall-clock 总耗时。

与旧任务比较时，优先报告“每个 generator 槽”的平均值、中位数、最小值和最大值；不得把 accepted QA 的耗时与固定槽耗时混为同一指标。旧参考值约为 Qwen3.6 reasoning `67 分钟/final QA slot`、Qwen3.8 fast fix `35 分钟/final QA slot`、Qwen3.8 原始 fast `69 分钟/final QA slot`，比较时必须注明旧值统计口径与本实验是否完全一致。

## 9. 错误处理与可恢复性

- asset 恢复失败：保留来源路径、group、异常和已完成检查；不启动 GPU 正式作业。
- 单槽 generator 或解析失败：记录失败并继续后续槽，不补题。
- 单个 judge 调用失败：记录对应 judge 的运行错误；不将缺失结论推断为 pass 或 fail，也不重新生成。
- 作业级异常：保留已完成槽位和时间记录，允许后续诊断，但续跑不得覆盖原 JobID 产物。
- 所有正式提交使用新的 JobID 和 JobID 派生输出目录；不覆盖 `16699348`、`16794616` 或 `16820668`。

## 10. 验证策略

实现阶段采用针对性测试覆盖以下合同：

1. 从合成 group asset 恢复六个不同 speaker packet，并剔除非必要巨型诊断字段。
2. 三组预展开后恰有 30 行、30 个唯一 slot，每组 10 行、每 speaker 全局 5 行。
3. `MAX_ATTEMPTS=1` 时 rejected 和 parse failure 均不触发重生成。
4. generator 与 judge 分别使用 `0.5/131072` 和 `0.25/65536`，且媒体缓存互不污染。
5. Answerability 两次调用接收相同 `canonical_facts`；Evidence 所有入口统一使用简单版。
6. 30 个槽全部有最终状态或明确运行错误，统计分母始终为 30。
7. 现有六用户任务合同不因新增 one-shot 入口而回归。

本地测试和登录节点零 GPU 检查只能证明数据合同、导入、路径与调度逻辑可执行，不能证明 GPU runtime、速度提升或 QA 质量。正式作业的 runtime 日志和 30 槽产物才用于速度与质量结论。

## 11. 完成标准

本设计的实现完成标准为：

- 不重跑 mining，成功从 `16699348` 恢复 18 个紧凑 speaker packet；
- 生成严格均衡的 30 行固定输入；
- 正式作业对每槽最多调用一次 generator，并使用约定的阶段级视频质量；
- 每个槽均保留生成、评审、状态和耗时记录；
- 最终报告以 30 为固定分母，同时给出按 group、speaker 和失败类型拆分的质量与耗时统计；
- 任何提交均使用新 JobID，不影响既有作业和产物。
