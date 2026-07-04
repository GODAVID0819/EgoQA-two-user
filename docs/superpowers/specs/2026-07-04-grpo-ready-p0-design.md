# GRPO-Ready P0 历史奖励回放设计

## 1. 目标与范围

本阶段只实现 `GRPO_READY_AGENT_IMPLEMENTATION_CN.md` 中的 P0：从
`outputs/qa_mcq.intermediate.jsonl` 恢复全部历史生成尝试，按 Reward v0
计算可观测奖励，并生成机器可读与人工可读的回放结果。

本阶段不修改现有生成主循环，不实现在线采样、冻结 evaluator 或 GRPO
训练。P1 后续直接复用本阶段的记录结构与纯奖励函数。

## 2. 方案选择

采用独立模块方案：新增 `grpo_ready` 包，将历史数据提取、奖励计算和结果
汇总与 `video_qa_loop.py` 隔离。只复用现有 JSON 解析、schema 校验和
answerability 语义，不重构现有推理流水线。

未采用的方案：

- 直接扩展 `video_qa_loop.py`：会把离线分析与在线生成耦合，回归风险高。
- 一次性实现 P0/P1/P2：远端媒体和 GPU 尚未验证，会阻塞本地可验收的 P0。

## 3. 模块边界

- `grpo_ready/records.py`：定义稳定、可序列化的 AttemptRecord 与 RewardRecord。
- `grpo_ready/extract_attempts.py`：逐 packet 恢复 attempt；`accepted` 只读取
  attempt 自身的最终决定，不继承 packet 状态。
- `grpo_ready/rewards.py`：无文件 I/O 的 Reward v0 纯函数；缺失 evaluator
  组件保持 missing，不伪造成 FAIL。
- `grpo_ready/replay.py`：CLI 编排、JSONL/CSV 写入、统计与矛盾案例选择。

## 4. 数据流

1. 逐行读取 intermediate JSONL，不一次性复制完整数据集。
2. 对每个 packet 遍历历史 attempts，并恢复 generation raw output。
3. 解析 raw QA，执行 schema 校验，提取 formality、groundedness 与
   answerability 观测。
4. 计算各奖励分项、observed total、complete-case 标记和缺失组件。
5. 输出逐 attempt 结果，再按 accepted/rejected、失败组件和标签矛盾汇总。

## 5. 奖励与缺失值语义

Reward v0 为：

\[
R=R_p+R_s+R_f+R_g+R_c+R_l.
\]

只有实际观测到的组件才计入 observed total。完整奖励要求 parse、schema、
formality、groundedness、combined 与 speaker leakage 六项均可判定；否则
`is_complete_reward=false`，并在 `missing_components` 中列明缺失项。

parse 失败时只计算 parse 分数，所有依赖 QA 语义的组件标为 unavailable。
evidence provider 单独答对记 0 分，不视为泄漏；speaker 单独答对记 -2 分。

## 6. 输出与错误处理

固定输出：

- `reward_replay_results.jsonl`
- `reward_replay_results.csv`
- `reward_replay_summary.json`
- `reward_replay_summary.md`
- `run_manifest.json`

输入文件不存在、JSONL 行损坏、attempt 缺 raw generation 或计数不满足预期时，
CLI 以非零状态退出并报告 evidence/attempt 标识，不静默跳过。统计时将完整奖励
与不完整奖励的覆盖率分开报告。

## 7. 测试与验收

严格按测试驱动顺序实现：

1. 先用最小构造数据覆盖 accepted attempt、rejected generation trace 和缺失字段。
2. 再锁定 Reward v0 各分项及 missing/unavailable 语义。
3. 最后在真实 intermediate 文件上执行集成测试，验证 55 packets、121 attempts、
   27 accepted attempts以及全部 raw generation 非空。
4. 运行 Replay CLI，验证五类输出存在、可重新解析且统计自洽。

完成标准是 P0 测试全部通过并生成真实回放产物；结论仅说明历史数据已具备
GRPO-ready reward replay 能力，不宣称模型质量提升。
