# 六用户 provider-only 剪枝与双重回答性验证设计

> 历史合同：本设计中的 provider-only 剪枝已被
> `2026-08-27-six-user-zip-temporal-pruning-design.md` 取代。回答性完整原视频合同继续有效。

## 目标

对当前六用户 QA 路径做两项调整：

1. 回答性 Gate 同时验证 speaker-only 与六段完整原视频；只有 speaker-only 选错且 all-six 选对时才通过。
2. 当前生产剪枝改为 speaker 与所有 provider cluster 的全量两两比较；只删除达到阈值的 provider cluster，永不删除 speaker cluster。

现有 QA formality、evidence groundedness、六个 speaker 的固定顺序遍历、最短保留时长约束和两用户路径保持不变。现有 Torch 作业不修改、不取消。

## 回答性合同

六用户回答性评估固定生成两个 condition：

- `speaker_only::<speaker>`：只提供 speaker 的完整原视频。
- `combined_all_six_users::<user1+...+user6>`：按既有用户顺序提供六段完整原视频。

Gate 按以下规则判定：

- speaker-only 返回声明的正确选项：失败，标记 `speaker_only_correct`。
- speaker-only 返回其他合法 A–E 选项：speaker-only 子条件通过。
- speaker-only 缺失或不可解析：失败，分别标记 `speaker_only_missing` 或 `speaker_only_unparsed`。
- all-six 返回声明的正确选项：all-six 子条件通过。
- all-six 返回错误选项、缺失或不可解析：失败，分别记录错误、缺失或不可解析原因。
- 两个子条件均通过时，回答性 Gate 才通过。

结果保留两次评估、两次调用记录、媒体路径、耗时、`speaker_only_choice`、`speaker_only_correct`、`all_six_choice`、`all_six_correct` 和 `answerability_evaluated_condition_count=2`。QA formality 仍是独立阻断 Gate。

## 当前 provider-only 全量剪枝

每个 speaker 候选分别执行以下流程：

1. 对 speaker 和五个 provider 各自聚类，默认每段视频得到 12 个 cluster。
2. 将 speaker 的 12 个 cluster 与五个 provider 合计 60 个 cluster 全量比较，共计算 \(12\times60=720\) 组余弦相似度。
3. 任一 provider cluster 只要与任一 speaker cluster 满足 `similarity >= high_similarity_threshold`，就标记该 provider cluster。
4. speaker cluster 不标记、不删除；不存在 3-of-5 触发条件，也不只看 provider 中的 argmax。
5. 同一 provider cluster 被多个 speaker cluster 命中时只删除一次，但在审计事件中保留所有达到阈值的 speaker 匹配及相似度。
6. 没有 provider cluster 达到阈值时，当前 speaker 尝试失败并记录原因；裁剪后任一视频低于现有最短保留时长时同样失败。无论成功或失败，都继续按 `1→2→3→4→5→6` 尝试后续 speaker。

生产结果使用明确的新 method 标识，避免将其误写成共识剪枝。

## 保留 3-of-5 备用实现

现有 `clustered_speaker_consensus_pruning` 函数及其 3-of-5 测试继续保留，作为可显式调用的备用策略。该函数仍按原合同执行：每个 speaker cluster 在每个 provider 内选择 argmax，至少三个 provider 达到阈值后，删除 speaker cluster 与达到阈值的 provider argmax cluster。

新增独立的 provider-only 全量配对函数，当前六用户生产路径只调用新函数。两套算法不通过隐式环境变量或隐藏分支混合，调用方可从函数名和 method 字段明确区分。

## 实现范围

预计修改：

- `group_relative_clip_sampling.py`：新增 provider-only 全量剪枝函数并切换当前六用户生产调用；保留旧共识函数。
- `video_qa_loop.py`：恢复 all-six 回答性 condition、第二次 judge 调用和双条件 Gate。
- `tests/test_six_user_group_relative_sampling.py`：保留旧 3-of-5 测试，新增新剪枝的最小回归测试。
- `tests/test_six_user_video_qa_loop.py`、`tests/test_six_user_torch_job_contract.py`：更新双重回答性合同和调用计数测试。
- 与当前六用户 probe/pilot 验收字段直接相关的既有脚本：仅在其仍写死单次回答性调用时做必要同步。

不修改 GRPO、DPO、reviewer、优化器、checkpoint、两用户回答性合同或现有 Torch 作业。不引入 hash 标识。

## 最小测试集

1. 六用户生成 speaker-only 与 all-six 两个 condition。
2. speaker-only 选错且 all-six 选对时通过。
3. speaker-only 选对、任一结果缺失或不可解析、all-six 选错时分别失败。
4. runner 调用两次，媒体路径分别为 speaker 完整原视频和六段完整原视频。
5. 非 argmax 的 provider cluster 达到阈值时也会删除。
6. 相似度恰好等于阈值时删除对应 provider cluster。
7. 多个 speaker cluster 命中同一 provider cluster 时删除去重，并保留匹配来源。
8. speaker 的任何 cluster 都不删除。
9. 旧 3-of-5 备用函数的现有测试继续通过。
10. QA formality 失败仍阻断六用户 QA。

## Git 与实验边界

开发分支固定为 `feature/six-user`。只暂存和提交本次相关文件，保留 worktree 中已有的无关 dirty 改动。代码与本地测试完成后推送该分支；交付不使用提交哈希作为实验或结果标识。本轮不连接 Torch、不提交新作业、不取消任何已提交作业。
