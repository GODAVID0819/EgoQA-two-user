# 六用户 3-of-5 共识裁剪与 speaker-only 回答性设计

## 目标

在现有六用户 speaker-consensus 路径上做三项窄范围调整：

1. 保留现有阻塞式 QA formality gate，不重复实现，并增加一项六用户回归检查。
2. speaker cluster 的删除触发条件由至少四个 provider 过阈值改为至少三个。
3. 六用户 answerability 只评估 speaker 的完整原视频；speaker 选错即通过，不再调用 all-six answerability judge。

两用户旧路径、evidence groundedness、生成媒体路由和六 speaker 固定顺序遍历保持不变。现有 Torch 作业不取消、不修改。

## 共识裁剪合同

每个 speaker cluster 分别在五个 provider 中寻找一个相似度最高的 provider cluster，相似度阈值继续使用 `>= 0.82`。

- 0、1、2 个 provider 过阈值：不触发删除。
- 3 个 provider 过阈值：删除 speaker cluster 和这三个 provider 的对应 cluster。
- 4 个 provider 过阈值：删除 speaker cluster 和这四个 provider 的对应 cluster。
- 5 个 provider 过阈值：删除 speaker cluster 和全部五个 provider 的对应 cluster。

未过阈值的 provider cluster 不删除。同一 cluster 被多个事件命中时只物理删除一次，但保留全部事件来源。任何视频违反最短保留时长时，当前 speaker 候选失败并继续下一个 speaker。

## 回答性合同

六用户只生成一个 answerability condition：`speaker_only::<speaker>`。该 condition 使用 speaker 的完整原视频。

- 返回正确选项：gate 失败，标签为 `speaker_only_correct`。
- 返回其他合法选项：gate 通过。
- 未返回合法的 A–E：gate 失败，标签为 `speaker_only_unparsed`。
- 缺少 speaker-only 结果：gate 失败，标签为 `speaker_only_missing`。

新六用户结果保留 `speaker_only_choice`、`speaker_only_correct` 和 `answerability_evaluated_condition_count=1`，不再产生 all-six 和 cross-view-gain 专属指标。

QA formality 继续使用现有 text-only 模型分支、确定性 schema 分支和五个语义子检查；任一分支失败仍阻塞 QA。Evidence groundedness 继续使用六个完整原视频。

## 实现边界

修改范围限定为：

- `group_relative_clip_sampling.py`
- `video_qa_loop.py`
- 与六用户行为直接相关的测试
- 六用户 runtime probe 与 pilot 的结果验收字段

不修改 GRPO、DPO、reviewer、优化器、checkpoint、两用户回答性合同或既有 Torch 作业。新候选标识、文件名和 manifest 使用可读字段，不引入哈希标识。

## 最小测试集

测试保持精简，只覆盖能够区分新旧合同的边界：

1. 3-of-5 触发，且只删除 speaker 与三个过阈值 provider cluster。
2. 2-of-5 不触发。
3. 六用户只生成一个 speaker-only condition。
4. speaker 选错通过、选对失败、不可解析失败。
5. runner 只调用一次且只接收 speaker 完整原视频。
6. QA formality FAIL 仍出现在阻塞失败中。
7. Torch 脚本记录一次 answerability 调用，并删除 all-six 专属验收字段。

## Git 与实验边界

开发分支为 `feature/six-user`。只提交本功能需要的文件，保留 worktree 中无关 dirty 改动。完成本地验证后推送该分支；推送后只汇报分支名、文件和测试证据，不使用提交哈希作为实验或交付标识。

在用户批准后续实验计划前，不同步新代码到 Torch，不提交新作业，也不取消现有作业。
