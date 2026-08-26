# 六用户 ZIP 时间感知双侧剪枝设计

## 决策

六用户正式候选路径采用 `multi-user.zip` 中
`temporal_kmeans_grid_sidecar.py` 与
`cross_user_temporal_gate_grid_sidecar.py` 的剪枝算法，不再使用当前纯视觉、
provider-only 的生产剪枝算法。ZIP 中的距离公式、初始化、cluster assignment、
中心更新、medoid 选择、跨用户时间差定义、保留时长保护和诊断字段均作为实现依据。

固定参数为：

- `time_weight=0.1`；
- `temporal_unit_seconds=30.0`；
- `max_iterations=25`；
- `high_similarity_threshold=0.82`；
- `cross_gap_mode="center"`；
- `max_cross_gap_seconds=10.0`；
- `pruning_protection_mode="min_percent"`；
- `min_pruned_video_percent=20.0`；
- `sample_interval_seconds=1.0`；
- `seconds_per_cluster=2.5`。

`G=10` 秒是未经跨用户 gate grid 和人工 QA 复核的生产初值。选择它是因为它位于
ZIP 候选网格 `5/10/15` 秒的中间，并把同一事件允许的主要时间偏移限制在默认
30 秒时间尺度的三分之一。该选择不能表述成实验最优值。

## 时间感知聚类

单视频内部使用 ZIP 中的 spherical K-means 距离：

\[
d^2(i,c)=2\left(1-\cos(e_i,e_c)\right)
+0.1\left(\frac{t_i-t_c}{30}\right)^2.
\]

初始化、assignment、归一化视觉均值、时间均值和 medoid 选择使用同一目标。
聚类数按完整输入窗口计算：

\[
K=\left\lceil\frac{D}{2.5}\right\rceil.
\]

因此 30 秒、180 秒、360 秒和 600 秒分别使用 `K=12/72/144/240`。长视频采用
ZIP sidecar 的全窗口聚类，不再按本地新增实现切成多个独立 30 秒 block。
`time_weight=0` 必须与原 `cluster_embedding_medoids` 的 labels 和 medoids 完全一致，
作为算法移植的兼容性测试。

## speaker-provider 双侧剪枝

对每个候选 speaker，分别对五个 speaker-provider pair 使用 ZIP 的 pair 剪枝：

1. speaker 和当前 provider 各自使用上述时间感知聚类；聚类结果可缓存复用。
2. 比较该 pair 的全部 cluster representative。
3. 一个 cluster pair 只有同时满足下式才被接受：

\[
\cos(e_s,e_p)\ge 0.82
\quad\land\quad
\left|\bar t_s-\bar t_p\right|\le 10\text{ 秒}.
\]

4. 接受的 pair 同时标记 speaker cluster 和 provider cluster，保持 ZIP 的双侧剪枝。
5. 每个视频按 ZIP 的 `min_percent` 保护恢复被标记帧，直到至少保留输入窗口的
   20%；恢复顺序继续使用 ZIP 的逐帧最佳纯 CLIP 匹配。
6. 每个 speaker-provider pair 使用 ZIP 的 `passed` 语义：两侧达到保护目标、
   `target_met` 为真且实际删除时长大于零。
7. 六用户 speaker 候选要求五个 pair 均成功；失败时保留每个 pair 的完整诊断，
   然后继续尝试下一个 speaker。

五次 pair 计算产生五份 speaker 剪枝诊断。它们不合并为一段实际 speaker 剪枝视频，
因为最终 QA 媒体合同要求 speaker 使用完整原视频。每个 provider 只属于一个
speaker-provider pair，因此直接使用该 pair 的 provider keep/remove intervals。

## QA 媒体路由

剪枝判定与最终 QA 媒体输入分离：

- speaker 在聚类、时间门、双侧标记、保留时长保护和 pair `passed` 判定中完整参与；
- 最终进入 QA 生成链路的 speaker 媒体始终是完整原视频，不物化或传入剪枝 speaker；
- provider 使用各自 speaker-provider pair 产生的剪枝视频；
- `speaker_only::<speaker>` 回答性条件继续使用 speaker 完整原视频；
- `combined_all_six_users::<user1+...+user6>` 回答性条件继续使用六段完整原视频；
- groundedness、answerability 和人工终点评估继续保留完整原视频来源，不以剪枝媒体
  替代证据审查输入。

因此，speaker-side 剪枝只影响 pair 是否满足 ZIP 剪枝合同及其诊断，不改变最终
QA 所见的 speaker 视频。

## 诊断与失败处理

每个 pair 保留 ZIP 字段，包括：

- `time_weight`、`temporal_unit_seconds` 和距离公式；
- representative similarity matrix；
- center、interval 和 medoid gap；
- gate eligible、accepted 和 rejected pair 计数；
- 左右 marked clusters、marked frames、restored frames；
- 左右 keep/remove intervals 和 retained/removed duration；
- protection target、`target_met` 和最终 `passed`。

六用户汇总层只增加 speaker/provider 索引、用户名、五个 pair 的状态和失败原因，
不重新解释或重算 ZIP 的 pair 诊断。没有 accepted pair、删除时长为零、任一侧未达到
20% 保留目标或媒体物化失败，都使当前 speaker 候选失败；不得静默回退到旧
provider-only 剪枝。

## 备份与修改边界

实现前把所有实际受影响且当前存在的本地源码、测试和设计文件按原相对路径复制到
一个时间戳备份目录，并记录每个文件的 SHA-256。备份目录不加入 Python import path、
测试发现路径或正式 Git 暂存范围。不得使用 `git reset`、`git checkout --` 或清理命令
处理当前 dirty worktree。

实现范围限于：

- 从 ZIP 提取并复用时间感知聚类、gap 和 pair 剪枝内核；
- 六用户候选挖掘、媒体物化和 CLI 参数传播；
- 相关六用户及 ZIP sidecar 测试；
- 本设计文档。

不修改 GRPO、DPO、reviewer、优化器、checkpoint、prompt judge 语义、两用户生产路径
或 Torch 作业；本轮不连接 Torch、不上传文件、不提交或取消 Slurm 作业。

## 最小测试集

1. `time_weight=0` 与原聚类 labels/medoids 完全一致。
2. `time_weight=0.1` 使用 ZIP 的时间感知初始化、assignment、中心和 medoid 规则。
3. `center gap=10` 接受边界内匹配，并拒绝超过 10 秒的高 CLIP 相似匹配。
4. 相似度恰好为 `0.82`、center gap 恰好为 `10` 秒时仍接受。
5. 接受的 pair 同时标记 speaker 和 provider clusters。
6. `min_percent=20` 按 ZIP 顺序恢复帧，并正确计算 pair `passed`。
7. 180 秒窗口使用一次全局 `K=72`，不调用本地 30 秒 blockwise 剪枝。
8. 每个 speaker 产生五个 pair 结果；任一 pair 失败时当前 speaker 失败并继续下一位。
9. provider 物化使用 pair 的右侧 keep intervals；speaker 始终路由完整原视频。
10. speaker-only 和 all-six 回答性条件继续分别使用一段和六段完整原视频。
11. 两用户路径以及 GRPO、DPO、reviewer、优化器和 checkpoint 模块无行为变化。

## 证据边界

本地单元测试只能证明算法移植、参数传播、媒体路由和输出合同符合 ZIP。它不能证明
`G=10` 优于其他候选值，也不能证明 QA groundedness、answerability 或生成质量改善。
在没有新的远端日志、产物和人工复核前，必须报告“远端未验证”。
