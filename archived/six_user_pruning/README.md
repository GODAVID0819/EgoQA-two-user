# 六用户剪枝历史归档

`speaker_provider_argmax_consensus_4_of_5.py` 保留早期 speaker 中心的 provider argmax / 4-of-5 共识方法，仅作历史参考，不得从生产流程导入。

当前活跃方法位于 `group_relative_clip_sampling.py` 的 `clustered_speaker_provider_all_pairs_pruning`：只剪 provider，每个 provider cluster 与全部 speaker clusters 比较，任一相似度达到阈值即剪掉该 provider cluster。
