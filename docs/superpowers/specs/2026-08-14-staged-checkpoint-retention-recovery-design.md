# 分段训练检查点保留与再次恢复设计

## 背景与故障结论

三个 epoch 2 恢复任务均从 `global_step=66` 正常恢复，Swift 返回码均为 0，并在退出前报告 `checkpoint-132`。外层脚本随后无法找到该目录，说明当前检查点在训练结束阶段被轮转逻辑删除，而不是训练、显存、时限或持久化复制失败。

恢复状态中的 `best_model_checkpoint` 指向上一分段任务的 scratch 路径，而分段训练当前使用 `--save_total_limit 1`。该组合会触发部分 Transformers 版本针对保留上限 1 的特殊结束清理路径，从而删除刚生成但不等于旧 best 路径的当前检查点。

## 修复范围

1. 只将 `hpc/grpo_v3/annotated_preference/staged_train.sbatch` 的 `--save_total_limit` 从 1 调整为 3。
2. 不修改普通训练、smoke 或 overfit probe 的保存策略。
3. 保留现有检查点合同：只有目标步数、适配器、优化器、调度器和 trainer state 全部存在时，才提升为正式持久化 checkpoint。
4. 保留上一轮失败恢复清单，不覆盖或删除历史记录；再次提交时生成新的时间戳清单和活动指针。
5. 新 epoch 3 任务继续通过 `afterok` 依赖各自的新 epoch 2 任务，禁止从失败的 epoch 2 继续。

## 选择保留上限 3 的理由

修复所需的关键条件是避开 `save_total_limit == 1` 的特殊清理分支。选择 3 比选择 2 留出更保守的余量，可同时容纳恢复状态中的旧 best、当前分段检查点以及可能存在的额外保存点。每个 staged 任务使用独立 scratch 输出目录，且分段步数受控，因此额外空间开销有界。

## 测试与验收

本地回归测试必须先证明当前值 1 不满足新合同，再验证：

- staged 脚本明确包含 `--save_total_limit 3`；
- 其他训练脚本不被意外修改；
- 恢复提交仍生成 epoch 2 到 epoch 3 的 `afterok` 依赖；
- 旧恢复记录得到保留，新恢复清单不会静默覆盖历史；
- 完整相关测试通过，且 `git diff --check` 无格式错误。

远端验收分两级：

- 运行验收：三个新 epoch 2 任务均产生正式 `checkpoint`，其中 `trainer_state.json.global_step == 132`，随后对应 epoch 3 才可启动。
- 最终验收：三个 epoch 3 均产生正式 `checkpoint`，其中 `trainer_state.json.global_step == 198`。Slurm 的 `COMPLETED` 本身不替代检查点合同验收。

## 非目标

- 不改写已有 `trainer_state.json`。
- 不清除 `best_model_checkpoint`。
- 不删除失败任务输出、日志或清单。
- 不以训练 reward 或 Slurm 成功状态选择最终学习率。
