# 奖励候选 QA 双作业收集设计

## 目标

在 Torch 上并发提交两个单 H100 作业，从 DAY1–DAY4 的 06:00–18:00 时间窗收集共 100 个奖励模型候选 evidence packet。两个作业各处理 50 个 evidence_id，每个作业的 Slurm 时限为 15 小时。

## 时间与数量划分

- 作业 1：06:00（含）至 12:00（不含），50 个 evidence_id。
- 作业 2：12:00（含）至 18:00（不含），50 个 evidence_id。
- 两个作业独立并发运行，使用不同的窗口标签和输出目录，不共享可变输出。
- `REWARD_TOTAL_PACKET_COUNT=100` 表示总评估 evidence packet 目标，不保证产生 100 条被下游质量门接受的 QA。

## 文件布局

- `hpc/qa/rlhf/run_reward_candidate_collection_qwen36_27b.sbatch`：单个时间半区间的 worker。
- `hpc/qa/rlhf/submit_reward_candidate_collection_two_halves.sh`：计算时间中点与数量划分并提交两个 worker。
- `hpc/shared/cuda.py`：CUDA 利用率 keeper 基础实现。
- `hpc/shared/cuda_slurm.py`：Slurm 可见设备到 NVML 设备的映射适配层。
- `hpc/shared/env_qwen3vl.sh`：Qwen/CLIP Conda 与缓存环境引导。

worker 和提交器均从自身真实路径推导项目根目录；worker 显式从 `hpc/shared/` 加载三个共享运行时文件。

## 运行与输出约束

- 资源：每个作业 1 节点、1 个任务、8 CPU、1 张 H100、128 GB 内存、15 小时。
- 模型：默认 `Qwen/Qwen3.6-27B`。
- 两个输出目录由带 `part1`/`part2` 和时钟范围的唯一标签区分。
- 运行前检查 Conda、CUDA、FFmpeg、模型依赖、双人同步证据数量和媒体完整性。
- 运行后校验 evidence packet 数、每包候选数、分组汇总与 `collection_summary.json` 一致性。

## 验证

本地只执行不会启动 GPU 作业的静态验证：

1. `bash -n` 检查两个 Shell/Slurm 脚本语法。
2. Python 编译检查 `cuda.py` 与 `cuda_slurm.py`。
3. 使用 `REWARD_SUBMIT_DRY_RUN=1` 验证输入命令被拆成两个 50-packet 作业，窗口分别为 06:00–12:00 与 12:00–18:00。
4. 检查 worker 的 `#SBATCH --time=15:00:00`、共享运行时路径和输出路径。

这些检查只能证明本地脚本结构与参数拆分正确；远程同步、环境可用性、Slurm 完成状态和候选产物仍需在 Torch 上按 JobID 验证。
