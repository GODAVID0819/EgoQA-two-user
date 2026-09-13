# 时间维度 K-means 剪枝 Sidecar 实验

## 目标与隔离边界

该实验只改变单个视频内部的聚类距离，不修改现有
`cluster_embedding_medoids`、`clustered_frame_representatives` 或
`clustered_temporal_similarity_pruning`。跨视频 medoid 匹配仍使用纯 CLIP cosine
similarity，因此原有 `0.82` 阈值的含义不变。

Sidecar 距离为：

\[
d^2(i,c)=2(1-\cos(e_i,e_c))
+w\left(\frac{t_i-t_c}{S}\right)^2
\]

- `w`：`time_weight`，时间权重；`w=0` 是当前实现的精确对照。
- `S`：`temporal_unit_seconds`，默认 30 秒。相差 `S` 秒时，时间项等于
  `w`；相差越小，惩罚快速接近零。
- CLIP center 使用归一化均值；time center 使用时间均值。初始化、分配和
  medoid 选择均使用同一距离。

这是一条软约束。它减少“视觉相似但相隔很远的帧被同一 medoid 连带剪除”，
但不会禁止相隔很远的两个跨视频 medoid 发生纯 CLIP 匹配。后者是独立实验，
不应与本实验同时改变。

## 固定 cohort 与默认网格

实验先从 manifest 中寻找完整的 10 分钟同步序列，固定抽取 50 对视频。每对
视频只采样和 CLIP encode 一次 600 秒窗口，随后使用嵌套前缀评估：

- 30 秒；
- 180 秒（3 分钟）；
- 360 秒（6 分钟）；
- 600 秒（10 分钟）。

因此四个 duration 使用完全相同的 50 对视频，不存在 duration 之间的 cohort
漂移。默认每秒采样一帧。

默认网格：

- `time_weight`: `0,0.1,0.25,0.5,1,2,4`；
- `seconds_per_cluster`: `2.5`，即每 30 秒固定 12 个 clusters；
- CLIP pruning threshold: `0.82`；
- `temporal_unit_seconds`: `30`。

`K = ceil(duration / seconds_per_cluster)`。固定 `2.5` 后，30 秒、3 分钟、
6 分钟和 10 分钟分别使用 `K=12,72,144,240`，保持与当前 30 秒 `K=12`
相同的 cluster intensity。默认每对视频共有 `4 × 7 = 28` 个配置，50 对共有
1400 个结果。CLI 仍允许显式传入其他 density，但标准实验不再 sweep 它。

## 输出与参数选择

主要输出：

- `summary.json`：cohort、冻结条件、配置数和跳过原因；
- `cohort.jsonl`：50 对固定视频及 embedding cache；
- `grid_metrics.csv/jsonl`：逐 pair、duration、K、time weight 的结果；
- `aggregate_metrics.csv/jsonl`：跨 50 对聚合及相对 `w=0` 的 matched delta；
- `summary.html`：时间泄漏与视觉一致性的 Pareto 表；
- `pairs/*/diagnostics/`：默认前三对视频的完整 cluster/pruning trace；
- `pairs/*/embedding_cache.npz`：600 秒 CLIP embedding cache，重跑网格时复用。

选择参数时优先同时检查：

1. `quarter_duration_gap_reduction_vs_w0` 是否为正：距离 medoid 超过窗口
   25% 的 cluster membership 是否下降；
2. `pruned_quarter_duration_gap_reduction_vs_w0` 是否为正：被剪帧中的同类
   远距离 membership 是否下降；
3. `visual_similarity_delta_vs_w0` 是否可接受：不能用纯时间切片换取虚假的改善；
4. `mean_removed_percent`、`pass_rate`、`no_removal_rate`：剪枝不能失效或塌缩；
5. 不同 duration 上是否稳定，而不是只适配 30 秒。

同时保留 `member_gap_reduction_vs_w0` 和 `pruned_gap_reduction_vs_w0`，它们使用
固定的 `temporal_unit_seconds`（默认 30 秒），用于比较长窗口上的绝对时间泄漏。
相对窗口指标保证 30 秒条件不会因为最大 gap 小于 30 秒而失去区分度。

`summary.html` 的绿色行是在“更低的相对远距离 membership”和“更高的视觉一致性”
两项上不被其他 time weight 支配的配置。它是候选集，不是自动研究结论。

## 运行

### 本地到集群的同步映射

假设集群 `PROJECT_ROOT=/scratch/${USER}/Long-video-understanding-clip`：

必需同步：

| 本地（相对 `multi-user/`） | 集群目标 |
|---|---|
| `temporal_kmeans_grid_sidecar.py` | `${PROJECT_ROOT}/egolife_two_user_qa/temporal_kmeans_grid_sidecar.py` |
| `hpc/qa/preprocessing/run_temporal_kmeans_grid_50.sbatch` | `${PROJECT_ROOT}/hpc/qa/preprocessing/run_temporal_kmeans_grid_50.sbatch` |
| `hpc/shared/cuda.py` | `${PROJECT_ROOT}/hpc/shared/cuda.py`（若集群已有同版本可只校验） |

推荐同步以便集群复验和查阅：

| 本地（相对 `multi-user/`） | 集群目标 |
|---|---|
| `tests/test_temporal_kmeans_grid_sidecar.py` | `${PROJECT_ROOT}/egolife_two_user_qa/tests/test_temporal_kmeans_grid_sidecar.py` |
| `docs/others/TEMPORAL_KMEANS_GRID_SIDECAR.md` | `${PROJECT_ROOT}/egolife_two_user_qa/docs/others/TEMPORAL_KMEANS_GRID_SIDECAR.md` |

Launcher 会在 storage 和视频 runtime preflight 之后启动
`${PROJECT_ROOT}/hpc/shared/cuda.py`，启动失败时 fail fast，并通过 exit trap 在作业
结束时停止 keeper。日志写入当前 JobID 对应的输出目录。

集群推荐入口：

```bash
cd /scratch/${USER}/Long-video-understanding-clip
sbatch hpc/qa/preprocessing/run_temporal_kmeans_grid_50.sbatch
```

默认结果目录为
`${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/temporal_kmeans_grid_50/job_${SLURM_JOB_ID}`。
仍可通过 `EGOLIFE2U_OUTPUT_ROOT` 显式覆盖根目录。

### 超时恢复

若旧作业超时但 `experiment/pairs/*/embedding_cache.json` 和 `.npz` 已存在，
新作业可以复用这些缓存：

```bash
OLD_PAIRS=/scratch/${USER}/Long-video-understanding-clip/egolife_two_user_qa/multi-user/outputs/temporal_kmeans_grid_50/job_<OLD_JOB_ID>/experiment/pairs
sbatch --export=ALL,TEMPORAL_KMEANS_RESUME_PAIRS_DIR="${OLD_PAIRS}" \
  hpc/qa/preprocessing/run_temporal_kmeans_grid_50.sbatch
```

第一次从旧版超时作业恢复时，已缓存 pair 的 grid 指标需要从 embedding
重新计算，但不会重新下载视频、运行 FFmpeg 或运行 CLIP encoder。新版作业会在
每个 pair 完成后原子更新 `progress.json`、`cohort.jsonl`、`grid_metrics.*`、
`aggregate_metrics.*` 和 `summary.html`，并在 pair 目录写入
`pair_complete.json`。若新版作业再次超时，把下一次恢复路径指向最新作业的
`experiment/pairs`；匹配当前 grid 的已完成指标会直接载入，不再重复计算。

`hpc/shared/cuda.py` 在导入 PyTorch 前会从模块搜索路径中移除其自身目录；
启动器也遵循现有 runtime-probe 作业的方式，通过 `python -P` 启动该脚本。
两层防护共同避免新版本 PyTorch 导入 NVIDIA `cuda.bindings` 时被同名脚本遮蔽并形成循环导入。

启动器会依次从显式 `FFMPEG_ENV`、当前激活的 Conda 环境、以及 `PATH`
解析同一运行时中的 `ffmpeg`/`ffprobe`。不再要求默认的
`/scratch/${USER}/envs/egoqa-ffmpeg-runtime` 必须存在；如果三个来源都不可用，
作业会在加载模型前给出检查过的路径和修复提示并退出。

直接运行 sidecar：

```bash
python -m egolife_two_user_qa.temporal_kmeans_grid_sidecar \
  --manifest outputs/temporal_kmeans_grid/manifest.json \
  --output-dir outputs/temporal_kmeans_grid/experiment \
  --cache-dir /scratch/${USER}/egolife_temporal_kmeans_cache \
  --pair-count 50 \
  --durations-seconds 30,180,360,600 \
  --time-weights 0,0.1,0.25,0.5,1,2,4 \
  --seconds-per-cluster-values 2.5 \
  --temporal-unit-seconds 30 \
  --similarity-thresholds 0.82 \
  --download-media
```

该 grid 默认不物化 4200 组长视频，以免产生不可控的大型产物。先根据聚合
结果选出少量 Pareto 配置，再对这些配置单独物化和盲审。

## 证据边界

本地测试只能证明距离公式、`w=0` 兼容性、时间平移不变性、默认网格合同和
Slurm 静态合同。完整远端作业完成后，可以证明所选 50 对视频上的聚类/剪枝
统计变化；它仍不能单独证明 QA groundedness、answerability 或最终生成质量改善。

## 跨用户 hard temporal gate 实验

第二阶段实验复用已完成 50-pair 作业的 `embedding_cache.json/.npz`，不再下载、
采样或 CLIP encode 视频。每个 duration 先运行两个对照：

1. `production_baseline`：单视频聚类 `w=0`，跨用户只检查 cosine `>=0.82`；
2. `temporal_ungated`：单视频聚类固定 `w=0.1`，跨用户仍只检查 cosine。

随后在 `w=0.1` clusters 上分别 sweep：

- `center`：两个 cluster 的 mean timestamp 差；
- `interval`：两个 `[min(member_time), max(member_time)]` 区间之间的最近距离；
- `G`: `0,5,10,15,30,60,120` 秒。

Gated pruning 的必要条件为：

\[
\cos(e_L,e_R)\ge 0.82
\quad\land\quad
gap(L,R)\le G.
\]

每个 duration 共 `2 + 2×7 = 16` 个 arms，四个 duration 每 pair 共 64 个配置，
50 pairs 共 3200 行。聚类结果、representative cosine matrix、full-frame best
matches 均按 pair/duration/weight 缓存，避免不同 `G` 重复矩阵乘法和 frame matching。

主要新增指标包括 gate eligible/accepted/rejected pair 数、trigger retention、
medoid/center/interval gap 的 mean/P95/max、超过 30 秒和 quarter-duration 的比例、
unique triggered clusters、restored frames、removed percent、pass/no-removal rate。
`aggregate_metrics.*` 同时给出相对 production 和 `temporal_ungated` 的 matched delta。

必需同步：

| 本地（相对 `multi-user/`） | 集群目标 |
|---|---|
| `temporal_kmeans_grid_sidecar.py` | `${PROJECT_ROOT}/egolife_two_user_qa/temporal_kmeans_grid_sidecar.py` |
| `cross_user_temporal_gate_grid_sidecar.py` | `${PROJECT_ROOT}/egolife_two_user_qa/cross_user_temporal_gate_grid_sidecar.py` |
| `hpc/qa/preprocessing/run_temporal_kmeans_grid_50.sbatch` | `${PROJECT_ROOT}/hpc/qa/preprocessing/run_temporal_kmeans_grid_50.sbatch` |
| `hpc/qa/preprocessing/run_cross_user_temporal_gate_grid_50.sbatch` | `${PROJECT_ROOT}/hpc/qa/preprocessing/run_cross_user_temporal_gate_grid_50.sbatch` |

默认 source 是已完成的 `job_16370116/experiment`；建议提交时显式写出，以避免
错误复用其他 cohort：

```bash
cd /scratch/${USER}/Long-video-understanding-clip
SOURCE=/scratch/${USER}/Long-video-understanding-clip/egolife_two_user_qa/multi-user/outputs/temporal_kmeans_grid_50/job_16370116/experiment
sbatch --export=ALL,CROSS_GATE_SOURCE_EXPERIMENT_DIR="${SOURCE}" \
  hpc/qa/preprocessing/run_cross_user_temporal_gate_grid_50.sbatch
```

结果位于：

```text
${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/
  cross_user_temporal_gate_grid_50/job_${SLURM_JOB_ID}/experiment/
```

作业逐 pair 原子更新 checkpoint。若超时，source 仍指向 embedding-cache 作业，
并把最新 gate 作业的 `experiment` 作为 metrics resume source：

```bash
OLD_GATE=/scratch/${USER}/Long-video-understanding-clip/egolife_two_user_qa/multi-user/outputs/cross_user_temporal_gate_grid_50/job_<OLD_JOB_ID>/experiment
sbatch --export=ALL,CROSS_GATE_SOURCE_EXPERIMENT_DIR="${SOURCE}",CROSS_GATE_RESUME_EXPERIMENT_DIR="${OLD_GATE}" \
  hpc/qa/preprocessing/run_cross_user_temporal_gate_grid_50.sbatch
```

该实验能证明 hard gate 如何改变 cross-user trigger 时间分布和剪枝统计；没有 QA
或人工 evidence 复核时，不能证明被过滤的远时间 matches 一定是错误 matches。
