# 六视频 speaker-consensus pruning 实施与 Torch 提交计划

> **执行要求：** 在当前会话内顺序执行。每项行为修改先写测试并观察预期失败，再写最小实现。使用复选框记录进度；不创建 commit，不使用 SHA 或其他哈希。

**目标：** 将六用户“双 anchor、3 pruned + 3 full”路径替换为固定遍历六个 speaker 的 consensus pruning，使每个同步组产生 0–6 个候选；本地和 Torch 登录节点验证通过后，在同一轮提交 runtime probe 与依赖它的 pilot40。

**架构：** 二用户路径保持不变。六用户路径复用现有 CLIP embedding、deterministic cosine k-means、medoid、区间与 FFmpeg helper；新增纯 speaker-consensus 计算与六视频物化。QA、groundedness、双条件 answerability 和 Qwen3.6-27B memory-safe 执行层只做必要字段适配。

**技术栈：** Python、pytest/unittest、NumPy、CLIP、FFmpeg、Qwen3-VL、Bash、Slurm、Paramiko 共享桥、交互式 SFTP。

**范围约束：** 不更新 Torch Runbook；不修改 GRPO、DPO、reviewer、optimizer、checkpoint；不清理现有脏工作树；不 push；不取消任何作业。

---

## 任务 1：纯 speaker-consensus 算法

**文件：**

- 修改：`tests/test_six_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py`

- [ ] 新增失败测试：`5-of-5`、`4-of-5`、`3-of-5`、`similarity == 0.82`、每个 provider 只取 argmax、非 argmax 即使过阈值也不删除、重复 cluster 去重但保留来源、删除全部 member frames。
- [ ] 运行定向测试，确认因 `clustered_speaker_consensus_pruning` 尚不存在或旧语义不符而失败：

```powershell
python -m pytest tests/test_six_user_group_relative_sampling.py -q
```

- [ ] 新增 `clustered_speaker_consensus_pruning(...)`。输入为六组 sampled frames/embeddings、`speaker_index`、`cluster_count=12`、`high_similarity_threshold=0.82`、`min_high_provider_matches=4` 与窗口参数；输出包含逐视频 cluster 诊断、每个 speaker cluster 的五个 argmax、删除事件、marked clusters/member frames、remove/keep intervals 和时长。
- [ ] `4-of-5` 事件只标记 speaker 与四个过阈值 provider clusters；`5-of-5` 标记六个；`3-of-5` 不触发。
- [ ] 不调用 provider-provider similarity，不使用 `max_pair_time_difference_seconds`。
- [ ] 重跑定向测试并保持现有二用户 pruning 测试通过：

```powershell
python -m pytest tests/test_six_user_group_relative_sampling.py tests/test_pruning_ablation.py tests/test_paired_evidence_pruning.py -q
```

## 任务 2：固定遍历六个 speaker 并物化全部成功候选

**文件：**

- 修改：`tests/test_six_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py`

- [ ] 新增失败测试：speaker 严格按已排序用户 `1,2,3,4,5,6` 遍历；不调用随机 speaker 选择；失败保存原因后继续；成功保存候选后仍继续；单组可返回 0–6 个候选。
- [ ] 新增失败测试：每个成功候选均按 `[speaker, provider1, ..., provider5]` 物化六个 pruned generator videos，同时保留六个 full originals；任一视频低于最短保留时长只淘汰当前 speaker。
- [ ] 运行测试确认旧代码只选择一个双 anchor 结构且媒体模式仍为 `3 pruned + 3 full`。
- [ ] 在 `selected_count == 6` 分支绕开 `relative_group_scores()`、`score_video_pairs()`、`build_six_user_role_structures()` 和 `materialize_six_user_role_structure()`；二用户分支继续使用原逻辑。
- [ ] 为六位 speaker 逐一调用纯 consensus 函数和六视频物化；累积 `speaker_attempts` 与全部成功候选，而非遇到首个成功即停止。
- [ ] packet 改为 `candidate_type=six_user_speaker_consensus`、`generator_media_mode=six_pruned_videos`、`provider_users`、`speaker_consensus_pruning`、`speaker_attempts`；活跃六用户输出移除 anchor/additional/edge 字段。
- [ ] 重跑任务 1–2 测试并确认二用户兼容。

## 任务 3：Prompt、schema 与双条件 answerability 适配

**文件：**

- 修改：`tests/test_six_user_prompts.py`
- 修改：`tests/test_six_user_video_qa_loop.py`
- 修改：`prompts.py`
- 修改：`schema.py`
- 修改：`video_qa_loop.py`

- [ ] 先写失败测试：角色只有一名 speaker 与五名 providers；prompt 不出现 anchor/additional；generator 为六个 pruned；groundedness 为六个 full；answerability 仍只调用 speaker-only full 与 all-six full 两次。
- [ ] 写失败测试：accepted 仍要求 `speaker_only_correct=false`、`all_six_correct=true`、`cross_view_gain=1`、`answerability_evaluated_condition_count=2`；all-six wrong 保持中性失败。
- [ ] 运行并确认失败：

```powershell
python -m pytest tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py -q
```

- [ ] 最小修改 prompt/schema/audit metadata；保留 `required_users`、`evidence_provider_user(s)` 兼容语义，但不重新引入 anchor/additional 角色。
- [ ] 重跑 prompt、video QA 和质量指标回归测试。

## 任务 4：更新 Torch 作业合同，不写 Runbook

**文件：**

- 修改：`tests/test_six_user_torch_job_contract.py`
- 修改：`hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`
- 修改：`hpc/qa/experiments/run_six_user_qa_pilot_40.sbatch`
- 不修改：`docs/SIX_USER_QA_TORCH_RUNBOOK_CN.md`

- [ ] 先写失败合同测试：两个作业检查 `six_pruned_videos`、固定六 speaker 遍历、0–6 候选、consensus 诊断和六 pruned/六 full 路由；不再检查 anchor edges 或 `three_pruned_three_full_videos`。
- [ ] 保留当前未提交的 Qwen3.6-27B、decord、显式 min/max pixels、job-specific scratch/cache、storage preflight、JobID 输出目录和 `partial` 语义。
- [ ] 运行合同测试确认旧审计字段失败，再最小更新两个 `.sbatch`。
- [ ] 执行：

```powershell
python -m pytest tests/test_six_user_torch_job_contract.py tests/test_qwen_runner_compat.py -q
bash -n hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
bash -n hpc/qa/experiments/run_six_user_qa_pilot_40.sbatch
```

## 任务 5：本地新鲜验证与范围审计

- [ ] 运行六用户定向套件：

```powershell
python -m pytest tests/test_six_user_group_relative_sampling.py tests/test_six_user_prompts.py tests/test_six_user_video_qa_loop.py tests/test_six_user_torch_job_contract.py tests/test_qwen_runner_compat.py -q
```

- [ ] 运行相关二用户回归、Python 语法和差异检查：

```powershell
python -m pytest tests/test_pruning_ablation.py tests/test_paired_evidence_pruning.py tests/test_quality_entropy.py -q
python -m compileall -q group_relative_clip_sampling.py prompts.py schema.py video_qa_loop.py qwen3vl_runner.py
git diff --check
git status --short --branch
```

- [ ] 扫描活跃六用户旧合同，仅允许旧函数定义或二用户兼容代码残留，不允许新 packet/prompt/作业继续依赖：

```powershell
rg -n "three_pruned_three_full_videos|anchor_provider_users|additional_provider_users|selected_anchor_edges|six_user_two_anchor" group_relative_clip_sampling.py prompts.py schema.py video_qa_loop.py hpc/qa tests
```

- [ ] 确认未修改 GRPO、DPO、reviewer、optimizer、checkpoint 线路，未新增哈希字段，未覆盖用户原有 27B/pilot40 改动。

## 任务 6：Torch 实时审计与窄同步

- [ ] 复用当前共享桥前重新核验 `READY`、PID 存活和带唯一请求 ID 的新鲜只读探针。
- [ ] 每条远端命令使用唯一文件名、`REQUEST_BEGIN/REQUEST_END`，并在命令内部重新声明所有路径和环境。
- [ ] 在登录节点只读查询当前 account、partition、QOS、H100、远端 checkout、dirty state、模型、环境、输入媒体、已有作业，以及同链路历史 `Elapsed/MaxRSS`；冲突时打印 `STOP:`，不提交。
- [ ] 使用窄范围交互式 SFTP 上传且只上传本次必要文件：

```text
group_relative_clip_sampling.py
prompts.py
schema.py
video_qa_loop.py
tests/test_six_user_group_relative_sampling.py
tests/test_six_user_prompts.py
tests/test_six_user_video_qa_loop.py
tests/test_six_user_torch_job_contract.py
hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch
hpc/qa/experiments/run_six_user_qa_pilot_40.sbatch
```

- [ ] 上传后在 Torch 登录节点运行同一组定向测试、`compileall`、两个 `.sbatch` 的 `bash -n`、真实 parser/CLI 检查，以及当前既定的 decord 与 Qwen 视频预处理零 GPU验证；不在登录节点运行模型或训练。
- [ ] 只有登录节点验证全部通过后才允许请求 GPU 资源。
- [ ] 根据实时资源与同链路历史证据确定最小安全申请：GPU 数量不得低于模型实际需要；CPU、内存和 walltime 只有在历史 `MaxRSS`、实际吞吐或同链路证据支持时才下调。没有可比证据时保留当前已论证的安全值，不用猜测换取更短排队。

## 任务 7：连续提交 runtime probe 与 pilot40

- [ ] 以自包含命令再次声明项目、输出、环境、模型、数据、account 和脚本路径，确认实时 account/partition/QOS/H100 与脚本一致。
- [ ] 使用 `sbatch --parsable` 提交 runtime probe，兼容 cluster 后缀提取数字 JobID，并立即写入新的时间戳 submission manifest；失败 JobID 和既有 Job `15959693` 均保留为 provenance。
- [ ] 不等待 probe 排队或运行；立即以全新自包含命令使用 `sbatch --parsable --dependency=afterok:${PROBE_JOB_ID}` 提交 `run_six_user_qa_pilot_40.sbatch`。依赖只防止 probe 顶层作业非零退出时启动 pilot，不声称 probe 产物已验收。
- [ ] 提取 pilot 数字 JobID，并立即追加到同一时间戳 manifest；manifest 明确记录 probe JobID、pilot JobID、`afterok` 依赖、脚本、提交时间、资源、walltime、输出根目录和任务关系，不记录任何哈希。
- [ ] 用 `sacct`/`squeue` 确认两个 JobID 均已被 Slurm 接收；允许 probe 为普通 pending、pilot 为 dependency pending。
- [ ] 到此停止并结束本轮对话。最终汇报本地验证、Torch 登录节点验证、两个 JobID、调度状态、资源与 walltime、manifest 路径，以及当前只能证明“已提交”，不能证明 runtime、产物或 QA 质量。

## 任务 8：用户通知结果后的后续验收（不属于本轮）

- [ ] 用户看到任务产生结果后再恢复本任务；从 manifest 读取两个 JobID，不要求用户手工重输。
- [ ] 分别检查 `sacct`、stdout、stderr、`storage_preflight.json`、`job_manifest.json`、JobID 派生结果目录和 `six_user_qa_result.json`。
- [ ] 将批处理状态、runtime、consensus/media 合同、自动 QA 指标和人工质量分开汇报，再决定是否存在下一步；不因失败自动取消或重投。

## 明确停止条件

本轮只在以下两种状态之一停止：

1. **成功停止：** Torch 登录节点验证通过，runtime probe 与带 `afterok` 依赖的 pilot40 均已被 Slurm 接收，两个 JobID 均已写入时间戳 manifest；
2. **失败停止：** 本地验证、Torch 登录节点验证、资源合同或任一提交命令失败，已保留证据且没有取消或盲目重提已有作业。

任何情况下都不取消已有作业，不因桥断线重提作业，不使用旧 PIN，不依赖聊天记忆保存 JobID。
