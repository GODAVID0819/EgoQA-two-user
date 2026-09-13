# 六用户 provider-only 剪枝与双重回答性验证实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 将六用户生产剪枝切换为只删除 provider cluster 的 720 组全量比较，同时恢复 speaker-only 与 all-six 两次回答性验证，并保留旧 3-of-5 实现。

**架构：** 旧 `clustered_speaker_consensus_pruning` 保持独立不变；新增 `clustered_speaker_provider_all_pairs_pruning` 并由六用户生产路径调用。回答性继续复用现有 condition 循环，只增加 all-six condition，并在六用户专属 Gate 中同时验证 speaker 错、all-six 对。

**技术栈：** Python 3.11、pytest/unittest、NumPy 风格余弦相似度辅助函数、Slurm 作业合同静态测试。

---

### 任务 1：锁定 provider-only 全量剪枝合同

**文件：**
- 修改：`tests/test_six_user_group_relative_sampling.py`
- 修改：`group_relative_clip_sampling.py`

- [ ] 先新增失败测试：导入 `clustered_speaker_provider_all_pairs_pruning`，构造两个 speaker cluster 和 provider cluster；断言非 argmax 但达到阈值的 provider cluster 被删除、阈值相等时被删除、speaker 标记始终为空、重复命中被去重且保留匹配来源。
- [ ] 运行 `& .\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py -q`，确认因新函数不存在或旧生产路由语义而失败。
- [ ] 新增独立函数：复用 `clustered_frame_representatives` 与 `frame_similarity_matrix`，遍历每个 provider matrix 的全部单元格，按 provider cluster 聚合过阈值匹配，只向 provider 的 `marked_clusters` 写入索引；返回新 method、比较总数、事件、视频区间与通过状态。
- [ ] 将六用户生产调用、候选媒体角色元数据和失败原因切换到新 method；删除生产结果中的 `min_high_provider_matches`，但不修改旧函数及其测试。
- [ ] 重跑该测试文件并确认通过。

### 任务 2：恢复六用户双重回答性 Gate

**文件：**
- 修改：`tests/test_six_user_video_qa_loop.py`
- 修改：`video_qa_loop.py`

- [ ] 先将测试改为期望两个 condition：`speaker_only` 与 `combined_all_six_users`；增加 speaker 错且 all-six 对通过，以及 all-six 错、缺失、不可解析失败的最小断言。
- [ ] 将 runner 路由测试改为两次调用，依次断言 `[speaker_full_video]` 与全部六段 `full_local_video`。
- [ ] 运行 `& .\.venv\Scripts\python.exe -m pytest tests/test_six_user_video_qa_loop.py -q`，确认旧单次实现产生预期失败。
- [ ] 在 `build_answerability_conditions` 为六用户返回两个 condition；在六用户专属 `answerability_gate` 验证两个结果均存在且可解析，并要求 `speaker_choice != correct` 且 `all_six_choice == correct`。
- [ ] 保持 `run_answerability_eval` 的通用循环不变，让新增 condition 自然触发第二次完整媒体调用。
- [ ] 重跑该测试文件并确认通过。

### 任务 3：同步 runtime probe/pilot 静态验收合同

**文件：**
- 修改：`tests/test_six_user_torch_job_contract.py`
- 修改：`hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch`

- [ ] 先将静态测试改为要求每个已接受 QA 恰好两次 answerability 调用、speaker-only 使用一段完整视频、all-six 使用六段完整视频，并要求结果含 `all_six_correct=true` 与评估数 2。
- [ ] 运行 `& .\.venv\Scripts\python.exe -m pytest tests/test_six_user_torch_job_contract.py -q`，确认旧脚本合同失败。
- [ ] 修改 probe 内嵌验收逻辑：按 `condition_type` 分别找到两行，验证媒体角色和视频路径；验收 Gate 中 speaker 为错、all-six 为对；结果摘要写入调用数 2、评估数 2 和 all-six 字段。
- [ ] 重跑静态合同测试并确认通过。

### 任务 4：完整验证、提交与推送

**文件：**
- 检查：本计划涉及的全部源代码、测试、脚本和规格文件

- [ ] 运行 `& .\.venv\Scripts\python.exe -m pytest tests/test_six_user_group_relative_sampling.py tests/test_six_user_video_qa_loop.py tests/test_six_user_torch_job_contract.py -q`，要求全部通过。
- [ ] 运行 `& .\.venv\Scripts\python.exe -m py_compile group_relative_clip_sampling.py video_qa_loop.py`，要求退出码为 0。
- [ ] 运行 `git diff --check`，并逐项核对规格：720 组全量比较、provider-only、speaker 不删、双重回答性、旧 3-of-5 保留、无 hash。
- [ ] 只暂存本计划相关文件，检查 `git diff --cached --name-only`，确保无既有无关 dirty 文件。
- [ ] 提交本次实现并将 `feature/six-user` 推送到源仓库；不连接 Torch、不提交或取消作业。
