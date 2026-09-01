# Qwen 双条件视频 QA 评审设计

日期：2026-09-01

## 1. 目标

使用本地 Transformers Qwen runner，对同一批人工审核通过的多选视频 QA 执行成对评审：

1. `minimum_set`：只输入该题人工确认的最小用户视频集合；
2. `all_six`：输入同一视频组的全部六个用户视频。

两个条件只允许视频集合不同。模型、题目、选项、prompt、解码方式和推理参数保持一致。最终报告只在两个条件均成功且答案解析有效的同一批 QA 上比较准确率。

本设计不修改正式 QA generator retry loop，不启动 OpenRouter 或 Gemini，也不修改既有人工标注结论。

## 2. 已批准输入

### 2.1 人工题目来源

来源一：

`C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts\six_user_qa_10min_16628910_snapshot_10qa_20260901\QA.md`

- 当前包含 7 题；
- 用户已明确确认这 7 题全部作为人工审核通过的 gold；
- 每题包含问题、A–E 选项、正确答案、视频组和 minimum set。

来源二：

`C:\Users\20661\Documents\xwechat_files\wxid_i096w25uhusk22_e748\msg\file\2026-09\qa_curated_17_trace_review_v3.jsonl`

- 共 17 行；
- 17 行均为 `review_status=pass`；
- 每行均包含五个选项、正确答案和非空 `required_users`；
- 覆盖 10 个视频组。

### 2.2 模型与 runner

- 模型 ID：`Qwen/Qwen3.8-27B`；
- Torch 模型目录：`/scratch/xl6775/models/Qwen3.8-27B`；
- runner backend：`transformers-local-memory-safe`；
- 复用 `qwen3vl_runner.py` 中的 `make_runner()` 和 `Qwen3VLMemorySafeTransformersRunner`；
- 使用 greedy 解码；
- 单 GPU 调用由现有 memory-safe runner 串行执行。

## 3. 总体架构

采用“独立核心模块 + 薄命令行入口”。

### 3.1 核心模块

新增 `qwen_two_condition_review.py`，负责：

- 读取并归一化两个人工题目来源；
- 校验 gold、minimum set 和视频组合同；
- 去重并生成审计记录；
- 构造两个条件的视频路径；
- 构造不含 gold 的统一 prompt；
- 调用注入的 Qwen runner；
- 严格解析 A–E 单选输出；
- 增量保存逐条件结果；
- 恢复未完成运行；
- 生成配对结果与总体统计；
- 生成中文人工审核报告。

核心模块不自行连接 Torch，不提交 Slurm 作业，也不负责下载视频。

### 3.2 命令行入口

新增 `tools/run_qwen_two_condition_review.py`，负责：

- 解析输入、媒体根目录和输出目录；
- 创建 `Qwen/Qwen3.8-27B` memory-safe runner；
- 将实际推理配置写入运行 manifest；
- 调用核心模块。

不修改现有 `cli.py`，避免扩大当前 dirty worktree 中的重叠修改面，也避免复现历史上的 CLI dispatch 漏传问题。

### 3.3 测试

新增 `tests/test_qwen_two_condition_review.py`，使用注入的轻量 fake runner 验证数据、prompt、调用形状、解析、恢复和统计。测试不加载模型、不解码真实视频，也不需要 GPU。

## 4. 统一数据合同

两个来源归一化为以下字段：

- `qa_id`
- `source`
- `source_item_id`
- `evidence_id`
- `generation_group_id`
- `question`
- `options`
- `correct`
- `answer`
- `minimum_required_users`
- `review_status`

JSONL 字段映射：

- `generation_group` → `generation_group_id`
- `required_users` → `minimum_required_users`
- `review_status=pass` → 允许进入 gold 候选集

`options` 必须恰好包含五个非空选项。`correct` 必须是 A–E 中的一个字母，并与 `answer` 指向同一个选项。`minimum_required_users` 必须非空、无重复，且所有用户均属于：

- Jake
- Alice
- Tasha
- Lucia
- Katrina
- Shure

任一必填字段缺失或不一致时，该记录不能进入模型评审。

## 5. 去重合同

### 5.1 精确题面去重

题面去重键由以下归一化产生：

- Unicode 归一化；
- 转为小写；
- 去除首尾空白；
- 连续空白压缩为一个空格。

归一化题面相同的记录视为重复。重复时，`qa_curated_17_trace_review_v3.jsonl` 的 trace review v3 记录优先于 `QA.md`。

保留记录的 `options`、`correct` 和 `answer` 必须整体保留。不得把一个来源的选项顺序与另一个来源的正确字母拼接，否则选项重排会造成 gold 错位。

当前两个来源之间有 3 条精确题面重复，因此当前预计唯一题数为：

\[
7+17-3=21
\]

这个 21 条计数作为当前输入数据的针对性回归证据，不写死为核心模块对未来数据的通用要求。

### 5.2 相似题面

同一视频组内描述相似事件、但归一化题面不同的记录不自动删除。系统只在 `deduplication_report.json` 中列出相似候选，供人工复核。程序不依靠语义模型猜测并删除人工 gold。

## 6. 媒体合同

固定六用户顺序为：

1. Jake
2. Alice
3. Tasha
4. Lucia
5. Katrina
6. Shure

视频组目录映射示例：

- `DAY1::17200000` → `DAY1_17200000`
- `DAY3::17000000` → `DAY3_17000000`
- `DAY4::21400000` → `DAY4_21400000`

每个用户视频路径为：

`stitched/<视频组目录>/<用户>.mp4`

只允许读取完整 `stitched` 成片。不得读取 `_segments_from_urls`、原始分段、缓存或临时候选媒体。

### 6.1 两个条件

`minimum_set`：

- `input_users` 严格等于人工确认的 `minimum_required_users`；
- 保留人工提供的用户顺序；
- 不额外加入任何用户。

`all_six`：

- `input_users` 严格等于固定六用户顺序；
- 与 minimum 条件使用相同视频组。

### 6.2 当前媒体边界

本地已有 3 个完整视频组：

- `DAY1_17200000`
- `DAY3_17000000`
- `DAY4_21400000`

新增 JSONL 还需要以下 7 个当前本地未找到的完整视频组：

- `DAY1_11200000`
- `DAY1_17410000`
- `DAY4_11540000`
- `DAY4_12000000`
- `DAY4_12200000`
- `DAY5_16200000`
- `DAY6_15523000`

模型调用前必须生成媒体预检结果。题目只要缺少任一条件所需视频，就标记为 `not_run_missing_media`，不调用模型、不记为答错，也不进入有效配对分母。

## 7. Prompt 与防 gold 泄漏

两个条件使用完全相同的 prompt：

```text
You are given one or more videos and a multiple-choice question.
Answer the question using only the provided videos.

Question:
{question}

Options:
A. {option_a}
B. {option_b}
C. {option_c}
D. {option_d}
E. {option_e}

Select exactly one option.
Output exactly two lines:
CHOICE: <A, B, C, D, or E>
ANSWER: <brief answer>
```

花括号字段只表示运行时从题面和五个选项插入文本，不是待人工填写的文档占位符。

prompt 不包含：

- `correct`
- `answer`
- `minimum_required_users`
- 用户姓名或 minimum set 名称
- `review_status`
- accepted/rejected 信息
- generator rationale
- judge 结果
- 另一条件的用户集合或预测

正确选项文本会作为五个普通选项之一出现，但不会标记哪一项正确。

runner 调用只接收 prompt、空 `image_paths`、当前条件的 `video_paths`、相同解码设置和相同调用 profile。完整 gold 对象不得传给 runner。

## 8. 输出与恢复

每次运行使用独立目录。本地检查目录使用时间戳区分；Torch 正式目录从实际 `SLURM_JOB_ID` 派生。不得使用 `latest_*` 或覆盖历史作业目录。

运行目录包含：

### 8.1 `selection.jsonl`

保存去重后的统一 gold 集及来源。当前预计 21 行。

### 8.2 `deduplication_report.json`

记录保留项、被去重项、优先级理由，以及仅提示而未自动删除的相似题目。

### 8.3 `media_preflight.json`

逐题记录两个条件的媒体完整性，并列出缺失的视频组、用户和路径。

### 8.4 `run_manifest.json`

记录实际模型、backend、FPS、像素限制、输出长度、解码模式、thinking 设置、六用户顺序、输入文件和开始时间。

### 8.5 `predictions.jsonl`

每完成一个条件立即追加一行，字段至少包括：

- `qa_id`
- `condition_id`
- `input_users`
- `video_paths`
- `predicted_choice`
- `correct_choice`
- `is_correct`
- `raw_output`
- `parse_status`
- `run_status`
- `model_id`
- `elapsed_seconds`
- `condition_order`
- `attempt`

### 8.6 `paired_results.jsonl`

每题一行，并列保存两个条件结果、有效配对状态、未配对原因和成败类别。

### 8.7 `summary.json`

保存机器可读统计。

### 8.8 `report_cn.md`

保存中文实验边界、逐题结果、总体统计、无效输出、缺失媒体和耗时，不嵌入大段原始 JSON。

### 8.9 恢复规则

- 每完成一个条件立即落盘；
- `(qa_id, condition_id)` 是恢复键；
- 同一运行目录中已成功且可解析的条件直接跳过；
- 无效解析或调用异常重新执行时增加 `attempt`，保留旧记录；
- 改变模型或推理配置时创建新的运行目录；
- 解析无效不自动重试；
- runner 抛出 GPU、视频解码或模型异常时，先写错误记录，然后停止后续模型调用；修复后显式恢复。

## 9. 输出解析

解析优先接受 `CHOICE: X`，其中 X 是 A–E。

如果没有该行，只在完整输出明确表达唯一答案时接受：

- 单独的 `B`
- `B.`
- `(B)`

多个互相冲突的选项记为 `invalid_ambiguous`；没有选项记为 `invalid_missing`。无效解析保留 `raw_output`，不自动转换为错误答案。

## 10. 配对统计

定义：

\[
N_{\text{gold}}=21
\]

\[
N_{\text{media-ready}}=
\text{两个条件所需视频均完整的唯一题数}
\]

\[
N_{\text{paired}}=
\text{两个条件均成功且解析有效的题数}
\]

最终准确率只使用同一批 (N_{\text{paired}})：

\[
\mathrm{Acc}_{\min}=
\frac{\text{minimum\_set 正确数}}{N_{\text{paired}}}
\]

\[
\mathrm{Acc}_{\mathrm{all}}=
\frac{\text{all\_six 正确数}}{N_{\text{paired}}}
\]

\[
\Delta=\mathrm{Acc}_{\mathrm{all}}-\mathrm{Acc}_{\min}
\]

同时报告：

- 两个条件都正确；
- 两个条件都错误；
- 仅 minimum set 正确；
- 仅 all six 正确；
- 各条件解析失败数；
- 推理异常数；
- 缺失媒体题数；
- 未配对题数；
- 每题与总体耗时。

当前样本最多 21 题，不默认执行显著性检验，也不把一次模型运行描述为稳定统计结论。固定执行顺序为先 `minimum_set`、后 `all_six`；两次调用相邻执行。耗时只作描述，不把模型预热影响解释为条件本身更快。

## 11. TDD 验证范围

实现前先写失败测试并确认失败原因正确，然后写最小实现。测试覆盖：

1. 两种输入来源的字段归一化；
2. 非 pass JSONL、缺失字段、错误选项数、无效正确字母和空 minimum set 的拒绝；
3. JSONL v3 对精确重复项的优先级；
4. 选项、正确字母和答案整体保留；
5. 当前输入去重后为 21 条；
6. 视频组到 `stitched` 路径的映射；
7. minimum 用户顺序和固定六用户顺序；
8. 缺失媒体阻止模型调用；
9. 两个条件 prompt 完全相同且只有视频路径不同；
10. gold sentinel 不进入 prompt 或 runner 参数；
11. A–E 有效、缺失和歧义解析；
12. 每种配对成败类别、准确率和百分点差；
13. 缺失媒体、解析失败和异常不进入有效配对分母；
14. 增量写入、成功记录跳过和 attempt 递增。

## 12. Torch 运行边界

本地实现和零 GPU 测试通过后，才进入远端阶段。

连接 Torch 前必须完整读取共享连接 SOP 及其中三份权威文档，并先检查 `.codex_runtime\torch_auth` 下是否存在可复用的 `READY` 共享桥。没有可复用桥时创建全新 session，等待用户完成本次设备认证后才继续。

缺失的 7 个视频组必须从真实结构化媒体映射或远端现有数据中定位，不猜测远端路径或文件名。

远端同步只允许覆盖本任务新增代码、选择文件和必要小型配置，不上传本地 18 个大型成片，不覆盖远端历史输出。

由于这是新的 Qwen3.8 双条件多视频入口，正式批量运行前执行一次最小 smoke：选择一条媒体完整的 QA，完成两个条件调用并验证输出结构。smoke 通过后不再串联其他规模的 smoke。

正式作业必须：

- 使用 `sbatch --parsable`；
- 立即把 JobID 写入时间戳 manifest；
- 从新 JobID 派生输出目录；
- 在模型加载前封闭 job-specific HOME、cache 和临时目录；
- 保存 storage preflight 结果；
- 不指定固定节点；
- 不复用旧 Job 输出目录。

`COMPLETED/0:0` 只证明调度运行结束。最终验收还必须检查 predictions、媒体完整性、有效配对数、解析状态、统计和报告。

未经用户后续明确授权，不自动上传、提交 Slurm 作业或下载远端产物。不得因连接、媒体、smoke、解析或验收失败自动取消任何作业。

如需从 Torch 下载结果，只使用交互式 SFTP，并按约定逐文件下载小型结果；视频只处理完整 `stitched` 成片。

## 13. 结论边界

最终报告分为三层：

1. 工程层：输入、去重、媒体、prompt、防泄漏、调用和输出合同；
2. 运行层：实际模型、媒体完整题数、有效配对数、解析失败、异常和耗时；
3. 科学层：两条件准确率、百分点差、四类配对结果和小样本限制。

本地测试不能证明远端 GPU runtime 成功；Slurm 完成不能证明产物完整；产物完整不能自动证明两条件差异稳定或具有统计显著性。
