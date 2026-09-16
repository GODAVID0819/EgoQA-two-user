# 六用户 RLHF 人工标注界面

## 作用

`rlhf_human_labeling.py` 把 production generation job 发布的一个或多个 metadata-only `.tgz` 转成可在浏览器中使用的人工标注包。输入与 `run_six_user_rlhf_qa_generation.sbatch` 对齐：

- 推荐输入：`RLHF_metadata/packets_XXXXXX_XXXXXX_metadata_only.tgz`
- 每个 archive 的旁边必须有同名 `.tgz.sha256`；builder 会先验证 archive hash，再验证内层 `CHECKPOINT_READY.json` 中列出的 checkpoint payload hash。
- archive 内必须包含 checkpoint 的 `qa_mcq.intermediate.jsonl`、`labeling_queue.jsonl` 和每个 packet 的 `packet.json` metadata。
- archive 不得包含图片、视频、frame 文件夹或 CLIP embeddings。发现这些内容时 builder 会拒绝构建。

人工标注池来自 `qa_mcq.intermediate.jsonl` 中的**全部 generation attempts**，而不是 accepted-only 的 `labeling_queue.jsonl`。当前四个 checkpoint 一共有 720 次生成尝试：其中 670 次形成了可供 judge 标注的 parsed QA，全部进入界面；另外 50 次连 QA JSON 都没有形成，单独计为 `unparseable_generation_attempt_count`，并写入 `unlabelable_generation_attempts.jsonl` 以保证 720 次尝试全部可核账。Archive 没有保存这 50 次的原始输出文本，因此不会伪造问题或答案后混入 judge 训练。

同一个 question loop 的每次重试都是独立样本，使用 `<evidence_id>::attempt_<NN>` 作为稳定 `candidate_id`。自动 judge 的 verdict、reason、subchecks、最终 accepted/rejected 状态、prompt 与 generation trace 都不会进入 annotator payload，避免标签泄漏。`source_evidence_id` 与 `generation_attempt` 只用于标签导出后回连原始 trajectory；界面不会向标注员显示先前自动判定。

少数 parsed QA 本身可能不满足 deterministic schema（例如缺问题、选项数不对或缺正确选项）。这些正是 formality judge 需要学习的负样本，因此不会被 builder 删除；界面会使用英文占位文本，例如 `[Missing question]`、`[No options were produced]`、`[Missing correct option]`。

## 当前人工标签合同

人工标注只要求二元/布尔标签；不会强制 annotator 为每个 judge 重写 reason、fix 或 evidence list。Formality 和 Evidence groundedness 直接选择 `pass` 或 `fail`：

```json
{
  "verdict": "pass | fail"
}
```

Answerability 保留 legacy zero-shot 的名称、布尔方向与两次独立判断：

```json
{
  "speaker_only": {
    "answerable": false
  },
  "combined_all_six_users": {
    "answerable": true
  }
}
```

代码仅在 `speaker_only.answerable == false` 且 `combined_all_six_users.answerable == true` 时派生出 answerability `pass`；这里没有重命名字段，也没有反转 `answerable`。

每个 attempt 另有：

- `failure_modes`：可选的常见失败模式 checkbox 列表；当前包括 unclear/unnatural wording or question、non-first-person、asker asks about themselves、bad options、concurrent-activity question、participant-name leakage、timestamp leakage、unsupported premise、unsupported answer、ambiguous answers 和 missing/occluded event。这些是人工分析 tags，不是重新放回模型 output contract 的 subchecks。
- `notes`：一个 per-attempt 的可选人工备注字段。

CSV 同时导出 `formality_score`、`evidence_grounding_score`、`answerability_score`，确定性编码为 `1 = fail`、`2 = pass`。新界面不会产生 3 分；已有的旧 1–3 分样本继续保留，并可由 binary reviewer 按 `1 -> fail`、`2/3 -> pass` 读取。

## Packet-first 导航

- 顶部下拉框与左右箭头按 packet 切换，不再按 generation attempt 切换。
- 当前 packet 的所有可标注 candidates 显示为一排匿名 Q1、Q2… tabs；tab 上只显示 asker 和当前标注状态。Generation attempt number 不在 UI 中显示，以免标注员偏好后期 retry；它只保留在导出 provenance 中用于回连原始 trajectory。
- `Save & next pending` 优先进入当前 packet 的下一条 pending attempt；当前 packet 完成后才进入后续 packet。
- packet filter 按“含 pending”“全部完成”“含 skipped”筛选。

## 证据界面

- 六位用户都有独立 tab；当前 asker 有明确标记。
- 每个用户的 10 分钟 timeline 由 20 个连续的 30 秒 Hugging Face MP4 组成；页面一次只加载当前用户、当前 segment。
- 六个用户共享相同的 0–10 分钟位置；切换用户时保持当前 segment/时间，便于同步对照。
- 支持前后移动 30 秒、直接选择任一 segment、拖动 10 分钟 timeline，以及可关闭的 segment 自动续播。
- Candidate 上方列出 generator 自报的精确 `user + absolute clock + packet-relative MM:SS`。可解析的条目可点击，界面会直接切到对应用户、30 秒 clip 和 clip 内位置；无法可靠规范化的原始 timestamp 会保留并标记为 `unresolved`。
- 定位信息合并 `referred_timestamps` 与可恢复的 `evidence.frames_used`。它们用于快速定位，但仍是 generator claims，不等于人工确认的 ground truth。

### Generator 时间声明的来源与转换

Builder 不会从人类标签或旧自动 judge 推断事件时间。定位条目来自 generator 自己写入 archive 的两个字段：

1. `qa_mcq.intermediate.jsonl -> attempts[].generation.parsed_qa.referred_timestamps[]`，每项包含 `user`、`timestamp_seconds` 和 `moment`。670 个 parsed candidates 都有这个字段，但有些为空或编码不规范。
2. 可恢复的 postprocessed QA `evidence[].frames_used[]`，例如 `Tasha 15:05:02:19`。这类信息能从 accepted rows 和 exhausted-rejected rows 中恢复；later-accepted loop 的部分早期失败 retry 没有保存 postprocessed evidence。

规范化按以下顺序进行：

- 数值在 `[0, 600]`：直接当作 packet-relative seconds。
- 可解释为绝对 `HHMMSS` 且落在 packet 起点后的 10 分钟内：减去 packet `clip_clock` 得到相对秒。
- 可解释为相对 `MMSS` 且不超过 10 分钟：转换为相对秒。
- `frames_used` 中明确的 `HH:MM:SS(:frame)`：减去 packet `clip_clock` 得到相对秒；frame suffix 保留在原始字符串中，但跳转精度为秒。
- 其他格式：标记为 `unresolved`，不提供跳转按钮。

解析后的相对秒通过 `segment_index = floor(relative_seconds / 30)` 选择 30 秒 source clip，并用 `relative_seconds % 30` 定位 clip 内时间。`user` 只在能匹配该 packet 的 `agent_name`、`agent_dir` 或 `agent_id` 时才可点击。相同 user + second 的重复声明会合并。

这些位置只回答“generator 声称自己在哪里看到了证据”，不能证明该事件真的发生在那里。真正的 generator 输入帧仍应由 `packet.json` 的 frame timestamp 与 `keep_masks.npz`/`asker_views.json` 的保留集合重建。
- 页面中不包含、也不会下载 sampled frames、图片或完整视频文件；视频播放直接使用 archive metadata 中的 Hugging Face URL。

## 构建和打开

推荐直接使用已下载的 metadata-only archives。PowerShell 示例：

```powershell
$metadata = "C:\path\to\multi-user\RLHF_metadata"
python multi-user\rlhf_human_labeling.py `
  --metadata-bundle "$metadata\packets_000001_000030_metadata_only.tgz" `
  --metadata-bundle "$metadata\packets_000031_000060_metadata_only.tgz" `
  --output-dir "C:\path\to\labeling\annotator_a" `
  --assignment-id "annotator_a"
```

可以重复传入 `--metadata-bundle` 合并多个 checkpoint；也可用 `--packet-start` 与 `--packet-count` 按 source packet 切分 assignment。切分不会跳过没有 accepted question 的 packet，因为标注数据包含 rejected attempts。生成一个 packet 的 smoke-test 包：

```powershell
python multi-user\rlhf_human_labeling.py `
  --metadata-bundle "$metadata\packets_000001_000030_metadata_only.tgz" `
  --output-dir "C:\path\to\labeling\smoke" `
  --assignment-id "schema_v2_smoke" `
  --packet-count 1
```

旧的已解压 checkpoint + sampled-frame dataset 输入仍保留作兼容，不会删除：

```powershell
python multi-user\rlhf_human_labeling.py `
  --checkpoint-dir "C:\path\to\checkpoint" `
  --dataset-root "C:\path\to\egolife_rlhf_evidence_v1" `
  --output-dir "C:\path\to\labeling\legacy" `
  --assignment-id "legacy_input"
```

输出目录包含：

- `rlhf_labeling.html`
- `labeling_data.json`
- `assignment_manifest.json`
- `human_labels_template.csv`
- `unlabelable_generation_attempts.jsonl`（无法形成 QA JSON 的 generation attempts；不作为 judge 标签任务）
- `serve_rlhf_labeling.py`
- `open_rlhf_labeling.cmd`

Windows 上双击 `open_rlhf_labeling.cmd`。其他系统运行 `python serve_rlhf_labeling.py`。本地 server 只暴露标注包本身；Hugging Face MP4 由浏览器按需读取，标注包内没有媒体文件。

界面会按 assignment fingerprint 自动保存到浏览器 `localStorage`，并支持 JSON backup/restore。正式交付标签时导出 JSONL 和 CSV；CSV 提供训练兼容的数值列，JSONL 保留完整的具体 reason、fix 与两组 answerability evidence arrays。

全部 judge prompts 与 output contracts 的逐项审阅稿见 `LEGACY_ZERO_SHOT_JUDGE_PROMPTS_REVIEW.md`。
