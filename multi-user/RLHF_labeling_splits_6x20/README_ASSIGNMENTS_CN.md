# 六人标注分配说明

## 分配表

| 分配 | Source packets | 可标注 candidates | Generation attempts | 无法解析、仅审计 |
|---|---:|---:|---:|---:|
| `annotator_01_packets_001_020` | 1–20 | 109 | 120 | 11 |
| `annotator_02_packets_021_040` | 21–40 | 113 | 120 | 7 |
| `annotator_03_packets_041_060` | 41–60 | 112 | 120 | 8 |
| `annotator_04_packets_061_080` | 61–80 | 112 | 120 | 8 |
| `annotator_05_packets_081_100` | 81–100 | 112 | 120 | 8 |
| `annotator_06_packets_101_120` | 101–120 | 112 | 120 | 8 |
| **合计** | **1–120** | **670** | **720** | **50** |

六个 assignment 的 packet IDs 与 candidate IDs 均无重叠；并集与完整 120-packet/670-candidate 标注集完全一致。

## 应当给每位 annotator 什么

给每位 annotator：

1. 只给其对应的整个文件夹，而不是只发裸 HTML。文件夹内的 `open_rlhf_labeling.cmd` 会启动本地 HTTP server，避免直接打开 HTML 时出现浏览器加载问题。
2. 同时给一份公共的 `ANNOTATOR_INSTRUCTIONS.md`。
3. 给 annotator 分配一个唯一 ID，并要求在页面顶部 `Annotator ID` 中填写。ID 建议使用不含姓名的稳定代号，例如 `ann01`。

标注包不含图片或视频；annotator 需要网络访问 Hugging Face。视频按需从 Hugging Face 加载。

## Annotator 应当交回什么

至少交回：

1. 页面导出的 `six_user_binary_labels_*.jsonl`，作为主要标签文件。
2. 页面导出的 backup JSON，作为状态恢复与审计副本。

CSV 可以同时交回用于快速检查，但 JSONL 是保留嵌套 binary labels、failure modes 和 provenance 的权威输出。不要让 annotator 修改或重命名 `candidate_id`、`evidence_id` 或 assignment 文件。

## 从 backup 恢复

重新打开标注包后，点击页面顶部的 `Import backup JSON`，选择之前由 `Backup` 导出的 `.json` 文件。页面会按 `candidate_id` 合并本 assignment 中匹配的记录，恢复已完成与部分完成的标签、failure-mode 复选框、notes、skip 状态、计时信息和 annotator ID；备份中不属于当前 20-packet assignment 的候选项会被忽略，当前 assignment 中未出现在备份里的进度不会被清空。数据 fingerprint 不匹配、不是合法 JSON、或没有任何匹配候选项时，导入会拒绝并在右下角显示原因。

## 合并前检查

- 六个 JSONL 的 `assignment_id` 应各不相同。
- `reviewer_id` 不应为空，并应与分配记录一致。
- `candidate_id` 在六份之间应完全唯一。
- 完成后应有 670 条可标注 candidate rows；50 条 generator JSON 解析失败仍只存在于各文件夹的 `unlabelable_generation_attempts.jsonl`，不应伪造成标签。
