# Qwen3.8 legacy two-pass schema v2 生成统计

## 口径与数据来源

本报告直接统计以下四个 metadata-only archives，并在读取前核验了每个 `.tgz.sha256`：

- `packets_000001_000030_metadata_only.tgz`
- `packets_000031_000060_metadata_only.tgz`
- `packets_000061_000090_metadata_only.tgz`
- `packets_000091_000120_metadata_only.tgz`

这里有三个不同的计数单位：

- **source packet**：一个六用户、同步 10 分钟的数据单元。
- **question loop**：为一个 asker 生成并最多重试三次的一轮。
- **generation attempt**：一次实际的 generator 输出及其后续检查。

因此“通过了多少个”应分别报告为：**84 个 accepted questions**，来自 **62/120 个至少产出一道通过题的 source packets**。不能把 84 直接理解为 84 个不同 packet。

本次 archives 虽然位于 `qwen38_legacy_two_pass_schema_v2`，其中的自动 judge 输出仍然是旧的 `legacy_review_passed`/subcheck-heavy contract。这些统计描述的是这批旧 contract 输出；不是尚待人工标注的新 binary reward-model 准确率。

## Judge 训练标注范围

最终是否 accepted 只是旧自动 pipeline 的筛选结果，**不是人工 judge 训练集的纳入条件**。人工标注 builder 读取 `qa_mcq.intermediate.jsonl` 的全部重试轨迹：

- 670 个形成了 parsed QA 的 generation attempts 全部作为独立候选进入人工标注，包括 accepted、rejected 与 6 个 deterministic-schema-invalid 候选。
- 50 个 generator 输出无法解析成 QA JSON，因而没有可展示的 question/options，也没有进入过三个 judge；它们保留为 generator-format failure 统计，但不能伪造成 judge 训练候选。
- annotator payload 不含旧自动 judge 输出或每个候选的 accepted/rejected 状态。

因此可人工标注的 judge 数据规模是 **670**，不是 84。84 只回答“旧 pipeline 最终放行了多少题”。

## 总览

| 指标 | 数量 | 比率 |
|---|---:|---:|
| Source packets | 120 | 100.00% |
| 完成的 generation attempts | 720 / 720 | 100.00% |
| Question loops | 290 | — |
| Accepted questions / loops | 84 | 84/290 = 28.97% |
| Exhausted rejected loops | 206 | 206/290 = 71.03% |
| 有至少一道 accepted question 的 packets | 62 | 62/120 = 51.67% |
| 没有 accepted question 的 packets | 58 | 58/120 = 48.33% |
| 按全部 generation attempts 计的最终 accepted | 84 | 84/720 = 11.67% |
| Infrastructure-skipped rows | 0 | 0.00% |

每个 packet 的 accepted question 数量分布：

| Accepted questions / packet | Packet 数量 |
|---:|---:|
| 0 | 58 |
| 1 | 45 |
| 2 | 13 |
| 3 | 3 |
| 4 | 1 |

## 分 checkpoint 结果

每个 checkpoint 都包含 30 个 source packets 和 180 次 generation attempts。

| Checkpoint | Loops | Accepted questions | Rejected loops | 有通过题的 packets | 无通过题的 packets | Loop acceptance |
|---|---:|---:|---:|---:|---:|---:|
| 000001–000030 | 73 | 22 | 51 | 15 | 15 | 30.14% |
| 000031–000060 | 69 | 17 | 52 | 10 | 20 | 24.64% |
| 000061–000090 | 78 | 28 | 50 | 22 | 8 | 35.90% |
| 000091–000120 | 70 | 17 | 53 | 15 | 15 | 24.29% |
| **合计** | **290** | **84** | **206** | **62** | **58** | **28.97%** |

第三个 checkpoint 的产出最好：28 道 accepted questions，22/30 个 packet 有至少一道通过题。第二和第四个 checkpoint 各只有 17 道 accepted questions。

## 各 judge 的筛选表现

50/720 次 attempt 的 generator 输出不是合法 JSON，因此没有进入任何 judge；三个 judge 的实际共同分母是 670。下表的通过率均以这 670 个已评估 candidate 为分母。

| Judge / deterministic branch | PASS | FAIL | NOT RUN | 已评估通过率 |
|---|---:|---:|---:|---:|
| QA formality | 482 | 188 | 50 | 71.94% |
| Evidence groundedness | 554 | 116 | 50 | 82.69% |
| Answerability | 150 | 520 | 50 | 22.39% |
| Deterministic schema branch | 664 | 6 | 50 | 99.10% |

这里的“表现”是自动 judge 的**筛选行为和选择性**，不是准确率。没有人类标签时，无法计算 precision、recall、agreement 或 bias-corrected accuracy。

最明显的瓶颈是 answerability：它拒绝了 77.61% 的已评估 candidate。三个 judge 同时 PASS 的组合只有 84 次，也就是最终 accepted questions。

### Judge 组合

| QA formality | Evidence groundedness | Answerability | Attempts |
|---|---|---|---:|
| PASS | PASS | PASS | 84 |
| PASS | PASS | FAIL | 315 |
| PASS | FAIL | PASS | 25 |
| PASS | FAIL | FAIL | 58 |
| FAIL | PASS | PASS | 31 |
| FAIL | PASS | FAIL | 124 |
| FAIL | FAIL | PASS | 10 |
| FAIL | FAIL | FAIL | 23 |
| NOT RUN | NOT RUN | NOT RUN | 50 |

单一 judge 独占失败的数量进一步说明 answerability 是主要过滤器：

- 只有 answerability FAIL：315。
- 只有 QA formality FAIL：31。
- 只有 evidence groundedness FAIL：25。

## Answerability 细分

Legacy two-pass gate 的理想模式保持原方向不变：

```text
speaker_only.answerable == false
combined_all_six_users.answerable == true
```

Gate 结果：

| Failure label / outcome | 数量 |
|---|---:|
| Passed（`false -> true`） | 150 |
| `speaker_only_answerable` | 470 |
| `all_six_not_answerable` | 16 |
| `speaker_only_unparsed` | 29 |
| `all_six_unparsed` | 5 |
| Judge 未运行（generator JSON 无效） | 50 |

最重要的诊断是：470 次失败来自 asker-only 已被判定足以作答，说明大多数被拒题没有真正依赖多用户信息。另有 34 次 condition-level 解析失败，值得在新 output contract 和训练数据清洗中单独追踪。

## QA formality 旧 subcheck 诊断

以下字段只存在于这批旧输出，可用于分析，但不应重新放回新 binary output contract：

| 旧 semantic subcheck | FAIL 次数 |
|---|---:|
| `other_person_activity_query` | 168 |
| `first_person_perspective` | 13 |
| `naturalness_and_clarity` | 13 |
| `direct_name_leakage` | 3 |
| `timestamp_citation` | 2 |

Formality 的主要失败原因是 `other_person_activity_query`，占 188 个 formality-failed attempts 中的绝大多数。一个 attempt 可以同时失败多个 subcheck，因此这些行不能相加后当作 attempt 总数。

## 重试效果

| Attempt index | Attempts | Accepted | Acceptance | Formality P/F/NR | Grounding P/F/NR | Answerability P/F/NR |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 290 | 27 | 9.31% | 168 / 105 / 17 | 231 / 42 / 17 | 48 / 225 / 17 |
| 2 | 239 | 35 | 14.64% | 174 / 46 / 19 | 181 / 39 / 19 | 60 / 160 / 19 |
| 3 | 191 | 22 | 11.52% | 140 / 37 / 14 | 142 / 35 / 14 | 42 / 135 / 14 |

第二次尝试的 accepted rate 最高；第三次尝试没有继续提升，说明现有 generator feedback/retry 策略在第二次以后收益有限。

## 按 asker 的产出

| Asker | Attempts | Accepted | Acceptance |
|---|---:|---:|---:|
| A1 | 116 | 17 | 14.66% |
| A2 | 143 | 19 | 13.29% |
| A3 | 111 | 12 | 10.81% |
| A4 | 135 | 14 | 10.37% |
| A5 | 105 | 11 | 10.48% |
| A6 | 110 | 11 | 10.00% |

## 数据完整性与媒体清单

- 四个 archive 的外层 SHA-256 均通过。
- 每个 archive 都含 30 个 `packet.json`、30 个 `clusters.json`、30 个 `asker_views.json`、30 个 `keep_masks.npz`、一个 checkpoint ready 文件和一个 `MEDIA_MANIFEST.jsonl`。
- 每个 source packet 都有 6 位用户；每位用户恰好有 20 个连续 30 秒 Hugging Face source URL。
- 总计 14,400 个 source-video URL（120 × 6 × 20）。
- Archives 中没有图片、视频文件、frame 目录或 CLIP embeddings。
- 50 次无 judge 结果全部来自 generator JSON 解析失败，不是 infrastructure failure；另有 6 次 parsed candidate 未通过 deterministic schema branch。

## Generator 证据与事件时间（全部 670 个 parsed candidates）

这里必须区分两类数据：generator **自报它依据了什么**，以及 archive **可确定它实际获得了哪些输入帧**。前者适合辅助审计，不能当作精确 ground truth；后者可以由代码确定性重建。

### Generator 自报字段

`generation.parsed_qa` 中 670/670 都有 `referred_timestamps` 字段，651 个候选为非空，共 1,214 条 timestamp reference；所有 reference 的 `user` 都能匹配该 packet 的六位用户。另有：

- `generator_rationale`：669/670 非空。
- `why_two_users_needed`：668/670 非空。
- `per_user_evidence_claims`：668/670 非空，共 1,725 条 claim。

但是 `referred_timestamps[].timestamp_seconds` 的编码并不一致：

| 可解释方式 | Reference 数量 |
|---|---:|
| 合同规定的相对秒 `[0, 600]` | 552 |
| 可修复的相对 `MMSS` 数字编码 | 86 |
| 可结合 packet 起始时钟修复的绝对 `HHMMSS` 编码 | 74 |
| 不能无歧义恢复 | 502 |

只有 396/670 个候选的全部 reference 都能按上述规则恢复；274 个至少含一个不可恢复 reference。因此，不能直接把原始 `timestamp_seconds` 当作统一的 0–600 秒坐标训练或定位。

postprocessed QA 中还有更丰富的 `evidence[]`（`user`、`needed_fact`、`timeframe`、`frames_used`）。Archive 能为 598 个候选恢复 postprocessed QA，其中 597 个 evidence 非空，共 1,629 个 evidence item；其余 72 个是后来成功 loop 里的先前失败尝试，trajectory 只保留了 parsed QA，没有把 postprocessed `evidence[]` 再写入。已保留的 evidence 中：

- 1,624 个 `timeframe` 非空，其中 584 个含明确的 `HH:MM:SS`。
- 2,972 条 `frames_used` 描述中，1,818 条含明确的 `HH:MM:SS`（有些还带 frame suffix）。

这些都是 generator 输出的文字声明，可能与真正输入帧不一致，应作为审计信息而不是权威媒体索引。

### 可确定性重建的实际 generator 输入

每个候选都可由 `evidence_id` 回连到以下文件：

- `packet.json`：六位用户每个 2 秒采样帧的 `timestamp_seconds` 与 source segment。
- `keep_masks.npz`：每个 asker 视角下六位用户哪些帧被保留。
- `asker_views.json`：asker/provider 角色及每位用户的 original、retained、removed frame counts。
- `MEDIA_MANIFEST.jsonl`：20 段连续 30 秒 Hugging Face source video URL。

这批数据共有 290 个独立 question-loop media views；同一 loop 的多次 retry 使用同一组媒体输入。每个 view 中 asker 始终保留全部 300 帧（10 分钟、2 秒一步）；五个 provider 分别经过 pruning，单个 provider 的 retained frame count 范围为 120–300，均值 173.74、中位数 146。六人合计每个 view 平均 1,168.72 帧。

因此如果要定位“generator 实际可能看到的事件时间”，应使用 packet frame timestamp 与 keep mask 的交集；如果要研究“generator 自己声称在什么时候看到什么”，再单独使用 `referred_timestamps`、`evidence.timeframe` 和 `frames_used`，并保留它们的格式质量标记。

## 人工标注后应补充的指标

等 binary 人类标签完成后，再按 judge 分别计算：human/model confusion matrix、PASS/FAIL precision 与 recall、balanced accuracy、Cohen's kappa，以及按 annotator 和 asker 分层的偏差。训练时的 class-weighted CE 权重应从**训练 split 的人类标签分布**计算，不能直接用本报告里的自动 judge 通过率代替。
