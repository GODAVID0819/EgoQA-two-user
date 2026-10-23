# 三轮 Judge 反馈环出题评估

分支 `xth-fbloop-eval`，worktree `/scratch/tx856/multi_user_qa/EgoQA-fbloop`；运行目录 `/scratch/tx856/multi_user_qa/dual_full120`。
`run_feedback_loop/` 是实际提交所用 sbatch 与配置的副本（运行时读取的是运行目录里的同名文件）；
`human_review/` 是整理审核 json、拼视频和审核网页的脚本副本（运行时在 `dual_full120/human_review/`）。

## 流程

沿用原流水线 `multi-user/video_qa_loop.py`（`--max-attempts 3`）的 生成 → 评审 → feedback_to_generator → 重生成，
但 Generator 和 Judge 都换成 GRPO 训练时用的版本：

1. **生成**：Generator（Qwen3.8-27B 基座，或基座 + GRPO LoRA）读一个 10 分钟窗口里六个用户的抽帧，出一道五选一题（JSON）。
   prompt 与 GRPO 训练逐字相同：`prompts.build_video_generation_prompt` + `generation.BREVITY_INSTRUCTIONS`（profile `compact_full_qa_v1`），
   已预先写在 `data/trainval/validation.jsonl` 的 `messages` 里（`six_user_binary/data.make_row`，由 `prepare_dual_full120.py` 生成）。
2. **判定**：三个已训练、冻结的 Judge（`configs/judge_mixed_f504.json`）用各自 SFT 时的 prompt，取 verdict token 的 P(pass)：

   | Judge | adapter | prompt |
   |---|---|---|
   | formality | `adapters/formality_checkpoint-504` | `prompts.build_qa_formality_judge_prompt` |
   | groundedness | `adapters/groundedness_epoch4` | `prompts.build_evidence_groundedness_judge_prompt` |
   | answerability（六人合并） | `adapters/answerability_checkpoint-340` | `six_user_binary/joint_answerability_prompt.py` |

   每个 Judge P(pass) > 0.5 算通过（Judge SFT 的判定标准，与 GRPO 奖励、pass-rate 评估一致）；**三个都过才接受**，接受的题不再重生成。
   Generator 输出不是合法 QA（非 JSON、选项不是 5 个、answer 与 correct 不一致等）按三项都不通过。不跑原流水线的零样本 Judge 和 strict validation。
3. **反馈**：每个判 fail 的 Judge 在训练时的输出前缀 `{"verdict":"` 后强制写入 `fail`，由同一 Judge 贪心续写 reason / fix
   （answerability 是 reason + missing_evidence），最多 384 token。groundedness LoRA 在 fail 后会直接结束，
   此时按 prompt 输出 schema 的字段顺序预填 `fail", "reason": "` 再续写一次（原续写保存在 `unprefilled_continuation`）。
   反馈只用于生成文本，不改变判定。按原流水线格式拼成 `"{check}: {reason} {fix}"`，多个用 ` | ` 连接；
   不合法输出给英文的 `Generator output was not valid ...` 反馈。
4. **重生成**：用原流水线的 `prompts._feedback_block(feedback, previous_generation=上一轮输出)`，插在原位置
   （`Evidence packet metadata:` 之前），与原流水线带反馈的 prompt 逐字一致（`test_feedback_loop.py` 有对照测试）。最多 3 轮。

采样：temperature 0.85 / top_p 0.95 / top_k 40，最多 2048 token；每题种子
`42 + 1e6·轮次 + 1000·chunk序号 + chunk首条的验证序号 + chunk内位置`，实际种子逐题记录。

输入：抽帧 0.5 fps（每人每 10 分钟 300 帧，Generator 每题约 905 帧），原图 1408×1408；`max_pixels = 24576`，
swift 先缩到 156×156，qwen_vl_utils 再取 32 的倍数 → **160×160，每帧 25 个视觉 token**。

## 代码

| 文件 | 作用 |
|---|---|
| `multi-user/training/grpo_v3/six_user_binary/feedback_loop.py` | 每个 rank 的反馈环：生成、调用 Judge 打分和写反馈、逐轮保存 `rank*/feedback_loop.json`（可续跑） |
| `multi-user/training/grpo_v3/six_user_binary/service.py` | Judge 服务新增 `/explain_batch`（只给判 fail 的 Judge 写反馈），打分路径不变 |
| `multi-user/training/grpo_v3/six_user_binary/predictor.py` | `VllmJudge.explain_many`：强制 verdict 后贪心续写 |
| `eval_feedback_loop.py` | 启动器：每卡一个冻结 Judge 服务 + 一个 Generator 进程（同卡交替显存），按 rank 切分输入，汇总 `feedback_loop_summary.json` |
| `test_feedback_loop.py` | 单测：反馈插入位置、反馈解析/拼接、接受与续跑、逐轮汇总、`/explain_batch` 路由与 reason 预填 |
| `run_feedback_loop/eval_feedback_loop.sbatch`、`fbloop_allsix_sft.json` | 提交脚本与配置 |
| `human_review/` | 审核 json、拼视频、审核网页（见下） |

## 运行

```bash
# 单测
cd /scratch/tx856/multi_user_qa/EgoQA-fbloop/dual_run
PYTHONPATH=$PWD/multi-user:$PWD /scratch/tx856/.conda/envs/grpo-swift/bin/python -m unittest test_feedback_loop

cd /scratch/tx856/multi_user_qa/dual_full120
C=$PWD/configs/fbloop_allsix_sft.json
# 未训练（基座）：前100条验证输入 × 3 题，最多3轮（4×H200）
sbatch --parsable --job-name=fbloop_untrained --export=ALL,RUN_CONFIG=$C eval_feedback_loop.sbatch
# 训练后（GRPO step204 LoRA）
sbatch --parsable --job-name=fbloop_trained \
  --export=ALL,RUN_CONFIG=$C,ADAPTER=$PWD/eval_ckpts/step204_20161348 eval_feedback_loop.sbatch
# 可调：COUNT（验证输入条数，需能被卡数整除）、SAMPLES（每条几题）、ATTEMPTS（最多几轮）；
# RESUME_DIR=outputs/feedback_loop/fbloop_<job> 续跑。卡数只能是 2 或 4（validation_partition）。
```

输出 `outputs/feedback_loop/fbloop_<job>/`：
- `rank*/feedback_loop.json`：每题每轮的 seed、收到的反馈、Generator 原始输出、三个 Judge 的 P(pass)/是否通过、Judge 原始续写、给下一轮的反馈；
- `feedback_loop_summary.json`：每轮结束后各 Judge 通过率（已接受的题保持接受时结果，未接受的按最近一轮）、本轮评审题内通过率、各轮接受数。

## 结果（2026-10-21，100 条验证输入 × 3 = 300 题）

每轮结束后的通过率：

| Judge | 未训练 R1 | R2 | R3 | step204 R1 | R2 | R3 |
|---|---|---|---|---|---|---|
| formality | 33.7% | 45.3% | 46.7% | 72.0% | 83.0% | 87.7% |
| groundedness | 35.7% | 43.0% | 40.7% | 57.7% | 73.3% | 80.0% |
| answerability（六人） | 39.3% | 36.3% | 41.3% | 59.3% | 69.0% | 72.7% |
| 三项全过（接受） | 5.3% | 9.3% | **14.0%** | 28.0% | 48.7% | **59.0%** |

各轮接受数：未训练 16 / 12 / 14（共 42）；step204 84 / 62 / 31（共 177）。
作业：未训练 20519345（7.5 h），step204 20519348（5.5 h），commit aa1c24d。

## 人工审核

```bash
cd /scratch/tx856/multi_user_qa/dual_full120/human_review
python build_fbloop_review_json.py      # fbloop/{untrained,trained}_{1,2,3}.json（接受的题按 验证序号,样本号 排序后三等分）+ fbloop/video_manifest.json
python prepare_videos.py --manifest fbloop/video_manifest.json   # 每人 20 个 30 秒 clip 拼成 10 分钟、640×640 的 videos/<packet_id>/<agent_dir>.mp4
python serve.py --port 8765 --sets fbloop/trained_1 fbloop/untrained_1   # 只听 127.0.0.1，带随机 token；本机 ssh -L 转发后打开打印的链接
```

json 里每题有：接受那一轮的题目/选项/答案/证据、三个 Judge 的 P(pass)、`accepted_attempt`、`feedback_received`、
`previous_attempts`（之前各轮的题、Judge 分数和反馈）、六路视频信息（speaker 在前；帧路径相对 `packets/<packet_id>/`，
clip 原始地址 `segments[].video_url`）。不含本机绝对路径：拿到 packets、`videos/` 和这些 json 即可自建审核页。
审核结论保存在 `reviews/fbloop/<set>.json`。
