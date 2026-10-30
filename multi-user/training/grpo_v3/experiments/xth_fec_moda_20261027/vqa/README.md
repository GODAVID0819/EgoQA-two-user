# 六用户 MCQ 的 VQA 评测

评测对象是人工全 pass 的 169 道六用户五选一题（HF：`tianxia2/egolife-six-user-mcqa-human-pass`）。每题由一个人（speaker / asker）以第一人称提问，设计上要靠其他人的视角才能答。这里的脚本用 vLLM 跑 Qwen3-VL-8B-Instruct 和 Qwen2.5-VL-7B-Instruct，回答：

- 只给题面能不能答（text only）；
- 只给 speaker 自己的视角能不能答（speaker only）；
- 加上其他人的视角后能答对多少。

在此基础上筛出两个子集，并公开到 HF。

所有路径都是 NYU Torch 本机路径（`/scratch/tx856/multi_user_qa/...`），换机器要改脚本开头的常量。GPU 任务用 `torch_pr_111_general`，`--gres=gpu:1`，任意卡型都能跑。

## 文件

| 路径 | 作用 |
|---|---|
| `vqa_qwen25vl.py`, `vqa.sbatch` | **视频版 VQA**：输入 10 分钟原始 mp4，每段均匀抽 `--frames` 帧，缩放到 `--size`（Qwen2.5 默认 336，Qwen3 默认 320）。设置：`all6` / `evidence2` / `speaker1` / `speaker_plus1`；`--shuffle-seed` 重排选项；`--ids-file` 只跑指定子集。 |
| `grpo_frames/prep_frames.py`, `prep.sbatch` | 用 lxr_run 的 `make_row` 为每题重建 **GRPO 生成器当时看到的同一套帧**，写成 `frames.json`（CPU 任务）：<br>• speaker 300 帧（每 2 s 一帧）；<br>• provider 只留 `keep_masks` 存活的帧；<br>• 与训练行逐项比对一致；<br>• 记录三个 processor 的实际分辨率；<br>• 截取 GRPO prompt 里描述输入的段落。 |
| `grpo_frames/vqa_grpo_frames.py`, `vqa_grpo.sbatch` | **GRPO 帧版 VQA**。<br>• 输入：帧和分辨率与生成器相同，每帧先 swift `rescale_image` 到面积 ≤ 24576（1408² → 156²），Qwen3 再到 256²、64 token，Qwen2.5 到 168²、36 token。<br>• prompt：GRPO 的输入描述原文 + metadata，再接问题和选项；要求回答 `Answer: X` / `Reason: ...`。<br>• 设置见下表。 |
| `grpo_frames/subsets/*.json` | 子集题号：`hard56`、`textwrong_any131`、`speaker_not_in_gen_evidence`（生成器 evidence 不含 speaker 的 76 题）、`human_review_speaker_only_20261030`（人工复审 29 题的 pass/fail 名单，不含备注）。 |
| `grpo_frames/speaker_viz/` | **人工复审页面**。<br>• `build.py`：生成 `data.json`，以及与模型输入同帧、同分辨率的六人 0.5 fps 视频；provider 没输入的时间格画成深灰。<br>• `serve.py`：只读服务，只监听 127.0.0.1，带随机 token；通过/不通过/备注 POST 回 `reviews.json`。<br>• `index.html`：页面本身。 |
| `hf_upload/` | 打包、上传 HF 数据集：<br>• `stage_*.py` 整理 JSON，去掉 reviewer 身份、备注和机器路径，并复制帧；<br>• `add_videos.py` 硬链接完整视频；<br>• `upload_*.sbatch` 负责上传。 |

### `vqa_grpo_frames.py` 的设置（`SETTING=`）

| 设置 | 给模型的帧 |
|---|---|
| `text_only` | 不给帧，只有题面和选项 |
| `speaker1` | 只给 speaker 的 300 帧 |
| `evidence_providers` | 只给生成器 evidence 里 speaker 以外的人：有两个给两个，只有一个给一个；evidence 只有 speaker 的题跳过，记进 `skipped.json` |
| `evidence2` | `vqa.json` 的 `evidence_videos` 两人（只有一个 provider 时补上 speaker） |
| `all6` | 六人全部（speaker 300 帧 + 各 provider 存活帧，平均约 1100 帧） |

## 运行

```bash
cd /scratch/tx856/multi_user_qa/vqa_eval/grpo_frames
# 先建 frames.json（只需一次）
sbatch --parsable prep.sbatch
# Qwen3-VL-8B，选项 seed 0 重排，跑 hard56 子集的 evidence_providers
sbatch --parsable --account=torch_pr_111_general --gres=gpu:1 \
  --export=ALL,SETTING=evidence_providers,MODEL=Qwen/Qwen3-VL-8B-Instruct,MODEL_TAG=qwen3vl8b,SHUFFLE=0,IDS=hard56 vqa_grpo.sbatch
# Qwen2.5-VL-7B 上限 128000，要加 MAX_LEN=100000
sbatch --parsable --account=torch_pr_111_general --gres=gpu:1 \
  --export=ALL,SETTING=all6,MODEL=Qwen/Qwen2.5-VL-7B-Instruct,MODEL_TAG=qwen25vl7b,SHUFFLE=0,MAX_LEN=100000,IDS=hard56 vqa_grpo.sbatch
```

输出写在 `outputs/<tag>_<setting>_shuf<seed>[_<IDS>]_<jobid>/`，包括：

- `predictions.jsonl`：每题的预测、理由、原始输出；
- `summary.json`；
- `prompt_example.txt`。

视频版：`cd vqa_eval && sbatch --export=ALL,SETTING=all6,FRAMES=384,SHUFFLE=0,MAX_LEN=180000 vqa.sbatch`。Qwen2.5 的 all6 384 帧约 166k token，还需要 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`。

复审页面：`cd grpo_frames/speaker_viz && sbatch build.sbatch && python3 serve.py --port 8770`，然后本机执行 `ssh -L 8770:localhost:8770 <user>@<该登录节点>`，打开打印出来的链接。

## 结果

所有结果都是选项按 seed 0 重排后、贪心解码得到的。五选一，随机猜的正确率是 20%。

### 视频版，169 题

| 模型 / 每段帧数 | all6 | evidence2 | speaker1 |
|---|---|---|---|
| Qwen2.5-VL-7B，64 帧，336px | 55.0 | 57.4 | 50.3 |
| Qwen2.5-VL-7B，384 帧，336px | 63.9 | 63.9 | 49.7 |
| Qwen3-VL-8B，64 帧，320px | 67.5 | 65.1 | 51.5 |
| Qwen3-VL-8B，768 帧 | 65.7（自动降到 160px） | 66.9 | – |

### GRPO 帧版，167 题

2 题在 GRPO 数据集里没有 packet，因此是 167 题。

| 模型 | text only | speaker only |
|---|---|---|
| Qwen3-VL-8B | 42.5 | 46.7 |
| Qwen2.5-VL-7B | 35.3 | 43.7 |

text only 答对的题，speaker only 的正确率约 68%；text only 答错的题，speaker only 的正确率约 30%。

### `textwrong_any131`：两个模型 text only 至少一个答错（131 题）

| 模型 | speaker only | evidence_providers（130 题，1 题 evidence 只有 speaker） | all6 |
|---|---|---|---|
| Qwen3-VL-8B | 40.5 | 68.5 | 64.9 |
| Qwen2.5-VL-7B | 35.1 | 56.9 | 58.0 |

### `hard56`：两个模型 text only 都答错（56 题）

56 题由两部分组成：

- speaker only 两个模型都答错的 44 题；
- speaker only 至少一个答对、但人工复审通过的 12 题。

按定义，这 56 题的 text only 都是 0。

| 模型 | all6 | evidence_providers |
|---|---|---|
| Qwen3-VL-8B | 55.4 | 67.9 |
| Qwen2.5-VL-7B | 53.6 | 55.4 |

只给 evidence provider 的帧（约 250 帧）不比给全部六人（约 1100 帧）差。Qwen3 在 hard56 上反而高 12.5 分。

## HF 数据集

| Repo | 内容 |
|---|---|
| `tianxia2/egolife-six-user-mcqa-human-pass` | 169 题；480 段完整视频 |
| `tianxia2/egolife-six-user-mcqa-hard56` | 56 题；生成器 evidence；baseline 结果；每题 GRPO 输入帧（49,113 张，1408²）；252 段完整视频 |
| `tianxia2/egolife-six-user-mcqa-textwrong131` | 131 题，字段同上；人工复审结论保留在 `selection.human_review_speaker_only`，没有按它过滤；89,255 帧；414 段完整视频 |
