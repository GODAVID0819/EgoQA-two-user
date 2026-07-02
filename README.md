# EgoLife 双用户 QA Pilot

这个模块用于从 EgoLife 视频中构造 20 条 pilot 多选题。每道题都要求至少两个用户的第一视角视频共同提供证据，单个用户的视频不能完整回答。

默认模型是 `Qwen/Qwen3-VL-8B-Instruct`。流程不使用 OpenRouter/Gemini 等商业 API key。`HF_TOKEN` 只作为 Hugging Face 下载或限流辅助，不作为推理 API key。

## 主流程

当前主路径是 video-first。也就是说，Qwen3-VL 直接接收对齐后的 EgoLife 原始视频，而不是先把视频转成 caption/observation 再出题。之后用 judger 和 answerability evaluation 过滤掉单用户可答、合并视频也答不准、或者问题口吻不自然的题。

```text
EgoLife video + EyeGaze/EyeTracking tree
-> build_manifest
-> prepare_evidence: 按 day / time token 对齐至少两个用户，并缓存视频/gaze
-> generate_video_qa_loop:
   -> generator: 直接看多用户视频，生成 commonality/difference MCQ
   -> judger: 解释为什么问这个问题，并给 generator 反馈
   -> answerability eval: 分别测试单用户视频和合并视频能否答题
-> validate_outputs: 做确定性的 schema/gate 检查
```

`observe_clips` 和 `mine_candidates` 只保留作调试辅助，不作为 pilot 主路径。正式 QA 生成、judger、answerability evaluation 和最终 review 都在 `generate_video_qa_loop` 内完成，避免旧 prompt 和当前 judge rubric 混用。

## CLIP-Pruned Benchmark Prep

`hpc/run_clip_pruned_benchmark_100.sbatch` prepares a 100-packet benchmark for
later QA pipeline experiments. For each synchronized timestamp, it randomly
selects exactly two videos, samples one frame per second from each selected
30-second video, embeds those sampled frames with CLIP, clusters embeddings
within each selected video into 12 clusters by default, and compares cross-video representative frames.
When representative frames are highly similar across the two selected videos,
the prep step prunes the corresponding sampled-frame interval plus every frame
assigned to that representative cluster.

The emitted evidence packets route media deliberately: `local_video` points to
the CLIP-guided pruned MP4 for generation, while `full_local_video` and
`original_local_video` point to the un-pruned 30-second MP4 for judges and
answerability. `generate_video_qa_loop` uses `media_role="generator"` for the
generator and `media_role="full"` for all three verification branches.

## CLIP Anchor / Evidence-Gap Toy Demo

`clip_gap_demo` 是一个独立的预处理实验，不会直接生成问题。它读取
`prepare_evidence` 产生的双用户 evidence packet，对短视频窗口采样帧，
使用 CLIP embedding 做用户内聚类，再寻找：

- 两个用户之间的 mutual-nearest shared anchors；
- Alice 中找不到 Bob 近邻的高 novelty evidence gaps；
- Bob 中找不到 Alice 近邻的高 novelty evidence gaps。

运行后会写出 JSON similarity results 和一张 contact sheet，先供人工检查，
再决定是否把候选片段交给自由度更高的 VLM question generator。完整示例见
`CLIP_GAP_DEMO.md`。

## Gaze 投影说明

EgoLife EyeGaze CSV 不是 EgoEverything 里的 image pixel gaze。它给的是 Project Aria CPF 坐标系下的 yaw/pitch/depth，例如 `left_yaw_rads_cpf`、`right_yaw_rads_cpf`、`pitch_rads_cpf` 和 `depth_m`。所以代码不能凭空构造 `gaze_x/gaze_y`。

默认情况下 gaze summary 会标记为：

```json
{"projection_status": "missing_calibration"}
```

如果想启用 EgoEverything 那种 2D gaze point 到 object bbox center 的距离/Gaussian sampling，需要传入 Aria RGB calibration：

```bash
python -m egolife_two_user_qa observe_clips \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.json \
  --output egolife_two_user_qa/outputs/pilot_20/observations.jsonl \
  --aria-calibration-dir /path/to/aria_calibrations
```

更严格的 Aria 投影需要提供 VRS/no-image VRS 文件或 `online_calibration.jsonl`，并安装 `projectaria-tools`。代码会优先走 Project Aria 原生 `CameraCalibration.project()`。JSON calibration 也可以用，但必须显式包含 RGB intrinsics 加 `T_camera_cpf`，或者 `T_device_camera` 和 `T_device_cpf`。如果只使用公开 EgoLife Hugging Face 文件且没有 calibration/VRS，正确行为就是保持 2D projection unavailable，只使用视频帧和未投影的 3D gaze 统计。

## 本地 CPU Dry Run

dry run 用来验证 Hugging Face manifest、evidence packet、video-first prompt 和 schema 工具链。它不会加载 Qwen3-VL，也不会真的生成高质量 QA。

```bash
python -m egolife_two_user_qa build_manifest \
  --days DAY1 \
  --agents A1_JAKE,A2_ALICE \
  --max-per-agent-day 2 \
  --output egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json

python -m egolife_two_user_qa prepare_evidence \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.dryrun.json \
  --output egolife_two_user_qa/outputs/pilot_20/evidence_manifest.dryrun.jsonl \
  --target-count 2 \
  --users-per-case 2 \
  --frames-per-clip 2 \
  --no-download-media

python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence egolife_two_user_qa/outputs/pilot_20/evidence_manifest.dryrun.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20/qa_mcq.video_first.dryrun.jsonl \
  --prompts-output egolife_two_user_qa/outputs/pilot_20/video_first_prompts.dryrun.jsonl \
  --intermediate-output egolife_two_user_qa/outputs/pilot_20/video_first_intermediate.dryrun.jsonl \
  --target-count 1 \
  --dry-run
```

## GPU Pilot Run

正式生成 20 条 QA 需要 GPU 或支持视频输入的本地 VLM server。

```bash
bash scripts/run_qwen3vl_gpu.sh \
  --target-count 20 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-new-tokens 1536
```

如果使用本地 OpenAI-compatible server，比如 vLLM/SGLang/llama.cpp，先启动 server，然后运行：

```bash
python -m egolife_two_user_qa generate_video_qa_loop \
  --backend openai-compatible-local \
  --base-url http://127.0.0.1:8000/v1 \
  --evidence egolife_two_user_qa/outputs/pilot_20_video_first/evidence_manifest.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20_video_first/qa_mcq.jsonl \
  --prompts-output egolife_two_user_qa/outputs/pilot_20_video_first/video_first_prompts.jsonl \
  --intermediate-output egolife_two_user_qa/outputs/pilot_20_video_first/qa_mcq.intermediate.jsonl \
  --allow-openai-video-input
```

如果不传 `--allow-openai-video-input`，OpenAI-compatible backend 会退回 sampled frame images，因为不是每个本地 server 都支持 video data URL。

## Discovery Control

`discovery_control` is a control mode for the discovery ablation. It records a
separate generation mode but skips the discovery/planning call and uses the same
direct generation prompt as `baseline`. This tests whether the discovery
thinking phase itself changes outcomes.

To reuse an existing five-row evidence manifest and run only this control, set
`EGOLIFE2U_EVIDENCE_JSONL` to the existing JSONL and `EGOLIFE2U_CACHE_DIR` to
the cache root. The cache is expected to be organized as
`CACHE_DIR/<agent_dir>/<DAY_N>/<video_file>` or
`CACHE_DIR/<agent_dir>/<DAYN>/<video_file>`, for example
`egolife_two_user_qa_cache/A1_JAKE/DAY_1/DAY1_A1_JAKE_11100000.mp4`
or `egolife_two_user_qa_cache/A1_JAKE/DAY1/DAY1_A1_JAKE_11100000.mp4`.
The sbatch writes a resolved copy of the evidence JSONL with `local_video`
filled from that cache layout.

```bash
EGOLIFE2U_EVIDENCE_JSONL=egolife_two_user_qa/outputs/<existing_run>/evidence_manifest.jsonl \
EGOLIFE2U_CACHE_DIR=/scratch/${USER}/egolife_two_user_qa_cache \
EGOLIFE2U_OUTPUT_ROOT=egolife_two_user_qa/outputs/discovery_control_5 \
EGOLIFE2U_EVAL_PACKET_COUNT=5 \
sbatch hpc/run_egolife_discovery_control_only.sbatch
```

## 输出 Schema

`qa_mcq.jsonl` 每一行是一条 QA，包含：

- `qa_id`
- `question`
- `options`
- `correct`
- `answer`
- `category`
- `question_type`
- `required_users`
- `evidence`
- `single_user_answerability`
- `combined_answerability`
- `generator_rationale`
- `why_two_users_needed`
- `per_user_evidence_claims`
- `attempt_count`
- `video_evidence`
- `referred_timestamps`
- `human_audit`
- `generation_trace`
- `review`
- `model_id`
- `source_urls`

最终 `review` 由 `generate_video_qa_loop` 根据 judger、answerability evaluation 和 deterministic schema validation 生成。strict validation 要求 `review.status == "passed"`、`review.review_passed == true`，并且下面这些 judger blocking checks 全部为 `PASS`：

- `qa_formality`
- `evidence_groundedness`

`generation_trace` 保存人眼核查需要的 intermediate data，包括 generation prompt/raw output、judger prompt/raw output、retry 时传回 generator 的 feedback、answerability conditions，以及每个 condition 实际使用的视频路径。只要传入 `--intermediate-output`，同样的 trace 也会单独写成 JSONL，方便后续人工检查。

运行严格校验：

```bash
python -m egolife_two_user_qa validate_outputs \
  --qa egolife_two_user_qa/outputs/pilot_20/qa_mcq.jsonl \
  --csv-output egolife_two_user_qa/outputs/pilot_20/qa_mcq.csv \
  --report egolife_two_user_qa/outputs/pilot_20/generation_report.md \
  --strict-review
```
