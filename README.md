# EgoLife 双用户 QA Pilot

这个模块用于从 EgoLife 视频中构造 20 条 pilot 多选题。每道题都要求至少两个用户的第一视角视频共同提供证据，单个用户的视频不能完整回答。

默认模型是 `Qwen/Qwen3.6-27B`。流程不使用 OpenRouter/Gemini 等商业 API key。`HF_TOKEN` 只作为 Hugging Face 下载或限流辅助，不作为推理 API key。

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

### Opt-in ten-minute, memory-safe path

The original `prepare_evidence` default and the historically named
`hpc/run_egolife_three_modes.sbatch` remain 30-second paths; that launcher is
now baseline-only. Ten-minute evidence is isolated in a separate launcher:

```bash
sbatch hpc/run_egolife_three_modes_10min_memory_safe.sbatch
```

That launcher explicitly requests 600-second synchronized windows. EgoLife
stores each recording as consecutive 30-second files, so preparation requires
all 20 source segments for a user and first attempts to losslessly concatenate
them into one cached MP4. If a complete set of segments is slightly shorter
than its nominal timestamps, the assembler pads the final video frame to
exactly 600 seconds; it rejects shortfalls over 30 seconds as incomplete.
Windows are non-overlapping and aligned to ten-minute clock
boundaries; a window is emitted only when at least two users have the complete
sequence. Each evidence row retains all segment URLs, cache paths, gaze
summaries, and offsets in `source_segments` for auditability.

The dedicated path uses one resident `Qwen/Qwen3.6-27B` instance through the
`transformers-local-memory-safe` backend. Complete model calls are serialized,
including CPU video decoding, processor encoding, and `model.generate`, so the
parallel judge coordinator cannot retain overlapping per-request video tensors
or KV caches. It defaults to 1 FPS, 131,072 pixels per frame, a 131,072-token
hard input ceiling, and a 5 GiB GPU workspace reserve beyond the request's
estimated KV cache. FlashAttention 2, per-call peak-VRAM telemetry, and
allocator-cache release are enabled. The two-video estimate is about 76,800
visual tokens, or 84,992 input tokens with the conservative text allowance.

The launcher requests 160 GiB of host RAM. It requires at least 96 GiB of
system/cgroup-aware available RAM before loading Qwen3.6-27B, then preserves at
least 16 GiB before video decoding and after processor encoding. At 1 FPS, the
two videos' 512-edge float32 frame tensors are at most about 3.5 GiB for one
copy and about 10.5 GiB with a conservative three-copy processing allowance.
Before Qwen sees a video, FFmpeg physically transcodes and caches it at 1 FPS
and a 512-pixel maximum edge. This prevents decoder implementations from first
materializing all frames of the original 10-minute, full-frame-rate MP4.

Before any QA generation, the ten-minute launcher must prune each already
assembled synchronized two-video pair. It samples each full 600-second video at
one frame per second and uses K=80 clusters per video. This is the duration-scaled
equivalent of K=4 for a 30-second video: both average 7.5 sampled frames per
cluster. CLIP encoding is split into batches of 32 frames to bound host and GPU
working memory. The resulting evidence routes `local_video` to the K=80-pruned
MP4 for generation, while `full_local_video` and `original_local_video` retain
the unpruned ten-minute MP4 for every judge and answerability call. Pair
selection/filtering is not rerun at this stage, and even
`EGOLIFE2U_REUSE_PREPROCESSED=1` reruns pruning from the unpruned ten-minute
evidence manifest.

The production pipeline exposes only the `baseline` generation mode.
`clip_guided`, `discovery`, and `discovery_control` are archived and rejected by
the CLI before any model call.

Its Qwen memory controls can be adjusted only for this path with `QWEN_MEMORY_SAFE_VIDEO_FPS`,
`QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS`, `QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS`,
`QWEN_MEMORY_SAFE_GPU_RESERVE_GIB`, `QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB`,
`QWEN_MEMORY_SAFE_PREFLIGHT_MIN_AVAILABLE_RAM_GIB`, and
`QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION`. The physical decoder-input cache is
controlled by `QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE` and
`QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR`; physical transcoding is mandatory in the
ten-minute launcher to protect host RAM.

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

Pruning can be protected from collapsing videos to very short clips. Set
`--pruning-protection-mode min_seconds --min-pruned-video-seconds 8` to restore
enough least-similar high-threshold sampled-frame intervals to keep at least 8
seconds per selected video, or set `--pruning-protection-mode min_percent
--min-pruned-video-percent 40` to keep at least 40% of the input window. The
restored intervals are chosen from frames whose best cross-video CLIP similarity
is still at or above `--high-similarity-interval-threshold`, ordered from least
similar to most similar.

### Complete 30-second pruning ablation

`run_pruning_ablation` is the standalone visual ablation for CLIP pruning. It
does not call the QA generator or any judge, and it only operates on fixed
30-second synchronized pairs. Every video is sampled and CLIP-encoded once at
the densest requested rate; lower-FPS timelines are deterministic subsets of
that shared cache.

The default experiment materializes four one-factor-at-a-time sweeps. Every
sweep contains its own current-pipeline control (`1 FPS`, `K=12`, threshold
`0.82`, timestamp-agnostic matching), while all non-target settings remain
fixed:

- temporal policy: current matching, hard `1s`/`2s`/`5s` gates, `2s`
  mutual-nearest matching, and `2s` mutual-nearest matching with non-contiguous
  visual clusters split into temporal runs;
- CLIP similarity threshold: `0.78,0.80,0.82,0.84,0.86,0.88`;
- sampling rate: `0.5,1,2,4 FPS`;
- cosine K-means clusters: `4,8,12,16,20,24,30`.

These are 23 controlled configurations per synchronized pair. Each
configuration writes both pruned MP4s, pruning intervals, cluster membership,
medoid images, trigger-pair timestamp differences, and fragmentation metrics.
The combined `review.html` groups videos first by pair and then by sweep.

```bash
python -m egolife_two_user_qa run_pruning_ablation \
  --manifest outputs/pruning_ablation_30s/manifest.json \
  --output-dir outputs/pruning_ablation_30s/experiment \
  --cache-dir /scratch/${USER}/egolife_two_user_qa_cache \
  --pair-count 10 \
  --fps-values 0.5,1,2,4 \
  --k-values 4,8,12,16,20,24,30 \
  --threshold-values 0.78,0.80,0.82,0.84,0.86,0.88 \
  --temporal-policies current,gate_1s,gate_2s,gate_5s,gate_2s_mnn,gate_2s_mnn_contiguous \
  --download-media
```

The complete manifest-plus-ablation cluster job is:

```bash
sbatch hpc/run_pruning_ablation_30s.sbatch
```

The Slurm launcher requires `hpc/cuda.py`, starts the CUDA keeper before
manifest construction or CLIP loading, keeps it alive through every sweep, and
stops it with an exit trap. The job fails instead of silently continuing if the
keeper script, dependencies, or controller process are unavailable. Override
`CUDA_KEEPER_THRESHOLD`, `CUDA_KEEPER_GPUS`, `CUDA_KEEPER_RESERVE`, or
`CUDA_KEEPER_AUTO_INSTALL` when needed; its log is written beside the
experiment output as `cuda_keeper_<job-id>.log`.

Primary outputs are `ablation_metrics.csv`, `sweep_aggregates.csv`,
`cluster_assignments.csv`, `trigger_pairs.csv`, `centroid_frames.csv`, and
`review.html`. The experiment is intentionally separate from the ten-minute
pipeline and generation loop.

### Fixed-pair pruning K grid

`run_pruning_k_grid` is the controlled cluster-count experiment. It samples each
synchronized two-video pair once, extracts the 30 one-FPS frames once, computes
CLIP embeddings once, and applies every requested K to those same inputs. It
does not use the K-dependent pair-survival filters from benchmark mining.

The default grid is `4,8,12,16,20,24,30`. Sampling remains at one FPS, so the
existing pruning interval remains fixed at +/-0.5 seconds. Duration protection
defaults to `min_seconds` with an eight-second floor so aggressive K values can
still be reviewed; zero-removal and fully collapsed variants are retained in the
diagnostics instead of being silently skipped.

```bash
python -m egolife_two_user_qa run_pruning_k_grid \
  --manifest outputs/pruning_k_grid/manifest.json \
  --output-dir outputs/pruning_k_grid/grid \
  --cache-dir /scratch/${USER}/egolife_two_user_qa_cache \
  --pair-count 10 \
  --k-values 4,8,12,16,20,24,30 \
  --download-media
```

On the cluster, the complete manifest-plus-grid job is:

```bash
sbatch hpc/run_pruning_k_grid_10.sbatch
```

The grid produces normalized original videos, `K_XX/left_pruned.mp4` and
`K_XX/right_pruned.mp4` variants, per-variant `pruning.json` traces,
`grid_metrics.csv`, aggregate `summary.json`, and a side-by-side `review.html`.
The review page displays the complete sampled one-FPS timeline for both videos.
For every K it also shows each medoid frame (the effective center actually used
for cross-video comparison), every frame assigned to that cluster, final
removed/restored/kept member status, and the medoid-pair similarities that
triggered pruning. The same information is stored as `K_XX/cluster_trace.json`,
with flattened tables in `cluster_assignments.csv` and `trigger_pairs.csv`.
Each effective cluster's representative image is also copied explicitly to
`K_XX/centroid_frames/{left,right}/cluster_XX_centroid_frame_*.png`. These are
the sampled medoids nearest the cosine-k-means centroid vectors and therefore
the actual frames used for pruning comparisons. `K_XX/centroid_frames.json`
and the global `centroid_frames.csv` index every exported image.

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
  --evidence-duration-seconds 30 \
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

### Gemini 2.5 Flash Backend

The QA loop can also use Gemini through the native Gemini API while preserving
the same evidence JSONL paths and media routing. Qwen remains the default; use
Gemini only by selecting the backend explicitly.

```bash
export GEMINI_API_KEY="..."

python -m egolife_two_user_qa generate_video_qa_loop \
  --backend gemini \
  --model-id gemini-3.5-flash \
  --evidence egolife_two_user_qa/outputs/pilot_20_video_first/evidence_manifest.jsonl \
  --output egolife_two_user_qa/outputs/pilot_20_video_first/qa_mcq.gemini.jsonl \
  --prompts-output egolife_two_user_qa/outputs/pilot_20_video_first/video_first_prompts.gemini.jsonl \
  --intermediate-output egolife_two_user_qa/outputs/pilot_20_video_first/qa_mcq.gemini.intermediate.jsonl
```

For sbatch runs that call `generate_video_qa_loop`, keep the existing evidence
root/path variables and set:

```bash
export VLM_BACKEND=gemini
export GEMINI_API_KEY="..."
# optional; defaults to gemini-3.5-flash when VLM_BACKEND=gemini
export VLM_MODEL_ID=gemini-3.5-flash
```

Unset `VLM_BACKEND` or set `VLM_BACKEND=transformers-local` to switch back to
Qwen. Existing `local_video`, `original_local_video`, and `full_local_video`
fields are still used; Gemini uploads those local files through its Files API
at call time.

### K=40 sampling with local Qwen judges

The K=40 sampling job uses one resident `Qwen/Qwen3.6-27B` runner for
generation, the text-only qa_formality judge, the full-video
evidence_groundedness judge, and all answerability conditions. No judge backend
override is passed, which is deliberate: it avoids constructing a second copy
of the same local model. Model weights remain loaded across the 50 packets, but
each generation or judge call has its own fresh KV cache.

Run the maintained sampling launcher with K fixed to 40:

```bash
SAMPLING_TOP_K=40 sbatch hpc/run_clip_pruned_sampling_neutral_pf_50.sbatch
```

The generator receives the pruned videos at 1 FPS. The visual Qwen verification
branches receive the full original videos at 1 FPS, while qa_formality remains
text-only. Both model-judge prompts receive the generator's
`generator_rationale` so they can inspect the intended cross-view relation; the
visual judge is still instructed to verify every rationale claim against the
full originals rather than treating it as evidence. Both judges now use only a
binary PASS/FAIL contract. The former 1/2/3 score, quality rationale, quota, and
quota-rebuttal path is archived and is not included in production prompts,
traces, or accepted rows.

PASS/FAIL choice-logit and decision-entropy JSON is a legacy archived
experiment. Production judge calls no longer request or attach it, and strict
acceptance validation does not require it. The A-E answerability evaluator also
uses ordinary JSON generation; it no longer requests or stores choice logits or
entropy.

Production generation and judging are category-free. Generated items do not emit
`category` or `category_rationale`; schema validation, CSV/review exports, and the
judge payload do not request or verify those fields. The old broad taxonomy is
retained only as an offline analysis catalog.

A comparison of the recent runs found that object identification, location,
ordinary task details, and temporal continuation already appeared naturally
without taxonomy steering. Count-specific steering was tested and then removed
because literal absence alone did not make counting useful. The active
category-free prompt now contains optional, equal-status hints for the genuinely
underrepresented directions: cross-view comparison/asymmetry, identity or role
linkage, post-handoff follow-up, concrete state verification, and strict
concurrent activity-pair comparison. These are definitions and benchmark-grounded
examples, not labels or output fields; the generator is told not to imitate or
converge on them. Ordinary grounded information-gap questions remain valid.

The concurrent form applies only when both pruned videos contain several bounded
activities and asks which pairing of one activity visible in each view overlaps
on the original synchronized timeline. An activity may be performed by the wearer
or by anyone clearly visible to that wearer. Options must recombine real visible
activities so neither single video can select the pair alone. The old generic
concurrent-activity block remains archived and is not rendered.

Because retained intervals are concatenated independently for the two pruned
videos, prompt metadata now includes a per-video pruned-to-original time map. The
generator is explicitly told never to equate compressed playback positions across
the two videos.

Run the category-free implicit underrepresented-family experiment with:

```bash
sbatch hpc/run_implicit_underrepresented_families_50.sbatch
```

The launcher uses neutral generation with sampling temperature 0.7, top-p 0.9,
and optional top-k, writing to
`egolife_two_user_qa/outputs/implicit_underrepresented_families_50` by default. Before loading
the model it verifies that the generator and judge prompts contain no category
language, that the generation schema has no category fields, that the retired
count direction remains absent, and that all five optional definitions, the
strict pair-comparison contract, and pruned-to-original temporal maps are rendered.

### Generator rationale-removal paired ablation

`generator_rationale_ablation` takes one completed
`qa_mcq.intermediate.jsonl` file and selects only attempt 1 from each packet.
Retry attempts are never substituted. If attempt 1 did not form a parseable
question, that packet is skipped by the paired ablation. Previous pipeline
decisions are retained only as output audit metadata and are never shown to
either ablation judge.

Each question receives exactly two independent evidence-groundedness calls
against the same full original videos. The prompt, question, options, declared
answer, video order, resolution, and decoding are identical. The only treatment
difference is that one payload contains `generator_rationale` and the other
does not. Other generator-authored reasoning fields such as
`why_two_users_needed`, evidence claims, and referred timestamps are absent in
both conditions. Call order alternates deterministically across questions so
one condition is not always first.

This run is binary only: each call returns `PASS` or `FAIL` plus a short visual
reason. There are no 1/2/3 scores, ranks, global quotas, cross-candidate memory,
or all-at-once dense prompt/raw-output JSON traces. The sole artifact is a
compact paired JSONL with the two decisions and their status pair.
Malformed judge serialization is retried up to three times with a format-only
repair instruction; invalid raw responses are never stored. The compact row
records only the number of format attempts for each condition. Override this
with `RATIONALE_ABLATION_MAX_FORMAT_ATTEMPTS` if needed.

The judge is explicitly told to verify every object noun rather than inherit
the generator's interpretation. For example, a question that calls a bowl of
dough a bowl of chips must fail even if its rationale fluently claims that the
bowl contains chips. `without_FAIL__with_PASS` is the direct pattern supporting
the hypothesis that rationale removal exposes a mistake. `without_PASS__with_PASS`
means neither condition rejected the item, but it should be called "misled
regardless" only after a human confirms that the item really contains a visual
mistake.

The Slurm workflow then automatically starts a fresh 50-packet production run
on the matching evidence JSONL. It uses the maintained neutral sampling setup
(temperature 0.7, top-p 0.9, top-k 40 by default), feeds `generator_rationale`
to both review judges, and uses the same binary PASS/FAIL and category contract
described above. Judge and answerability logits, entropy diagnostics, and point
scoring remain archived and are not collected. A preflight
requires the first 50 evidence IDs to exactly match the 50 baseline intermediate
rows.

Submit the baseline intermediate and its evidence file:

```bash
sbatch hpc/run_generator_rationale_ablation_qwen.sbatch \
  path/to/qa_mcq.intermediate.jsonl \
  path/to/evidence_pruned_pairs.jsonl
```

The paired output is `paired_pass_fail.jsonl`. The production outputs are under
`production_rationale_binary_50/`, including `acceptance_comparison.json` with
the baseline and new production acceptance rates, absolute percentage-point
change, generation-attempt totals, and a higher/lower/no-change result. The comparison
uses the same evidence packets, but the questions are freshly sampled; it is an
end-to-end production comparison rather than a paired causal estimate of
rationale removal alone. After verification, accepted and rejected production
JSONLs retain `attempt_count` and the judge configuration but drop dense
`generation_trace`/`attempts` histories. `RATIONALE_ABLATION_OUTDIR` overrides
the root output directory. Both phases start fresh by default; explicitly set
`RATIONALE_ABLATION_RESUME=1` or `RATIONALE_PRODUCTION_RESUME=1` only when a
resume is desired. The
launcher also starts `hpc/cuda.py` before the ablation and keeps it running
through the production phase; it is stopped automatically when the job exits.
Set `CUDA_KEEPER_ENABLE=0` to disable it, or override
`CUDA_KEEPER_THRESHOLD`, `CUDA_KEEPER_GPUS`, and `CUDA_KEEPER_RESERVE`.

## Archived generation modes

The former `clip_guided`, `discovery`, and `discovery_control` ablations are no
longer production modes. Their helper prompts remain in source only for reading
old artifacts and offline historical reproduction. They are excluded from
`GENERATION_MODES`, hidden from CLI choices, and have no active routing in the
generation loop. The old discovery-control launcher exits immediately with an
archive notice; the historical implementation remains below that guard.

## 输出 Schema

`qa_mcq.jsonl` 每一行是一条 QA，包含：

- `qa_id`
- `question`
- `options`
- `correct`
- `answer`
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
