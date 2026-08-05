# EgoLife 双用户 QA Pilot

这个模块用于从 EgoLife 视频中构造 20 条 pilot 多选题。每道题都要求至少两个用户的第一视角视频共同提供证据，单个用户的视频不能完整回答。

默认模型是 `Qwen/Qwen3.6-27B`。流程不使用 OpenRouter/Gemini 等商业 API key。`HF_TOKEN` 只作为 Hugging Face 下载或限流辅助，不作为推理 API key。

On HPC, the checkout root is
`/scratch/${USER}/Long-video-understanding-clip`. All Slurm launchers, workers,
environment scripts, and helper programs are resolved from
`${PROJECT_ROOT}/hpc` directly. The nested `egolife_two_user_qa/` directory is
used for Python package data and outputs only; no HPC executable is expected
under `egolife_two_user_qa/hpc`.

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

PASS/FAIL choice logits are disabled in ordinary production. The opt-in entropy
mode below adds a separate diagnostic judge call; strict acceptance validation
does not require its verdict or entropy. The A-E answerability evaluator still
uses ordinary JSON generation and does not request or store choice logits or
entropy.

### Independent minimal-verdict production entropy

The former entropy run captured the nested `checks.<judge>.status` token after
the model had already emitted `review_passed`. Its near-zero values therefore
mostly measured consistency with a verdict already present in the generated
prefix, not uncertainty about the judge decision.

The redesigned mode runs each model judge twice. The first call is the original
detailed production judge: it returns `review_passed`, nested checks, reasons,
fixes, blocking failures, and generator feedback. Its nested
`checks.<judge>.status` remains the only model-judge decision used for
acceptance and retries.

The second call is an independent diagnostic probe. It receives the same judge
rubric, generated QA candidate, and the same media as that judge, but it is not
shown the detailed judge's output. Its entire response must be exactly one of:

```json
{"verdict":"pass"}
```

```json
{"verdict":"fail"}
```

No checks, reasons, fixes, feedback, scores, or other fields are permitted. The
probe verdict and its entropy are recorded only for analysis; they cannot alter
acceptance, retry feedback, or final QA selection. Deterministic schema checks
and the separate answerability gate continue to work exactly as before.

Qwen3.6-27B encodes both lowercase values as single tokens (`pass` token 6184
and `fail` token 17854 in the cached tokenizer). The run performs this tokenizer
preflight before inference. The minimal probe requires `verdict` to be its only
JSON field. A malformed or unavailable probe is marked unavailable while the
separate detailed production decision remains unchanged.

Run a fresh production generation over the exact saved 001-100 evidence cohort:

```bash
sbatch hpc/run_production_entropy_qa_001_100.sbatch
```

The new QA run writes under
`outputs/qa_001_100_production_independent_minimal_entropy_v3/packets_001_100/`.
The `_v3` root preserves both the completed entropy-less run produced by the
CLI forwarding bug and the earlier coupled first-verdict run.
Alongside the accepted, rejected, prompt, and intermediate QA artifacts it
writes:

- `judge_entropy_attempts.jsonl`: both model judges for every judged generation
  attempt, including probe probabilities, entropy, the independent probe
  verdict, the authoritative production status, retry outcome, and eventual
  packet acceptance/rejection;
- `judge_entropy_summary.json`: aggregates by judge, probe verdict, production
  post-merge status, retry outcome, and final packet status;
- `judge_entropy_report.md`: a compact human-readable comparison.

The same two flattened measurements are also stored at
`qa_mcq.intermediate.jsonl -> attempts[*].judge_entropy` (or
`generation_trace[*].judge_entropy` for a finally rejected packet), so the
entropy, retry decision, and final packet result can be inspected in one trace.
The production entropy launcher defaults to `RESUME_GENERATION=0`: rerunning it
recreates the batch instead of skipping outputs from a prior non-entropy run.
Set resume to `1` only when continuing an interrupted run that already contains
integrated entropy artifacts.

This makes the relationship traceable without changing the production
experiment: the detailed verdict affects retries and final QA selection, while
the second minimal verdict lets us compare diagnostic entropy with that outcome.

### Offline first-verdict sidecar

The offline sidecar remains available for rerunning judges on an existing trace
without generating new QAs. It retains the older detailed first-verdict
compatibility contract and does not affect the saved run's original gates. It
is distinct from the production pipeline's new one-field minimal probe.

`ENTROPY_SOURCE_INTERMEDIATE` means the prior QA run's
`qa_mcq.intermediate.jsonl` trace. It is not an entropy artifact: unlike the
accepted-only `qa_mcq.jsonl`, it retains every generation attempt together with
the exact judge prompts, original model PASS/FAIL labels, and media paths needed
to rerun the judges offline.

The offline launcher defaults to the saved fixed 001-100 baseline cohort:
`outputs/qa_300_three_jobs_safe_prompts/packets_001_100/qa_mcq.intermediate.jsonl`.
Run that cohort from the project root with:

```bash
sbatch hpc/run_judge_entropy_sidecar.sbatch
```

To probe a different completed QA run, override the source and give it a
separate output directory:

```bash
ENTROPY_SOURCE_INTERMEDIATE=egolife_two_user_qa/outputs/<run>/qa_mcq.intermediate.jsonl \
ENTROPY_OUTPUT_DIR=egolife_two_user_qa/outputs/judge_entropy_first_verdict_detailed/<run> \
sbatch hpc/run_judge_entropy_sidecar.sbatch
```

The output directory contains the selected task manifest, preflight class
counts, raw JSONL/CSV sidecar results, `summary.json`, and `report.md`. By default
the launcher keeps the natural label prevalence across all attempts and requires
at least 20 examples in every judge-by-status cell. Set
`ENTROPY_BALANCE_STATUSES=1` only for a status-balanced sensitivity analysis;
calibration metrics should use the default natural cohort.

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

## Retained-centroid-frame generator sidecar

The centroid-frame sidecar is an additive generation experiment. It consumes
the existing 300 CLIP-pruned evidence packets and reads their saved cluster
decisions. For each required user, it copies one CLIP medoid for every cluster
that still has retained content after pruning and duration protection. Those
images are ordered by original timestamp and sent directly to the generator;
the sidecar does not extract one-second intervals, run ffmpeg, or build a new
pruned MP4. Evidence-groundedness and answerability still receive the full
original synchronized videos.

The centroid experiment remains on its separate launchers, evidence JSONL,
frame assets, QA outputs, and post-run media-routing verification:

Run these commands from the project root
`/scratch/${USER}/Long-video-understanding-clip`. The sidecar launchers, its
worker, `run_qa_packet_slice_100.sh`, `env_qwen3vl.sh`, and `cuda.py` are all
resolved exclusively from `${PROJECT_ROOT}/hpc`; the sidecar never expects an
`egolife_two_user_qa/hpc` directory on HPC.

```bash
sbatch hpc/run_centroid_frame_qa_packets_001_100.sbatch
sbatch hpc/run_centroid_frame_qa_packets_101_200.sbatch
sbatch hpc/run_centroid_frame_qa_packets_201_300.sbatch
```

Outputs default to `outputs/qa_300_centroid_frame_sidecar/`. Override the
existing pruned packet source with `CENTROID_SIDECAR_SOURCE_EVIDENCE` or the
sidecar output root with `CENTROID_SIDECAR_OUTPUT_ROOT`.

For a local or one-off conversion without generation:

```bash
python -m egolife_two_user_qa.centroid_frame_sidecar prepare \
  --evidence outputs/clip_pruned_packets_300_mixed_temporal/evidence_pruned_pairs.jsonl \
  --output outputs/centroid_frame_sidecar/evidence.jsonl \
  --output-dir outputs/centroid_frame_sidecar/assets \
  --start-index 0 \
  --max-packets 10
```

Each sidecar packet uses `generator_media_mode="centroid_frames_only"`, stores
the ordered images in `clips[*].frames`, removes generator-video routing, and
retains `clips[*].full_local_video` for the visual checks.

## Default retained-cluster-member-frame generation

The default production launchers give the generator the frames in clusters that
survived pruning, without reconstructing an MP4. They read the saved CLIP
pruning diagnostics and, for every retained cluster, send all of that cluster's
CLIP-sampled member frames. For a cluster otherwise pruned but partly restored
by duration protection, they send only the explicitly restored members. Images
are ordered by original timestamp within each user. These are the frames sampled
for CLIP (normally one per second), not every native video frame. Judges and
answerability continue to receive the full original videos. Both pruning-output
schemas are supported: newer packets read `sampled_frames` from packet
diagnostics, while legacy group-relative packets recover the same sampled
timeline from the saved frame files beside each cluster medoid.

Run the default 300-packet production split from
`/scratch/${USER}/Long-video-understanding-clip`:

```bash
sbatch hpc/run_qa_packets_001_100.sbatch
sbatch hpc/run_qa_packets_101_200.sbatch
sbatch hpc/run_qa_packets_201_300.sbatch
```

These standard launchers now delegate to
`run_cluster_member_frame_qa_packet_slice_100.sh`. The explicitly named
`run_cluster_member_frame_qa_packets_*.sbatch` launchers remain equivalent
aliases. The lower-level `run_qa_packet_slice_100.sh` worker still accepts
already-prepared evidence directly, so controlled pruned-video ablations remain
available without changing this production default.

Outputs default to `outputs/qa_300_cluster_member_frame_sidecar/`. Override the
source with `CLUSTER_MEMBER_SIDECAR_SOURCE_EVIDENCE` or the output root with
`CLUSTER_MEMBER_SIDECAR_OUTPUT_ROOT`. Each prepared packet uses
`generator_media_mode="retained_cluster_frames_only"` and routes an ordered
image list to Qwen with no generator MP4.

## DAY3-7 RLHF annotation run

Two independent 48-hour H100 jobs create 100 new evidence pairs apiece and
generate questions for future annotation. The jobs use only `DAY3` through
`DAY7`; `DAY1` and `DAY2` are excluded. Synchronized `(day, time_token)` groups
are assigned by a stable SHA-256 parity partition, so the two jobs can run
concurrently without selecting the same source group.

```bash
sbatch hpc/run_rlhf_annotation_day3_7_pairs_001_100.sbatch
sbatch hpc/run_rlhf_annotation_day3_7_pairs_101_200.sbatch
```

Both jobs write beneath the exact directory
`egolife_two_user_qa/outputs/RLHF annotation run/`:

```text
source_packets/
  pairs_001_100/evidence_pruned_pairs.jsonl
  pairs_101_200/evidence_pruned_pairs.jsonl
question_generation/
  pairs_001_100/
  pairs_101_200/
```

Each source cohort is verified to contain exactly 100 unique evidence IDs and
100 unique synchronized groups from `DAY3`-`DAY7`. Question generation uses the
default retained-cluster-member representation: the generator receives all
one-FPS sampled members of clusters that survived pruning, while visual judges
and answerability checks receive the full original videos. Each generation
directory includes accepted and rejected JSONL, prompt traces, a CSV annotation
export, a human-review sheet, and routing verification.

The shared worker is
`hpc/run_rlhf_annotation_day3_7_100_worker.sh`. Override
`RLHF_ANNOTATION_OUTPUT_ROOT` to relocate the complete run or
`EGOLIFE2U_CACHE_DIR` to change the video cache. Model and generation overrides
supported by the established QA worker, such as `QWEN_MODEL_ID`,
`MAX_ATTEMPTS`, and `RESUME_GENERATION`, are inherited.

## Fixed-six reward-model candidate collection

`reward_candidate_collection.py` is a separate data-collection controller for
future reward-model training. The Slurm launcher builds its evidence from
scratch: it fetches a raw EgoLife manifest for the requested days, filters it
to the requested clock window, mines and downloads synchronized two-user
pairs with the current CLIP-pruning pipeline, and materializes the retained
cluster-member frames. The resulting internal
`evidence_build/evidence_cluster_member_frames.jsonl` is then frozen for the
candidate collector. None of these stages alters the production QA loop.

For each selected packet the controller retains exactly six structurally valid
generations. A feedback loop has at most three attempts and stops on the first
production-gate pass. The controller then starts a fresh, independent loop
until the fixed quota is reached. Both passing and failing valid attempts are
retained, so progressive revisions remain available as natural preference
trajectories. Malformed JSON is recorded and replaced, but it consumes neither
the six-candidate quota nor one of the three valid attempts in its current
judge loop. Raw generator calls and valid loop attempts have separate counters.
Only a structurally valid, judged candidate can be the semantic parent of a
feedback refinement; a replacement before the first valid candidate is an
independent root. Structural errors from malformed calls are passed forward as
format-repair feedback without making the malformed text part of the candidate
lineage.

Generation uses the existing retained CLIP-cluster-member frames. Its prompt
explicitly describes them as sparse, chronologically ordered samples and
forbids unsupported inference about actions, transitions, or moments between
frames. The existing parallel formality, groundedness, and answerability path
remains unchanged and receives the full original videos. Those same two
full-video paths are the media references recorded for future reward-model
examples. Automated PASS or FAIL is metadata rather than a human label.
Sampling defaults to temperature `0.70`, top-p `0.95`, and top-k `40`, with a
stable unique seed for every generation identity.

On the cluster, both launchers belong in the project-root `hpc/` directory,
next to the current `env_qwen3vl.sh`, `cuda_slurm.py`, and `cuda.py`. The worker
discovers the project root from that location rather than depending on the
project directory's name. It activates the existing
`/scratch/$USER/conda/envs/qwen3vl-smoke` environment through the root helper
and verifies its Python prefix, required Qwen/CLIP imports, one visible CUDA
GPU, and bfloat16 support before building evidence.

The recommended submission divides one start-inclusive/end-exclusive interval
into two non-overlapping halves and immediately submits one independent H100
job for each half. `REWARD_TOTAL_PACKET_COUNT` is the total across both jobs;
100 becomes 50 packets and 300 questions per half:

```bash
cd /scratch/$USER/Long-video-understanding-clip

REWARD_WINDOW_LABEL=day1_4_full_day \
REWARD_DAYS=DAY1,DAY2,DAY3,DAY4 \
REWARD_START_CLOCK=06:00 \
REWARD_END_CLOCK=18:00 \
REWARD_TOTAL_PACKET_COUNT=100 \
bash hpc/submit_reward_candidate_collection_two_halves.sh
```

This example submits `[06:00,12:00)` and `[12:00,18:00)` concurrently with
distinct output labels. Set `REWARD_SUBMIT_DRY_RUN=1` to print and validate the
two submissions without calling `sbatch`. Use `REWARD_OUTPUT_ROOT` to relocate
both outputs; the submitter intentionally ignores the single-job
`REWARD_OUTPUT_DIR` override so the concurrent jobs cannot collide. The
underlying worker remains
available for a single custom window through
`hpc/run_reward_candidate_collection_qwen36_27b.sbatch` and its per-job
`REWARD_PACKET_COUNT` variable.

The worker defaults to `Qwen/Qwen3.6-27B`, one GPU, and one resident model
shared by generation and the serialized local judge calls. Different window
labels create different output directories and can run concurrently. Set
`REWARD_OUTPUT_DIR` explicitly and `REWARD_RESUME=1` to resume an interrupted
window with the exact same configuration.

Each window directory retains the raw and filtered manifests, CLIP-pruned pair
evidence, and generated frame sidecar beneath `evidence_build/`. No external
reward-evidence path is required. Candidate outputs are divided into groups of
25 source packets. A 100-packet run therefore creates:

```text
packet_groups/
  packets_001_025/  # 25 packets, exactly 150 candidates
  packets_026_050/  # 25 packets, exactly 150 candidates
  packets_051_075/  # 25 packets, exactly 150 candidates
  packets_076_100/  # 25 packets, exactly 150 candidates
```

Each group contains its own `evidence_manifest.jsonl`, `candidates.jsonl`,
`candidate_details.jsonl`, `packet_summaries.jsonl`,
`malformed_outputs.jsonl`, `discarded_packets.jsonl`, `group_summary.json`, and
`packet_records/*.json` restart checkpoints. The root contains only the small
`packet_groups.jsonl`, `collection_summary.json`, run manifest, and complete
evidence manifest. All artifacts are ordinary uncompressed JSON or JSONL. A
successful 100-packet run is required to contain all 600 candidates; a missing
six-candidate packet makes final verification fail instead of silently
producing a short dataset.

## Fixed 001–100 QA ablations

Two one-factor arms reuse the saved
`outputs/qa_300_three_jobs_safe_prompts/packets_001_100/evidence_slice.jsonl`
directly. Both therefore use the same 100 evidence IDs, original videos, and
pruned videos:

| Arm | CLIP-pruning FPS | Visual judge media |
| --- | ---: | --- |
| `fps_0p5` | `0.5` | full originals |
| `pruned_judges` | `1.0` | pruned videos |

The first arm rebuilds the pruned videos from the same full originals with a
2-second sample interval (`0.5 FPS`) while holding `K=12`, threshold `0.82`,
minimum-8-second pruning protection, model inference, and full-original judge
routing fixed. Its input check allows the saved nominally 30-second clips to
measure within 3 seconds of that target, accommodating container durations such
as 28.6 seconds without changing the 30-second pruning window. The second
reuses the existing baseline 1-FPS pruned videos and
routes both evidence-groundedness and every answerability condition to the same
pruned media seen by the generator. The text-only formality judge remains
text-only.

```bash
sbatch hpc/run_ablation_qa_001_100_fps_0p5.sbatch
sbatch hpc/run_ablation_qa_001_100_pruned_judges.sbatch
```

Run these commands from the project root. The launchers, shared worker, environment
script, and `cuda.py` all live in the project-level `hpc/` directory; only package
data and outputs resolve through `egolife_two_user_qa/`.

Set `EGOLIFE2U_REFERENCE_001_100` if the saved slice lives elsewhere. Outputs
are separated under `outputs/qa_001_100_two_ablations/`. Each job verifies
100-row coverage, unique and ordered evidence-ID identity after re-pruning, the
declared pruning FPS, and visual-judge media routing before completing.

### Threshold and K follow-up arms

Two additional one-factor launchers use the same saved 001–100 cohort and the
same shared worker:

| Arm | CLIP-pruning FPS | K | Threshold | Visual judge media |
| --- | ---: | ---: | ---: | --- |
| `threshold_0p85` | `1.0` | `12` | `0.85` | full originals |
| `k_8` | `1.0` | `8` | `0.82` | full originals |

`K=8` is the closest established grid point below the `K=12` control. It gives
a meaningful 33% reduction in cluster count without the more aggressive jump
to `K=4`. Each arm changes only its named factor; sampling, pruning protection,
generation settings, and judge routing remain at the baseline values.

```bash
sbatch hpc/run_ablation_qa_001_100_threshold_0p85.sbatch
sbatch hpc/run_ablation_qa_001_100_k_8.sbatch
```

Outputs are isolated under
`outputs/qa_001_100_threshold_k_ablations/{threshold_0p85,k_8}/`. The worker
checks the realized FPS, K, threshold, evidence-ID order, full-video judge
routing, and complete 100-pair coverage, then records the arm metadata in both
the rebuilt evidence packets and `generation_summary.json`.

## CPU-only Gemini answerability verification

After a generation-loop job completes, the accepted `qa_mcq.jsonl` rows can be
checked once more with `google/gemini-3.5-flash` through OpenRouter:

```bash
sbatch hpc/run_answerability_verification_gemini35_flash_cpu.sbatch
```

Put every generation-run directory to verify in
`hpc/answerability_verification_run_dirs.txt`, one per line. Entries may be
absolute, or relative to `egolife_two_user_qa/`:

```text
outputs/qa_300_three_jobs_safe_prompts/packets_001_100
outputs/qa_300_three_jobs_safe_prompts/packets_101_200
outputs/qa_300_three_jobs_safe_prompts/packets_201_300
outputs/qa_001_100_two_ablations/fps_0p5/packets_001_100
outputs/qa_001_100_two_ablations/pruned_judges/packets_001_100
outputs/qa_001_100_threshold_k_ablations/threshold_0p85/packets_001_100
outputs/qa_001_100_threshold_k_ablations/k_8/packets_001_100
```

Only list completed directories containing `qa_mcq.jsonl`,
`evidence_slice.jsonl`, and `generation_summary.json`. The job validates every
listed directory before its first billable API call. For a one-off run,
`EGOLIFE2U_QA_RUN_DIR` overrides the manifest.

`OPENROUTER_API_KEY` must be exported when the job is submitted. This is a
CPU-only job: it requests no GPU, starts no CUDA keeper, and loads no local
model. It makes three verification calls per accepted two-user QA, preserves
that QA's original `full` or `pruned` judge-media route, and writes
`verification.jsonl`, `prompts.jsonl`, and `summary.json` under
`answerability_verification_gemini35_flash_minimal_reasoning/`.
Gemini 3.5 Flash requires reasoning on OpenRouter, so the launcher uses its
lowest supported `minimal` effort instead of the rejected `none` setting.
`RESUME_VERIFICATION=1` is the default, so completed QA IDs are not submitted
again after a retry. Before upload, videos are cached as moderate 720px, 2-FPS,
CRF-23 CPU transcodes to reduce OpenRouter gateway timeouts without aggressive
downsampling; set
`OPENROUTER_VIDEO_MAX_EDGE=0` and
`OPENROUTER_VIDEO_FPS=0` only when original-encoding uploads are required.

## Golden-label external-verifier benchmark

Use `answerability_verification_benchmark.py` to compare stronger external
verifiers against the manual review CSV. The benchmark uses only accepted QAs
with a human `review_status` of `Pass` or `Fail`. `Pending`, `Unsure`, blank,
and `Unset` rows are excluded rather than guessed.

Every model arm receives the same three independent conditions for each QA:
asker video only, evidence-provider video only, and both videos together. The
deterministic gate still requires the asker-only call to miss the declared
answer and the combined call to recover it; provider-only correctness remains
an allowed diagnostic. There is exactly one external pass and no verifier
output is returned to the generator or any retry loop.

The checked-in benchmark configuration uses the six runs represented in
`ablation_manual_review_fa662de7a5885094dd12.csv` and enables these arms:

- `google/gemini-3.5-flash` at `high`, isolating reasoning effort from the old
  minimal-effort experiment.
- `google/gemini-3.1-pro-preview` at `high`, testing a larger frontier Gemini
  model with native video input.
- `qwen/qwen3-vl-235b-a22b-thinking` at provider-default thinking, testing a
  large model from a different family.

The old `google/gemini-3.5-flash`/`minimal` arm remains in the JSON config with
`"enabled": false`; enable it only when a same-cohort baseline rerun is useful.

On the cluster, copy the exported gold CSV to the default path shown below (or
set `ANSWERABILITY_GOLD_CSV` to its absolute path), export the API key, and
submit the CPU-only job:

```bash
cp ablation_manual_review_fa662de7a5885094dd12.csv \
  egolife_two_user_qa/hpc/ablation_manual_review_fa662de7a5885094dd12.csv
export OPENROUTER_API_KEY="..."
sbatch egolife_two_user_qa/hpc/run_answerability_verifier_benchmark_cpu.sbatch
```

The launcher first runs a non-billable validation pass over the annotation-to-QA
join, accepted status, evidence coverage, exact full/pruned video files, and
OpenRouter video/reasoning capabilities. For the 140 labeled QAs, each enabled
arm makes 420 calls. With the three default stronger arms, the plan reports
1,260 calls before execution.

Results are written under `outputs/answerability_verifier_benchmark/`:

- `comparison.csv` and `comparison.json`: confusion matrices, accuracy,
  balanced accuracy, failure recall/precision, false-accept rate, confidence
  intervals, per-run metrics, per-error-tag metrics, and paired arm counts.
- `predictions.jsonl`: the gold label, final gate, and all three condition
  choices for every QA/model arm.
- `disagreements.jsonl`: only model-versus-human disagreements for inspection.
- `benchmark_plan.json`: the validated cohort and exact call count before the
  first model request.

To rescore completed outputs without another API call:

```bash
python -m egolife_two_user_qa.answerability_verification_benchmark score \
  --config egolife_two_user_qa/hpc/answerability_verifier_benchmark.json
```

### Small manually annotated cohorts

The benchmark also supports a single small annotation export without requiring
the six-run configuration. `hpc/answerability_verifier_small_benchmark.json`
expects one annotation run named `experiment` and takes the accepted-QA and
evidence paths from environment variables. For the 19-row
`ablation_manual_review_7d3817bdcbeb84ada5bd.csv` export (12 Pass, 7 Fail), run:

```bash
export OPENROUTER_API_KEY="..."
export ANSWERABILITY_BENCHMARK_CONFIG="${PWD}/egolife_two_user_qa/hpc/answerability_verifier_small_benchmark.json"
export ANSWERABILITY_GOLD_CSV="/path/to/ablation_manual_review_7d3817bdcbeb84ada5bd.csv"
export ANSWERABILITY_ACCEPTED_QA="/path/to/the/source/qa_mcq.jsonl"
export ANSWERABILITY_EVIDENCE_JSONL="/path/to/the/source/evidence_slice.jsonl"
export ANSWERABILITY_BENCHMARK_OUTPUT_DIR="${PWD}/egolife_two_user_qa/outputs/answerability_verifier_benchmark_19"
sbatch egolife_two_user_qa/hpc/run_answerability_verifier_benchmark_cpu.sbatch
```

The validation step matches the 19 annotated `qa_id` and `evidence_id` values
against that accepted-only `qa_mcq.jsonl`; passing an unrelated QA file fails
before any API request. The three enabled arms make 171 calls total (19 QAs ×
3 conditions × 3 arms). Treat this cohort as a screening stage: one item changes
raw accuracy by about 5.3 percentage points, so use balanced accuracy, the
reported Wilson intervals, and the row-level disagreements before promoting the
best arm to the 140-QA benchmark.

### Accepted-only `qa_mcq.jsonl` test set

For a new accepted-only set that does not yet have matching manual labels, use
the test-set commands instead of assigning unrelated gold rows. The checked-in
`hpc/answerability_verifier_qa_mcq_testset.json` targets the repository-root
`qa_mcq.jsonl`. That file currently contains 27 valid, unique, accepted JSONL
records (despite the informal 29-question count) and also contains one unique
top-level evidence packet per QA, so it is safely used as both the accepted-QA
and evidence input.

On the cluster:

```bash
export OPENROUTER_API_KEY="..."
sbatch egolife_two_user_qa/hpc/run_answerability_verifier_qa_mcq_testset_cpu.sbatch
```

The three configured OpenRouter arms make 243 calls total (27 QAs x 3 video
conditions x 3 arms). `validate-set` checks accepted status, evidence IDs, and
the exact full-video files before the first paid request. `run-set` writes
`testset_summary.json`, `testset_summary.csv`, `testset_predictions.jsonl`, and
`testset_disagreements.jsonl`. These report gate rates and cross-model
agreement, but deliberately do not rank a model or claim accuracy without
matching human Pass/Fail labels.

To rebuild the summary after a resumed job without issuing calls:

```bash
python -m egolife_two_user_qa.answerability_verification_benchmark summarize-set \
  --config egolife_two_user_qa/hpc/answerability_verifier_qa_mcq_testset.json
```

Once this exact set has been exported from the manual reviewer, use the normal
gold-label `validate`, `run`, and `score` commands with that CSV. The existing
19-row manual export has zero exact `qa_id` matches to this file, and the
140-row export has only three, so neither is a valid full-set gold file.

### Ten-QA Gemini 3.5 Flash pilot

`hpc/answerability_verifier_pilot10_gemini35_high.json` defines a credit-capped
pilot from the 27 accepted cluster-frame QAs and their matching
`ablation_manual_review_cluster_frames.csv` export. The cohort is fixed by QA ID
and contains exactly five human Pass rows and five human Fail rows. The Pass
examples have manual answerability score A2; the Fail examples have A1, including
clean asker-alone leakage cases and mixed grounding/answerability failures. The
selected QA records are copied intact, while the manual F/E/A values are parsed
from `reviewer_notes` into structured `manual_judge_scores` fields in
`gold_labels.jsonl` and the scored predictions.

Gemini 3.5 Flash exposes `high` as its maximum valid thinking level through
OpenRouter, so the single arm sends `reasoning.effort="high"`. Each QA is called
exactly three logical times: asker video only, provider video only, and the
video pair. The plan is therefore 10 QAs, one arm, and 30 logical inference
calls. Client-side OpenRouter retries default to zero in this launcher to avoid
silently exceeding the pilot request budget.

On the cluster, copy the matching manual export once and submit:

```bash
cp /path/to/ablation_manual_review_cluster_frames.csv \
  egolife_two_user_qa/hpc/ablation_manual_review_cluster_frames.csv
export OPENROUTER_API_KEY="..."
sbatch egolife_two_user_qa/hpc/run_answerability_verifier_pilot10_cpu.sbatch
```

After the normal annotation/media/catalog preflight, the launcher reads
`benchmark_plan.json` and hard-fails unless it contains 10 QAs, 5 Pass, 5 Fail,
one Gemini 3.5 Flash/high arm, and 30 expected calls. Resume is enabled for
completed QAs, but a request failure is not retried automatically.

### Remaining-17 Gemini 3.5 Flash run

After the ten-QA pilot, `hpc/answerability_verifier_remaining17_gemini35_high.json`
selects the exact complement from the same 27 accepted cluster-frame QAs. The
cohort contains 17 unique QAs (14 manual Pass and 3 manual Fail), and all 17
have manual answerability score A2; the three overall Fail labels arise from
other review dimensions. The model, `high` reasoning effort, full-video media,
and three conditions per QA are unchanged from the pilot.

The launcher disables client retries and hard-fails unless the validated plan
contains one arm and exactly 51 logical calls:

```bash
export OPENROUTER_API_KEY="..."
sbatch egolife_two_user_qa/hpc/run_answerability_verifier_remaining17_cpu.sbatch
```

Results are written under
`outputs/answerability_verifier_remaining17_gemini35_high/`.

### Question-only balanced option-rotation follow-up

`question_only_option_rotation_benchmark.py` tests whether Gemini can recover
the declared answer without seeing either video and whether a preferred answer
letter explains asker-only successes. The frozen cohort is the 17 QAs marked
`Pass` in `ablation_manual_review_cluster_frames.updated.csv`.

Each question receives five independent text-only calls. The correct semantic
answer occupies A, B, C, D, and E exactly once. The implementation uses a
cyclic Latin-square rotation, so every distractor also occupies every answer
position exactly once; no rotation gets an independently randomized distractor
order. Prompts contain only the raw question, the five displayed choices, and
a JSON response-format instruction. No image path, video path, evidence claim,
user identity, rationale, or original answer letter is sent to the model.

The single arm is `google/gemini-3.5-flash` at `high` reasoning with greedy
decoding. The validated hard cap is 17 questions x 5 rotations = 85 billable
calls, with zero media inputs and zero automatic retries or JSON-repair calls.

```bash
export OPENROUTER_API_KEY="..."
sbatch egolife_two_user_qa/hpc/run_question_only_option_rotation_17_cpu.sbatch
```

The launcher validates the exact 17 manual-Pass IDs, materializes all 85
prompts before the first API call, and proves that every semantic option appears
once at each letter. `--resume` skips completed call keys after an infrastructure
failure without repeating them.

Results are written under
`outputs/question_only_option_rotation_17_gemini35_high/`:

- `summary.json`: overall accuracy, chosen-letter distribution, accuracy at
  each correct-answer position, and counts of 5/5 semantic solutions versus
  fixed-letter behavior.
- `per_question.csv` and `per_question.jsonl`: five-condition choice pattern,
  correct count, and classification for every QA.
- `position_summary.csv`: aggregate results when the correct answer is at each
  of A through E.
- `calls.jsonl`: raw model response and canonical/display option mappings for
  every call.
- `non_five_of_five.jsonl`: questions that were not answered correctly under
  all five rotations.
- `benchmark_plan.json`, `rotation_plan.jsonl`, and `prompts.jsonl`: the exact
  preflighted cohort and text-only intervention for audit.

To rebuild summaries without another model call:

```bash
python -m egolife_two_user_qa.question_only_option_rotation_benchmark summarize \
  --config egolife_two_user_qa/hpc/question_only_option_rotation_17_gemini35_high.json
```

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
