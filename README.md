# EgoLife 鍙岀敤鎴?QA Pilot

杩欎釜妯″潡鐢ㄤ簬浠?EgoLife 瑙嗛涓瀯閫?20 鏉?pilot 澶氶€夐銆傛瘡閬撻閮借姹傝嚦灏戜袱涓敤鎴风殑绗竴瑙嗚瑙嗛鍏卞悓鎻愪緵璇佹嵁锛屽崟涓敤鎴风殑瑙嗛涓嶈兘瀹屾暣鍥炵瓟銆?

榛樿妯″瀷鏄?`Qwen/Qwen3.6-27B`銆傛祦绋嬩笉浣跨敤 OpenRouter/Gemini 绛夊晢涓?API key銆俙HF_TOKEN` 鍙綔涓?Hugging Face 涓嬭浇鎴栭檺娴佽緟鍔╋紝涓嶄綔涓烘帹鐞?API key銆?

## 涓绘祦绋?

褰撳墠涓昏矾寰勬槸 video-first銆備篃灏辨槸璇达紝Qwen3-VL 鐩存帴鎺ユ敹瀵归綈鍚庣殑 EgoLife 鍘熷瑙嗛锛岃€屼笉鏄厛鎶婅棰戣浆鎴?caption/observation 鍐嶅嚭棰樸€備箣鍚庣敤 judger 鍜?answerability evaluation 杩囨护鎺夊崟鐢ㄦ埛鍙瓟銆佸悎骞惰棰戜篃绛斾笉鍑嗐€佹垨鑰呴棶棰樺彛鍚讳笉鑷劧鐨勯銆?

```text
EgoLife video + EyeGaze/EyeTracking tree
-> build_manifest
-> prepare_evidence: 鎸?day / time token 瀵归綈鑷冲皯涓や釜鐢ㄦ埛锛屽苟缂撳瓨瑙嗛/gaze
-> generate_video_qa_loop:
   -> generator: 鐩存帴鐪嬪鐢ㄦ埛瑙嗛锛岀敓鎴?commonality/difference MCQ
   -> judger: 瑙ｉ噴涓轰粈涔堥棶杩欎釜闂锛屽苟缁?generator 鍙嶉
   -> answerability eval: 鍒嗗埆娴嬭瘯鍗曠敤鎴疯棰戝拰鍚堝苟瑙嗛鑳藉惁绛旈
-> validate_outputs: 鍋氱‘瀹氭€х殑 schema/gate 妫€鏌?
```

`observe_clips` 鍜?`mine_candidates` 鍙繚鐣欎綔璋冭瘯杈呭姪锛屼笉浣滀负 pilot 涓昏矾寰勩€傛寮?QA 鐢熸垚銆乯udger銆乤nswerability evaluation 鍜屾渶缁?review 閮藉湪 `generate_video_qa_loop` 鍐呭畬鎴愶紝閬垮厤鏃?prompt 鍜屽綋鍓?judge rubric 娣风敤銆?

### Opt-in ten-minute, memory-safe path

The original `prepare_evidence` default and the historically named
`hpc/qa/production/run_egolife_three_modes.sbatch` remain 30-second paths; that launcher is
now baseline-only. Ten-minute evidence is isolated in a separate launcher:

```bash
sbatch hpc/qa/production/run_egolife_three_modes_10min_memory_safe.sbatch
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

`hpc/qa/production/run_clip_pruned_benchmark_100.sbatch` prepares a 100-packet benchmark for
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
sbatch hpc/qa/preprocessing/run_pruning_ablation_30s.sbatch
```

The Slurm launcher requires `hpc/shared/cuda.py`, starts the CUDA keeper before
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
sbatch hpc/qa/preprocessing/run_pruning_k_grid_10.sbatch
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

`clip_gap_demo` 鏄竴涓嫭绔嬬殑棰勫鐞嗗疄楠岋紝涓嶄細鐩存帴鐢熸垚闂銆傚畠璇诲彇
`prepare_evidence` 浜х敓鐨勫弻鐢ㄦ埛 evidence packet锛屽鐭棰戠獥鍙ｉ噰鏍峰抚锛?
浣跨敤 CLIP embedding 鍋氱敤鎴峰唴鑱氱被锛屽啀瀵绘壘锛?

- 涓や釜鐢ㄦ埛涔嬮棿鐨?mutual-nearest shared anchors锛?
- Alice 涓壘涓嶅埌 Bob 杩戦偦鐨勯珮 novelty evidence gaps锛?
- Bob 涓壘涓嶅埌 Alice 杩戦偦鐨勯珮 novelty evidence gaps銆?

杩愯鍚庝細鍐欏嚭 JSON similarity results 鍜屼竴寮?contact sheet锛屽厛渚涗汉宸ユ鏌ワ紝
鍐嶅喅瀹氭槸鍚︽妸鍊欓€夌墖娈典氦缁欒嚜鐢卞害鏇撮珮鐨?VLM question generator銆傚畬鏁寸ず渚嬭
`CLIP_GAP_DEMO.md`銆?

## Gaze 鎶曞奖璇存槑

EgoLife EyeGaze CSV 涓嶆槸 EgoEverything 閲岀殑 image pixel gaze銆傚畠缁欑殑鏄?Project Aria CPF 鍧愭爣绯讳笅鐨?yaw/pitch/depth锛屼緥濡?`left_yaw_rads_cpf`銆乣right_yaw_rads_cpf`銆乣pitch_rads_cpf` 鍜?`depth_m`銆傛墍浠ヤ唬鐮佷笉鑳藉嚟绌烘瀯閫?`gaze_x/gaze_y`銆?

榛樿鎯呭喌涓?gaze summary 浼氭爣璁颁负锛?

```json
{"projection_status": "missing_calibration"}
```

濡傛灉鎯冲惎鐢?EgoEverything 閭ｇ 2D gaze point 鍒?object bbox center 鐨勮窛绂?Gaussian sampling锛岄渶瑕佷紶鍏?Aria RGB calibration锛?

```bash
python -m egolife_two_user_qa observe_clips \
  --manifest egolife_two_user_qa/outputs/pilot_20/manifest.json \
  --output egolife_two_user_qa/outputs/pilot_20/observations.jsonl \
  --aria-calibration-dir /path/to/aria_calibrations
```

鏇翠弗鏍肩殑 Aria 鎶曞奖闇€瑕佹彁渚?VRS/no-image VRS 鏂囦欢鎴?`online_calibration.jsonl`锛屽苟瀹夎 `projectaria-tools`銆備唬鐮佷細浼樺厛璧?Project Aria 鍘熺敓 `CameraCalibration.project()`銆侸SON calibration 涔熷彲浠ョ敤锛屼絾蹇呴』鏄惧紡鍖呭惈 RGB intrinsics 鍔?`T_camera_cpf`锛屾垨鑰?`T_device_camera` 鍜?`T_device_cpf`銆傚鏋滃彧浣跨敤鍏紑 EgoLife Hugging Face 鏂囦欢涓旀病鏈?calibration/VRS锛屾纭涓哄氨鏄繚鎸?2D projection unavailable锛屽彧浣跨敤瑙嗛甯у拰鏈姇褰辩殑 3D gaze 缁熻銆?

## 鏈湴 CPU Dry Run

dry run 鐢ㄦ潵楠岃瘉 Hugging Face manifest銆乪vidence packet銆乿ideo-first prompt 鍜?schema 宸ュ叿閾俱€傚畠涓嶄細鍔犺浇 Qwen3-VL锛屼篃涓嶄細鐪熺殑鐢熸垚楂樿川閲?QA銆?

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

姝ｅ紡鐢熸垚 20 鏉?QA 闇€瑕?GPU 鎴栨敮鎸佽棰戣緭鍏ョ殑鏈湴 VLM server銆?

```bash
bash scripts/run_qwen3vl_gpu.sh \
  --target-count 20 \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-new-tokens 1536
```

濡傛灉浣跨敤鏈湴 OpenAI-compatible server锛屾瘮濡?vLLM/SGLang/llama.cpp锛屽厛鍚姩 server锛岀劧鍚庤繍琛岋細

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

濡傛灉涓嶄紶 `--allow-openai-video-input`锛孫penAI-compatible backend 浼氶€€鍥?sampled frame images锛屽洜涓轰笉鏄瘡涓湰鍦?server 閮芥敮鎸?video data URL銆?

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
SAMPLING_TOP_K=40 sbatch hpc/qa/experiments/run_clip_pruned_sampling_neutral_pf_50.sbatch
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
sbatch hpc/qa/experiments/run_implicit_underrepresented_families_50.sbatch
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
sbatch hpc/qa/experiments/run_generator_rationale_ablation_qwen.sbatch \
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
launcher also starts `hpc/shared/cuda.py` before the ablation and keeps it running
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

## 杈撳嚭 Schema

`qa_mcq.jsonl` 姣忎竴琛屾槸涓€鏉?QA锛屽寘鍚細

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

鏈€缁?`review` 鐢?`generate_video_qa_loop` 鏍规嵁 judger銆乤nswerability evaluation 鍜?deterministic schema validation 鐢熸垚銆俿trict validation 瑕佹眰 `review.status == "passed"`銆乣review.review_passed == true`锛屽苟涓斾笅闈㈣繖浜?judger blocking checks 鍏ㄩ儴涓?`PASS`锛?

- `qa_formality`
- `evidence_groundedness`

`generation_trace` 淇濆瓨浜虹溂鏍告煡闇€瑕佺殑 intermediate data锛屽寘鎷?generation prompt/raw output銆乯udger prompt/raw output銆乺etry 鏃朵紶鍥?generator 鐨?feedback銆乤nswerability conditions锛屼互鍙婃瘡涓?condition 瀹為檯浣跨敤鐨勮棰戣矾寰勩€傚彧瑕佷紶鍏?`--intermediate-output`锛屽悓鏍风殑 trace 涔熶細鍗曠嫭鍐欐垚 JSONL锛屾柟渚垮悗缁汉宸ユ鏌ャ€?

杩愯涓ユ牸鏍￠獙锛?

```bash
python -m egolife_two_user_qa validate_outputs \
  --qa egolife_two_user_qa/outputs/pilot_20/qa_mcq.jsonl \
  --csv-output egolife_two_user_qa/outputs/pilot_20/qa_mcq.csv \
  --report egolife_two_user_qa/outputs/pilot_20/generation_report.md \
  --strict-review
```
