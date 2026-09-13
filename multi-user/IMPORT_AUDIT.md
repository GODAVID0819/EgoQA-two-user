# Six-user branch snapshot and audit

## Provenance

- Upstream repository: <https://github.com/GODAVID0819/EgoQA-two-user>
- Upstream branch: `feature/six-user`
- Imported commit: `1106fe37372383ce4c203f6c0e96e50ddb35e78a`
- Commit subject: `fix: use available attention backend for six-user jobs`
- Imported on: 2026-08-20 (America/New_York)
- Import method: Git archive of the upstream commit. The upstream `.git` directory is intentionally not included.

The directory began as an unmodified snapshot of the tracked files at the commit above. It now includes local fixes made on 2026-08-21 for the two blocking contract defects, the requested full-speaker/pruned-provider media contract, six-user prompt guidance, and regression coverage. `.gitignore` has one appended vendoring block that re-includes the snapshot's tracked `docs/`, `tests/`, and `hpc/` files when nested under this repository; generated caches, outputs, secrets, and HPC logs remain ignored.

## What the current six-user path does

1. Candidate mining samples six synchronized videos from a manifest group.
2. It tries each of the six people as the speaker, in index order.
3. For each speaker, CLIP frame embeddings are clustered per video. Every speaker cluster is compared with every cluster in the other five videos.
4. Provider clusters at or above the similarity threshold are removed; the speaker video stays full. A successful candidate therefore contains one full speaker video and five similarity-pruned provider videos.
5. Generation consumes those six videos. Text-only formality review runs alongside full-video groundedness review and answerability review.
6. The six-user answerability gate makes two model calls: the full speaker video alone must produce a valid wrong A-E choice, and all six full videos must produce the declared correct choice.
7. The production Slurm runtime validates storage, CUDA, FFmpeg/decord/TorchCodec, exact packet count, all-pairs pruning diagnostics, packet media, prompt traces, accepted/rejected artifacts, and strict output validation.
8. The production wrapper requests exactly 100 evidence packets and permits up to three feedback-driven generation attempts per packet. QA acceptance may be partial, but candidate packet materialization and packet-level generation coverage may not.

The older two-anchor mixed-media implementation and the older 3-of-5 consensus implementation remain in the file as callable helpers, but the current `selected_count == 6` production path does not use them.

## Audit findings

### Resolved high: mined packets failed the generation-loop role contract

`materialize_six_user_consensus_candidate()` emits `speaker_reference_unpruned` and `provider_similarity_pruned`. `build_candidate_packet()` preserves those values in `media_roles`. However, `six_user_role_metadata()` and `validate_qa_item()` require `speaker_consensus_pruned` and `provider_consensus_pruned`.

Resolution: `six_user_role_metadata()` and `validate_qa_item()` now require the roles emitted by the live sampler. The generation-loop regression fixture is built through `build_candidate_packet()`, so a future sampler-to-generator role drift fails one test rather than leaving two independently green test groups.

### Resolved high: the Slurm result validator looked up the wrong video-evidence key

`video_evidence_for_packet()` serializes the participant name under `user`. The embedded validator in `run_six_user_qa_runtime_probe.sbatch` searches for `agent_name`, so any accepted QA reaches `accepted row must identify exactly one speaker video` and fails the job's final validation.

Resolution: the final validator now looks up `video_evidence[].user`. Its contract test also rejects reintroduction of the incorrect `agent_name` lookup.

## Media-contract audit

- The live `selected_count == 6` path calls `materialize_six_user_consensus_candidate()` once for every successful speaker assignment.
- The ordered speaker clip is materialized with `keep_intervals=None`. Its `generator_local_video` and `full_local_video` are the same preserved full file, `generator_media_mode` is `full_video`, and `is_pruned` is false.
- Each of the five provider clips is materialized from its consensus keep intervals. Its generator and full paths are distinct, `generator_media_mode` is `pruned_video`, and `is_pruned` is true.
- Generation consumes the ordered full-speaker plus five-pruned-provider `local_video` paths.
- Groundedness review consumes all six full originals. Answerability makes exactly two calls: full speaker alone, then all six full originals.
- The Slurm candidate validator and unit tests now check path identity/difference, media modes, pruning flags, and role order rather than relying only on role labels.

## Cluster-pruning audit

- The live six-user path calls `clustered_speaker_provider_all_pairs_pruning()`; it does not call the older argmax-consensus helper.
- The two-user `topk`, mean-sim, pair-count, shared-anchor, and pruning-protection options are not part of the active six-user decision path. The six-user branch returns before `score_video_pairs()` is reached. They have therefore been removed from the active launcher and from six-user summary settings rather than being displayed as misleading active configuration.
- `high_similarity_interval_threshold` remains active and necessary: it is the provider-cluster deletion threshold, not a pair-selection threshold. `pruning_clusters_per_video` and `min_pruned_video_seconds` likewise remain active.
- Each provider receives a complete similarity matrix with shape `asker_cluster_count × provider_cluster_count`. Matrix dimensions are checked before pruning.
- The expected comparison count is `asker_cluster_count * sum(provider_cluster_counts)` across all five providers. Both expected and observed counts are serialized into every packet.
- Every provider cluster with at least one asker-cluster similarity at or above the configured threshold becomes one pruning event. All above-threshold asker matches are retained in that event's diagnostics, including non-argmax matches.
- Each event deletes only its provider cluster. The asker's marked clusters and removal intervals remain empty, and materialization ignores any asker keep intervals by using the full copied video.
- The Slurm packet validator rechecks the method, comparison scope, all five provider indexes, comparison count, threshold events, provider deletion targets, and empty asker pruning state before model generation starts.

## Six-user prompt audit

- The production generator is `build_video_generation_prompt()`. Its six-user branch explicitly says to ask a question the speaker would naturally have and genuinely want to ask, while the speaker's video alone remains insufficient.
- The same branch states the actual input contract: one full unpruned speaker reference and five similarity-pruned provider videos. It requires the speaker view to ground the reason to ask and one or more provider views to supply the missing answer-bearing fact, continuation, handoff, state, identity, or endpoint.
- The prompt does not require all five providers to contribute. Unused providers are valid, and metadata defaults no longer preclassify every provider as sufficient.
- A detailed structural example follows a marked red tape dispenser across two handoffs and into the top drawer of a blue rolling cart. The example includes the asker's practical reason for asking, five parallel location options, and an explicit warning not to copy the people, object, places, wording, answer, or option order.
- The text-only formality judge rejects contrived third-party quizzes that give the speaker no plausible reason to ask. The full-video groundedness judge verifies the speaker-side experience and provider-side answer evidence without requiring every provider. The two answerability prompts explicitly describe the full-speaker-only and all-six-full-video conditions.
- `build_relation_discovery_prompt()` and `build_relation_mcq_prompt()` still contain two-user wording, but both are explicitly archived and are not called by the production generation loop. The `why_two_users_needed` JSON key remains only as a legacy schema field; its six-user instructions and fallback value describe one speaker plus one or more providers.

### Partially resolved medium: legacy architecture remains callable

The active Torch runbook now describes the current full-asker/five-pruned-provider all-pairs design and the 100-packet production job. Older two-anchor and 3-of-5 helpers remain callable in `group_relative_clip_sampling.py` for historical reproduction, but the live `selected_count == 6` branch returns before those paths.

### Resolved medium: the targeted tests did not cover the two real handoffs

The focused suite now feeds an actual `build_candidate_packet()` result into `complete_generator_metadata()` and schema validation. It also directly materializes a consensus candidate and proves that the speaker generator path is the full path while all five provider generator paths are pruned. The Slurm test remains a static job-contract test; actual cluster execution is still required to exercise the embedded validator end to end.

### Resolved low: candidate mining could exceed `target_count`

The six-user append loop now checks the target before each speaker candidate. A regression test presents six successful candidates from one group with a target of four and verifies that the JSONL and summary both contain exactly four. The Slurm validator independently requires the JSONL row count to equal `EVIDENCE_TARGET`; the production wrapper sets that value to 100.

## Slurm identity and 100-packet audit

- All former account-specific login, scratch, environment, model, SFTP, and archived preflight references under `multi-user/` now use `hm2991`.
- The Torch launchers now model the deployed repository topology explicitly: `hpc/` is read from the `Long-video-understanding-clip` project root, while the six-user package is read from `egolife_two_user_qa/multi-user` beneath that root.
- The local deployment source for the active launchers is likewise the project-level `egolife_two_user_qa/hpc/qa/...`, not `multi-user/hpc/qa/...`. The nested copies are retained as package mirrors, and a contract test requires both pairs to remain byte-for-byte equivalent so edits cannot silently target only the non-deployed copy again.
- `#SBATCH --account=torch_pr_674_tandon_advanced` is intentionally unchanged because it names a Slurm project allocation, not the login user. The runbook requires a live `sacctmgr` association check for `hm2991` before submission.
- The active launchers do not pin `partition` or `qos`; this deployment lets Slurm select them from the requested account, one-GPU GRES, and H100 constraint. The runbook goes directly to the 100-packet submission without a separate probe or `sbatch --test-only` job.
- `hpc/qa/production/run_six_user_qa_packets_100.sbatch` fixes `EVIDENCE_TARGET=100`, `ACCEPTED_TARGET=100`, `MAX_GROUPS=800`, `MAX_ATTEMPTS=3`, and `ALLOW_PARTIAL=1`.
- `ALLOW_PARTIAL` applies only to accepted QA count. Missing or extra candidate packets fail before generation. The final validator requires every unique candidate `evidence_id` to enter generation, permits only the contiguous attempt sequence `1..N` with `N <= 3`, requires accepted packets to stop on their final recorded attempt, and requires rejected packets to exhaust all three attempts.
- The active six-user runtime starts the project-root `hpc/cuda.py` keeper before storage/manifest work, stops it through an EXIT trap, emits timestamped stage transitions, and persists the last running/failed/cancelled/completed stage in `stage_status.json` for postmortem diagnosis.
- The FFmpeg runtime is resolved after conda activation from, in order, an explicitly configured `FFMPEG_ENV`, the training environment, or the activated `PATH`. The resolved `ffmpeg` and `ffprobe` paths and their source are logged, and mining receives the resolved FFmpeg executable explicitly; a missing account-specific standalone FFmpeg environment is no longer an unconditional blocker.
- Every Python call in the active runtime uses the absolute `${TRAIN_ENV}/bin/python` interpreter. Immediately after conda activation, a dedicated stage verifies both `sys.prefix` and `CONDA_PREFIX` against the configured training environment and logs the actual executable and prefixes, preventing the login shell or an inherited `PATH` from silently selecting another environment.
- Python is now invoked with safe-path mode (`-P`) because the project working directory also contains the legacy two-user `egolife_two_user_qa` package. A pre-mining import check resolves both the package and `qwen3vl_runner.py` and requires their directories to equal the deployed `multi-user` package root; this prevents the legacy package from silently winning import precedence.
- The six-user memory-safe runner now reads `QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS`, forwards it through the base runner, and serializes it into every video item. The production values are `min_video_pixels=3136` and `max_image_pixels=65536`, preventing qwen-vl-utils from selecting a default video floor above the configured maximum.
- The imported `/scratch/<upstream-user>/models/Qwen3.6-27B` assumption was not valid for hm2991 and has been removed rather than mechanically renaming its owner. The runtime now follows the existing project-level sbatches: `MODEL_ID=Qwen/Qwen3.6-27B`, persistent Hugging Face cache `/scratch/${USER}/hf_cache`, and an early `AutoConfig.from_pretrained` model-resolution stage before candidate mining.

## Verification

- Imported tracked files through `git archive`, excluding only upstream Git metadata.
- Verified every imported upstream path against its Git blob after applying the repository's line-ending filters; only the documented `.gitignore` vendoring block differs.
- Reproduced both blocking defects before patching them.
- Added focused unit and job-contract coverage for the corrected roles, video-evidence key, media routing, and six-user prompt text.
- Ran the original 48-test focused six-user suite after the contract fixes. After removing the inactive pair-selection configuration, reran the 27 directly impacted sampler and Slurm-contract tests; all passed, including in-memory compilation of every embedded Python heredoc in the runtime worker.
- Ran `bash -n` against the runtime probe, retained 40-row pilot wrapper, and new 100-packet production wrapper; all passed.
- Broad discovery from this nested vendored folder is not a clean Windows-local signal: it imports the snapshot under a different directory name than its expected `egolife_two_user_qa` package and many unrelated training tests write to the sandbox-blocked system temp directory. The focused suite installs the intended package alias and writes only inside the workspace.
- Did not run GPU/model, FFmpeg media-materialization, network API, or Slurm execution tests locally.
