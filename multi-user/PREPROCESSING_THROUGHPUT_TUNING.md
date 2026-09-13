# Six-user preprocessing throughput tuning

## Baseline evidence

The supplied `job-17426822-20260912_085414-freshprep-results.tgz` bundle was
verified against its SHA-256 sidecar:

```text
0fc4a8396fcec9682536c2ee3510a65a21b61885c4b3f89b6e57c4a4962b4f95
```

The optimized cold profile (`CLIP batch=128`, video sampling workers `3`,
media preparation workers `3`) completed useful preprocessing in 524 seconds.
Its main measured components were:

- source analysis: 453.805 worker-wall seconds across two source windows;
- direct RGB decode: 186.435 summed worker seconds across 12 videos;
- CLIP embedding: 112.160 seconds, including a 53.314-second first model load;
- media preparation: 136.715 summed window seconds, overlapped by double buffering;
- GPU utilization including the CUDA keeper: 45.26% mean, with 50.81% of samples
  below 10% utilization.

The timeline shows that each three-video decode wave reported roughly 13–19
seconds of ffmpeg work, but the next wave or embedding did not begin for roughly
another minute. The most likely unmeasured gap is lossless PNG persistence.
New instrumentation reports decode, PNG persistence, total per-video sampling,
aggregate sampling wall time, and CLIP embedding separately so the next result
can confirm this instead of relying on inference.

## Correct generator-media contract

CLIP clustering and pruning remain part of evidence analysis and candidate
selection. They do not alter generator inputs. The generator receives every
one-per-second sampled frame from each of the six complete, unpruned 600-second
videos, in per-user chronological order.

The contract is enforced while candidates are materialized, accepted by the
ten-minute setup, and consumed by the QA runtime. Each output packet records:

```text
policy = complete_full_unpruned_sampled_frames
generator_media_mode = full_unpruned_sampled_frames_only
analysis_pruning_applied_to_generator = false
```

The sweep summarizer rejects a profile if any packet loses a frame between its
full analysis sample set and model input set.

## Cold-cache sweep

The launcher brackets the sweep with two baseline replicates. All arms use a
separate cold media/frame/embedding cache, the same evidence seed and freshness
history, direct segment RGB, window double buffering, no assembled long MP4,
and no generator/judge phase.

| Profile | CLIP batch | Video workers | Media workers | Purpose |
|---|---:|---:|---:|---|
| `baseline_a_b128_w3_m3` | 128 | 3 | 3 | opening baseline |
| `sample6_b128_w6_m3` | 128 | 6 | 3 | isolate one-wave six-video sampling |
| `batch256_b256_w6_m3` | 256 | 6 | 3 | isolate larger H200 CLIP batches |
| `sample6_media6_b256_w6_m6` | 256 | 6 | 6 | test full six-user media concurrency |
| `candidate_repeat_b256_w6_m6` | 256 | 6 | 6 | repeat likely best profile |
| `baseline_b_b128_w3_m3` | 128 | 3 | 3 | closing baseline/order check |

Run from the cluster project root:

```bash
sbatch hpc/qa/experiments/run_six_user_preprocessing_throughput_sweep_0p5.sbatch
```

The job writes `preprocessing_throughput_sweep.json`, containing per-profile
timings, bracketed-baseline speedups, output-equivalence checks, full-unpruned
generator contract checks, and GPU summaries. CUDA-keeper utilization remains
scheduler overhead and is not counted as useful preprocessing throughput.
