# Score-only reviewer GRPO runtime

This directory connects the score-only ordinal reviewer to the maintained
EgoLife two-user generation pipeline. The frozen reviewer scores three ordered
1-3 fields—evidence grounding, answerability, and QA formality—and maps their
expected scores to a reward in `[0, 1]` with weights `0.4/0.4/0.2`.

The policy and reviewer intentionally receive different visual inputs:

- The policy receives only the surviving retained CLIP-cluster-member still
  frames from `clips[*].frames[*].path`. ms-swift consumes those paths through
  the row's `images` column and one `<image>` placeholder per frame.
- Frames are flattened in `required_users` order. Within each person, they stay
  in chronological `input_order_within_user`/`timestamp_seconds` order. The
  prompt identifies the packet-image range belonging to each person.
- `generator_image_paths` is an immutable copy used by the reward plugin to
  recheck the dataset-to-packet binding after ms-swift preprocessing.
- The frozen reviewer receives exactly two full synchronized MP4s through
  `reviewer_video_paths`, selected from `full_local_video`,
  `original_local_video`, or `source_local_video` in `required_users` order.
- A GRPO row containing a `videos` column, a `<video>` placeholder, a generator
  `local_video`, missing/empty frames, out-of-order frames, or mismatched paths
  fails validation before GPU work begins.

## Repository and HPC layout

On Torch/HPC, `hpc` is under the project root, not inside the Python package:

```bash
export QWEN3VL_PROJECT_ROOT=/scratch/${USER}/Long-video-understanding-clip
export PROJECT_ROOT=${QWEN3VL_PROJECT_ROOT}
export PACKAGE_ROOT=${PROJECT_ROOT}/egolife_two_user_qa
export HPC_ROOT=${PROJECT_ROOT}/hpc
```

Python modules live below `${PACKAGE_ROOT}/training/grpo_v3`. Slurm entrypoints
live below `${HPC_ROOT}/grpo_v3/score_only_reward`.

The reviewer checkpoint is a runtime artifact. Materialize the reviewed
nine-file checkpoint at the default location below, or set
`REVIEWER_CHECKPOINT` explicitly:

```text
${PACKAGE_ROOT}/checkpoints/score_only_reviewer/ckpt1
```

## Evidence input

Build GRPO data from the retained-frame sidecar produced by the verified reward
candidate collection, for example:

```text
${PACKAGE_ROOT}/outputs/reward_candidate_collection_qwen36_27b/<window>/evidence_build/evidence_cluster_member_frames.jsonl
```

Use the actual `<window>` from the selected collection run. Do not pass
`evidence_pruned_pairs.jsonl` directly: it still represents generator-side
pruned MP4s and has not yet materialized the exact surviving frame sequence.

Each packet must have:

- `generator_media_mode="retained_cluster_frames_only"` on the packet and both
  clips;
- exactly one clip for each of two distinct `required_users`;
- one or more readable, non-empty image files in each clip's `frames` list;
- no generator `local_video` or `generator_local_video`; and
- one readable, non-empty full MP4 per user through a reviewer alias.

## Build and validate the annotated 90-packet split

The maintained generator contract is neutral-only. Every train and evaluation
row uses `question_type="neutral"` and `generation_mode="baseline"`. The split
manifest and validator reject the archived question-type and generation-mode
paths.

Dataset construction always imports the live project-root
`egolife_two_user_qa.prompts.build_video_generation_prompt`; it has no alternate
or copied prompt-builder option. Each row records the prompt implementation
path/hash and rendered-prompt hash. Preflight recomputes them against the
current checkout and requires the production asker/provider attribution audit:
provider-only facts may supply the unknown answer detail, but must never be
written as something the asker saw, did, handled, or experienced. Any stale
prompt or perspective-mixing prompt fails before training.

The annotated bundle contains 90 evidence packets and six human-scored reviewer
candidates per packet. The candidate rows are reviewer supervision, not 540
generator prompts. GRPO regenerates from the evidence, so its fixed split is 60
train prompts, 10 validation prompts, and 20 held-out test prompts.

The checked-in Day 5-6 evidence manifests cover 70 of those IDs (48/9/13 by
split). The other 20 IDs are Day 1 (12/1/7). Dataset construction deliberately
fails until all named IDs are supplied; it never silently changes the split.

The available manifests were created under
`/scratch/hm2991/Long-video-understanding-clip`. If the same processed media was
copied below the current project, rebase that exact old project prefix to
`${PROJECT_ROOT}`. The builder rewrites only retained generator-frame paths and
full reviewer-video aliases, then requires every target file to exist.

Run from `${PROJECT_ROOT}` after restoring the Day 1 evidence manifest and all
media under the current project:

```bash
export CONDA_ENV_NAME=${TRAIN_ENV:-/scratch/${USER}/conda/envs/egoqa-ms-swift-v4.2.2-vllm024}
source "${HPC_ROOT}/env_qwen3vl.sh"
PYTHON=$(command -v python)

export DATA_DIR=${PACKAGE_ROOT}/outputs/grpo_score_only/annotated90_v3_data
export REVIEWER_SPLIT_AUDIT=${PACKAGE_ROOT}/score-only-reward-model/data_RLHF/reviewer_v1/annotation_audit_reshuffled_90_seed_20260809.json
export OLD_PROJECT_ROOT=/scratch/hm2991/Long-video-understanding-clip

"${PYTHON}" -m egolife_two_user_qa.training.grpo_v3.baseline.gate3_dataset \
  --evidence "${PACKAGE_ROOT}/outputs/RLHF_dataset/day5-7_6-12/packet_groups/packets_001_025/evidence_manifest.jsonl" \
  --evidence "${PACKAGE_ROOT}/outputs/RLHF_dataset/day5-7_6-12/packet_groups/packets_026_050/evidence_manifest.jsonl" \
  --evidence "${PACKAGE_ROOT}/outputs/RLHF_dataset/day5-7_12-18/packet_groups/packets_001_025/evidence_manifest.jsonl" \
  --evidence "${PACKAGE_ROOT}/outputs/RLHF_dataset/day5-7_12-18/packet_groups/packets_026_050/evidence_manifest.jsonl" \
  --evidence "<RESTORED_DAY1_RETAINED_FRAME_EVIDENCE_JSONL>" \
  --reviewer-split-audit "${REVIEWER_SPLIT_AUDIT}" \
  --source-project-root "${OLD_PROJECT_ROOT}" \
  --target-project-root "${PROJECT_ROOT}" \
  --output-dir "${DATA_DIR}" \
  --generation-mode baseline

"${PYTHON}" -m egolife_two_user_qa.training.grpo_v3.shared.validate_dataset \
  --dataset "${DATA_DIR}/gate3_v3_train_retained_frames.jsonl" \
  --expected-rows 60 --split train \
  --split-manifest "${DATA_DIR}/gate3_v3_split_manifest.json" \
  --output "${DATA_DIR}/train_dataset_contract_audit.json"

"${PYTHON}" -m egolife_two_user_qa.training.grpo_v3.shared.validate_dataset \
  --dataset "${DATA_DIR}/gate3_v3_eval_retained_frames.jsonl" \
  --expected-rows 10 --split eval \
  --split-manifest "${DATA_DIR}/gate3_v3_split_manifest.json" \
  --output "${DATA_DIR}/eval_dataset_contract_audit.json"

"${PYTHON}" -m egolife_two_user_qa.training.grpo_v3.shared.validate_dataset \
  --dataset "${DATA_DIR}/gate3_v3_test_retained_frames.jsonl" \
  --expected-rows 20 --split test \
  --split-manifest "${DATA_DIR}/gate3_v3_split_manifest.json" \
  --output "${DATA_DIR}/test_dataset_contract_audit.json"
```

All three validators must end with `DATASET_CONTRACT_PASSED`. They check row schema,
exact packet/media binding, frame order and placeholder count, all materialized
image/full-video paths, neutral/baseline mode, unique evidence IDs, manifest
counts and order, the production prompt source/content hashes and attribution
contract, and train/eval disjointness.

## Run order

Run the single-GPU reviewer probe before the two-GPU GRPO smoke test:

```bash
mkdir -p "${HPC_ROOT}/logs"
export DATASET=${DATA_DIR}/gate3_v3_train_retained_frames.jsonl
export VAL_DATASET=${DATA_DIR}/gate3_v3_eval_retained_frames.jsonl
export SPLIT_MANIFEST=${DATA_DIR}/gate3_v3_split_manifest.json

sbatch --export=ALL,DATASET="${DATASET}",SPLIT_MANIFEST="${SPLIT_MANIFEST}",EXPECTED_ROWS=60 \
  "${HPC_ROOT}/grpo_v3/score_only_reward/reviewer_probe.sbatch"

# Submit only after reviewer_probe_result.json reports status=passed.
sbatch --export=ALL,DATASET="${DATASET}",SPLIT_MANIFEST="${SPLIT_MANIFEST}",EXPECTED_ROWS=60,MAX_STEPS=1,EXPECTED_GROUPS=1,NUM_GENERATIONS=4 \
  "${HPC_ROOT}/grpo_v3/score_only_reward/grpo_smoke1.sbatch"

# Repeat only after G=4 passes. A pass includes at least 8 GiB policy-GPU headroom.
sbatch --export=ALL,DATASET="${DATASET}",SPLIT_MANIFEST="${SPLIT_MANIFEST}",EXPECTED_ROWS=60,MAX_STEPS=1,EXPECTED_GROUPS=1,NUM_GENERATIONS=8 \
  "${HPC_ROOT}/grpo_v3/score_only_reward/grpo_smoke1.sbatch"

# One-epoch pilot: 60 optimizer steps, 8 completions per training prompt,
# and 4 completions per validation prompt at the end of the epoch.
sbatch --export=ALL,DATASET="${DATASET}",VAL_DATASET="${VAL_DATASET}",SPLIT_MANIFEST="${SPLIT_MANIFEST}",EXPECTED_ROWS=60,VAL_EXPECTED_ROWS=10 \
  "${HPC_ROOT}/grpo_v3/score_only_reward/grpo_train.sbatch"
```

GPU 0 trains the policy; GPU 1 serves the frozen reviewer on loopback. The
smoke acceptance artifact is `smoke_result.json`. A pass requires the expected
reward rows, finite nonconstant rewards, at least two reviewer-scored
candidates, a completed optimizer step, nonzero finite gradients, and changed
LoRA-B tensors. It also records per-GPU peak memory at one-second resolution
and requires at least 8 GiB of policy-GPU headroom before calling a group size
comfortable.

The training launcher uses `beta=0.04`, `temperature=0.85`, `top_p=0.95`, and
`top_k=40`. The higher temperature is an intentional GRPO exploration setting;
it does not change the production prompt or retained-frame media pipeline.
With LoRA, ms-swift computes the reference log-probabilities by disabling the
policy adapter rather than loading a second copy of the 8B base model, but the
extra reference forward pass still costs time and working memory.

The validation JSONL is now passed as `--val_dataset`, with
`--eval_strategy epoch` and `--split_dataset_ratio 0`. Therefore validation
actually generates completions, calls the frozen reviewer, and records held-out
reward/KL metrics after each epoch. The 20-packet test split is not passed to
the trainer and remains reserved for post-training evaluation.
