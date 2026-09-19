# Qwen3.8-27B binary judge training

This package trains the existing language-model vocabulary logits directly. It
adds no classification head and supervises no scalar score.

## Exact target and media contract

Every invocation ends at the fixed assistant prefix:

```text
{"verdict":"
```

`pass` and `fail` must each be one tokenizer token. If their next-token logits
are `z_pass` and `z_fail`, the binary margin is `z_pass - z_fail` and training
uses `BCEWithLogits(margin, target)`, with PASS=1. At inference, code compares
only these two logits. Free-generated text never determines the verdict.

The production generation run did not retain six monolithic training videos.
Its durable judge input is in
`/scratch/$USER/egolife_rlhf_evidence_v1/packets`: six ordered, packet-owned
timelines of exactly 300 JPEGs each, already sampled from 600 seconds at 0.5
FPS. Groundedness and all-six answerability see all 1,800 frames; speaker-only
answerability sees the speaker's 300 frames; formality is text-only.

The collator mirrors the production Qwen runner's dynamic per-image resolution
cap. With the defaults below, a 1,800-frame call is capped at 91,728 pixels per
image, while a 300-frame call remains at the configured 262,144-pixel cap:

```text
max_input_tokens=262144
target_fraction=0.85
text_token_reserve=8192
item_token_overhead=2
min_pixels=3136
configured_max_pixels=262144
```

Manifests are compact. A visual row stores `frame_packet`, ordered
`frame_user_indices`, and `frame_order`; the loader resolves and verifies all
300 frames per selected user from the relative paths recorded in `packet.json`.
The on-cluster packet layout is:

```text
/scratch/$USER/egolife_rlhf_evidence_v1/
└── packets/
    └── <packet-id>/
        ├── packet.json
        └── frames/
            ├── user_0_A1_JAKE/
            │   └── ...jpg
            ├── user_1_.../
            │   └── ...jpg
            └── ...
```

Frame paths inside `packet.json` are packet-relative (for example,
`frames/user_0_A1_JAKE/...jpg`); the training manifest references the packet
directory, not the `frames` directory directly.

## Human-label conversion

`prepare_real_data` joins each label by
`source_evidence_id::attempt_NN` to `qa_mcq.intermediate.jsonl`. This is
necessary because the 221 labeled candidates include rejected generation
attempts, not only rows in `qa_mcq.jsonl`. The join recovers the full QA and its
original schema errors, reconstructs current prompts, verifies packet/user/URL
provenance, and refuses missing or duplicate matches.

The preparation module resolves `prompts.py` from this exact `multi-user`
checkout and builds the full judge view directly from each persisted
`packet.json`. It does not import the legacy `rlhf_evidence_preprocessing.py`
pipeline, and every visual judge row references the complete unpruned sampled
frames rather than the generation-time provider-pruned view.
The annotation and packet provenance must contain the identical multiset of
source-video URLs, including duplicate counts; their serialized/display order
may differ because frame ordering comes from the packet's user and frame indices.

Each labeled candidate becomes up to four invocations:

- one formality PASS/FAIL row;
- one groundedness PASS/FAIL row;
- one speaker-only answerability row from `asker_only_answerable`;
- one all-six answerability row from `all_six_answerable`.

The aggregate `answerability_verdict` is checked against the gate
`not asker_only_answerable and all_six_answerable`, but it is not a training
target. If a human PASS contradicts a deterministic FAIL embedded in the
formality prompt, only that impossible task row is excluded and recorded in
`excluded_task_rows.jsonl`; the other three labels remain usable.

Every usable row from all 40 supplied packets is written to `train.jsonl`; no
internal validation manifest is created. `data_audit.json` records label and
generation hashes, full-training counts, exclusions, and the media contract.
`prompt_snapshots.json` contains one complete prompt per judge condition. A
future, separately labeled standalone validation/test collection will select
among the saved epoch checkpoints.

## Loss weighting and first run

Class imbalance is corrected independently per task with smoothed, capped,
mean-one inverse-frequency weights. Task sampling scales make the dataset-level
objective exactly:

```text
0.2 * mean(formality BCE)
+ 0.4 * mean(groundedness BCE)
+ 0.4 * mean(answerability BCE)
```

The first real run uses BF16 Qwen3.8-27B with language-side attention and MLP
LoRA on `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj`, rank 8, alpha 16, dropout 0.05. The launcher audits that every
requested family has trainable language-side LoRA parameters. The vision
encoder, merger/aligner, embeddings, LM head, and all base weights remain
frozen. Attention defaults to PyTorch native SDPA, avoiding an external
`flash-attn` build while still allowing H200 fused SDPA kernels. Set
`ATTN_IMPLEMENTATION=flash_attention_2` only in an environment with a compatible
`flash_attn` installation. The full launcher uses two H200s, fused AdamW, LR `2e-5`, betas
`(0.9, 0.95)`, weight decay `0.01`, 10% warmup, cosine decay, ten epochs,
microbatch one per GPU, gradient accumulation sixteen (effective batch 32),
gradient checkpointing, clipping at 1.0, and ZeRO-3. There is no in-training
evaluation or early stopping. Every epoch checkpoint is saved without a
retention cap, and `final_adapter` stores the last epoch for convenience.
The launcher preserves the 10% warmup across Transformers APIs: it uses
`warmup_ratio=0.1` where supported and the Transformers v5.2+
`warmup_steps=0.1` ratio form otherwise.

## What the fixed assistant prefix actually does

The prefix is part of the tokenized input context; it is not claimed as model
output. The collator first asks Qwen's chat template to render the user message
and open an assistant turn with thinking disabled. Qwen3.8 represents this with
a canonical closed empty thinking block. The collator then concatenates the
literal verdict prefix:

```text
<rendered user content><assistant marker><think>\n\n</think>\n\n{"verdict":"
```

The empty block is template control syntax, not generated reasoning. An open or
non-empty thinking block is rejected before tokenization.

Because a causal LM's logits at the final input position predict the next
token, those logits answer: "what token follows the opening verdict quote?"
Training reads only the `pass` and `fail` entries of that vocabulary vector and
applies BCE to their difference. It does not append the human verdict, run
generation, or supervise the remainder of the JSON.

Inference uses one continuous autoregressive generation. At generation step
one, a logits processor records the original vocabulary scores, compares only
`pass` and `fail`, and permits exactly the selected verdict token. Starting at
step two it becomes a no-op, so the same `model.generate` call continues with
the closing quote, later contract fields, closing brace, and EOS. There is no
second prompt and the model is not re-asked for its verdict.

For example, the single generated continuation after the code-supplied prefix
may be:

```text
pass","reason":null,"fix":null}
```

Together, the input prefix and generated continuation form:

```json
{"verdict":"pass","reason":null,"fix":null}
```

For FAIL, the same uninterrupted generation writes the concrete `reason` and
`fix`. The complete object is schema-validated, while the authoritative verdict
always remains the first constrained token. The adapter-reload smoke test now
exercises both the binary logit decision and complete continuous JSON
generation. Training itself still computes loss only at the verdict position;
the available human annotations contain verdict labels, not supervised
reason/fix targets.

## Cluster sequence

Put exactly these two completed exports in a cluster directory (the filenames
may retain their original suffixes):

```text
six_user_binary_labels_annotator-01-packets-001-020-v2_hm2991.jsonl
six_user_binary_labels_annotator-05-packets-081-100-v2_ac.jsonl
```

The default is `/scratch/$USER/egolife_judge_sft_labels`. Then run each gate in
order, substituting the exact manifest output path printed by the first job:

```bash
mkdir -p hpc/logs
sbatch hpc/judge_sft/prepare_real_manifests.sbatch

sbatch --export=ALL,MANIFEST_DIR=/scratch/$USER/egolife_judge_sft/manifests_40_packets_JOBID \
  hpc/judge_sft/runtime_smoke_qwen38_27b.sbatch

sbatch --export=ALL,MANIFEST_DIR=/scratch/$USER/egolife_judge_sft/manifests_40_packets_JOBID \
  hpc/judge_sft/train_one_step_qwen38_27b.sbatch

sbatch --export=ALL,MANIFEST_DIR=/scratch/$USER/egolife_judge_sft/manifests_40_packets_JOBID \
  hpc/judge_sft/train_real_40_packets_qwen38_27b.sbatch
```

The runtime gate processes a real 1,800-frame example without loading model
weights. The one-step gate proves one optimizer update, finite loss/gradient,
nonzero LoRA-B tensors, save/reload, binary-logit selection, and continuous
generation of a complete verdict-first JSON object. Only after both pass should
the ten-epoch job run. These gates establish data/runtime/training plumbing;
they do not establish held-out judge quality. The full job writes
`checkpoint_inventory.json` and fails if it cannot find a checkpoint for every
configured epoch.

Both GPU training launchers start the same utilization-aware CUDA keeper used
by the production question-generation job. It is enabled by default, watches
both allocated GPUs, uses the conservative generation defaults (including a
0.25 GiB maximum preallocation), writes `cuda_keeper.log` inside the job output,
and is stopped by the exit trap. The CPU manifest job and processor-only runtime
probe do not start it. The shared helper must exist at
`$project/hpc/shared/cuda_high_duty.py`, and the training environment must
provide `pynvml` and `psutil`. Set `ENABLE_CUDA_KEEPER=0` only when deliberately
running without the keeper.

All launchers default to the actual production locations:

```text
project:    /scratch/$USER/Long-video-understanding-clip
package:    $project/egolife_two_user_qa/multi-user
frames:     /scratch/$USER/egolife_rlhf_evidence_v1
generation: /scratch/$USER/egolife_rlhf_qa_generation/qwen38_legacy_two_pass_schema_v2
```

They default to the generation environment at
`/scratch/$USER/conda/envs/qwen38-vllm`. It must also contain PEFT, Accelerate,
DeepSpeed, `qwen-vl-utils`, and Safetensors. The launchers do
not install or upgrade packages; if those training dependencies live in a
separate environment, pass its exact directory as `TRAIN_ENV`.
If the shared Hugging Face cache contains more than one Qwen3.8-27B snapshot,
set `MODEL_PATH` to one exact snapshot directory.
