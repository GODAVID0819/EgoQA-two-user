# CLIP Anchor and Evidence-Gap Demo

This toy experiment extends the current evidence-packet pipeline without changing
question generation. It uses a short window from one aligned two-user packet:

For a ready-to-run Google Colab version, open the notebook below. It asks you
to upload a ZIP of the local `egolife_two_user_qa` directory; it does not clone
or pull from GitHub:

`notebooks/clip_anchor_evidence_gap_colab.ipynb`

The notebook pins Pillow to `>=10,<12` to remain compatible with Colab's
preinstalled Gradio packages.

```text
short video windows
-> CLIP frame embeddings
-> within-user cosine K-means grouping
-> representative frames
-> cross-user similarity
-> shared anchors + per-user evidence gaps
```

An evidence gap means that a sampled representative has no close visual match in
the other user's representatives. It is a retrieval hypothesis, not proof that the
other user never saw the object or event.

## 1. Prepare one short evidence packet

From the parent repository directory:

```bash
python -m egolife_two_user_qa build_manifest \
  --days DAY1 \
  --agents A1_JAKE,A2_ALICE \
  --max-per-agent-day 4 \
  --output egolife_two_user_qa/outputs/clip_gap_demo/manifest.json

python -m egolife_two_user_qa prepare_evidence \
  --manifest egolife_two_user_qa/outputs/clip_gap_demo/manifest.json \
  --output egolife_two_user_qa/outputs/clip_gap_demo/evidence.jsonl \
  --output-root egolife_two_user_qa/outputs/clip_gap_demo \
  --cache-dir egolife_two_user_qa/outputs/clip_gap_demo/hf_cache \
  --target-count 1 \
  --users-per-case 2 \
  --frames-per-clip 8
```

## 2. Run the CLIP comparison

Install a compatible PyTorch/Torchvision pair plus Transformers and Pillow.
The first run downloads `openai/clip-vit-base-patch32`.

```bash
python -m egolife_two_user_qa clip_gap_demo \
  --evidence egolife_two_user_qa/outputs/clip_gap_demo/evidence.jsonl \
  --output-dir egolife_two_user_qa/outputs/clip_gap_demo/mined \
  --duration-seconds 12 \
  --sample-interval-seconds 1.5 \
  --clusters-per-user 4 \
  --anchor-threshold 0.75 \
  --top-k 3
```

This resamples the first 12 seconds of each local video and requires `ffmpeg`.
To use the frames already saved by `prepare_evidence`, add:

```bash
--use-existing-frames
```

## Outputs

- `clip_gap_results.json`: clustering groups, pairwise similarities, anchors,
  and per-user novelty-ranked gaps.
- `clip_gap_contact_sheet.jpg`: side-by-side visual inspection sheet.

For five reproducible random windows, add:

```bash
--random-trials 5 --random-seed 42
```

This writes one trial directory per window and an aggregate
`random_trials_summary.json`. CLIP is loaded once and reused across all trials.

For the more useful diversity experiment, first prepare five packets with
`--random-seed 42 --stratify-by-day`, then run:

```bash
--diverse-packet-trials 5 --random-seed 42
```

This uses five different synchronized instances rather than five windows from
the same 30-second clip.

The next experiment should give the contact-sheet candidates to a lightly
prompted VLM and ask it to describe relationships before it writes any question.

The Colab notebook also includes a multi-window sweep. By default it evaluates
overlapping 12-second windows every 6 seconds, reuses one loaded CLIP model,
plots similarity/novelty trends, and ranks windows for manual review.
