#!/usr/bin/env bash
#SBATCH --job-name=judge38-hm-split
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -Eeuo pipefail

PROJECT_ROOT=/scratch/$USER/Long-video-understanding-clip
PACKAGE_ROOT=$PROJECT_ROOT/egolife_two_user_qa/multi-user
TRAIN_ENV=/scratch/$USER/conda/envs/qwen38-vllm

DATASET_ROOT=/scratch/$USER/egolife_rlhf_evidence_v1
GENERATION_ROOT=/scratch/$USER/egolife_rlhf_qa_generation/qwen38_legacy_two_pass_schema_v2

SPLIT_ROOT=/scratch/$USER/egolife_judge_sft_labels_hm2991_only/small_split
OUTPUT_ROOT=/scratch/$USER/egolife_judge_sft/hm2991_small_split_manifests_${SLURM_JOB_ID}

PYTHON=$TRAIN_ENV/bin/python

export PYTHONPATH="$PACKAGE_ROOT:$PROJECT_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

mkdir -p \
  "$OUTPUT_ROOT/train" \
  "$OUTPUT_ROOT/heldout"

cd "$PACKAGE_ROOT"

TRAIN_LABELS="$SPLIT_ROOT/train_annotations.jsonl"
TEST_LABELS="$SPLIT_ROOT/test_annotations.jsonl"

# ------------------------------------------------------------
# Determine exact candidate + packet counts automatically.
# ------------------------------------------------------------
read TRAIN_N TRAIN_PACKETS < <(
"$PYTHON" - "$TRAIN_LABELS" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(x) for x in open(p) if x.strip()]
print(len(rows), len({x["evidence_id"] for x in rows}))
PY
)

read TEST_N TEST_PACKETS < <(
"$PYTHON" - "$TEST_LABELS" <<'PY'
import json, sys
p = sys.argv[1]
rows = [json.loads(x) for x in open(p) if x.strip()]
print(len(rows), len({x["evidence_id"] for x in rows}))
PY
)

echo "train_candidates=$TRAIN_N"
echo "train_packets=$TRAIN_PACKETS"
echo "heldout_candidates=$TEST_N"
echo "heldout_packets=$TEST_PACKETS"

# ------------------------------------------------------------
# Build TRAIN manifest
# ------------------------------------------------------------
"$PYTHON" -P -m training.judge_sft.prepare_real_data \
  --labels "$TRAIN_LABELS" \
  --generation-root "$GENERATION_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/train" \
  --expected-candidates "$TRAIN_N" \
  --expected-packets "$TRAIN_PACKETS" \
  --contradiction-policy exclude-task \
  > "$OUTPUT_ROOT/train/prepare_stdout.json"

# ------------------------------------------------------------
# Build HELD-OUT manifest
#
# prepare_real_data always calls its output train.jsonl.
# That's okay; semantically this directory is held-out.
# ------------------------------------------------------------
"$PYTHON" -P -m training.judge_sft.prepare_real_data \
  --labels "$TEST_LABELS" \
  --generation-root "$GENERATION_ROOT" \
  --dataset-root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_ROOT/heldout" \
  --expected-candidates "$TEST_N" \
  --expected-packets "$TEST_PACKETS" \
  --contradiction-policy exclude-task \
  > "$OUTPUT_ROOT/heldout/prepare_stdout.json"

# Convenience copy with a semantically correct name.
cp "$OUTPUT_ROOT/heldout/train.jsonl" "$OUTPUT_ROOT/test.jsonl"

# Record where this experiment lives.
cat > "$OUTPUT_ROOT/split_summary.json" <<JSON
{
  "train_candidates": $TRAIN_N,
  "heldout_candidates": $TEST_N,
  "train_packets": $TRAIN_PACKETS,
  "heldout_packets": $TEST_PACKETS,
  "train_manifest": "$OUTPUT_ROOT/train/train.jsonl",
  "heldout_manifest": "$OUTPUT_ROOT/test.jsonl"
}
JSON

echo
echo "========================================"
echo "PREPARATION COMPLETE"
echo "OUTPUT_ROOT=$OUTPUT_ROOT"
echo "TRAIN=$OUTPUT_ROOT/train/train.jsonl"
echo "TEST=$OUTPUT_ROOT/test.jsonl"
echo "========================================"
