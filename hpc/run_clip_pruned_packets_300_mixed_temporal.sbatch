#!/usr/bin/env bash
#SBATCH --job-name=egolife_pkt300_t10
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${EGOLIFE2U_PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa}"
if [[ ! -d "${PACKAGE_ROOT}" ]]; then
  PACKAGE_ROOT="${PROJECT_ROOT}"
fi

OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-${PACKAGE_ROOT}/outputs/clip_pruned_packets_300_mixed_temporal}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

REGULAR_PACKET_COUNT="${REGULAR_PACKET_COUNT:-200}"
TEMPORAL_PACKET_COUNT="${TEMPORAL_PACKET_COUNT:-100}"
TOTAL_PACKET_COUNT=$((REGULAR_PACKET_COUNT + TEMPORAL_PACKET_COUNT))
REGULAR_MAX_GROUPS="${REGULAR_MAX_GROUPS:-3000}"
TEMPORAL_MAX_GROUPS="${TEMPORAL_MAX_GROUPS:-2000}"
REGULAR_RANDOM_SEED="${REGULAR_RANDOM_SEED:-42}"
TEMPORAL_RANDOM_SEED="${TEMPORAL_RANDOM_SEED:-20260721}"

CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
PRUNING_CLUSTERS_PER_VIDEO="${PRUNING_CLUSTERS_PER_VIDEO:-12}"
HIGH_SIMILARITY_THRESHOLD="${HIGH_SIMILARITY_THRESHOLD:-0.82}"
# The production API's max gate is inclusive. With one-FPS timestamps, 9.999999
# implements the requested strict timestamp-difference < 10-second rule.
TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS="${TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS:-9.999999}"
MIN_PRUNED_VIDEO_SECONDS="${MIN_PRUNED_VIDEO_SECONDS:-8}"
PRUNING_PROTECTION_MODE="${PRUNING_PROTECTION_MODE:-min_seconds}"
MIN_PRUNED_VIDEO_PERCENT="${MIN_PRUNED_VIDEO_PERCENT:-}"

CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PACKAGE_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_PID=""

ENV_SCRIPT="${PROJECT_ROOT}/hpc/env_qwen3vl.sh"
if [[ ! -s "${ENV_SCRIPT}" ]]; then
  ENV_SCRIPT="${PACKAGE_ROOT}/hpc/env_qwen3vl.sh"
fi

export CONDA_ROOT="${CONDA_ROOT:-/share/apps/anaconda3/2025.06}"
export CONDA_ENV_NAME="${CONDA_ENV_NAME:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    echo "stage=stop_cuda_keeper pid=${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "${PROJECT_ROOT}"
mkdir -p hpc/logs "${OUTDIR}" "${CACHE_DIR}" "/scratch/${USER}/hf_cache"

if [[ ! -s "${ENV_SCRIPT}" ]]; then
  echo "error=environment_script_missing path=${ENV_SCRIPT}" >&2
  exit 1
fi
source "${ENV_SCRIPT}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "date_start=$(date --iso-8601=seconds)"
echo "project_root=${PROJECT_ROOT}"
echo "package_root=${PACKAGE_ROOT}"
echo "outdir=${OUTDIR}"
echo "cache_dir=${CACHE_DIR}"
echo "packet_plan=regular:${REGULAR_PACKET_COUNT},temporal:${TEMPORAL_PACKET_COUNT},total:${TOTAL_PACKET_COUNT}"
echo "regular_max_groups=${REGULAR_MAX_GROUPS}"
echo "temporal_max_groups=${TEMPORAL_MAX_GROUPS}"
echo "pruning_clusters_per_video=${PRUNING_CLUSTERS_PER_VIDEO}"
echo "high_similarity_threshold=${HIGH_SIMILARITY_THRESHOLD}"
echo "temporal_max_timestamp_difference_seconds=${TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS}"
echo "temporal_pruning_rule=clip_similarity_at_least_${HIGH_SIMILARITY_THRESHOLD}_and_timestamp_difference_less_than_10s"
echo "pruning_protection_mode=${PRUNING_PROTECTION_MODE}"
echo "min_pruned_video_seconds=${MIN_PRUNED_VIDEO_SECONDS}"
echo "cuda_keeper=required threshold=${CUDA_KEEPER_THRESHOLD} gpus=${CUDA_KEEPER_GPUS} reserve_gib=${CUDA_KEEPER_RESERVE}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
python --version
nvidia-smi || true

if [[ ! -s "${CUDA_KEEPER_SCRIPT}" ]]; then
  echo "error=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}" >&2
  exit 1
fi

echo "stage=preflight_cuda_keeper_imports"
if ! python - <<'PY'
import importlib.util

missing = [name for name in ("torch", "pynvml", "psutil") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_cuda_keeper_dependencies=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_imports=ok")
PY
then
  if [[ "${CUDA_KEEPER_AUTO_INSTALL}" == "1" ]]; then
    echo "stage=install_cuda_keeper_dependencies"
    python -m pip install nvidia-ml-py3 psutil
  fi
fi

python - <<'PY'
import importlib.util

missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("cuda keeper dependencies remain missing: " + ",".join(missing))
print("cuda_keeper_required_imports=ok")
PY

echo "stage=start_cuda_keeper"
python "${CUDA_KEEPER_SCRIPT}" \
  --threshold "${CUDA_KEEPER_THRESHOLD}" \
  --gpus "${CUDA_KEEPER_GPUS}" \
  --reserve "${CUDA_KEEPER_RESERVE}" \
  > "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log" 2>&1 &
CUDA_KEEPER_PID=$!
sleep 2
if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
  echo "error=cuda_keeper_exited_early log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log" >&2
  wait "${CUDA_KEEPER_PID}" || true
  CUDA_KEEPER_PID=""
  exit 1
fi
echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"

echo "stage=build_manifest"
python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json" \
  --no-overlays

protection_args=(--pruning-protection-mode "${PRUNING_PROTECTION_MODE}")
if [[ -n "${MIN_PRUNED_VIDEO_PERCENT}" ]]; then
  protection_args+=(--min-pruned-video-percent "${MIN_PRUNED_VIDEO_PERCENT}")
fi

echo "stage=prepare_regular_packets count=${REGULAR_PACKET_COUNT}"
python -m egolife_two_user_qa prepare_clip_pruned_benchmark \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/regular_packets.jsonl" \
  --output-dir "${OUTDIR}/regular/clip_pruned_pairs" \
  --review-dir "${OUTDIR}/regular/manual_review" \
  --cache-dir "${CACHE_DIR}" \
  --model-id "${CLIP_MODEL_ID}" \
  --target-count "${REGULAR_PACKET_COUNT}" \
  --max-groups "${REGULAR_MAX_GROUPS}" \
  --min-group-size 2 \
  --selected-count 2 \
  --duration-seconds 30 \
  --sample-interval-seconds 1.0 \
  --start-seconds 0 \
  --pairs-per-group 1 \
  --topk 3 \
  --min-topk-sim 0.01 \
  --min-mean-sim 0.01 \
  --max-mean-sim 0.99 \
  --high-similarity-interval-threshold "${HIGH_SIMILARITY_THRESHOLD}" \
  --pruning-clusters-per-video "${PRUNING_CLUSTERS_PER_VIDEO}" \
  --preserve-shared-anchor-seconds 0 \
  --min-pruned-video-seconds "${MIN_PRUNED_VIDEO_SECONDS}" \
  "${protection_args[@]}" \
  --random-seed "${REGULAR_RANDOM_SEED}" \
  --download-media

echo "stage=build_disjoint_temporal_manifest"
python - "${OUTDIR}/manifest.json" "${OUTDIR}/regular_packets.jsonl" "${OUTDIR}/temporal_manifest.json" "${REGULAR_PACKET_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
regular_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
expected = int(sys.argv[4])

with regular_path.open("r", encoding="utf-8") as handle:
    regular = [json.loads(line) for line in handle if line.strip()]
if len(regular) != expected:
    raise SystemExit(f"expected {expected} regular packets, found {len(regular)}")

used_groups = {(row.get("day"), row.get("time_token")) for row in regular}
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
original_count = len(manifest.get("clips", []))
manifest["clips"] = [
    clip
    for clip in manifest.get("clips", [])
    if (clip.get("day"), clip.get("time_token")) not in used_groups
]
manifest.setdefault("summary", {})["clip_count"] = len(manifest["clips"])
manifest["summary"]["excluded_regular_packet_group_count"] = len(used_groups)
output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"regular_packet_count={len(regular)}")
print(f"excluded_group_count={len(used_groups)}")
print(f"temporal_manifest_clip_count={len(manifest['clips'])}/{original_count}")
PY

echo "stage=prepare_time_sensitive_packets count=${TEMPORAL_PACKET_COUNT}"
python -m egolife_two_user_qa prepare_clip_pruned_benchmark \
  --manifest "${OUTDIR}/temporal_manifest.json" \
  --output "${OUTDIR}/temporal_packets.jsonl" \
  --output-dir "${OUTDIR}/temporal_10s/clip_pruned_pairs" \
  --review-dir "${OUTDIR}/temporal_10s/manual_review" \
  --cache-dir "${CACHE_DIR}" \
  --model-id "${CLIP_MODEL_ID}" \
  --target-count "${TEMPORAL_PACKET_COUNT}" \
  --max-groups "${TEMPORAL_MAX_GROUPS}" \
  --min-group-size 2 \
  --selected-count 2 \
  --duration-seconds 30 \
  --sample-interval-seconds 1.0 \
  --start-seconds 0 \
  --pairs-per-group 1 \
  --topk 3 \
  --min-topk-sim 0.01 \
  --min-mean-sim 0.01 \
  --max-mean-sim 0.99 \
  --high-similarity-interval-threshold "${HIGH_SIMILARITY_THRESHOLD}" \
  --pruning-clusters-per-video "${PRUNING_CLUSTERS_PER_VIDEO}" \
  --max-pair-time-difference-seconds "${TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS}" \
  --preserve-shared-anchor-seconds 0 \
  --min-pruned-video-seconds "${MIN_PRUNED_VIDEO_SECONDS}" \
  "${protection_args[@]}" \
  --random-seed "${TEMPORAL_RANDOM_SEED}" \
  --download-media

echo "stage=combine_and_verify_packets"
python - \
  "${OUTDIR}/regular_packets.jsonl" \
  "${OUTDIR}/temporal_packets.jsonl" \
  "${OUTDIR}/evidence_pruned_pairs.jsonl" \
  "${OUTDIR}/packet_build_summary.json" \
  "${REGULAR_PACKET_COUNT}" \
  "${TEMPORAL_PACKET_COUNT}" \
  "${PRUNING_CLUSTERS_PER_VIDEO}" \
  "${TEMPORAL_MAX_TIMESTAMP_DIFFERENCE_SECONDS}" <<'PY'
import json
import math
import sys
from pathlib import Path

regular_path = Path(sys.argv[1])
temporal_path = Path(sys.argv[2])
combined_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])
expected_regular = int(sys.argv[5])
expected_temporal = int(sys.argv[6])
expected_k = int(sys.argv[7])
temporal_gate = float(sys.argv[8])

def load_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def selected_pruning(packet):
    return packet["group_relative_clip_similarity"]["selection"]["selected_pair"]["temporal_pruning"]

regular = load_jsonl(regular_path)
temporal = load_jsonl(temporal_path)
if len(regular) != expected_regular:
    raise SystemExit(f"expected {expected_regular} regular packets, found {len(regular)}")
if len(temporal) != expected_temporal:
    raise SystemExit(f"expected {expected_temporal} temporal packets, found {len(temporal)}")

regular_groups = {(row.get("day"), row.get("time_token")) for row in regular}
temporal_groups = {(row.get("day"), row.get("time_token")) for row in temporal}
overlap = regular_groups & temporal_groups
if overlap:
    raise SystemExit(f"regular/temporal timestamp groups overlap: {sorted(overlap)[:5]}")

combined = regular + temporal
evidence_ids = [str(row.get("evidence_id")) for row in combined]
if len(evidence_ids) != len(set(evidence_ids)):
    raise SystemExit("duplicate evidence_id values found in combined packet set")

for index, packet in enumerate(regular):
    pruning = selected_pruning(packet)
    if int(pruning["cluster_count"]) != expected_k:
        raise SystemExit(f"regular packet {index} did not use k={expected_k}")
    if pruning.get("max_pair_time_difference_seconds") is not None:
        raise SystemExit(f"regular packet {index} unexpectedly used temporal gating")

for index, packet in enumerate(temporal):
    pruning = selected_pruning(packet)
    if int(pruning["cluster_count"]) != expected_k:
        raise SystemExit(f"temporal packet {index} did not use k={expected_k}")
    observed_gate = pruning.get("max_pair_time_difference_seconds")
    if observed_gate is None or not math.isclose(float(observed_gate), temporal_gate):
        raise SystemExit(f"temporal packet {index} has gate={observed_gate}, expected {temporal_gate}")
    for pair in pruning.get("high_similarity_representative_pairs", []):
        difference = float(pair["timestamp_difference_seconds"])
        if difference > temporal_gate + 1e-9:
            raise SystemExit(
                f"temporal packet {index} contains pruned centroid pair with time difference {difference}"
            )

with combined_path.open("w", encoding="utf-8") as handle:
    for packet in combined:
        handle.write(json.dumps(packet, ensure_ascii=False) + "\n")

summary = {
    "packet_count": len(combined),
    "regular_packet_count": len(regular),
    "temporal_packet_count": len(temporal),
    "packet_order": [
        {"cohort": "regular_timestamp_agnostic", "start_index": 0, "end_index": len(regular) - 1},
        {
            "cohort": "clip_plus_temporal_gate",
            "start_index": len(regular),
            "end_index": len(combined) - 1,
        },
    ],
    "pruning_clusters_per_video": expected_k,
    "temporal_max_pair_time_difference_seconds": temporal_gate,
    "temporal_timestamp_difference_rule": "< 10 seconds",
    "regular_and_temporal_groups_are_disjoint": True,
    "regular_source": str(regular_path),
    "temporal_source": str(temporal_path),
    "combined_output": str(combined_path),
}
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"combined_packet_count={len(combined)}")
print(f"regular_packet_count={len(regular)}")
print(f"temporal_packet_count={len(temporal)}")
print(f"unique_evidence_id_count={len(set(evidence_ids))}")
PY

echo "outputs:"
echo "  combined_packets=${OUTDIR}/evidence_pruned_pairs.jsonl"
echo "  build_summary=${OUTDIR}/packet_build_summary.json"
echo "  regular_packets=${OUTDIR}/regular_packets.jsonl"
echo "  temporal_packets=${OUTDIR}/temporal_packets.jsonl"
echo "  regular_summary=${OUTDIR}/regular/clip_pruned_pairs/group_relative_clip_summary.json"
echo "  temporal_summary=${OUTDIR}/temporal_10s/clip_pruned_pairs/group_relative_clip_summary.json"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-manual}.log"
echo "date_done=$(date --iso-8601=seconds)"
