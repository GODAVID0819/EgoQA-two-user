#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=160G
#SBATCH --time=08:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

RUN_MODE="${RUN_MODE:-six_user_qa_runtime_probe}"
ACCEPTED_TARGET="${ACCEPTED_TARGET:-1}"
EVIDENCE_TARGET="${EVIDENCE_TARGET:-8}"
MAX_GROUPS="${MAX_GROUPS:-80}"

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-grpo-clean}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/outputs/six_user_qa}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/xl6775/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/xl6775/envs/egoqa-ffmpeg-runtime}"
MODEL_DIR="${MODEL_DIR:-/scratch/xl6775/models/Qwen3.6-27B}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/${RUN_MODE}_${SLURM_JOB_ID}"
OUTDIR="${OUTPUT_ROOT}/${RUN_MODE}_${SLURM_JOB_ID}"
CACHE_DIR="${JOB_SCRATCH_ROOT}/media_cache"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"

export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${JOB_SCRATCH_ROOT}/hf"
export HF_DATASETS_CACHE="${JOB_SCRATCH_ROOT}/hf_datasets"
export MODELSCOPE_CACHE="${JOB_SCRATCH_ROOT}/modelscope"
export TORCH_HOME="${JOB_SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${JOB_SCRATCH_ROOT}/triton"
export TORCHINDUCTOR_CACHE_DIR="${JOB_SCRATCH_ROOT}/torchinductor"
export VLLM_CACHE_ROOT="${JOB_SCRATCH_ROOT}/vllm"
export CUDA_CACHE_PATH="${JOB_SCRATCH_ROOT}/cuda"
export FLASHINFER_WORKSPACE_BASE="${JOB_SCRATCH_ROOT}/flashinfer"
export TMPDIR="${JOB_SCRATCH_ROOT}/tmp"
export TMP="${JOB_SCRATCH_ROOT}/tmp"
export TEMP="${JOB_SCRATCH_ROOT}/tmp"
export VLLM_NO_USAGE_STATS=1
export TOKENIZERS_PARALLELISM=false
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
export CONDA_PKGS_DIRS="${JOB_SCRATCH_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export QWEN_MEMORY_SAFE_VIDEO_FPS="1.0"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="32768"
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB="5"
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="32"
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="flash_attention_2"
export QWEN_MEMORY_SAFE_TRANSCODE_LOCAL_VIDEOS="1"
export QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE="512"
export QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR="${JOB_SCRATCH_ROOT}/qwen_video_cache"

mkdir -p \
  "${OUTDIR}" \
  "${CACHE_DIR}" \
  "${PYTHON_ALIAS_ROOT}" \
  "${HOME}" \
  "${XDG_CACHE_HOME}" \
  "${HF_HOME}" \
  "${HF_DATASETS_CACHE}" \
  "${MODELSCOPE_CACHE}" \
  "${TORCH_HOME}" \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${CUDA_CACHE_PATH}" \
  "${FLASHINFER_WORKSPACE_BASE}" \
  "${TMPDIR}" \
  "${CONDA_PKGS_DIRS}" \
  "${PIP_CACHE_DIR}"

cd "${PROJECT_ROOT}"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${TRAIN_ENV}"
PYTHON="${TRAIN_ENV}/bin/python"
ln -s "${PROJECT_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"

test -x "${PYTHON}"
test -x "${FFMPEG_ENV}/bin/ffmpeg"
test -x "${FFMPEG_ENV}/bin/ffprobe"
test -s "${MODEL_DIR}/config.json"
python -m training.torch_storage_preflight \
  --allowed-root "${JOB_SCRATCH_ROOT}" \
  --output "${OUTDIR}/storage_preflight.json"
python -m pip check
python -m egolife_two_user_qa --help >/dev/null
"${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 1
"${FFMPEG_ENV}/bin/ffprobe" -version | head -n 1
python - <<'PY'
import torch
from torchcodec.decoders import VideoDecoder

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"video_decoder={VideoDecoder.__module__}")
PY

BRANCH_NAME="$(git branch --show-current 2>/dev/null || true)"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  DIRTY_STATE="dirty"
else
  DIRTY_STATE="clean"
fi
python - "${OUTDIR}/job_manifest.json" <<PY
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "status": "running",
    "job_id": os.environ["SLURM_JOB_ID"],
    "run_mode": "${RUN_MODE}",
    "branch": "${BRANCH_NAME}",
    "dirty_state": "${DIRTY_STATE}",
    "project_root": "${PROJECT_ROOT}",
    "output_dir": "${OUTDIR}",
    "job_scratch_root": "${JOB_SCRATCH_ROOT}",
    "model_dir": "${MODEL_DIR}",
    "accepted_target": int("${ACCEPTED_TARGET}"),
    "evidence_target": int("${EVIDENCE_TARGET}"),
    "selected_count": 6,
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

echo "stage=build_manifest"
python -m egolife_two_user_qa build_manifest \
  --output "${OUTDIR}/manifest.json" \
  --no-overlays

echo "stage=mine_six_user_candidates"
python "${PROJECT_ROOT}/group_relative_clip_sampling.py" \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/six_user_candidates.jsonl" \
  --output-dir "${OUTDIR}/candidate_assets" \
  --review-dir "${OUTDIR}/candidate_review" \
  --cache-dir "${CACHE_DIR}" \
  --model-id "${CLIP_MODEL_ID}" \
  --target-count "${EVIDENCE_TARGET}" \
  --max-groups "${MAX_GROUPS}" \
  --min-group-size 6 \
  --selected-count 6 \
  --duration-seconds 30 \
  --sample-interval-seconds 1 \
  --pairs-per-group 1 \
  --topk 3 \
  --min-topk-sim 0.75 \
  --min-mean-sim 0.25 \
  --max-mean-sim 0.70 \
  --high-similarity-interval-threshold 0.82 \
  --preserve-shared-anchor-seconds 4 \
  --min-pruned-video-seconds 8 \
  --pruning-protection-mode reject \
  --random-seed 42 \
  --download-media

python - "${OUTDIR}/six_user_candidates.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("no six-user candidate was materialized")
expected_roles = [
    "speaker_pruned",
    "anchor_provider_pruned",
    "anchor_provider_pruned",
    "additional_provider_full",
    "additional_provider_full",
    "additional_provider_full",
]
for row in rows:
    users = row.get("input_users")
    clips = row.get("clips")
    if row.get("candidate_type") != "six_user_two_anchor_mixed_media":
        raise SystemExit("candidate_type mismatch")
    if row.get("generator_media_mode") != "three_pruned_three_full_videos":
        raise SystemExit("generator media contract mismatch")
    if not isinstance(users, list) or len(users) != 6 or row.get("required_users") != users:
        raise SystemExit("ordered six-user contract mismatch")
    if not isinstance(clips, list) or [clip.get("media_role") for clip in clips] != expected_roles:
        raise SystemExit("six media roles are not ordered correctly")
    if len(row.get("selected_anchor_edges") or []) != 2:
        raise SystemExit("expected two selected anchor edges")
    if len(row.get("diagnostic_pair_edges") or []) != 15:
        raise SystemExit("expected all 15 diagnostic pair edges")
    for clip in clips:
        generator_path = Path(str(clip.get("generator_local_video") or ""))
        full_path = Path(str(clip.get("full_local_video") or ""))
        if not generator_path.is_file() or not full_path.is_file():
            raise SystemExit(f"missing candidate media for {clip.get('agent_name')}")
print(f"candidate_contract_passed rows={len(rows)}")
PY

echo "stage=generate_six_user_qa"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${OUTDIR}/six_user_candidates.jsonl" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${ACCEPTED_TARGET}" \
  --max-attempts 3 \
  --backend transformers-local-memory-safe \
  --model-id "${MODEL_DIR}" \
  --dtype bfloat16 \
  --max-image-pixels 65536 \
  --max-new-tokens 2048 \
  --disable-thinking \
  --generation-mode baseline \
  --judge-video-source full \
  --fixed-question-type-schedule \
  --question-types neutral

python - "${OUTDIR}" "${ACCEPTED_TARGET}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
accepted_target = int(sys.argv[2])

def read_rows(name):
    path = output_dir / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

accepted = read_rows("qa_mcq.jsonl")
rejected = read_rows("qa_mcq.rejected.jsonl")
prompts = read_rows("video_first_prompts.jsonl")
candidates = read_rows("six_user_candidates.jsonl")
generation = [row for row in prompts if row.get("stage") == "generation"]
groundedness = [row for row in prompts if row.get("stage") == "evidence_groundedness_judge"]
answerability = [row for row in prompts if row.get("stage") == "answerability"]

for row in generation:
    if len(row.get("video_paths") or []) != 6:
        raise SystemExit("generator_video_count must be 6")
for row in groundedness:
    if len(row.get("video_paths") or []) != 6 or row.get("media_role") != "full":
        raise SystemExit("groundedness_video_count must be 6 full videos")

by_qa = {}
for row in answerability:
    by_qa.setdefault(str(row.get("qa_id")), []).append(row)
for qa_id, rows in by_qa.items():
    if len(rows) != 2:
        raise SystemExit(f"answerability_call_count must be 2 for {qa_id}")
    counts = sorted(len(row.get("video_paths") or []) for row in rows)
    if counts != [1, 6]:
        raise SystemExit(f"answerability media counts must be 1 and 6 for {qa_id}: {counts}")

all_six_wrong_count = 0
answerability_attempt_count = 0
for final_row in [*accepted, *rejected]:
    for trace in final_row.get("generation_trace") or []:
        gate = ((trace.get("answerability") or {}).get("gate") or {})
        if gate:
            answerability_attempt_count += 1
        if gate.get("failure_label") == "all_six_wrong":
            all_six_wrong_count += 1

for row in accepted:
    gate = (((row.get("review") or {}).get("answerability") or {}).get("gate") or {})
    required = {
        "speaker_only_correct": False,
        "all_six_correct": True,
        "cross_view_gain": 1,
        "answerability_evaluated_condition_count": 2,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise SystemExit(f"accepted row has invalid {key}: {gate.get(key)!r}")

attempted_count = len(accepted) + len(rejected)
all_six_wrong_rate = (
    all_six_wrong_count / answerability_attempt_count
    if answerability_attempt_count
    else 0.0
)
status = "passed" if len(accepted) >= accepted_target else "failed"
result = {
    "status": status,
    "candidate_count": len(candidates),
    "attempted_count": attempted_count,
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
    "accepted_target": accepted_target,
    "generator_video_count": 6,
    "groundedness_video_count": 6,
    "answerability_call_count": 2,
    "speaker_only_correct": False if accepted else None,
    "all_six_correct": True if accepted else None,
    "cross_view_gain": 1 if accepted else None,
    "answerability_evaluated_condition_count": 2,
    "all_six_wrong_count": all_six_wrong_count,
    "all_six_wrong_rate": round(all_six_wrong_rate, 6),
    "all_six_wrong_interpretation": "neutral_failure_label_not_automatic_noise_attribution",
}
(output_dir / "six_user_qa_result.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
manifest_path = output_dir / "job_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["status"] = status
manifest["result_path"] = str(output_dir / "six_user_qa_result.json")
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False))
if status != "passed":
    raise SystemExit(
        f"accepted_count={len(accepted)} is below accepted_target={accepted_target}; "
        "inspect rejected and intermediate rows before another GPU submission"
    )
PY

python -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review

python -m egolife_two_user_qa materialize_review_videos \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --output-dir "${OUTDIR}/review_videos"

echo "SIX_USER_QA_RUNTIME_PROBE_PASSED output=${OUTDIR}"
