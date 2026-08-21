#!/usr/bin/env bash
#SBATCH --job-name=egoqa_6u_probe
#SBATCH --account=torch_pr_674_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err

set -euo pipefail

RUN_MODE="${RUN_MODE:-six_user_qa_runtime_probe}"
ACCEPTED_TARGET="${ACCEPTED_TARGET:-1}"
EVIDENCE_TARGET="${EVIDENCE_TARGET:-1}"
MAX_GROUPS="${MAX_GROUPS:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-1}"

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
export FORCE_QWENVL_VIDEO_READER="decord"
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
export CONDA_PKGS_DIRS="${JOB_SCRATCH_ROOT}/conda_pkgs"
export PIP_CACHE_DIR="${JOB_SCRATCH_ROOT}/pip_cache"
export PATH="${FFMPEG_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export QWEN_MEMORY_SAFE_VIDEO_FPS="1.0"
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="3136"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="32768"
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB="5"
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="32"
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="sdpa"
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
    "status": "preflight",
    "job_id": os.environ["SLURM_JOB_ID"],
    "run_mode": "${RUN_MODE}",
    "branch": "${BRANCH_NAME}",
    "dirty_state": "${DIRTY_STATE}",
    "project_root": "${PROJECT_ROOT}",
    "output_dir": "${OUTDIR}",
    "job_scratch_root": "${JOB_SCRATCH_ROOT}",
    "model_dir": "${MODEL_DIR}",
    "video_backend": "${FORCE_QWENVL_VIDEO_READER}",
    "min_video_pixels": int("${QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS}"),
    "max_image_pixels": 65536,
    "pip_check_status": "pending",
    "pip_check_path": "${OUTDIR}/pip_check.txt",
    "accepted_target": int("${ACCEPTED_TARGET}"),
    "evidence_target": int("${EVIDENCE_TARGET}"),
    "max_attempts": int("${MAX_ATTEMPTS}"),
    "allow_partial": bool(int("${ALLOW_PARTIAL}")),
    "selected_count": 6,
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

PIP_CHECK_STATUS="passed"
if python -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then
  cat "${OUTDIR}/pip_check.txt"
else
  cat "${OUTDIR}/pip_check.txt"
  UNEXPECTED_PIP_CHECK="$(
    grep -Ev '^decord [^[:space:]]+ is not supported on this platform$' \
      "${OUTDIR}/pip_check.txt" || true
  )"
  if [[ -n "${UNEXPECTED_PIP_CHECK}" ]]; then
    echo "STOP: pip check 发现 decord 平台元数据提示之外的依赖错误。" >&2
    printf '%s\n' "${UNEXPECTED_PIP_CHECK}" >&2
    exit 1
  fi
  PIP_CHECK_STATUS="known_decord_platform_metadata_warning"
  echo "WARNING: pip check 仅报告已通过真实视频解码验证的 decord 平台元数据提示。"
fi
python -m egolife_two_user_qa --help >/dev/null
"${FFMPEG_ENV}/bin/ffmpeg" -version | head -n 1
"${FFMPEG_ENV}/bin/ffprobe" -version | head -n 1
python - <<'PY'
import decord
import torch
from qwen_vl_utils.vision_process import get_video_reader_backend

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
video_backend = get_video_reader_backend()
if video_backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {video_backend}")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"decord={decord.__version__}")
print(f"video_backend={video_backend}")
PY

python - "${OUTDIR}/job_manifest.json" <<PY
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = "running"
payload["pip_check_status"] = "${PIP_CHECK_STATUS}"
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
    "speaker_reference_unpruned",
    "provider_similarity_pruned",
    "provider_similarity_pruned",
    "provider_similarity_pruned",
    "provider_similarity_pruned",
    "provider_similarity_pruned",
]
for row in rows:
    users = row.get("input_users")
    clips = row.get("clips")
    if row.get("candidate_type") != "six_user_speaker_consensus":
        raise SystemExit("candidate_type mismatch")
    if row.get("generator_media_mode") != "speaker_full_five_provider_pruned_videos":
        raise SystemExit("generator media contract mismatch")
    if not isinstance(users, list) or len(users) != 6 or row.get("required_users") != users:
        raise SystemExit("ordered six-user contract mismatch")
    if not isinstance(clips, list) or [clip.get("media_role") for clip in clips] != expected_roles:
        raise SystemExit("six media roles are not ordered correctly")
    if row.get("provider_users") != users[1:]:
        raise SystemExit("ordered provider contract mismatch")
    attempts = row.get("speaker_attempts") or []
    if [attempt.get("speaker_index") for attempt in attempts] != list(range(6)):
        raise SystemExit("speaker attempts must visit indexes 0 through 5 in order")
    for clip in clips:
        generator_path = Path(str(clip.get("generator_local_video") or ""))
        full_path = Path(str(clip.get("full_local_video") or ""))
        if not generator_path.is_file() or not full_path.is_file():
            raise SystemExit(f"missing candidate media for {clip.get('agent_name')}")
print(f"candidate_contract_passed rows={len(rows)}")
PY

python - "${OUTDIR}/six_user_candidates.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from qwen_vl_utils import process_vision_info

path = Path(sys.argv[1])
first_row = json.loads(next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
video_paths = [str(clip["generator_local_video"]) for clip in first_row["clips"]]
if len(video_paths) != 6 or not all(Path(item).is_file() for item in video_paths):
    raise SystemExit("six-user video preflight requires six materialized generator videos")

messages = [{
    "role": "user",
    "content": [
        *[
            {
                "type": "video",
                "video": item,
                "min_pixels": 3136,
                "max_pixels": 65536,
                "fps": 1.0,
            }
            for item in video_paths
        ],
        {"type": "text", "text": "six-user runtime preflight"},
    ],
}]
try:
    _, video_inputs, _ = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
except TypeError:
    try:
        _, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        _, video_inputs = process_vision_info(messages)
if video_inputs is None or len(video_inputs) != 6:
    raise SystemExit(f"Qwen video preflight decoded {0 if video_inputs is None else len(video_inputs)} videos")
print("six_user_video_preflight_passed video_count=6 backend=decord")
PY

echo "stage=generate_six_user_qa"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${OUTDIR}/six_user_candidates.jsonl" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${ACCEPTED_TARGET}" \
  --max-attempts "${MAX_ATTEMPTS}" \
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

python - "${OUTDIR}" "${ACCEPTED_TARGET}" "${ALLOW_PARTIAL}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
accepted_target = int(sys.argv[2])
allow_partial = bool(int(sys.argv[3]))

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
formality = [row for row in prompts if row.get("stage") == "qa_formality_judge"]
groundedness = [row for row in prompts if row.get("stage") == "evidence_groundedness_judge"]
answerability = [row for row in prompts if row.get("stage") == "answerability"]

for row in generation:
    if len(row.get("video_paths") or []) != 6:
        raise SystemExit("generator_video_count must be 6")
for row in groundedness:
    if len(row.get("video_paths") or []) != 6 or row.get("media_role") != "full":
        raise SystemExit("groundedness_video_count must be 6 full videos")

by_qa_attempt = {}
for row in answerability:
    key = (str(row.get("qa_id")), row.get("attempt"))
    by_qa_attempt.setdefault(key, []).append(row)

formality_by_qa = {
    str(row.get("qa_id"))
    for row in formality
}

accepted_answerability_call_counts = []
for row in accepted:
    qa_id = str(row.get("qa_id"))
    attempt = row.get("attempt_count")
    rows = by_qa_attempt.get((qa_id, attempt), [])
    accepted_answerability_call_counts.append(len(rows))
    if len(rows) != 2:
        raise SystemExit(
            f"accepted QA must have exactly 2 answerability calls: qa_id={qa_id} "
            f"attempt={attempt} count={len(rows)}"
        )
    answerability_rows = {
        str(item.get("condition_id") or "").split("::", 1)[0]: item
        for item in rows
    }
    speaker_answerability = answerability_rows.get("speaker_only")
    all_six_answerability = answerability_rows.get("combined_all_six_users")
    if speaker_answerability is None or all_six_answerability is None:
        raise SystemExit(f"accepted QA is missing an answerability condition: {qa_id}")
    speaker_user = str(row.get("speaker_user") or "")
    if not speaker_user:
        raise SystemExit(f"accepted row is missing speaker_user: {qa_id}")
    if speaker_answerability.get("condition_id") != f"speaker_only::{speaker_user}":
        raise SystemExit(f"answerability condition does not match speaker: {qa_id}")
    if speaker_answerability.get("media_role") != "full":
        raise SystemExit(f"speaker-only answerability must use full media: {qa_id}")
    if all_six_answerability.get("media_role") != "full":
        raise SystemExit(f"all-six answerability must use full media: {qa_id}")
    speaker_evidence = [
        item
        for item in (row.get("video_evidence") or [])
        if str(item.get("agent_name") or "") == speaker_user
    ]
    if len(speaker_evidence) != 1:
        raise SystemExit(f"accepted row must identify exactly one speaker video: {qa_id}")
    speaker_full_video = str(speaker_evidence[0].get("full_local_video") or "")
    if not speaker_full_video:
        raise SystemExit(f"accepted row speaker full video is missing: {qa_id}")
    if speaker_answerability.get("video_paths") != [speaker_full_video]:
        raise SystemExit(f"answerability video does not match speaker full video: {qa_id}")
    full_videos = [
        str(item.get("full_local_video") or "")
        for item in (row.get("video_evidence") or [])
    ]
    if len(full_videos) != 6 or not all(full_videos):
        raise SystemExit(f"accepted row must identify six full videos: {qa_id}")
    if all_six_answerability.get("video_paths") != full_videos:
        raise SystemExit(f"all-six answerability videos do not match full videos: {qa_id}")
    gate = (((row.get("review") or {}).get("answerability") or {}).get("gate") or {})
    required = {
        "speaker_only_correct": False,
        "all_six_correct": True,
        "answerability_evaluated_condition_count": 2,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise SystemExit(f"accepted row has invalid {key}: {gate.get(key)!r}")
    if str(row.get("qa_id")) not in formality_by_qa:
        raise SystemExit(f"accepted row is missing qa_formality judge prompt: {row.get('qa_id')}")
    formality_check = (
        ((((row.get("review") or {}).get("judger") or {}).get("checks") or {}).get(
            "qa_formality"
        ) or {})
    )
    if formality_check.get("status") != "PASS":
        raise SystemExit(f"accepted row has non-passing qa_formality gate: {row.get('qa_id')}")

attempted_count = len(accepted) + len(rejected)
completed_review_count = 0
for row in [*accepted, *rejected]:
    review = row.get("review") or {}
    checks = ((review.get("judger") or {}).get("checks") or {})
    evaluations = ((review.get("answerability") or {}).get("evaluations") or [])
    condition_types = {
        str(item.get("condition_type") or "")
        for item in evaluations
    }
    if (
        (checks.get("qa_formality") or {}).get("status") in {"PASS", "FAIL"}
        and (checks.get("evidence_groundedness") or {}).get("status") in {"PASS", "FAIL"}
        and (checks.get("answerability") or {}).get("status") in {"PASS", "FAIL"}
        and condition_types == {"speaker_only", "combined_all_six_users"}
    ):
        completed_review_count += 1
answerability_call_count = (
    accepted_answerability_call_counts[0]
    if accepted_answerability_call_counts
    else (2 if completed_review_count else 0)
)
if len(accepted) >= accepted_target:
    status = "passed"
elif allow_partial and completed_review_count > 0:
    status = "partial"
else:
    status = "failed"
result = {
    "status": status,
    "candidate_count": len(candidates),
    "attempted_count": attempted_count,
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
    "accepted_target": accepted_target,
    "target_reached": len(accepted) >= accepted_target,
    "allow_partial": allow_partial,
    "generator_video_count": 6,
    "groundedness_video_count": 6,
    "answerability_call_count": answerability_call_count,
    "runtime_completed_review_count": completed_review_count,
    "speaker_only_correct": False if accepted else None,
    "all_six_correct": True if accepted else None,
    "answerability_evaluated_condition_count": 2,
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
if status == "failed":
    raise SystemExit(
        f"accepted_count={len(accepted)} is below accepted_target={accepted_target}; "
        "inspect rejected and intermediate rows before another GPU submission"
    )
PY

if [[ -s "${OUTDIR}/qa_mcq.jsonl" ]]; then
  python -m egolife_two_user_qa validate_outputs \
    --qa "${OUTDIR}/qa_mcq.jsonl" \
    --csv-output "${OUTDIR}/qa_mcq.csv" \
    --human-review-output "${OUTDIR}/human_review_sheet.md" \
    --report "${OUTDIR}/generation_report.md" \
    --strict-review

  python -m egolife_two_user_qa materialize_review_videos \
    --qa "${OUTDIR}/qa_mcq.jsonl" \
    --output-dir "${OUTDIR}/review_videos"
else
  echo "No accepted QA rows; semantic rejections remain recorded for pilot analysis."
fi

echo "SIX_USER_QA_JOB_FINISHED run_mode=${RUN_MODE} result=${OUTDIR}/six_user_qa_result.json"
