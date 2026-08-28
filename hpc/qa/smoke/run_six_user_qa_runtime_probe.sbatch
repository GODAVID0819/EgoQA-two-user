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
EVIDENCE_DURATION_SECONDS="${EVIDENCE_DURATION_SECONDS:-30}"
PRUNING_BLOCK_SECONDS="${PRUNING_BLOCK_SECONDS:-30}"
PRUNING_MAX_CROSS_GAP_SECONDS="${PRUNING_MAX_CROSS_GAP_SECONDS:-10}"
QA_TIME_BUDGET_MODE="${QA_TIME_BUDGET_MODE:-0}"
RUNTIME_PROBE_ALLOW_ZERO_ACCEPTED="${RUNTIME_PROBE_ALLOW_ZERO_ACCEPTED:-0}"
QA_DEADLINE_EPOCH_SECONDS="${QA_DEADLINE_EPOCH_SECONDS:-0}"
MAX_GENERATION_SLOTS="${MAX_GENERATION_SLOTS:-}"
SINGLE_CANDIDATE_GROUP="${SINGLE_CANDIDATE_GROUP:-0}"
TARGET_GENERATION_GROUPS="${TARGET_GENERATION_GROUPS:-1}"
EXPECTED_QA_PER_GROUP="${EXPECTED_QA_PER_GROUP:-0}"
ONE_CANDIDATE_PER_GROUP="${ONE_CANDIDATE_PER_GROUP:-0}"
GENERATOR_DECODE_MODE="${GENERATOR_DECODE_MODE:-greedy}"
GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.7}"
GENERATOR_TOP_P="${GENERATOR_TOP_P:-0.9}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
FORMALITY_MAX_NEW_TOKENS="${FORMALITY_MAX_NEW_TOKENS:-2048}"
SIX_USER_TEN_MINUTE_REASONING_PROFILE="${SIX_USER_TEN_MINUTE_REASONING_PROFILE:-0}"
export PRUNING_MAX_CROSS_GAP_SECONDS MAX_NEW_TOKENS FORMALITY_MAX_NEW_TOKENS
export SIX_USER_TEN_MINUTE_REASONING_PROFILE
export EXPECTED_QA_PER_GROUP
export ONE_CANDIDATE_PER_GROUP

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

CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-0}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/shared/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_MAX_PREALLOC="${CUDA_KEEPER_MAX_PREALLOC:-2.5}"
CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-8192}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-0}"
CUDA_KEEPER_PID=""
TERMINATION_REQUESTED="0"

cleanup() {
  cleanup_status=$?
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    echo "stage=stop_cuda_keeper pid=${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
  return "${cleanup_status}"
}

mark_termination() {
  TERMINATION_REQUESTED="1"
  echo "stage=termination_requested signal=TERM"
}

trap cleanup EXIT
trap mark_termination INT TERM

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
export QWEN_MEMORY_SAFE_VIDEO_FPS="${QWEN_MEMORY_SAFE_VIDEO_FPS:-1.0}"
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="${QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS:-3136}"
export QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="${QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS:-65536}"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="${QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS:-32768}"
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
    "max_image_pixels": int("${QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS}"),
    "pip_check_status": "pending",
    "pip_check_path": "${OUTDIR}/pip_check.txt",
    "accepted_target": int("${ACCEPTED_TARGET}"),
    "evidence_target": int("${EVIDENCE_TARGET}"),
    "target_generation_groups": int("${TARGET_GENERATION_GROUPS}"),
    "expected_qa_per_generation_group": int("${EXPECTED_QA_PER_GROUP}"),
    "one_candidate_per_generation_group": bool(int("${ONE_CANDIDATE_PER_GROUP}")),
    "max_attempts": int("${MAX_ATTEMPTS}"),
    "allow_partial": bool(int("${ALLOW_PARTIAL}")),
    "evidence_duration_seconds": float("${EVIDENCE_DURATION_SECONDS}"),
    "pruning_max_cross_gap_seconds": float("${PRUNING_MAX_CROSS_GAP_SECONDS}"),
    "max_new_tokens": int("${MAX_NEW_TOKENS}"),
    "formality_max_new_tokens": int("${FORMALITY_MAX_NEW_TOKENS}"),
    "ten_minute_reasoning_profile": bool(int("${SIX_USER_TEN_MINUTE_REASONING_PROFILE}")),
    "runtime_probe_allow_zero_accepted": bool(int("${RUNTIME_PROBE_ALLOW_ZERO_ACCEPTED}")),
    "selected_count": 6,
    "generator_decode": {
        "mode": "${GENERATOR_DECODE_MODE}",
        "temperature": float("${GENERATOR_TEMPERATURE}"),
        "top_p": float("${GENERATOR_TOP_P}"),
    },
    "cuda_keeper": {
        "enabled": bool(int("${CUDA_KEEPER_ENABLE}")),
        "script": "${CUDA_KEEPER_SCRIPT}",
        "threshold": int("${CUDA_KEEPER_THRESHOLD}"),
        "gpus": "${CUDA_KEEPER_GPUS}",
        "reserve_gb": float("${CUDA_KEEPER_RESERVE}"),
        "max_prealloc_gb": float("${CUDA_KEEPER_MAX_PREALLOC}"),
        "start_used_mib": int("${CUDA_KEEPER_START_USED_MIB}"),
        "log_path": "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID}.log",
    },
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

if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  if [[ -s "${CUDA_KEEPER_SCRIPT}" ]]; then
    echo "stage=preflight_cuda_keeper_imports"
    CUDA_KEEPER_IMPORTS_OK="0"
    if "${PYTHON}" - <<'PY'
import importlib.util

missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_required=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_imports=ok")
PY
    then
      CUDA_KEEPER_IMPORTS_OK="1"
    elif [[ "${CUDA_KEEPER_AUTO_INSTALL}" == "1" ]]; then
      echo "stage=install_cuda_keeper_deps"
      "${PYTHON}" -m pip install nvidia-ml-py3 psutil || true
      if "${PYTHON}" - <<'PY'
import importlib.util

missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_required_after_install=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_imports=ok")
PY
      then
        CUDA_KEEPER_IMPORTS_OK="1"
      fi
    fi

    if [[ "${CUDA_KEEPER_IMPORTS_OK}" == "1" ]]; then
      echo "stage=start_cuda_keeper"
      "${PYTHON}" "${CUDA_KEEPER_SCRIPT}" \
        --threshold "${CUDA_KEEPER_THRESHOLD}" \
        --gpus "${CUDA_KEEPER_GPUS}" \
        --reserve "${CUDA_KEEPER_RESERVE}" \
        --max-prealloc "${CUDA_KEEPER_MAX_PREALLOC}" \
        --start-used-mib "${CUDA_KEEPER_START_USED_MIB}" \
        > "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log" 2>&1 &
      CUDA_KEEPER_PID=$!
      sleep 2
      if kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
        echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"
      else
        echo "WARNING: cuda_keeper_exited_early; continuing without keeper"
        CUDA_KEEPER_PID=""
      fi
    else
      echo "WARNING: cuda_keeper_imports_missing; continuing without keeper"
    fi
  else
    echo "WARNING: cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}; continuing without keeper"
  fi
fi

python - "${OUTDIR}/job_manifest.json" <<PY
import json
import os
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
CANDIDATE_MODE_ARGS=()
if [[ "${SINGLE_CANDIDATE_GROUP}" == "1" ]]; then
  CANDIDATE_MODE_ARGS+=(--single-candidate-group)
fi
python "${PROJECT_ROOT}/group_relative_clip_sampling.py" \
  --manifest "${OUTDIR}/manifest.json" \
  --output "${OUTDIR}/six_user_candidates.jsonl" \
  --output-dir "${OUTDIR}/candidate_assets" \
  --review-dir "${OUTDIR}/candidate_review" \
  --cache-dir "${CACHE_DIR}" \
  --model-id "${CLIP_MODEL_ID}" \
  --target-count "${EVIDENCE_TARGET}" \
  --target-generation-groups "${TARGET_GENERATION_GROUPS}" \
  --max-groups "${MAX_GROUPS}" \
  --min-group-size 6 \
  --selected-count 6 \
  --duration-seconds "${EVIDENCE_DURATION_SECONDS}" \
  --pruning-block-seconds "${PRUNING_BLOCK_SECONDS}" \
  --pruning-cross-gap-mode center \
  --pruning-max-cross-gap-seconds "${PRUNING_MAX_CROSS_GAP_SECONDS}" \
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
  --download-media \
  "${CANDIDATE_MODE_ARGS[@]}"

python - "${OUTDIR}/six_user_candidates.jsonl" "${TARGET_GENERATION_GROUPS}" "${ONE_CANDIDATE_PER_GROUP}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
target_generation_groups = int(sys.argv[2])
one_candidate_per_group = bool(int(sys.argv[3]))
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not rows:
    raise SystemExit("no six-user candidate was materialized")
all_generation_group_ids = sorted(
    {str(row.get("generation_group_id")) for row in rows if row.get("generation_group_id")}
)
if len(all_generation_group_ids) != target_generation_groups:
    raise SystemExit(
        "generation_group_count must equal target_generation_groups: "
        f"actual={len(all_generation_group_ids)} target={target_generation_groups} "
        f"group_ids={all_generation_group_ids}"
    )
if one_candidate_per_group:
    selected_by_group = {}
    for row in rows:
        group_id = str(row.get("generation_group_id") or "")
        selected_by_group.setdefault(group_id, row)
    rows = [selected_by_group[group_id] for group_id in all_generation_group_ids]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
generation_group_ids = sorted(
    {str(row.get("generation_group_id")) for row in rows if row.get("generation_group_id")}
)
generation_group_count = len(generation_group_ids)
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
group_paths = {}
for row in rows:
    group_id = str(row.get("generation_group_id") or "")
    group_paths.setdefault(
        group_id,
        tuple(str(clip.get("full_local_video") or "") for clip in row.get("clips") or []),
    )
if len(group_paths) != target_generation_groups:
    raise SystemExit(
        f"expected exactly {target_generation_groups} distinct generation groups, "
        f"got {sorted(group_paths)}"
    )
if len(set(group_paths.values())) != len(group_paths):
    raise SystemExit("generation groups must reference different six-user video sets")
print(
    f"candidate_contract_passed rows={len(rows)} "
    f"generation_groups={generation_group_count} group_ids={generation_group_ids} "
    f"one_candidate_per_group={one_candidate_per_group}"
)
PY

python - "${OUTDIR}/six_user_candidates.jsonl" "${QWEN_MEMORY_SAFE_VIDEO_FPS}" \
  "${QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR}" "${QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE}" <<'PY'
import json
import os
import subprocess
import sys
from pathlib import Path

from qwen_vl_utils import process_vision_info


def cached_memory_safe_video(source: str, cache_dir: Path, *, fps: float, max_edge: int, crf: int = 23) -> str:
    """Prepare exactly the same low-memory media used by the inference runner."""

    source_path = Path(source).resolve()
    stat = source_path.stat()
    cache_label = ".".join(
        (
            f"{stat.st_dev:x}",
            f"{stat.st_ino:x}",
            str(stat.st_size),
            str(stat.st_mtime_ns),
            f"fps-{fps:g}",
            f"edge-{max_edge}",
            f"crf-{crf}",
        )
    )
    output = cache_dir / f"{source_path.stem}.{cache_label}.mp4"
    if output.is_file() and output.stat().st_size > 0:
        return str(output)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_dir / f".{source_path.stem}.{cache_label}.{os.getpid()}.tmp.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-vf",
        ",".join(
            (
                f"fps={fps:g}",
                f"scale={max_edge}:{max_edge}:force_original_aspect_ratio=decrease",
                "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            )
        ),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temporary),
    ]
    print(
        "qwen_memory_safe_preflight_transcode "
        f"source={source_path} fps={fps:g} max_edge={max_edge}",
        flush=True,
    )
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not temporary.is_file() or temporary.stat().st_size <= 0:
            raise RuntimeError(f"ffmpeg did not create a usable preflight file: {temporary}")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return str(output)


path = Path(sys.argv[1])
video_fps = float(sys.argv[2])
cache_dir = Path(sys.argv[3])
max_edge = int(sys.argv[4])
first_row = json.loads(next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
source_paths = [str(clip["generator_local_video"]) for clip in first_row["clips"]]
if len(source_paths) != 6 or not all(Path(item).is_file() for item in source_paths):
    raise SystemExit("six-user video preflight requires six materialized generator videos")
preflight_video_paths = [
    cached_memory_safe_video(item, cache_dir, fps=video_fps, max_edge=max_edge)
    for item in source_paths
]
messages = [{
    "role": "user",
    "content": [
        *[
            {
                "type": "video",
                "video": str(output),
                "min_pixels": int(os.environ["QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS"]),
                "max_pixels": int(os.environ["QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS"]),
                "fps": video_fps,
            }
            for output in preflight_video_paths
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
print("six_user_video_preflight_passed video_count=6 backend=decord", flush=True)
PY

echo "stage=generate_six_user_qa"
QA_TIME_ARGS=()
QA_PROFILE_ARGS=(--disable-thinking)
if [[ "${SIX_USER_TEN_MINUTE_REASONING_PROFILE}" == "1" ]]; then
  QA_PROFILE_ARGS=(
    --six-user-ten-minute-reasoning-profile
    --formality-max-new-tokens "${FORMALITY_MAX_NEW_TOKENS}"
  )
fi
if [[ "${QA_TIME_BUDGET_MODE}" == "1" ]]; then
  QA_TIME_ARGS+=(
    --deadline-epoch-seconds "${QA_DEADLINE_EPOCH_SECONDS}"
    --repeat-evidence
    --attempts-output "${OUTDIR}/qa_mcq.attempts.jsonl"
  )
  if [[ -n "${MAX_GENERATION_SLOTS}" ]]; then
    QA_TIME_ARGS+=(--max-generation-slots "${MAX_GENERATION_SLOTS}")
  fi
fi
set +e
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
  --max-image-pixels "${QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generation-mode baseline \
  --generator-decode-mode "${GENERATOR_DECODE_MODE}" \
  --generator-temperature "${GENERATOR_TEMPERATURE}" \
  --generator-top-p "${GENERATOR_TOP_P}" \
  --judge-video-source full \
  --fixed-question-type-schedule \
  --question-types neutral \
  "${QA_PROFILE_ARGS[@]}" \
  "${QA_TIME_ARGS[@]}"
GENERATION_EXIT_CODE="$?"
set -e

if [[ "${GENERATION_EXIT_CODE}" -ne 0 && "${TERMINATION_REQUESTED}" != "1" ]]; then
  echo "STOP: QA generation failed exit_code=${GENERATION_EXIT_CODE}"
  exit "${GENERATION_EXIT_CODE}"
fi

python - "${OUTDIR}" "${ACCEPTED_TARGET}" "${ALLOW_PARTIAL}" "${MAX_ATTEMPTS}" \
  "${QA_TIME_BUDGET_MODE}" "${EVIDENCE_DURATION_SECONDS}" "${PRUNING_BLOCK_SECONDS}" \
  "${QA_DEADLINE_EPOCH_SECONDS}" "${MAX_GENERATION_SLOTS}" \
  "${TERMINATION_REQUESTED}" "${GENERATION_EXIT_CODE}" \
  "${RUNTIME_PROBE_ALLOW_ZERO_ACCEPTED}" \
  "${GENERATOR_DECODE_MODE}" "${GENERATOR_TEMPERATURE}" "${GENERATOR_TOP_P}" \
  "${EXPECTED_QA_PER_GROUP}" <<'PY'
import json
import sys
import time
from pathlib import Path

from egolife_two_user_qa.video_qa_loop import (
    prompt_rows_by_generation_identity,
    summarize_review_gate_attempts,
)

output_dir = Path(sys.argv[1])
accepted_target = int(sys.argv[2])
allow_partial = bool(int(sys.argv[3]))
max_attempts = int(sys.argv[4])
time_budget_mode = bool(int(sys.argv[5]))
evidence_duration_seconds = float(sys.argv[6])
if evidence_duration_seconds <= 0 or evidence_duration_seconds % 30 != 0:
    raise SystemExit(
        "evidence_duration_seconds must be a positive multiple of 30: "
        f"actual={evidence_duration_seconds}"
    )
expected_segments_per_user = int(evidence_duration_seconds / 30)
pruning_block_seconds = float(sys.argv[7])
deadline_epoch_seconds = float(sys.argv[8])
max_generation_slots = int(sys.argv[9]) if sys.argv[9] else None
termination_requested = bool(int(sys.argv[10]))
generation_exit_code = int(sys.argv[11])
runtime_probe_allow_zero_accepted = bool(int(sys.argv[12]))
generator_decode_mode = sys.argv[13]
generator_temperature = float(sys.argv[14])
generator_top_p = float(sys.argv[15])
expected_qa_per_group = int(sys.argv[16])

def read_rows(name):
    path = output_dir / name
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

accepted = read_rows("qa_mcq.jsonl")
rejected = read_rows("qa_mcq.rejected.jsonl")
prompts = read_rows("video_first_prompts.jsonl")
candidates = read_rows("six_user_candidates.jsonl")
intermediate = read_rows("qa_mcq.intermediate.jsonl")
attempt_rows = read_rows("qa_mcq.attempts.jsonl")
time_budget_partials = [
    row for row in intermediate if row.get("status") == "time_budget_partial"
]
generation = [row for row in prompts if row.get("stage") == "generation"]
formality = [row for row in prompts if row.get("stage") == "qa_formality_judge"]
groundedness = [row for row in prompts if row.get("stage") == "evidence_groundedness_judge"]
evidence_segments = [
    row for row in prompts if row.get("stage") == "evidence_segment_observation"
]
evidence_aggregations = [
    row for row in prompts if row.get("stage") == "evidence_groundedness_aggregation"
]
answerability = [row for row in prompts if row.get("stage") == "answerability"]

for row in generation:
    if len(row.get("video_paths") or []) != 6:
        raise SystemExit("generator_video_count must be 6")
for row in groundedness:
    if len(row.get("video_paths") or []) != 6 or row.get("media_role") != "full":
        raise SystemExit("groundedness_video_count must be 6 full videos")

by_answerability_identity = prompt_rows_by_generation_identity(
    prompts,
    stage="answerability",
)
by_formality_identity = prompt_rows_by_generation_identity(
    prompts,
    stage="qa_formality_judge",
)
by_segment_identity = prompt_rows_by_generation_identity(
    prompts,
    stage="evidence_segment_observation",
)
by_aggregation_identity = prompt_rows_by_generation_identity(
    prompts,
    stage="evidence_groundedness_aggregation",
)

accepted_answerability_call_counts = []
for row in accepted:
    generation_slot_id = str(row.get("generation_slot_id") or "")
    qa_id = str(row.get("qa_id"))
    attempt = row.get("attempt_count")
    identity = (str(row.get("generation_slot_id")), qa_id, attempt)
    if not generation_slot_id:
        raise SystemExit(f"accepted QA is missing generation_slot_id: qa_id={qa_id}")
    answerability_prompt_rows = by_answerability_identity.get(identity, [])
    accepted_answerability_call_counts.append(len(answerability_prompt_rows))
    if len(answerability_prompt_rows) != 2:
        raise SystemExit(
            f"accepted QA must have exactly 2 answerability calls: "
            f"generation_slot_id={generation_slot_id} qa_id={qa_id} "
            f"attempt={attempt} count={len(answerability_prompt_rows)}"
        )
    answerability_rows = {
        str(item.get("condition_type") or ""): item
        for item in answerability_prompt_rows
    }
    speaker_answerability = answerability_rows.get("speaker_only")
    all_six_answerability = answerability_rows.get("combined_all_six_users")
    if (
        set(answerability_rows) != {"speaker_only", "combined_all_six_users"}
        or speaker_answerability is None
        or all_six_answerability is None
    ):
        raise SystemExit(f"accepted QA is missing an answerability condition: {qa_id}")
    speaker_user = str(row.get("speaker_user") or "")
    if not speaker_user:
        raise SystemExit(f"accepted row is missing speaker_user: {qa_id}")
    if speaker_answerability.get("condition_id") != f"speaker_only::{speaker_user}":
        raise SystemExit(f"answerability condition does not match speaker: {qa_id}")
    if speaker_answerability.get("media_role") != "full":
        raise SystemExit(f"speaker-only answerability must use full media: {qa_id}")
    speaker_evidence = [
        item
        for item in (row.get("video_evidence") or [])
        if str(item.get("user") or "") == speaker_user
    ]
    if len(speaker_evidence) != 1:
        raise SystemExit(f"accepted row must identify exactly one speaker video: {qa_id}")
    speaker_full_video = str(speaker_evidence[0].get("full_local_video") or "")
    if not speaker_full_video:
        raise SystemExit(f"accepted row speaker full video is missing: {qa_id}")
    if speaker_answerability.get("video_paths") != [speaker_full_video]:
        raise SystemExit(f"answerability video does not match speaker full video: {qa_id}")
    if len(all_six_answerability.get("video_paths") or []) != 6:
        raise SystemExit(f"all-six answerability must use six full videos: {qa_id}")
    gate = (((row.get("review") or {}).get("answerability") or {}).get("gate") or {})
    required = {
        "speaker_only_answerable": False,
        "all_six_answerable": True,
        "answerability_evaluated_condition_count": 2,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise SystemExit(f"accepted row has invalid {key}: {gate.get(key)!r}")
    segment_rows = by_segment_identity.get(identity, [])
    if len(segment_rows) != 6:
        raise SystemExit(
            f"accepted QA must have 6 evidence segment observations: "
            f"generation_slot_id={generation_slot_id} qa_id={qa_id} "
            f"attempt={attempt} count={len(segment_rows)}"
        )
    segment_users = {str(item.get("user") or "") for item in segment_rows}
    if segment_users != set(row.get("required_users") or []):
        raise SystemExit(f"evidence segment users do not match required_users: {qa_id}")
    if any(
        len(item.get("video_paths") or []) != expected_segments_per_user
        or len(item.get("segments") or []) != expected_segments_per_user
        or item.get("media_role") != "full_original_30s_segments"
        for item in segment_rows
    ):
        raise SystemExit(
            "evidence segment rows must use the duration-derived 30-second segments: "
            f"qa_id={qa_id} expected={expected_segments_per_user}"
        )
    aggregation_rows = by_aggregation_identity.get(identity, [])
    if len(aggregation_rows) != 1:
        raise SystemExit(
            f"accepted QA must have 1 evidence aggregation: "
            f"generation_slot_id={generation_slot_id} qa_id={qa_id} "
            f"attempt={attempt} count={len(aggregation_rows)}"
        )
    if aggregation_rows[0].get("video_paths") not in ([], None):
        raise SystemExit(f"evidence aggregation must be text-only: {qa_id}")
    if len(by_formality_identity.get(identity, [])) != 1:
        raise SystemExit(f"accepted row is missing qa_formality judge prompt: {row.get('qa_id')}")
    formality_check = (
        ((((row.get("review") or {}).get("judger") or {}).get("checks") or {}).get(
            "qa_formality"
        ) or {})
    )
    if formality_check.get("status") != "PASS":
        raise SystemExit(f"accepted row has non-passing qa_formality gate: {row.get('qa_id')}")

generated_count = len(accepted) + len(rejected) + len(time_budget_partials)
attempted_count = generated_count if time_budget_mode else len(accepted) + len(rejected)
completed_review_count = 0
completed_rows = [
    *accepted,
    *rejected,
    *[row.get("qa") or {} for row in time_budget_partials],
]

def has_complete_three_judge_review(row):
    review = row.get("review") or {}
    checks = ((review.get("judger") or {}).get("checks") or {})
    evaluations = ((review.get("answerability") or {}).get("evaluations") or [])
    condition_types = {
        str(item.get("condition_type") or "")
        for item in evaluations
    }
    return (
        (checks.get("qa_formality") or {}).get("status") in {"PASS", "FAIL"}
        and (checks.get("evidence_groundedness") or {}).get("status") in {"PASS", "FAIL"}
        and (checks.get("answerability") or {}).get("status") in {"PASS", "FAIL"}
        and condition_types == {"speaker_only", "combined_all_six_users"}
    )

for row in completed_rows:
    if has_complete_three_judge_review(row):
        completed_review_count += 1
completed_three_judge_attempt_count = sum(
    1 for row in attempt_rows if has_complete_three_judge_review(row.get("qa") or {})
)
answerability_call_count = (
    accepted_answerability_call_counts[0]
    if accepted_answerability_call_counts
    else (2 if completed_review_count else 0)
)
review_summary = summarize_review_gate_attempts(
    accepted,
    rejected,
    max_attempts=max_attempts,
)
deadline_stop = bool(time_budget_partials) or (
    time_budget_mode and time.time() >= deadline_epoch_seconds
) or termination_requested
completed_slots_by_speaker = {}
for row in [*accepted, *rejected, *time_budget_partials]:
    speaker = str(row.get("speaker_index"))
    completed_slots_by_speaker[speaker] = completed_slots_by_speaker.get(speaker, 0) + 1
attempt_count_distribution = {}
for row in [*accepted, *rejected, *time_budget_partials]:
    count = row.get("attempt_count")
    if count is None:
        count = len(row.get("attempts") or [])
    key = str(count)
    attempt_count_distribution[key] = attempt_count_distribution.get(key, 0) + 1
generation_group_ids = {
    str(row.get("generation_group_id"))
    for row in [*accepted, *rejected, *time_budget_partials]
    if row.get("generation_group_id")
}
completed_slots_by_group = {}
for row in [*accepted, *rejected, *time_budget_partials]:
    group_id = str(row.get("generation_group_id") or "missing-group")
    completed_slots_by_group[group_id] = completed_slots_by_group.get(group_id, 0) + 1
qa_per_group_target_reached = (
    expected_qa_per_group <= 0
    or (
        len(completed_slots_by_group) == len(generation_group_ids)
        and all(
            count >= expected_qa_per_group
            for count in completed_slots_by_group.values()
        )
    )
)
if time_budget_mode:
    if (
        len(accepted) > 0
        and generated_count > 0
        and completed_review_count > 0
        and qa_per_group_target_reached
    ):
        status = "partial" if deadline_stop else "completed"
    elif runtime_probe_allow_zero_accepted and completed_review_count > 0:
        status = "runtime_passed_no_accepted"
    else:
        status = "failed"
elif len(accepted) >= accepted_target:
    status = "passed"
elif runtime_probe_allow_zero_accepted and len(accepted) == 0 and completed_review_count > 0:
    status = "runtime_passed_no_accepted"
elif allow_partial and len(accepted) > 0 and completed_review_count > 0:
    status = "partial"
else:
    status = "failed"
result = {
    "status": status,
    "objective": "time_budgeted_generation" if time_budget_mode else "accepted_target_generation",
    "candidate_count": len(candidates),
    "attempted_count": attempted_count,
    "generated_count": generated_count,
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
    "time_budget_partial_count": len(time_budget_partials),
    "completed_attempt_count": len(attempt_rows),
    "completed_three_judge_attempt_count": completed_three_judge_attempt_count,
    "attempt_count_distribution": attempt_count_distribution,
    "completed_slots_by_speaker": completed_slots_by_speaker,
    "evidence_duration_seconds": evidence_duration_seconds,
    "pruning_block_seconds": pruning_block_seconds,
    "generation_group_ids": sorted(generation_group_ids),
    "completed_slots_by_group": completed_slots_by_group,
    "expected_qa_per_generation_group": expected_qa_per_group,
    "qa_per_group_target_reached": qa_per_group_target_reached,
    "deadline_stop": deadline_stop,
    "termination_requested": termination_requested,
    "generation_exit_code": generation_exit_code,
    "runtime_probe_allow_zero_accepted": runtime_probe_allow_zero_accepted,
    "max_generation_slots": max_generation_slots,
    "generation_slot_target_reached": (
        max_generation_slots is None or generated_count >= max_generation_slots
    ),
    "acceptance_rate": review_summary["acceptance_rate"],
    "generator_attempt_count": review_summary["generator_attempt_count"],
    "accepted_by_attempt": review_summary["accepted_by_attempt"],
    "rejected_after_max_attempts": review_summary["rejected_after_max_attempts"],
    "gate_review_by_attempt": review_summary["gate_review_by_attempt"],
    "gate_review_overall": review_summary["gate_review_overall"],
    "accepted_target": accepted_target,
    "target_reached": len(accepted) >= accepted_target,
    "allow_partial": allow_partial,
    "generator_video_count": 6,
    "groundedness_video_count": 0,
    "evidence_segment_observation_count": len(evidence_segments),
    "evidence_groundedness_aggregation_count": len(evidence_aggregations),
    "answerability_call_count": answerability_call_count,
    "runtime_completed_review_count": completed_review_count,
    "speaker_only_answerable": False if accepted else None,
    "all_six_answerable": True if accepted else None,
    "answerability_evaluated_condition_count": 2,
    "generator_decode": {
        "mode": generator_decode_mode,
        "temperature": generator_temperature,
        "top_p": generator_top_p,
    },
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

if [[ -s "${OUTDIR}/six_user_qa_result.json" ]] && [[ -f "${PROJECT_ROOT}/tools/render_six_user_10min_review.py" ]]; then
  python "${PROJECT_ROOT}/tools/render_six_user_10min_review.py" \
    --output-dir "${OUTDIR}" \
    --output "${OUTDIR}/six_user_qa_review_cn.md"
fi

echo "SIX_USER_QA_JOB_FINISHED run_mode=${RUN_MODE} result=${OUTDIR}/six_user_qa_result.json"
