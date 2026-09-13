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

set -Eeuo pipefail

RUN_MODE="${RUN_MODE:-six_user_qa_runtime_probe}"
ACCEPTED_TARGET="${ACCEPTED_TARGET:-1}"
EVIDENCE_TARGET="${EVIDENCE_TARGET:-1}"
MAX_GROUPS="${MAX_GROUPS:-16}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
ALLOW_PARTIAL="${ALLOW_PARTIAL:-0}"
PREPARED_EVIDENCE_SOURCE="${PREPARED_EVIDENCE_SOURCE:-}"
PREPARED_EVIDENCE_OFFSET="${PREPARED_EVIDENCE_OFFSET:-0}"
GENERATOR_DECODE_MODE="sampling"
GENERATOR_TEMPERATURE="0.7"
GENERATOR_TOP_P="0.9"
GENERATOR_TOP_K="40"

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"
PACKAGE_ROOT="${PACKAGE_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/egolife_two_user_qa/multi-user/outputs/six_user_qa}"
LOGIN_HF_TOKEN_PATH="${HF_TOKEN_PATH:-${HF_HOME:-${HOME}/.cache/huggingface}/token}"
TRAIN_ENV="${TRAIN_ENV:-/scratch/${USER}/conda/envs/qwen3vl-smoke}"
FFMPEG_ENV="${FFMPEG_ENV:-/scratch/${USER}/envs/egoqa-ffmpeg-runtime}"
MODEL_ID="${MODEL_ID:-${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/scratch/${USER}/hf_cache}"
CLIP_MODEL_ID="${CLIP_MODEL_ID:-openai/clip-vit-base-patch32}"
PYTHON="${TRAIN_ENV}/bin/python"
JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/${RUN_MODE}_${SLURM_JOB_ID}"
OUTDIR="${OUTPUT_ROOT}/${RUN_MODE}_${SLURM_JOB_ID}"
CACHE_DIR="${JOB_SCRATCH_ROOT}/media_cache"
PYTHON_ALIAS_ROOT="${JOB_SCRATCH_ROOT}/pythonpath"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-12.0}"
CUDA_KEEPER_PID=""
CUDA_KEEPER_LOG="${OUTDIR}/cuda_keeper_${SLURM_JOB_ID}.log"
STAGE_STATUS_PATH="${OUTDIR}/stage_status.json"
CURRENT_STAGE="runtime_bootstrap"

write_stage_status() {
  local status_value="$1"
  local stage_value="$2"
  local exit_code_value="${3:-0}"
  local signal_value="${4:-}"
  local timestamp_value
  timestamp_value="$(date --iso-8601=seconds)"
  if [[ -d "${OUTDIR}" ]]; then
    printf '{"status":"%s","stage":"%s","timestamp":"%s","job_id":"%s","host":"%s","exit_code":%s,"signal":"%s"}\n' \
      "${status_value}" "${stage_value}" "${timestamp_value}" \
      "${SLURM_JOB_ID:-unknown}" "$(hostname)" "${exit_code_value}" \
      "${signal_value}" > "${STAGE_STATUS_PATH}"
  fi
}

stage_start() {
  CURRENT_STAGE="$1"
  printf 'stage=%s status=started timestamp=%s job_id=%s host=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" \
    "${SLURM_JOB_ID:-unknown}" "$(hostname)"
  write_stage_status "running" "${CURRENT_STAGE}" 0
}

stage_finish() {
  CURRENT_STAGE="$1"
  printf 'stage=%s status=completed timestamp=%s job_id=%s host=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" \
    "${SLURM_JOB_ID:-unknown}" "$(hostname)"
  write_stage_status "completed" "${CURRENT_STAGE}" 0
}

cleanup() {
  if [[ -n "${CUDA_KEEPER_PID}" ]] && kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
    printf 'stage=stop_cuda_keeper status=started timestamp=%s pid=%s\n' \
      "$(date --iso-8601=seconds)" "${CUDA_KEEPER_PID}"
    kill "${CUDA_KEEPER_PID}" 2>/dev/null || true
    wait "${CUDA_KEEPER_PID}" 2>/dev/null || true
  fi
}

on_error() {
  local exit_code=$?
  local line_number="$1"
  local failed_command="$2"
  trap - ERR
  printf 'stage=%s status=failed timestamp=%s exit_code=%s line=%s command=%q\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" "${exit_code}" \
    "${line_number}" "${failed_command}" >&2
  write_stage_status "failed" "${CURRENT_STAGE}" "${exit_code}"
  exit "${exit_code}"
}

on_signal() {
  local signal_name="$1"
  local exit_code="$2"
  trap - INT TERM
  printf 'stage=%s status=cancelled timestamp=%s signal=%s exit_code=%s\n' \
    "${CURRENT_STAGE}" "$(date --iso-8601=seconds)" "${signal_name}" \
    "${exit_code}" >&2
  write_stage_status "cancelled" "${CURRENT_STAGE}" "${exit_code}" "${signal_name}"
  exit "${exit_code}"
}

trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap 'on_signal INT 130' INT
trap 'on_signal TERM 143' TERM
trap cleanup EXIT

stage_start "runtime_bootstrap"

if [[ -z "${HF_TOKEN:-}" && -s "${LOGIN_HF_TOKEN_PATH}" ]]; then
  export HF_TOKEN_PATH="${LOGIN_HF_TOKEN_PATH}"
fi
export HOME="${JOB_SCRATCH_ROOT}/home"
export XDG_CACHE_HOME="${JOB_SCRATCH_ROOT}/xdg_cache"
export HF_HOME="${MODEL_CACHE_ROOT}"
export HF_HUB_CACHE="${MODEL_CACHE_ROOT}/hub"
export TRANSFORMERS_CACHE="${MODEL_CACHE_ROOT}"
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
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export MALLOC_ARENA_MAX="${MALLOC_ARENA_MAX:-2}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export QWEN_MEMORY_SAFE_VIDEO_FPS="1.0"
export QWEN_MEMORY_SAFE_MIN_VIDEO_PIXELS="3136"
export QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="32768"
export QWEN_MEMORY_SAFE_GPU_RESERVE_GIB="5"
export QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB="32"
export QWEN_MEMORY_SAFE_ATTN_IMPLEMENTATION="sdpa"
export QWEN_MEMORY_SAFE_DEVICE_MAP="${QWEN_MEMORY_SAFE_DEVICE_MAP:-cuda}"
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
  "${HF_HUB_CACHE}" \
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
printf 'hf_authentication=%s\n' \
  "$([[ -n "${HF_TOKEN:-}" || -s "${HF_TOKEN_PATH:-/nonexistent}" ]] && printf configured || printf anonymous)"

stage_start "start_cuda_keeper"
test -x "${PYTHON}"
test -s "${CUDA_KEEPER_SCRIPT}"
"${PYTHON}" -P -c "import torch, pynvml, psutil"
"${PYTHON}" -P "${CUDA_KEEPER_SCRIPT}" \
  --threshold "${CUDA_KEEPER_THRESHOLD}" \
  --gpus "${CUDA_KEEPER_GPUS}" \
  --reserve "${CUDA_KEEPER_RESERVE}" \
  > "${CUDA_KEEPER_LOG}" 2>&1 &
CUDA_KEEPER_PID=$!
sleep 3
if ! kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
  echo "CUDA keeper exited during startup: ${CUDA_KEEPER_LOG}" >&2
  tail -n 100 "${CUDA_KEEPER_LOG}" >&2 || true
  exit 2
fi
printf 'stage=start_cuda_keeper status=ready timestamp=%s pid=%s log=%s\n' \
  "$(date --iso-8601=seconds)" "${CUDA_KEEPER_PID}" "${CUDA_KEEPER_LOG}"

stage_start "activate_environment"
cd "${PROJECT_ROOT}"
source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate "${TRAIN_ENV}"

stage_start "validate_python_environment"
"${PYTHON}" -P - "${TRAIN_ENV}" <<'PY'
import os
import sys
from pathlib import Path

expected_prefix = Path(sys.argv[1]).resolve()
actual_prefix = Path(sys.prefix).resolve()
conda_prefix = Path(os.environ.get("CONDA_PREFIX", "")).resolve()
if actual_prefix != expected_prefix:
    raise SystemExit(
        f"wrong Python environment: expected_prefix={expected_prefix} "
        f"actual_prefix={actual_prefix} executable={sys.executable}"
    )
if conda_prefix != expected_prefix:
    raise SystemExit(
        f"wrong conda activation: expected_prefix={expected_prefix} "
        f"CONDA_PREFIX={conda_prefix}"
    )
print(f"python_executable={sys.executable}")
print(f"python_prefix={actual_prefix}")
print(f"conda_prefix={conda_prefix}")
PY

stage_start "configure_pytorch_runtime_libraries"
PYTHON_SITE_PACKAGES="$("${PYTHON}" -P -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
PYTORCH_CUDNN_LIB="${PYTHON_SITE_PACKAGES}/nvidia/cudnn/lib"
PYTORCH_CORE_LIB="${PYTHON_SITE_PACKAGES}/torch/lib"
TORCH_RUNTIME_LIBRARY_PATH=""
if compgen -G "${PYTORCH_CUDNN_LIB}/libcudnn.so*" >/dev/null; then
  TORCH_RUNTIME_LIBRARY_PATH="${PYTORCH_CUDNN_LIB}"
fi
if [[ -d "${PYTORCH_CORE_LIB}" ]]; then
  TORCH_RUNTIME_LIBRARY_PATH="${TORCH_RUNTIME_LIBRARY_PATH}${TORCH_RUNTIME_LIBRARY_PATH:+:}${PYTORCH_CORE_LIB}"
fi
if [[ -n "${TORCH_RUNTIME_LIBRARY_PATH}" ]]; then
  export LD_LIBRARY_PATH="${TORCH_RUNTIME_LIBRARY_PATH}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
printf 'python_site_packages=%s pytorch_runtime_library_path=%s\n' \
  "${PYTHON_SITE_PACKAGES}" "${TORCH_RUNTIME_LIBRARY_PATH:-bundled_rpath_only}"

stage_start "validate_runtime_paths"
test -d "${PACKAGE_ROOT}"
test -s "${PACKAGE_ROOT}/__main__.py"
test -s "${PACKAGE_ROOT}/group_relative_clip_sampling.py"
ln -s "${PACKAGE_ROOT}" "${PYTHON_ALIAS_ROOT}/egolife_two_user_qa"
export PYTHONPATH="${PYTHON_ALIAS_ROOT}:${PROJECT_ROOT}:${PACKAGE_ROOT}:${PYTHONPATH:-}"

stage_start "validate_multi_user_package_import"
"${PYTHON}" -P - "${PACKAGE_ROOT}" <<'PY'
import sys
from pathlib import Path

import egolife_two_user_qa
import egolife_two_user_qa.prompts as prompts
import egolife_two_user_qa.qwen3vl_runner as qwen3vl_runner

expected = Path(sys.argv[1]).resolve()
package_dir = Path(egolife_two_user_qa.__file__).resolve().parent
prompts_dir = Path(prompts.__file__).resolve().parent
runner_dir = Path(qwen3vl_runner.__file__).resolve().parent
if package_dir != expected or prompts_dir != expected or runner_dir != expected:
    raise SystemExit(
        "wrong EgoLife package imported: "
        f"expected={expected} package_dir={package_dir} prompts_dir={prompts_dir} "
        f"runner_dir={runner_dir}"
    )
print(f"multi_user_package_dir={package_dir}")
print(f"multi_user_prompts_path={Path(prompts.__file__).resolve()}")
print(f"multi_user_runner_path={Path(qwen3vl_runner.__file__).resolve()}")
PY

stage_start "validate_baseline_neutral_prompt_contract"
"${PYTHON}" -P - <<'PY'
from egolife_two_user_qa.prompts import (
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
    build_answerability_condition_aggregation_prompt,
    build_answerability_fact_plan_prompt,
    build_answerability_user_fact_audit_prompt,
    build_evidence_observation_aggregation_prompt,
    build_evidence_segment_observation_prompt,
    build_qa_formality_judge_prompt,
    build_video_generation_prompt,
)

users = [f"User{index}" for index in range(6)]
packet = {
    "evidence_id": "prompt_contract_probe",
    "required_users": users,
    "provider_users": users[1:],
    "clips": [
        {
            "agent_name": user,
            "generator_media_mode": (
                "all_clustering_frames_only"
                if index == 0
                else "retained_cluster_frames_only"
            ),
            "media_role": (
                "speaker_all_clustering_frames"
                if index == 0
                else "provider_retained_cluster_frames"
            ),
            "frames": [{"path": f"unused-{index}.jpg", "timestamp_seconds": index}],
        }
        for index, user in enumerate(users)
    ],
}
qa_item = {
    "qa_id": "prompt_contract_probe",
    "question": "What happened after I stepped away?",
    "options": ["Option one", "Option two", "Option three", "Option four", "Option five"],
    "correct": "A",
    "answer": "Option one",
    "required_users": users,
}
condition = {
    "condition_id": "combined_all_six_users::all",
    "condition_type": "combined_all_six_users",
    "users": users,
}
fact_plan = {
    "reason": "The question anchor and answer-bearing detail are needed.",
    "needed_facts": [
        {"fact_id": "F1", "fact": "The anchor is visible.", "why_needed": "It grounds the question."},
        {"fact_id": "F2", "fact": "The external detail is visible.", "why_needed": "It resolves the question."},
    ],
}
observations = [
    {
        "user": users[0],
        "claims": [
            {
                "claim": "The material event is visible.",
                "status": "SUPPORTED",
                "segment_references": ["segment_001"],
                "visual_description": "The event is directly visible.",
            }
        ],
    }
]

rendered = {
    "generation": build_video_generation_prompt(packet, "neutral"),
    "qa_formality": build_qa_formality_judge_prompt(qa_item, packet),
    "evidence_map": build_evidence_segment_observation_prompt(
        qa_item, user=users[0], segment_count=1
    ),
    "evidence_reduce": build_evidence_observation_aggregation_prompt(
        qa_item, packet, observations=observations
    ),
    "answerability_plan": build_answerability_fact_plan_prompt(qa_item),
    "answerability_map": build_answerability_user_fact_audit_prompt(
        qa_item, user=users[0], fact_plan=fact_plan, segment_count=1
    ),
    "answerability_reduce": build_answerability_condition_aggregation_prompt(
        qa_item,
        condition=condition,
        fact_plan=fact_plan,
        user_audits=[],
    ),
}
forbidden = (
    "The following are structural hints",
    "Information gap.",
    "Cross-view comparison or asymmetry.",
    "Post-handoff recipient follow-up.",
    "Temporal difference.",
    "State verification or change.",
    "Six-user interaction-chain example",
    "red tape dispenser with a torn white label",
    "2D gaze",
    "gaze summary",
    "gaze information",
    "Object-detection hints",
    "CLIP retrieval hints",
    "packet_image_index",
    "original_timestamp_seconds",
    "packet_image_order",
)
for stage, prompt in rendered.items():
    hits = [fragment for fragment in forbidden if fragment.lower() in prompt.lower()]
    if hits:
        raise SystemExit(f"retired prompt content rendered in {stage}: {hits}")

generation = rendered["generation"]
required_generation = (
    "speaker would naturally have and genuinely want to ask",
    "speaker's sampled frames alone must remain insufficient",
    "every CLIP-sampled frame from the speaker",
    "only sampled members of provider clusters that survived pruning",
    "It receives no MP4",
    "Concurrent-activity restriction",
    "Do not generate a question whose answer is what one person was doing",
    "Do not generate options that encode pairs of concurrent activities",
)
missing = [fragment for fragment in required_generation if fragment not in generation]
if missing:
    raise SystemExit(f"baseline-neutral generation contract is incomplete: {missing}")
if "other_person_activity_query" not in QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES:
    raise SystemExit("qa_formality no longer blocks concurrent-activity questions")
formality = rendered["qa_formality"]
required_formality = (
    "3. other_person_activity_query",
    "one provider's event used to query another provider's activity",
    "FAIL pair-matching questions",
)
missing = [fragment for fragment in required_formality if fragment not in formality]
if missing:
    raise SystemExit(f"concurrent-activity formality restriction is incomplete: {missing}")
for stage_name, marker in {
    "evidence_map": "Stage marker: evidence_segment_observation",
    "evidence_reduce": "Stage marker: evidence_groundedness_aggregation",
    "answerability_plan": "Stage marker: answerability_fact_plan",
    "answerability_map": "Stage marker: answerability_user_fact_audit",
    "answerability_reduce": "Stage marker: answerability_condition_aggregation",
}.items():
    if marker not in rendered[stage_name]:
        raise SystemExit(f"six-user map/reduce prompt is missing {marker}")
for stage_name, prompt in rendered.items():
    for forbidden_mapping in ("local_video", "timestamp_seconds", "window_start_seconds"):
        if forbidden_mapping in prompt:
            raise SystemExit(
                f"six-user prompt exposed exact media metadata in {stage_name}: "
                f"{forbidden_mapping}"
            )
print("baseline_neutral_prompt_contract=passed")
PY

test -x "${PYTHON}"
if [[ -d "${MODEL_ID}" ]]; then
  test -s "${MODEL_ID}/config.json"
  MODEL_SOURCE="local_directory"
elif [[ "${MODEL_ID}" == */* ]]; then
  MODEL_SOURCE="huggingface_model_id"
else
  echo "Invalid MODEL_ID: ${MODEL_ID}" >&2
  exit 1
fi
export MODEL_ID MODEL_SOURCE MODEL_CACHE_ROOT

stage_start "resolve_ffmpeg_runtime"
FFMPEG_BINARY=""
FFPROBE_BINARY=""
FFMPEG_SOURCE=""
if [[ -x "${FFMPEG_ENV}/bin/ffmpeg" && -x "${FFMPEG_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${FFMPEG_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${FFMPEG_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="FFMPEG_ENV"
elif [[ -x "${TRAIN_ENV}/bin/ffmpeg" && -x "${TRAIN_ENV}/bin/ffprobe" ]]; then
  FFMPEG_BINARY="${TRAIN_ENV}/bin/ffmpeg"
  FFPROBE_BINARY="${TRAIN_ENV}/bin/ffprobe"
  FFMPEG_SOURCE="TRAIN_ENV"
else
  FFMPEG_BINARY="$(command -v ffmpeg || true)"
  FFPROBE_BINARY="$(command -v ffprobe || true)"
  FFMPEG_SOURCE="activated_PATH"
fi
if [[ -z "${FFMPEG_BINARY}" || ! -x "${FFMPEG_BINARY}" || \
      -z "${FFPROBE_BINARY}" || ! -x "${FFPROBE_BINARY}" ]]; then
  printf 'FFmpeg runtime not found. Checked FFMPEG_ENV=%s, TRAIN_ENV=%s, and activated PATH=%s\n' \
    "${FFMPEG_ENV}" "${TRAIN_ENV}" "${PATH}" >&2
  printf 'ffmpeg_candidate=%s ffprobe_candidate=%s\n' \
    "${FFMPEG_BINARY:-missing}" "${FFPROBE_BINARY:-missing}" >&2
  exit 1
fi
export FFMPEG_BINARY FFPROBE_BINARY
export PATH="$(dirname "${FFMPEG_BINARY}"):$(dirname "${FFPROBE_BINARY}"):${PATH}"
printf 'stage=resolve_ffmpeg_runtime status=ready timestamp=%s source=%s ffmpeg=%s ffprobe=%s\n' \
  "$(date --iso-8601=seconds)" "${FFMPEG_SOURCE}" \
  "${FFMPEG_BINARY}" "${FFPROBE_BINARY}"

stage_start "storage_preflight"
"${PYTHON}" -P -m training.torch_storage_preflight \
  --allowed-root "/scratch/${USER}" \
  --output "${OUTDIR}/storage_preflight.json"
BRANCH_NAME="$(git branch --show-current 2>/dev/null || true)"
if [[ -n "$(git status --porcelain 2>/dev/null || true)" ]]; then
  DIRTY_STATE="dirty"
else
  DIRTY_STATE="clean"
fi
stage_start "write_job_manifest"
"${PYTHON}" -P - "${OUTDIR}/job_manifest.json" <<PY
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
    "package_root": "${PACKAGE_ROOT}",
    "output_dir": "${OUTDIR}",
    "job_scratch_root": "${JOB_SCRATCH_ROOT}",
    "model_id": "${MODEL_ID}",
    "model_source": "${MODEL_SOURCE}",
    "model_cache_root": "${MODEL_CACHE_ROOT}",
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
    "prepared_evidence_source": "${PREPARED_EVIDENCE_SOURCE}",
    "prepared_evidence_offset": int("${PREPARED_EVIDENCE_OFFSET}"),
    "generator_decode": {
        "mode": "${GENERATOR_DECODE_MODE}",
        "temperature": float("${GENERATOR_TEMPERATURE}"),
        "top_p": float("${GENERATOR_TOP_P}"),
        "top_k": int("${GENERATOR_TOP_K}"),
    },
}
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

stage_start "dependency_and_cuda_preflight"
PIP_CHECK_STATUS="passed"
if "${PYTHON}" -P -m pip check > "${OUTDIR}/pip_check.txt" 2>&1; then
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
"${PYTHON}" -P -m egolife_two_user_qa --help >/dev/null
"${FFMPEG_BINARY}" -version | head -n 1
"${FFPROBE_BINARY}" -version | head -n 1
"${PYTHON}" -P - <<'PY'
import decord
import torch
from torchcodec.decoders import VideoDecoder
from qwen_vl_utils.vision_process import get_video_reader_backend

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
cudnn_version = torch.backends.cudnn.version()
if cudnn_version is None or cudnn_version < 90300:
    raise SystemExit(f"incompatible cuDNN runtime: {cudnn_version!r}")
video_backend = get_video_reader_backend()
if video_backend != "decord":
    raise SystemExit(f"unexpected Qwen video backend: {video_backend}")
print(f"torch={torch.__version__}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"cudnn={cudnn_version}")
print(f"decord={decord.__version__}")
print(f"video_backend={video_backend}")
print(f"torchcodec_decoder={VideoDecoder.__module__}")
PY

stage_start "validate_model_runtime"
"${PYTHON}" -P - "${MODEL_ID}" <<'PY'
import sys

from transformers import AutoConfig

model_id = sys.argv[1]
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
print(f"model_id={model_id}")
print(f"model_type={getattr(config, 'model_type', 'unknown')}")
PY

"${PYTHON}" -P - "${OUTDIR}/job_manifest.json" <<PY
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = "running"
payload["pip_check_status"] = "${PIP_CHECK_STATUS}"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

if [[ -n "${PREPARED_EVIDENCE_SOURCE}" ]]; then
  stage_start "select_prepared_six_user_candidates"
  "${PYTHON}" -P - \
    "${PREPARED_EVIDENCE_SOURCE}" \
    "${OUTDIR}/six_user_candidates.jsonl" \
    "${OUTDIR}/prepared_evidence_selection.json" \
    "${OUTPUT_ROOT}" \
    "${EVIDENCE_TARGET}" \
    "${PREPARED_EVIDENCE_OFFSET}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

source = Path(sys.argv[1]).resolve()
destination = Path(sys.argv[2]).resolve()
selection_path = Path(sys.argv[3]).resolve()
allowed_root = Path(sys.argv[4]).resolve()
target_count = int(sys.argv[5])
offset = int(sys.argv[6])

if target_count < 1 or offset < 0:
    raise SystemExit(
        f"invalid prepared evidence selection: target_count={target_count} offset={offset}"
    )
try:
    common = Path(os.path.commonpath([str(source), str(allowed_root)]))
except ValueError as exc:
    raise SystemExit(f"prepared evidence source is outside OUTPUT_ROOT: {source}") from exc
if common != allowed_root:
    raise SystemExit(
        f"prepared evidence source must be inside OUTPUT_ROOT: "
        f"source={source} output_root={allowed_root}"
    )
if not source.is_file():
    raise SystemExit(f"prepared evidence source is missing: {source}")
source_bytes = source.read_bytes()
rows = [
    json.loads(line)
    for line in source_bytes.decode("utf-8").splitlines()
    if line.strip()
]
end = offset + target_count
if len(rows) < end:
    raise SystemExit(
        f"prepared evidence source has too few rows: source_rows={len(rows)} "
        f"offset={offset} target_count={target_count}"
    )
selected = rows[offset:end]
evidence_ids = [str(row.get("evidence_id") or "") for row in selected]
if not all(evidence_ids) or len(set(evidence_ids)) != len(evidence_ids):
    raise SystemExit("selected prepared evidence IDs must be non-empty and unique")

destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
    encoding="utf-8",
)
selection = {
    "source": str(source),
    "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "source_row_count": len(rows),
    "offset": offset,
    "selected_count": len(selected),
    "selected_evidence_ids": evidence_ids,
    "destination": str(destination),
    "source_assets_policy": "read_only_reuse",
}
selection_path.write_text(
    json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(selection, ensure_ascii=False))
PY
else
  stage_start "build_manifest"
  "${PYTHON}" -P -m egolife_two_user_qa build_manifest \
    --output "${OUTDIR}/manifest.json" \
    --no-overlays

  stage_start "mine_six_user_candidates"
  "${PYTHON}" -P "${PACKAGE_ROOT}/group_relative_clip_sampling.py" \
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
    --high-similarity-interval-threshold 0.82 \
    --min-pruned-video-seconds 8 \
    --random-seed 42 \
    --ffmpeg-binary "${FFMPEG_BINARY}" \
    --download-media
fi

stage_start "validate_six_user_candidates"
"${PYTHON}" -P - "${OUTDIR}/six_user_candidates.jsonl" "${EVIDENCE_TARGET}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_count = int(sys.argv[2])
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
if len(rows) != expected_count:
    raise SystemExit(
        f"expected exactly {expected_count} six-user packets, materialized {len(rows)}"
    )
expected_roles = [
    "speaker_all_clustering_frames",
    "provider_retained_cluster_frames",
    "provider_retained_cluster_frames",
    "provider_retained_cluster_frames",
    "provider_retained_cluster_frames",
    "provider_retained_cluster_frames",
]
for row in rows:
    users = row.get("input_users")
    clips = row.get("clips")
    if row.get("candidate_type") != "six_user_speaker_consensus":
        raise SystemExit("candidate_type mismatch")
    if row.get("generator_media_mode") != "speaker_all_clustering_frames_five_provider_retained_cluster_frames":
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
    pruning = row.get("speaker_consensus_pruning") or {}
    if pruning.get("method") != "speaker_provider_all_pairs_provider_only":
        raise SystemExit("six-user pruning method must use provider-only all-pairs comparison")
    if pruning.get("comparison_scope") != "every_speaker_cluster_x_every_provider_cluster":
        raise SystemExit("six-user pruning must compare every asker-provider cluster pair")
    if pruning.get("pruned_side") != "providers_only" or pruning.get("speaker_preserved") is not True:
        raise SystemExit("six-user pruning must preserve the asker and prune providers only")
    speaker_index = int(pruning.get("speaker_index", -1))
    provider_indices = [int(index) for index in (pruning.get("provider_indices") or [])]
    if provider_indices != [index for index in range(6) if index != speaker_index]:
        raise SystemExit("six-user pruning must cover all five providers")
    speaker_cluster_count = int(pruning.get("speaker_cluster_count", -1))
    provider_cluster_counts = pruning.get("provider_cluster_counts") or {}
    expected_comparisons = speaker_cluster_count * sum(
        int(provider_cluster_counts.get(str(index), -1))
        for index in provider_indices
    )
    if expected_comparisons <= 0:
        raise SystemExit("six-user pruning cluster counts must be positive")
    if int(pruning.get("expected_pairwise_comparison_count", -1)) != expected_comparisons:
        raise SystemExit("six-user pruning expected comparison count is inconsistent")
    if int(pruning.get("pairwise_comparison_count", -1)) != expected_comparisons:
        raise SystemExit("six-user pruning did not compare every asker-provider cluster pair")
    pruning_videos = {
        int(video.get("video_index", -1)): video
        for video in (pruning.get("videos") or [])
    }
    if set(pruning_videos) != set(range(6)):
        raise SystemExit("six-user pruning diagnostics must cover all six videos")
    speaker_pruning = pruning_videos[speaker_index]
    if (
        speaker_pruning.get("marked_cluster_indices")
        or speaker_pruning.get("remove_intervals")
        or float(speaker_pruning.get("removed_duration_seconds") or 0.0) != 0.0
    ):
        raise SystemExit("asker clusters or intervals were pruned")
    threshold = float(pruning.get("high_similarity_threshold"))
    marked_by_provider = {index: set() for index in provider_indices}
    above_threshold_pair_count = 0
    for event in pruning.get("events") or []:
        provider_index = int(event.get("provider_index", -1))
        provider_cluster_index = int(event.get("provider_cluster_index", -1))
        if provider_index not in marked_by_provider or provider_cluster_index < 0:
            raise SystemExit("pruning event targets a non-provider cluster")
        speaker_matches = event.get("speaker_matches") or []
        if not speaker_matches or any(
            float(match.get("similarity", -1.0)) < threshold
            for match in speaker_matches
        ):
            raise SystemExit("pruning event contains a below-threshold cluster pair")
        deleted_clusters = event.get("deleted_clusters") or []
        if deleted_clusters != [{"video_index": provider_index, "cluster_index": provider_cluster_index}]:
            raise SystemExit("pruning event must delete only its matched provider cluster")
        marked_by_provider[provider_index].add(provider_cluster_index)
        above_threshold_pair_count += len(speaker_matches)
    if int(pruning.get("above_threshold_pair_count", -1)) != above_threshold_pair_count:
        raise SystemExit("above-threshold comparison count is inconsistent")
    for provider_index, marked_clusters in marked_by_provider.items():
        if pruning_videos[provider_index].get("marked_cluster_indices") != sorted(marked_clusters):
            raise SystemExit("provider pruning diagnostics do not match threshold events")
    for index, clip in enumerate(clips):
        full_path = Path(str(clip.get("full_local_video") or ""))
        if not full_path.is_file():
            raise SystemExit(f"missing candidate media for {clip.get('agent_name')}")
        if clip.get("local_video") or clip.get("generator_local_video"):
            raise SystemExit("generator packet must not expose a generator MP4")
        if clip.get("force_frame_inputs") is not True:
            raise SystemExit("six-user generator clips must force image inputs")
        frames = clip.get("frames") or []
        if not frames or not all(Path(str(frame.get("path") or "")).is_file() for frame in frames):
            raise SystemExit(f"missing generator frames for {clip.get('agent_name')}")
        source_video_index = int((clip.get("temporal_pruning") or {}).get("source_video_index", -1))
        video_pruning = pruning_videos.get(source_video_index)
        if video_pruning is None:
            raise SystemExit("clip pruning metadata references an unknown source video")
        marked_clusters = {
            int(value) for value in (video_pruning.get("marked_cluster_indices") or [])
        }
        expected_frame_count = sum(
            len(cluster.get("member_indices") or [])
            for cluster in (video_pruning.get("clusters") or [])
            if index == 0 or int(cluster.get("cluster_index", -1)) not in marked_clusters
        )
        if len(frames) != expected_frame_count:
            raise SystemExit(
                f"generator frame count does not match retained cluster membership for "
                f"{clip.get('agent_name')}: expected={expected_frame_count} actual={len(frames)}"
            )
        if [int(frame.get("sampled_frame_index", -1)) for frame in frames] != sorted(
            int(frame.get("sampled_frame_index", -1)) for frame in frames
        ):
            raise SystemExit("generator frames must remain in chronological sampled-frame order")
        if index == 0:
            if clip.get("is_pruned") is not False:
                raise SystemExit("speaker generator frame set must be marked unpruned")
            if clip.get("generator_media_mode") != "all_clustering_frames_only":
                raise SystemExit("speaker generator must receive all clustering frames")
            if marked_clusters:
                raise SystemExit("speaker generator frame set cannot omit a cluster")
        else:
            if clip.get("is_pruned") is not True:
                raise SystemExit("provider generator frame sets must be marked pruned")
            if clip.get("generator_media_mode") != "retained_cluster_frames_only":
                raise SystemExit("provider generator must receive retained cluster frames only")
            if any(int(frame.get("cluster_index", -1)) in marked_clusters for frame in frames):
                raise SystemExit("provider generator received a frame from a pruned cluster")
print(f"candidate_contract_passed rows={len(rows)}")
PY

stage_start "decode_generator_images_preflight"
"${PYTHON}" -P - "${OUTDIR}/six_user_candidates.jsonl" <<'PY'
import json
import sys
from pathlib import Path

from qwen_vl_utils import process_vision_info

path = Path(sys.argv[1])
first_row = json.loads(next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
image_paths = [
    str(frame["path"])
    for clip in first_row["clips"]
    for frame in (clip.get("frames") or [])
]
if not image_paths or not all(Path(item).is_file() for item in image_paths):
    raise SystemExit("six-user image preflight requires materialized generator frames")

messages = [{
    "role": "user",
    "content": [
        *[
            {
                "type": "image",
                "image": item,
                "max_pixels": 65536,
            }
            for item in image_paths
        ],
        {"type": "text", "text": "six-user runtime preflight"},
    ],
}]
try:
    image_inputs, video_inputs, _ = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
except TypeError:
    try:
        image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
    except TypeError:
        image_inputs, video_inputs = process_vision_info(messages)
if image_inputs is None or len(image_inputs) != len(image_paths):
    raise SystemExit(
        f"Qwen image preflight decoded {0 if image_inputs is None else len(image_inputs)} "
        f"of {len(image_paths)} images"
    )
if video_inputs:
    raise SystemExit("Qwen image preflight unexpectedly decoded generator videos")
print(f"six_user_image_preflight_passed image_count={len(image_inputs)}")
PY

stage_start "generate_six_user_qa"
"${PYTHON}" -P -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${OUTDIR}/six_user_candidates.jsonl" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${ACCEPTED_TARGET}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local-memory-safe \
  --model-id "${MODEL_ID}" \
  --dtype bfloat16 \
  --max-image-pixels 65536 \
  --max-new-tokens 2048 \
  --disable-thinking \
  --generation-mode baseline \
  --generator-decode-mode "${GENERATOR_DECODE_MODE}" \
  --generator-temperature "${GENERATOR_TEMPERATURE}" \
  --generator-top-p "${GENERATOR_TOP_P}" \
  --generator-top-k "${GENERATOR_TOP_K}" \
  --judge-video-source full \
  --fixed-question-type-schedule \
  --question-types neutral

stage_start "validate_generated_qa"
"${PYTHON}" -P - "${OUTDIR}" "${ACCEPTED_TARGET}" "${ALLOW_PARTIAL}" "${MAX_ATTEMPTS}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
accepted_target = int(sys.argv[2])
allow_partial = bool(int(sys.argv[3]))
max_attempts = int(sys.argv[4])
if max_attempts < 1:
    raise SystemExit("max_attempts must be at least 1")

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
groundedness_maps = [row for row in prompts if row.get("stage") == "evidence_segment_observation"]
groundedness_reductions = [
    row for row in prompts if row.get("stage") == "evidence_groundedness_aggregation"
]
answerability_plans = [row for row in prompts if row.get("stage") == "answerability_fact_plan"]
answerability_maps = [row for row in prompts if row.get("stage") == "answerability_user_fact_audit"]
answerability_reductions = [
    row for row in prompts if row.get("stage") == "answerability_condition_aggregation"
]

candidate_evidence_ids = [str(row.get("evidence_id") or "") for row in candidates]
if not all(candidate_evidence_ids) or len(set(candidate_evidence_ids)) != len(candidates):
    raise SystemExit("candidate evidence IDs must be non-empty and unique")
candidate_evidence_id_set = set(candidate_evidence_ids)
candidate_image_counts = {
    str(row.get("evidence_id")): sum(
        len(clip.get("frames") or []) for clip in (row.get("clips") or [])
    )
    for row in candidates
}
generation_attempts_by_evidence = {evidence_id: [] for evidence_id in candidate_evidence_ids}
for row in generation:
    evidence_id = str(row.get("evidence_id") or "")
    attempt = row.get("attempt")
    if evidence_id not in candidate_evidence_id_set:
        raise SystemExit(f"generation trace contains unknown evidence_id: {evidence_id!r}")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise SystemExit(f"generation attempt must be an integer: {evidence_id} {attempt!r}")
    generation_attempts_by_evidence[evidence_id].append(attempt)
for evidence_id, attempts in generation_attempts_by_evidence.items():
    if not attempts:
        raise SystemExit(f"every candidate packet must enter generation: {evidence_id}")
    expected_attempts = list(range(1, len(attempts) + 1))
    if attempts != expected_attempts:
        raise SystemExit(
            f"generation attempts must be contiguous and ordered for {evidence_id}: "
            f"expected={expected_attempts} actual={attempts}"
        )
    if len(attempts) > max_attempts:
        raise SystemExit(
            f"generation attempts exceed MAX_ATTEMPTS for {evidence_id}: "
            f"count={len(attempts)} max={max_attempts}"
        )

for row in generation:
    evidence_id = str(row.get("evidence_id") or "")
    if row.get("video_paths"):
        raise SystemExit("generator_video_count must be 0")
    if len(row.get("image_paths") or []) != candidate_image_counts[evidence_id]:
        raise SystemExit("generator_image_count must match the candidate frame contract")
    expected_decode = {
        "mode": "sampling",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
    }
    if row.get("generator_decode") != expected_decode:
        raise SystemExit(
            f"generator decoding contract mismatch: "
            f"expected={expected_decode} actual={row.get('generator_decode')}"
        )
    prompt = str(row.get("prompt") or "")
    required_prompt_fragments = [
        "speaker would naturally have and genuinely want to ask",
        "speaker's sampled frames alone must remain insufficient",
        "every CLIP-sampled frame from the speaker",
        "only sampled members of provider clusters that survived pruning",
        "It receives no MP4",
        "Concurrent-activity restriction",
        "Do not generate options that encode pairs of concurrent activities",
    ]
    missing_fragments = [fragment for fragment in required_prompt_fragments if fragment not in prompt]
    if missing_fragments:
        raise SystemExit(f"generation prompt is missing six-user guidance: {missing_fragments}")
    forbidden_prompt_fragments = [
        "The following are structural hints",
        "Information gap.",
        "Cross-view comparison or asymmetry.",
        "Post-handoff recipient follow-up.",
        "Temporal difference.",
        "State verification or change.",
        "Six-user interaction-chain example",
        "red tape dispenser with a torn white label",
        "2D gaze",
        "gaze summary",
        "gaze information",
        "Object-detection hints",
        "CLIP retrieval hints",
        "packet_image_index",
        "original_timestamp_seconds",
        "packet_image_order",
    ]
    forbidden_hits = [
        fragment
        for fragment in forbidden_prompt_fragments
        if fragment.lower() in prompt.lower()
    ]
    if forbidden_hits:
        raise SystemExit(
            f"generation prompt rendered retired content: {forbidden_hits}"
        )
for row in formality:
    prompt = str(row.get("prompt") or "")
    required_fragments = [
        "3. other_person_activity_query",
        "one provider's event used to query another provider's activity",
        "FAIL pair-matching questions",
    ]
    missing_fragments = [fragment for fragment in required_fragments if fragment not in prompt]
    if missing_fragments:
        raise SystemExit(
            "qa_formality prompt is missing the concurrent-activity blocker: "
            f"{missing_fragments}"
        )
for row in groundedness_maps:
    media_summary = row.get("media_summary") or {}
    if media_summary.get("video_count") != 1:
        raise SystemExit("each groundedness map call must receive one 30-second source segment")
    if media_summary.get("media_role") != "ordered_source_segments_one_user":
        raise SystemExit("groundedness map call has the wrong media role")
    prompt = str(row.get("prompt") or "")
    if "Stage marker: evidence_segment_observation" not in prompt:
        raise SystemExit("groundedness map prompt is missing its stage contract")
for row in groundedness_reductions:
    if (row.get("media_summary") or {}).get("video_count") != 0:
        raise SystemExit("groundedness reduction must be text only")
for row in answerability_plans:
    if "Stage marker: answerability_fact_plan" not in str(row.get("prompt") or ""):
        raise SystemExit("answerability fact plan prompt is missing")
for row in answerability_maps:
    if (row.get("media_summary") or {}).get("video_count") != 1:
        raise SystemExit("each answerability map call must receive one source segment")
    prompt = str(row.get("prompt") or "")
    if "Stage marker: answerability_user_fact_audit" not in prompt:
        raise SystemExit("answerability map prompt is missing its stage contract")
    if "You must choose exactly one answer" in prompt or '"choice"' in prompt:
        raise SystemExit("six-user answerability map prompt requests a forced-choice answer")
for row in answerability_reductions:
    if (row.get("media_summary") or {}).get("video_count") != 0:
        raise SystemExit("answerability condition reduction must be text only")

by_qa_attempt = {}
for row in [*answerability_plans, *answerability_maps, *answerability_reductions]:
    key = (str(row.get("qa_id")), row.get("attempt"))
    by_qa_attempt.setdefault(key, []).append(row)

formality_by_qa = {
    str(row.get("qa_id"))
    for row in formality
}

accepted_answerability_call_counts = []
for row in accepted:
    evidence_id = str(row.get("evidence_id") or "")
    qa_id = str(row.get("qa_id"))
    attempt = row.get("attempt_count")
    if evidence_id not in generation_attempts_by_evidence:
        raise SystemExit(f"accepted QA contains unknown evidence_id: {evidence_id!r}")
    if attempt != generation_attempts_by_evidence[evidence_id][-1]:
        raise SystemExit(
            f"accepted QA must stop on its final generation attempt: {evidence_id} "
            f"accepted_attempt={attempt!r} attempts={generation_attempts_by_evidence[evidence_id]}"
        )
    rows = by_qa_attempt.get((qa_id, attempt), [])
    accepted_answerability_call_counts.append(len(rows))
    if len(rows) != 9:
        raise SystemExit(
            f"accepted QA must have exactly 9 answerability calls: qa_id={qa_id} "
            f"attempt={attempt} count={len(rows)}"
        )
    stage_counts = {
        stage: sum(item.get("stage") == stage for item in rows)
        for stage in {
            "answerability_fact_plan",
            "answerability_user_fact_audit",
            "answerability_condition_aggregation",
        }
    }
    if stage_counts != {
        "answerability_fact_plan": 1,
        "answerability_user_fact_audit": 6,
        "answerability_condition_aggregation": 2,
    }:
        raise SystemExit(f"accepted QA has invalid answerability stages: {stage_counts}")
    answerability_rows = {
        str(item.get("condition_id") or "").split("::", 1)[0]: item
        for item in rows
        if item.get("stage") == "answerability_condition_aggregation"
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
    if (speaker_answerability.get("media_summary") or {}).get("video_count") != 0:
        raise SystemExit(f"speaker-only reduction must be text only: {qa_id}")
    if (all_six_answerability.get("media_summary") or {}).get("video_count") != 0:
        raise SystemExit(f"all-six reduction must be text only: {qa_id}")
    gate = (((row.get("review") or {}).get("answerability") or {}).get("gate") or {})
    required = {
        "answerability_mode": "shared_fact_visibility_audit",
        "speaker_only_answerable": False,
        "all_six_answerable": True,
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

accepted_evidence_ids = [str(row.get("evidence_id") or "") for row in accepted]
rejected_evidence_ids = [str(row.get("evidence_id") or "") for row in rejected]
outcome_evidence_ids = accepted_evidence_ids + rejected_evidence_ids
if not all(outcome_evidence_ids):
    raise SystemExit("accepted and rejected packet outcomes must have non-empty evidence IDs")
if len(set(outcome_evidence_ids)) != len(outcome_evidence_ids):
    raise SystemExit("each candidate packet must have exactly one accepted or rejected outcome")
if set(outcome_evidence_ids) != candidate_evidence_id_set:
    raise SystemExit("accepted and rejected outcomes must cover every candidate packet")
for evidence_id in rejected_evidence_ids:
    attempts = generation_attempts_by_evidence[evidence_id]
    if len(attempts) != max_attempts:
        raise SystemExit(
            f"rejected packet must exhaust MAX_ATTEMPTS: {evidence_id} "
            f"count={len(attempts)} max={max_attempts}"
        )

attempted_count = len(outcome_evidence_ids)
answerability_call_count = (
    accepted_answerability_call_counts[0]
    if accepted_answerability_call_counts
    else 0
)
if len(accepted) >= accepted_target:
    status = "passed"
elif allow_partial and accepted:
    status = "partial"
else:
    status = "failed"
result = {
    "status": status,
    "candidate_count": len(candidates),
    "generation_packet_count": len(generation_attempts_by_evidence),
    "generation_attempt_count": len(generation),
    "retried_packet_count": sum(
        len(attempts) > 1 for attempts in generation_attempts_by_evidence.values()
    ),
    "max_observed_attempts_per_packet": max(
        (len(attempts) for attempts in generation_attempts_by_evidence.values()),
        default=0,
    ),
    "max_attempts_per_packet": max_attempts,
    "attempted_count": attempted_count,
    "accepted_count": len(accepted),
    "rejected_count": len(rejected),
    "accepted_target": accepted_target,
    "target_reached": len(accepted) >= accepted_target,
    "allow_partial": allow_partial,
    "generator_media_type": "images_only",
    "generator_video_count": 0,
    "minimum_generator_image_count": min(candidate_image_counts.values(), default=0),
    "maximum_generator_image_count": max(candidate_image_counts.values(), default=0),
    "groundedness_visual_call_count": 6,
    "groundedness_max_videos_per_call": 1,
    "groundedness_mode": "per_user_source_segment_map_reduce",
    "answerability_call_count": answerability_call_count,
    "answerability_mode": "shared_fact_visibility_audit",
    "speaker_only_answerable": False if accepted else None,
    "all_six_answerable": True if accepted else None,
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

stage_start "validate_outputs"
"${PYTHON}" -P -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review

stage_start "materialize_review_videos"
"${PYTHON}" -P -m egolife_two_user_qa materialize_review_videos \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --output-dir "${OUTDIR}/review_videos"

stage_finish "six_user_qa_complete"
echo "SIX_USER_QA_JOB_FINISHED run_mode=${RUN_MODE} result=${OUTDIR}/six_user_qa_result.json"
