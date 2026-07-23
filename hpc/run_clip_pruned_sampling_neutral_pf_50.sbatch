#!/usr/bin/env bash
#SBATCH --job-name=egolife_impl50
#SBATCH --account=torch_pr_111_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=hpc/logs/%x_%j.out
#SBATCH --error=hpc/logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=hm2991@nyu.edu

set -euo pipefail

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"

# Reuse the existing CLIP-pruned two-user evidence packets.
EVIDENCE_ROOT="${EGOLIFE2U_EVIDENCE_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${EVIDENCE_ROOT}/evidence_pruned_pairs.jsonl}"

OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/relation_design_text_only_formality_pf_50}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-50}"
QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
JUDGE_BACKEND="${JUDGE_BACKEND:-transformers-local}"
JUDGE_MODEL_ID="${JUDGE_MODEL_ID:-${QWEN_MODEL_ID}}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
JUDGE_MAX_NEW_TOKENS="${JUDGE_MAX_NEW_TOKENS:-4096}"
JUDGE_REASONING_EFFORT="${JUDGE_REASONING_EFFORT:-}"
ALLOW_OPENAI_VIDEO_INPUT="${ALLOW_OPENAI_VIDEO_INPUT:-0}"
QA_FORMALITY_USE_GENERATOR="${QA_FORMALITY_USE_GENERATOR:-0}"
GENERATION_MODE="${GENERATION_MODE:-baseline}"
if [[ "${GENERATION_MODE}" != "baseline" ]]; then
  echo "Archived generation mode requested: ${GENERATION_MODE}. Only baseline is active." >&2
  exit 2
fi
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
DISABLE_THINKING="${DISABLE_THINKING:-1}"
RESUME_GENERATION="${RESUME_GENERATION:-0}"

SAMPLING_TEMPERATURE="${SAMPLING_TEMPERATURE:-0.7}"
SAMPLING_TOP_P="${SAMPLING_TOP_P:-0.9}"
SAMPLING_TOP_K="${SAMPLING_TOP_K:-}"

CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"
CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/cuda.py}"
CUDA_KEEPER_THRESHOLD="${CUDA_KEEPER_THRESHOLD:-60}"
CUDA_KEEPER_GPUS="${CUDA_KEEPER_GPUS:-0}"
CUDA_KEEPER_RESERVE="${CUDA_KEEPER_RESERVE:-8.0}"
CUDA_KEEPER_AUTO_INSTALL="${CUDA_KEEPER_AUTO_INSTALL:-1}"
CUDA_KEEPER_PID=""

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

source "${PROJECT_ROOT}/hpc/env_qwen3vl.sh"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/scratch/${USER}/hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "job_id=${SLURM_JOB_ID:-}"
echo "hostname=$(hostname)"
echo "pwd=$(pwd)"
echo "date_start=$(date --iso-8601=seconds)"
echo "evidence_root=${EVIDENCE_ROOT}"
echo "evidence_path=${EVIDENCE_PATH}"
echo "outdir=${OUTDIR}"
echo "cache_dir=${CACHE_DIR}"
echo "target_count=${TARGET_COUNT}"
echo "generator_backend=transformers-local"
echo "generator_model_id=${QWEN_MODEL_ID}"
echo "judge_backend=${JUDGE_BACKEND}"
echo "judge_model_id=${JUDGE_MODEL_ID}"
echo "judge_base_url=${JUDGE_BASE_URL}"
echo "judge_max_new_tokens=${JUDGE_MAX_NEW_TOKENS}"
echo "judge_reasoning_effort=${JUDGE_REASONING_EFFORT:-provider_default}"
echo "allow_openai_video_input=${ALLOW_OPENAI_VIDEO_INPUT}"
echo "qa_formality_use_generator=${QA_FORMALITY_USE_GENERATOR}"
echo "generation_mode=${GENERATION_MODE}"
echo "archived_generation_modes=clip_guided,discovery,discovery_control"
echo "question_types=neutral"
echo "generator_decode_mode=sampling"
echo "review_decode_mode=greedy"
echo "sampling_temperature=${SAMPLING_TEMPERATURE}"
echo "sampling_top_p=${SAMPLING_TOP_P}"
echo "sampling_top_k=${SAMPLING_TOP_K}"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "disable_thinking=${DISABLE_THINKING}"
echo "resume_generation=${RESUME_GENERATION}"
echo "media_routing=generator_uses_existing_pruned_videos;judges_and_answerability_use_full_30s_originals"
echo "judge_contract=binary_PASS_FAIL_gates"
if [[ "${JUDGE_BACKEND}" == "transformers-local" \
   && "${JUDGE_MODEL_ID}" == "${QWEN_MODEL_ID}" \
   && "${JUDGE_MAX_NEW_TOKENS}" == "${MAX_NEW_TOKENS}" \
   && -z "${JUDGE_BASE_URL}" \
   && -z "${JUDGE_REASONING_EFFORT}" ]]; then
  echo "expected_runner_topology=one_shared_generator_and_judge_model"
else
  echo "expected_runner_topology=separate_generator_and_judge_runners"
fi
echo "judge_input_contract=qa_formality_text_only_without_generator_rationale;evidence_groundedness_uses_generator_rationale"
echo "pass_fail_entropy_logits=legacy_archived_not_collected"
echo "answerability_choice_logits=legacy_archived_not_collected"
echo "point_scoring=legacy_archived_not_active"
echo "generation_category_contract=absent"
echo "generator_relation_design=private_broad_search_without_family_default"
echo "concurrency_contract=either_view_anchor_or_complete_activity_pairs"
echo "qa_formality_contract=deterministic_schema_plus_five_text_only_semantic_subchecks"
echo "timestamp_detection=qa_formality_model_judge_only_no_deterministic_parser"
echo "judge_json_repair=one_text_only_attempt"
python --version
nvidia-smi || true

echo "stage=preflight_relation_design_text_only_formality"
python - <<'PY'
from pathlib import Path

import egolife_two_user_qa.prompts as prompts_module
from egolife_two_user_qa.prompts import (
    POSITIVE_EXAMPLES_GUIDANCE,
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
    VIDEO_GENERATION_SCHEMA,
    build_answerability_prompt,
    build_evidence_groundedness_judge_prompt,
    build_judge_json_repair_prompt,
    build_qa_formality_judge_prompt,
    build_video_generation_prompt,
    judge_schema_for_check,
    qa_formality_errors,
    user_facing_participant_name_errors,
)
from egolife_two_user_qa.video_qa_loop import single_judge_output_errors

actual_prompts_path = Path(prompts_module.__file__).resolve()
expected_prompts_path = (Path.cwd() / "egolife_two_user_qa" / "prompts.py").resolve()
print(f"prompts_module_path={actual_prompts_path}")
if actual_prompts_path != expected_prompts_path:
    raise SystemExit(
        "preflight imported prompts.py from the wrong checkout: "
        f"expected={expected_prompts_path} actual={actual_prompts_path}"
    )

packet = {
    "evidence_id": "preflight_prompt_contract",
    "required_users": ["asker", "provider"],
    "participant_names": ["Alice", "Bob"],
    "clips": [
        {
            "agent_name": "asker",
            "temporal_pruning": {
                "keep_intervals": [[2.0, 4.0], [7.0, 10.0]],
                "kept_duration_seconds": 5.0,
                "removed_duration_seconds": 5.0,
            },
        },
        {
            "agent_name": "provider",
            "temporal_pruning": {
                "keep_intervals": [[0.0, 1.0], [5.0, 10.0]],
                "kept_duration_seconds": 6.0,
                "removed_duration_seconds": 4.0,
            },
        },
    ],
}
qa_item = {
    "qa_id": "preflight",
    "question_type": "neutral",
    "question": "After I handed over the parcel, what happened to it next?",
    "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
    "correct": "A",
    "answer": "Option A",
    "required_users": ["asker", "provider"],
    "generator_rationale": "PRIVATE_RATIONALE_SENTINEL_DO_NOT_EXPOSE",
    "single_user_answerability": {"asker": "insufficient", "provider": "sufficient"},
    "combined_answerability": "sufficient because the available views support one option",
    "why_two_users_needed": "The asker supplies context and the provider supplies the answer.",
    "per_user_evidence_claims": {"asker": "handoff", "provider": "follow-up"},
    "referred_timestamps": {"asker": [2.0], "provider": [5.0]},
}
preflight_prompt = build_video_generation_prompt(packet, "neutral")
formality_prompt = build_qa_formality_judge_prompt(qa_item, packet, schema_errors=[])
grounding_prompt = build_evidence_groundedness_judge_prompt(qa_item, packet)
answerability_prompt = build_answerability_prompt(
    qa_item,
    {"name": "both_users", "available_users": ["asker", "provider"]},
)
repair_prompt = build_judge_json_repair_prompt(
    "not valid json",
    judge_schema_for_check("qa_formality", pass_fail_only=True),
)
banned_strict_dependency_phrases = (
    "First look for a strict relation",
    "Prefer a strict two-view relation",
    "fallback only if no grounded strict relation exists",
)
for phrase in banned_strict_dependency_phrases:
    if phrase in preflight_prompt:
        raise SystemExit(f"generator prompt retained strict-dependency preference: {phrase}")
if POSITIVE_EXAMPLES_GUIDANCE not in preflight_prompt:
    raise SystemExit("generator prompt omitted the positive examples block")
for required_relation_instruction in (
    "Privately consider several substantively different possible information needs",
    "Choose the strongest natural grounded relation supported by the videos.",
    "Do not force a family or default automatically to an object or isolated-activity question.",
    'Openings such as "When I," "After I," or "I was" are allowed when natural',
    "Relation-design rules:",
    "Ordinary information gap:",
    "Comparison: each view supplies one operand",
    "Identity or role linkage:",
    "Post-handoff follow-up:",
    "State verification:",
    "Concurrent single-anchor matching: a concrete event from either user's view may be fixed",
    "Concurrent pair matching: every option contains one concrete activity associated with each view",
    "activities are recombined across plausible options",
    '"pruned_to_original_time_map"',
    '"temporal_alignment_contract"',
    "equal pruned playback positions do not prove concurrency",
):
    if required_relation_instruction not in preflight_prompt:
        raise SystemExit(
            "generator prompt omitted active relation-design contract: "
            + required_relation_instruction
        )
for archived_generator_instruction in (
    "You can try the following question families when applicable",
    "optional, equal-status directions, not labels to choose or fields to output",
    "You can try this question family when applicable",
    "Clear visible quantity:",
    "three open boxes",
):
    if archived_generator_instruction in preflight_prompt:
        raise SystemExit(
            "generator prompt retained archived steering: "
            + archived_generator_instruction
        )
if "required_users[1] may be sufficient alone" not in preflight_prompt:
    raise SystemExit("generator prompt incorrectly restored a strict global two-view dependency")
for removed_example_token in ("guitar", "microwave"):
    if removed_example_token in preflight_prompt.casefold():
        raise SystemExit(f"generator prompt retained distinctive removed example: {removed_example_token}")
if {"category", "category_rationale"} & set(VIDEO_GENERATION_SCHEMA):
    raise SystemExit("production generation schema still requests category fields")
for stage, prompt in (
    ("generation", preflight_prompt),
    ("qa_formality", formality_prompt),
    ("evidence_groundedness", grounding_prompt),
    ("answerability", answerability_prompt),
):
    lowered = prompt.casefold()
    if "category" in lowered:
        raise SystemExit(f"{stage} prompt contains category language")
    for forbidden in (
        "broad two-user reasoning categories",
        "generator-declared category",
        '"category_rationale"',
        '"category":',
        "cross_view_concurrent_activity",
    ):
        if forbidden in lowered:
            raise SystemExit(f"{stage} prompt retained category machinery: {forbidden}")

expected_subchecks = {
    "first_person_perspective",
    "naturalness_and_clarity",
    "other_person_activity_query",
    "direct_name_leakage",
    "timestamp_citation",
}
if set(QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES) != expected_subchecks:
    raise SystemExit(
        "qa_formality semantic subcheck mismatch: "
        + repr(QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES)
    )
for required_formality_instruction in (
    "pure text-only semantic judge and do not see the videos",
    "Judge only the deterministic schema result and the user-facing question and options.",
    "PASS a pair-matching form when every option states a complete cross-view activity pair.",
    "a fixed provider-side event may ask which asker-side event was simultaneous",
    "Known participant names for leakage detection only:",
    "User-facing question-answer item:",
    *expected_subchecks,
):
    if required_formality_instruction not in formality_prompt:
        raise SystemExit(
            "qa_formality prompt omitted text-only semantic contract: "
            + required_formality_instruction
        )
for hidden_formality_value in (
    "PRIVATE_RATIONALE_SENTINEL_DO_NOT_EXPOSE",
    '"generator_rationale"',
    '"single_user_answerability"',
    '"combined_answerability"',
    '"why_two_users_needed"',
    '"per_user_evidence_claims"',
    '"referred_timestamps"',
    '"temporal_pruning"',
    '"pruned_to_original_time_map"',
):
    if hidden_formality_value in formality_prompt:
        raise SystemExit(
            "qa_formality prompt leaked excluded context: " + hidden_formality_value
        )
if "PRIVATE_RATIONALE_SENTINEL_DO_NOT_EXPOSE" not in grounding_prompt:
    raise SystemExit("evidence_groundedness prompt omitted the configured generator rationale")
for required_grounding_instruction in (
    "Incorrect distractors do not need to occur in the videos for an ordinary object",
    "verify that every component activity used across the options actually occurs",
    "Either view may supply the fixed event.",
    "exactly the declared pair overlaps on the original synchronized timeline",
):
    if required_grounding_instruction not in grounding_prompt:
        raise SystemExit(
            "evidence_groundedness prompt omitted grounding contract: "
            + required_grounding_instruction
        )
for required_answerability_instruction in (
    "If evidence is missing, ambiguous, supports more than one option",
    "a concrete event from either user's view as the relative temporal key",
    "For a cross-view activity-pair question",
    "shows only one side's activities but not which cross-view pair overlaps",
):
    if required_answerability_instruction not in answerability_prompt:
        raise SystemExit(
            "answerability prompt omitted ambiguity or symmetry contract: "
            + required_answerability_instruction
        )

timestamp_item = {
    "question": "What did I do at 00:42?",
    "options": ["Option A", "Option B", "Option C", "Option D", "Option E"],
    "correct": "A",
    "answer": "Option A",
}
if qa_formality_errors(timestamp_item, [], participant_names=[]):
    raise SystemExit("deterministic timestamp parsing was unexpectedly reactivated")
for archived_timestamp_symbol in (
    "USER_FACING_TIMESTAMP_PATTERNS",
    "user_facing_timestamp_errors",
):
    if hasattr(prompts_module, archived_timestamp_symbol):
        raise SystemExit(
            "deterministic timestamp parser symbol still exists: "
            + archived_timestamp_symbol
        )
if not user_facing_participant_name_errors(
    {"question": "What did Alice hand me?", "options": []},
    ["Alice"],
):
    raise SystemExit("deterministic participant-name leakage check is inactive")

missing_subchecks_payload = {
    "review_passed": True,
    "checks": {
        "qa_formality": {
            "status": "PASS",
            "reason": "passed",
            "fix": "",
        }
    },
    "blocking_failures": [],
    "feedback_to_generator": "",
}
contract_errors = single_judge_output_errors(missing_subchecks_payload, "qa_formality")
if not any("semantic_subchecks" in error for error in contract_errors):
    raise SystemExit("judge JSON contract does not require qa_formality semantic subchecks")
for required_repair_instruction in (
    "Preserve the same decision and content",
    "return only one valid JSON object",
    '"semantic_subchecks"',
):
    if required_repair_instruction not in repair_prompt:
        raise SystemExit(
            "judge JSON repair prompt omitted bounded repair contract: "
            + required_repair_instruction
        )

print("category_free_prompt_preflight=ok")
print("generator_relation_design_preflight=ok")
print("symmetric_concurrency_preflight=ok")
print("pruned_to_original_temporal_mapping=ok")
print("qa_formality_text_only_contract=ok")
print("qa_formality_five_semantic_subchecks=ok")
print("timestamp_judge_only_contract=ok")
print("participant_name_deterministic_check=ok")
print("grounding_distractor_contract=ok")
print("answerability_ambiguity_contract=ok")
print("judge_json_repair_contract=ok")
PY

if [[ "${JUDGE_BACKEND}" == "openrouter" && -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "error=openrouter_api_key_missing set OPENROUTER_API_KEY before sbatch" >&2
  exit 1
fi

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  if [[ -s "${CUDA_KEEPER_SCRIPT}" ]]; then
    echo "stage=preflight_cuda_keeper_imports"
    if ! python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml", "psutil") if importlib.util.find_spec(name) is None]
if missing:
    print("missing=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_imports=ok")
PY
    then
      if [[ "${CUDA_KEEPER_AUTO_INSTALL}" == "1" ]]; then
        echo "stage=install_cuda_keeper_deps"
        python -m pip install nvidia-ml-py3 psutil || true
      fi
    fi

    if python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
if missing:
    print("missing_required_after_install=" + ",".join(missing))
    raise SystemExit(1)
print("cuda_keeper_required_imports=ok")
PY
    then
      echo "stage=start_cuda_keeper"
      python "${CUDA_KEEPER_SCRIPT}" \
        --threshold "${CUDA_KEEPER_THRESHOLD}" \
        --gpus "${CUDA_KEEPER_GPUS}" \
        --reserve "${CUDA_KEEPER_RESERVE}" \
        > "${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log" 2>&1 &
      CUDA_KEEPER_PID=$!
      sleep 2
      if kill -0 "${CUDA_KEEPER_PID}" 2>/dev/null; then
        echo "cuda_keeper_pid=${CUDA_KEEPER_PID}"
      else
        echo "warning=cuda_keeper_exited_early; continuing without keeper"
        CUDA_KEEPER_PID=""
      fi
    else
      echo "warning=cuda_keeper_imports_missing; continuing without keeper"
    fi
  else
    echo "warning=cuda_keeper_script_missing path=${CUDA_KEEPER_SCRIPT}; continuing without keeper"
  fi
fi

resume_args=()
if [[ "${RESUME_GENERATION}" == "1" ]]; then
  resume_args+=(--resume)
fi

thinking_args=()
if [[ "${DISABLE_THINKING}" == "1" ]]; then
  thinking_args+=(--disable-thinking)
fi

sampling_args=(
  --generator-decode-mode sampling
  --generator-temperature "${SAMPLING_TEMPERATURE}"
  --generator-top-p "${SAMPLING_TOP_P}"
)
if [[ -n "${SAMPLING_TOP_K}" ]]; then
  sampling_args+=(--generator-top-k "${SAMPLING_TOP_K}")
fi

judge_args=()
if [[ "${JUDGE_BACKEND}" != "transformers-local" \
   || "${JUDGE_MODEL_ID}" != "${QWEN_MODEL_ID}" \
   || "${JUDGE_MAX_NEW_TOKENS}" != "${MAX_NEW_TOKENS}" \
   || -n "${JUDGE_BASE_URL}" \
   || -n "${JUDGE_REASONING_EFFORT}" ]]; then
  judge_args+=(
    --judge-backend "${JUDGE_BACKEND}"
    --judge-model-id "${JUDGE_MODEL_ID}"
    --judge-max-new-tokens "${JUDGE_MAX_NEW_TOKENS}"
  )
  if [[ -n "${JUDGE_BASE_URL}" ]]; then
    judge_args+=(--judge-base-url "${JUDGE_BASE_URL}")
  fi
  if [[ -n "${JUDGE_REASONING_EFFORT}" ]]; then
    judge_args+=(--judge-reasoning-effort "${JUDGE_REASONING_EFFORT}")
  fi
fi

judge_media_args=()
if [[ "${ALLOW_OPENAI_VIDEO_INPUT}" == "1" ]]; then
  judge_media_args+=(--allow-openai-video-input)
fi

qa_formality_args=()
if [[ "${QA_FORMALITY_USE_GENERATOR}" == "1" ]]; then
  qa_formality_args+=(--qa-formality-use-generator)
fi

echo "stage=generate_video_qa_loop"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_PATH}" \
  --output "${OUTDIR}/qa_mcq.jsonl" \
  --prompts-output "${OUTDIR}/video_first_prompts.jsonl" \
  --rejected-output "${OUTDIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${OUTDIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${TARGET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local \
  --model-id "${QWEN_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generation-mode "${GENERATION_MODE}" \
  --fixed-question-type-schedule \
  --question-types neutral \
  "${judge_args[@]}" \
  "${judge_media_args[@]}" \
  "${qa_formality_args[@]}" \
  "${sampling_args[@]}" \
  "${thinking_args[@]}" \
  "${resume_args[@]}"

echo "stage=verify_generation"
python - "${EVIDENCE_PATH}" "${OUTDIR}" "${TARGET_COUNT}" <<'PY'
import json
import sys
from pathlib import Path

from egolife_two_user_qa.prompts import (
    POSITIVE_EXAMPLES_GUIDANCE,
    QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES,
)

evidence_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
target = int(sys.argv[3])

def read_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

evidence_count = len(read_rows(evidence_path))
accepted = read_rows(output_dir / "qa_mcq.jsonl")
rejected = read_rows(output_dir / "qa_mcq.rejected.jsonl")
prompt_rows = read_rows(output_dir / "video_first_prompts.jsonl")
generation_prompts = [row for row in prompt_rows if row.get("stage") == "generation"]
formality_prompts = [
    row for row in prompt_rows if row.get("stage") == "qa_formality_judge"
]
grounding_prompts = [
    row for row in prompt_rows if row.get("stage") == "evidence_groundedness_judge"
]
answerability_prompts = [
    row for row in prompt_rows if row.get("stage") == "answerability"
]
judge_prompts = [
    row
    for row in prompt_rows
    if row.get("stage") in {"qa_formality_judge", "evidence_groundedness_judge"}
]

banned_strict_dependency_phrases = (
    "First look for a strict relation",
    "Prefer a strict two-view relation",
    "fallback only if no grounded strict relation exists",
)
category_prompt_phrases = (
    "broad two-user reasoning categories",
    "generator-declared category",
    '"category_rationale"',
    '"category":',
    "cross_view_concurrent_activity",
)

def assert_category_free(prompt, *, stage):
    lowered = prompt.casefold()
    if "category" in lowered:
        raise SystemExit(f"{stage} prompt contains category language")
    for phrase in category_prompt_phrases:
        if phrase in lowered:
            raise SystemExit(f"{stage} prompt retained category machinery: {phrase}")

for row in generation_prompts:
    mode = (row.get("generator_decode") or {}).get("mode")
    if mode != "sampling":
        raise SystemExit(f"non-sampling generator trace found: {mode}")
    if row.get("question_type") != "neutral":
        raise SystemExit(f"non-neutral generator trace found: {row.get('question_type')}")
    if row.get("generation_mode") != "baseline":
        raise SystemExit(f"archived generation mode found: {row.get('generation_mode')}")
    prompt = str(row.get("prompt") or "")
    for phrase in banned_strict_dependency_phrases:
        if phrase in prompt:
            raise SystemExit(f"generation prompt retained strict-dependency preference: {phrase}")
    if POSITIVE_EXAMPLES_GUIDANCE not in prompt:
        raise SystemExit("generation prompt omitted the positive examples block")
    for required_relation_instruction in (
        "Privately consider several substantively different possible information needs",
        "Do not force a family or default automatically to an object or isolated-activity question.",
        'Openings such as "When I," "After I," or "I was" are allowed when natural',
        "Relation-design rules:",
        "Ordinary information gap:",
        "Comparison: each view supplies one operand",
        "Identity or role linkage:",
        "Post-handoff follow-up:",
        "State verification:",
        "Concurrent single-anchor matching: a concrete event from either user's view may be fixed",
        "Concurrent pair matching: every option contains one concrete activity associated with each view",
        "activities are recombined across plausible options",
        '"pruned_to_original_time_map"',
        '"temporal_alignment_contract"',
        "equal pruned playback positions do not prove concurrency",
    ):
        if required_relation_instruction not in prompt:
            raise SystemExit(
                "generation prompt omitted active relation-design contract: "
                + required_relation_instruction
            )
    for archived_generator_instruction in (
        "You can try the following question families when applicable",
        "optional, equal-status directions, not labels to choose or fields to output",
        "You can try this question family when applicable",
        "Clear visible quantity:",
        "three open boxes",
    ):
        if archived_generator_instruction in prompt:
            raise SystemExit(
                "generation prompt retained archived steering: "
                + archived_generator_instruction
            )
    for removed_example_token in ("guitar", "microwave"):
        if removed_example_token in prompt.casefold():
            raise SystemExit(
                f"generation prompt retained distinctive removed example: {removed_example_token}"
            )
    assert_category_free(prompt, stage="generation")

expected_subchecks = set(QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES)
if expected_subchecks != {
    "first_person_perspective",
    "naturalness_and_clarity",
    "other_person_activity_query",
    "direct_name_leakage",
    "timestamp_citation",
}:
    raise SystemExit(f"qa_formality semantic subcheck mismatch: {expected_subchecks}")

for row in formality_prompts:
    if row.get("generator_rationale_included") is not False:
        raise SystemExit("qa_formality prompt unexpectedly includes generator rationale")
    if row.get("media_role") != "text_only":
        raise SystemExit("qa_formality prompt is not marked text_only")
    if row.get("image_paths") or row.get("video_paths"):
        raise SystemExit("qa_formality prompt unexpectedly received media")
    if row.get("pass_fail_only") is not True:
        raise SystemExit("qa_formality prompt is not using the binary PASS/FAIL contract")
    if "quality_quota" in row:
        raise SystemExit("qa_formality prompt retained archived quality-quota metadata")
    prompt = str(row.get("prompt") or "")
    assert_category_free(prompt, stage="qa_formality")
    for required_formality_instruction in (
        "pure text-only semantic judge and do not see the videos",
        "Judge only the deterministic schema result and the user-facing question and options.",
        "PASS a pair-matching form when every option states a complete cross-view activity pair.",
        "a fixed provider-side event may ask which asker-side event was simultaneous",
        *expected_subchecks,
    ):
        if required_formality_instruction not in prompt:
            raise SystemExit(
                "qa_formality prompt omitted text-only semantic contract: "
                + required_formality_instruction
            )
    for hidden_formality_field in (
        '"generator_rationale"',
        '"single_user_answerability"',
        '"combined_answerability"',
        '"why_two_users_needed"',
        '"per_user_evidence_claims"',
        '"referred_timestamps"',
        '"temporal_pruning"',
        '"pruned_to_original_time_map"',
        "Video set metadata:",
        "Evidence packet metadata:",
    ):
        if hidden_formality_field in prompt:
            raise SystemExit(
                "qa_formality prompt leaked excluded context: " + hidden_formality_field
            )
    for archived_term in ("quality_score rubric", "Global 3-point quota"):
        if archived_term in prompt:
            raise SystemExit(f"qa_formality prompt retained archived point scoring: {archived_term}")

for row in grounding_prompts:
    if row.get("generator_rationale_included") is not True:
        raise SystemExit("evidence_groundedness prompt omitted generator rationale")
    if row.get("media_role") != "full":
        raise SystemExit("evidence_groundedness prompt is not marked full-media")
    if row.get("pass_fail_only") is not True:
        raise SystemExit("evidence_groundedness prompt is not using the binary PASS/FAIL contract")
    prompt = str(row.get("prompt") or "")
    assert_category_free(prompt, stage="evidence_groundedness")
    for required_grounding_instruction in (
        "Incorrect distractors do not need to occur in the videos for an ordinary object",
        "verify that every component activity used across the options actually occurs",
        "Either view may supply the fixed event.",
        "exactly the declared pair overlaps on the original synchronized timeline",
    ):
        if required_grounding_instruction not in prompt:
            raise SystemExit(
                "evidence_groundedness prompt omitted grounding contract: "
                + required_grounding_instruction
            )
    for archived_term in ("quality_score rubric", "Global 3-point quota"):
        if archived_term in prompt:
            raise SystemExit(
                f"evidence_groundedness prompt retained archived point scoring: {archived_term}"
            )

for row in answerability_prompts:
    prompt = str(row.get("prompt") or "")
    assert_category_free(prompt, stage="answerability")
    for required_answerability_instruction in (
        "If evidence is missing, ambiguous, supports more than one option",
        "a concrete event from either user's view as the relative temporal key",
        "For a cross-view activity-pair question",
        "shows only one side's activities but not which cross-view pair overlaps",
    ):
        if required_answerability_instruction not in prompt:
            raise SystemExit(
                "answerability prompt omitted ambiguity or symmetry contract: "
                + required_answerability_instruction
            )

for row in [*accepted, *rejected]:
    for trace in row.get("generation_trace") or []:
        judge = trace.get("judge") if isinstance(trace, dict) else None
        if not isinstance(judge, dict) or not isinstance(judge.get("merged"), dict):
            continue
        if judge.get("pass_fail_entropy_logits") != "legacy_archived_not_collected":
            raise SystemExit("judge trace retained the legacy PASS/FAIL entropy path")
        if judge.get("answerability_choice_logits") != "legacy_archived_not_collected":
            raise SystemExit("judge trace retained the legacy answerability-logit path")
        if judge.get("point_scoring") != "legacy_archived_not_active":
            raise SystemExit("judge trace retained point scoring")
        merged = judge["merged"]
        if "decision_uncertainty_summary" in merged:
            raise SystemExit("merged judge retained legacy PASS/FAIL entropy JSON")
        for check_name in ("qa_formality", "evidence_groundedness"):
            check = (merged.get("checks") or {}).get(check_name) or {}
            for archived_field in (
                "quality_score", "quality_flag", "quality_reason", "quality_quota",
                "quota_rebuttal", "decision_uncertainty", "quality_uncertainty",
            ):
                if archived_field in check:
                    raise SystemExit(f"{check_name} retained archived field {archived_field}")
        formality_check = (merged.get("checks") or {}).get("qa_formality") or {}
        if str(formality_check.get("status") or "").upper() == "PASS":
            semantic_subchecks = formality_check.get("semantic_subchecks") or {}
            missing_subchecks = expected_subchecks - set(semantic_subchecks)
            if missing_subchecks:
                raise SystemExit(
                    "passing qa_formality check omitted semantic subchecks: "
                    + ",".join(sorted(missing_subchecks))
                )
            nonpassing_subchecks = [
                name
                for name in expected_subchecks
                if str((semantic_subchecks.get(name) or {}).get("status") or "").upper()
                != "PASS"
            ]
            if nonpassing_subchecks:
                raise SystemExit(
                    "passing qa_formality check retained non-PASS semantic subchecks: "
                    + ",".join(sorted(nonpassing_subchecks))
                )
        formality_trace = judge.get("qa_formality") or {}
        if formality_trace.get("generator_rationale_included") is not False:
            raise SystemExit("qa_formality trace unexpectedly includes generator rationale")

for row in accepted:
    if "category" in row or "category_rationale" in row:
        raise SystemExit("accepted row unexpectedly contains category output fields")
    evaluations = (((row.get("review") or {}).get("answerability") or {}).get("evaluations") or [])
    if any("choice_uncertainty" in evaluation for evaluation in evaluations):
        raise SystemExit("accepted row retained answerability choice-logit diagnostics")

print(f"trace_check evidence_count={evidence_count}")
print(f"trace_check generation_prompt_count={len(generation_prompts)}")
print(f"trace_check qa_formality_prompt_count={len(formality_prompts)}")
print(f"trace_check evidence_groundedness_prompt_count={len(grounding_prompts)}")
print(f"trace_check answerability_prompt_count={len(answerability_prompts)}")
print(f"trace_check judge_prompt_count={len(judge_prompts)}")
print(f"trace_check accepted_count={len(accepted)}")
print(f"trace_check rejected_count={len(rejected)}")

if evidence_count < target:
    raise SystemExit(f"expected at least {target} evidence packets, found {evidence_count}")
if len(generation_prompts) < target:
    raise SystemExit(f"expected at least {target} generation prompts, found {len(generation_prompts)}")
if len(formality_prompts) < target:
    raise SystemExit(
        f"expected at least {target} qa_formality prompts, found {len(formality_prompts)}"
    )
if len(grounding_prompts) < target:
    raise SystemExit(
        "expected at least "
        f"{target} evidence_groundedness prompts, found {len(grounding_prompts)}"
    )
if len(answerability_prompts) < target:
    raise SystemExit(
        f"expected at least {target} answerability prompts, found {len(answerability_prompts)}"
    )
if len(accepted) + len(rejected) < target:
    raise SystemExit(
        f"expected at least {target} final accepted/rejected rows, found {len(accepted) + len(rejected)}"
    )
PY

echo "stage=validate_outputs"
python -m egolife_two_user_qa validate_outputs \
  --qa "${OUTDIR}/qa_mcq.jsonl" \
  --csv-output "${OUTDIR}/qa_mcq.csv" \
  --human-review-output "${OUTDIR}/human_review_sheet.md" \
  --report "${OUTDIR}/generation_report.md" \
  --strict-review

echo "outputs:"
echo "  qa=${OUTDIR}/qa_mcq.jsonl"
echo "  intermediate=${OUTDIR}/qa_mcq.intermediate.jsonl"
echo "  rejected=${OUTDIR}/qa_mcq.rejected.jsonl"
echo "  prompts=${OUTDIR}/video_first_prompts.jsonl"
echo "  report=${OUTDIR}/generation_report.md"
echo "date_end=$(date --iso-8601=seconds)"
