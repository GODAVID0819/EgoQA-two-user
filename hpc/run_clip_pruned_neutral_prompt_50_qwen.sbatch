#!/usr/bin/env bash
#SBATCH --job-name=egolife_decodecmp50
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

# ARCHIVED LEGACY EXPERIMENT: this launcher validates 1/2/3 quality-score
# rubrics that the production judges no longer request or retain. Its body is
# preserved below as a historical record. Use
# run_clip_pruned_sampling_neutral_pf_50.sbatch for the maintained binary path.
echo "error=archived_scored_judge_experiment_not_a_production_pipeline" >&2
exit 2

PROJECT_ROOT="${QWEN3VL_PROJECT_ROOT:-/scratch/${USER}/Long-video-understanding-clip}"

# Reuse the same CLIP-pruned video-pair packets as the 50-pair decode ablation.
EVIDENCE_ROOT="${EGOLIFE2U_EVIDENCE_ROOT:-egolife_two_user_qa/outputs/clip_pruned_benchmark_100_floor_resample_2}"
EVIDENCE_PATH="${EGOLIFE2U_EVIDENCE_PATH:-${EVIDENCE_ROOT}/evidence_pruned_pairs.jsonl}"

OUTDIR="${EGOLIFE2U_OUTPUT_ROOT:-egolife_two_user_qa/outputs/clip_pruned_neutral_decode_comparison_50}"
CACHE_DIR="${EGOLIFE2U_CACHE_DIR:-/scratch/${USER}/egolife_two_user_qa_cache}"

TARGET_COUNT="${EGOLIFE2U_TARGET_COUNT:-50}"
EVAL_PACKET_COUNT="${EGOLIFE2U_EVAL_PACKET_COUNT:-${TARGET_COUNT}}"

QWEN_MODEL_ID="${QWEN_MODEL_ID:-Qwen/Qwen3.6-27B}"
GENERATION_MODE="${GENERATION_MODE:-baseline}"
# Only the generator changes decoding policy. All review calls remain greedy,
# so their 1/2/3 entropy distributions are evaluated on the same scale.
GENERATOR_DECODE_MODES="${GENERATOR_DECODE_MODES:-greedy sampling}"
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
echo "eval_packet_count=${EVAL_PACKET_COUNT}"
echo "qwen_model_id=${QWEN_MODEL_ID}"
echo "generator_backend=transformers-local"
echo "judge_backend=transformers-local"
echo "runner_sharing=generator_and_all_review_branches_reuse_one_qwen_runner"
echo "quality_evaluation=softmax_over_1_2_3_choice_logits_then_normalized_entropy"
echo "generation_mode=${GENERATION_MODE}"
echo "question_types=neutral"
echo "generator_decode_modes=${GENERATOR_DECODE_MODES}"
echo "judge_decode_mode=greedy"
echo "judge_temperature=unscaled_logits"
echo "max_attempts=${MAX_ATTEMPTS}"
echo "max_new_tokens=${MAX_NEW_TOKENS}"
echo "disable_thinking=${DISABLE_THINKING}"
echo "resume_generation=${RESUME_GENERATION}"
echo "cuda_keeper_enable=${CUDA_KEEPER_ENABLE}"
echo "cuda_keeper_script=${CUDA_KEEPER_SCRIPT}"
echo "cuda_keeper_threshold=${CUDA_KEEPER_THRESHOLD}"
echo "cuda_keeper_gpus=${CUDA_KEEPER_GPUS}"
echo "cuda_keeper_reserve=${CUDA_KEEPER_RESERVE}"
echo "cuda_keeper_auto_install=${CUDA_KEEPER_AUTO_INSTALL}"
echo "benchmark_contract=reuse_same_first_50_clip_pruned_packets_neutral_prompt"
echo "media_routing=generator_uses_existing_pruned_videos;judges_and_answerability_use_full_30s_originals"
python --version
nvidia-smi || true

if [[ ! -s "${EVIDENCE_PATH}" ]]; then
  echo "error=evidence_file_missing path=${EVIDENCE_PATH}" >&2
  exit 1
fi

echo "stage=preflight_choice_logit_entropy"
python - <<'PY'
import math

from egolife_two_user_qa.qwen3vl_runner import Qwen3VLTransformersRunner
from egolife_two_user_qa.video_qa_loop import quality_uncertainty_from_choice_logits

if not hasattr(Qwen3VLTransformersRunner, "generate_with_choice_logits"):
    raise SystemExit("Qwen runner does not expose generate_with_choice_logits; sync the entropy-enabled code")

uniform = quality_uncertainty_from_choice_logits(
    {
        "available": True,
        "choice_logits": {"1": 0.0, "2": 0.0, "3": 0.0},
        "weight_type": "logit",
        "generated_choice": "2",
        "token_index": 0,
    }
)
if uniform.get("available") is not True:
    raise SystemExit(f"entropy preflight unavailable: {uniform}")
if not math.isclose(float(uniform["entropy_nats"]), math.log(3.0), rel_tol=1e-7):
    raise SystemExit(f"uniform entropy should equal ln(3): {uniform}")
if not math.isclose(float(uniform["normalized_entropy"]), 1.0, rel_tol=1e-7):
    raise SystemExit(f"uniform normalized entropy should equal 1: {uniform}")
print("choice_logit_entropy_preflight=ok")
PY

if [[ "${CUDA_KEEPER_ENABLE}" == "1" ]]; then
  if [[ -s "${CUDA_KEEPER_SCRIPT}" ]]; then
    echo "stage=preflight_cuda_keeper_imports"
    if ! python - <<'PY'
import importlib.util
missing = [name for name in ("torch", "pynvml") if importlib.util.find_spec(name) is None]
optional_missing = [name for name in ("psutil",) if importlib.util.find_spec(name) is None]
if missing or optional_missing:
    print("missing_required=" + ",".join(missing))
    print("missing_optional=" + ",".join(optional_missing))
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

read -r -a decode_modes <<< "${GENERATOR_DECODE_MODES}"
if [[ "${#decode_modes[@]}" -ne 2 ]]; then
  echo "error=GENERATOR_DECODE_MODES_must_contain_exactly_greedy_and_sampling value=${GENERATOR_DECODE_MODES}" >&2
  exit 1
fi
seen_greedy=0
seen_sampling=0
for mode in "${decode_modes[@]}"; do
  case "${mode}" in
    greedy) seen_greedy=$((seen_greedy + 1)) ;;
    sampling) seen_sampling=$((seen_sampling + 1)) ;;
    *)
      echo "error=unsupported_generator_decode_mode value=${mode}" >&2
      exit 1
      ;;
  esac
done
if [[ "${seen_greedy}" -ne 1 || "${seen_sampling}" -ne 1 ]]; then
  echo "error=GENERATOR_DECODE_MODES_must_include_greedy_and_sampling_once value=${GENERATOR_DECODE_MODES}" >&2
  exit 1
fi

for GENERATOR_DECODE_MODE in "${decode_modes[@]}"; do
  decode_args=(--generator-decode-mode "${GENERATOR_DECODE_MODE}")
  if [[ "${GENERATOR_DECODE_MODE}" == "sampling" ]]; then
    decode_args+=(--generator-temperature "${SAMPLING_TEMPERATURE}")
    decode_args+=(--generator-top-p "${SAMPLING_TOP_P}")
    if [[ -n "${SAMPLING_TOP_K}" ]]; then
      decode_args+=(--generator-top-k "${SAMPLING_TOP_K}")
    fi
  fi

QA_DIR="${OUTDIR}/${GENERATION_MODE}_neutral_${GENERATOR_DECODE_MODE}"
mkdir -p "${QA_DIR}"

echo "stage=generate_video_qa_loop question_types=neutral decode_mode=${GENERATOR_DECODE_MODE}"
python -m egolife_two_user_qa generate_video_qa_loop \
  --evidence "${EVIDENCE_PATH}" \
  --output "${QA_DIR}/qa_mcq.jsonl" \
  --prompts-output "${QA_DIR}/video_first_prompts.jsonl" \
  --rejected-output "${QA_DIR}/qa_mcq.rejected.jsonl" \
  --intermediate-output "${QA_DIR}/qa_mcq.intermediate.jsonl" \
  --target-count "${EVAL_PACKET_COUNT}" \
  --max-attempts "${MAX_ATTEMPTS}" \
  --backend transformers-local \
  --model-id "${QWEN_MODEL_ID}" \
  --dtype bfloat16 \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --generation-mode "${GENERATION_MODE}" \
  --fixed-question-type-schedule \
  --question-types neutral \
  "${decode_args[@]}" \
  "${thinking_args[@]}" \
  "${resume_args[@]}"

echo "stage=verify_neutral_generation_traces"
python - "${EVIDENCE_PATH}" "${QA_DIR}" "${EVAL_PACKET_COUNT}" "${GENERATOR_DECODE_MODE}" <<'PY'
import json
import sys
from pathlib import Path

evidence_path = Path(sys.argv[1])
qa_dir = Path(sys.argv[2])
target = int(sys.argv[3])
decode_mode = sys.argv[4]

forbidden_prompt_fragments = [
    'question_type must be "neutral"',
    'Target question_type: "neutral"',
    "Create a commonality question",
    "Create a difference question",
    "commonality or difference subtype",
]

def count_lines(path):
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())

def inspect_generation_prompts(path):
    if not path.exists():
        return 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("stage") != "generation":
                continue
            total += 1
            prompt = row.get("prompt") or ""
            mode = row.get("generator_decode", {}).get("mode")
            if mode != decode_mode:
                raise SystemExit(f"decode_mode mismatch in prompt trace: expected {decode_mode}, got {mode}")
            for fragment in forbidden_prompt_fragments:
                if fragment in prompt:
                    raise SystemExit(f"neutral prompt contains forbidden fragment: {fragment}")
    return total

def inspect_judge_prompts(path):
    counts = {
        "qa_formality_judge": 0,
        "evidence_groundedness_judge": 0,
    }
    required_fragments = {
        "qa_formality_judge": [
            "qa_formality quality_score rubric",
            "direct_name_leakage",
            "quality_reason",
            "final_quality_score",
            "must be the final field",
        ],
        "evidence_groundedness_judge": [
            "evidence_groundedness quality_score rubric",
            "quality_reason",
            "final_quality_score",
            "must be the final field",
        ],
    }
    if not path.exists():
        return counts
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            stage = row.get("stage")
            if stage not in counts:
                continue
            counts[stage] += 1
            prompt = row.get("prompt") or ""
            if "quality_reward" in prompt:
                raise SystemExit(f"{stage} prompt still asks for quality_reward")
            for fragment in required_fragments[stage]:
                if fragment not in prompt:
                    raise SystemExit(f"{stage} prompt missing required fragment: {fragment}")
    return counts

def inspect_final_rows(path, *, require_entropy):
    if not path.exists():
        return 0, 0
    total = 0
    rows_with_entropy = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            total += 1
            row = json.loads(line)
            question_type = row.get("question_type")
            if question_type != "neutral":
                raise SystemExit(f"final row question_type mismatch: expected neutral, got {question_type}")
            judger = (row.get("review") or {}).get("judger") or {}
            checks = judger.get("checks") if isinstance(judger, dict) else {}
            row_has_all_entropy = True
            if isinstance(checks, dict):
                for check_name in ("qa_formality", "evidence_groundedness"):
                    check = checks.get(check_name)
                    if not isinstance(check, dict):
                        row_has_all_entropy = False
                        continue
                    score = check.get("quality_score")
                    flag = check.get("quality_flag")
                    reason = str(check.get("quality_reason") or "").strip()
                    expected_flags = {
                        1: "1_weak_or_reject",
                        2: "2_acceptable",
                        3: "3_strong",
                    }
                    if score not in expected_flags:
                        raise SystemExit(f"{path.name} row missing valid {check_name}.quality_score")
                    if flag != expected_flags[score]:
                        raise SystemExit(f"{path.name} row has mismatched {check_name}.quality_flag")
                    if not reason:
                        raise SystemExit(f"{path.name} row missing {check_name}.quality_reason")
                    if "quality_reward" in check:
                        raise SystemExit(f"{path.name} row still contains {check_name}.quality_reward")
                    uncertainty = check.get("quality_uncertainty")
                    if not isinstance(uncertainty, dict) or uncertainty.get("available") is not True:
                        row_has_all_entropy = False
                        if require_entropy:
                            detail = uncertainty.get("reason") if isinstance(uncertainty, dict) else "missing object"
                            raise SystemExit(
                                f"{path.name} row missing available {check_name} choice-logit entropy: {detail}"
                            )
                        continue
                    probabilities = uncertainty.get("probabilities")
                    if not isinstance(probabilities, dict):
                        raise SystemExit(f"{path.name} row missing {check_name} score probabilities")
                    try:
                        probability_values = [float(probabilities[str(value)]) for value in (1, 2, 3)]
                    except (KeyError, TypeError, ValueError) as exc:
                        raise SystemExit(
                            f"{path.name} row has invalid {check_name} probabilities: {probabilities}"
                        ) from exc
                    if any(value < 0.0 or value > 1.0 for value in probability_values):
                        raise SystemExit(f"{path.name} row has out-of-range {check_name} probabilities")
                    if abs(sum(probability_values) - 1.0) > 1e-5:
                        raise SystemExit(f"{path.name} row {check_name} probabilities do not sum to one")
                    entropy_nats = uncertainty.get("entropy_nats")
                    normalized_entropy = uncertainty.get("normalized_entropy")
                    if not isinstance(entropy_nats, (int, float)) or not 0.0 <= float(entropy_nats) <= 1.098613:
                        raise SystemExit(f"{path.name} row has invalid {check_name}.entropy_nats")
                    if not isinstance(normalized_entropy, (int, float)) or not 0.0 <= float(normalized_entropy) <= 1.0:
                        raise SystemExit(f"{path.name} row has invalid {check_name}.normalized_entropy")
                    argmax_score = uncertainty.get("argmax_score")
                    if argmax_score not in (1, 2, 3):
                        raise SystemExit(f"{path.name} row has invalid {check_name}.argmax_score")
                    if check.get("quality_score_source") == "choice_logits_argmax" and score != argmax_score:
                        raise SystemExit(f"{path.name} row {check_name} score does not match logit argmax")
            else:
                row_has_all_entropy = False
            summary = judger.get("quality_uncertainty_summary") if isinstance(judger, dict) else None
            if require_entropy:
                if not isinstance(summary, dict) or summary.get("available") is not True:
                    raise SystemExit(f"{path.name} row missing available merged entropy summary")
                entropy_by_check = summary.get("normalized_entropy_by_check")
                if not isinstance(entropy_by_check, dict) or any(
                    name not in entropy_by_check for name in ("qa_formality", "evidence_groundedness")
                ):
                    raise SystemExit(f"{path.name} row has incomplete merged entropy summary")
            if row_has_all_entropy:
                rows_with_entropy += 1
    return total, rows_with_entropy

evidence_count = count_lines(evidence_path)
prompts_path = qa_dir / "video_first_prompts.jsonl"
generation_prompt_count = inspect_generation_prompts(prompts_path)
judge_prompt_counts = inspect_judge_prompts(prompts_path)
accepted_count, accepted_entropy_count = inspect_final_rows(
    qa_dir / "qa_mcq.jsonl",
    require_entropy=True,
)
rejected_count, rejected_entropy_count = inspect_final_rows(
    qa_dir / "qa_mcq.rejected.jsonl",
    require_entropy=False,
)

print(f"trace_check evidence_count={evidence_count}")
print(f"trace_check generation_prompt_count={generation_prompt_count}")
for stage, count in sorted(judge_prompt_counts.items()):
    print(f"trace_check {stage}_count={count}")
print(f"trace_check accepted_count={accepted_count}")
print(f"trace_check rejected_count={rejected_count}")
print(f"trace_check accepted_entropy_count={accepted_entropy_count}")
print(f"trace_check rejected_entropy_count={rejected_entropy_count}")

if evidence_count < target:
    raise SystemExit(f"expected at least {target} evidence packets, found {evidence_count}")
if generation_prompt_count < target:
    raise SystemExit(f"expected at least {target} generation prompt traces, found {generation_prompt_count}")
if sum(judge_prompt_counts.values()) == 0:
    raise SystemExit("expected judge prompt traces with quality scoring rubrics, found none")
if accepted_count + rejected_count < target:
    raise SystemExit(
        f"expected at least {target} final accepted/rejected question-answer rows, found {accepted_count + rejected_count}"
    )
PY

echo "stage=validate_outputs"
python -m egolife_two_user_qa validate_outputs \
  --qa "${QA_DIR}/qa_mcq.jsonl" \
  --csv-output "${QA_DIR}/qa_mcq.csv" \
  --human-review-output "${QA_DIR}/human_review_sheet.md" \
  --report "${QA_DIR}/generation_report.md" \
  --strict-review

echo "mode_outputs decode_mode=${GENERATOR_DECODE_MODE}:"
echo "  qa=${QA_DIR}/qa_mcq.jsonl"
echo "  intermediate=${QA_DIR}/qa_mcq.intermediate.jsonl"
echo "  rejected=${QA_DIR}/qa_mcq.rejected.jsonl"
echo "  prompts=${QA_DIR}/video_first_prompts.jsonl"
echo "  report=${QA_DIR}/generation_report.md"
done

echo "stage=compare_greedy_vs_sampling_entropy"
python - "${OUTDIR}" "${GENERATION_MODE}" "${EVAL_PACKET_COUNT}" "${SAMPLING_TEMPERATURE}" "${SAMPLING_TOP_P}" "${SAMPLING_TOP_K}" <<'PY'
import json
import math
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
generation_mode = sys.argv[2]
target = int(sys.argv[3])
sampling_temperature = float(sys.argv[4])
sampling_top_p = float(sys.argv[5])
sampling_top_k = sys.argv[6] or None
modes = ("greedy", "sampling")
check_names = ("qa_formality", "evidence_groundedness")


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values):
    return sum(values) / len(values) if values else None


def rounded(value):
    return round(float(value), 8) if value is not None else None


def accepted_item_metrics(row):
    judger = (row.get("review") or {}).get("judger") or {}
    checks = judger.get("checks") or {}
    per_check = {}
    for check_name in check_names:
        check = checks.get(check_name) or {}
        uncertainty = check.get("quality_uncertainty") or {}
        if uncertainty.get("available") is not True:
            raise SystemExit(
                f"accepted row {row.get('evidence_id')} lacks {check_name} entropy: "
                f"{uncertainty.get('reason')}"
            )
        probabilities = uncertainty.get("probabilities") or {}
        try:
            expected_quality = sum(score * float(probabilities[str(score)]) for score in (1, 2, 3))
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"accepted row {row.get('evidence_id')} has invalid {check_name} probabilities"
            ) from exc
        normalized_entropy = float(uncertainty["normalized_entropy"])
        quality_01 = (expected_quality - 1.0) / 2.0
        confidence = 1.0 - normalized_entropy
        per_check[check_name] = {
            "expected_quality": expected_quality,
            "normalized_entropy": normalized_entropy,
            "confidence": confidence,
            "entropy_adjusted_quality": quality_01 * confidence,
        }
    return {
        "expected_quality": mean([value["expected_quality"] for value in per_check.values()]),
        "normalized_entropy": mean([value["normalized_entropy"] for value in per_check.values()]),
        "entropy_adjusted_quality": mean(
            [value["entropy_adjusted_quality"] for value in per_check.values()]
        ),
        "per_check": per_check,
    }


def summarize_mode(mode):
    qa_dir = output_root / f"{generation_mode}_neutral_{mode}"
    accepted = read_jsonl(qa_dir / "qa_mcq.jsonl")
    rejected = read_jsonl(qa_dir / "qa_mcq.rejected.jsonl")
    attempted = len(accepted) + len(rejected)
    if attempted < target:
        raise SystemExit(f"{mode} produced only {attempted} final rows; expected at least {target}")

    accepted_metrics = []
    utility_by_evidence = {}
    for row in rejected:
        evidence_id = str(row.get("evidence_id") or row.get("qa_id") or "")
        utility_by_evidence[evidence_id] = 0.0
    for row in accepted:
        item = accepted_item_metrics(row)
        accepted_metrics.append(item)
        evidence_id = str(row.get("evidence_id") or row.get("qa_id") or "")
        utility_by_evidence[evidence_id] = float(item["entropy_adjusted_quality"])

    per_check = {}
    for check_name in check_names:
        values = [item["per_check"][check_name] for item in accepted_metrics]
        per_check[check_name] = {
            "mean_expected_quality": rounded(mean([value["expected_quality"] for value in values])),
            "mean_normalized_entropy": rounded(mean([value["normalized_entropy"] for value in values])),
            "mean_entropy_adjusted_quality": rounded(
                mean([value["entropy_adjusted_quality"] for value in values])
            ),
        }

    accepted_adjusted = [float(item["entropy_adjusted_quality"]) for item in accepted_metrics]
    return {
        "mode": mode,
        "attempted_count": attempted,
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "acceptance_rate": rounded(len(accepted) / attempted if attempted else 0.0),
        "mean_expected_quality_1_to_3_accepted": rounded(
            mean([item["expected_quality"] for item in accepted_metrics])
        ),
        "mean_normalized_entropy_accepted": rounded(
            mean([item["normalized_entropy"] for item in accepted_metrics])
        ),
        "mean_entropy_adjusted_quality_accepted": rounded(mean(accepted_adjusted)),
        "overall_entropy_adjusted_yield": rounded(
            sum(accepted_adjusted) / attempted if attempted else 0.0
        ),
        "accepted_entropy_coverage": rounded(
            len(accepted_metrics) / len(accepted) if accepted else 1.0
        ),
        "per_check": per_check,
        "utility_by_evidence": utility_by_evidence,
    }


summaries = {mode: summarize_mode(mode) for mode in modes}
greedy = summaries["greedy"]
sampling = summaries["sampling"]
metric_name = "overall_entropy_adjusted_yield"
delta = float(sampling[metric_name]) - float(greedy[metric_name])
tie_tolerance = 1e-4
if abs(delta) <= tie_tolerance:
    winner = "tie"
    verdict = "No meaningful winner: entropy-adjusted yields are effectively tied."
elif delta > 0:
    winner = "sampling"
    verdict = "Sampling has the higher entropy-adjusted accepted-question yield."
else:
    winner = "greedy"
    verdict = "Greedy has the higher entropy-adjusted accepted-question yield."

common_ids = sorted(
    set(greedy["utility_by_evidence"]) & set(sampling["utility_by_evidence"])
)
paired_wins = {"greedy": 0, "sampling": 0, "tie": 0}
for evidence_id in common_ids:
    greedy_utility = float(greedy["utility_by_evidence"][evidence_id])
    sampling_utility = float(sampling["utility_by_evidence"][evidence_id])
    if math.isclose(greedy_utility, sampling_utility, abs_tol=tie_tolerance):
        paired_wins["tie"] += 1
    elif greedy_utility > sampling_utility:
        paired_wins["greedy"] += 1
    else:
        paired_wins["sampling"] += 1

for summary in summaries.values():
    summary.pop("utility_by_evidence", None)

report = {
    "comparison": "greedy_vs_sampling_with_fixed_greedy_qwen_judges",
    "same_evidence_packet_target": target,
    "judge_decode_mode": "greedy",
    "judge_entropy_temperature": "unscaled logits",
    "sampling_config": {
        "temperature": sampling_temperature,
        "top_p": sampling_top_p,
        "top_k": sampling_top_k,
    },
    "metric_definition": {
        "expected_quality": "sum(score * p(score)) over score choices 1, 2, and 3",
        "entropy_adjusted_quality": "((expected_quality - 1) / 2) * (1 - normalized_entropy)",
        "overall_entropy_adjusted_yield": (
            "sum of accepted-item entropy_adjusted_quality divided by all attempted packets; "
            "rejected items contribute zero"
        ),
    },
    "results": summaries,
    "paired_common_packet_count": len(common_ids),
    "paired_wins": paired_wins,
    "winner": winner,
    "winner_metric": metric_name,
    "sampling_minus_greedy": rounded(delta),
    "verdict": verdict,
    "caution": (
        "Entropy measures judge decisiveness, not correctness. This verdict combines expected quality, "
        "entropy, and acceptance rate and should be confirmed with additional sampling seeds or human review."
    ),
}

json_path = output_root / "greedy_vs_sampling_entropy_comparison.json"
json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Greedy vs Sampling: Entropy Comparison",
    "",
    f"**Verdict:** {verdict}",
    "",
    "| Mode | Accepted | Acceptance rate | Expected quality | Normalized entropy | Adjusted quality | Overall adjusted yield |",
    "|---|---:|---:|---:|---:|---:|---:|",
]
for mode in modes:
    row = summaries[mode]
    lines.append(
        f"| {mode} | {row['accepted_count']}/{row['attempted_count']} | "
        f"{row['acceptance_rate']:.4f} | "
        f"{row['mean_expected_quality_1_to_3_accepted'] or 0.0:.4f} | "
        f"{row['mean_normalized_entropy_accepted'] or 0.0:.4f} | "
        f"{row['mean_entropy_adjusted_quality_accepted'] or 0.0:.4f} | "
        f"{row['overall_entropy_adjusted_yield']:.4f} |"
    )
lines.extend(
    [
        "",
        f"Paired packet wins: greedy={paired_wins['greedy']}, sampling={paired_wins['sampling']}, "
        f"ties={paired_wins['tie']} across {len(common_ids)} common packets.",
        "",
        "The generator temperature differs between conditions, but all judges use greedy decoding and "
        "entropy from unscaled 1/2/3 logits. Therefore temperature affects entropy only indirectly by "
        "changing the generated question.",
        "",
        report["caution"],
    ]
)
markdown_path = output_root / "greedy_vs_sampling_entropy_comparison.md"
markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"comparison_json={json_path}")
print(f"comparison_markdown={markdown_path}")
print(f"winner={winner}")
print(f"verdict={verdict}")
PY

echo "comparison_outputs:"
echo "  reused_evidence=${EVIDENCE_PATH}"
echo "  source_summary=${EVIDENCE_ROOT}/clip_pruned_pairs/group_relative_clip_summary.json"
echo "  comparison_json=${OUTDIR}/greedy_vs_sampling_entropy_comparison.json"
echo "  comparison_markdown=${OUTDIR}/greedy_vs_sampling_entropy_comparison.md"
echo "  cuda_keeper_log=${OUTDIR}/cuda_keeper_${SLURM_JOB_ID:-local}.log"
echo "date_done=$(date --iso-8601=seconds)"
