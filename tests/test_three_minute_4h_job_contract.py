from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = (ROOT / "hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch").read_text(
    encoding="utf-8"
)
JOB_PATH = ROOT / "hpc/qa/experiments/run_six_user_qa_3min_4h.sbatch"


def test_three_minute_wrapper_reuses_four_hour_keeper_contract() -> None:
    assert JOB_PATH.is_file()
    job = JOB_PATH.read_text(encoding="utf-8")
    assert "#SBATCH --time=04:00:00" in job
    assert 'EVIDENCE_DURATION_SECONDS="180"' in job
    assert 'PRUNING_BLOCK_SECONDS="30"' in job
    assert 'MAX_ATTEMPTS="3"' in job
    assert 'TARGET_GENERATION_GROUPS="3"' in job
    assert 'SINGLE_CANDIDATE_GROUP="0"' in job
    assert "MAX_GENERATION_SLOTS=" not in job
    assert 'CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"' in job
    assert "--nodelist" not in job and "#SBATCH -w" not in job


def test_three_minute_wrapper_enables_generator_sampling_explicitly() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")

    assert 'GENERATOR_DECODE_MODE="${GENERATOR_DECODE_MODE:-sampling}"' in job
    assert 'GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.7}"' in job
    assert 'GENERATOR_TOP_P="${GENERATOR_TOP_P:-0.9}"' in job
    assert '--generator-decode-mode "${GENERATOR_DECODE_MODE}"' in RUNTIME
    assert '--generator-temperature "${GENERATOR_TEMPERATURE}"' in RUNTIME
    assert '--generator-top-p "${GENERATOR_TOP_P}"' in RUNTIME


def test_runtime_finalizer_receives_generator_decode_values_without_literal_shell_tokens() -> None:
    finalizer = RUNTIME.split("runtime_probe_allow_zero_accepted = bool(int(sys.argv[12]))", 1)[1]
    finalizer = finalizer.split('if status == "failed":', 1)[0]
    assert '"${GENERATOR_DECODE_MODE}" "${GENERATOR_TEMPERATURE}" "${GENERATOR_TOP_P}" <<\'PY\'' in RUNTIME
    assert "generator_decode_mode = sys.argv[13]" in RUNTIME
    assert "generator_temperature = float(sys.argv[14])" in RUNTIME
    assert "generator_top_p = float(sys.argv[15])" in RUNTIME
    assert '"mode": generator_decode_mode' in finalizer
    assert '"temperature": generator_temperature' in finalizer
    assert '"top_p": generator_top_p' in finalizer
    assert 'float("${GENERATOR_TEMPERATURE}")' not in finalizer
    assert 'float("${GENERATOR_TOP_P}")' not in finalizer


def test_three_minute_wrapper_exports_project_root_before_runtime_child() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")
    assignment = job.index(
        'PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-six-user}"'
    )
    export = job.index("export PROJECT_ROOT", assignment)
    runtime = job.index(
        'bash "${PROJECT_ROOT}/hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"'
    )

    assert assignment < export < runtime


def test_runtime_defaults_remain_thirty_seconds_and_accept_overrides() -> None:
    assert 'EVIDENCE_DURATION_SECONDS="${EVIDENCE_DURATION_SECONDS:-30}"' in RUNTIME
    assert 'PRUNING_BLOCK_SECONDS="${PRUNING_BLOCK_SECONDS:-30}"' in RUNTIME
    assert '--pruning-block-seconds "${PRUNING_BLOCK_SECONDS}"' in RUNTIME
    assert '--deadline-epoch-seconds "${QA_DEADLINE_EPOCH_SECONDS}"' in RUNTIME
    assert '--attempts-output "${OUTDIR}/qa_mcq.attempts.jsonl"' in RUNTIME
    assert "--single-candidate-group" in RUNTIME
    assert '--target-generation-groups "${TARGET_GENERATION_GROUPS}"' in RUNTIME
    assert 'target_generation_groups = int(sys.argv[2])' in RUNTIME
    assert 'generation_group_count < target_generation_groups' in RUNTIME
    assert 'QWEN_MEMORY_SAFE_VIDEO_FPS="${QWEN_MEMORY_SAFE_VIDEO_FPS:-1.0}"' in RUNTIME


def test_qwen_video_preflight_receives_configured_fps_in_its_own_heredoc() -> None:
    blocks = re.findall(
        r"python - ([^\n]+) <<'PY'\n(.*?)\nPY",
        RUNTIME,
        flags=re.DOTALL,
    )
    arguments, body = next(
        (arguments, body)
        for arguments, body in blocks
        if "process_vision_info" in body
    )

    assert '"${QWEN_MEMORY_SAFE_VIDEO_FPS}"' in arguments
    assert "video_fps = float(sys.argv[2])" in body
    assert '"fps": video_fps' in body


def test_time_budget_summary_counts_all_completed_output_classes() -> None:
    assert 'row.get("status") == "time_budget_partial"' in RUNTIME
    assert "generated_count = len(accepted) + len(rejected) + len(time_budget_partials)" in RUNTIME
    assert '"objective": "time_budgeted_generation"' in RUNTIME
    assert '"max_generation_slots": max_generation_slots' in RUNTIME


def test_runtime_finalizes_partial_results_after_external_sigterm() -> None:
    assert 'trap cleanup EXIT' in RUNTIME
    assert 'trap mark_termination INT TERM' in RUNTIME
    assert 'TERMINATION_REQUESTED="0"' in RUNTIME
    assert 'GENERATION_EXIT_CODE="$?"' in RUNTIME
    assert '"${TERMINATION_REQUESTED}" "${GENERATION_EXIT_CODE}"' in RUNTIME
    assert 'termination_requested = bool(int(sys.argv[10]))' in RUNTIME
    assert 'generation_exit_code = int(sys.argv[11])' in RUNTIME
    assert '"termination_requested": termination_requested' in RUNTIME
    assert '"generation_exit_code": generation_exit_code' in RUNTIME


def test_runtime_finalizer_is_slot_aware_for_new_judge_contracts() -> None:
    assert 'by_answerability_identity = prompt_rows_by_generation_identity(' in RUNTIME
    assert 'by_segment_identity = prompt_rows_by_generation_identity(' in RUNTIME
    assert 'by_aggregation_identity = prompt_rows_by_generation_identity(' in RUNTIME
    assert 'identity = (str(row.get("generation_slot_id")), qa_id, attempt)' in RUNTIME
    assert 'accepted QA must have exactly 2 answerability calls' in RUNTIME
    assert 'set(answerability_rows) != {"speaker_only", "combined_all_six_users"}' in RUNTIME
    assert 'speaker_only_answerable' in RUNTIME
    assert 'all_six_answerable' in RUNTIME
    assert 'evidence_segment_observation' in RUNTIME
    assert 'evidence_groundedness_aggregation' in RUNTIME
    assert 'accepted QA must have 6 evidence segment observations' in RUNTIME
    assert 'accepted QA must have 1 evidence aggregation' in RUNTIME
    assert 'accepted QA must have exactly 1 answerability call' not in RUNTIME


def test_runtime_probe_can_allow_zero_accepted_without_weakening_formal_job() -> None:
    job = JOB_PATH.read_text(encoding="utf-8")
    assert 'RUNTIME_PROBE_ALLOW_ZERO_ACCEPTED="0"' in job
    assert 'RUNTIME_PROBE_ALLOW_ZERO_ACCEPTED="${RUNTIME_PROBE_ALLOW_ZERO_ACCEPTED:-0}"' in RUNTIME
    assert 'runtime_probe_allow_zero_accepted = bool(int(sys.argv[12]))' in RUNTIME
    assert 'status = "runtime_passed_no_accepted"' in RUNTIME
    assert '"runtime_probe_allow_zero_accepted": runtime_probe_allow_zero_accepted' in RUNTIME
    assert 'if status == "failed":' in RUNTIME


def test_non_time_budget_probe_marks_zero_accepted_as_runtime_only() -> None:
    assert 'elif runtime_probe_allow_zero_accepted and len(accepted) == 0 and completed_review_count > 0:' in RUNTIME
    assert 'status = "runtime_passed_no_accepted"' in RUNTIME
