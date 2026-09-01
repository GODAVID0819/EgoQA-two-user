from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch"
FORMAL_JOB = ROOT / "hpc/qa/experiments/run_six_user_qa_10min_3groups_x20.sbatch"
QWEN38_FAST_FORMAL_JOB = (
    ROOT / "hpc/qa/experiments/run_six_user_qa_10min_3groups_x20_qwen38_fast.sbatch"
)
QWEN38_FAST_FIX_FORMAL_JOB = (
    ROOT / "hpc/qa/experiments/run_six_user_qa_10min_3groups_x20_qwen38_fast_fix.sbatch"
)
QWEN38_DOWNLOAD_JOB = ROOT / "hpc/qa/experiments/download_qwen38_27b.sbatch"
RUNTIME_PATH = ROOT / "hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"
CLI_PATH = ROOT / "cli.py"


def runtime_text() -> str:
    return RUNTIME_PATH.read_text(encoding="utf-8")


def test_ten_minute_reasoning_wrapper_contract() -> None:
    job = JOB.read_text(encoding="utf-8")

    assert "#SBATCH --mem=96G" in job
    assert "#SBATCH --time=04:00:00" in job
    assert 'EVIDENCE_DURATION_SECONDS="600"' in job
    assert 'PRUNING_BLOCK_SECONDS="30"' in job
    assert 'PRUNING_MAX_CROSS_GAP_SECONDS="30"' in job
    assert 'TARGET_GENERATION_GROUPS="1"' in job
    assert 'MAX_GENERATION_SLOTS="1"' in job
    assert 'MAX_NEW_TOKENS="16384"' in job
    assert 'FORMALITY_MAX_NEW_TOKENS="2048"' in job
    assert 'SIX_USER_TEN_MINUTE_REASONING_PROFILE="1"' in job
    assert 'CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-0}"' in job
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-six-user-10min-20260829}"' in job
    assert 'QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="65536"' in job
    assert 'QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="131072"' in job
    assert "--nodelist" not in job
    assert "#SBATCH -w" not in job
    assert "#SBATCH --partition" not in job


def test_runtime_propagates_cross_gap_and_reasoning_profile() -> None:
    runtime = runtime_text()

    assert (
        'PRUNING_MAX_CROSS_GAP_SECONDS="${PRUNING_MAX_CROSS_GAP_SECONDS:-10}"'
        in runtime
    )
    assert '--pruning-max-cross-gap-seconds "${PRUNING_MAX_CROSS_GAP_SECONDS}"' in runtime
    assert 'MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"' in runtime
    assert 'FORMALITY_MAX_NEW_TOKENS="${FORMALITY_MAX_NEW_TOKENS:-2048}"' in runtime
    assert (
        'SIX_USER_TEN_MINUTE_REASONING_PROFILE="${SIX_USER_TEN_MINUTE_REASONING_PROFILE:-0}"'
        in runtime
    )
    assert '--max-new-tokens "${MAX_NEW_TOKENS}"' in runtime
    assert '--formality-max-new-tokens "${FORMALITY_MAX_NEW_TOKENS}"' in runtime
    assert "--six-user-ten-minute-reasoning-profile" in runtime
    assert 'QA_PROFILE_ARGS=(--disable-thinking)' in runtime
    assert '"pruning_max_cross_gap_seconds": float("${PRUNING_MAX_CROSS_GAP_SECONDS}")' in runtime
    assert 'ONE_CANDIDATE_PER_GROUP="${ONE_CANDIDATE_PER_GROUP:-0}"' in runtime
    assert '"one_candidate_per_generation_group": bool(int("${ONE_CANDIDATE_PER_GROUP}"))' in runtime
    assert '"${ONE_CANDIDATE_PER_GROUP}" <<' in runtime
    assert '"max_new_tokens": int("${MAX_NEW_TOKENS}")' in runtime
    assert '"formality_max_new_tokens": int("${FORMALITY_MAX_NEW_TOKENS}")' in runtime
    assert (
        '"ten_minute_reasoning_profile": bool(int("${SIX_USER_TEN_MINUTE_REASONING_PROFILE}"))'
        in runtime
    )


def test_runtime_propagates_fast_profile_and_fail_fast_review() -> None:
    runtime = runtime_text()

    assert 'SIX_USER_TEN_MINUTE_FAST_PROFILE="${SIX_USER_TEN_MINUTE_FAST_PROFILE:-0}"' in runtime
    assert 'FAIL_FAST_REVIEW="${FAIL_FAST_REVIEW:-0}"' in runtime
    assert "--six-user-ten-minute-fast-profile" in runtime
    assert "--fail-fast-review" in runtime
    assert '"ten_minute_fast_profile": bool(int("${SIX_USER_TEN_MINUTE_FAST_PROFILE}"))' in runtime
    assert '"fail_fast_review": bool(int("${FAIL_FAST_REVIEW}"))' in runtime


def test_cli_forwards_fast_profile_and_fail_fast_review_to_generation_loop() -> None:
    cli = CLI_PATH.read_text(encoding="utf-8")

    assert "six_user_ten_minute_fast_profile=args.six_user_ten_minute_fast_profile" in cli
    assert "fail_fast_review=args.fail_fast_review" in cli


def test_runtime_does_not_require_segmented_evidence_review() -> None:
    runtime = runtime_text()

    assert "expected_segments_per_user" not in runtime
    assert "evidence segment rows must use" not in runtime
    assert "accepted QA must have exactly 1 simple evidence groundedness call" in runtime
    assert "elif allow_partial and len(accepted) > 0 and completed_review_count > 0:" in runtime


def test_runtime_requires_minimum_user_set_for_every_accepted_qa() -> None:
    runtime = runtime_text()

    assert 'minimum_required_users = row.get("minimum_required_users")' in runtime
    assert "accepted QA minimum_required_users must be a non-empty ordered subset" in runtime
    assert 'gate.get("minimum_required_users") != minimum_required_users' in runtime


def test_runtime_accepts_one_simple_evidence_call_per_accepted_qa() -> None:
    runtime = runtime_text()

    assert 'by_groundedness_identity = prompt_rows_by_generation_identity(' in runtime
    assert "accepted QA must have exactly 1 simple evidence groundedness call" in runtime
    assert '"groundedness_video_count": 6' in runtime
    assert '"evidence_segment_observation_count": 0' in runtime
    assert '"evidence_groundedness_aggregation_count": 0' in runtime
    assert "accepted QA must have 6 evidence segment observations" not in runtime


def test_runtime_tracks_minimum_slots_for_every_eligible_group_speaker() -> None:
    runtime = runtime_text()

    assert "selected_by_group.setdefault" not in runtime
    assert "completed_slots_by_group_and_speaker" in runtime
    assert 'minimum_slots_per_eligible_speaker = 2' in runtime
    assert '"speaker_slot_target_reached": speaker_slot_target_reached' in runtime


def test_runtime_propagates_ten_minute_memory_safe_limits() -> None:
    runtime = runtime_text()

    assert (
        'QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="${QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS:-65536}"'
        in runtime
    )
    assert (
        'QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="${QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS:-32768}"'
        in runtime
    )
    assert '--max-image-pixels "${QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS}"' in runtime


def test_runtime_cleanup_preserves_original_failure_status() -> None:
    runtime = runtime_text()

    assert "cleanup_status=$?" in runtime
    assert 'return "${cleanup_status}"' in runtime


def test_formal_wrapper_targets_three_distinct_groups_and_twenty_slots_each() -> None:
    job = FORMAL_JOB.read_text(encoding="utf-8")

    assert "#SBATCH --time=2-00:00:00" in job
    assert 'EVIDENCE_TARGET="3"' in job
    assert 'TARGET_GENERATION_GROUPS="3"' in job
    assert 'EXPECTED_QA_PER_GROUP="20"' in job
    assert 'ONE_CANDIDATE_PER_GROUP="0"' in job
    assert 'MAX_GENERATION_SLOTS="60"' in job
    assert 'MAX_NEW_TOKENS="16384"' in job
    assert 'QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="65536"' in job
    assert 'QA_TIME_BUDGET_MODE="1"' in job
    assert 'SIX_USER_TEN_MINUTE_REASONING_PROFILE="1"' in job
    assert 'CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-0}"' in job
    assert "--nodelist" not in job
    assert "#SBATCH -w" not in job


def test_qwen38_fast_formal_wrapper_keeps_full_three_by_twenty_contract() -> None:
    job = QWEN38_FAST_FORMAL_JOB.read_text(encoding="utf-8")

    assert '#SBATCH --job-name=egoqa_6u_10min_q38_fast' in job
    assert '#SBATCH --account=torch_pr_674_tandon_advanced' in job
    assert '#SBATCH --gres=gpu:1' in job
    assert '#SBATCH --constraint=h100' in job
    assert 'RUN_MODE="six_user_qa_10min_3groups_x20_qwen38_fast"' in job
    assert 'MODEL_DIR="/scratch/xl6775/models/Qwen3.8-27B"' in job
    assert 'EVIDENCE_DURATION_SECONDS="600"' in job
    assert 'TARGET_GENERATION_GROUPS="3"' in job
    assert 'EXPECTED_QA_PER_GROUP="20"' in job
    assert 'ONE_CANDIDATE_PER_GROUP="0"' in job
    assert 'MAX_GENERATION_SLOTS="60"' in job
    assert 'MAX_ATTEMPTS="3"' in job
    assert 'QWEN_MEMORY_SAFE_VIDEO_FPS="0.25"' in job
    assert 'QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="65536"' in job
    assert 'SIX_USER_TEN_MINUTE_FAST_PROFILE="1"' in job
    assert 'FAIL_FAST_REVIEW="1"' in job
    assert 'SIX_USER_TEN_MINUTE_REASONING_PROFILE="0"' in job
    assert "--nodelist" not in job
    assert "#SBATCH -w" not in job


def test_qwen38_download_job_is_cpu_only_and_audits_all_model_parts() -> None:
    job = QWEN38_DOWNLOAD_JOB.read_text(encoding="utf-8")

    assert '#SBATCH --job-name=download_qwen38_27b' in job
    assert '#SBATCH --account=torch_pr_674_tandon_advanced' in job
    assert '#SBATCH --cpus-per-task=4' in job
    assert '#SBATCH --mem=16G' in job
    assert '#SBATCH --gres' not in job
    assert 'MODEL_DIR="/scratch/xl6775/models/Qwen3.8-27B"' in job
    assert 'HF_CLI="/scratch/xl6775/conda/envs/qwen3vl-smoke/bin/hf"' in job
    assert 'hf download' not in job
    assert '"${HF_CLI}" download Qwen/Qwen3.8-27B' in job
    assert 'expected_shards = 18' in job
    assert 'AutoConfig.from_pretrained(model_dir, local_files_only=True)' in job
    assert 'AutoProcessor.from_pretrained(model_dir, local_files_only=True)' in job
    assert 'download_result.json' in job
    assert 'export HOME="${JOB_SCRATCH_ROOT}/home"' in job
    assert '--nodelist' not in job


def test_qwen38_fast_fix_wrapper_uses_new_isolated_project_root() -> None:
    job = QWEN38_FAST_FIX_FORMAL_JOB.read_text(encoding="utf-8")

    assert 'RUN_MODE="six_user_qa_10min_3groups_x20_qwen38_fast_fix"' in job
    assert 'PROJECT_ROOT="${PROJECT_ROOT:-/scratch/xl6775/projects/EgoQA-two-user-six-user-10min-speed-qwen38-fix-20260901}"' in job
    assert 'MODEL_DIR="/scratch/xl6775/models/Qwen3.8-27B"' in job
    assert 'SIX_USER_TEN_MINUTE_FAST_PROFILE="1"' in job
    assert 'FAIL_FAST_REVIEW="1"' in job
    assert 'ONE_CANDIDATE_PER_GROUP="0"' in job
    assert 'MAX_GENERATION_SLOTS="60"' in job
    assert 'MAX_ATTEMPTS="3"' in job
    assert 'EVIDENCE_DURATION_SECONDS="600"' in job
    assert "#SBATCH --nodelist" not in job
    assert "#SBATCH -w" not in job
