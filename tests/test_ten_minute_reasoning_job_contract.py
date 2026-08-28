from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = ROOT / "hpc/qa/experiments/run_six_user_qa_10min_reasoning.sbatch"
FORMAL_JOB = ROOT / "hpc/qa/experiments/run_six_user_qa_10min_3groups_x20.sbatch"
RUNTIME_PATH = ROOT / "hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"


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
    assert 'MAX_NEW_TOKENS="8192"' in job
    assert 'FORMALITY_MAX_NEW_TOKENS="2048"' in job
    assert 'SIX_USER_TEN_MINUTE_REASONING_PROFILE="1"' in job
    assert 'QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="131072"' in job
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
    assert '"max_new_tokens": int("${MAX_NEW_TOKENS}")' in runtime
    assert '"formality_max_new_tokens": int("${FORMALITY_MAX_NEW_TOKENS}")' in runtime
    assert (
        '"ten_minute_reasoning_profile": bool(int("${SIX_USER_TEN_MINUTE_REASONING_PROFILE}"))'
        in runtime
    )


def test_runtime_uses_duration_derived_segment_count() -> None:
    runtime = runtime_text()

    assert "expected_segments_per_user = int(evidence_duration_seconds / 30)" in runtime
    assert "len(item.get(\"video_paths\") or []) != expected_segments_per_user" in runtime
    assert "len(item.get(\"segments\") or []) != expected_segments_per_user" in runtime
    assert "elif allow_partial and len(accepted) > 0 and completed_review_count > 0:" in runtime


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


def test_formal_wrapper_targets_three_distinct_groups_and_twenty_slots_each() -> None:
    job = FORMAL_JOB.read_text(encoding="utf-8")

    assert 'EVIDENCE_TARGET="3"' in job
    assert 'TARGET_GENERATION_GROUPS="3"' in job
    assert 'EXPECTED_QA_PER_GROUP="20"' in job
    assert 'MAX_GENERATION_SLOTS="60"' in job
    assert 'QA_TIME_BUDGET_MODE="1"' in job
    assert 'SIX_USER_TEN_MINUTE_REASONING_PROFILE="1"' in job
    assert "--nodelist" not in job
    assert "#SBATCH -w" not in job
