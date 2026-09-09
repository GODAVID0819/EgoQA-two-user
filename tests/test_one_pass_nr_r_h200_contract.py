from pathlib import Path

import egolife_two_user_qa.video_qa_loop as video_qa_loop


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "hpc/qa/smoke/run_six_user_qa_runtime_probe.sbatch"
SMOKE = ROOT / "hpc/qa/experiments/run_six_user_qa_one_pass_smoke_h200.sbatch"
NR = ROOT / "hpc/qa/experiments/run_six_user_qa_one_pass_nr_h200.sbatch"
R = ROOT / "hpc/qa/experiments/run_six_user_qa_one_pass_r_h200.sbatch"
COMMON = ROOT / "hpc/qa/experiments/run_six_user_qa_one_pass_ab_h200_common.sh"


def test_nr_profile_matches_r_finalizer_budgets_without_reasoning() -> None:
    factory = getattr(video_qa_loop, "six_user_one_pass_nr_profiles", None)
    assert callable(factory), "缺少 six_user_one_pass_nr_profiles"
    nr = factory()
    reasoning = video_qa_loop.six_user_one_pass_profiles()

    assert "generator_reasoning" not in nr
    assert "generator_finalizer" not in nr
    assert nr["generator"].disable_thinking is True
    assert nr["generator"].max_new_tokens == reasoning["generator_finalizer"].max_new_tokens
    pairs = {
        "evidence_groundedness": "evidence_groundedness_finalizer",
        "speaker_only_answerability": "speaker_only_answerability_finalizer",
        "all_six_answerability": "all_six_answerability_finalizer",
        "minimum_set_answerability": "minimum_set_answerability_finalizer",
    }
    for nr_name, r_name in pairs.items():
        assert nr[nr_name].disable_thinking is True
        assert nr[nr_name].max_new_tokens == reasoning[r_name].max_new_tokens


def test_runtime_selects_explicit_one_pass_nr_or_r_profile() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    assert 'ONE_PASS_REASONING_MODE="${ONE_PASS_REASONING_MODE:-r}"' in text
    assert '"nr") QA_PROFILE_ARGS=(--six-user-one-pass-nr-profile) ;;' in text
    assert '"r") QA_PROFILE_ARGS=(--six-user-one-pass-profile) ;;' in text
    assert '"one_pass_reasoning_mode": "${ONE_PASS_REASONING_MODE}"' in text
    assert '--expected-slots "${EXPECTED_TOTAL_SLOTS}"' in text
    assert "--allow-evidence-superset" in text


def test_formal_nr_and_r_wrappers_share_resources_and_scope() -> None:
    nr = NR.read_text(encoding="utf-8")
    reasoning = R.read_text(encoding="utf-8")
    for text in (nr, reasoning):
        assert "#SBATCH --account=torch_pr_674_tandon_advanced" in text
        assert "#SBATCH --cpus-per-task=16" in text
        assert "#SBATCH --gres=gpu:1" in text
        assert "#SBATCH --constraint=h200" in text
        assert "#SBATCH --mem=320G" in text
        assert "#SBATCH --time=14:00:00" in text
        assert "--partition" not in text
        assert "--nodelist" not in text
        assert (
            'source "${PROJECT_ROOT}/hpc/qa/experiments/'
            'run_six_user_qa_one_pass_ab_h200_common.sh"' in text
        )
        assert 'SCRIPT_DIR="$(cd --' not in text
    normalized_nr = nr.replace("one_pass_nr_3x10_h200", "RUN_LABEL").replace(
        'ONE_PASS_REASONING_MODE="nr"', 'ONE_PASS_REASONING_MODE="MODE"'
    )
    normalized_r = reasoning.replace("one_pass_r_3x10_h200", "RUN_LABEL").replace(
        'ONE_PASS_REASONING_MODE="r"', 'ONE_PASS_REASONING_MODE="MODE"'
    )
    assert normalized_nr == normalized_r


def test_common_body_freezes_three_packet_thirty_slot_contract() -> None:
    text = COMMON.read_text(encoding="utf-8")
    for value in (
        'EVIDENCE_TARGET="3"',
        'TARGET_GENERATION_GROUPS="3"',
        'EXPECTED_QA_PER_GROUP="10"',
        'EXPECTED_TOTAL_SLOTS="30"',
        'MAX_GENERATION_SLOTS="30"',
        'MAX_ATTEMPTS="1"',
        'ONE_PASS_30_SLOT_MODE="1"',
        'INFERENCE_BACKEND="transformers-local-memory-safe"',
        'TRAIN_ENV="/scratch/xl6775/conda/envs/qwen3vl-smoke"',
        'MODEL_DIR="/scratch/xl6775/models/Qwen3.8-27B"',
        'PRECOMPUTED_SOURCE_JOB_ID="16699348"',
        'CUDA_KEEPER_ENABLE="1"',
        'CUDA_KEEPER_START_AFTER_SECONDS="7200"',
        'QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="262144"',
    ):
        assert value in text
    assert "DAY1_17200000" in text
    assert "DAY3_17000000" in text
    assert "DAY4_21400000" in text


def test_single_smoke_uses_r_and_one_slot_only() -> None:
    text = SMOKE.read_text(encoding="utf-8")
    assert "#SBATCH --account=torch_pr_674_tandon_advanced" in text
    assert "#SBATCH --cpus-per-task=16" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --constraint=h200" in text
    assert "#SBATCH --mem=320G" in text
    assert "#SBATCH --time=01:30:00" in text
    assert "--partition" not in text
    assert "--nodelist" not in text
    assert 'ONE_PASS_REASONING_MODE="r"' in text
    assert 'MAX_GENERATION_SLOTS="1"' in text
    assert 'EXPECTED_TOTAL_SLOTS="1"' in text
    assert 'CUDA_KEEPER_ENABLE="0"' in text
