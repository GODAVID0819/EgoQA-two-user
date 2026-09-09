"""统一 reasoning A/B 对旧 one-pass reasoning 合同的替代检查。"""

from pathlib import Path

from egolife_two_user_qa.video_qa_loop import REASONING_MODES


ROOT = Path(__file__).resolve().parents[1]


def test_ten_minute_reasoning_uses_unified_ab_runtime() -> None:
    runtime = (
        ROOT
        / "hpc"
        / "qa"
        / "production"
        / "run_six_user_qa_10min_sequential_0p5_fresh30.sbatch"
    ).read_text(encoding="utf-8")
    nr = (
        ROOT
        / "hpc"
        / "qa"
        / "experiments"
        / "run_six_user_qa_reasoning_nr_h200.sbatch"
    ).read_text(encoding="utf-8")
    reasoning = (
        ROOT
        / "hpc"
        / "qa"
        / "experiments"
        / "run_six_user_qa_reasoning_r_h200.sbatch"
    ).read_text(encoding="utf-8")

    assert REASONING_MODES == ("nr", "r")
    assert '--reasoning-mode "${REASONING_MODE}"' in runtime
    assert 'REASONING_MODE="nr"' in nr
    assert 'REASONING_MODE="r"' in reasoning
    assert "--six-user-ten-minute-reasoning-profile" not in runtime


def test_unified_ab_runtime_retains_sequential_fact_audit() -> None:
    runtime = (
        ROOT
        / "hpc"
        / "qa"
        / "production"
        / "run_six_user_qa_10min_sequential_0p5_fresh30.sbatch"
    ).read_text(encoding="utf-8")

    assert (
        'SIX_USER_JUDGE_MODE="${SIX_USER_JUDGE_MODE:-sequential-separated-fact-audit}"'
        in runtime
    )
    assert '--six-user-judge-mode "${SIX_USER_JUDGE_MODE}"' in runtime
    assert '--max-packets-in-flight "${MAX_PACKETS_IN_FLIGHT}"' in runtime
    assert '--max-review-lanes "${MAX_REVIEW_LANES}"' in runtime
