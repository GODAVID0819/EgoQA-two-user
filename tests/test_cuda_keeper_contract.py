from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_cuda_keeper_uses_safe_stop_event_and_memory_controls():
    source = (REPO_ROOT / "hpc" / "shared" / "cuda.py").read_text(encoding="utf-8")

    assert "self._stop_event = threading.Event()" in source
    assert "self._stop = threading.Event()" not in source
    assert '"--max-prealloc"' in source
    assert '"--start-used-mib"' in source


def test_six_user_long_run_starts_and_cleans_up_keeper():
    pilot = (
        REPO_ROOT / "hpc" / "qa" / "experiments" / "run_six_user_qa_pilot_40.sbatch"
    ).read_text(encoding="utf-8")
    runtime = (
        REPO_ROOT / "hpc" / "qa" / "smoke" / "run_six_user_qa_runtime_probe.sbatch"
    ).read_text(encoding="utf-8")

    assert 'CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"' in pilot
    assert "stage=start_cuda_keeper" in runtime
    assert "stage=stop_cuda_keeper" in runtime
    assert "--max-prealloc" in runtime
    assert "--start-used-mib" in runtime


def test_formal_two_hour_wrapper_enables_keeper_after_model_load():
    wrapper = (
        REPO_ROOT
        / "hpc"
        / "qa"
        / "experiments"
        / "run_six_user_qa_10min_3groups_x20_qwen38_fast_fix_20260902.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --time=2-00:00:00" in wrapper
    assert 'CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"' in wrapper
    assert 'CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/shared/cuda.py}"' in wrapper
    assert 'CUDA_KEEPER_START_USED_MIB="${CUDA_KEEPER_START_USED_MIB:-49152}"' in wrapper
