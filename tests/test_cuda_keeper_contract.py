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
