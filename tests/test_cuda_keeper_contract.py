from pathlib import Path
import sys
import types


REPO_ROOT = Path(__file__).resolve().parents[1]

if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(REPO_ROOT)]
    sys.modules["egolife_two_user_qa"] = package


def test_cuda_keeper_uses_time_activation_and_memory_safety_controls():
    source = (REPO_ROOT / "hpc" / "shared" / "cuda.py").read_text(encoding="utf-8")

    assert "self._stop_event = threading.Event()" in source
    assert "self._stop = threading.Event()" not in source
    assert '"--max-prealloc"' in source
    assert '"--start-after-seconds"' in source
    assert '"--start-used-mib"' not in source
    assert "start_after_seconds" in source
    assert "waiting_for_start_time" in source


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
    assert "--start-after-seconds" in runtime
    assert "--start-used-mib" not in runtime


def test_formal_two_hour_wrapper_enables_keeper_after_two_hours():
    wrapper = (
        REPO_ROOT
        / "hpc"
        / "qa"
        / "experiments"
        / "run_six_user_qa_10min_3groups_x20_qwen38_fast_fix_20260902.sbatch"
    ).read_text(encoding="utf-8")

    assert "#SBATCH --time=2-00:00:00" in wrapper
    assert "#SBATCH --qos=gpu48" in wrapper
    assert "#SBATCH --chdir=/scratch/xl6775/projects/EgoQA-two-user-six-user-10min-speed-qwen38-fix-20260902" in wrapper
    assert "#SBATCH --output=/scratch/xl6775/projects/EgoQA-two-user-six-user-10min-speed-qwen38-fix-20260902/hpc/logs/%x_%j.out" in wrapper
    assert "#SBATCH --error=/scratch/xl6775/projects/EgoQA-two-user-six-user-10min-speed-qwen38-fix-20260902/hpc/logs/%x_%j.err" in wrapper
    assert 'CUDA_KEEPER_ENABLE="${CUDA_KEEPER_ENABLE:-1}"' in wrapper
    assert 'CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${PROJECT_ROOT}/hpc/shared/cuda.py}"' in wrapper
    assert 'CUDA_KEEPER_START_AFTER_SECONDS="${CUDA_KEEPER_START_AFTER_SECONDS:-7200}"' in wrapper
    assert "CUDA_KEEPER_START_USED_MIB" not in wrapper


def test_runtime_storage_preflight_package_is_present_for_remote_sync():
    runtime = (
        REPO_ROOT / "hpc" / "qa" / "smoke" / "run_six_user_qa_runtime_probe.sbatch"
    ).read_text(encoding="utf-8")

    assert (REPO_ROOT / "training" / "__init__.py").is_file()
    assert (REPO_ROOT / "training" / "torch_storage_preflight.py").is_file()
    assert "python -m training.torch_storage_preflight" in runtime


def test_fast_profile_disables_thinking_for_every_model_stage():
    from egolife_two_user_qa.video_qa_loop import six_user_ten_minute_fast_profiles

    profiles = six_user_ten_minute_fast_profiles()

    assert profiles
    assert all(profile.disable_thinking for profile in profiles.values())
