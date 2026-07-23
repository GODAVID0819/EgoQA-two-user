import argparse
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

from egolife_two_user_qa import full_loop_rationale_ablation as ablation
from egolife_two_user_qa.full_loop_rationale_ablation import (
    build_video_loop_argv,
    compare_trials,
    run_sequential_conditions,
    summarize_trial,
)
ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
SBATCH = PROJECT_ROOT / "hpc" / "run_full_loop_rationale_ablation_dual_h100.sbatch"
CUDA_SLURM = PROJECT_ROOT / "hpc" / "cuda_slurm.py"
SINGLE_SBATCH = PROJECT_ROOT / "hpc" / "run_full_loop_rationale_ablation_single_h100.sbatch"
RESUME_SBATCH = PROJECT_ROOT / "hpc" / "run_full_loop_rationale_ablation_resume_single_h100.sbatch"

CUDA_SLURM_SPEC = importlib.util.spec_from_file_location("cuda_slurm_test", CUDA_SLURM)
assert CUDA_SLURM_SPEC is not None and CUDA_SLURM_SPEC.loader is not None
CUDA_SLURM_MODULE = importlib.util.module_from_spec(CUDA_SLURM_SPEC)
CUDA_SLURM_SPEC.loader.exec_module(CUDA_SLURM_MODULE)
nvml_handle_for_logical_device = CUDA_SLURM_MODULE.nvml_handle_for_logical_device


def _check(score: int, status: str, previous: int) -> dict:
    return {
        "status": status,
        "quality_score": score,
        "quality_reason": f"specific reason for score {score}",
        "quota_rebuttal": "",
        "quality_quota": {
            "quota": 48,
            "previous_three_point_assignments": previous,
            "remaining_before_candidate": max(0, 48 - previous),
            "quota_rebuttal_required": False,
        },
    }


def _trace(
    *,
    attempt: int,
    accepted: bool,
    include_rationale: bool,
    formality: tuple[int, str, int],
    grounding: tuple[int, str, int],
    answerability_status: str,
    raw: str,
) -> dict:
    return {
        "attempt": attempt,
        "generation": {"raw_output": raw},
        "result": {"accepted": accepted},
        "judge": {
            "generator_rationale_included": include_rationale,
            "pass_fail_only": False,
            "pass_fail_entropy_logits": "legacy_archived_not_collected",
            "merged": {
                "checks": {
                    "qa_formality": _check(*formality),
                    "evidence_groundedness": _check(*grounding),
                    "answerability": {"status": answerability_status},
                }
            },
        },
    }


def _condition_rows(include_rationale: bool, *, swap_outcomes: bool = False) -> list[dict]:
    first_accepted = not swap_outcomes
    second_accepted = swap_outcomes
    first = _trace(
        attempt=1,
        accepted=first_accepted,
        include_rationale=include_rationale,
        formality=(3, "PASS", 0),
        grounding=(1, "FAIL", 0),
        answerability_status="FAIL" if not first_accepted else "PASS",
        raw='{"question":"same first generation 1"}',
    )
    second_attempt_1 = _trace(
        attempt=1,
        accepted=False,
        include_rationale=include_rationale,
        formality=(2, "PASS", 1),
        grounding=(3, "PASS", 0),
        answerability_status="FAIL",
        raw='{"question":"same first generation 2"}',
    )
    second_attempt_2 = _trace(
        attempt=2,
        accepted=second_accepted,
        include_rationale=include_rationale,
        formality=(1, "FAIL", 1),
        grounding=(2, "PASS", 1),
        answerability_status="PASS" if second_accepted else "FAIL",
        raw='{"question":"retry"}',
    )
    return [
        {
            "evidence_id": "e1",
            "status": "accepted" if first_accepted else "rejected",
            "attempts": [first] if first_accepted else None,
            "generation_trace": None if first_accepted else [first],
        },
        {
            "evidence_id": "e2",
            "status": "accepted" if second_accepted else "rejected",
            "attempts": [second_attempt_1, second_attempt_2] if second_accepted else None,
            "generation_trace": (
                None if second_accepted else [second_attempt_1, second_attempt_2]
            ),
        },
    ]


def _completed_rows(count: int, include_rationale: bool) -> list[dict]:
    rows = []
    for index in range(1, count + 1):
        trace = _trace(
            attempt=1,
            accepted=True,
            include_rationale=include_rationale,
            formality=(2, "PASS", 0),
            grounding=(2, "PASS", 0),
            answerability_status="PASS",
            raw=f'{{"question":"paired generation {index}"}}',
        )
        rows.append(
            {
                "evidence_id": f"e{index}",
                "status": "accepted",
                "attempts": [trace],
            }
        )
    return rows


def test_condition_invocations_differ_only_in_rationale_disclosure_flag() -> None:
    common = {
        "evidence_path": "evidence.jsonl",
        "target_count": 50,
        "max_attempts": 3,
        "model_id": "Qwen/Qwen3.6-27B",
        "max_new_tokens": 4096,
        "max_image_pixels": 262144,
        "dtype": "bfloat16",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "quality_quota": 48,
    }
    with_args = build_video_loop_argv(
        condition="with_rationale",
        output_dir="with",
        **common,
    )
    without_args = build_video_loop_argv(
        condition="without_rationale",
        output_dir="without",
        **common,
    )

    assert "--judge-hide-generator-rationale" not in with_args
    assert "--judge-hide-generator-rationale" in without_args
    assert "--resume" not in with_args + without_args
    assert with_args[with_args.index("--backend") + 1] == "transformers-local"
    assert with_args[with_args.index("--generator-top-k") + 1] == "40"
    assert "--experimental-scored-judge" not in with_args
    assert "--judge-quality-quota" not in with_args


def test_trial_summary_counts_all_attempts_not_only_attempt_one() -> None:
    original_rows = ablation._rows
    ablation._rows = lambda _: _condition_rows(True)
    try:
        summary = summarize_trial(
            Path("with.jsonl"),
            condition="with_rationale",
            packet_count=2,
            quality_quota=48,
        )
    finally:
        ablation._rows = original_rows

    assert summary["scope"] == "all_attempts_and_final_packet_outcomes"
    assert summary["accepted_count"] == 1
    assert summary["rejected_count"] == 1
    assert summary["total_generation_attempts"] == 3
    assert summary["judged_attempts"] == 3
    assert summary["generation_attempt_histogram"] == {"1": 1, "2": 1}
    assert summary["checks"]["qa_formality"]["score_distribution"] == {
        "1": 1,
        "2": 1,
        "3": 1,
    }
    assert summary["checks"]["evidence_groundedness"]["three_point_assignment_count"] == 1


def test_trial_summary_audits_legacy_remaining_capacity_without_blocking_resume() -> None:
    rows = _condition_rows(True)
    rows[0]["attempts"][0]["judge"]["merged"]["checks"]["qa_formality"][
        "quality_quota"
    ]["remaining_before_candidate"] = 47
    original_rows = ablation._rows
    ablation._rows = lambda _: rows
    try:
        summary = summarize_trial(
            Path("with.jsonl"),
            condition="with_rationale",
            packet_count=2,
            quality_quota=48,
        )
    finally:
        ablation._rows = original_rows

    mismatches = summary["checks"]["qa_formality"][
        "quota_metadata_mismatch_records"
    ]
    assert len(mismatches) == 1
    assert mismatches[0]["observed"] == 47
    assert mismatches[0]["expected"] == 48


def test_full_loop_comparison_pairs_final_outcomes_and_first_generations() -> None:
    rows_by_name = {
        "with.jsonl": _condition_rows(True),
        "without.jsonl": _condition_rows(False, swap_outcomes=True),
    }
    original_rows = ablation._rows
    ablation._rows = lambda path: rows_by_name[Path(path).name]
    try:
        comparison = compare_trials(
            with_rationale_intermediate=Path("with.jsonl"),
            without_rationale_intermediate=Path("without.jsonl"),
            packet_count=2,
            quality_quota=48,
        )
    finally:
        ablation._rows = original_rows

    assert comparison["scope"] == "two_independent_full_generation_loops_all_attempts"
    assert comparison["final_packet_outcome_pairs"] == {
        "without_accepted__with_rejected": 1,
        "without_rejected__with_accepted": 1,
    }
    assert comparison["first_attempt_generation_pairing"]["exact_match_count"] == 2
    assert comparison["first_attempt_generation_pairing"]["mismatch_count"] == 0


def test_dual_h100_sbatch_runs_one_full_condition_per_gpu() -> None:
    script = SBATCH.read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --constraint=h100" in script
    assert "#SBATCH --ntasks=2" in script
    assert "--exclusive" in script
    assert "--exact" in script
    assert "--gpus-per-task=1" in script
    assert "--gpu-bind=single:1" in script
    assert 'launch_trial with_rationale "${WITH_DIR}" &' in script
    assert 'launch_trial without_rationale "${WITHOUT_DIR}" &' in script
    assert 'CUDA_KEEPER_GPUS="0,1"' in script
    assert '${HPC_ROOT}/cuda_slurm.py' in script
    assert 'echo "cuda_keeper=required gpus=${CUDA_KEEPER_GPUS}' in script
    assert 'HPC_ROOT="${PROJECT_ROOT}/hpc"' in script
    assert 'source "${HPC_ROOT}/env_qwen3vl.sh"' in script
    assert 'export QWEN3VL_PROJECT_ROOT="${PROJECT_ROOT}"' in script
    assert 'CUDA_KEEPER_BASE_SCRIPT="${CUDA_KEEPER_BASE_SCRIPT:-${HPC_ROOT}/cuda.py}"' in script
    assert 'cuda_keeper_both_gpu_controllers=healthy' in script
    assert 'cuda_keeper_did_not_map_both_slurm_devices' in script
    assert 'wrong experiment module imported' in script
    assert 'media preflight failed:' in script
    assert 'model_config_preflight=ok' in script
    assert 'processor_preflight=ok' in script
    assert "input_contract=evidence_jsonl_and_videos_only" in script
    assert "run_generator_rationale_ablation_qwen.sbatch" not in script
    assert "INTERMEDIATE_PATH=" not in script


def test_slurm_cuda_keeper_maps_logical_devices_to_physical_nvml_handles() -> None:
    script = CUDA_SLURM.read_text(encoding="utf-8")

    assert "_get_nvml_device_index" in script
    assert "CUDA_VISIBLE_DEVICES" in script
    assert "nvmlDeviceGetHandleByUUID" in script
    assert "cuda_keeper_device_mapping" in script
    assert "class SlurmAwareController" in script
    assert "del self._stop" in script
    assert 'os.getenv("CUDA_KEEPER_BASE_SCRIPT")' in script

    class FakeNVML:
        def nvmlInit(self) -> None:
            pass

        def nvmlDeviceGetHandleByIndex(self, index: int) -> str:
            return f"physical-{index}"

        def nvmlDeviceGetHandleByUUID(self, uuid: str | bytes) -> str:
            return f"uuid-{uuid!r}"

    keeper = SimpleNamespace(
        nvml=FakeNVML(),
        torch=SimpleNamespace(
            cuda=SimpleNamespace(_get_nvml_device_index=lambda logical: (3, 7)[logical])
        ),
    )
    assert nvml_handle_for_logical_device(keeper, 0) == "physical-3"
    assert nvml_handle_for_logical_device(keeper, 1) == "physical-7"

    old_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    os.environ["CUDA_VISIBLE_DEVICES"] = "GPU-first,GPU-second"
    try:
        keeper.torch.cuda._get_nvml_device_index = None
        assert nvml_handle_for_logical_device(keeper, 1) == "uuid-'GPU-second'"
    finally:
        if old_visible is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_visible


def test_single_h100_launcher_runs_both_conditions_sequentially() -> None:
    script = SINGLE_SBATCH.read_text(encoding="utf-8")

    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --constraint=h100" in script
    assert "run-sequential" in script
    assert "--gpus 0" in script
    assert "cuda_keeper_gpu0_controller=healthy" in script
    assert "launch_trial with_rationale" not in script
    assert "--gpus-per-task" not in script
    assert 'source "${HPC_ROOT}/env_qwen3vl.sh"' in script
    assert 'CUDA_KEEPER_SCRIPT="${CUDA_KEEPER_SCRIPT:-${HPC_ROOT}/cuda_slurm.py}"' in script


def test_sequential_mode_loads_runner_once_and_resets_each_arm() -> None:
    from egolife_two_user_qa import video_qa_loop

    original_factory = video_qa_loop.make_runner
    original_run_condition = ablation.run_condition
    original_compare = ablation.compare_trials
    original_write_json = ablation.write_json
    original_seed = ablation.seed_single_gpu_process
    factory_calls = []
    conditions = []
    writes = []
    shared_runner = object()

    def fake_factory(backend: str, *args, **kwargs):
        factory_calls.append((backend, args, kwargs))
        return shared_runner

    def fake_run_condition(args):
        conditions.append(args.condition)
        runner = video_qa_loop.make_runner(
            "transformers-local",
            model_id="Qwen/Qwen3.6-27B",
            base_url="http://127.0.0.1:8000/v1",
            max_new_tokens=4096,
            max_image_pixels=262144,
            dtype="bfloat16",
            allow_cpu=False,
            allow_openai_video_input=False,
            disable_thinking=True,
            api_key=None,
        )
        assert runner is shared_runner
        return {"condition": args.condition}

    try:
        video_qa_loop.make_runner = fake_factory
        ablation.run_condition = fake_run_condition
        ablation.compare_trials = lambda **kwargs: {"comparison": "ok"}
        ablation.write_json = lambda path, data: writes.append((path, data))
        ablation.seed_single_gpu_process = lambda seed: {"seed": seed}
        comparison = run_sequential_conditions(
            argparse.Namespace(
                output_root=".",
                target_count=50,
                quality_quota=48,
                seed=1729,
                model_id="Qwen/Qwen3.6-27B",
                max_new_tokens=4096,
                max_image_pixels=262144,
                dtype="bfloat16",
            )
        )
    finally:
        video_qa_loop.make_runner = original_factory
        ablation.run_condition = original_run_condition
        ablation.compare_trials = original_compare
        ablation.write_json = original_write_json
        ablation.seed_single_gpu_process = original_seed

    assert conditions == ["with_rationale", "without_rationale"]
    assert len(factory_calls) == 1
    assert comparison["execution"]["shared_model_load_count"] == 1
    assert comparison["execution"]["model_preloaded_before_condition_seeding"] is True
    assert comparison["execution"]["seed_reset_before_each_condition"] is True
    assert len(writes) == 1


def test_resume_launcher_targets_only_partial_without_rationale_arm() -> None:
    script = RESUME_SBATCH.read_text(encoding="utf-8")

    assert "#SBATCH --gres=gpu:1" in script
    assert "#SBATCH --time=24:00:00" in script
    assert "RATIONALE_RESUME_SOURCE_JOB_ID" in script
    assert "RATIONALE_RESUME_OUTPUT_ROOT" in script
    assert "resume-sequential" in script
    assert "resume_condition=without_rationale" in script
    assert "completed_with_rationale_reused=true" in script
    assert "--gpus 0" in script
    assert 'source "${HPC_ROOT}/env_qwen3vl.sh"' in script

    common = {
        "condition": "without_rationale",
        "evidence_path": "evidence.jsonl",
        "output_dir": "without_rationale",
        "target_count": 50,
        "max_attempts": 3,
        "model_id": "Qwen/Qwen3.6-27B",
        "max_new_tokens": 4096,
        "max_image_pixels": 262144,
        "dtype": "bfloat16",
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "quality_quota": 48,
    }
    argv = build_video_loop_argv(**common, resume=True)
    assert "--resume" in argv
    assert "--judge-hide-generator-rationale" in argv


def test_resume_mode_reuses_completed_arm_and_finishes_partial_prefix() -> None:
    from egolife_two_user_qa import video_qa_loop

    original_rows = ablation._rows
    original_factory = video_qa_loop.make_runner
    original_run_condition = ablation.run_condition
    original_write_json = ablation.write_json
    original_seed = ablation.seed_single_gpu_process
    evidence_rows = [{"evidence_id": f"e{index}"} for index in range(1, 51)]
    rows_by_path = {
        "evidence.jsonl": evidence_rows,
        "with_rationale/qa_mcq.intermediate.jsonl": _completed_rows(50, True),
        "without_rationale/qa_mcq.intermediate.jsonl": _completed_rows(33, False),
        "without_rationale/qa_mcq.jsonl": [
            {"evidence_id": f"e{index}"} for index in range(1, 34)
        ],
        "without_rationale/qa_mcq.rejected.jsonl": [],
    }
    factory_calls = []
    resumed_args = []
    writes = []
    shared_runner = object()

    def normalized(path) -> str:
        return str(Path(path)).replace("\\", "/").lstrip("./")

    def fake_rows(path):
        return rows_by_path.get(normalized(path), [])

    def fake_factory(backend: str, *args, **kwargs):
        factory_calls.append((backend, args, kwargs))
        return shared_runner

    def fake_run_condition(args):
        resumed_args.append(args)
        runner = video_qa_loop.make_runner(
            "transformers-local",
            model_id="Qwen/Qwen3.6-27B",
            base_url="http://127.0.0.1:8000/v1",
            max_new_tokens=4096,
            max_image_pixels=262144,
            dtype="bfloat16",
            allow_cpu=False,
            allow_openai_video_input=False,
            disable_thinking=True,
            api_key=None,
        )
        assert runner is shared_runner
        rows_by_path["without_rationale/qa_mcq.intermediate.jsonl"] = _completed_rows(
            50, False
        )
        rows_by_path["without_rationale/qa_mcq.jsonl"] = [
            {"evidence_id": f"e{index}"} for index in range(1, 51)
        ]
        return {"condition": "without_rationale"}

    try:
        ablation._rows = fake_rows
        video_qa_loop.make_runner = fake_factory
        ablation.run_condition = fake_run_condition
        ablation.write_json = lambda path, data: writes.append((path, data))
        ablation.seed_single_gpu_process = lambda seed: {"seed": seed}
        comparison = ablation.resume_sequential_without_rationale(
            argparse.Namespace(
                evidence="evidence.jsonl",
                output_root=".",
                target_count=50,
                quality_quota=48,
                seed=1729,
                model_id="Qwen/Qwen3.6-27B",
                max_new_tokens=4096,
                max_image_pixels=262144,
                dtype="bfloat16",
            )
        )
    finally:
        ablation._rows = original_rows
        video_qa_loop.make_runner = original_factory
        ablation.run_condition = original_run_condition
        ablation.write_json = original_write_json
        ablation.seed_single_gpu_process = original_seed

    assert len(resumed_args) == 1
    assert resumed_args[0].condition == "without_rationale"
    assert resumed_args[0].resume is True
    assert len(factory_calls) == 1
    assert comparison["execution"]["resumed_from_completed_packet_count"] == 33
    assert comparison["execution"]["completed_with_rationale_reused"] is True
    assert comparison["execution"]["quota_state_restored_from_intermediate"] is True
    assert len(writes) == 1
