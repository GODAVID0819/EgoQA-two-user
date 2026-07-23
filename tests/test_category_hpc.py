import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COUNT_COMPATIBILITY_LAUNCHER = ROOT / "hpc" / "run_implicit_count_nudge_50.sbatch"
CATEGORY_COMPATIBILITY_LAUNCHER = ROOT / "hpc" / "run_category_guided_50.sbatch"
PRODUCTION_LAUNCHER = ROOT / "hpc" / "run_clip_pruned_sampling_neutral_pf_50.sbatch"
CONCURRENT_LAUNCHER = ROOT / "hpc" / "run_concurrent_activity_comparison_50.sbatch"
IMPLICIT_FAMILIES_LAUNCHER = ROOT / "hpc" / "run_implicit_underrepresented_families_50.sbatch"
ARCHIVED_SCORED_JUDGE_LAUNCHERS = (
    ROOT / "hpc" / "run_clip_pruned_neutral_prompt_50_qwen.sbatch",
    ROOT / "hpc" / "run_clip_pruned_neutral_prompt_50_mix.sbatch",
)
ARCHIVED_QUOTA_JUDGE_LAUNCHERS = (
    ROOT / "hpc" / "run_clip_pruned_rationale_quota48_single_model_50.sbatch",
    ROOT
    / "hpc"
    / "run_clip_pruned_gemini25_openrouter_generator_qwen_judge_quota48_50.sbatch",
)


def test_legacy_scoring_and_quota_launchers_fail_fast_as_archived() -> None:
    for path in ARCHIVED_SCORED_JUDGE_LAUNCHERS:
        script = path.read_text(encoding="utf-8")
        guard = 'error=archived_scored_judge_experiment_not_a_production_pipeline'
        assert guard in script
        assert script.index(guard) < script.index("exit 2")
        assert script.index("exit 2") < script.index("PROJECT_ROOT=")
        assert "run_clip_pruned_sampling_neutral_pf_50.sbatch" in script

    for path in ARCHIVED_QUOTA_JUDGE_LAUNCHERS:
        script = path.read_text(encoding="utf-8")
        guard = 'error=archived_scored_quota_judge_experiment_not_a_production_pipeline'
        assert guard in script
        assert script.index(guard) < script.index("exit 2")
        assert script.index("exit 2") < script.index("PROJECT_ROOT=")
        assert "run_clip_pruned_sampling_neutral_pf_50.sbatch" in script


def test_retired_experiment_launchers_only_forward_to_current_production_run() -> None:
    count_script = COUNT_COMPATIBILITY_LAUNCHER.read_text(encoding="utf-8")
    category_script = CATEGORY_COMPATIBILITY_LAUNCHER.read_text(encoding="utf-8")

    assert "Count-specific steering was removed" in count_script
    assert "category-guided experiment has" in category_script
    assert "been retired" in category_script
    assert "current category-free production launcher" in count_script
    assert "current category-free production launcher" in category_script
    for script in (count_script, category_script):
        assert "run_clip_pruned_sampling_neutral_pf_50.sbatch" in script
        assert "QUESTION_CATEGORY_GUIDANCE" not in script


def test_production_launcher_preflights_active_relation_and_judge_contracts() -> None:
    script = PRODUCTION_LAUNCHER.read_text(encoding="utf-8")

    assert "#SBATCH --job-name=egolife_impl50" in script
    assert "relation_design_text_only_formality_pf_50" in script
    preflight_index = script.index('echo "stage=preflight_relation_design_text_only_formality"')
    generation_index = script.index('echo "stage=generate_video_qa_loop"')
    assert preflight_index < generation_index
    assert "POSITIVE_EXAMPLES_GUIDANCE" in script
    assert "QA_FORMALITY_SEMANTIC_SUBCHECK_NAMES" in script
    assert "VIDEO_GENERATION_SCHEMA" in script
    assert "production generation schema still requests category fields" in script
    assert "actual_prompts_path" in script
    assert "preflight imported prompts.py from the wrong checkout" in script
    assert "required_users[1] may be sufficient alone" in script
    assert "category_free_prompt_preflight=ok" in script
    assert "generator_relation_design_preflight=ok" in script
    assert "symmetric_concurrency_preflight=ok" in script
    assert "pruned_to_original_temporal_mapping=ok" in script
    assert "qa_formality_text_only_contract=ok" in script
    assert "qa_formality_five_semantic_subchecks=ok" in script
    assert "timestamp_judge_only_contract=ok" in script
    assert "participant_name_deterministic_check=ok" in script
    assert "grounding_distractor_contract=ok" in script
    assert "answerability_ambiguity_contract=ok" in script
    assert "judge_json_repair_contract=ok" in script
    assert "generator prompt retained archived steering" in script
    assert "generator prompt omitted active relation-design contract" in script
    assert "Relation-design rules:" in script
    assert "Concurrent single-anchor matching" in script
    assert "Concurrent pair matching" in script
    assert "pure text-only semantic judge and do not see the videos" in script
    assert "USER_FACING_TIMESTAMP_PATTERNS" in script
    assert "deterministic timestamp parsing was unexpectedly reactivated" in script
    assert "QUESTION_CATEGORY_GUIDANCE" not in script
    assert "JUDGE_CATEGORY_GUIDANCE" not in script


def test_production_launcher_verifies_recorded_category_free_prompts() -> None:
    script = PRODUCTION_LAUNCHER.read_text(encoding="utf-8")

    verify_index = script.index('echo "stage=verify_generation"')
    validation_index = script.index('echo "stage=validate_outputs"')
    assert verify_index < validation_index
    assert "def assert_category_free" in script
    assert "prompt retained category machinery" in script
    assert "generation prompt retained strict-dependency preference" in script
    assert "generation prompt omitted the positive examples block" in script
    assert "generation prompt omitted active relation-design contract" in script
    assert "qa_formality prompt unexpectedly includes generator rationale" in script
    assert "qa_formality prompt unexpectedly received media" in script
    assert "qa_formality prompt leaked excluded context" in script
    assert "evidence_groundedness prompt omitted grounding contract" in script
    assert "answerability prompt omitted ambiguity or symmetry contract" in script
    assert "passing qa_formality check omitted semantic subchecks" in script
    assert "accepted_category_counts" not in script


def test_implicit_family_launchers_forward_to_verified_production_run() -> None:
    scripts = (
        CONCURRENT_LAUNCHER.read_text(encoding="utf-8"),
        IMPLICIT_FAMILIES_LAUNCHER.read_text(encoding="utf-8"),
    )

    for script in scripts:
        assert "#SBATCH --job-name=egolife_impl50" in script
        assert "implicit_underrepresented_families_50" in script
        assert "run_clip_pruned_sampling_neutral_pf_50.sbatch" in script


def test_category_free_hpc_embedded_python_blocks_compile() -> None:
    script = PRODUCTION_LAUNCHER.read_text(encoding="utf-8")
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script, flags=re.DOTALL)
    assert blocks
    for index, block in enumerate(blocks, start=1):
        compile(block, f"{PRODUCTION_LAUNCHER.name}:heredoc:{index}", "exec")
