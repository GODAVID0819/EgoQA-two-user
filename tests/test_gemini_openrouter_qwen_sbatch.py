from pathlib import Path

from egolife_two_user_qa.prompts import quality_quota_prompt


SBATCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "hpc"
    / "run_clip_pruned_gemini25_openrouter_generator_qwen_judge_quota48_50.sbatch"
)


def test_hybrid_sbatch_routes_only_generation_to_gemini_openrouter() -> None:
    script = SBATCH_PATH.read_text(encoding="utf-8")

    assert "archived_scored_quota_judge_experiment_not_a_production_pipeline" in script
    assert script.index("exit 2") < script.index('GENERATOR_BACKEND="openrouter"')
    assert "#SBATCH --gres=gpu:1" in script
    assert 'GENERATOR_BACKEND="openrouter"' in script
    assert 'GENERATOR_MODEL_ID="google/gemini-2.5-flash"' in script
    assert 'OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"' in script
    assert 'JUDGE_BACKEND="transformers-local"' in script
    assert 'Qwen/Qwen3.6-27B' in script
    assert '--allow-openai-video-input' in script
    assert 'OPENROUTER_VIDEO_MAX_EDGE:-0' in script
    assert 'OPENROUTER_VIDEO_FPS:-0' in script
    assert '--judge-backend "${JUDGE_BACKEND}"' in script
    assert '--judge-model-id "${JUDGE_MODEL_ID}"' in script
    assert not any(
        line.strip().startswith("--qa-formality-use-generator")
        for line in script.splitlines()
    )


def test_hybrid_sbatch_keeps_sampling_generator_only_and_runs_keeper() -> None:
    script = SBATCH_PATH.read_text(encoding="utf-8")

    assert "GENERATOR_TEMPERATURE=0.7" in script
    assert "GENERATOR_TOP_P=0.9" in script
    assert "GENERATOR_TOP_K=40" in script
    assert '--generator-decode-mode sampling' in script
    assert '--generator-top-k "${GENERATOR_TOP_K}"' in script
    assert "judge_decode=greedy" in script
    assert 'python "${CUDA_KEEPER_SCRIPT}"' in script
    assert "QUALITY_QUOTA=48" in script
    assert '--judge-quality-quota "${QUALITY_QUOTA}"' in script


def test_hybrid_sbatch_validates_openrouter_key_before_loading_qwen() -> None:
    script = SBATCH_PATH.read_text(encoding="utf-8")

    assert 'OPENROUTER_API_KEY="PASTE_OPENROUTER_API_KEY_HERE"' in script
    assert "export OPENROUTER_API_KEY" in script
    assert '"${OPENROUTER_API_KEY}" == "PASTE_OPENROUTER_API_KEY_HERE"' in script
    auth_index = script.index('echo "stage=preflight_openrouter_auth"')
    keeper_index = script.index('echo "stage=start_cuda_keeper"')
    generation_index = script.index('echo "stage=generate_video_qa_loop"')
    assert auth_index < keeper_index < generation_index
    assert 'python - "${OPENROUTER_BASE_URL}/key"' in script
    assert '"Authorization": f"Bearer {key}"' in script
    assert "OPENROUTER_API_KEY has leading/trailing whitespace" in script
    assert "quotation marks became part of OPENROUTER_API_KEY" in script


def test_quota_prompt_is_plain_text_without_markdown_bold_markers() -> None:
    prompt = quality_quota_prompt(previous_three_point_assignments=0, quota=48)

    assert "The prompt budget for this category is at most 48 3-point assignments." in prompt
    assert "Previous 3-point assignments already observed: 0." in prompt
    assert "Remaining 3-point capacity before this candidate: 48." in prompt
    assert "*" * 2 not in prompt


def test_hybrid_sbatch_is_an_isolated_output_path() -> None:
    script = SBATCH_PATH.read_text(encoding="utf-8")

    assert "EGOLIFE2U_GEMINI25_TEST_OUTPUT_ROOT" in script
    assert "EGOLIFE2U_OUTPUT_ROOT" not in script
    assert "clip_pruned_gemini25_openrouter_generator_qwen_judge_quota48_50_" in script
    assert "run_clip_pruned_rationale_quota48_single_model_50.sbatch" not in script
