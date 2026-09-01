from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEDIA_JOB = ROOT / "hpc" / "qa" / "experiments" / "prepare_qwen_two_condition_review_media.sbatch"
REVIEW_JOB = ROOT / "hpc" / "qa" / "experiments" / "run_qwen_two_condition_review.sbatch"


def test_media_job_uses_cpu_job_specific_storage_and_real_preflight() -> None:
    text = MEDIA_JOB.read_text(encoding="utf-8")
    assert "#SBATCH --account=torch_pr_674_tandon_advanced" in text
    assert "#SBATCH --partition=cpu_short" in text
    assert "#SBATCH --qos=" not in text
    assert "#SBATCH --time=00:50:00" in text
    assert "#SBATCH --gres" not in text
    assert "--nodelist" not in text
    assert 'JOB_SCRATCH_ROOT="/scratch/${USER}/job_scratch/' in text
    assert 'MEDIA_RUN_ROOT="${OUTPUT_ROOT}/media_${SLURM_JOB_ID}"' in text
    assert "python -m training.torch_storage_preflight" in text
    assert "prepare_qwen_review_stitched_media.py" in text
    assert "storage_preflight.json" in text
    assert "pip_check.txt" in text
    assert "media_manifest.json" in text


def test_review_job_keeps_smoke_and_formal_in_job_specific_outputs() -> None:
    text = REVIEW_JOB.read_text(encoding="utf-8")
    assert "#SBATCH --account=torch_pr_674_tandon_advanced" in text
    assert "#SBATCH --partition=h100_tandon" in text
    assert "#SBATCH --qos=gpu48" in text
    assert "#SBATCH --gres=gpu:1" in text
    assert "#SBATCH --constraint=h100" in text
    assert "#SBATCH --time=00:50:00" in text
    assert "--nodelist" not in text
    assert 'OUTDIR="${OUTPUT_ROOT}/${REVIEW_MODE}_${SLURM_JOB_ID}"' in text
    assert "python -m training.torch_storage_preflight" in text
    assert 'QWEN_MEMORY_SAFE_VIDEO_FPS="0.25"' in text
    assert 'QWEN_MEMORY_SAFE_MAX_IMAGE_PIXELS="65536"' in text
    assert 'QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS="131072"' in text
    assert "egolife_two_user_qa.tools.run_qwen_two_condition_review" in text
    assert "--max-new-tokens 256" in text
    assert "--disable-thinking" in text
    assert 'QA_LIMIT_ARGS=(--qa-limit 1)' in text
    assert "predictions.jsonl" in text
    assert "summary.json" in text
    assert "job_result.json" in text
    assert "pip_check.txt" in text
