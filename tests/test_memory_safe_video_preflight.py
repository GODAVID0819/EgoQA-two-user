from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "hpc" / "qa" / "smoke" / "run_six_user_qa_runtime_probe.sbatch"


def test_ten_minute_video_preflight_uses_runner_compatible_transcode_cache() -> None:
    text = RUNTIME.read_text(encoding="utf-8")

    assert "qwen_memory_safe_preflight_transcode" in text
    assert "QWEN_MEMORY_SAFE_TRANSCODE_MAX_EDGE" in text
    assert 'QWEN_MEMORY_SAFE_VIDEO_CACHE_DIR' in text
    assert '"fps-{fps:g}"' in text
    assert '"edge-{max_edge}"' in text
    assert '"crf-{crf}"' in text
    assert 'subprocess.run(command, check=True' in text
    assert "preflight_video_paths" in text


def test_preflight_does_not_decode_original_six_videos_directly() -> None:
    text = RUNTIME.read_text(encoding="utf-8")
    start = text.index("qwen_memory_safe_preflight_transcode")
    end = text.index("echo \"stage=generate_six_user_qa\"", start)
    block = text[start:end]

    assert '"video": str(output)' in block
    assert '"video": item' not in block
