from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[5]
SCRIPT = REPO_ROOT / "hpc" / "grpo_v3" / "qa_cross_view_relation" / "smoke1.sbatch"


class CrossViewRelationSlurmTests(unittest.TestCase):
    def test_native_video_smoke_uses_ffmpeg_runtime_and_torchcodec_preflight(self) -> None:
        text = SCRIPT.read_text(encoding="utf-8")

        self.assertIn("FFMPEG_ENV", text)
        self.assertIn('PATH="${FFMPEG_ENV}/bin:${PATH}"', text)
        self.assertIn('LD_LIBRARY_PATH="${FFMPEG_ENV}/lib:${LD_LIBRARY_PATH:-}"', text)
        self.assertIn('"${FFMPEG_ENV}/bin/ffmpeg" -version', text)
        self.assertIn("from torchcodec.decoders import VideoDecoder", text)
        self.assertLess(text.index("from torchcodec.decoders import VideoDecoder"), text.index('"${SWIFT}" rlhf'))


if __name__ == "__main__":
    unittest.main()
