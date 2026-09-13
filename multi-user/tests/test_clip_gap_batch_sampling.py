from __future__ import annotations

import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.clip_gap_demo import sample_short_video  # noqa: E402


class BatchedVideoSamplingTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        self.tmp_path = tmp_root / f"batch_sampling_{uuid.uuid4().hex}"
        self.tmp_path.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp_path, True)

    def test_one_ffmpeg_process_extracts_the_complete_uniform_frame_set(self) -> None:
        video = self.tmp_path / "video.mp4"
        video.write_bytes(b"video")
        commands: list[list[str]] = []

        def fake_run(command, *, check):
            self.assertTrue(check)
            command = [str(value) for value in command]
            commands.append(command)
            frame_count = int(command[command.index("-frames:v") + 1])
            pattern = command[-1]
            for index in range(frame_count):
                path = Path(pattern.replace("%06d", f"{index:06d}"))
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"frame-{index}".encode("ascii"))

        output_dir = self.tmp_path / "frames"
        with (
            mock.patch(
                "egolife_two_user_qa.clip_gap_demo.shutil.which",
                return_value="/test/bin/ffmpeg",
            ),
            mock.patch(
                "egolife_two_user_qa.clip_gap_demo.subprocess.run",
                side_effect=fake_run,
            ),
        ):
            frames = sample_short_video(
                video,
                output_dir,
                duration_seconds=4.0,
                sample_interval_seconds=1.0,
                start_seconds=5.0,
            )
            cached_frames = sample_short_video(
                video,
                output_dir,
                duration_seconds=4.0,
                sample_interval_seconds=1.0,
                start_seconds=5.0,
            )

        self.assertEqual(len(commands), 1)
        self.assertEqual([row["timestamp_seconds"] for row in frames], [5.0, 6.0, 7.0, 8.0])
        self.assertEqual(frames, cached_frames)
        self.assertEqual(commands[0][commands[0].index("-frames:v") + 1], "4")
        self.assertEqual(commands[0][commands[0].index("-ss") + 1], "5.000")
        self.assertEqual(commands[0][commands[0].index("-t") + 1], "4.000")
        self.assertIn("fps=1:start_time=0", commands[0][commands[0].index("-vf") + 1])
        self.assertTrue(all(Path(row["path"]).is_file() for row in frames))
        self.assertFalse(any(path.name.startswith(".ffmpeg_batch_") for path in output_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
