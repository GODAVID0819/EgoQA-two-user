from __future__ import annotations

import shutil
import unittest
import uuid
from pathlib import Path
from unittest import mock

from egolife_two_user_qa.evidence import concatenate_video_segments


class EvidenceAssemblyPaddingTests(unittest.TestCase):
    def setUp(self) -> None:
        workspace_tmp = Path(__file__).resolve().parents[1] / "tmp"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        self.tmp_path = workspace_tmp / f"evidence_padding_{uuid.uuid4().hex}"
        self.tmp_path.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.tmp_path, True)
        self.sources = []
        for index in range(20):
            source = self.tmp_path / f"segment_{index:02d}.mp4"
            source.write_bytes(b"source")
            self.sources.append(source)

    def test_nearly_complete_window_is_padded_to_nominal_duration(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command, *, check):
            self.assertTrue(check)
            commands.append(command)
            Path(command[-1]).write_bytes(b"copied" if len(commands) == 1 else b"padded")

        output = self.tmp_path / "window.mp4"
        with (
            mock.patch("egolife_two_user_qa.evidence.shutil.which", return_value="ffmpeg"),
            mock.patch("egolife_two_user_qa.evidence.subprocess.run", side_effect=fake_run),
            mock.patch(
                "egolife_two_user_qa.evidence.ffprobe_duration",
                side_effect=[592.673, 600.0],
            ),
        ):
            concatenate_video_segments(self.sources, output, duration_seconds=600.0)

        self.assertEqual(output.read_bytes(), b"padded")
        self.assertEqual(len(commands), 2)
        padding_command = commands[1]
        self.assertIn("-vf", padding_command)
        self.assertIn("tpad=stop_mode=clone:stop_duration=8.327", padding_command)
        self.assertEqual(padding_command[padding_command.index("-t") + 1], "600.000")
        self.assertEqual(padding_command[padding_command.index("-c:a") + 1], "copy")

    def test_large_shortfall_is_rejected_as_incomplete(self) -> None:
        commands: list[list[str]] = []

        def fake_run(command, *, check):
            self.assertTrue(check)
            commands.append(command)
            Path(command[-1]).write_bytes(b"copied")

        output = self.tmp_path / "window.mp4"
        with (
            mock.patch("egolife_two_user_qa.evidence.shutil.which", return_value="ffmpeg"),
            mock.patch("egolife_two_user_qa.evidence.subprocess.run", side_effect=fake_run),
            mock.patch("egolife_two_user_qa.evidence.ffprobe_duration", return_value=560.0),
        ):
            with self.assertRaisesRegex(RuntimeError, "exceeds the safe padding limit"):
                concatenate_video_segments(self.sources, output, duration_seconds=600.0)

        self.assertEqual(len(commands), 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
