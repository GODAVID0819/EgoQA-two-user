from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "build_six_user_review_media.py"


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError(f"review media script is missing: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location("build_six_user_review_media", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def candidate_rows() -> list[dict]:
    users = [
        ("Jake", "A1_JAKE"),
        ("Alice", "A2_ALICE"),
        ("Tasha", "A3_TASHA"),
        ("Lucia", "A4_LUCIA"),
        ("Katrina", "A5_KATRINA"),
        ("Shure", "A6_SHURE"),
    ]
    tokens = ["20060000", "20063000", "20070000", "20073000", "20080000", "20083000"]
    clips = []
    for user, agent_dir in users:
        clips.append(
            {
                "agent_name": user,
                "agent_dir": agent_dir,
                "segments": [
                    {
                        "time_token": token,
                        "clip_clock": f"20:0{6 + index // 2}:{30 * (index % 2):02d}.00",
                        "video_url": f"https://example.test/{agent_dir}/{token}.mp4",
                    }
                    for index, token in enumerate(tokens)
                ],
            }
        )
    return [
        {
            "evidence_id": "candidate-s5",
            "generation_group_id": "DAY6::20060000",
            "clips": clips,
        },
        {
            "evidence_id": "candidate-s6",
            "generation_group_id": "DAY6::20060000",
            "clips": list(reversed(clips)),
        },
    ]


class ReviewMediaScriptTests(unittest.TestCase):
    def test_extracts_one_ordered_36_segment_plan_per_group(self) -> None:
        module = load_module()

        plan = module.extract_review_media_plan(
            candidate_rows(),
            generation_group_id="DAY6::20060000",
        )

        self.assertEqual(plan["generation_group_id"], "DAY6::20060000")
        self.assertEqual(list(plan["users"]), ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"])
        self.assertTrue(all(len(row["segments"]) == 6 for row in plan["users"].values()))
        self.assertEqual(
            [row["time_token"] for row in plan["users"]["Jake"]["segments"]],
            ["20060000", "20063000", "20070000", "20073000", "20080000", "20083000"],
        )
        self.assertEqual(
            len(
                {
                    segment["video_url"]
                    for user in plan["users"].values()
                    for segment in user["segments"]
                }
            ),
            36,
        )

    def test_download_uses_part_file_and_atomic_rename(self) -> None:
        module = load_module()

        class Response(io.BytesIO):
            status = 200

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "segment.mp4"
            result = module.download_atomic(
                "https://example.test/segment.mp4",
                target,
                open_url=lambda *_args, **_kwargs: Response(b"video-bytes"),
            )

            self.assertEqual(result, target)
            self.assertEqual(target.read_bytes(), b"video-bytes")
            self.assertFalse(target.with_name(".segment.mp4.part").exists())

    def test_concat_uses_ordered_stream_copy_and_verifies_output(self) -> None:
        module = load_module()
        calls = []

        def fake_runner(command, **_kwargs):
            calls.append(list(command))
            output = Path(command[-1])
            output.write_bytes(b"stitched")

            class Result:
                stdout = ""
                stderr = ""
                returncode = 0

            return Result()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            segments = []
            for index in range(6):
                path = root / f"segment-{index}.mp4"
                path.write_bytes(b"segment")
                segments.append(path)
            output = root / "stitched.mp4"

            module.concat_segments(
                segments,
                output,
                ffmpeg_binary="ffmpeg-test",
                command_runner=fake_runner,
            )

            self.assertEqual(calls[0][0], "ffmpeg-test")
            self.assertIn("-f", calls[0])
            self.assertIn("concat", calls[0])
            self.assertIn("copy", calls[0])
            concat_file = Path(calls[0][calls[0].index("-i") + 1])
            concat_text = concat_file.read_text(encoding="utf-8")
            self.assertLess(concat_text.index("segment-0.mp4"), concat_text.index("segment-5.mp4"))
            self.assertEqual(output.read_bytes(), b"stitched")

    def test_report_update_lists_36_urls_and_preserves_annotations(self) -> None:
        module = load_module()
        plan = module.extract_review_media_plan(
            candidate_rows(),
            generation_group_id="DAY6::20060000",
        )
        url_markdown = module.render_review_media_markdown(plan, stitched_paths={})
        self.assertEqual(url_markdown.count("https://example.test/"), 36)

        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "report.md"
            report.write_text(
                "# Report\n\n## 2. 统一六路视频 URL\n\nold\n\n## 3. 完整统计\n\n"
                "> 人工标注必须保留\n",
                encoding="utf-8",
            )
            module.update_report_media_section(report, url_markdown)
            updated = report.read_text(encoding="utf-8")

            self.assertIn("https://example.test/A1_JAKE/20060000.mp4", updated)
            self.assertIn("> 人工标注必须保留", updated)
            self.assertNotIn("\nold\n", updated)


if __name__ == "__main__":
    unittest.main()
