from __future__ import annotations

import unittest

from training.grpo_v3.baseline.gate0_probe import assert_framework_version, build_gate0_result, collate_encoded_batch


class Gate0ResultTests(unittest.TestCase):
    def test_collates_exactly_four_encoded_rows(self) -> None:
        class Template:
            def data_collator(self, rows):
                self.rows = rows
                return {"input_ids": [row["input_ids"] for row in rows]}

        template = Template()
        batch = collate_encoded_batch(template, [{"input_ids": [index]} for index in range(4)])
        self.assertEqual(len(template.rows), 4)
        self.assertEqual(len(batch["input_ids"]), 4)
        with self.assertRaisesRegex(ValueError, "4"):
            collate_encoded_batch(template, [{"input_ids": [0]}])

    def test_requires_exact_ms_swift_release(self) -> None:
        assert_framework_version("4.2.2")
        with self.assertRaisesRegex(RuntimeError, "4.2.2"):
            assert_framework_version("4.2.1")

    def test_passes_only_when_all_probe_assertions_hold(self) -> None:
        result = build_gate0_result(
            single_video_ok=True,
            dual_video_ok=True,
            batch_size=4,
            trainable_lora_parameters=128,
            visual_trainable_parameters=0,
            aligner_trainable_parameters=0,
            media_metadata={
                "video_grid_thw": [[1, 2, 3]],
                "actual_video_count": 2,
                "visual_token_count": 6,
                "processor_class": "Qwen3VLProcessor",
                "video_backend": "qwen_vl_utils",
            },
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["batch_size"], 4)

        for override in (
            {"single_video_ok": False},
            {"dual_video_ok": False},
            {"batch_size": 3},
            {"trainable_lora_parameters": 0},
            {"visual_trainable_parameters": 1},
            {"aligner_trainable_parameters": 1},
            {"media_metadata": {}},
            {
                "media_metadata": {
                    "video_grid_thw": None,
                    "actual_video_count": 2,
                    "visual_token_count": 0,
                    "processor_class": "Qwen3VLProcessor",
                    "video_backend": "qwen_vl_utils",
                }
            },
        ):
            values = {
                "single_video_ok": True,
                "dual_video_ok": True,
                "batch_size": 4,
                "trainable_lora_parameters": 128,
                "visual_trainable_parameters": 0,
                "aligner_trainable_parameters": 0,
                "media_metadata": {
                    "video_grid_thw": [[1, 2, 3]],
                    "actual_video_count": 2,
                    "visual_token_count": 6,
                    "processor_class": "Qwen3VLProcessor",
                    "video_backend": "qwen_vl_utils",
                },
            }
            values.update(override)
            self.assertEqual(build_gate0_result(**values)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
