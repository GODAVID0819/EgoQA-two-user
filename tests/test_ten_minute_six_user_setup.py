from __future__ import annotations

import sys
import types
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.group_relative_clip_sampling import (  # noqa: E402
    SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
)
from egolife_two_user_qa.ten_minute_six_user_setup import (  # noqa: E402
    collect_processed_evidence,
    question_fingerprint,
    prepare_ten_minute_six_user,
    representative_six_user_packet,
    six_user_context_plan,
    validate_fresh_output,
    validate_prompt_contract,
    write_memory_safe_context_script,
)
from egolife_two_user_qa import ten_minute_six_user_setup  # noqa: E402
from egolife_two_user_qa.qwen3vl_runner import (  # noqa: E402
    QWEN_VISION_TOKEN_PIXEL_AREA,
    Qwen3VLMemorySafeTransformersRunner,
    memory_safe_image_max_pixels,
)


class TenMinuteSixUserSetupTests(unittest.TestCase):
    def test_context_plan_preserves_frames_and_adapts_generator_resolution(self) -> None:
        plan = six_user_context_plan()

        self.assertEqual(plan["users_per_packet"], 6)
        self.assertEqual(plan["duration_seconds_per_user"], 600.0)
        self.assertEqual(plan["aggregate_media_seconds"], 3_600.0)
        self.assertEqual(plan["clip_analysis_sample_fps"], 1.0)
        self.assertEqual(plan["pruning_sample_fps"], 1.0)
        self.assertEqual(plan["pruning_sampled_frames_per_user"], 600)
        self.assertEqual(plan["pruning_aggregate_sampled_frames"], 3_600)
        self.assertEqual(plan["pruning_cluster_window_seconds"], 30.0)
        self.assertEqual(plan["pruning_clusters_per_window"], 12)
        self.assertEqual(plan["judge_video_fps"], 0.75)
        self.assertEqual(plan["judge_review_mode"], "per_user_source_segment_map_reduce")
        self.assertEqual(plan["judge_visual_scope"], "one_user_per_visual_call")
        self.assertEqual(plan["judge_source_segments_per_visual_call"], 20)
        self.assertEqual(plan["estimated_judge_frames_per_visual_call"], 450)
        self.assertEqual(plan["generator_aggregate_frame_budget"], 3_600)
        self.assertEqual(plan["largest_visual_frame_count"], 3_600)
        self.assertEqual(
            plan["generator_worst_case_adaptive_max_image_pixels"], 44_688
        )
        self.assertEqual(plan["generator_worst_case_tokens_per_frame"], 57)
        self.assertEqual(plan["estimated_generator_input_tokens"], 220_592)
        self.assertEqual(plan["estimated_judge_aggregate_frames"], 2_700)
        self.assertEqual(plan["estimated_judge_input_tokens"], 36_992)
        self.assertEqual(plan["estimated_input_tokens"], 220_592)
        self.assertEqual(plan["max_input_tokens"], 262_144)
        self.assertEqual(plan["context_target_tokens"], 222_822)
        self.assertLess(plan["estimated_input_tokens"], plan["max_input_tokens"])

    def test_adaptive_image_budget_keeps_all_images_under_the_target(self) -> None:
        for image_count in (1, 1_850, 3_600):
            cap = memory_safe_image_max_pixels(
                image_count=image_count,
                configured_max_image_pixels=65_536,
                max_input_tokens=262_144,
            )
            estimated_tokens = (
                8_192
                + image_count * 2
                + image_count * (cap // QWEN_VISION_TOKEN_PIXEL_AREA)
            )
            self.assertLessEqual(estimated_tokens, int(262_144 * 0.85))
        self.assertEqual(
            memory_safe_image_max_pixels(
                image_count=1,
                configured_max_image_pixels=65_536,
                max_input_tokens=262_144,
            ),
            65_536,
        )

    def test_memory_safe_runner_uses_actual_image_count_for_pixel_cap(self) -> None:
        runner = object.__new__(Qwen3VLMemorySafeTransformersRunner)
        runner.adaptive_image_pixels = True
        runner.max_image_pixels = 65_536
        runner.max_input_tokens = 262_144
        runner.image_context_target_fraction = 0.85
        runner.image_text_token_reserve = 8_192
        runner.image_item_token_overhead = 2

        self.assertEqual(runner._image_max_pixels_for_call(3_600), 44_688)
        self.assertEqual(runner._image_max_pixels_for_call(1_850), 65_536)

    def test_representative_packet_preserves_asker_and_uses_frames_only(self) -> None:
        packet = representative_six_user_packet()
        clips = packet["clips"]

        self.assertEqual(len(clips), 6)
        self.assertEqual(
            packet["speaker_consensus_pruning"]["temporal_policy"],
            SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
        )
        self.assertFalse(
            packet["speaker_consensus_pruning"]["mutual_nearest_only"]
        )
        self.assertTrue(
            packet["speaker_consensus_pruning"]["asker_preserved"]
        )
        self.assertEqual(
            packet["speaker_consensus_pruning"]["pruned_side"],
            "providers_only",
        )
        self.assertEqual(
            packet["speaker_consensus_pruning"]["cluster_window_seconds"], 30.0
        )
        self.assertEqual(
            packet["speaker_consensus_pruning"]["cluster_count_per_window"], 12
        )
        self.assertFalse(clips[0]["is_pruned"])
        self.assertEqual(
            clips[0]["generator_media_mode"], "all_clustering_frames_only"
        )
        self.assertTrue(all("local_video" not in clip for clip in clips))
        self.assertTrue(all(clip["force_frame_inputs"] for clip in clips))
        self.assertTrue(
            all(
                clip["context_sampling"]["policy"]
                == "complete_surviving_sampled_frames"
                for clip in clips
            )
        )

    def test_preflight_injects_five_optional_examples(self) -> None:
        report = validate_prompt_contract()

        self.assertTrue(report["passed"])
        self.assertEqual(report["selection_count"], 5)
        self.assertEqual(report["long_horizon_example_count"], 5)
        self.assertGreater(report["judge_map_prompt_characters"], 0)
        self.assertGreater(report["judge_reduce_prompt_characters"], 0)
        self.assertEqual(
            set(report["legacy_judge_prompt_characters"]),
            {
                "qa_formality",
                "evidence_groundedness",
                "answerability_speaker_only",
                "answerability_combined_all_six_users",
            },
        )
        self.assertTrue(
            all(
                characters > 0
                for characters in report["legacy_judge_prompt_characters"].values()
            )
        )
        self.assertEqual(
            report["temporal_policy"], SIX_USER_TIME_AWARE_TEMPORAL_POLICY
        )

    def test_generated_context_shell_script_is_lf_only(self) -> None:
        tmp_dir = ROOT / "tmp" / f"context_script_{uuid4().hex}"
        tmp_dir.mkdir()
        path = tmp_dir / "qwen_memory_safe_six_user_10min.sh"
        try:
            write_memory_safe_context_script(six_user_context_plan(), tmp_dir)
            raw = path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
            tmp_dir.rmdir()

        self.assertTrue(raw.startswith(b"# Generated six-user"))
        self.assertNotIn(b"\r", raw)
        self.assertIn(b"QWEN_MEMORY_SAFE_ADAPTIVE_IMAGE_PIXELS=1", raw)
        self.assertIn(b"QWEN_MEMORY_SAFE_VIDEO_FPS=0.75", raw)
        self.assertIn(b"QWEN_MEMORY_SAFE_MAX_INPUT_TOKENS=262144", raw)
        self.assertIn(
            b"QWEN_MEMORY_SAFE_IMAGE_CONTEXT_TARGET_FRACTION=0.85", raw
        )
        self.assertIn(b"QWEN_MEMORY_SAFE_MIN_AVAILABLE_RAM_GIB=160", raw)
        self.assertIn(b"QWEN_MEMORY_SAFE_DEVICE_MAP=cuda", raw)
        self.assertIn(b"MALLOC_ARENA_MAX=2", raw)

    def test_prepare_wires_exact_time_aware_pruning_and_sampling_contract(self) -> None:
        prepared_packet = representative_six_user_packet()
        prepared_packet["group_relative_clip_similarity"] = {}
        source_packet = {
            "evidence_id": "source-window",
            "day": "2025-01-01",
            "time_token": "120000",
            "clip_clock": "12:00:00",
            "clips": [{"agent_name": f"user_{index}"} for index in range(6)],
        }
        analysis = {"speaker_candidates": [{"candidate": 1}]}

        class OnePacketThenFail:
            def __init__(self) -> None:
                self.next_calls = 0

            def __iter__(self):
                return self

            def __next__(self):
                self.next_calls += 1
                if self.next_calls > 1:
                    raise AssertionError(
                        "lazy preparation requested another source after reaching target"
                    )
                return source_packet

        source_iterator = OnePacketThenFail()

        with (
            mock.patch.object(Path, "mkdir"),
            mock.patch.object(
                ten_minute_six_user_setup,
                "iter_evidence_packets",
                return_value=source_iterator,
            ) as evidence_iterator_mock,
            mock.patch.object(
                ten_minute_six_user_setup,
                "TransformersClipEncoder",
                return_value=SimpleNamespace(model_id="mock-clip"),
            ),
            mock.patch.object(
                ten_minute_six_user_setup,
                "analyze_group_relative_similarity",
                return_value=analysis,
            ) as analyze_mock,
            mock.patch.object(
                ten_minute_six_user_setup,
                "build_candidate_packet",
                return_value=prepared_packet,
            ),
            mock.patch.object(
                ten_minute_six_user_setup,
                "write_memory_safe_context_script",
                return_value="context.sh",
            ),
            mock.patch.object(ten_minute_six_user_setup, "write_json"),
            mock.patch.object(ten_minute_six_user_setup, "write_jsonl"),
            mock.patch.object(
                ten_minute_six_user_setup,
                "write_prompt_previews",
                return_value={"neutral": "generator_prompt_neutral.txt"},
            ),
        ):
            summary = prepare_ten_minute_six_user(
                manifest_path=ROOT / "unused_manifest.json",
                output_root=ROOT / "unused_output",
                cache_dir=ROOT / "unused_cache",
                target_count=1,
                source_window_count=1,
            )

        kwargs = analyze_mock.call_args.kwargs
        self.assertEqual(
            evidence_iterator_mock.call_args.kwargs["excluded_group_keys"], set()
        )
        self.assertTrue(
            evidence_iterator_mock.call_args.kwargs["skip_failed_groups"]
        )
        self.assertEqual(source_iterator.next_calls, 1)
        self.assertEqual(kwargs["duration_seconds"], 600.0)
        self.assertEqual(kwargs["sample_interval_seconds"], 1.0)
        self.assertEqual(kwargs["pruning_clusters_per_video"], 12)
        self.assertEqual(kwargs["max_pair_time_difference_seconds"], 30.0)
        self.assertFalse(kwargs["mutual_nearest_only"])
        self.assertTrue(kwargs["split_noncontiguous_clusters"])
        self.assertEqual(kwargs["max_cluster_member_gap_seconds"], 1.5)
        self.assertEqual(kwargs["cluster_window_seconds"], 30.0)
        self.assertEqual(kwargs["generator_frame_budget"], 3_600)
        self.assertEqual(kwargs["pruning_protection_mode"], "min_percent")
        self.assertEqual(kwargs["min_pruned_video_percent"], 40.0)
        self.assertTrue(summary["time_aware_pruning"]["speaker_preserved"])
        self.assertEqual(summary["source_window_limit"], 1)
        self.assertEqual(summary["source_window_count"], 1)
        self.assertEqual(
            summary["acceleration"],
            {
                "processed_windows_filtered_before_download_assembly_and_clip": True,
                "lazy_source_window_preparation": True,
                "ffmpeg_processes_per_sampled_video": 1,
                "clusters_computed_once_per_video_and_reused_for_all_askers": True,
                "full_judge_video_materialization": "hardlink_with_copy_fallback",
                "per_user_source_segment_map_reduce_judges": True,
                "pruning_contract_changed": False,
            },
        )
        self.assertEqual(
            summary["media_routing"]["generator"],
            (
                "complete 1 FPS asker samples and complete surviving "
                "provider-cluster samples"
            ),
        )
        self.assertIn("six independent 600-second", summary["media_routing"]["groundedness"])
        self.assertIn("one shared fact plan", summary["media_routing"]["answerability"])

    def test_processed_history_collects_windows_and_blocks_question_reuse(self) -> None:
        root = ROOT / "tmp" / f"freshness_{uuid4().hex}"
        prior = root / "prior"
        prior_candidates = prior / "time_aware_candidates"
        prior_candidates.mkdir(parents=True)
        candidate_path = prior_candidates / "six_user_10min_time_aware.jsonl"
        candidate_path.write_text(
            json.dumps(
                {
                    "evidence_id": "EGOLIFE6U_CONSENSUS_DAY3_12300000_S1",
                    "day": "DAY3",
                    "time_token": "12300000",
                    "question": "Which item did the other participant bring?",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        accepted_path = prior / "qa_mcq.jsonl"
        accepted_path.write_text(
            json.dumps(
                {"question": "Where was the object placed after I left?"}
            )
            + "\n",
            encoding="utf-8",
        )
        freshness_path = root / "freshness.json"
        fresh_qa_path = root / "fresh.jsonl"
        repeated_qa_path = root / "repeated.jsonl"
        def cleanup() -> None:
            for path in (
                candidate_path,
                accepted_path,
                freshness_path,
                fresh_qa_path,
                repeated_qa_path,
            ):
                path.unlink(missing_ok=True)
            prior_candidates.rmdir()
            prior.rmdir()
            root.rmdir()

        self.addCleanup(cleanup)

        processed = collect_processed_evidence([prior])

        self.assertEqual(processed["window_keys"], [("DAY3", "12300000")])
        self.assertEqual(processed["scanned_file_count"], 2)
        self.assertEqual(len(processed["question_hashes"]), 2)
        freshness_path.write_text(
            json.dumps({"prior_question_hashes": processed["question_hashes"]}),
            encoding="utf-8",
        )
        fresh_qa_path.write_text(
            json.dumps({"question": "What happened to the container next?"}) + "\n",
            encoding="utf-8",
        )
        repeated_qa_path.write_text(
            json.dumps(
                {"question": "  WHERE was the object placed after I left?  "}
            )
            + "\n",
            encoding="utf-8",
        )

        report = validate_fresh_output(
            qa_path=fresh_qa_path,
            freshness_manifest_path=freshness_path,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(report["question_count"], 1)
        self.assertIn(
            question_fingerprint("Where was the object placed after I left?"),
            processed["question_hashes"],
        )
        with self.assertRaisesRegex(RuntimeError, "prior_duplicates=1"):
            validate_fresh_output(
                qa_path=repeated_qa_path,
                freshness_manifest_path=freshness_path,
            )


if __name__ == "__main__":
    unittest.main()
