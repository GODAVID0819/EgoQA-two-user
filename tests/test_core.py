from __future__ import annotations

from contextlib import contextmanager
import shutil
import unittest
import json
import uuid
from pathlib import Path
from unittest.mock import patch

from egolife_two_user_qa.candidate_mining import mine_candidates
from egolife_two_user_qa.clip_gap_demo import (
    TransformersClipEncoder,
    cluster_embedding_medoids,
    mine_anchors_and_gaps,
    random_window_starts,
    run_clip_gap_demo,
    summarize_trial,
)
from egolife_two_user_qa.clip_exclusive_mining import (
    mine_clip_exclusive_candidates,
    summarize_exclusiveness,
)
from egolife_two_user_qa.evidence import (
    choose_required_clips,
    group_manifest_clips,
    select_evidence_groups,
    summarize_gaze_csv,
)
from egolife_two_user_qa.gaze_projection import gaussian_bbox_score, load_aria_projection_calibration, project_gaze_row
from egolife_two_user_qa.manifest import parse_egolife_path, seconds_from_time_token
from egolife_two_user_qa.prompts import build_relation_discovery_prompt, build_video_generation_prompt
from egolife_two_user_qa.qwen3vl_runner import DryRunRunner, normalize_video_kwargs, split_video_inputs_and_metadata
from egolife_two_user_qa.review_media import materialize_review_videos
from egolife_two_user_qa.schema import extract_json_object, validate_qa_item, write_human_review_sheet
from egolife_two_user_qa.video_qa_loop import (
    answerability_gate,
    build_review_from_gates,
    complete_generator_metadata,
    dry_run_qa,
    generate_video_qa_loop,
    judge_gate,
    qa_for_judger_prompt,
)


@contextmanager
def workspace_temp_dir():
    root = Path(__file__).resolve().parents[1] / "tmp" / "test_runs"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ManifestTests(unittest.TestCase):
    def test_parse_video_path(self) -> None:
        parsed = parse_egolife_path("A1_JAKE/DAY1/DAY1_A1_JAKE_11094208.mp4")
        self.assertEqual(parsed.day, "DAY1")
        self.assertEqual(parsed.agent_dir, "A1_JAKE")
        self.assertEqual(parsed.agent_name, "Jake")
        self.assertEqual(parsed.time_token, "11094208")
        self.assertEqual(parsed.clip_clock, "11:09:42.08")

    def test_seconds_from_time_token(self) -> None:
        self.assertAlmostEqual(seconds_from_time_token("11094208"), 40182.08)

    def test_group_manifest_clips(self) -> None:
        manifest = {
            "clips": [
                {"day": "DAY1", "time_token": "11100000", "agent_dir": "A1_JAKE"},
                {"day": "DAY1", "time_token": "11100000", "agent_dir": "A2_ALICE"},
                {"day": "DAY1", "time_token": "11103000", "agent_dir": "A1_JAKE"},
            ]
        }
        groups = group_manifest_clips(manifest)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["agents"], ["A1_JAKE", "A2_ALICE"])


class EvidenceTests(unittest.TestCase):
    def test_select_evidence_groups_stratifies_across_days(self) -> None:
        groups = [
            {"day": day, "time_token": f"{index:08d}", "clips": []}
            for day in ["DAY1", "DAY2", "DAY3", "DAY4", "DAY5"]
            for index in range(3)
        ]
        selected = select_evidence_groups(
            groups,
            target_count=5,
            random_seed=42,
            stratify_by_day=True,
        )
        self.assertEqual({group["day"] for group in selected}, {"DAY1", "DAY2", "DAY3", "DAY4", "DAY5"})

    def test_choose_required_clips_can_randomize_user_pair(self) -> None:
        group = {
            "clips": [
                {"agent_dir": "A1_JAKE"},
                {"agent_dir": "A2_ALICE"},
                {"agent_dir": "A3_TASHA"},
                {"agent_dir": "A4_LUCIA"},
            ]
        }
        selected = choose_required_clips(group, 2, rng=__import__("random").Random(7))
        self.assertEqual(len(selected), 2)
        self.assertNotEqual([clip["agent_dir"] for clip in selected], ["A1_JAKE", "A2_ALICE"])

    def test_summarize_gaze_csv(self) -> None:
        with workspace_temp_dir() as tmp:
            path = Path(tmp) / "gaze.csv"
            path.write_text(
                "tracking_timestamp_us,left_yaw_rads_cpf,right_yaw_rads_cpf,pitch_rads_cpf,depth_m\n"
                "1,0.1,0.3,-0.2,1.5\n"
                "2,0.2,0.4,-0.1,2.0\n",
                encoding="utf-8",
            )
            summary = summarize_gaze_csv(path)
        self.assertEqual(summary["row_count"], 2)
        self.assertEqual(summary["yaw_rads_summary"]["median"], 0.25)
        self.assertEqual(summary["depth_m_summary"]["max"], 2.0)
        self.assertEqual(summary["projection_status"], "missing_calibration")
        self.assertIsNone(summary["projected_gaze_summary"])

    def test_project_gaze_with_explicit_calibration(self) -> None:
        with workspace_temp_dir() as tmp:
            calibration_path = Path(tmp) / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "camera": {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
                        "T_camera_cpf": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            calibration = load_aria_projection_calibration(calibration_path)
            projected = project_gaze_row(
                {
                    "tracking_timestamp_us": "1",
                    "left_yaw_rads_cpf": "0.0",
                    "right_yaw_rads_cpf": "0.0",
                    "pitch_rads_cpf": "0.0",
                    "depth_m": "2.0",
                },
                calibration,
            )
        self.assertIsNotNone(projected)
        self.assertEqual(projected["x"], 320.0)
        self.assertEqual(projected["y"], 240.0)
        self.assertTrue(projected["in_frame"])

    def test_summarize_gaze_projects_only_with_calibration(self) -> None:
        with workspace_temp_dir() as tmp:
            gaze_path = Path(tmp) / "gaze.csv"
            gaze_path.write_text(
                "tracking_timestamp_us,left_yaw_rads_cpf,right_yaw_rads_cpf,pitch_rads_cpf,depth_m\n"
                "1,0.0,0.0,0.0,2.0\n",
                encoding="utf-8",
            )
            calibration_path = Path(tmp) / "calibration.json"
            calibration_path.write_text(
                json.dumps(
                    {
                        "camera": {"fx": 100.0, "fy": 100.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480},
                        "T_camera_cpf": [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = summarize_gaze_csv(gaze_path, calibration_path=calibration_path)
        self.assertEqual(summary["projection_status"], "projected")
        self.assertEqual(summary["projected_gaze_summary"]["median_x"], 320.0)
        self.assertEqual(summary["projected_gaze_summary"]["median_y"], 240.0)

    def test_gaussian_bbox_score_prefers_near_center(self) -> None:
        near = gaussian_bbox_score((10.0, 10.0), (8.0, 8.0, 12.0, 12.0), sigma=10.0)
        far = gaussian_bbox_score((10.0, 10.0), (100.0, 100.0, 120.0, 120.0), sigma=10.0)
        self.assertGreater(near, far)


class CandidateMiningTests(unittest.TestCase):
    def test_mine_candidates_from_complementary_observations(self) -> None:
        rows = [
            {
                "clip_id": "DAY1_A1_JAKE_11100000",
                "clip": {
                    "clip_id": "DAY1_A1_JAKE_11100000",
                    "day": "DAY1",
                    "agent_dir": "A1_JAKE",
                    "agent_id": "A1",
                    "agent_name": "Jake",
                    "time_token": "11100000",
                    "clip_clock": "11:10:00.00",
                    "clock_seconds": 40200.0,
                    "video_url": "video_a",
                    "gaze_url": "gaze_a",
                    "frames": [],
                    "gaze_summary": {},
                },
                "observation": {
                    "status": "ok",
                    "location_guess": "kitchen table",
                    "visible_people": ["Alice"],
                    "salient_objects": ["red mug", "table"],
                    "actions": ["Jake sees Alice pick up the red mug"],
                    "gaze_focus": ["red mug"],
                    "key_facts": ["Alice picks up the red mug near the kitchen table"],
                },
            },
            {
                "clip_id": "DAY1_A2_ALICE_11100000",
                "clip": {
                    "clip_id": "DAY1_A2_ALICE_11100000",
                    "day": "DAY1",
                    "agent_dir": "A2_ALICE",
                    "agent_id": "A2",
                    "agent_name": "Alice",
                    "time_token": "11100000",
                    "clip_clock": "11:10:00.00",
                    "clock_seconds": 40200.0,
                    "video_url": "video_b",
                    "gaze_url": "gaze_b",
                    "frames": [],
                    "gaze_summary": {},
                },
                "observation": {
                    "status": "ok",
                    "location_guess": "kitchen table",
                    "visible_people": ["Jake"],
                    "salient_objects": ["red mug", "sink"],
                    "actions": ["Alice places the red mug beside the sink"],
                    "gaze_focus": ["sink"],
                    "key_facts": ["The red mug ends up beside the sink"],
                },
            },
        ]
        with workspace_temp_dir() as tmp:
            obs_path = Path(tmp) / "observations.jsonl"
            out_path = Path(tmp) / "candidates.jsonl"
            obs_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            candidates = mine_candidates(
                observations_path=obs_path,
                output_path=out_path,
                target_count=1,
                min_score=0,
            )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidate_type"], "semantic_complementarity")
        self.assertEqual(candidates[0]["required_users"], ["Jake", "Alice"])
        self.assertIn("complementarity", candidates[0])


class ClipGapDemoTests(unittest.TestCase):
    class _FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def norm(self, *args, **kwargs):
            return self

    class _FakeProjection:
        in_features = 768
        out_features = 512

        def __init__(self):
            self.called = False

        def __call__(self, tensor):
            self.called = True
            return ClipGapDemoTests._FakeTensor((tensor.shape[0], self.out_features))

    class _FakeModelOutput:
        def __init__(self, pooled):
            self.pooler_output = pooled

    def test_transformers_clip_encoder_accepts_projected_pooler_output(self) -> None:
        projection = self._FakeProjection()
        encoder = TransformersClipEncoder.__new__(TransformersClipEncoder)
        encoder.model = type("FakeModel", (), {"visual_projection": projection})()

        features = encoder._coerce_image_features(
            self._FakeModelOutput(self._FakeTensor((2, 512)))
        )

        self.assertEqual(features.shape, (2, 512))
        self.assertFalse(projection.called)

    def test_transformers_clip_encoder_projects_hidden_pooler_output(self) -> None:
        projection = self._FakeProjection()
        encoder = TransformersClipEncoder.__new__(TransformersClipEncoder)
        encoder.model = type("FakeModel", (), {"visual_projection": projection})()

        features = encoder._coerce_image_features(
            self._FakeModelOutput(self._FakeTensor((2, 768)))
        )

        self.assertEqual(features.shape, (2, 512))
        self.assertTrue(projection.called)

    def test_random_window_starts_are_reproducible_and_distinct(self) -> None:
        starts = random_window_starts(
            max_start_seconds=18.0,
            trial_count=5,
            sample_interval_seconds=1.5,
            seed=42,
        )
        self.assertEqual(starts, [0.0, 3.0, 9.0, 10.5, 15.0])
        self.assertEqual(len(starts), len(set(starts)))

    def test_cluster_embedding_medoids_groups_near_duplicates(self) -> None:
        embeddings = [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ]
        labels, medoids = cluster_embedding_medoids(embeddings, 2)
        self.assertEqual(len(set(labels)), 2)
        self.assertEqual(len(medoids), 2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])

    def test_mine_anchors_and_gaps_finds_shared_and_unique_frames(self) -> None:
        left_frames = [
            {"path": "alice_shared.jpg", "timestamp_seconds": 1.0},
            {"path": "alice_unique.jpg", "timestamp_seconds": 4.0},
        ]
        right_frames = [
            {"path": "bob_shared.jpg", "timestamp_seconds": 1.2},
            {"path": "bob_unique.jpg", "timestamp_seconds": 5.0},
        ]
        result = mine_anchors_and_gaps(
            left_frames,
            right_frames,
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.99, 0.01, 0.0], [0.0, 0.0, 1.0]],
            anchor_threshold=0.9,
            top_k=2,
        )
        self.assertEqual(len(result["anchors"]), 1)
        self.assertEqual(result["anchors"][0]["left_frame"]["path"], "alice_shared.jpg")
        self.assertEqual(result["anchors"][0]["right_frame"]["path"], "bob_shared.jpg")
        self.assertEqual(result["left_evidence_gaps"][0]["left_frame"]["path"], "alice_unique.jpg")
        self.assertEqual(result["right_evidence_gaps"][0]["right_frame"]["path"], "bob_unique.jpg")
        self.assertNotIn(
            "alice_shared.jpg",
            [row["left_frame"]["path"] for row in result["left_evidence_gaps"]],
        )
        self.assertTrue(result["left_novelty_ranked"][1]["is_anchor"])

    def test_run_demo_with_existing_frames_and_fake_encoder(self) -> None:
        class FakeEncoder:
            model_id = "fake-clip"

            def encode(self, image_paths):
                lookup = {
                    "alice_shared.jpg": [1.0, 0.0, 0.0],
                    "alice_unique.jpg": [0.0, 1.0, 0.0],
                    "bob_shared.jpg": [0.99, 0.01, 0.0],
                    "bob_unique.jpg": [0.0, 0.0, 1.0],
                }
                return [lookup[Path(path).name] for path in image_paths]

        packet = {"evidence_id": "toy", "clips": [{}, {}]}
        frame_rows = [
            {
                "user": "Alice",
                "clip": {},
                "frames": [
                    {"path": "alice_shared.jpg", "timestamp_seconds": 1.0},
                    {"path": "alice_unique.jpg", "timestamp_seconds": 4.0},
                ],
            },
            {
                "user": "Bob",
                "clip": {},
                "frames": [
                    {"path": "bob_shared.jpg", "timestamp_seconds": 1.2},
                    {"path": "bob_unique.jpg", "timestamp_seconds": 5.0},
                ],
            },
        ]
        with (
            patch("egolife_two_user_qa.clip_gap_demo.load_evidence_packet", return_value=packet),
            patch("egolife_two_user_qa.clip_gap_demo.packet_frames", return_value=frame_rows),
            patch("egolife_two_user_qa.clip_gap_demo.write_json") as write_json_mock,
            patch("egolife_two_user_qa.clip_gap_demo.write_contact_sheet") as sheet_mock,
        ):
            result = run_clip_gap_demo(
                evidence_path="evidence.jsonl",
                output_dir="output",
                duration_seconds=10.0,
                clusters_per_user=2,
                anchor_threshold=0.9,
                resample_videos=False,
                encoder=FakeEncoder(),
            )
        self.assertEqual(result["anchors"][0]["left_frame"]["path"], "alice_shared.jpg")
        write_json_mock.assert_called_once()
        sheet_mock.assert_called_once()

    def test_summarize_trial_reports_anchor_and_gap_extremes(self) -> None:
        summary = summarize_trial(
            {
                "left_user": "Jake",
                "right_user": "Alice",
                "window": {"start_seconds": 3.0, "duration_seconds": 12.0},
                "anchors": [{"similarity": 0.8}, {"similarity": 0.9}],
                "left_evidence_gaps": [{"novelty": 0.2}, {"novelty": 0.1}],
                "right_evidence_gaps": [{"novelty": 0.3}],
            },
            2,
        )
        self.assertEqual(summary["end_seconds"], 15.0)
        self.assertEqual(summary["anchor_count"], 2)
        self.assertIsNone(summary["mean_cross_user_similarity"])
        self.assertEqual(summary["max_anchor_similarity"], 0.9)
        self.assertEqual(summary["mean_anchor_similarity"], 0.85)
        self.assertEqual(summary["max_Jake_novelty"], 0.2)
        self.assertEqual(summary["max_Alice_novelty"], 0.3)
        self.assertEqual(summary["largest_user_novelty"], 0.3)
        self.assertEqual(summary["review_priority"], 0.27)

    def test_summarize_exclusiveness_combines_global_and_user_novelty(self) -> None:
        summary = summarize_exclusiveness(
            {
                "left_user": "Jake",
                "right_user": "Alice",
                "similarity_matrix": [[0.2, 0.4], [0.6, 0.8]],
                "left_evidence_gaps": [{"novelty": 0.7}],
                "right_evidence_gaps": [{"novelty": 0.5}],
                "anchors": [],
            }
        )
        self.assertEqual(summary["mean_cross_user_similarity"], 0.5)
        self.assertEqual(summary["cross_user_dissimilarity"], 0.5)
        self.assertEqual(summary["largest_user_novelty"], 0.7)
        self.assertEqual(summary["score"], 0.6)

    def test_mine_clip_exclusive_candidates_ranks_packets(self) -> None:
        class FakeEncoder:
            model_id = "fake-clip"

            def encode(self, image_paths):
                lookup = {
                    "p1_left.jpg": [1.0, 0.0, 0.0],
                    "p1_right.jpg": [0.0, 1.0, 0.0],
                    "p2_left.jpg": [1.0, 0.0, 0.0],
                    "p2_right.jpg": [0.99, 0.01, 0.0],
                }
                return [lookup[Path(path).name] for path in image_paths]

        with workspace_temp_dir() as tmp:
            root = Path(tmp)
            for name in ["p1_left.jpg", "p1_right.jpg", "p2_left.jpg", "p2_right.jpg"]:
                (root / name).write_bytes(b"fake")
            packets = [
                {
                    "evidence_id": "packet_less_exclusive",
                    "day": "DAY1",
                    "time_token": "11103000",
                    "required_users": ["Jake", "Alice"],
                    "clips": [
                        {"agent_name": "Jake", "frames": [{"path": str(root / "p2_left.jpg"), "timestamp_seconds": 1.0}]},
                        {"agent_name": "Alice", "frames": [{"path": str(root / "p2_right.jpg"), "timestamp_seconds": 1.0}]},
                    ],
                },
                {
                    "evidence_id": "packet_more_exclusive",
                    "day": "DAY1",
                    "time_token": "11100000",
                    "required_users": ["Jake", "Alice"],
                    "clips": [
                        {"agent_name": "Jake", "frames": [{"path": str(root / "p1_left.jpg"), "timestamp_seconds": 1.0}]},
                        {"agent_name": "Alice", "frames": [{"path": str(root / "p1_right.jpg"), "timestamp_seconds": 1.0}]},
                    ],
                },
            ]
            evidence_path = root / "evidence.jsonl"
            evidence_path.write_text(
                "\n".join(json.dumps(packet) for packet in packets) + "\n",
                encoding="utf-8",
            )
            output_path = root / "ranked.jsonl"
            rows = mine_clip_exclusive_candidates(
                evidence_path=evidence_path,
                output_path=output_path,
                output_dir=root / "mined",
                clusters_per_user=1,
                top_k=1,
                contact_sheet_count=0,
                resample_videos=False,
                encoder=FakeEncoder(),
            )
            preserved_rows = mine_clip_exclusive_candidates(
                evidence_path=evidence_path,
                output_path=root / "preserved.jsonl",
                output_dir=root / "preserved_mined",
                clusters_per_user=1,
                top_k=1,
                contact_sheet_count=0,
                resample_videos=False,
                preserve_order=True,
                encoder=FakeEncoder(),
            )

        self.assertEqual([row["evidence_id"] for row in rows], ["packet_more_exclusive", "packet_less_exclusive"])
        self.assertEqual(rows[0]["clip_exclusiveness"]["rank"], 1)
        self.assertGreater(rows[0]["clip_exclusiveness"]["score"], rows[1]["clip_exclusiveness"]["score"])
        self.assertIn("Jake", rows[0]["clip_exclusiveness"]["exclusive_frames_by_user"])
        self.assertEqual(
            [row["evidence_id"] for row in preserved_rows],
            ["packet_less_exclusive", "packet_more_exclusive"],
        )
        self.assertEqual([row["clip_exclusiveness"]["rank"] for row in preserved_rows], [2, 1])


class SchemaTests(unittest.TestCase):
    def passed_judge_checks(self):
        return {
            name: {"status": "PASS", "reason": "ok", "fix": ""}
            for name in [
                "first_person_naturalness",
                "agent_perspective",
                "source_scope",
                "question_type_semantics",
                "multi_video_necessity",
                "visual_grounding",
                "mcq_option_quality",
                "gaze_safety",
                "human_auditability",
            ]
        }

    def valid_item(self):
        judger = {
            "review_passed": True,
            "checks": self.passed_judge_checks(),
            "blocking_failures": [],
            "feedback_to_generator": "",
            "gate": {"passed": True, "reason": "all structured judger checks passed", "failed_checks": []},
        }
        answerability = {
            "evaluations": [
                {"condition_id": "single_user::Jake", "condition_type": "single_user", "choice": "insufficient"},
                {"condition_id": "single_user::Alice", "condition_type": "single_user", "choice": "B"},
                {"condition_id": "combined_all_users::Jake+Alice", "condition_type": "combined_all_users", "choice": "A"},
            ],
            "gate": {"passed": True, "reason": "combined correct and singles insufficient or wrong"},
        }
        return {
            "qa_id": "QA_001",
            "question": "What did we put near the table?",
            "options": ["A cup", "A plate", "A book", "A phone", "A key"],
            "correct": "A",
            "answer": "A cup",
            "category": "environmental_interaction",
            "required_users": ["Jake", "Alice"],
            "evidence": [
                {"user": "Jake", "needed_fact": "saw the cup", "frames_used": ["f1"]},
                {"user": "Alice", "needed_fact": "saw the table", "frames_used": ["f2"]},
            ],
            "single_user_answerability": {
                "Jake": "insufficient because he only saw the object",
                "Alice": "insufficient because she only saw the destination",
            },
            "combined_answerability": "sufficient because together they support the answer",
            "review": {
                "status": "passed",
                "review_passed": True,
                "judger": judger,
                "answerability": answerability,
                "schema_validation": {"passed": True, "errors": []},
                "final_decision": {
                    "accepted": True,
                    "rejection_stage": None,
                    "reason": "passed all gates",
                },
            },
            "question_type": "commonality",
            "generator_rationale": "The question is natural and grounded in both users' views.",
            "why_two_users_needed": "Jake and Alice each provide a necessary visual fact.",
            "per_user_evidence_claims": [
                {"user": "Jake", "claim": "Jake saw the cup"},
                {"user": "Alice", "claim": "Alice saw the table"},
            ],
            "attempt_count": 1,
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "source_urls": {"videos": []},
            "video_evidence": [
                {
                    "user": "Jake",
                    "day": "DAY1",
                    "time_token": "11100000",
                    "video_url": "video_a",
                    "local_video": "jake.mp4",
                    "sampled_frames": [],
                }
            ],
            "referred_timestamps": [],
            "human_audit": {"evidence_id": "E1", "video_evidence": []},
            "generation_trace": [
                {
                    "attempt": 1,
                    "generation": {"prompt": "p", "raw_output": "{}"},
                    "judge": {"prompt": "j", "raw_output": "{}"},
                    "answerability": {},
                    "result": {"accepted": True},
                }
            ],
        }

    def test_validate_valid_item(self) -> None:
        self.assertEqual(validate_qa_item(self.valid_item(), strict_review=True), [])

    def test_validate_requires_two_users(self) -> None:
        item = self.valid_item()
        item["required_users"] = ["Jake"]
        errors = validate_qa_item(item)
        self.assertTrue(any("at least two" in error for error in errors))

    def test_extract_json_object_from_codeblock(self) -> None:
        self.assertEqual(extract_json_object("```json\n{\"a\": 1}\n```"), {"a": 1})

    def test_strict_validation_requires_video_first_fields(self) -> None:
        item = self.valid_item()
        del item["question_type"]
        errors = validate_qa_item(item, strict_review=True)
        self.assertTrue(any("missing video-first fields" in error for error in errors))

    def test_difference_question_type_validates(self) -> None:
        item = self.valid_item()
        item["question_type"] = "difference"
        self.assertEqual(validate_qa_item(item, strict_review=True), [])

    def test_strict_validation_requires_structured_judge_checks(self) -> None:
        item = self.valid_item()
        item["review"]["judger"] = {"review_passed": True, "gate": {"passed": True}}
        errors = validate_qa_item(item, strict_review=True)
        self.assertTrue(any("review.judger.checks" in error for error in errors))

    def test_strict_validation_trusts_judge_gate_over_inconsistent_review_passed(self) -> None:
        item = self.valid_item()
        item["review"]["judger"]["review_passed"] = False
        item["review"]["judger"]["gate"] = {
            "passed": True,
            "reason": "all structured judger checks passed",
            "failed_checks": [],
            "model_review_passed": False,
            "warning": "ignored inconsistent top-level review_passed because all structured checks passed",
        }
        self.assertEqual(validate_qa_item(item, strict_review=True), [])

    def test_strict_validation_uses_review_not_top_level_review_fields(self) -> None:
        item = self.valid_item()
        self.assertNotIn("judge_feedback", item)
        self.assertNotIn("answerability_eval", item)
        self.assertEqual(validate_qa_item(item, strict_review=True), [])

    def test_strict_validation_requires_review_answerability_gate(self) -> None:
        item = self.valid_item()
        item["review"]["answerability"]["gate"] = {"passed": False, "reason": "single user leaked answer"}
        errors = validate_qa_item(item, strict_review=True)
        self.assertTrue(any("review.answerability.gate.passed" in error for error in errors))

    def test_write_human_review_sheet(self) -> None:
        with workspace_temp_dir() as tmp:
            qa_path = Path(tmp) / "qa_mcq.jsonl"
            sheet_path = Path(tmp) / "human_review_sheet.md"
            item = self.valid_item()
            qa_path.write_text(json.dumps(item) + "\n", encoding="utf-8")

            count = write_human_review_sheet(qa_path, sheet_path)

            text = sheet_path.read_text(encoding="utf-8")
            self.assertEqual(count, 1)
            self.assertIn("# EgoLife Human Review Sheet", text)
            self.assertIn(item["question"], text)
            self.assertIn(item["answer"], text)
            self.assertIn("Jake and Alice each provide a necessary visual fact.", text)
            self.assertIn("jake.mp4", text)

class VideoFirstTests(unittest.TestCase):
    def test_dry_run_runner_accepts_video_paths(self) -> None:
        raw = DryRunRunner().generate("prompt", image_paths=["a.jpg"], video_paths=["a.mp4", "b.mp4"])
        parsed = json.loads(raw)
        self.assertEqual(parsed["image_count"], 1)
        self.assertEqual(parsed["video_count"], 2)

    def test_normalize_video_kwargs_collapses_fps_list(self) -> None:
        self.assertEqual(normalize_video_kwargs({"fps": [1.0, 1.0]})["fps"], 1.0)
        self.assertEqual(normalize_video_kwargs({"fps": []})["fps"], 1.0)
        self.assertEqual(normalize_video_kwargs({"fps": 2.0})["fps"], 2.0)

    def test_split_video_inputs_and_metadata(self) -> None:
        video = object()
        metadata = {"fps": 30.0, "frames_indices": [0, 15], "total_num_frames": 60}
        videos, kwargs = split_video_inputs_and_metadata([(video, metadata)], {"fps": [1.0]})
        self.assertEqual(videos, [video])
        self.assertTrue(kwargs["return_metadata"])
        self.assertEqual(kwargs["fps"], 1.0)
        self.assertEqual(kwargs["video_metadata"][0].fps, 30.0)
        self.assertEqual(kwargs["video_metadata"][0].frames_indices, [0, 15])

    def test_video_generation_prompt_does_not_use_observation(self) -> None:
        packet = {
            "evidence_id": "E1",
            "required_users": ["Jake", "Alice"],
            "clips": [
                {
                    "agent_name": "Jake",
                    "day": "DAY1",
                    "clip_clock": "11:10:00.00",
                    "local_video": "jake.mp4",
                    "video_url": "video_a",
                    "observation": {"key_facts": ["SHOULD_NOT_APPEAR"]},
                    "gaze_summary": {"projection_status": "missing_calibration"},
                },
                {
                    "agent_name": "Alice",
                    "day": "DAY1",
                    "clip_clock": "11:10:00.00",
                    "local_video": "alice.mp4",
                    "video_url": "video_b",
                    "observation": {"key_facts": ["SHOULD_NOT_APPEAR"]},
                    "gaze_summary": {"projection_status": "missing_calibration"},
                },
            ],
        }
        prompt = build_video_generation_prompt(packet, "commonality")
        self.assertIn("Look directly at the videos", prompt)
        self.assertIn("local_video", prompt)
        self.assertNotIn("SHOULD_NOT_APPEAR", prompt)
        self.assertIn("single_user_answerability", prompt)
        self.assertIn("combined_answerability", prompt)
        self.assertIn("why_two_users_needed", prompt)

    def test_clip_guided_prompt_includes_retrieval_hints(self) -> None:
        packet = {
            "evidence_id": "E1",
            "required_users": ["Jake", "Alice"],
            "clips": [
                {"agent_name": "Jake", "day": "DAY1", "clip_clock": "11:10:00.00", "local_video": "jake.mp4"},
                {"agent_name": "Alice", "day": "DAY1", "clip_clock": "11:10:00.00", "local_video": "alice.mp4"},
            ],
            "clip_exclusiveness": {
                "model_id": "fake-clip",
                "rank": 1,
                "score": 0.7,
                "left_user": "Jake",
                "right_user": "Alice",
                "exclusive_frames_by_user": {
                    "Jake": [
                        {
                            "left_frame": {"timestamp_seconds": 4.0, "path": "jake_4.jpg"},
                            "closest_right_frame": {"timestamp_seconds": 7.0, "path": "alice_7.jpg"},
                            "novelty": 0.8,
                            "closest_similarity": 0.2,
                        }
                    ]
                },
                "anchors": [],
            },
        }

        prompt = build_video_generation_prompt(packet, "difference", generation_mode="clip_guided")

        self.assertIn("CLIP retrieval hints", prompt)
        self.assertIn("jake_4.jpg", prompt)
        self.assertIn("Verify every semantic claim from the raw videos", prompt)

    def test_discovery_prompt_is_template_free_planning_stage(self) -> None:
        packet = {
            "evidence_id": "E1",
            "required_users": ["Jake", "Alice"],
            "clips": [
                {"agent_name": "Jake", "day": "DAY1", "clip_clock": "11:10:00.00", "local_video": "jake.mp4"},
                {"agent_name": "Alice", "day": "DAY1", "clip_clock": "11:10:00.00", "local_video": "alice.mp4"},
            ],
        }

        prompt = build_relation_discovery_prompt(packet, "commonality")

        self.assertIn("Do not write the MCQ yet", prompt)
        self.assertIn("List 3-5 possible cross-user information needs", prompt)
        self.assertIn("likely_answerable_by_one_video_alone", prompt)
        self.assertNotIn("Good example", prompt)

    def test_discovery_dry_run_writes_discovery_prompt_stage(self) -> None:
        with workspace_temp_dir() as tmp:
            root = Path(tmp)
            evidence_path = root / "evidence.jsonl"
            output_path = root / "qa.jsonl"
            prompts_path = root / "prompts.jsonl"
            intermediate_path = root / "intermediate.jsonl"
            rejected_path = root / "rejected.jsonl"
            packet = {
                "evidence_id": "E1",
                "required_users": ["Jake", "Alice"],
                "source_urls": {"videos": []},
                "clips": [
                    {"agent_name": "Jake", "agent_dir": "A1_JAKE", "frames": []},
                    {"agent_name": "Alice", "agent_dir": "A2_ALICE", "frames": []},
                ],
            }
            evidence_path.write_text(json.dumps(packet) + "\n", encoding="utf-8")

            rows = generate_video_qa_loop(
                evidence_path=evidence_path,
                output_path=output_path,
                prompts_path=prompts_path,
                rejected_path=rejected_path,
                intermediate_path=intermediate_path,
                backend="transformers-local",
                target_count=1,
                dry_run=True,
                generation_mode="discovery",
            )

            prompt_rows = [json.loads(line) for line in prompts_path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["generation_mode"], "discovery")
        self.assertEqual(prompt_rows[0]["stage"], "discovery")
        self.assertEqual(prompt_rows[1]["stage"], "generation")

    def test_fixed_question_type_schedule_is_packet_order_based(self) -> None:
        with workspace_temp_dir() as tmp:
            root = Path(tmp)
            evidence_path = root / "evidence.jsonl"
            packets = [
                {
                    "evidence_id": f"E{index}",
                    "required_users": ["Jake", "Alice"],
                    "source_urls": {"videos": []},
                    "clips": [
                        {"agent_name": "Jake", "agent_dir": "A1_JAKE", "frames": []},
                        {"agent_name": "Alice", "agent_dir": "A2_ALICE", "frames": []},
                    ],
                }
                for index in range(2)
            ]
            evidence_path.write_text(
                "".join(json.dumps(packet) + "\n" for packet in packets),
                encoding="utf-8",
            )

            rows = generate_video_qa_loop(
                evidence_path=evidence_path,
                output_path=root / "qa.jsonl",
                prompts_path=root / "prompts.jsonl",
                rejected_path=root / "rejected.jsonl",
                backend="transformers-local",
                target_count=2,
                dry_run=True,
                fixed_question_type_schedule=True,
            )

        self.assertEqual([row["question_type"] for row in rows], ["commonality", "difference"])

    def test_complete_generator_metadata_repairs_old_generator_shape(self) -> None:
        packet = {"required_users": ["Jake", "Alice"]}
        qa = {
            "qa_id": "Q1",
            "question": "After I left the table, what was still happening there?",
            "options": ["food prep", "phone charging", "dish washing", "door opening", "bag packing"],
            "correct": "A",
            "answer": "wrong text",
            "required_users": ["Jake", "Alice"],
            "evidence": [
                {
                    "user": "Jake",
                    "needed_fact": "Jake left the table.",
                    "timeframe": "early in the clip",
                    "frames_used": ["around 5s"],
                }
            ],
            "model_id": "dry-run",
            "source_urls": {},
        }
        complete_generator_metadata(qa, packet=packet, question_type="commonality")
        self.assertEqual(qa["answer"], "food prep")
        self.assertEqual(qa["question_type"], "commonality")
        self.assertIn("insufficient", qa["single_user_answerability"]["Jake"])
        self.assertIn("sufficient", qa["combined_answerability"])
        self.assertEqual(validate_qa_item(qa), [])

    def test_answerability_gate_requires_combined_correct_and_singles_not_correct(self) -> None:
        qa = {"correct": "A"}
        passed = answerability_gate(
            qa,
            [
                {"condition_id": "single_user::Jake", "condition_type": "single_user", "choice": "insufficient"},
                {"condition_id": "single_user::Alice", "condition_type": "single_user", "choice": "B"},
                {"condition_id": "combined_all_users::Jake+Alice", "condition_type": "combined_all_users", "choice": "A"},
            ],
        )
        self.assertTrue(passed["passed"])
        failed = answerability_gate(
            qa,
            [
                {"condition_id": "single_user::Jake", "condition_type": "single_user", "choice": "A"},
                {"condition_id": "combined_all_users::Jake+Alice", "condition_type": "combined_all_users", "choice": "A"},
            ],
        )
        self.assertFalse(failed["passed"])

    def test_judge_gate_requires_each_structured_check_to_pass(self) -> None:
        checks = {
            name: {"status": "PASS", "reason": "ok", "fix": ""}
            for name in [
                "first_person_naturalness",
                "agent_perspective",
                "source_scope",
                "question_type_semantics",
                "multi_video_necessity",
                "visual_grounding",
                "mcq_option_quality",
                "gaze_safety",
                "human_auditability",
            ]
        }
        self.assertTrue(judge_gate({"review_passed": True, "checks": checks})["passed"])
        inconsistent = judge_gate({"review_passed": False, "checks": checks, "blocking_failures": []})
        self.assertTrue(inconsistent["passed"])
        self.assertEqual(inconsistent["model_review_passed"], False)
        self.assertIn("warning", inconsistent)
        checks["multi_video_necessity"] = {
            "status": "FAIL",
            "reason": "one video already answers",
            "fix": "ask for a complementary clue from the second user",
        }
        failed = judge_gate({"review_passed": True, "checks": checks})
        self.assertFalse(failed["passed"])
        self.assertIn("multi_video_necessity", failed["failed_checks"])

    def test_judge_gate_blocks_unnatural_non_first_person_question(self) -> None:
        checks = {
            name: {"status": "PASS", "reason": "ok", "fix": ""}
            for name in [
                "first_person_naturalness",
                "agent_perspective",
                "source_scope",
                "question_type_semantics",
                "multi_video_necessity",
                "visual_grounding",
                "mcq_option_quality",
                "gaze_safety",
                "human_auditability",
            ]
        }
        checks["first_person_naturalness"] = {
            "status": "FAIL",
            "reason": "question names Jake and Alice instead of asking from first person",
            "fix": "rewrite as a natural everyday first-person question using I or we",
        }
        failed = judge_gate({"review_passed": True, "checks": checks})
        self.assertFalse(failed["passed"])
        self.assertIn("first_person_naturalness", failed["failed_checks"])

    def test_build_review_from_gates_for_accepted_row(self) -> None:
        review = build_review_from_gates(
            judge={"review_passed": True, "gate": {"passed": True}},
            answerability={"gate": {"passed": True}, "evaluations": []},
            schema_errors=[],
            accepted=True,
            final_reason="passed all gates",
        )
        self.assertEqual(review["status"], "passed")
        self.assertTrue(review["review_passed"])
        self.assertTrue(review["judger"]["gate"]["passed"])
        self.assertTrue(review["answerability"]["gate"]["passed"])

    def test_build_review_from_gates_for_judger_rejection(self) -> None:
        review = build_review_from_gates(
            judge={
                "review_passed": False,
                "feedback_to_generator": "ask from the speaker's own memory gap",
                "gate": {"passed": False, "reason": "not first-person"},
            },
            answerability=None,
            schema_errors=[],
            accepted=False,
            rejection_stage="judger",
            final_reason="not first-person",
        )
        self.assertEqual(review["status"], "rejected_by_judger")
        self.assertFalse(review["review_passed"])
        self.assertEqual(review["final_decision"]["rejection_stage"], "judger")

    def test_build_review_from_gates_for_answerability_rejection(self) -> None:
        review = build_review_from_gates(
            judge={"review_passed": True, "gate": {"passed": True}},
            answerability={"gate": {"passed": False, "reason": "single user answered correctly"}},
            schema_errors=[],
            accepted=False,
            rejection_stage="answerability",
            final_reason="single user answered correctly",
        )
        self.assertEqual(review["status"], "rejected_by_answerability")
        self.assertFalse(review["answerability"]["gate"]["passed"])

    def test_build_review_from_gates_for_schema_rejection(self) -> None:
        review = build_review_from_gates(
            judge={"review_passed": True, "gate": {"passed": True}},
            answerability={"gate": {"passed": True}, "evaluations": []},
            schema_errors=["answer must equal options[correct]"],
            accepted=False,
            rejection_stage="schema",
            final_reason="strict validation failed",
        )
        self.assertEqual(review["status"], "rejected_by_schema")
        self.assertFalse(review["schema_validation"]["passed"])
        self.assertEqual(review["schema_validation"]["errors"], ["answer must equal options[correct]"])

    def test_dry_run_qa_includes_video_evidence_provenance(self) -> None:
        qa = dry_run_qa(
            {
                "evidence_id": "E1",
                "required_users": ["Jake", "Alice"],
                "source_urls": {"videos": ["video_a", "video_b"]},
                "clips": [
                    {
                        "agent_name": "Jake",
                        "agent_dir": "A1_JAKE",
                        "agent_id": "A1",
                        "day": "DAY1",
                        "time_token": "11100000",
                        "clip_clock": "11:10:00.00",
                        "duration_seconds": 30.0,
                        "video_url": "video_a",
                        "local_video": "jake.mp4",
                        "frames": [{"timestamp_seconds": 10.0, "path": "jake_10.jpg"}],
                    }
                ],
            },
            "commonality",
        )
        self.assertEqual(qa["video_evidence"][0]["user"], "Jake")
        self.assertEqual(qa["video_evidence"][0]["video_url"], "video_a")
        self.assertEqual(qa["video_evidence"][0]["sampled_frames"][0]["timestamp_seconds"], 10.0)
        self.assertEqual(qa["referred_timestamps"], [])
        self.assertIn("human_audit", qa)
        self.assertIn("generation_trace", qa)
        self.assertEqual(qa["generation_trace"][0]["stage"], "dry_run")

    def test_materialize_review_videos_copies_existing_local_video(self) -> None:
        with workspace_temp_dir() as tmp:
            root = Path(tmp)
            source_video = root / "source.mp4"
            source_video.write_bytes(b"fake video")
            evidence_path = root / "evidence.jsonl"
            evidence_path.write_text(
                json.dumps(
                    {
                        "evidence_id": "E1",
                        "day": "DAY1",
                        "time_token": "11100000",
                        "required_users": ["Jake"],
                        "clips": [
                            {
                                "agent_name": "Jake",
                                "agent_dir": "A1_JAKE",
                                "agent_id": "A1",
                                "day": "DAY1",
                                "time_token": "11100000",
                                "clip_clock": "11:10:00.00",
                                "video_url": "https://example.invalid/source.mp4",
                                "local_video": str(source_video),
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = materialize_review_videos(
                evidence_path=evidence_path,
                output_dir=root / "review",
                download_missing=False,
            )

        video = manifest["rows"][0]["videos"][0]
        self.assertEqual(manifest["video_count_ok"], 1)
        self.assertEqual(video["status"], "ok")
        self.assertTrue(Path(video["review_video"]).name.endswith("source.mp4"))

    def test_qa_for_judger_prompt_excludes_trace_fields(self) -> None:
        item = SchemaTests("test_validate_valid_item").valid_item()
        compact = qa_for_judger_prompt(item)
        self.assertIn("question", compact)
        self.assertIn("evidence", compact)
        self.assertNotIn("generation_trace", compact)
        self.assertNotIn("human_audit", compact)
        self.assertNotIn("video_evidence", compact)
        self.assertNotIn("source_urls", compact)


if __name__ == "__main__":
    unittest.main()
