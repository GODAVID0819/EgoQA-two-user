from __future__ import annotations

import random
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

from egolife_two_user_qa.group_relative_clip_sampling import (  # noqa: E402
    analyze_group_relative_similarity,
    build_candidate_packet,
    build_six_user_role_structures,
    materialize_six_user_role_structure,
    mine_group_relative_clip_candidates,
)
from egolife_two_user_qa import group_relative_clip_sampling  # noqa: E402


def pair_scores(*, kept_keys: set[str]) -> list[dict[str, object]]:
    rows = []
    for left_index in range(6):
        for right_index in range(left_index + 1, 6):
            pair_key = f"{left_index}-{right_index}"
            kept = pair_key in kept_keys
            rows.append(
                {
                    "pair_key": pair_key,
                    "left_index": left_index,
                    "right_index": right_index,
                    "status": "kept" if kept else "rejected",
                    "rejection_reasons": [] if kept else ["synthetic_rejection"],
                }
            )
    return rows


class SpeakerProviderAllPairsPruningTests(unittest.TestCase):
    @staticmethod
    def cluster_result(
        representative_embeddings: list[list[float]],
        member_indices: list[list[int]],
    ) -> dict[str, object]:
        representatives = []
        labels = []
        for cluster_index, members in enumerate(member_indices):
            labels.extend([cluster_index] * len(members))
            representatives.append(
                {
                    "cluster_index": cluster_index,
                    "frame_index": members[0],
                    "timestamp_seconds": float(members[0] + 1),
                    "path": f"cluster-{cluster_index}.jpg",
                    "member_indices": members,
                    "member_timestamps": [float(index + 1) for index in members],
                    "member_count": len(members),
                }
            )
        return {
            "cluster_count_requested": 12,
            "cluster_count": len(representatives),
            "visual_cluster_count": len(representatives),
            "labels": labels,
            "representatives": representatives,
            "representative_embeddings": representative_embeddings,
        }

    @staticmethod
    def frames(count: int = 2) -> list[dict[str, object]]:
        return [
            {"timestamp_seconds": float(index + 1), "path": f"frame-{index}.jpg"}
            for index in range(count)
        ]

        provider = self.cluster_result(
            [[1.0, 0.0], [0.82, 0.5723635209], [0.0, 1.0]],
            [[0], [1], [2]],
        )
        unrelated = self.cluster_result([[-1.0, 0.0]], [[0]])
        with mock.patch.object(
            group_relative_clip_sampling,
            "clustered_frame_representatives",
            side_effect=[speaker, provider, unrelated, unrelated, unrelated, unrelated],
        ):
            result = group_relative_clip_sampling.clustered_speaker_provider_all_pairs_pruning(
                [self.frames(3) for _ in range(6)],
                [[[1.0, 0.0]] * 3 for _ in range(6)],
                speaker_index=0,
                start_seconds=0.0,
                duration_seconds=10.0,
                sample_interval_seconds=1.0,
                min_pruned_video_seconds=2.0,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["pairwise_comparison_count"], 14)
        self.assertEqual(result["videos"][0]["marked_cluster_indices"], [])
        self.assertEqual(result["videos"][1]["marked_cluster_indices"], [0, 1, 2])
        self.assertEqual(result["events"][1]["max_similarity"], 0.82)

    def test_provider_only_all_pairs_deduplicates_provider_cluster_matches(self) -> None:
        speaker = self.cluster_result(
            [[1.0, 0.0], [0.99, 0.1410673598]],
            [[0], [1]],
        )
        provider = self.cluster_result([[1.0, 0.0]], [[0]])
        unrelated = self.cluster_result([[-1.0, 0.0]], [[0]])
        with mock.patch.object(
            group_relative_clip_sampling,
            "clustered_frame_representatives",
            side_effect=[speaker, provider, unrelated, unrelated, unrelated, unrelated],
        ):
            result = group_relative_clip_sampling.clustered_speaker_provider_all_pairs_pruning(
                [self.frames() for _ in range(6)],
                [[[1.0, 0.0]] * 2 for _ in range(6)],
                speaker_index=0,
                start_seconds=0.0,
                duration_seconds=10.0,
                sample_interval_seconds=1.0,
                min_pruned_video_seconds=2.0,
            )

        self.assertEqual(result["videos"][1]["marked_cluster_indices"], [0])
        self.assertEqual(len(result["events"]), 1)
        self.assertEqual(
            [match["speaker_cluster_index"] for match in result["events"][0]["speaker_matches"]],
            [0, 1],
        )

    def test_zip_temporal_pruning_builds_five_pairs_and_keeps_full_speaker(self) -> None:
        frames = [self.frames(3) for _ in range(6)]
        embeddings = [[[1.0, 0.0]] * 3 for _ in range(6)]
        clusters = self.cluster_result([[1.0, 0.0]], [[0, 1, 2]])
        clusters["clustering"] = {
            "time_weight": 0.1,
            "temporal_unit_seconds": 30.0,
        }

        def pair_result(*_args, **_kwargs):
            return {
                "passed": True,
                "high_similarity_representative_pairs": [
                    {
                        "left_cluster_index": 0,
                        "right_cluster_index": 0,
                        "similarity": 0.9,
                    }
                ],
                "high_similarity_representative_pair_count": 1,
                "left_marked_frame_indices": [0],
                "right_marked_frame_indices": [1],
                "left_remove_intervals": [[0.5, 1.5]],
                "right_remove_intervals": [[1.5, 2.5]],
                "left_keep_intervals": [[0.0, 0.5], [1.5, 30.0]],
                "right_keep_intervals": [[0.0, 1.5], [2.5, 30.0]],
                "left_kept_duration_seconds": 29.0,
                "right_kept_duration_seconds": 29.0,
                "left_removed_duration_seconds": 1.0,
                "right_removed_duration_seconds": 1.0,
            }

        with (
            mock.patch(
                "egolife_two_user_qa.temporal_kmeans_grid_sidecar."
                "time_aware_clustered_frame_representatives",
                return_value=clusters,
            ) as cluster,
            mock.patch(
                "egolife_two_user_qa.temporal_kmeans_grid_sidecar."
                "prune_time_aware_cluster_pair",
                side_effect=pair_result,
            ) as prune,
        ):
            result = (
                group_relative_clip_sampling.clustered_six_user_zip_temporal_pruning(
                    frames,
                    embeddings,
                    speaker_index=0,
                    start_seconds=0.0,
                    duration_seconds=30.0,
                    sample_interval_seconds=1.0,
                )
            )

        self.assertEqual(cluster.call_count, 6)
        self.assertEqual(prune.call_count, 5)
        self.assertTrue(
            all(
                call.kwargs["min_pruned_video_percent"] == 20.0
                and call.kwargs["cross_gap_mode"] == "center"
                and call.kwargs["max_cross_gap_seconds"] == 10.0
                for call in prune.call_args_list
            )
        )
        self.assertEqual(len(result["pair_results"]), 5)
        self.assertEqual(result["cluster_count"], 12)
        self.assertEqual(result["videos"][0]["keep_intervals"], [[0.0, 30.0]])
        self.assertEqual(result["videos"][1]["remove_intervals"], [[1.5, 2.5]])
        self.assertTrue(result["passed"])

    def test_zip_temporal_pruning_runs_real_pair_kernel(self) -> None:
        result = group_relative_clip_sampling.clustered_six_user_zip_temporal_pruning(
            [self.frames(3) for _ in range(6)],
            [[[1.0, 0.0]] * 3 for _ in range(6)],
            speaker_index=0,
            start_seconds=0.0,
            duration_seconds=10.0,
            sample_interval_seconds=1.0,
        )

        self.assertEqual(result["cluster_count"], 4)
        self.assertEqual(len(result["pair_results"]), 5)
        self.assertTrue(result["passed"])
        self.assertTrue(
            result["pair_results"][0]["pruning"]["left_remove_intervals"]
        )
        self.assertEqual(result["videos"][0]["remove_intervals"], [])
        self.assertTrue(result["videos"][1]["remove_intervals"])


class SixUserRoleSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        self.tmp_path = tmp_root / f"six_user_sampling_{uuid.uuid4().hex}"
        self.tmp_path.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp_path, True)

    def six_rows(self) -> list[dict[str, object]]:
        rows = []
        for index in range(6):
            source = self.tmp_path / f"user_{index}.mp4"
            source.write_bytes(b"full-video")
            rows.append(
                {
                    "user": f"user_{index}",
                    "clip": {
                        "clip_id": f"clip_{index}",
                        "agent_dir": f"agent_{index}",
                        "agent_name": f"user_{index}",
                        "local_video": str(source),
                    },
                    "frames": [],
                }
            )
        return rows

    def test_materializer_ignores_speaker_pair_pruning_and_uses_original_video(
        self,
    ) -> None:
        rows = self.six_rows()
        consensus = {
            "method": "zip_temporal_kmeans_center_gate_pair_pruning_v1",
            "speaker_index": 0,
            "pair_results": [
                {
                    "speaker_index": 0,
                    "provider_index": 1,
                    "pruning": {"left_remove_intervals": [[0.5, 1.5]]},
                }
            ],
            "videos": [
                {
                    "video_index": index,
                    "keep_intervals": (
                        [[0.0, 30.0]]
                        if index == 0
                        else [[0.0, 1.5], [2.5, 30.0]]
                    ),
                    "remove_intervals": [] if index == 0 else [[1.5, 2.5]],
                    "marked_cluster_indices": [],
                    "trigger_event_indices": [],
                    "kept_duration_seconds": 30.0 if index == 0 else 29.0,
                    "passed": True,
                }
                for index in range(6)
            ],
            "passed": True,
        }

        def fake_materialize(clip, *, media_role, keep_intervals, **_kwargs):
            return {
                **clip,
                "media_role": media_role,
                "is_pruned": keep_intervals is not None,
                "received_keep_intervals": keep_intervals,
            }

        with mock.patch.object(
            group_relative_clip_sampling,
            "_materialize_six_user_clip",
            side_effect=fake_materialize,
        ):
            clips = group_relative_clip_sampling.materialize_six_user_consensus_candidate(
                rows,
                consensus,
                output_dir=self.tmp_path / "zip_temporal",
                ffmpeg_binary="ffmpeg",
            )

        self.assertEqual(
            clips[0]["local_video"],
            rows[0]["clip"]["local_video"],
        )
        self.assertEqual(clips[0]["media_role"], "speaker_reference_unpruned")
        self.assertFalse(clips[0]["is_pruned"])
        self.assertIsNone(clips[0]["received_keep_intervals"])
        self.assertTrue(all(clip["is_pruned"] for clip in clips[1:]))

    @staticmethod
    def anchor_edge(anchor_index: int, *, speaker_remove: list[list[float]]) -> dict[str, object]:
        return {
            "pair_key": f"0-{anchor_index}",
            "left_index": 0,
            "right_index": anchor_index,
            "status": "kept",
            "temporal_pruning": {
                "method": "synthetic",
                "left_remove_intervals": speaker_remove,
                "left_keep_intervals": [[0.0, 10.0]],
                "left_kept_duration_seconds": 10.0,
                "right_remove_intervals": [[1.0, 3.0]],
                "right_keep_intervals": [[0.0, 1.0], [3.0, 10.0]],
                "right_kept_duration_seconds": 8.0,
            },
        }

    def test_exactly_two_speaker_edges_are_enough(self) -> None:
        scores = pair_scores(kept_keys={"0-1", "0-2"})

        result = build_six_user_role_structures(scores, rng=random.Random(7))

        self.assertEqual(len(result["diagnostic_pair_edges"]), 15)
        self.assertEqual(result["kept_degrees"], [2, 1, 1, 0, 0, 0])
        self.assertEqual(len(result["role_structures"]), 1)
        structure = result["role_structures"][0]
        self.assertEqual(structure["speaker_index"], 0)
        self.assertEqual(structure["anchor_indices"], [1, 2])
        self.assertEqual(structure["additional_indices"], [3, 4, 5])
        self.assertEqual(
            [edge["pair_key"] for edge in structure["selected_anchor_edges"]],
            ["0-1", "0-2"],
        )

    def test_one_kept_neighbor_produces_no_role_structure(self) -> None:
        result = build_six_user_role_structures(
            pair_scores(kept_keys={"0-1"}),
            rng=random.Random(2),
        )

        self.assertEqual(result["role_structures"], [])
        self.assertEqual(result["kept_degrees"], [1, 1, 0, 0, 0, 0])
        self.assertEqual(result["eligible_speaker_indices"], [])

    def test_provider_provider_rejections_do_not_block_valid_star(self) -> None:
        result = build_six_user_role_structures(
            pair_scores(kept_keys={"3-4", "3-5"}),
            rng=random.Random(5),
        )

        self.assertEqual(len(result["role_structures"]), 1)
        self.assertEqual(result["role_structures"][0]["speaker_index"], 3)
        self.assertEqual(result["role_structures"][0]["anchor_indices"], [4, 5])
        self.assertEqual(result["role_structures"][0]["additional_indices"], [0, 1, 2])

    def test_seeded_order_is_deterministic_with_multiple_structures(self) -> None:
        scores = pair_scores(kept_keys={"0-1", "0-2", "0-3", "1-2", "1-4"})

        first = build_six_user_role_structures(scores, rng=random.Random(19))
        second = build_six_user_role_structures(scores, rng=random.Random(19))

        self.assertGreater(len(first["role_structures"]), 1)
        self.assertEqual(first["role_structures"], second["role_structures"])
        for structure in first["role_structures"]:
            self.assertEqual(len(structure["anchor_indices"]), 2)
            self.assertEqual(len(structure["additional_indices"]), 3)
            self.assertEqual(len(structure["selected_anchor_edges"]), 2)

    def test_missing_pair_edge_is_rejected(self) -> None:
        scores = pair_scores(kept_keys={"0-1", "0-2"})[:-1]

        with self.assertRaisesRegex(ValueError, "15"):
            build_six_user_role_structures(scores, rng=random.Random(1))

    def test_materializes_three_pruned_and_three_full_videos_in_role_order(self) -> None:
        first_edge = self.anchor_edge(1, speaker_remove=[[2.0, 4.0]])
        second_edge = self.anchor_edge(2, speaker_remove=[[3.0, 5.0]])
        structure = {
            "candidate_rank": 1,
            "speaker_index": 0,
            "anchor_indices": [1, 2],
            "additional_indices": [3, 4, 5],
            "selected_anchor_edges": [first_edge, second_edge],
        }
        materialize_calls = []

        def fake_materialize(source_video, output_video, keep_intervals, **kwargs):
            materialize_calls.append(
                {
                    "source": str(source_video),
                    "output": str(output_video),
                    "keep_intervals": list(keep_intervals),
                }
            )
            output = Path(output_video)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"pruned-video")
            return output

        with mock.patch.object(
            group_relative_clip_sampling,
            "materialize_pruned_video",
            fake_materialize,
        ):
            clips = materialize_six_user_role_structure(
                self.six_rows(),
                structure,
                output_dir=self.tmp_path / "assets",
                start_seconds=0.0,
                duration_seconds=10.0,
                min_pruned_video_seconds=2.0,
                ffmpeg_binary="ffmpeg",
            )

        self.assertEqual([clip["agent_name"] for clip in clips], [f"user_{i}" for i in range(6)])
        self.assertEqual(
            [clip["media_role"] for clip in clips],
            [
                "speaker_pruned",
                "anchor_provider_pruned",
                "anchor_provider_pruned",
                "additional_provider_full",
                "additional_provider_full",
                "additional_provider_full",
            ],
        )
        self.assertEqual(len(materialize_calls), 3)
        self.assertEqual(
            materialize_calls[0]["keep_intervals"],
            [(0.0, 2.0), (5.0, 10.0)],
        )
        self.assertEqual(
            [row["keep_intervals"] for row in materialize_calls[1:]],
            [[[0.0, 1.0], [3.0, 10.0]], [[0.0, 1.0], [3.0, 10.0]]],
        )
        for clip in clips[:3]:
            self.assertTrue(clip["is_pruned"])
            self.assertNotEqual(clip["generator_local_video"], clip["full_local_video"])
        for clip in clips[3:]:
            self.assertFalse(clip["is_pruned"])
            self.assertEqual(clip["generator_local_video"], clip["full_local_video"])

    def test_rejects_role_structure_when_merged_speaker_retention_is_too_short(self) -> None:
        first_edge = self.anchor_edge(1, speaker_remove=[[0.0, 6.0]])
        second_edge = self.anchor_edge(2, speaker_remove=[[5.0, 9.5]])
        structure = {
            "candidate_rank": 1,
            "speaker_index": 0,
            "anchor_indices": [1, 2],
            "additional_indices": [3, 4, 5],
            "selected_anchor_edges": [first_edge, second_edge],
        }

        with self.assertRaisesRegex(ValueError, "speaker.*too short"):
            materialize_six_user_role_structure(
                self.six_rows(),
                structure,
                output_dir=self.tmp_path / "assets",
                start_seconds=0.0,
                duration_seconds=10.0,
                min_pruned_video_seconds=2.0,
                ffmpeg_binary="ffmpeg",
            )

    def test_six_user_analysis_visits_all_speakers_and_keeps_every_success(self) -> None:
        rows = self.six_rows()
        for index, row in enumerate(rows):
            row["frames"] = [{"path": f"frame-{index}.jpg"}]

        class Encoder:
            model_id = "fake/clip"

            def __init__(self) -> None:
                self.calls = []

            def encode(self, paths):
                self.calls.append(list(paths))
                return [[1.0, 0.0] for _ in paths]

        encoder = Encoder()
        consensus_attempts = []
        successful_speakers = {1, 2, 4, 5}

        def fake_provider_pruning(_frames, _embeddings, *, speaker_index, **kwargs):
            consensus_attempts.append(speaker_index)
            return {
                "method": "speaker_provider_all_pairs_provider_only",
                "speaker_index": speaker_index,
                "events": [{"event_index": 0}],
                "videos": [
                    {
                        "video_index": index,
                        "keep_intervals": [(0.0, 9.0)],
                        "remove_intervals": [(9.0, 10.0)],
                        "kept_duration_seconds": 9.0,
                        "removed_duration_seconds": 1.0,
                        "marked_cluster_indices": [0],
                    }
                    for index in range(6)
                ],
                "passed": speaker_index in successful_speakers,
            }

        def fake_materialize(_rows, consensus, **kwargs):
            speaker_index = consensus["speaker_index"]
            if not consensus["passed"]:
                raise ValueError(f"synthetic speaker {speaker_index} failure")
            ordered_indices = [
                speaker_index,
                *[index for index in range(6) if index != speaker_index],
            ]
            return [
                {
                    **dict(_rows[index]["clip"]),
                    "media_role": (
                        "speaker_reference_unpruned"
                        if position == 0
                        else "provider_similarity_pruned"
                    ),
                    "is_pruned": position != 0,
                }
                for position, index in enumerate(ordered_indices)
            ]

        group = {
            "day": "DAY1",
            "time_token": "120000",
            "clip_clock": "12:00:00",
            "clips": [dict(row["clip"]) for row in rows],
        }
        with (
            mock.patch.object(
                group_relative_clip_sampling,
                "group_clip_frames",
                return_value=rows,
            ),
            mock.patch.object(group_relative_clip_sampling, "score_video_pairs") as score_mock,
            mock.patch.object(
                group_relative_clip_sampling,
                "relative_group_scores",
            ) as relative_scores_mock,
            mock.patch.object(
                group_relative_clip_sampling,
                "clustered_six_user_zip_temporal_pruning",
                side_effect=fake_provider_pruning,
            ),
            mock.patch.object(
                group_relative_clip_sampling,
                "materialize_six_user_consensus_candidate",
                side_effect=fake_materialize,
            ),
        ):
            result = analyze_group_relative_similarity(
                group,
                output_dir=self.tmp_path / "analysis",
                cache_dir=self.tmp_path / "cache",
                encoder=encoder,
                selected_count=6,
                rng=random.Random(11),
            )

        self.assertEqual(len(encoder.calls), 6)
        score_mock.assert_not_called()
        relative_scores_mock.assert_not_called()
        self.assertEqual(consensus_attempts, [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(result["speaker_attempts"]), 6)
        self.assertEqual([row["status"] for row in result["speaker_attempts"]], [
            "failed", "succeeded", "succeeded", "failed", "succeeded", "succeeded"
        ])
        self.assertEqual(len(result["speaker_candidates"]), 4)
        self.assertEqual(
            [row["selection"]["speaker_index"] for row in result["speaker_candidates"]],
            [1, 2, 4, 5],
        )
        self.assertEqual(result["selection"]["selected_count"], 6)
        self.assertEqual(result["selection"]["method"], "six_user_speaker_consensus_all_speakers")

        packet = build_candidate_packet(result["speaker_candidates"][0])
        users = [clip["agent_name"] for clip in result["speaker_candidates"][0]["selected_clips"]]
        self.assertEqual(packet["candidate_type"], "six_user_speaker_consensus")
        self.assertEqual(packet["input_users"], users)
        self.assertEqual(packet["required_users"], users)
        self.assertEqual(packet["speaker_user"], users[0])
        self.assertEqual(packet["provider_users"], users[1:])
        self.assertEqual(packet["evidence_provider_user"], users[1])
        self.assertEqual(packet["evidence_provider_users"], users[1:])
        self.assertEqual(
            packet["generator_media_mode"],
            "speaker_full_five_provider_pruned_videos",
        )
        self.assertIn("speaker-only", packet["requirement"])
        self.assertIn("all-six condition", packet["requirement"])
        self.assertIn("w=0.1", packet["requirement"])
        self.assertIn("10.0-second center-gap gate", packet["requirement"])
        self.assertNotIn("anchor_provider_users", packet)
        self.assertNotIn("selected_anchor_edges", packet)
        self.assertEqual(set(packet["media_roles"]), set(users))

    def test_invalid_selected_count_fails_before_encoder_initialization(self) -> None:
        with (
            mock.patch.object(
                group_relative_clip_sampling,
                "TransformersClipEncoder",
            ) as encoder_class,
            mock.patch.object(group_relative_clip_sampling, "read_json") as read_json_mock,
        ):
            with self.assertRaisesRegex(ValueError, "2 or 6"):
                mine_group_relative_clip_candidates(
                    manifest_path=self.tmp_path / "manifest.json",
                    output_path=self.tmp_path / "output.jsonl",
                    output_dir=self.tmp_path / "output",
                    cache_dir=self.tmp_path / "cache",
                    selected_count=3,
                )

        encoder_class.assert_not_called()
        read_json_mock.assert_not_called()

    def test_two_user_candidate_packet_keeps_legacy_shape(self) -> None:
        selected_clips = [
            {"agent_name": "speaker", "agent_id": "A", "media_role": "legacy"},
            {"agent_name": "provider", "agent_id": "B", "media_role": "legacy"},
        ]
        group_result = {
            "day": "DAY1",
            "time_token": "120000",
            "clip_clock": "12:00:00",
            "selected_clips": selected_clips,
            "selection": {"selected_pair": {"pair_key": "0-1"}},
        }

        packet = build_candidate_packet(group_result)

        self.assertEqual(packet["candidate_type"], "random_synchronized_pair_cluster_pruned_video")
        self.assertEqual(packet["required_users"], ["speaker", "provider"])
        self.assertNotIn("input_users", packet)


if __name__ == "__main__":
    unittest.main()
