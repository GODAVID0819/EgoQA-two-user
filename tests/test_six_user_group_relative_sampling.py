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
    SIX_USER_TIME_AWARE_TEMPORAL_POLICY,
    analyze_group_relative_similarity,
    apply_six_user_generator_frame_budget,
    build_candidate_packet,
    build_six_user_role_structures,
    clustered_speaker_consensus_pruning,
    clustered_frame_representatives_by_time_window,
    materialize_six_user_consensus_candidate,
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


class SpeakerConsensusPruningTests(unittest.TestCase):
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

    def test_long_video_clustering_keeps_twelve_clusters_per_thirty_seconds(self) -> None:
        frames = [
            {"timestamp_seconds": float(index), "path": f"frame-{index}.jpg"}
            for index in range(60)
        ]
        embeddings = [[float(index), 1.0] for index in range(60)]

        def local_clusters(
            local_frames: list[dict[str, object]],
            local_embeddings: list[list[float]],
            **kwargs: object,
        ) -> dict[str, object]:
            self.assertEqual(len(local_frames), 30)
            self.assertEqual(len(local_embeddings), 30)
            self.assertEqual(kwargs["cluster_count"], 12)
            self.assertTrue(kwargs["split_noncontiguous_clusters"])
            self.assertEqual(kwargs["max_member_gap_seconds"], 1.5)
            representatives = []
            labels = [-1] * 30
            representative_embeddings = []
            for cluster_index in range(12):
                members = list(range(cluster_index, 30, 12))
                for member in members:
                    labels[member] = cluster_index
                representatives.append(
                    {
                        "cluster_index": cluster_index,
                        "visual_cluster_index": cluster_index,
                        "temporal_component_index": 0,
                        "frame_index": members[0],
                        "timestamp_seconds": local_frames[members[0]][
                            "timestamp_seconds"
                        ],
                        "path": local_frames[members[0]]["path"],
                        "member_indices": members,
                        "member_timestamps": [
                            local_frames[index]["timestamp_seconds"]
                            for index in members
                        ],
                        "member_count": len(members),
                    }
                )
                representative_embeddings.append(local_embeddings[members[0]])
            return {
                "cluster_count_requested": 12,
                "cluster_count": 12,
                "visual_cluster_count": 12,
                "split_noncontiguous_clusters": True,
                "max_member_gap_seconds": 1.5,
                "labels": labels,
                "representatives": representatives,
                "representative_embeddings": representative_embeddings,
            }

        with mock.patch.object(
            group_relative_clip_sampling,
            "clustered_frame_representatives",
            side_effect=local_clusters,
        ) as cluster_mock:
            result = clustered_frame_representatives_by_time_window(
                frames,
                embeddings,
                cluster_count_per_window=12,
                cluster_window_seconds=30.0,
                start_seconds=0.0,
                duration_seconds=60.0,
                split_noncontiguous_clusters=True,
                max_member_gap_seconds=1.5,
            )

        self.assertEqual(cluster_mock.call_count, 2)
        self.assertEqual(result["window_count"], 2)
        self.assertEqual(result["cluster_count_per_window"], 12)
        self.assertEqual(result["cluster_count"], 24)
        self.assertEqual(
            [window["output_cluster_count"] for window in result["windows"]],
            [12, 12],
        )
        self.assertEqual(
            result["representatives"][12]["member_indices"][0], 30
        )
        self.assertTrue(all(label >= 0 for label in result["labels"]))

    def run_consensus(self, provider_best_similarities: list[float]) -> dict[str, object]:
        speaker = self.cluster_result([[1.0, 0.0]], [[0, 1]])
        providers = []
        for similarity in provider_best_similarities:
            orthogonal = max(0.0, 1.0 - similarity**2) ** 0.5
            providers.append(
                self.cluster_result(
                    [
                        [similarity, orthogonal],
                        [0.81, max(0.0, 1.0 - 0.81**2) ** 0.5],
                    ],
                    [[0], [1]],
                )
            )
        with mock.patch.object(
            group_relative_clip_sampling,
            "clustered_frame_representatives",
            side_effect=[speaker, *providers],
        ):
            return clustered_speaker_consensus_pruning(
                [self.frames() for _ in range(6)],
                [[[1.0, 0.0], [1.0, 0.0]] for _ in range(6)],
                speaker_index=0,
                start_seconds=0.0,
                duration_seconds=10.0,
                sample_interval_seconds=1.0,
                min_pruned_video_seconds=2.0,
            )

    def test_five_of_five_deletes_speaker_and_all_provider_argmax_clusters(self) -> None:
        result = self.run_consensus([0.95, 0.94, 0.93, 0.92, 0.91])

        self.assertTrue(result["passed"])
        self.assertEqual(len(result["events"]), 1)
        event = result["events"][0]
        self.assertEqual(event["high_provider_count"], 5)
        self.assertEqual(
            event["deleted_clusters"],
            [
                {"video_index": 0, "cluster_index": 0},
                {"video_index": 1, "cluster_index": 0},
                {"video_index": 2, "cluster_index": 0},
                {"video_index": 3, "cluster_index": 0},
                {"video_index": 4, "cluster_index": 0},
                {"video_index": 5, "cluster_index": 0},
            ],
        )
        self.assertEqual(result["videos"][0]["marked_frame_indices"], [0, 1])

    def test_four_of_five_does_not_delete_below_threshold_provider(self) -> None:
        result = self.run_consensus([0.95, 0.94, 0.93, 0.82, 0.70])

        event = result["events"][0]
        self.assertEqual(event["high_provider_count"], 4)
        self.assertEqual(
            [row["video_index"] for row in event["deleted_clusters"]],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(result["videos"][5]["marked_cluster_indices"], [])

    def test_three_of_five_deletes_only_speaker_and_high_provider_clusters(self) -> None:
        result = self.run_consensus([0.95, 0.94, 0.93, 0.70, 0.69])

        self.assertTrue(result["passed"])
        event = result["events"][0]
        self.assertEqual(event["high_provider_count"], 3)
        self.assertEqual(
            [row["video_index"] for row in event["deleted_clusters"]],
            [0, 1, 2, 3],
        )
        self.assertEqual(result["videos"][4]["marked_cluster_indices"], [])
        self.assertEqual(result["videos"][5]["marked_cluster_indices"], [])

    def test_two_of_five_does_not_trigger_consensus_deletion(self) -> None:
        result = self.run_consensus([0.95, 0.94, 0.70, 0.69, 0.68])

        self.assertFalse(result["passed"])
        self.assertEqual(result["events"], [])
        self.assertTrue(all(not video["marked_cluster_indices"] for video in result["videos"]))

    def test_argmax_only_and_duplicate_cluster_deletion_is_deduplicated(self) -> None:
        speaker = self.cluster_result([[1.0, 0.0], [0.99, 0.01]], [[0], [1]])
        provider = self.cluster_result([[0.95, 0.0], [0.90, 0.0]], [[0], [1]])
        with mock.patch.object(
            group_relative_clip_sampling,
            "clustered_frame_representatives",
            side_effect=[speaker, provider, provider, provider, provider, provider],
        ):
            result = clustered_speaker_consensus_pruning(
                [self.frames() for _ in range(6)],
                [[[1.0, 0.0], [1.0, 0.0]] for _ in range(6)],
                speaker_index=0,
                start_seconds=0.0,
                duration_seconds=10.0,
                sample_interval_seconds=1.0,
                min_pruned_video_seconds=2.0,
            )

        self.assertEqual(len(result["events"]), 2)
        self.assertEqual(result["videos"][1]["marked_cluster_indices"], [0])
        self.assertEqual(result["videos"][1]["trigger_event_indices"], [0, 1])

    def test_provider_only_all_pairs_deletes_non_argmax_at_threshold_and_keeps_speaker(self) -> None:
        speaker = self.cluster_result([[1.0, 0.0], [0.0, 1.0]], [[0], [1]])
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

    def test_provider_only_all_pairs_covers_every_cluster_pair_for_all_five_providers(self) -> None:
        cluster_counts = [3, 2, 1, 3, 2, 4]
        cluster_results = [
            self.cluster_result(
                [[1.0, 0.0] for _ in range(count)],
                [[index] for index in range(count)],
            )
            for count in cluster_counts
        ]
        matrices = [
            [[0.90, 0.10], [0.20, 0.83], [0.10, 0.40]],
            [[0.10], [0.20], [0.30]],
            [[0.10, 0.84, 0.20], [0.30, 0.40, 0.95], [0.10, 0.20, 0.30]],
            [[0.82, 0.10], [0.20, 0.30], [0.40, 0.50]],
            [
                [0.90, 0.10, 0.20, 0.30],
                [0.10, 0.20, 0.88, 0.40],
                [0.30, 0.40, 0.50, 0.60],
            ],
        ]
        with (
            mock.patch.object(
                group_relative_clip_sampling,
                "clustered_frame_representatives",
                side_effect=cluster_results,
            ),
            mock.patch.object(
                group_relative_clip_sampling,
                "frame_similarity_matrix",
                side_effect=matrices,
            ) as matrix_mock,
        ):
            result = group_relative_clip_sampling.clustered_speaker_provider_all_pairs_pruning(
                [self.frames(4) for _ in range(6)],
                [[[1.0, 0.0]] * 4 for _ in range(6)],
                speaker_index=0,
                start_seconds=0.0,
                duration_seconds=10.0,
                sample_interval_seconds=1.0,
                min_pruned_video_seconds=2.0,
            )

        self.assertEqual(matrix_mock.call_count, 5)
        self.assertEqual(result["comparison_scope"], "every_speaker_cluster_x_every_provider_cluster")
        self.assertEqual(result["pruned_side"], "providers_only")
        self.assertTrue(result["speaker_preserved"])
        self.assertEqual(result["speaker_cluster_count"], 3)
        self.assertEqual(result["provider_cluster_counts"], {1: 2, 2: 1, 3: 3, 4: 2, 5: 4})
        self.assertEqual(result["expected_pairwise_comparison_count"], 36)
        self.assertEqual(result["pairwise_comparison_count"], 36)
        self.assertEqual(result["above_threshold_pair_count"], 7)
        self.assertEqual(result["videos"][0]["marked_cluster_indices"], [])
        self.assertEqual(result["videos"][0]["remove_intervals"], [])
        self.assertEqual(
            [video["marked_cluster_indices"] for video in result["videos"][1:]],
            [[0, 1], [], [1, 2], [0], [0, 2]],
        )

    def test_time_aware_provider_pruning_compares_all_clusters_within_plus_minus_30s(self) -> None:
        speaker = self.cluster_result(
            [[1.0, 0.0], [0.0, 1.0]],
            [[0], [100]],
        )
        provider = self.cluster_result(
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            [[0], [40], [100]],
        )
        unrelated = self.cluster_result([[-1.0, 0.0]], [[0]])
        matrices = [
            [[0.90, 0.95, 0.10], [0.95, 0.10, 0.90]],
            *([[[0.10], [0.10]]] * 4),
        ]
        with (
            mock.patch.object(
                group_relative_clip_sampling,
                "clustered_frame_representatives",
                side_effect=[speaker, provider, unrelated, unrelated, unrelated, unrelated],
            ) as cluster_mock,
            mock.patch.object(
                group_relative_clip_sampling,
                "frame_similarity_matrix",
                side_effect=matrices,
            ),
        ):
            result = group_relative_clip_sampling.clustered_speaker_provider_all_pairs_pruning(
                [self.frames(101) for _ in range(6)],
                [[[1.0, 0.0]] * 101 for _ in range(6)],
                speaker_index=0,
                start_seconds=0.0,
                duration_seconds=110.0,
                sample_interval_seconds=1.0,
                min_pruned_video_seconds=2.0,
                max_pair_time_difference_seconds=30.0,
                mutual_nearest_only=False,
                split_noncontiguous_clusters=True,
                max_cluster_member_gap_seconds=1.5,
            )

        self.assertTrue(result["passed"])
        self.assertEqual(result["temporal_policy"], SIX_USER_TIME_AWARE_TEMPORAL_POLICY)
        self.assertEqual(
            result["comparison_scope"],
            "every_asker_cluster_x_every_provider_cluster_within_plus_minus_time_window",
        )
        self.assertEqual(result["cluster_time_window_seconds"], 30.0)
        self.assertFalse(result["mutual_nearest_only"])
        self.assertTrue(result["asker_preserved"])
        self.assertEqual(result["eligible_pairwise_comparison_count"], 6)
        self.assertEqual(result["videos"][0]["marked_cluster_indices"], [])
        self.assertEqual(result["videos"][0]["marked_frame_indices"], [])
        self.assertEqual(result["videos"][1]["marked_cluster_indices"], [0, 2])
        self.assertNotIn(1, result["videos"][1]["marked_cluster_indices"])
        self.assertTrue(
            all(
                match["timestamp_difference_seconds"] <= 30.0
                for event in result["events"]
                for match in event["speaker_matches"]
            )
        )
        self.assertTrue(
            all(
                call.kwargs["split_noncontiguous_clusters"] is True
                and call.kwargs["max_member_gap_seconds"] == 1.5
                for call in cluster_mock.call_args_list
            )
        )

    def test_frame_budget_keeps_every_surviving_sample_and_rejects_posthoc_thinning(self) -> None:
        counts = [600, 480, 480, 480, 480, 480]
        clips = [
            {
                "agent_name": f"user_{index}",
                "duration_seconds": 600.0,
                "frames": [
                    {
                        "path": f"user_{index}_{frame_index}.jpg",
                        "sampled_frame_index": frame_index,
                    }
                    for frame_index in range(count)
                ],
                "temporal_pruning": {"analysis_sample_fps": 1.0},
            }
            for index, count in enumerate(counts)
        ]

        budgeted, summary = apply_six_user_generator_frame_budget(
            clips, aggregate_frame_budget=3_600
        )

        self.assertEqual([len(clip["frames"]) for clip in budgeted], counts)
        self.assertEqual(summary["source_frame_count"], 3_000)
        self.assertEqual(summary["model_input_frame_count"], 3_000)
        self.assertEqual(summary["policy"], "complete_surviving_sampled_frames")
        self.assertEqual(
            [frame["sampled_frame_index"] for frame in budgeted[0]["frames"]],
            list(range(600)),
        )

        oversized = [
            {**clip, "frames": [*clip["frames"], {"path": "extra.jpg"}]}
            for clip in [
                {
                    "duration_seconds": 600.0,
                    "frames": [{"path": f"frame_{index}.jpg"} for index in range(600)],
                    "temporal_pruning": {"analysis_sample_fps": 1.0},
                }
                for _ in range(6)
            ]
        ]
        with self.assertRaisesRegex(ValueError, "do not lower the 1 FPS pruning rate"):
            apply_six_user_generator_frame_budget(
                oversized, aggregate_frame_budget=3_600
            )


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
            frames = []
            for frame_index in range(4):
                frame_path = self.tmp_path / f"user_{index}_frame_{frame_index}.jpg"
                frame_path.write_bytes(b"frame")
                frames.append(
                    {
                        "timestamp_seconds": float(frame_index),
                        "path": str(frame_path),
                    }
                )
            rows.append(
                {
                    "user": f"user_{index}",
                    "clip": {
                        "clip_id": f"clip_{index}",
                        "agent_dir": f"agent_{index}",
                        "agent_name": f"user_{index}",
                        "local_video": str(source),
                    },
                    "frames": frames,
                }
            )
        return rows

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

    def test_consensus_candidate_routes_all_speaker_and_retained_provider_frames(self) -> None:
        consensus = {
            "passed": True,
            "method": "speaker_provider_all_pairs_provider_only",
            "speaker_index": 2,
            "events": [{"event_index": 0}],
            "videos": [
                {
                    "video_index": index,
                    "passed": True,
                    "keep_intervals": [[0.0, 1.0], [3.0, 10.0]],
                    "remove_intervals": [[1.0, 3.0]],
                    "marked_cluster_indices": [] if index == 2 else [0],
                    "trigger_event_indices": [0],
                    "clusters": [
                        {
                            "cluster_index": 0,
                            "frame_index": 0,
                            "member_indices": [0, 1],
                        },
                        {
                            "cluster_index": 1,
                            "frame_index": 2,
                            "member_indices": [2, 3],
                        },
                    ],
                    "kept_duration_seconds": 8.0,
                    "removed_duration_seconds": 2.0,
                }
                for index in range(6)
            ],
        }
        materialize_calls = []

        def fake_materialize(source_video, output_video, keep_intervals, **kwargs):
            materialize_calls.append(
                {
                    "source": str(source_video),
                    "keep_intervals": list(keep_intervals),
                }
            )
            output = Path(output_video)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"pruned-video")
            return output

        rows = self.six_rows()
        with mock.patch.object(
            group_relative_clip_sampling,
            "materialize_pruned_video",
            fake_materialize,
        ):
            clips = materialize_six_user_consensus_candidate(
                rows,
                consensus,
                output_dir=self.tmp_path / "consensus-assets",
                ffmpeg_binary="ffmpeg",
            )

        self.assertEqual(
            [clip["agent_name"] for clip in clips],
            ["user_2", "user_0", "user_1", "user_3", "user_4", "user_5"],
        )
        self.assertEqual(
            [clip["media_role"] for clip in clips],
            ["speaker_all_clustering_frames", *(["provider_retained_cluster_frames"] * 5)],
        )
        self.assertEqual([clip["is_pruned"] for clip in clips], [False, *([True] * 5)])
        self.assertEqual(
            [clip["generator_media_mode"] for clip in clips],
            ["all_clustering_frames_only", *(["retained_cluster_frames_only"] * 5)],
        )
        self.assertTrue(all(clip["force_frame_inputs"] is True for clip in clips))
        self.assertTrue(all("local_video" not in clip for clip in clips))
        self.assertTrue(all("generator_local_video" not in clip for clip in clips))
        self.assertTrue(all(Path(clip["full_local_video"]).is_file() for clip in clips))
        source_by_user = {
            row["clip"]["agent_name"]: Path(row["clip"]["local_video"])
            for row in rows
        }
        self.assertTrue(
            all(
                Path(clip["full_local_video"]).samefile(
                    source_by_user[clip["agent_name"]]
                )
                for clip in clips
            )
        )
        self.assertTrue(
            all(clip["full_video_materialization"] == "hardlink" for clip in clips)
        )
        self.assertEqual([len(clip["frames"]) for clip in clips], [4, 2, 2, 2, 2, 2])
        self.assertEqual(
            [frame["sampled_frame_index"] for frame in clips[0]["frames"]],
            [0, 1, 2, 3],
        )
        self.assertTrue(
            all(
                [frame["sampled_frame_index"] for frame in clip["frames"]] == [2, 3]
                for clip in clips[1:]
            )
        )
        self.assertEqual(materialize_calls, [])

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
        reused_cluster_object_ids = []
        successful_speakers = {1, 2, 4, 5}

        def fake_provider_pruning(_frames, _embeddings, *, speaker_index, **kwargs):
            consensus_attempts.append(speaker_index)
            reused_cluster_object_ids.append(
                id(kwargs["precomputed_clusters_by_video"])
            )
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
                        "speaker_all_clustering_frames"
                        if position == 0
                        else "provider_retained_cluster_frames"
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
                "cluster_six_user_frame_representatives",
                wraps=group_relative_clip_sampling.cluster_six_user_frame_representatives,
            ) as cluster_mock,
            mock.patch.object(
                group_relative_clip_sampling,
                "clustered_speaker_provider_all_pairs_pruning",
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
        self.assertEqual(cluster_mock.call_count, 1)
        self.assertEqual(len(set(reused_cluster_object_ids)), 1)
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
            "speaker_all_clustering_frames_five_provider_retained_cluster_frames",
        )
        self.assertIn("speaker-only", packet["requirement"])
        self.assertIn("all-six condition", packet["requirement"])
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

    def test_six_user_mining_stops_at_the_exact_packet_target(self) -> None:
        output_dir = self.tmp_path / "mining-output"
        output_path = self.tmp_path / "candidates.jsonl"
        result = {
            "day": "DAY1",
            "time_token": "120000",
            "speaker_candidates": [
                {"candidate_index": index}
                for index in range(6)
            ],
            "speaker_attempts": [],
        }

        with (
            mock.patch.object(group_relative_clip_sampling, "read_json", return_value={}),
            mock.patch.object(
                group_relative_clip_sampling,
                "group_manifest_clips",
                return_value=[{"day": "DAY1", "time_token": "120000", "clips": [{}] * 6}],
            ),
            mock.patch.object(
                group_relative_clip_sampling,
                "analyze_group_relative_similarity",
                return_value=result,
            ) as analyze_mock,
            mock.patch.object(
                group_relative_clip_sampling,
                "write_review_bundle",
                return_value=self.tmp_path / "review",
            ),
            mock.patch.object(
                group_relative_clip_sampling,
                "build_candidate_packet",
                side_effect=lambda candidate: {
                    "evidence_id": f"candidate-{candidate['candidate_index']}",
                    "group_relative_clip_similarity": {},
                },
            ),
        ):
            candidates = mine_group_relative_clip_candidates(
                manifest_path=self.tmp_path / "manifest.json",
                output_path=output_path,
                output_dir=output_dir,
                cache_dir=self.tmp_path / "cache",
                target_count=4,
                selected_count=6,
                encoder=types.SimpleNamespace(model_id="fake/clip"),
            )

        self.assertEqual(len(candidates), 4)
        self.assertEqual(
            [candidate["evidence_id"] for candidate in candidates],
            ["candidate-0", "candidate-1", "candidate-2", "candidate-3"],
        )
        self.assertEqual(len(output_path.read_text(encoding="utf-8").splitlines()), 4)
        for legacy_argument in (
            "pairs_per_group",
            "topk",
            "min_topk_sim",
            "min_mean_sim",
            "max_mean_sim",
            "temporal_neighborhood_seconds",
            "preserve_shared_anchor_seconds",
            "pruning_protection_mode",
            "min_pruned_video_percent",
            "max_pair_time_difference_seconds",
            "random_pair_first",
        ):
            self.assertNotIn(legacy_argument, analyze_mock.call_args.kwargs)
        summary = group_relative_clip_sampling.read_json(
            output_dir / "group_relative_clip_summary.json"
        )
        self.assertEqual(summary["candidate_count"], 4)
        self.assertEqual(summary["settings"]["target_count"], 4)
        self.assertEqual(
            summary["settings"]["selection_mode"],
            "six_user_speaker_provider_all_pairs",
        )
        self.assertFalse(summary["settings"]["legacy_pair_filter_active"])
        self.assertEqual(
            summary["settings"]["cluster_comparison_scope"],
            "every_speaker_cluster_x_every_provider_cluster",
        )
        self.assertEqual(summary["settings"]["pruned_side"], "providers_only")
        self.assertTrue(summary["settings"]["speaker_preserved"])
        for legacy_setting in (
            "pairs_per_group",
            "topk",
            "min_topk_sim",
            "min_mean_sim",
            "max_mean_sim",
            "temporal_neighborhood_seconds",
            "preserve_shared_anchor_seconds",
            "pruning_protection_mode",
            "min_pruned_video_percent",
            "max_pair_time_difference_seconds",
            "random_pair_first",
        ):
            self.assertNotIn(legacy_setting, summary["settings"])

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
