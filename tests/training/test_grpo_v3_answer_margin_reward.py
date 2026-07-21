from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from training.grpo_v3_answer_margin import (
    ANSWER_MARGIN_REWARD_REVISION,
    LABELS,
    PermutationKey,
)
from training.grpo_v3_answer_margin_reward import (
    EXPERIMENT_REVISION,
    resolve_ordered_videos,
    score_completion,
)
from training.grpo_v3_answer_scorer import (
    LabelScore,
    PromptAudit,
    ScoreResponse,
)


def packet(root: Path) -> dict:
    videos = []
    for user in ("u1", "u2"):
        path = root / f"{user}.mp4"
        path.write_bytes(b"video")
        videos.append({"agent_name": user, "local_video": str(path)})
    return {"evidence_id": "E1", "required_users": ["u1", "u2"], "clips": videos}


def key(candidate_index: int = 0) -> PermutationKey:
    return PermutationKey(
        experiment_condition_id="temperature_0.5",
        phase="train",
        evidence_id="E1",
        generation_seed_or_call_index=7,
        candidate_index=candidate_index,
        reward_revision=ANSWER_MARGIN_REWARD_REVISION,
    )


def completion(**extra: str) -> str:
    value = {
        "question": "Where is the mug?",
        "options": ["desk", "sink", "shelf", "bag", "table"],
        "correct": "C",
        "answer": "shelf",
        "rationale": "generator-only reason",
        **extra,
    }
    return json.dumps(value)


class RecordingScorer:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.calls = []
        self.scores = scores or {label: -float(index + 1) for index, label in enumerate(LABELS)}

    def score(self, request, *, audit_material=None):
        self.calls.append((request, audit_material))
        rendered = "safe rendered prompt"
        return ScoreResponse(
            scores={
                label: LabelScore(label, [100 + index], [self.scores[label]], self.scores[label])
                for index, label in enumerate(LABELS)
            },
            prompt_audit=PromptAudit(
                hashlib.sha256(rendered.encode()).hexdigest(),
                True,
                ["generator_field_marker_scan", "excluded_value_scan"],
                [],
            ),
            rendered_prompt=rendered,
        )


class AnswerMarginRewardCoreTests(unittest.TestCase):
    def test_unrecoverable_qa_returns_floor_without_scorer(self):
        scorer = RecordingScorer()
        result = score_completion("bad", {}, "E1", 0, scorer=scorer, key=key())
        self.assertEqual(result["reward"], -1.0)
        self.assertEqual(scorer.calls, [])
        record = result["record"]
        self.assertEqual(record["reward_source"], "core_qa_unrecoverable_floor")
        self.assertFalse(record["masked"])
        self.assertTrue(record["eligible_for_grpo"])
        self.assertEqual(record["raw_completion"], "bad")
        self.assertEqual(record["evidence_id"], "E1")
        self.assertEqual(record["candidate_index"], 0)
        self.assertEqual(record["permutation_key"]["phase"], "train")

    def test_scores_permuted_qa_and_writes_complete_audit_trace(self):
        scores = {"A": -4.0, "B": -3.0, "C": -2.0, "D": -1.0, "E": -5.0}
        scorer = RecordingScorer(scores)
        with tempfile.TemporaryDirectory() as tmp:
            result = score_completion(
                completion(review="generator review"),
                packet(Path(tmp)),
                "E1",
                0,
                scorer=scorer,
                key=key(),
                question_type="commonality",
                generation_mode="baseline",
            )
        request, audit_material = scorer.calls[0]
        self.assertEqual(len(request.videos), 2)
        self.assertEqual(tuple(Path(item).name for item in request.videos), ("u1.mp4", "u2.mp4"))
        self.assertEqual(set(request.options), {"desk", "sink", "shelf", "bag", "table"})
        self.assertEqual(
            audit_material.excluded_values,
            {
                "correct": "C",
                "answer": "shelf",
                "rationale": "generator-only reason",
                "review": "generator review",
            },
        )
        record = result["record"]
        self.assertEqual(record["schema_version"], 1)
        self.assertEqual(record["reward_revision"], ANSWER_MARGIN_REWARD_REVISION)
        self.assertEqual(record["experiment_revision"], EXPERIMENT_REVISION)
        self.assertEqual(record["raw_completion"], completion(review="generator review"))
        self.assertEqual(record["core_qa"]["question"], "Where is the mug?")
        self.assertIn("repair_operations", record["format_validation"])
        self.assertEqual(record["permutation_key"]["candidate_index"], 0)
        self.assertEqual(sorted(record["permutation"]), list(range(5)))
        self.assertEqual(sorted(record["inverse_permutation"]), list(range(5)))
        self.assertEqual(len(record["permutation_digests"]), 5)
        self.assertTrue(record["prompt_audit"]["passed"])
        self.assertEqual(len(record["prompt_audit"]["prompt_sha256"]), 64)
        self.assertEqual(set(record["label_scores"]), set(LABELS))
        for label, item in record["label_scores"].items():
            self.assertEqual(item["label"], label)
            self.assertTrue(item["token_ids"])
            self.assertTrue(all(math.isfinite(value) for value in item["token_logprobs"]))
            self.assertTrue(math.isfinite(item["sequence_logprob"]))
            self.assertTrue(math.isfinite(item["local_log_probability"]))
        self.assertTrue(math.isfinite(record["raw_margin"]))
        self.assertTrue(math.isfinite(record["clipped_margin"]))
        self.assertEqual(record["normalized_reward"], result["reward"])
        self.assertIn(record["top1"], LABELS)
        self.assertIsInstance(record["tie"], bool)
        self.assertEqual(record["reward_source"], ANSWER_MARGIN_REWARD_REVISION)
        self.assertFalse(record["masked"])
        self.assertTrue(record["eligible_for_grpo"])

    def test_scorer_timeout_aborts(self):
        class TimeoutScorer:
            def score(self, *_args, **_kwargs):
                raise TimeoutError("scorer timeout")

        with tempfile.TemporaryDirectory() as tmp, self.assertRaisesRegex(TimeoutError, "scorer timeout"):
            score_completion(completion(), packet(Path(tmp)), "E1", 0, scorer=TimeoutScorer(), key=key())

    def test_metadata_misalignment_aborts_even_when_core_qa_is_bad(self):
        wrong = PermutationKey(
            experiment_condition_id="temperature_0.5",
            phase="train",
            evidence_id="other",
            generation_seed_or_call_index=7,
            candidate_index=0,
            reward_revision=ANSWER_MARGIN_REWARD_REVISION,
        )
        with self.assertRaisesRegex(ValueError, "错位"):
            score_completion("bad", {}, "E1", 0, scorer=RecordingScorer(), key=wrong)

    def test_resolve_ordered_videos_rejects_mapping_and_media_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = packet(root)
            self.assertEqual(tuple(Path(item).name for item in resolve_ordered_videos(valid, "E1")), ("u1.mp4", "u2.mp4"))
            cases = [
                ({**valid, "evidence_id": "wrong"}, "evidence"),
                ({**valid, "required_users": ["u1", "u1"]}, "required_users"),
                ({**valid, "clips": valid["clips"][:1]}, "恰好"),
                ({**valid, "clips": [*valid["clips"], {"agent_name": "u3", "local_video": valid["clips"][0]["local_video"]}]}, "两段"),
                ({**valid, "clips": [{**valid["clips"][0], "local_video": "frames.jpg"}, valid["clips"][1]]}, "mp4"),
            ]
            for value, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                    resolve_ordered_videos(value, "E1")


if __name__ == "__main__":
    unittest.main()
