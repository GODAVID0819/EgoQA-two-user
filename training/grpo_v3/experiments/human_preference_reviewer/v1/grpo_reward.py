"""Convert one generator completion into a frozen score-only reviewer reward."""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

from ....shared.media import (
    ordered_generator_frame_paths,
    ordered_reviewer_video_paths,
    required_users,
    validate_bound_paths,
)

REWARD_REVISION = "score_only_ordinal_reviewer_v1"


def _parse_completion(raw: str) -> dict[str, Any]:
    text = str(raw).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("completion contains no JSON object")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("completion JSON must be an object")
    options = value.get("options")
    if not isinstance(options, list) or len(options) != 5 or any(
        not isinstance(item, str) or not item.strip() for item in options
    ):
        raise ValueError("completion must contain five non-empty options")
    correct = str(value.get("correct") or "").strip().upper()
    if correct not in "ABCDE" or len(correct) != 1:
        raise ValueError("completion correct must be A-E")
    if value.get("answer") != options[ord(correct) - ord("A")]:
        raise ValueError("completion answer must equal options[correct]")
    if not str(value.get("question") or "").strip():
        raise ValueError("completion question is required")
    return value


def prepare_completion_for_review(
    raw_completion: str,
    packet: Mapping[str, Any],
    *,
    evidence_id: str,
    candidate_index: int,
    dataset_generator_images: Any | None = None,
    dataset_reviewer_videos: Any | None = None,
) -> dict[str, Any]:
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    if str(packet.get("evidence_id") or "") != str(evidence_id):
        raise ValueError("packet evidence_id does not match completion metadata")
    users = required_users(packet)
    generator_images = ordered_generator_frame_paths(packet)
    full_videos = ordered_reviewer_video_paths(packet)
    if dataset_generator_images is not None:
        validate_bound_paths(
            dataset_generator_images,
            generator_images,
            field="generator_image_paths",
        )
    if dataset_reviewer_videos is not None:
        validate_bound_paths(
            dataset_reviewer_videos,
            full_videos,
            field="reviewer_videos",
        )
    try:
        qa = _parse_completion(raw_completion)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return {"status": "candidate_rejected", "candidate": None, "reason": str(error)}
    return {
        "status": "eligible",
        "reason": None,
        "candidate": {
            "review_key": f"{evidence_id}:candidate_{candidate_index}",
            "evidence_id": str(evidence_id),
            "question": str(qa["question"]),
            "options": list(qa["options"]),
            "correct": str(qa["correct"]).upper(),
            "answer": str(qa["answer"]),
            "video_a_path": full_videos[0],
            "video_b_path": full_videos[1],
            "video_a_user": users[0],
            "video_b_user": users[1],
        },
    }


def score_completion(
    raw_completion: str,
    packet: Mapping[str, Any],
    *,
    evidence_id: str,
    candidate_index: int,
    client: Any,
    dataset_generator_images: Any | None = None,
    dataset_reviewer_videos: Any | None = None,
) -> dict[str, Any]:
    prepared = prepare_completion_for_review(
        raw_completion,
        packet,
        evidence_id=evidence_id,
        candidate_index=candidate_index,
        dataset_generator_images=dataset_generator_images,
        dataset_reviewer_videos=dataset_reviewer_videos,
    )
    if prepared["status"] == "candidate_rejected":
        return {
            "reward": 0.0,
            "record": {
                "reward_revision": REWARD_REVISION,
                "reward_source": "deterministic_rejection",
                "rejection_reason": prepared["reason"],
                "reviewer_score": None,
            },
        }
    candidate = prepared["candidate"]
    reviewer_score = client.score(candidate)
    if reviewer_score.get("review_key") != candidate["review_key"] or reviewer_score.get("evidence_id") != candidate["evidence_id"]:
        raise RuntimeError("reviewer response identity does not match request")
    reward = float(reviewer_score.get("reward"))
    if not math.isfinite(reward) or not 0.0 <= reward <= 1.0:
        raise RuntimeError("reviewer reward must be finite and in [0, 1]")
    return {
        "reward": reward,
        "record": {
            "reward_revision": REWARD_REVISION,
            "reward_source": "score_only_ordinal_reviewer",
            "reviewer_score": reviewer_score,
        },
    }
