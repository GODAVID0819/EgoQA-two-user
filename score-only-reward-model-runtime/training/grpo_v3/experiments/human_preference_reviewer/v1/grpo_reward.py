"""Convert one generator completion into a frozen score-only reviewer reward."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

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


def _ordered_full_videos(packet: Mapping[str, Any], users: list[str]) -> list[str]:
    clips = packet.get("clips")
    if not isinstance(clips, list):
        raise ValueError("packet clips must be a list")
    by_user = {
        str(clip.get("agent_name") or clip.get("user") or ""): clip
        for clip in clips if isinstance(clip, Mapping)
    }
    paths = []
    for user in users:
        clip = by_user.get(user)
        if clip is None:
            raise ValueError(f"packet has no clip for required user {user}")
        value = next((
            str(clip.get(name) or "").strip()
            for name in ("full_local_video", "original_local_video", "source_local_video", "local_video")
            if str(clip.get(name) or "").strip()
        ), "")
        if not value or not Path(value).is_file():
            raise FileNotFoundError(f"materialized full video is missing for {user}: {value}")
        paths.append(value)
    return paths


def prepare_completion_for_review(
    raw_completion: str,
    packet: Mapping[str, Any],
    *,
    evidence_id: str,
    candidate_index: int,
) -> dict[str, Any]:
    if isinstance(candidate_index, bool) or not isinstance(candidate_index, int) or candidate_index < 0:
        raise ValueError("candidate_index must be a nonnegative integer")
    if str(packet.get("evidence_id") or "") != str(evidence_id):
        raise ValueError("packet evidence_id does not match completion metadata")
    users_value = packet.get("required_users")
    if not isinstance(users_value, list) or len(users_value) != 2:
        raise ValueError("packet must contain exactly two required_users")
    users = [str(value) for value in users_value]
    if len(set(users)) != 2:
        raise ValueError("packet required_users must be distinct")
    try:
        qa = _parse_completion(raw_completion)
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        return {"status": "candidate_rejected", "candidate": None, "reason": str(error)}
    videos = _ordered_full_videos(packet, users)
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
            "video_a_path": videos[0],
            "video_b_path": videos[1],
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
) -> dict[str, Any]:
    prepared = prepare_completion_for_review(
        raw_completion, packet, evidence_id=evidence_id, candidate_index=candidate_index
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
