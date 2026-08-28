from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from egolife_two_user_qa.evidence_chunk_review import (
    aggregate_evidence_user_votes,
    evidence_segment_specs,
    validate_segment_observation,
)


USERS = ["Jake", "Alice", "Lucia", "Katrina", "Shure", "Tasha"]


def time_tokens(count: int) -> list[str]:
    start = datetime(2000, 1, 1, 20, 6, 0)
    return [
        (start + timedelta(seconds=30 * index)).strftime("%H%M%S") + "00"
        for index in range(count)
    ]


def packet_with_segments(count: int) -> dict[str, object]:
    tokens = time_tokens(count)
    return {
        "evidence_id": "packet",
        "required_users": USERS,
        "clips": [
            {
                "agent_name": user,
                "segments": [
                    {"time_token": token, "video_url": f"https://example/{user}/{token}"}
                    for token in tokens
                ],
            }
            for user in USERS
        ],
    }


def full_video_paths() -> list[str]:
    return [f"{user}.mp4" for user in USERS]


def valid_observation(*, user: str = "Jake", segment_count: int = 20) -> dict[str, object]:
    return {
        "user": user,
        "segments": [
            {
                "segment_index": index,
                "time_token": token,
                "claims": [
                    {
                        "claim": "the screen shows option B",
                        "status": "SUPPORTED",
                        "confidence": "HIGH",
                        "visual_description": "The screen text is legible.",
                        "original_time_range": "00:00-00:05",
                    }
                ],
            }
            for index, token in enumerate(time_tokens(segment_count))
        ],
        "user_vote": {
            "visible": True,
            "confidence": "HIGH",
            "supported_option": "B",
            "supporting_segment_indices": [0],
            "reason": "The option is directly legible.",
        },
    }


def make_observations(votes: list[str | None], confidences: list[str] | None = None):
    rows = []
    for index, (user, option) in enumerate(zip(USERS, votes)):
        confidence = (confidences or ["HIGH"] * len(votes))[index]
        rows.append(
            {
                "user": user,
                "user_vote": {
                    "visible": option is not None,
                    "confidence": confidence,
                    "supported_option": option,
                    "supporting_segment_indices": [0] if option is not None else [],
                    "reason": "direct view" if option is not None else "not visible",
                },
            }
        )
    return rows


def test_evidence_segment_specs_accepts_twenty_segments_for_ten_minutes() -> None:
    specs = evidence_segment_specs(packet_with_segments(20), full_video_paths())

    assert all(len(rows) == 20 for rows in specs.values())
    assert specs["Jake"][0]["start_seconds"] == 0.0
    assert specs["Jake"][-1]["start_seconds"] == 570.0


def test_evidence_segment_specs_preserves_six_segment_three_minute_contract() -> None:
    specs = evidence_segment_specs(packet_with_segments(6), full_video_paths())

    assert all(len(rows) == 6 for rows in specs.values())


def test_evidence_segment_specs_rejects_non_uniform_segment_counts() -> None:
    packet = packet_with_segments(20)
    packet["clips"][0]["segments"].pop()

    with pytest.raises(ValueError, match="same complete segment count"):
        evidence_segment_specs(packet, full_video_paths())


def test_evidence_segment_specs_rejects_discontinuous_time_tokens() -> None:
    packet = packet_with_segments(20)
    packet["clips"][0]["segments"][3]["time_token"] = "21000000"

    with pytest.raises(ValueError, match="consecutive 30-second time tokens"):
        evidence_segment_specs(packet, full_video_paths())


def test_validate_segment_observation_requires_high_confidence_vote() -> None:
    observation = valid_observation()
    observation["user_vote"] = {
        "visible": True,
        "confidence": "LOW",
        "supported_option": "B",
        "supporting_segment_indices": [3],
        "reason": "The object is blurry.",
    }

    errors = validate_segment_observation(
        observation,
        expected_user="Jake",
        expected_time_tokens=time_tokens(20),
    )

    assert "non-HIGH vote must not select an option" in errors


def test_validate_segment_observation_requires_supporting_segment_for_vote() -> None:
    observation = valid_observation()
    observation["user_vote"]["supporting_segment_indices"] = []

    errors = validate_segment_observation(
        observation,
        expected_user="Jake",
        expected_time_tokens=time_tokens(20),
    )

    assert "HIGH visible vote must cite at least one supporting segment" in errors


@pytest.mark.parametrize(
    ("votes", "passed"),
    [
        (["B", "B", "B", None, None, None], True),
        (["B", "B", "A", None, None, None], True),
        (["B", "B", "A", "A", None, None], False),
        (["B", "B", "B", "A", "A", "A"], False),
        ([None, None, None, None, None, None], False),
    ],
)
def test_aggregate_evidence_user_votes(votes: list[str | None], passed: bool) -> None:
    summary = aggregate_evidence_user_votes("B", make_observations(votes))

    assert summary["passed"] is passed


def test_aggregate_evidence_user_votes_excludes_non_high_confidence_views() -> None:
    summary = aggregate_evidence_user_votes(
        "B",
        make_observations(
            ["B", "B", "A", "A", None, None],
            ["HIGH", "HIGH", "LOW", "MEDIUM", "LOW", "LOW"],
        ),
    )

    assert summary["passed"] is True
    assert summary["visible_user_count"] == 2
    assert summary["option_support_counts"]["B"] == 2
    assert summary["option_support_counts"]["A"] == 0


def test_aggregate_evidence_user_votes_rejects_competing_threshold_option() -> None:
    summary = aggregate_evidence_user_votes(
        "B",
        make_observations(["B", "B", "B", "A", "A", "A"]),
    )

    assert summary["threshold_options"] == ["A", "B"]
    assert summary["passed"] is False
