from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
if "egolife_two_user_qa" not in sys.modules:
    package = types.ModuleType("egolife_two_user_qa")
    package.__path__ = [str(ROOT)]
    sys.modules["egolife_two_user_qa"] = package

from egolife_two_user_qa.one_pass_evidence import (  # noqa: E402
    compact_speaker_packets,
    expand_one_pass_slots,
)
from egolife_two_user_qa.one_pass_summary import (  # noqa: E402
    summarize_one_pass_rows,
    update_one_pass_manifest,
)


USERS = ["Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"]
START_BY_GROUP = {
    "DAY1::17200000": 0,
    "DAY3::17000000": 4,
    "DAY4::21400000": 2,
}


def _clip(tmp_path: Path, user: str, speaker_index: int) -> dict[str, object]:
    full = tmp_path / f"{user}.full.mp4"
    generator = tmp_path / f"{user}.generator.mp4"
    full.write_bytes(b"full")
    generator.write_bytes(b"generator")
    return {
        "agent_name": user,
        "agent_dir": user.lower(),
        "media_role": (
            "speaker_reference_unpruned"
            if user == USERS[speaker_index]
            else "provider_similarity_pruned"
        ),
        "local_video": str(generator),
        "generator_local_video": str(generator),
        "full_local_video": str(full),
        "original_local_video": str(full),
        "duration_seconds": 600.0,
        "temporal_pruning": {
            "keep_intervals": [[0.0, 30.0]],
            "block_diagnostics": [{"should_not": "survive compaction"}],
        },
        "frames": [{"timestamp_seconds": 12.0, "path": str(tmp_path / f"{user}.jpg")}],
        "original_time_range": [0.0, 30.0],
    }


def _asset(tmp_path: Path, group_id: str) -> Path:
    day, time_token = group_id.split("::")
    candidates = []
    speaker_attempts = [
        {
            "speaker_index": speaker_index,
            "speaker_user": speaker_user,
            "status": "succeeded",
            "candidate_index": speaker_index,
        }
        for speaker_index, speaker_user in enumerate(USERS)
    ]
    for speaker_index, speaker_user in enumerate(USERS):
        candidates.append(
            {
                "day": day,
                "time_token": time_token,
                "clip_clock": "00:00:00-00:10:00",
                "selection": {
                    "method": "six_user_speaker_consensus",
                    "speaker_index": speaker_index,
                    "speaker_user": speaker_user,
                },
                "speaker_consensus_pruning": {
                    "method": "clustered_speaker_provider_all_pairs_pruning",
                    "block_diagnostics": [{"huge": "diagnostic"}],
                },
                "selected_clips": [
                    _clip(tmp_path, user, speaker_index)
                    for user in [
                        USERS[speaker_index],
                        *[user for user in USERS if user != USERS[speaker_index]],
                    ]
                ],
                "speaker_attempts": speaker_attempts,
                "similarity_matrix": [[1.0]],
            }
        )
    path = tmp_path / f"{day}_{time_token}_group_relative_clip.json"
    path.write_text(
        json.dumps(
            {
                "day": day,
                "time_token": time_token,
                "speaker_attempts": speaker_attempts,
                "speaker_candidates": candidates,
                "similarity_matrix": [[1.0]],
                "block_diagnostics": [{"huge": "group diagnostic"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_compact_speaker_packets_keep_all_speakers_and_media_contract(tmp_path: Path) -> None:
    asset_path = _asset(tmp_path, "DAY1::17200000")

    packets = compact_speaker_packets(asset_path, source_job_id="16699348")

    assert [packet["speaker_index"] for packet in packets] == list(range(6))
    assert [packet["speaker_user"] for packet in packets] == USERS
    assert all(len(packet["clips"]) == 6 for packet in packets)
    assert all(
        all(
            clip["full_local_video"]
            and clip["generator_local_video"]
            and clip["temporal_pruning"]
            for clip in packet["clips"]
        )
        for packet in packets
    )
    assert all("block_diagnostics" not in packet for packet in packets)
    assert all("similarity_matrix" not in packet for packet in packets)
    assert all(packet["provenance"]["source_job_id"] == "16699348" for packet in packets)


def test_expand_one_pass_slots_has_fixed_30_rows_and_balanced_speakers(tmp_path: Path) -> None:
    packets = []
    for group_id in START_BY_GROUP:
        packets.extend(compact_speaker_packets(_asset(tmp_path, group_id), source_job_id="16699348"))

    slots = expand_one_pass_slots(packets, slots_per_group=10)

    assert len(slots) == 30
    assert len({slot["generation_slot_id"] for slot in slots}) == 30
    assert {group: sum(slot["generation_group_id"] == group for slot in slots) for group in START_BY_GROUP} == {
        group: 10 for group in START_BY_GROUP
    }
    speaker_counts = {user: 0 for user in USERS}
    for slot in slots:
        speaker_counts[str(slot["speaker_user"])] += 1
        assert slot["base_evidence_id"]
        assert slot["generation_slot_id"].startswith(f"{slot['base_evidence_id']}::one_pass_")
    assert speaker_counts == {user: 5 for user in USERS}


def test_compact_speaker_packets_reject_incomplete_speaker_set(tmp_path: Path) -> None:
    asset_path = _asset(tmp_path, "DAY1::17200000")
    payload = json.loads(asset_path.read_text(encoding="utf-8"))
    payload["speaker_candidates"] = payload["speaker_candidates"][:5]
    asset_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="six successful speaker candidates"):
        compact_speaker_packets(asset_path, source_job_id="16699348")


def test_summarize_one_pass_rows_uses_fixed_denominator_and_keeps_parse_failure() -> None:
    evidence = [
        {
            "generation_slot_id": f"slot-{index:03d}",
            "generation_group_id": "DAY1::17200000" if index < 10 else "DAY3::17000000",
            "speaker_index": index % 6,
            "speaker_user": USERS[index % 6],
        }
        for index in range(30)
    ]
    accepted = [{**evidence[0], "qa_id": "qa-0"}]
    rejected = [
        {
            **row,
            "attempts": [
                {
                    "attempt": 1,
                    "reason": "judge rejected the generated question",
                }
            ],
        }
        for row in evidence[1:]
    ]
    rejected[0]["attempts"][0]["reason"] = "Generator output was not valid JSON: parse failed"
    rejected[1]["attempts"] = [
        {
            "attempt": 1,
            "reason": "qa_formality rejected the generated question",
            "qa": {
                "review": {
                    "final_decision": {"rejection_stage": "judger"},
                    "judger": {
                        "checks": {
                            "qa_formality": {"status": "FAIL"},
                            "evidence_groundedness": {"status": "PASS"},
                            "answerability": {"status": "PASS"},
                        }
                    },
                }
            },
        }
    ]
    prompts = [
        {
            "stage": "generation",
            "generation_slot_id": row["generation_slot_id"],
            "elapsed_seconds": 1.0,
        }
        for row in evidence
    ]

    result = summarize_one_pass_rows(
        evidence_rows=evidence,
        accepted_rows=accepted,
        rejected_rows=rejected,
        prompt_rows=prompts,
        attempt_rows=[],
        expected_slot_count=30,
    )

    assert result["status"] == "completed"
    assert result["slot_count"] == 30
    assert result["completed_slot_count"] == 30
    assert result["accepted_count"] == 1
    assert result["rejected_count"] == 29
    assert result["parse_failed_count"] == 1
    assert result["rejected_by_formality_count"] == 1
    assert result["acceptance_rate"] == pytest.approx(1 / 30)
    assert result["attempt_count_distribution"] == {"1": 30}


def test_update_one_pass_manifest_records_result_status(tmp_path: Path) -> None:
    manifest = tmp_path / "job_manifest.json"
    result_path = tmp_path / "six_user_qa_result.json"
    manifest.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    result_path.write_text("{}", encoding="utf-8")

    update_one_pass_manifest(manifest, result_path, status="incomplete")

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload == {
        "status": "incomplete",
        "result_path": str(result_path),
    }
