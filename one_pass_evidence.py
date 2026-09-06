"""复用已完成的六用户 candidate asset，构造一次性 QA 输入。"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_GROUP_ORDER = (
    "DAY1::17200000",
    "DAY3::17000000",
    "DAY4::21400000",
)
SPEAKER_START_INDEX = {
    "DAY1::17200000": 0,
    "DAY3::17000000": 4,
    "DAY4::21400000": 2,
}
DROP_KEYS = {
    "block_diagnostics",
    "similarity_matrix",
    "frame_embeddings",
    "embeddings",
    "pair_results",
    "review_bundle",
}


def _compact_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _compact_value(item)
            for key, item in value.items()
            if str(key) not in DROP_KEYS
        }
    if isinstance(value, list):
        return [_compact_value(item) for item in value]
    return value


def _compact_clip(clip: dict[str, Any], *, group_id: str, speaker_index: int) -> dict[str, Any]:
    compact = _compact_value(clip)
    if not isinstance(compact, dict):
        raise ValueError(f"{group_id} speaker_{speaker_index}: selected clip is not an object")
    return compact


def _candidate_speaker_index(candidate: dict[str, Any], *, group_id: str) -> int:
    selection = candidate.get("selection")
    if not isinstance(selection, dict):
        raise ValueError(f"{group_id}: speaker candidate is missing selection")
    try:
        speaker_index = int(selection["speaker_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{group_id}: speaker candidate has invalid speaker_index") from exc
    if speaker_index not in range(6):
        raise ValueError(f"{group_id}: speaker_index must be in 0..5, got {speaker_index}")
    return speaker_index


def _compact_attempts(
    attempts: Any,
    *,
    group_id: str,
    expected_users: list[str],
) -> list[dict[str, Any]]:
    compact_attempts = []
    for attempt in attempts if isinstance(attempts, list) else []:
        if not isinstance(attempt, dict):
            continue
        speaker_index = attempt.get("speaker_index")
        if not isinstance(speaker_index, int) or speaker_index not in range(6):
            continue
        compact_attempts.append(
            {
                key: attempt.get(key)
                for key in ("speaker_index", "speaker_user", "status", "candidate_index")
                if attempt.get(key) is not None
            }
        )
    by_index = {int(row["speaker_index"]): row for row in compact_attempts}
    if set(by_index) != set(range(6)):
        raise ValueError(f"{group_id}: speaker_attempts must cover all six speaker indexes")
    for index, row in by_index.items():
        if row.get("status") != "succeeded":
            raise ValueError(f"{group_id} speaker_{index}: speaker attempt is not succeeded")
        if row.get("speaker_user") != expected_users[index]:
            raise ValueError(
                f"{group_id} speaker_{index}: speaker identity does not match selected clips"
            )
    return [by_index[index] for index in range(6)]


def compact_speaker_packets(
    asset_path: str | Path,
    *,
    source_job_id: str,
) -> list[dict[str, Any]]:
    """从一个 group asset 提取六个可直接供 QA loop 使用的 speaker packet。"""

    path = Path(asset_path)
    if not path.is_file():
        raise FileNotFoundError(f"candidate asset does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"candidate asset must contain one JSON object: {path}")
    day = payload.get("day")
    time_token = payload.get("time_token")
    group_id = f"{day}::{time_token}"
    if group_id not in EXPECTED_GROUP_ORDER:
        raise ValueError(f"unexpected one-pass generation group: {group_id}")

    candidates = payload.get("speaker_candidates")
    if not isinstance(candidates, list) or len(candidates) != 6:
        raise ValueError(
            f"{group_id}: expected six successful speaker candidates, "
            f"got {0 if not isinstance(candidates, list) else len(candidates)}"
        )

    candidates_by_index: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError(f"{group_id}: speaker candidate is not an object")
        speaker_index = _candidate_speaker_index(candidate, group_id=group_id)
        if speaker_index in candidates_by_index:
            raise ValueError(f"{group_id}: duplicate speaker_index {speaker_index}")
        candidates_by_index[speaker_index] = candidate
    if set(candidates_by_index) != set(range(6)):
        raise ValueError(f"{group_id}: six successful speaker candidates must cover indexes 0..5")

    first_candidate = candidates_by_index[0]
    first_clips = first_candidate.get("selected_clips")
    if not isinstance(first_clips, list) or len(first_clips) != 6:
        raise ValueError(f"{group_id} speaker_0: selected_clips must contain six videos")
    first_users = [clip.get("agent_name") for clip in first_clips if isinstance(clip, dict)]
    if len(first_users) != 6 or len(set(first_users)) != 6 or any(not user for user in first_users):
        raise ValueError(f"{group_id} speaker_0: selected clips must contain six unique users")
    users_by_speaker_index = {}
    for index, candidate in candidates_by_index.items():
        clips = candidate.get("selected_clips")
        if not isinstance(clips, list) or not clips or not isinstance(clips[0], dict):
            raise ValueError(f"{group_id} speaker_{index}: selected speaker clip is missing")
        speaker_user = clips[0].get("agent_name")
        if not isinstance(speaker_user, str) or not speaker_user:
            raise ValueError(f"{group_id} speaker_{index}: selected speaker identity is missing")
        users_by_speaker_index[index] = speaker_user
    if len(set(users_by_speaker_index.values())) != 6:
        raise ValueError(f"{group_id}: speaker candidates do not identify six unique users")
    global_users = [users_by_speaker_index[index] for index in range(6)]
    attempts = payload.get("speaker_attempts") or first_candidate.get("speaker_attempts")
    compact_attempts = _compact_attempts(
        attempts,
        group_id=group_id,
        expected_users=global_users,
    )

    packets: list[dict[str, Any]] = []
    for speaker_index in range(6):
        candidate = candidates_by_index[speaker_index]
        selected_clips = candidate.get("selected_clips")
        if not isinstance(selected_clips, list) or len(selected_clips) != 6:
            raise ValueError(
                f"{group_id} speaker_{speaker_index}: selected_clips must contain six videos"
            )
        users = [clip.get("agent_name") for clip in selected_clips if isinstance(clip, dict)]
        if len(users) != 6 or set(users) != set(global_users):
            raise ValueError(f"{group_id} speaker_{speaker_index}: selected clips have the wrong user set")
        if users[0] != global_users[speaker_index]:
            raise ValueError(
                f"{group_id} speaker_{speaker_index}: first clip is not the selected speaker"
            )
        compact_clips = [
            _compact_clip(clip, group_id=group_id, speaker_index=speaker_index)
            for clip in selected_clips
            if isinstance(clip, dict)
        ]
        for clip_index, clip in enumerate(compact_clips):
            for field in ("generator_local_video", "full_local_video"):
                video_path = clip.get(field)
                if not isinstance(video_path, str) or not video_path:
                    raise ValueError(
                        f"{group_id} speaker_{speaker_index} clip_{clip_index}: missing {field}"
                    )
                if not Path(video_path).is_file():
                    raise ValueError(
                        f"{group_id} speaker_{speaker_index} clip_{clip_index}: missing video {video_path}"
                    )
        speaker_user = users[0]
        base_evidence_id = f"EGOLIFE6U_CONSENSUS_{day}_{time_token}_S{speaker_index + 1}"
        media_roles = {
            str(clip.get("agent_name")): str(clip.get("media_role"))
            for clip in compact_clips
        }
        packets.append(
            {
                "evidence_id": base_evidence_id,
                "generation_group_id": group_id,
                "candidate_type": "six_user_speaker_consensus",
                "day": day,
                "time_token": time_token,
                "clip_clock": candidate.get("clip_clock") or payload.get("clip_clock"),
                "input_users": users,
                "required_users": users,
                "speaker_index": speaker_index,
                "speaker_user": speaker_user,
                "provider_users": users[1:],
                "evidence_provider_user": users[1],
                "evidence_provider_users": users[1:],
                "media_roles": media_roles,
                "requirement": (
                    "Six synchronized input videos are ordered as one speaker and five providers. "
                    "Generation uses the full speaker video and five temporally pruned provider "
                    "videos; groundedness and answerability use the six full original videos."
                ),
                "generator_media_mode": "speaker_full_five_provider_pruned_videos",
                "clips": compact_clips,
                "source_urls": {
                    "videos": [clip.get("video_url") for clip in compact_clips],
                    "gazes": [clip.get("gaze_url") for clip in compact_clips],
                    "overlays": [
                        clip.get("overlay_url")
                        for clip in compact_clips
                        if clip.get("overlay_url")
                    ],
                },
                "speaker_attempts": copy.deepcopy(compact_attempts),
                "provenance": {
                    "source_job_id": str(source_job_id),
                    "source_asset": str(path),
                    "source_candidate_index": speaker_index,
                    "source_generation_group_id": group_id,
                },
            }
        )
    return packets


def _packets_by_group(packets: Iterable[dict[str, Any]]) -> dict[str, dict[int, dict[str, Any]]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for packet in packets:
        group_id = str(packet.get("generation_group_id") or "")
        speaker_index = packet.get("speaker_index")
        if group_id not in EXPECTED_GROUP_ORDER or speaker_index not in range(6):
            raise ValueError(f"invalid one-pass packet identity: {group_id!r}, {speaker_index!r}")
        group = grouped.setdefault(group_id, {})
        if speaker_index in group:
            raise ValueError(f"duplicate one-pass packet: {group_id} speaker_{speaker_index}")
        group[speaker_index] = packet
    if set(grouped) != set(EXPECTED_GROUP_ORDER) or any(
        set(grouped[group_id]) != set(range(6)) for group_id in EXPECTED_GROUP_ORDER
    ):
        raise ValueError("one-pass evidence must contain exactly six speaker packets for each expected group")
    return grouped


def expand_one_pass_slots(
    packets_by_group: Iterable[dict[str, Any]],
    *,
    slots_per_group: int = 10,
) -> list[dict[str, Any]]:
    """按固定起点展开三组各十槽，并使每个 speaker 全局获得五槽。"""

    if slots_per_group != 10:
        raise ValueError("one-pass mode requires exactly 10 slots per generation group")
    grouped = _packets_by_group(packets_by_group)
    slots: list[dict[str, Any]] = []
    global_slot_index = 0
    for group_id in EXPECTED_GROUP_ORDER:
        speaker_order = [
            (SPEAKER_START_INDEX[group_id] + offset) % 6 for offset in range(6)
        ]
        speaker_order.extend(speaker_order[:4])
        for group_slot_index, speaker_index in enumerate(speaker_order):
            slot = copy.deepcopy(grouped[group_id][speaker_index])
            base_evidence_id = str(slot["evidence_id"])
            slot_id = f"{base_evidence_id}::one_pass_{global_slot_index:03d}"
            slot["base_evidence_id"] = base_evidence_id
            slot["evidence_id"] = slot_id
            slot["generation_slot_id"] = slot_id
            slot["generation_round_index"] = global_slot_index
            slot["one_pass_slot_index"] = global_slot_index
            slot["one_pass_group_slot_index"] = group_slot_index
            slots.append(slot)
            global_slot_index += 1
    return slots


def _write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> int:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count


def write_one_pass_evidence(
    asset_paths: Iterable[str | Path],
    *,
    compact_output: str | Path,
    expanded_output: str | Path,
    source_job_id: str,
) -> dict[str, Any]:
    """逐个读取 asset，并写出 18 packet 与 30 slot 两份 JSONL。"""

    compact_packets: list[dict[str, Any]] = []
    for asset_path in asset_paths:
        compact_packets.extend(
            compact_speaker_packets(asset_path, source_job_id=source_job_id)
        )
    slots = expand_one_pass_slots(compact_packets)
    compact_count = _write_jsonl(compact_output, compact_packets)
    expanded_count = _write_jsonl(expanded_output, slots)
    return {
        "source_job_id": str(source_job_id),
        "generation_groups": list(EXPECTED_GROUP_ORDER),
        "compact_packet_count": compact_count,
        "expanded_slot_count": expanded_count,
        "slots_per_group": 10,
        "speaker_slot_counts": {
            str(user): sum(slot.get("speaker_user") == user for slot in slots)
            for user in ("Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure")
        },
    }
