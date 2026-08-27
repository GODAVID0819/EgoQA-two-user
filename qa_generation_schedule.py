from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from typing import Any


DIVERSITY_RELATION_FOCUSES = (
    "missing detail",
    "comparison",
    "identity link",
    "handoff follow-up",
    "state verification",
    "temporal relation",
    "sequence",
    "interaction",
)


def diversity_focus_for_round(
    packet: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    providers = list(packet.get("required_users") or [])[1:]
    band_index = int(round_index) % 6
    return {
        "round_index": int(round_index),
        "temporal_band_seconds": [band_index * 30, (band_index + 1) * 30],
        "focal_provider": (
            str(providers[int(round_index) % len(providers)])
            if providers
            else None
        ),
        "relation_focus": DIVERSITY_RELATION_FOCUSES[
            int(round_index) % len(DIVERSITY_RELATION_FOCUSES)
        ],
    }


def generation_slot_id(evidence_id: str, round_index: int) -> str:
    return f"{evidence_id}::round_{round_index:04d}"


def round_robin_generation_slots(
    packets: Iterable[dict[str, Any]],
    *,
    max_slots: int | None = None,
) -> Iterator[dict[str, Any]]:
    available = list(packets)
    if not available:
        return

    emitted = 0
    round_index = 0
    while max_slots is None or emitted < max_slots:
        for packet in available:
            if max_slots is not None and emitted >= max_slots:
                return
            slot = dict(packet)
            slot["base_evidence_id"] = str(packet["evidence_id"])
            slot["generation_round_index"] = round_index
            slot["generation_diversity_focus"] = diversity_focus_for_round(
                packet,
                round_index,
            )
            slot["generation_slot_id"] = generation_slot_id(
                str(packet["evidence_id"]), round_index
            )
            yield slot
            emitted += 1
        round_index += 1


def deadline_reached(
    deadline_epoch_seconds: float,
    *,
    now_epoch_seconds: float | None = None,
) -> bool:
    now = time.time() if now_epoch_seconds is None else now_epoch_seconds
    return now >= deadline_epoch_seconds
