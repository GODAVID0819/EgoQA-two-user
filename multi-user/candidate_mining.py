"""Mine complementary multi-user evidence candidates from per-user observations."""

from __future__ import annotations

import argparse
import itertools
import re
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl, stable_id, write_jsonl


STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "into",
    "onto",
    "near",
    "person",
    "someone",
    "something",
    "wearer",
    "user",
}


def norm_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_'-]{2,}", text.lower())
        if token not in STOPWORDS
    }


def as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def observation_text(obs: dict[str, Any]) -> str:
    parts = []
    for key in ["location_guess", "visible_people", "salient_objects", "actions", "gaze_focus", "key_facts"]:
        value = obs.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def key_facts(row: dict[str, Any]) -> list[str]:
    obs = row.get("observation", {})
    facts = as_list(obs.get("key_facts"))
    if facts:
        return facts
    fallback = []
    fallback.extend(as_list(obs.get("actions")))
    fallback.extend(as_list(obs.get("gaze_focus")))
    fallback.extend(as_list(obs.get("salient_objects")))
    return fallback


def shared_anchors(rows: list[dict[str, Any]]) -> list[str]:
    token_sets = [norm_tokens(observation_text(row.get("observation", {}))) for row in rows]
    if not token_sets:
        return []
    anchors = set.intersection(*token_sets)
    return sorted(anchors)[:12]


def unique_facts_by_user(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    out = {}
    all_fact_tokens = [norm_tokens(" ".join(key_facts(row))) for row in rows]
    for idx, row in enumerate(rows):
        user = row["clip"]["agent_name"]
        other_tokens = set().union(*(tokens for j, tokens in enumerate(all_fact_tokens) if j != idx))
        facts = []
        for fact in key_facts(row):
            tokens = norm_tokens(fact)
            if tokens and len(tokens - other_tokens) > 0:
                facts.append(fact)
        out[user] = facts[:8]
    return out


def complementarity_score(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    anchors = shared_anchors(rows)
    uniques = unique_facts_by_user(rows)
    unique_count = sum(1 for facts in uniques.values() if facts)
    total_unique_facts = sum(len(facts) for facts in uniques.values())
    score = len(anchors) * 2 + unique_count * 3 + total_unique_facts
    if not anchors:
        score -= 2
    explanation = {
        "score": score,
        "shared_anchors": anchors,
        "unique_facts_by_user": uniques,
        "why_candidate": (
            "Candidate has shared visual/event anchors plus distinct per-user facts; "
            "QA generation should use at least one unique fact from each required user."
        ),
    }
    return score, explanation


def load_observations(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for row in iter_jsonl(path):
        obs = row.get("observation", {})
        if obs.get("status") not in {"ok", "manual", "dry_run"}:
            continue
        clip = row.get("clip", {})
        if not clip:
            continue
        rows.append(row)
    return rows


def build_candidate_packet(rows: list[dict[str, Any]], explanation: dict[str, Any]) -> dict[str, Any]:
    clips = [row["clip"] for row in rows]
    day = clips[0]["day"]
    time_tokens = [clip["time_token"] for clip in clips]
    packet_id = stable_id("EGOLIFE2U_SEM", day, min(time_tokens), *[clip["agent_id"] for clip in clips])
    packet_clips = []
    for row in rows:
        clip = dict(row["clip"])
        clip["observation"] = row.get("observation", {})
        packet_clips.append(clip)
    return {
        "evidence_id": packet_id,
        "candidate_type": "semantic_complementarity",
        "day": day,
        "time_tokens": time_tokens,
        "clip_clock_range": [clip.get("clip_clock") for clip in clips],
        "required_users": [clip["agent_name"] for clip in clips],
        "requirement": (
            "The final question must combine distinct facts from at least two listed users; "
            "each single-user evidence slice should be insufficient."
        ),
        "complementarity": explanation,
        "clips": packet_clips,
        "source_urls": {
            "videos": [clip["video_url"] for clip in clips],
            "gazes": [clip["gaze_url"] for clip in clips],
            "overlays": [clip.get("overlay_url") for clip in clips if clip.get("overlay_url")],
        },
    }


def mine_candidates(
    *,
    observations_path: str | Path,
    output_path: str | Path,
    target_count: int = 20,
    users_per_case: int = 2,
    max_time_gap_seconds: float = 90.0,
    min_score: int = 5,
) -> list[dict[str, Any]]:
    rows = load_observations(observations_path)
    by_day: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_day.setdefault(row["clip"]["day"], []).append(row)
    scored: list[tuple[int, dict[str, Any]]] = []
    for day_rows in by_day.values():
        day_rows = sorted(day_rows, key=lambda row: (row["clip"]["clock_seconds"], row["clip"]["agent_dir"]))
        for combo in itertools.combinations(day_rows, max(2, users_per_case)):
            users = [row["clip"]["agent_dir"] for row in combo]
            if len(set(users)) != len(users):
                continue
            times = [float(row["clip"]["clock_seconds"]) for row in combo]
            if max(times) - min(times) > max_time_gap_seconds:
                continue
            score, explanation = complementarity_score(list(combo))
            if score < min_score:
                continue
            scored.append((score, build_candidate_packet(list(combo), explanation)))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    packets = [packet for _, packet in scored[:target_count]]
    write_jsonl(output_path, packets)
    return packets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mine complementary two-user evidence candidates")
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--users-per-case", type=int, default=2)
    parser.add_argument("--max-time-gap-seconds", type=float, default=90.0)
    parser.add_argument("--min-score", type=int, default=5)
    args = parser.parse_args(argv)
    packets = mine_candidates(
        observations_path=args.observations,
        output_path=args.output,
        target_count=args.target_count,
        users_per_case=args.users_per_case,
        max_time_gap_seconds=args.max_time_gap_seconds,
        min_score=args.min_score,
    )
    print(f"wrote {len(packets)} semantic candidates to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

