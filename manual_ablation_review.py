"""Build an evidence-first manual review package from QA or intermediate JSONL."""

from __future__ import annotations

import argparse
import collections
import copy
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = (
    ("base_1", Path(r"C:\Users\haoya\Downloads\qa_mcq_base_1.jsonl")),
    ("base_2", Path(r"C:\Users\haoya\Downloads\qa_mcq_base_2.jsonl")),
    (
        "pruned_judges",
        Path(r"C:\Users\haoya\Downloads\qa_mcq_pruned_judges.jsonl"),
    ),
    ("fps_0p5", Path(r"C:\Users\haoya\Downloads\qa_mcq_fps_0p5.jsonl")),
    (
        "threshold_0p85",
        Path(r"C:\Users\haoya\Downloads\qa_mcq_thresh_0p85.jsonl"),
    ),
    ("k_8", Path(r"C:\Users\haoya\Downloads\qa_mcq_k=8.jsonl")),
)

RUN_LABELS = {
    "base_1": "Base 1",
    "base_2": "Base 2",
    "pruned_judges": "Pruned judges",
    "fps_0p5": "0.5 FPS",
    "threshold_0p85": "Threshold 0.85",
    "k_8": "K = 8",
}

SIX_PARTICIPANTS = (
    ("A1_JAKE", "Jake"),
    ("A2_ALICE", "Alice"),
    ("A3_TASHA", "Tasha"),
    ("A4_LUCIA", "Lucia"),
    ("A5_KATRINA", "Katrina"),
    ("A6_SHURE", "Shure"),
)
VIDEO_DATASET_URL = "https://huggingface.co/datasets/lmms-lab/EgoLife/resolve/main"
_VIDEO_KEY_RE = re.compile(
    r"/(?P<agent_dir>A[1-6]_[A-Z]+)/(?P<day>DAY[1-7])/"
    r"(?P=day)_(?P=agent_dir)_(?P<time_token>\d{8})\.mp4(?:$|[?#])",
    re.IGNORECASE,
)

REVIEW_COLUMNS = [
    "evidence_order",
    "evidence_id",
    "qa_count_for_evidence",
    "run",
    "question_type",
    "question",
    "option_A",
    "option_B",
    "option_C",
    "option_D",
    "option_E",
    "model_correct_letter",
    "model_answer",
    "review_status",
    "reviewer_answer",
    "answer_correct",
    "evidence_grounded",
    "asker_alone_answerable",
    "two_user_needed",
    "wording_formality",
    "error_tags",
    "reviewer_notes",
    "reviewed_at",
    "qa_id",
    "source_file",
    "required_users",
    "attempt_count",
    "judge_video_source",
    "video_count",
    "video_1_user",
    "video_1_url",
    "video_2_user",
    "video_2_url",
    "video_3_user",
    "video_3_url",
    "video_4_user",
    "video_4_url",
    "video_5_user",
    "video_5_url",
    "video_6_user",
    "video_6_url",
    "generator_rationale",
    "evidence_claims",
    "referred_timestamps",
    "single_user_answerability",
    "combined_answerability",
    "why_two_users_needed",
    "review_key",
]


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _serialized_bytes(value: Any) -> int:
    return len(_compact_json(value).encode("utf-8"))


def _count_nested_key(value: Any, target: str) -> int:
    if isinstance(value, dict):
        return (1 if target in value else 0) + sum(
            _count_nested_key(child, target) for child in value.values()
        )
    if isinstance(value, list):
        return sum(_count_nested_key(child, target) for child in value)
    return 0


def _read_jsonl(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        raw.decode("utf-8-sig").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(parsed)
    return raw, rows


def _qa_shaped(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and bool(str(value.get("question") or "").strip())
        and isinstance(value.get("options"), list)
    )


def _accepted_value(value: Any) -> bool:
    if value is True:
        return True
    return str(value or "").strip().lower() in {"accept", "accepted", "pass", "passed"}


def _attempt_was_accepted(attempt: dict[str, Any]) -> bool:
    result = attempt.get("result")
    return (
        _accepted_value(attempt.get("accepted"))
        or _accepted_value(attempt.get("status"))
        or (isinstance(result, dict) and _accepted_value(result.get("accepted")))
    )


def _qa_from_attempt(attempt: dict[str, Any]) -> dict[str, Any] | None:
    """Recover the final QA from standard and cyclic intermediate layouts."""

    for candidate in (
        attempt.get("qa"),
        (attempt.get("candidate") or {}).get("qa")
        if isinstance(attempt.get("candidate"), dict)
        else None,
    ):
        if _qa_shaped(candidate):
            return copy.deepcopy(candidate)

    generation = attempt.get("generation")
    if not isinstance(generation, dict):
        return None
    parsed = generation.get("parsed_qa")
    normalized = generation.get("normalized_qa")
    if not _qa_shaped(parsed):
        return None
    qa = copy.deepcopy(normalized) if isinstance(normalized, dict) else {}
    qa.update(copy.deepcopy(parsed))
    return qa


def _attempt_human_audit(attempt: dict[str, Any]) -> dict[str, Any]:
    media = attempt.get("media")
    if isinstance(media, dict) and isinstance(media.get("human_audit"), dict):
        return media["human_audit"]
    trace = attempt.get("generation_trace")
    if isinstance(trace, dict):
        media = trace.get("media")
        if isinstance(media, dict) and isinstance(media.get("human_audit"), dict):
            return media["human_audit"]
    return {}


def _enrich_intermediate_qa(
    qa: dict[str, Any],
    *,
    row: dict[str, Any],
    attempt: dict[str, Any],
) -> dict[str, Any]:
    qa = copy.deepcopy(qa)
    audit = _attempt_human_audit(attempt)
    for key in ("evidence_id", "qa_id", "question_type", "generation_mode"):
        if not qa.get(key):
            qa[key] = attempt.get(key) or row.get(key)
    attempt_number = attempt.get("attempt") or row.get("attempt")
    qa["attempt_count"] = int(attempt_number or qa.get("attempt_count") or 0)

    for key in ("required_users", "source_urls", "video_evidence", "human_audit"):
        if not qa.get(key):
            if key == "human_audit":
                value = audit
            else:
                value = audit.get(key)
            if value:
                qa[key] = copy.deepcopy(value)

    if not qa.get("judge_video_source"):
        media = attempt.get("media")
        qa["judge_video_source"] = (
            media.get("judge_media_role") if isinstance(media, dict) else None
        ) or row.get("judge_video_source") or ""

    options = qa.get("options")
    correct = str(qa.get("correct") or "").strip().upper()
    if not qa.get("answer") and isinstance(options, list) and correct in "ABCDE":
        index = ord(correct) - ord("A")
        if index < len(options):
            qa["answer"] = options[index]
    return qa


def _accepted_qa_rows(
    rows: list[dict[str, Any]],
    *,
    source: Path,
) -> tuple[list[dict[str, Any]], int]:
    """Select accepted QAs and reconstruct them when the source is intermediate."""

    accepted: list[dict[str, Any]] = []
    skipped = 0
    for row_number, row in enumerate(rows, 1):
        status = str(row.get("status") or "").strip().lower()
        is_explicit_intermediate = bool(status) or isinstance(row.get("attempts"), list)

        if _qa_shaped(row) and (not status or _accepted_value(status)):
            accepted.append(copy.deepcopy(row))
            continue
        if not is_explicit_intermediate or not _accepted_value(status):
            skipped += 1
            continue

        attempts = row.get("attempts")
        attempt_rows = [item for item in attempts or [] if isinstance(item, dict)]
        chosen: dict[str, Any] | None = None
        qa: dict[str, Any] | None = None
        for attempt in reversed(attempt_rows):
            candidate = _qa_from_attempt(attempt)
            if candidate is not None and _attempt_was_accepted(attempt):
                chosen, qa = attempt, candidate
                break
        if qa is None:
            for attempt in reversed(attempt_rows):
                candidate = _qa_from_attempt(attempt)
                if candidate is not None:
                    chosen, qa = attempt, candidate
                    break
        if qa is None and _qa_shaped(row.get("qa")):
            chosen, qa = row, copy.deepcopy(row["qa"])
        if qa is None or chosen is None:
            raise ValueError(
                f"{source}:{row_number}: accepted intermediate row does not contain "
                "a recoverable parsed QA"
            )
        accepted.append(_enrich_intermediate_qa(qa, row=row, attempt=chosen))
    return accepted, skipped


def _parse_inputs(values: list[str]) -> list[tuple[str, Path]]:
    if not values:
        return list(DEFAULT_INPUTS)
    parsed: list[tuple[str, Path]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--input must use LABEL=PATH, received {value!r}")
        label, raw_path = value.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"--input has an empty label: {value!r}")
        parsed.append((label, Path(raw_path)))
    return parsed


def _source_videos(row: dict[str, Any]) -> list[dict[str, Any]]:
    source_urls = row.get("source_urls")
    urls = source_urls.get("videos") if isinstance(source_urls, dict) else None
    evidence = row.get("video_evidence")
    evidence_rows = evidence if isinstance(evidence, list) else []
    videos: list[dict[str, Any]] = []
    if isinstance(urls, list):
        for index, raw_url in enumerate(urls):
            if not raw_url:
                continue
            metadata = (
                evidence_rows[index]
                if index < len(evidence_rows)
                and isinstance(evidence_rows[index], dict)
                else {}
            )
            videos.append(
                {
                    "user": str(
                        metadata.get("user")
                        or metadata.get("agent_dir")
                        or f"User {index + 1}"
                    ),
                    "agent_dir": str(metadata.get("agent_dir") or ""),
                    "day": str(metadata.get("day") or ""),
                    "time_token": str(metadata.get("time_token") or ""),
                    "clip_clock": str(metadata.get("clip_clock") or ""),
                    "url": str(raw_url),
                    "role": "selected evidence pair",
                    "alignment": "source metadata",
                }
            )
    if videos:
        return videos
    for index, metadata in enumerate(evidence_rows):
        if not isinstance(metadata, dict):
            continue
        raw_url = metadata.get("video_url")
        if not raw_url:
            continue
        videos.append(
            {
                "user": str(
                    metadata.get("user")
                    or metadata.get("agent_dir")
                    or f"User {index + 1}"
                ),
                "agent_dir": str(metadata.get("agent_dir") or ""),
                "day": str(metadata.get("day") or ""),
                "time_token": str(metadata.get("time_token") or ""),
                "clip_clock": str(metadata.get("clip_clock") or ""),
                "url": str(raw_url),
                "role": "selected evidence pair",
                "alignment": "source metadata",
            }
        )
    return videos


def _video_key(video: dict[str, Any]) -> tuple[str, str, str]:
    agent_dir = str(video.get("agent_dir") or "").upper()
    day = str(video.get("day") or "").upper()
    time_token = str(video.get("time_token") or "")
    match = _VIDEO_KEY_RE.search(str(video.get("url") or ""))
    if match:
        agent_dir = agent_dir or match.group("agent_dir").upper()
        day = day or match.group("day").upper()
        time_token = time_token or match.group("time_token")
    return agent_dir, day, time_token


def _six_user_videos(
    videos: list[dict[str, Any]],
    *,
    evidence_id: str,
) -> list[dict[str, Any]]:
    """Expand a synchronized selected pair to the six EgoLife participants."""

    known_names = {name.lower(): agent_dir for agent_dir, name in SIX_PARTICIPANTS}
    by_agent: dict[str, dict[str, Any]] = {}
    day = ""
    time_token = ""
    for video in videos:
        agent_dir, video_day, video_token = _video_key(video)
        if not agent_dir:
            agent_dir = known_names.get(str(video.get("user") or "").lower(), "")
        if agent_dir:
            by_agent.setdefault(agent_dir, video)
        day = day or video_day
        time_token = time_token or video_token

    if not day or not time_token:
        raise ValueError(
            f"{evidence_id}: cannot derive DAY and time token needed for six-user review"
        )

    expanded: list[dict[str, Any]] = []
    for agent_dir, name in SIX_PARTICIPANTS:
        existing = by_agent.get(agent_dir)
        if existing is not None:
            item = copy.deepcopy(existing)
            item["user"] = str(item.get("user") or name)
            item["agent_dir"] = agent_dir
            item["day"] = str(item.get("day") or day)
            item["time_token"] = str(item.get("time_token") or time_token)
            item["role"] = str(item.get("role") or "selected evidence pair")
            item["alignment"] = str(item.get("alignment") or "source metadata")
        else:
            item = {
                "user": name,
                "agent_dir": agent_dir,
                "day": day,
                "time_token": time_token,
                "clip_clock": (
                    f"{time_token[0:2]}:{time_token[2:4]}:"
                    f"{time_token[4:6]}.{time_token[6:8]}"
                ),
                "url": (
                    f"{VIDEO_DATASET_URL}/{agent_dir}/{day}/"
                    f"{day}_{agent_dir}_{time_token}.mp4"
                ),
                "role": "additional participant context",
                "alignment": "constructed exact DAY/time candidate",
            }
        expanded.append(item)
    return expanded


def _evidence_claims(row: dict[str, Any]) -> str:
    claims = row.get("evidence")
    if not isinstance(claims, list):
        claims = row.get("per_user_evidence_claims")
    if not isinstance(claims, list):
        return ""
    rendered: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        user = str(claim.get("user") or "Unknown user")
        fact = str(claim.get("needed_fact") or claim.get("claim") or "")
        timeframe = str(claim.get("timeframe") or "")
        suffix = f" [{timeframe}]" if timeframe else ""
        rendered.append(f"{user}: {fact}{suffix}".strip())
    return " | ".join(rendered)


def _referred_timestamps(row: dict[str, Any]) -> str:
    timestamps = row.get("referred_timestamps")
    if not isinstance(timestamps, list):
        return ""
    rendered: list[str] = []
    for item in timestamps:
        if not isinstance(item, dict):
            continue
        user = str(item.get("user") or "Unknown user")
        timestamp = item.get("timestamp_seconds")
        moment = str(item.get("moment") or "")
        if isinstance(timestamp, (int, float)):
            rendered.append(f"{user} @ {timestamp:g}s: {moment}".strip())
        else:
            rendered.append(f"{user}: {moment}".strip())
    return " | ".join(rendered)


def _evidence_sort_key(item: dict[str, Any]) -> tuple[Any, ...]:
    videos = item.get("videos") or []
    first = videos[0] if videos else {}
    day_text = str(first.get("day") or "")
    day_match = re.search(r"(\d+)", day_text)
    day = int(day_match.group(1)) if day_match else 999
    token_text = str(first.get("time_token") or "")
    token_digits = re.sub(r"\D", "", token_text)
    token = int(token_digits) if token_digits else 999999999
    return day, token, str(item["evidence_id"])


def _normalize_qa(
    row: dict[str, Any],
    *,
    run_id: str,
    run_label: str,
    source_file: str,
) -> dict[str, Any]:
    options = row.get("options")
    normalized_options = [str(value) for value in options] if isinstance(options, list) else []
    normalized_options = (normalized_options + [""] * 5)[:5]
    qa_id = str(row.get("qa_id") or "")
    evidence_id = str(row.get("evidence_id") or "")
    review_key = f"{run_id}::{evidence_id}::{qa_id}"
    return {
        "review_key": review_key,
        "run_id": run_id,
        "run": run_label,
        "source_file": source_file,
        "qa_id": qa_id,
        "question_type": str(row.get("question_type") or ""),
        "question": str(row.get("question") or ""),
        "options": normalized_options,
        "correct": str(row.get("correct") or ""),
        "answer": str(row.get("answer") or ""),
        "required_users": [
            str(value) for value in (row.get("required_users") or [])
        ],
        "attempt_count": int(row.get("attempt_count") or 0),
        "judge_video_source": str(row.get("judge_video_source") or ""),
        "generator_rationale": str(row.get("generator_rationale") or ""),
        "evidence_claims": _evidence_claims(row),
        "referred_timestamps": _referred_timestamps(row),
        "single_user_answerability": _compact_json(
            row.get("single_user_answerability") or {}
        ),
        "combined_answerability": str(row.get("combined_answerability") or ""),
        "why_two_users_needed": str(row.get("why_two_users_needed") or ""),
    }


def build_review_data(
    inputs: list[tuple[str, Path]],
    *,
    six_user: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    run_order = {run_id: index for index, (run_id, _) in enumerate(inputs)}
    run_summaries: list[dict[str, Any]] = []
    evidence_map: dict[str, dict[str, Any]] = {}
    id_sets: dict[str, set[str]] = {}
    file_hashes: list[str] = []
    shared_top_level_keys: set[str] | None = None

    for run_id, path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
        raw, source_rows = _read_jsonl(path)
        rows, skipped_row_count = _accepted_qa_rows(source_rows, source=path)
        if not rows:
            raise ValueError(f"{path}: no accepted QAs were found")
        file_hash = hashlib.sha256(raw).hexdigest()
        file_hashes.append(f"{run_id}:{file_hash}")
        key_counts = collections.Counter(key for row in rows for key in row)
        field_bytes: collections.Counter[str] = collections.Counter()
        for row in rows:
            for key, value in row.items():
                field_bytes[key] += _serialized_bytes(value)
        evidence_ids = [str(row.get("evidence_id") or "") for row in rows]
        if any(not evidence_id for evidence_id in evidence_ids):
            raise ValueError(f"{path}: every row must contain evidence_id")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError(f"{path}: duplicate evidence_id values are not supported")
        id_sets[run_id] = set(evidence_ids)
        top_level_keys = set(key_counts)
        shared_top_level_keys = (
            top_level_keys
            if shared_top_level_keys is None
            else shared_top_level_keys & top_level_keys
        )
        attempts = [int(row.get("attempt_count") or 0) for row in rows]
        trace_counts = [len(row.get("generation_trace") or []) for row in rows]
        run_summaries.append(
            {
                "run_id": run_id,
                "run": RUN_LABELS.get(run_id, run_id),
                "source_file": path.name,
                "source_path": str(path),
                "sha256": file_hash,
                "bytes": len(raw),
                "mib": round(len(raw) / 1024 / 1024, 3),
                "source_row_count": len(source_rows),
                "skipped_nonaccepted_row_count": skipped_row_count,
                "accepted_qa_count": len(rows),
                "unique_evidence_count": len(set(evidence_ids)),
                "bytes_per_qa": round(len(raw) / len(rows)) if rows else 0,
                "attempt_total": sum(attempts),
                "attempt_distribution": dict(
                    sorted(collections.Counter(attempts).items())
                ),
                "generation_trace_entries": sum(trace_counts),
                "generation_trace_bytes": field_bytes["generation_trace"],
                "review_bytes": field_bytes["review"],
                "human_audit_bytes": field_bytes["human_audit"],
                "video_evidence_bytes": field_bytes["video_evidence"],
                "cluster_decision_occurrences": sum(
                    _count_nested_key(row, "cluster_decisions") for row in rows
                ),
                "newline": "CRLF" if b"\r\n" in raw else "LF",
                "utf8_bom": raw.startswith(b"\xef\xbb\xbf"),
                "top_level_keys": sorted(top_level_keys),
                "largest_fields_bytes": [
                    {"field": key, "bytes": value}
                    for key, value in field_bytes.most_common(10)
                ],
            }
        )

        for row in rows:
            evidence_id = str(row["evidence_id"])
            videos = _source_videos(row)
            if six_user:
                videos = _six_user_videos(videos, evidence_id=evidence_id)
            entry = evidence_map.setdefault(
                evidence_id,
                {
                    "evidence_id": evidence_id,
                    "videos": videos,
                    "qas": [],
                    "warnings": [],
                },
            )
            existing_urls = tuple(video.get("url") for video in entry["videos"])
            new_urls = tuple(video.get("url") for video in videos)
            if existing_urls and new_urls and existing_urls != new_urls:
                warning = (
                    f"{RUN_LABELS.get(run_id, run_id)} supplied different source-video "
                    "URLs for the same evidence ID."
                )
                if warning not in entry["warnings"]:
                    entry["warnings"].append(warning)
            if not entry["videos"] and videos:
                entry["videos"] = videos
            entry["qas"].append(
                _normalize_qa(
                    row,
                    run_id=run_id,
                    run_label=RUN_LABELS.get(run_id, run_id),
                    source_file=path.name,
                )
            )

    evidence = sorted(evidence_map.values(), key=_evidence_sort_key)
    review_rows: list[dict[str, Any]] = []
    for evidence_index, entry in enumerate(evidence, start=1):
        entry["evidence_order"] = evidence_index
        entry["qas"].sort(key=lambda qa: run_order.get(qa["run_id"], 999))
        entry["qa_count"] = len(entry["qas"])
        entry["runs"] = [qa["run"] for qa in entry["qas"]]
        videos = entry["videos"]
        for qa in entry["qas"]:
            options = qa["options"]
            review_rows.append(
                {
                    "evidence_order": evidence_index,
                    "evidence_id": entry["evidence_id"],
                    "qa_count_for_evidence": entry["qa_count"],
                    "run": qa["run"],
                    "question_type": qa["question_type"],
                    "question": qa["question"],
                    "option_A": options[0],
                    "option_B": options[1],
                    "option_C": options[2],
                    "option_D": options[3],
                    "option_E": options[4],
                    "model_correct_letter": qa["correct"],
                    "model_answer": qa["answer"],
                    "review_status": "Pending",
                    "reviewer_answer": "",
                    "answer_correct": "Unset",
                    "evidence_grounded": "Unset",
                    "asker_alone_answerable": "Unset",
                    "two_user_needed": "Unset",
                    "wording_formality": "Unset",
                    "error_tags": "",
                    "reviewer_notes": "",
                    "reviewed_at": "",
                    "qa_id": qa["qa_id"],
                    "source_file": qa["source_file"],
                    "required_users": " | ".join(qa["required_users"]),
                    "attempt_count": qa["attempt_count"],
                    "judge_video_source": qa["judge_video_source"],
                    "video_count": len(videos),
                    "video_1_user": videos[0]["user"] if len(videos) > 0 else "",
                    "video_1_url": videos[0]["url"] if len(videos) > 0 else "",
                    "video_2_user": videos[1]["user"] if len(videos) > 1 else "",
                    "video_2_url": videos[1]["url"] if len(videos) > 1 else "",
                    "video_3_user": videos[2]["user"] if len(videos) > 2 else "",
                    "video_3_url": videos[2]["url"] if len(videos) > 2 else "",
                    "video_4_user": videos[3]["user"] if len(videos) > 3 else "",
                    "video_4_url": videos[3]["url"] if len(videos) > 3 else "",
                    "video_5_user": videos[4]["user"] if len(videos) > 4 else "",
                    "video_5_url": videos[4]["url"] if len(videos) > 4 else "",
                    "video_6_user": videos[5]["user"] if len(videos) > 5 else "",
                    "video_6_url": videos[5]["url"] if len(videos) > 5 else "",
                    "generator_rationale": qa["generator_rationale"],
                    "evidence_claims": qa["evidence_claims"],
                    "referred_timestamps": qa["referred_timestamps"],
                    "single_user_answerability": qa["single_user_answerability"],
                    "combined_answerability": qa["combined_answerability"],
                    "why_two_users_needed": qa["why_two_users_needed"],
                    "review_key": qa["review_key"],
                }
            )

    overlap_matrix = {
        left: {
            right: len(id_sets[left] & id_sets[right])
            for right, _ in inputs
        }
        for left, _ in inputs
    }
    qa_count_distribution = dict(
        sorted(
            collections.Counter(entry["qa_count"] for entry in evidence).items()
        )
    )
    fingerprint_lines = list(file_hashes)
    if six_user:
        fingerprint_lines.append("review_mode:six_user")
    dataset_fingerprint = hashlib.sha256(
        "\n".join(fingerprint_lines).encode("utf-8")
    ).hexdigest()[:20]
    analysis = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint,
        "review_mode": "six_user" if six_user else "source_evidence",
        "format_comparison": {
            "all_utf8_without_bom": all(not run["utf8_bom"] for run in run_summaries),
            "all_lf_newlines": all(run["newline"] == "LF" for run in run_summaries),
            "same_top_level_schema": all(
                run["top_level_keys"] == run_summaries[0]["top_level_keys"]
                for run in run_summaries
            ),
            "shared_top_level_keys": sorted(shared_top_level_keys or []),
        },
        "run_summaries": run_summaries,
        "overlap_matrix": overlap_matrix,
        "union_evidence_count": len(evidence),
        "total_accepted_qa_count": len(review_rows),
        "qa_count_per_evidence_distribution": qa_count_distribution,
    }
    data = {
        "generated_at": analysis["generated_at"],
        "dataset_fingerprint": dataset_fingerprint,
        "review_mode": analysis["review_mode"],
        "runs": [
            {
                "run_id": run["run_id"],
                "run": run["run"],
                "source_file": run["source_file"],
                "accepted_qa_count": run["accepted_qa_count"],
                "bytes": run["bytes"],
            }
            for run in run_summaries
        ],
        "summary": {
            "evidence_count": len(evidence),
            "qa_count": len(review_rows),
            "video_count": max((len(entry["videos"]) for entry in evidence), default=0),
            "qa_count_per_evidence_distribution": qa_count_distribution,
        },
        "evidence": evidence,
    }
    return data, analysis, review_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_local_server(path: Path) -> None:
    server = '''"""Serve the manual reviewer from its own directory and open it locally."""

from __future__ import annotations

import functools
import http.server
from pathlib import Path
import sys
import threading
import webbrowser


def main() -> int:
    root = Path(__file__).resolve().parent
    html_path = root / "manual_review.html"
    if not html_path.is_file():
        print(f"Could not find {html_path}")
        return 1

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler,
        directory=str(root),
    )
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/manual_review.html"
        print(f"Serving the EgoLife reviewer from: {root}")
        print(f"Open: {url}")
        print("Keep this window open while reviewing. Press Ctrl+C to stop.")
        if "--no-browser" not in sys.argv:
            threading.Timer(0.35, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\\nReviewer server stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
    path.write_text(server, encoding="utf-8")


def _write_windows_launcher(path: Path) -> None:
    launcher = r"""@echo off
setlocal

where py >nul 2>&1
if %errorlevel%==0 (
  py "%~dp0serve_manual_review.py"
) else (
  where python >nul 2>&1
  if errorlevel 1 (
    echo Python was not found. Install Python and run this launcher again.
    pause
    exit /b 1
  )
  python "%~dp0serve_manual_review.py"
)

echo The reviewer server has stopped.
pause
"""
    path.write_text(launcher, encoding="utf-8", newline="\r\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Accepted-QA or generation-intermediate JSONL input; repeat in the "
            "desired display order. Non-accepted intermediate rows are skipped."
        ),
    )
    parser.add_argument(
        "--six-user",
        action="store_true",
        help=(
            "Show all six EgoLife participant videos. Missing participant URLs are "
            "constructed from the accepted row's exact DAY/time token."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/manual_ablation_review_20260727"),
    )
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("manual_ablation_review_template.html"),
    )
    args = parser.parse_args(argv)

    inputs = _parse_inputs(args.input)
    data, analysis, review_rows = build_review_data(inputs, six_user=args.six_user)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    (args.output_dir / "review_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "source_analysis.json").write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "review_rows.json").write_text(
        json.dumps(review_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(args.output_dir / "qa_review_template.csv", review_rows)

    template = args.template.read_text(encoding="utf-8")
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__REVIEW_DATA__", embedded)
    if html == template:
        raise ValueError(f"{args.template}: missing __REVIEW_DATA__ placeholder")
    (args.output_dir / "manual_review.html").write_text(html, encoding="utf-8")
    _write_local_server(args.output_dir / "serve_manual_review.py")
    _write_windows_launcher(args.output_dir / "open_manual_review.cmd")

    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "evidence_count": data["summary"]["evidence_count"],
                "qa_count": data["summary"]["qa_count"],
                "video_count": data["summary"]["video_count"],
                "review_mode": data["review_mode"],
                "dataset_fingerprint": data["dataset_fingerprint"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
