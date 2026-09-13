"""Build a blinded, evidence-first RLHF packet-ranking package.

The builder consumes one or more reward-candidate collection roots. It skips
only explicitly discarded zero-candidate packets, keeps every complete set of
six candidates, and creates a packet-level assignment for 1-3 F/E/A scoring
and tie-aware aggregate ranking. Automatic judge outcomes remain hidden.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


# Retained only so older launch commands and Python callers do not break while
# assignments transition from pairwise comparison to six-question ranking.
DEFAULT_COMPARISONS_PER_PACKET = 6
LABEL_SCHEMA_VERSION = "egolife_rlhf_packet_ranking_v4"

EXPORT_COLUMNS = [
    "schema_version",
    "dataset_fingerprint",
    "assignment_id",
    "reviewer_id",
    "annotation_status",
    "packet_order",
    "evidence_id",
    "display_order",
    "candidate_id",
    "formality_score",
    "evidence_grounding_score",
    "answerability_score",
    "fea_total_score",
    "aggregate_rank",
    "packet_skipped",
    "skip_reason",
    "notes",
    "started_at",
    "completed_at",
    "active_seconds",
    "question",
    "options",
    "correct",
    "answer",
    "video_1_user",
    "video_1_source",
    "video_2_user",
    "video_2_source",
]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_hex(*parts: Any, length: int = 20) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected one JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def _collection_groups(root: Path) -> list[Path]:
    root = root.resolve()
    if (root / "candidate_details.jsonl").is_file():
        return [root]
    index_path = root / "packet_groups.jsonl"
    if index_path.is_file():
        groups = []
        for row in _read_jsonl(index_path):
            relative = row.get("relative_dir")
            if not relative:
                raise ValueError(f"{index_path}: group row is missing relative_dir")
            groups.append((root / str(relative)).resolve())
        return groups
    groups = sorted((root / "packet_groups").glob("packets_*_*"))
    if groups:
        return [path.resolve() for path in groups]
    raise FileNotFoundError(f"no reward-candidate packet groups found beneath {root}")


def _source_videos(packet: dict[str, Any]) -> list[dict[str, Any]]:
    clips = [row for row in packet.get("clips") or [] if isinstance(row, dict)]
    source_urls = packet.get("source_urls")
    raw_urls = source_urls.get("videos") if isinstance(source_urls, dict) else []
    urls = list(raw_urls) if isinstance(raw_urls, list) else []
    videos: list[dict[str, Any]] = []
    for index, clip in enumerate(clips):
        source_url = str(urls[index] or "") if index < len(urls) else ""
        source_url = source_url or str(clip.get("video_url") or "")
        local_path = str(
            clip.get("full_local_video")
            or clip.get("original_local_video")
            or clip.get("local_video")
            or ""
        )
        media_src = source_url
        if not media_src and local_path:
            try:
                media_src = Path(local_path).resolve().as_uri()
            except ValueError:
                media_src = local_path
        videos.append(
            {
                "user": str(
                    clip.get("agent_name")
                    or clip.get("user")
                    or f"View {index + 1}"
                ),
                "agent_dir": str(clip.get("agent_dir") or ""),
                "day": str(clip.get("day") or packet.get("day") or ""),
                "clip_clock": str(
                    clip.get("clip_clock") or packet.get("clip_clock") or ""
                ),
                "source_url": source_url,
                "local_path": local_path,
                "media_src": media_src,
            }
        )
    return videos


def _normalize_candidate(row: dict[str, Any]) -> dict[str, Any]:
    qa = row.get("raw_qa")
    if not isinstance(qa, dict):
        qa = row.get("normalized_qa")
    if not isinstance(qa, dict):
        raise ValueError(f"{row.get('candidate_id')}: candidate has no QA object")
    options = [str(value) for value in qa.get("options") or []]
    if len(options) != 5:
        raise ValueError(f"{row.get('candidate_id')}: expected five answer options")
    candidate_id = str(row.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("candidate row is missing candidate_id")
    correct = str(qa.get("correct") or "")
    answer = str(qa.get("answer") or "")
    return {
        "candidate_id": candidate_id,
        "candidate_index": int(row.get("candidate_index") or 0),
        "question": str(qa.get("question") or ""),
        "options": options,
        "correct": correct,
        "answer": answer,
        "question_type": str(qa.get("question_type") or "neutral"),
        "required_users": [str(value) for value in qa.get("required_users") or []],
        "required_users": [str(value) for value in qa.get("required_users") or []],
    }


def _packet_sort_key(packet: dict[str, Any]) -> tuple[Any, ...]:
    day_text = str(packet.get("day") or "")
    match = re.search(r"(\d+)", day_text)
    day = int(match.group(1)) if match else 999
    digits = re.sub(r"\D", "", str(packet.get("time_token") or ""))
    time_token = int(digits) if digits else 999999999
    return day, time_token, str(packet.get("evidence_id") or "")


def build_labeling_data(
    collection_dirs: Iterable[Path],
    *,
    assignment_id: str,
    comparisons_per_packet: int = DEFAULT_COMPARISONS_PER_PACKET,
    assignment_seed: int = 20260805,
    packet_start: int = 1,
    packet_count: int | None = None,
) -> dict[str, Any]:
    if comparisons_per_packet <= 0:
        raise ValueError("comparisons_per_packet must be positive")
    if packet_start <= 0:
        raise ValueError("packet_start is one-based and must be positive")

    packet_map: dict[str, dict[str, Any]] = {}
    candidate_map: dict[str, list[dict[str, Any]]] = {}
    discarded_map: dict[str, dict[str, Any]] = {}
    source_files: list[dict[str, Any]] = []
    fingerprint_parts: list[str] = []

    for collection_dir in collection_dirs:
        root = collection_dir.resolve()
        for group in _collection_groups(root):
            evidence_path = group / "evidence_manifest.jsonl"
            details_path = group / "candidate_details.jsonl"
            discarded_path = group / "discarded_packets.jsonl"
            if not evidence_path.is_file() or not details_path.is_file():
                raise FileNotFoundError(
                    f"{group}: expected evidence_manifest.jsonl and candidate_details.jsonl"
                )
            evidence_bytes = evidence_path.read_bytes()
            details_bytes = details_path.read_bytes()
            fingerprint_parts.extend(
                [
                    hashlib.sha256(evidence_bytes).hexdigest(),
                    hashlib.sha256(details_bytes).hexdigest(),
                    hashlib.sha256(
                        discarded_path.read_bytes() if discarded_path.is_file() else b""
                    ).hexdigest(),
                ]
            )
            source_files.append(
                {
                    "collection_dir": str(root),
                    "group_dir": str(group),
                    "evidence_manifest": str(evidence_path),
                    "candidate_details": str(details_path),
                    "discarded_packets": (
                        str(discarded_path) if discarded_path.is_file() else None
                    ),
                }
            )
            for packet in _read_jsonl(evidence_path):
                evidence_id = str(packet.get("evidence_id") or "")
                if not evidence_id:
                    raise ValueError(f"{evidence_path}: packet is missing evidence_id")
                if evidence_id in packet_map:
                    if packet_map[evidence_id] != packet:
                        raise ValueError(
                            f"conflicting evidence packet across collections: {evidence_id}"
                        )
                    continue
                packet_map[evidence_id] = packet
            for row in _read_jsonl(details_path):
                evidence_id = str(row.get("evidence_id") or "")
                if not evidence_id:
                    raise ValueError(f"{details_path}: candidate is missing evidence_id")
                candidate_map.setdefault(evidence_id, []).append(_normalize_candidate(row))

            if discarded_path.is_file():
                for row in _read_jsonl(discarded_path):
                    evidence_id = str(row.get("evidence_id") or "")
                    if not evidence_id:
                        raise ValueError(
                            f"{discarded_path}: discarded packet is missing evidence_id"
                        )
                    summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
                    compact = {
                        "evidence_id": evidence_id,
                        "failure_reason": str(
                            row.get("failure_reason")
                            or summary.get("failure_reason")
                            or "generation candidate quota not reached"
                        ),
                        "valid_candidate_count": int(
                            summary.get("valid_candidate_count") or 0
                        ),
                        "candidate_quota": int(summary.get("candidate_quota") or 6),
                        "raw_generation_count": int(
                            summary.get("raw_generation_count") or 0
                        ),
                        "malformed_output_count": int(
                            summary.get("malformed_output_count") or 0
                        ),
                        "collection_dir": str(root),
                        "group_dir": str(group),
                    }
                    existing = discarded_map.get(evidence_id)
                    if existing and existing != compact:
                        raise ValueError(
                            f"conflicting discarded-packet records: {evidence_id}"
                        )
                    discarded_map[evidence_id] = compact

    unknown_candidate_ids = set(candidate_map) - set(packet_map)
    if unknown_candidate_ids:
        raise ValueError(
            "candidate details reference unknown evidence IDs: "
            + ", ".join(sorted(unknown_candidate_ids))
        )
    unknown_discarded_ids = set(discarded_map) - set(packet_map)
    if unknown_discarded_ids:
        raise ValueError(
            "discarded records reference unknown evidence IDs: "
            + ", ".join(sorted(unknown_discarded_ids))
        )

    all_packets = sorted(packet_map.values(), key=_packet_sort_key)
    labelable_packets: list[dict[str, Any]] = []
    excluded_packets: list[dict[str, Any]] = []
    recovered_discarded_count = 0
    for packet in all_packets:
        evidence_id = str(packet["evidence_id"])
        candidate_count_for_packet = len(candidate_map.get(evidence_id, []))
        if candidate_count_for_packet == 6:
            labelable_packets.append(packet)
            if evidence_id in discarded_map:
                recovered_discarded_count += 1
            continue
        if candidate_count_for_packet == 0 and evidence_id in discarded_map:
            excluded_packets.append(
                {
                    **discarded_map[evidence_id],
                    "day": str(packet.get("day") or ""),
                    "time_token": str(packet.get("time_token") or ""),
                    "clip_clock": str(packet.get("clip_clock") or ""),
                    "exclusion_reason": "source packet has no complete six-candidate set",
                }
            )
            continue
        raise ValueError(
            f"{evidence_id}: expected six candidates, or zero candidates with an explicit "
            f"discarded_packets.jsonl record; found {candidate_count_for_packet}"
        )

    start_index = packet_start - 1
    stop_index = None if packet_count is None else start_index + packet_count
    packets = labelable_packets[start_index:stop_index]
    if not packets:
        raise ValueError("the requested assignment contains no labelable evidence packets")

    packet_entries: list[dict[str, Any]] = []
    for packet_order, packet in enumerate(packets, 1):
        evidence_id = str(packet["evidence_id"])
        candidates = sorted(
            candidate_map.get(evidence_id, []),
            key=lambda row: (row["candidate_index"], row["candidate_id"]),
        )
        candidates = sorted(
            candidates,
            key=lambda row: _stable_hex(
                assignment_seed,
                evidence_id,
                row["candidate_id"],
                length=40,
            ),
        )
        display_candidates = []
        for display_order, candidate in enumerate(candidates, 1):
            # Keep generation order and lineage out of the annotator payload.
            display_candidates.append(
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "candidate_index"
                }
                | {
                    "display_order": display_order,
                    "display_label": f"Question {display_order}",
                }
            )
        packet_entries.append(
            {
                "packet_order": packet_order,
                "evidence_id": evidence_id,
                "day": str(packet.get("day") or ""),
                "time_token": str(packet.get("time_token") or ""),
                "clip_clock": str(packet.get("clip_clock") or ""),
                "required_users": [str(value) for value in packet.get("required_users") or []],
                "videos": _source_videos(packet),
                "candidate_count": len(candidates),
                "candidates": display_candidates,
            }
        )

    dataset_fingerprint = hashlib.sha256(
        "\n".join(sorted(fingerprint_parts)).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": LABEL_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_fingerprint": dataset_fingerprint,
        "assignment": {
            "assignment_id": assignment_id,
            "assignment_seed": assignment_seed,
            "packet_start": packet_start,
            "packet_count": len(packet_entries),
            "requested_packet_count": packet_count,
            "packet_start_semantics": (
                "one-based over complete six-candidate packets after explicitly discarded "
                "zero-candidate packets are excluded"
            ),
            "scoring": "each candidate receives F/E/A scores from 1 to 3",
            "ranking": "rank 1 is best; ranks 1-6 may repeat to express ties",
            "blinding": (
                "candidate display order is deterministically randomized; automatic judge "
                "outcomes and generation lineage are not shown in the annotation interface"
            ),
        },
        "summary": {
            "packet_count": len(packet_entries),
            "candidate_count": len(packet_entries) * 6,
            "source_packet_count": len(all_packets),
            "labelable_packet_count": len(labelable_packets),
            "source_discarded_packet_count": len(discarded_map),
            "recovered_discarded_packet_count": recovered_discarded_count,
            "excluded_packet_count": len(excluded_packets),
            "maximum_derived_pair_count": len(packet_entries) * 15,
        },
        "packets": packet_entries,
        "excluded_packets": excluded_packets,
        "sources": source_files,
    }


def _write_csv_template(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPORT_COLUMNS)
        writer.writeheader()
        for packet in data["packets"]:
            for candidate in packet["candidates"]:
                writer.writerow(
                    {
                        "schema_version": data["schema_version"],
                        "dataset_fingerprint": data["dataset_fingerprint"],
                        "assignment_id": data["assignment"]["assignment_id"],
                        "packet_order": packet["packet_order"],
                        "evidence_id": packet["evidence_id"],
                        "display_order": candidate["display_order"],
                        "candidate_id": candidate["candidate_id"],
                        "question": candidate["question"],
                        "options": _canonical_json(candidate["options"]),
                        "correct": candidate["correct"],
                        "answer": candidate["answer"],
                        "video_1_user": packet["videos"][0]["user"] if packet["videos"] else "",
                        "video_1_source": (
                            packet["videos"][0]["source_url"]
                            or packet["videos"][0]["local_path"]
                            if packet["videos"]
                            else ""
                        ),
                        "video_2_user": packet["videos"][1]["user"] if len(packet["videos"]) > 1 else "",
                        "video_2_source": (
                            packet["videos"][1]["source_url"]
                            or packet["videos"][1]["local_path"]
                            if len(packet["videos"]) > 1
                            else ""
                        ),
                    }
                )


def _write_local_server(path: Path) -> None:
    server = '''"""Serve the RLHF labeling package locally."""

from __future__ import annotations

import functools
import http.server
from pathlib import Path
import sys
import threading
import webbrowser


def main() -> int:
    root = Path(__file__).resolve().parent
    html_path = root / "rlhf_labeling.html"
    if not html_path.is_file():
        print(f"Could not find {html_path}")
        return 1
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler) as server:
        port = server.server_address[1]
        url = f"http://127.0.0.1:{port}/rlhf_labeling.html"
        print(f"Serving the EgoLife RLHF labeler from: {root}")
        print(f"Open: {url}")
        print("Keep this window open while labeling. Press Ctrl+C to stop.")
        if "--no-browser" not in sys.argv:
            threading.Timer(0.35, lambda: webbrowser.open(url)).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\\nLabeling server stopped.")
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
  py "%~dp0serve_rlhf_labeling.py"
) else (
  python "%~dp0serve_rlhf_labeling.py"
)
pause
"""
    path.write_text(launcher, encoding="utf-8", newline="\r\n")


def write_labeling_package(
    data: dict[str, Any],
    *,
    output_dir: Path,
    template_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labeling_data.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "assignment_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": data["schema_version"],
                "generated_at": data["generated_at"],
                "dataset_fingerprint": data["dataset_fingerprint"],
                "assignment": data["assignment"],
                "summary": data["summary"],
                "excluded_packets": data["excluded_packets"],
                "sources": data["sources"],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv_template(output_dir / "human_labels_template.csv", data)
    template = template_path.read_text(encoding="utf-8")
    embedded = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = template.replace("__LABELING_DATA__", embedded)
    if html == template:
        raise ValueError(f"{template_path}: missing __LABELING_DATA__ placeholder")
    (output_dir / "rlhf_labeling.html").write_text(html, encoding="utf-8")
    _write_local_server(output_dir / "serve_rlhf_labeling.py")
    _write_windows_launcher(output_dir / "open_rlhf_labeling.cmd")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection-dir",
        action="append",
        required=True,
        type=Path,
        help="Completed reward-candidate collection root; repeat to combine time halves",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--assignment-id", required=True)
    parser.add_argument(
        "--comparisons-per-packet",
        type=int,
        default=DEFAULT_COMPARISONS_PER_PACKET,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--assignment-seed", type=int, default=20260805)
    parser.add_argument("--packet-start", type=int, default=1)
    parser.add_argument("--packet-count", type=int)
    parser.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).with_name("rlhf_human_labeling_template.html"),
    )
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.assignment_id):
        raise ValueError("assignment_id may contain only letters, numbers, dot, dash, underscore")
    data = build_labeling_data(
        args.collection_dir,
        assignment_id=args.assignment_id,
        comparisons_per_packet=args.comparisons_per_packet,
        assignment_seed=args.assignment_seed,
        packet_start=args.packet_start,
        packet_count=args.packet_count,
    )
    write_labeling_package(
        data,
        output_dir=args.output_dir,
        template_path=args.template,
    )
    print(
        _canonical_json(
            {
                "output_dir": str(args.output_dir.resolve()),
                "dataset_fingerprint": data["dataset_fingerprint"],
                **data["summary"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
