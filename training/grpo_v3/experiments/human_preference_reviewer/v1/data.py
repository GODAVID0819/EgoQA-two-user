"""Strict CSV and evidence-split contracts for Reviewer v1."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCORE_COLUMNS = {
    "evidence_grounding_score": "evidence_quality",
    "answerability_score": "answerability",
    "formality_score": "qa_formality",
}
INTERNAL_FIELDS = ("evidence_quality", "answerability", "qa_formality")
GRADE_TO_TARGET = {1: 0, 2: 1, 3: 2}
CONTRACT_VERSION = "human_preference_reviewer_absolute_v1"


def _text(value: object) -> str:
    return str(value or "").strip()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    evidence_id: str
    display_order: int
    question: str
    options: tuple[str, ...]
    correct: str
    answer: str
    evidence_quality: int | None
    answerability: int | None
    qa_formality: int | None
    overall_rank: int | None

    def model_features(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "options": list(self.options),
            "correct": self.correct,
            "answer": self.answer,
        }

    def labels(self) -> dict[str, int]:
        values = {
            "evidence_quality": self.evidence_quality,
            "answerability": self.answerability,
            "qa_formality": self.qa_formality,
        }
        if any(value not in GRADE_TO_TARGET for value in values.values()):
            raise ValueError(f"candidate {self.candidate_id} does not have three valid grades")
        return {name: int(value) for name, value in values.items()}


@dataclass(frozen=True)
class EvidenceRecord:
    evidence_id: str
    annotation_status: str
    video_a_user: str
    video_a_source: str
    video_b_user: str
    video_b_source: str
    candidates: tuple[CandidateRecord, ...]


@dataclass(frozen=True)
class AnnotationAudit:
    csv_path: str
    csv_sha256: str
    row_count: int
    evidence_count: int
    eligible_evidence: tuple[EvidenceRecord, ...]
    quarantined_scored_evidence_ids: tuple[str, ...]
    unscored_evidence_ids: tuple[str, ...]
    label_distribution: dict[str, dict[int, int]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": CONTRACT_VERSION,
            "csv_path": self.csv_path,
            "csv_sha256": self.csv_sha256,
            "row_count": self.row_count,
            "evidence_count": self.evidence_count,
            "eligible_evidence_count": len(self.eligible_evidence),
            "eligible_candidate_count": sum(len(row.candidates) for row in self.eligible_evidence),
            "quarantined_scored_evidence_ids": list(self.quarantined_scored_evidence_ids),
            "unscored_evidence_ids": list(self.unscored_evidence_ids),
            "label_distribution": {
                field: {str(grade): count for grade, count in sorted(values.items())}
                for field, values in self.label_distribution.items()
            },
        }


def _parse_grade(row: Mapping[str, str], column: str) -> int | None:
    raw = _text(row.get(column))
    if not raw:
        return None
    try:
        grade = int(raw)
    except ValueError as error:
        raise ValueError(f"invalid {column} grade {raw!r}") from error
    if grade not in GRADE_TO_TARGET:
        raise ValueError(f"invalid {column} grade {grade}; expected 1, 2, or 3")
    return grade


def _parse_candidate(row: Mapping[str, str]) -> CandidateRecord:
    evidence_id = _text(row.get("evidence_id"))
    candidate_id = _text(row.get("candidate_id"))
    if not evidence_id or not candidate_id:
        raise ValueError("evidence_id and candidate_id are required")
    if candidate_id.split("::", 1)[0] != evidence_id:
        raise ValueError(f"candidate {candidate_id} does not belong to {evidence_id}")
    try:
        display_order = int(_text(row.get("display_order")))
    except ValueError as error:
        raise ValueError(f"candidate {candidate_id} has invalid display_order") from error
    try:
        options_value = json.loads(_text(row.get("options")))
    except json.JSONDecodeError as error:
        raise ValueError(f"candidate {candidate_id} has invalid options JSON") from error
    if not isinstance(options_value, list) or len(options_value) != 5 or not all(_text(x) for x in options_value):
        raise ValueError(f"candidate {candidate_id} must contain exactly five non-empty options")
    options = tuple(_text(value) for value in options_value)
    correct = _text(row.get("correct")).upper()
    if correct not in tuple("ABCDE"):
        raise ValueError(f"candidate {candidate_id} has invalid correct letter")
    answer = _text(row.get("answer"))
    if answer != options[ord(correct) - ord("A")]:
        raise ValueError(f"candidate {candidate_id} answer does not match correct option")
    grades = {internal: _parse_grade(row, source) for source, internal in SCORE_COLUMNS.items()}
    populated = [value is not None for value in grades.values()]
    if any(populated) and not all(populated):
        raise ValueError(f"candidate {candidate_id} has partially populated grades")
    total_raw = _text(row.get("fea_total_score"))
    if all(populated):
        if total_raw and int(total_raw) != sum(int(value) for value in grades.values()):
            raise ValueError(f"candidate {candidate_id} has inconsistent fea_total_score")
    elif total_raw:
        raise ValueError(f"candidate {candidate_id} has fea_total_score without grades")
    rank_raw = _text(row.get("aggregate_rank"))
    overall_rank = int(rank_raw) if rank_raw else None
    if overall_rank is not None and not 1 <= overall_rank <= 6:
        raise ValueError(f"candidate {candidate_id} has invalid aggregate_rank")
    question = _text(row.get("question"))
    if not question:
        raise ValueError(f"candidate {candidate_id} has empty question")
    return CandidateRecord(
        candidate_id=candidate_id,
        evidence_id=evidence_id,
        display_order=display_order,
        question=question,
        options=options,
        correct=correct,
        answer=answer,
        evidence_quality=grades["evidence_quality"],
        answerability=grades["answerability"],
        qa_formality=grades["qa_formality"],
        overall_rank=overall_rank,
    )


def _parse_evidence(evidence_id: str, rows: Sequence[Mapping[str, str]]) -> EvidenceRecord:
    if len(rows) != 6:
        raise ValueError(f"evidence {evidence_id} must contain exactly 6 candidates; found {len(rows)}")
    candidates = tuple(sorted((_parse_candidate(row) for row in rows), key=lambda item: item.display_order))
    if [item.display_order for item in candidates] != list(range(1, 7)):
        raise ValueError(f"evidence {evidence_id} display_order must be exactly 1..6")
    if len({item.candidate_id for item in candidates}) != 6:
        raise ValueError(f"evidence {evidence_id} has duplicate candidate IDs")
    statuses = {_text(row.get("annotation_status")).lower() for row in rows}
    if len(statuses) != 1 or next(iter(statuses)) not in {"completed", "pending"}:
        raise ValueError(f"evidence {evidence_id} has inconsistent annotation_status")
    media = {
        (
            _text(row.get("video_1_user")), _text(row.get("video_1_source")),
            _text(row.get("video_2_user")), _text(row.get("video_2_source")),
        )
        for row in rows
    }
    if len(media) != 1 or not all(next(iter(media))):
        raise ValueError(f"evidence {evidence_id} must have one complete ordered video pair")
    video_a_user, video_a_source, video_b_user, video_b_source = next(iter(media))
    return EvidenceRecord(
        evidence_id=evidence_id,
        annotation_status=next(iter(statuses)),
        video_a_user=video_a_user,
        video_a_source=video_a_source,
        video_b_user=video_b_user,
        video_b_source=video_b_source,
        candidates=candidates,
    )


def load_annotation_csv(path: str | Path) -> AnnotationAudit:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[_text(row.get("evidence_id"))].append(row)
    if "" in grouped:
        raise ValueError("every CSV row must have evidence_id")
    evidence = tuple(_parse_evidence(evidence_id, grouped[evidence_id]) for evidence_id in sorted(grouped))
    eligible: list[EvidenceRecord] = []
    quarantined: list[str] = []
    unscored: list[str] = []
    for item in evidence:
        scored = all(candidate.evidence_quality is not None for candidate in item.candidates)
        if item.annotation_status == "completed" and scored:
            eligible.append(item)
        elif scored:
            quarantined.append(item.evidence_id)
        else:
            unscored.append(item.evidence_id)
    distribution = {field: Counter() for field in INTERNAL_FIELDS}
    for item in eligible:
        for candidate in item.candidates:
            for field, grade in candidate.labels().items():
                distribution[field][grade] += 1
    return AnnotationAudit(
        csv_path=str(csv_path.resolve()),
        csv_sha256=sha256_file(csv_path),
        row_count=len(rows),
        evidence_count=len(evidence),
        eligible_evidence=tuple(eligible),
        quarantined_scored_evidence_ids=tuple(quarantined),
        unscored_evidence_ids=tuple(unscored),
        label_distribution={field: dict(sorted(values.items())) for field, values in distribution.items()},
    )


def _support(records: Iterable[EvidenceRecord]) -> dict[str, dict[str, int]]:
    counters = {field: Counter({1: 0, 2: 0, 3: 0}) for field in INTERNAL_FIELDS}
    for evidence in records:
        for candidate in evidence.candidates:
            for field, grade in candidate.labels().items():
                counters[field][grade] += 1
    return {
        field: {str(grade): count for grade, count in sorted(counter.items())}
        for field, counter in counters.items()
    }


def validate_split_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_counts: tuple[int, int, int] | None = None,
    require_contract: bool = False,
) -> None:
    """Reject duplicate or overlapping evidence IDs before any split is consumed."""
    if require_contract:
        if manifest.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("split manifest contract_version mismatch")
        if manifest.get("split_unit") != "evidence_id":
            raise ValueError("split manifest split_unit must be evidence_id")
        if not str(manifest.get("csv_sha256") or "").strip():
            raise ValueError("split manifest csv_sha256 is required")
    names = ("train", "validation", "locked_test", "reserve")
    split_sets: dict[str, set[str]] = {}
    for name in names:
        key = f"{name}_evidence_ids"
        raw = manifest.get(key, [])
        if not isinstance(raw, list):
            raise ValueError(f"{key} must be a list")
        ids = [str(value) for value in raw]
        if name in {"train", "validation"} and not ids:
            raise ValueError(f"{key} must not be empty")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate evidence_id within {key}")
        split_sets[name] = set(ids)
    if expected_counts is not None:
        for name, expected in zip(names[:3], expected_counts):
            actual = len(split_sets[name])
            if actual != expected:
                raise ValueError(f"{name}_evidence_ids expected {expected}, found {actual}")
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            overlap = sorted(split_sets[left] & split_sets[right])
            if overlap:
                raise ValueError(f"evidence_id overlap between {left} and {right}: {overlap}")


def build_split_manifest(
    records: Sequence[EvidenceRecord],
    *,
    train_count: int,
    validation_count: int,
    locked_test_count: int,
    seed: int = 42,
    csv_sha256: str | None = None,
    require_full_class_support: bool = True,
) -> dict[str, Any]:
    required = train_count + validation_count + locked_test_count
    if any(not isinstance(value, int) or value <= 0 for value in (train_count, validation_count)):
        raise ValueError("train_count and validation_count must be positive integers")
    if not isinstance(locked_test_count, int) or locked_test_count < 0:
        raise ValueError("locked_test_count must be a non-negative integer")
    if len(records) < required:
        raise ValueError(f"need at least {required} eligible evidence IDs; found {len(records)}")
    by_id = {record.evidence_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("duplicate evidence_id in split input")
    maximum_attempts = 10_000 if require_full_class_support else 1
    missing_support: list[str] = []
    for selection_attempt in range(maximum_attempts):
        evidence_ids = sorted(by_id)
        random.Random(seed + selection_attempt).shuffle(evidence_ids)
        selected_ids = evidence_ids[:required]
        reserve_ids = sorted(evidence_ids[required:])
        split_ids = {
            "train": sorted(selected_ids[:train_count]),
            "validation": sorted(selected_ids[train_count:train_count + validation_count]),
            "locked_test": sorted(selected_ids[train_count + validation_count:required]),
        }
        label_support = {
            name: _support(by_id[evidence_id] for evidence_id in ids)
            for name, ids in split_ids.items() if ids
        }
        missing_support = [
            f"{split}:{field}:{grade}"
            for split, fields in label_support.items()
            for field, grades in fields.items()
            for grade, count in grades.items()
            if count == 0
        ]
        if not require_full_class_support or not missing_support:
            break
    else:
        raise ValueError(
            f"could not find a split with full class support after {maximum_attempts} attempts; "
            f"last missing support: {missing_support}"
        )
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "split_unit": "evidence_id",
        "seed": seed,
        "selection_attempt": selection_attempt,
        "csv_sha256": csv_sha256,
        "reserve_evidence_ids": reserve_ids,
        **{f"{name}_evidence_ids": values for name, values in split_ids.items()},
        "label_support": label_support,
        "full_class_support_required": require_full_class_support,
    }
    validate_split_manifest(
        manifest,
        expected_counts=(train_count, validation_count, locked_test_count),
    )
    return manifest
