"""Qwen minimum-set 与 all-six 成对视频 QA 评审核心逻辑。"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

SIX_USERS = ("Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure")
CHOICES = ("A", "B", "C", "D", "E")
SOURCE_PRIORITY = {"approved_markdown": 1, "curated_trace_v3": 2}


@dataclass(frozen=True)
class GoldItem:
    qa_id: str
    source: str
    source_item_id: str
    evidence_id: str
    generation_group_id: str
    question: str
    options: tuple[str, ...]
    correct: str
    answer: str
    minimum_required_users: tuple[str, ...]
    review_status: str


@dataclass(frozen=True)
class DeduplicationResult:
    items: tuple[GoldItem, ...]
    removed: tuple[dict[str, str], ...]
    same_group_nonduplicates: tuple[dict[str, str], ...]


def normalize_text(value: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        unicodedata.normalize("NFKC", value).strip(),
    ).casefold()


def validate_gold_item(item: GoldItem) -> GoldItem:
    if not item.qa_id or not item.question or not item.evidence_id:
        raise ValueError("qa_id, evidence_id, and question are required")
    if not re.fullmatch(r"DAY\d+::\d+", item.generation_group_id):
        raise ValueError("generation_group_id must match DAY<number>::<time>")
    if len(item.options) != 5 or any(not value.strip() for value in item.options):
        raise ValueError("options must contain exactly five non-empty values")
    if item.correct not in CHOICES:
        raise ValueError("correct must be A-E")
    if not item.minimum_required_users:
        raise ValueError("minimum_required_users must be non-empty")
    if len(set(item.minimum_required_users)) != len(item.minimum_required_users):
        raise ValueError("minimum_required_users must be unique")
    unknown = [user for user in item.minimum_required_users if user not in SIX_USERS]
    if unknown:
        raise ValueError(f"unknown user: {unknown[0]}")
    if item.source == "curated_trace_v3" and item.review_status != "pass":
        raise ValueError("curated_trace_v3 requires review_status=pass")
    expected = item.options[CHOICES.index(item.correct)]
    if normalize_text(item.answer).rstrip(".") != normalize_text(expected).rstrip("."):
        raise ValueError("answer must match the option selected by correct")
    return item


def load_curated_jsonl(path: str | Path) -> list[GoldItem]:
    rows: list[GoldItem] = []
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        value = json.loads(line)
        item = GoldItem(
            qa_id=str(value["qa_id"]),
            source="curated_trace_v3",
            source_item_id=str(value.get("original_item_number", line_number)),
            evidence_id=str(value["evidence_id"]),
            generation_group_id=str(value["generation_group"]),
            question=str(value["question"]),
            options=tuple(str(option) for option in value["options"]),
            correct=str(value["correct"]).upper(),
            answer=str(value["answer"]),
            minimum_required_users=tuple(
                str(user) for user in value["required_users"]
            ),
            review_status=str(value["review_status"]),
        )
        rows.append(validate_gold_item(item))
    return rows


def load_approved_markdown(path: str | Path) -> list[GoldItem]:
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"(?m)^#\s+(?=\d+\s*$)", text)[1:]
    items: list[GoldItem] = []
    for block in blocks:
        lines = block.splitlines()
        item_number = int(lines[0].strip())
        body = "\n".join(lines[1:])
        group = re.search(
            r"generation group[：:]\s*(\S+)", body, re.IGNORECASE
        )
        evidence = re.search(r"evidence id[：:]\s*(\S+)", body, re.IGNORECASE)
        users = re.search(
            r"minimum required users[：:]\s*([^\n]+)", body, re.IGNORECASE
        )
        if group is None or evidence is None or users is None:
            raise ValueError(f"Markdown item {item_number} is missing metadata")
        option_matches = list(
            re.finditer(
                r"(?m)^([A-E])\.\s+(.+?)(?:\s+←\s*声明答案)?\s*$",
                body,
            )
        )
        if len(option_matches) != 5:
            raise ValueError(f"Markdown item {item_number} must contain five options")
        correct_matches = re.findall(
            r"(?m)^([A-E])\.\s+.+?\s+←\s*声明答案\s*$",
            body,
        )
        if len(correct_matches) != 1:
            raise ValueError(f"Markdown item {item_number} must declare one answer")
        question_start = users.end()
        question_end = option_matches[0].start()
        question = body[question_start:question_end].strip()
        options = tuple(match.group(2).strip() for match in option_matches)
        correct = correct_matches[0]
        item = GoldItem(
            qa_id=f"APPROVED_MD_Q{item_number:02d}",
            source="approved_markdown",
            source_item_id=str(item_number),
            evidence_id=evidence.group(1),
            generation_group_id=group.group(1),
            question=question,
            options=options,
            correct=correct,
            answer=options[CHOICES.index(correct)],
            minimum_required_users=tuple(
                user.strip() for user in users.group(1).split(",") if user.strip()
            ),
            review_status="user_approved",
        )
        items.append(validate_gold_item(item))
    return items


def deduplicate_items(items: Iterable[GoldItem]) -> DeduplicationResult:
    chosen: dict[str, GoldItem] = {}
    removed: list[dict[str, str]] = []
    for item in items:
        validate_gold_item(item)
        key = normalize_text(item.question)
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = item
            continue
        keep, drop = sorted(
            (existing, item),
            key=lambda value: SOURCE_PRIORITY[value.source],
            reverse=True,
        )
        chosen[key] = keep
        removed.append(
            {
                "normalized_question": key,
                "kept_qa_id": keep.qa_id,
                "removed_qa_id": drop.qa_id,
                "reason": "normalized_question_duplicate_source_priority",
            }
        )
    ordered = tuple(chosen.values())
    same_group: list[dict[str, str]] = []
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if left.generation_group_id == right.generation_group_id:
                same_group.append(
                    {
                        "left_qa_id": left.qa_id,
                        "right_qa_id": right.qa_id,
                        "generation_group_id": left.generation_group_id,
                        "action": "retained_for_human_review",
                    }
                )
    return DeduplicationResult(ordered, tuple(removed), tuple(same_group))


def item_to_dict(item: GoldItem) -> dict[str, Any]:
    value = asdict(item)
    value["options"] = list(item.options)
    value["minimum_required_users"] = list(item.minimum_required_users)
    return value
