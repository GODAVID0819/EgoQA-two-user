from __future__ import annotations

import json
from pathlib import Path

import pytest

from egolife_two_user_qa.qwen_two_condition_review import (
    GoldItem,
    deduplicate_items,
    load_approved_markdown,
    load_curated_jsonl,
    validate_gold_item,
)


def _item(
    qa_id: str,
    question: str,
    *,
    source: str = "approved_markdown",
    correct: str = "B",
    options: tuple[str, ...] = ("one", "two", "three", "four", "five"),
) -> GoldItem:
    return GoldItem(
        qa_id=qa_id,
        source=source,
        source_item_id=qa_id,
        evidence_id="E1",
        generation_group_id="DAY1::17200000",
        question=question,
        options=options,
        correct=correct,
        answer=options[ord(correct) - ord("A")],
        minimum_required_users=("Jake", "Lucia"),
        review_status="user_approved" if source == "approved_markdown" else "pass",
    )


def test_load_approved_markdown_extracts_gold_contract(tmp_path: Path) -> None:
    path = tmp_path / "QA.md"
    path.write_text(
        """# 1
generation group：DAY1::17200000
evidence id：E1
speaker：Jake
minimum required users: Jake, Lucia

Which bottle was selected?

A. red
B. gold ← 声明答案
C. silver
D. blue
E. green
""",
        encoding="utf-8",
    )
    items = load_approved_markdown(path)
    assert len(items) == 1
    assert items[0].generation_group_id == "DAY1::17200000"
    assert items[0].minimum_required_users == ("Jake", "Lucia")
    assert items[0].options == ("red", "gold", "silver", "blue", "green")
    assert items[0].correct == "B"
    assert items[0].answer == "gold"


def test_load_curated_jsonl_maps_trace_v3_fields(tmp_path: Path) -> None:
    path = tmp_path / "curated.jsonl"
    row = {
        "qa_id": "CURATED_Q01",
        "evidence_id": "E1",
        "generation_group": "DAY3::17000000",
        "original_item_number": 1,
        "question": "What was on the scale?",
        "options": ["cereal", "grapes", "meat", "chips", "water"],
        "correct": "B",
        "answer": "grapes",
        "required_users": ["Jake", "Alice"],
        "review_status": "pass",
        "review_source": "trace-v3",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    items = load_curated_jsonl(path)
    assert len(items) == 1
    assert items[0].generation_group_id == "DAY3::17000000"
    assert items[0].minimum_required_users == ("Jake", "Alice")
    assert items[0].review_status == "pass"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"options": ("one", "two")}, "exactly five"),
        ({"correct": "F"}, "correct must be A-E"),
        ({"minimum_required_users": ()}, "minimum_required_users"),
        ({"minimum_required_users": ("Jake", "Unknown")}, "unknown user"),
        ({"review_status": "fail", "source": "curated_trace_v3"}, "review_status=pass"),
    ],
)
def test_validate_gold_item_rejects_invalid_contract(changes: dict, message: str) -> None:
    base = _item("Q1", "Question?")
    values = {
        "qa_id": base.qa_id,
        "source": base.source,
        "source_item_id": base.source_item_id,
        "evidence_id": base.evidence_id,
        "generation_group_id": base.generation_group_id,
        "question": base.question,
        "options": base.options,
        "correct": base.correct,
        "answer": base.answer,
        "minimum_required_users": base.minimum_required_users,
        "review_status": base.review_status,
    }
    values.update(changes)
    with pytest.raises(ValueError, match=message):
        validate_gold_item(GoldItem(**values))


def test_deduplicate_prefers_complete_curated_record() -> None:
    markdown = _item("M1", "Which bottle was selected?", correct="B")
    curated_options = ("red", "silver", "gold", "blue", "green")
    curated = _item(
        "C1",
        "  WHICH   BOTTLE WAS SELECTED?  ",
        source="curated_trace_v3",
        correct="C",
        options=curated_options,
    )
    result = deduplicate_items([markdown, curated])
    assert [item.qa_id for item in result.items] == ["C1"]
    assert result.items[0].options == curated_options
    assert result.items[0].correct == "C"
    assert result.items[0].answer == "gold"
    assert result.removed[0]["removed_qa_id"] == "M1"
    assert result.removed[0]["kept_qa_id"] == "C1"


def test_current_approved_inputs_deduplicate_to_21() -> None:
    markdown = Path(
        r"C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts"
        r"\six_user_qa_10min_16628910_snapshot_10qa_20260901\QA.md"
    )
    curated = Path(
        r"C:\Users\20661\Documents\xwechat_files\wxid_i096w25uhusk22_e748"
        r"\msg\file\2026-09\qa_curated_17_trace_review_v3.jsonl"
    )
    if not markdown.is_file() or not curated.is_file():
        pytest.skip("current approved input files are local review artifacts")
    result = deduplicate_items(
        [*load_approved_markdown(markdown), *load_curated_jsonl(curated)]
    )
    assert len(result.items) == 21
    assert len(result.removed) == 3
