from __future__ import annotations

import json
from pathlib import Path

import pytest

from egolife_two_user_qa.qwen_two_condition_review import (
    ALL_SIX_CONDITION,
    MINIMUM_SET_CONDITION,
    GoldItem,
    build_condition_specs,
    build_prompt,
    deduplicate_items,
    load_approved_markdown,
    load_curated_jsonl,
    parse_choice,
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


def test_build_conditions_changes_only_video_collection(tmp_path: Path) -> None:
    item = _item("Q1", "Which bottle was selected?")
    group = tmp_path / "DAY1_17200000"
    group.mkdir()
    for user in ("Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"):
        (group / f"{user}.mp4").write_bytes(b"video")
    specs = build_condition_specs(item, tmp_path)
    assert [spec.condition_id for spec in specs] == [
        MINIMUM_SET_CONDITION,
        ALL_SIX_CONDITION,
    ]
    assert specs[0].input_users == ("Jake", "Lucia")
    assert specs[1].input_users == (
        "Jake",
        "Alice",
        "Tasha",
        "Lucia",
        "Katrina",
        "Shure",
    )
    assert len(specs[0].video_paths) == 2
    assert len(specs[1].video_paths) == 6
    assert specs[0].missing_paths == ()
    assert specs[1].missing_paths == ()


def test_missing_media_is_explicit(tmp_path: Path) -> None:
    item = _item("Q1", "Which bottle was selected?")
    specs = build_condition_specs(item, tmp_path)
    assert len(specs[0].missing_paths) == 2
    assert len(specs[1].missing_paths) == 6


def test_prompt_uses_question_and_options_only() -> None:
    item = _item("SECRET_GOLD_ID", "Which bottle was selected?")
    prompt = build_prompt(item.question, item.options)
    assert "SECRET_GOLD_ID" not in prompt
    assert "minimum_required_users" not in prompt
    assert "correct" not in prompt.casefold()
    assert "Which bottle was selected?" in prompt
    assert "B. two" in prompt
    assert prompt.endswith("ANSWER: <brief answer>")


@pytest.mark.parametrize("raw", ["CHOICE: B\nANSWER: two", "B", "B.", "(B)"])
def test_parse_choice_accepts_one_unambiguous_choice(raw: str) -> None:
    parsed = parse_choice(raw)
    assert parsed.choice == "B"
    assert parsed.status == "valid"


@pytest.mark.parametrize(
    "raw, status",
    [
        ("I cannot tell.", "invalid_missing"),
        ("CHOICE: A\nCHOICE: B", "invalid_ambiguous"),
        ("A or B", "invalid_missing"),
    ],
)
def test_parse_choice_rejects_missing_or_conflicting_output(
    raw: str,
    status: str,
) -> None:
    parsed = parse_choice(raw)
    assert parsed.choice is None
    assert parsed.status == status


class FakeRunner:
    model_id = "fake-model"

    def __init__(self, outputs: list[str | Exception]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, **kwargs})
        value = self.outputs.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _complete_media(tmp_path: Path) -> Path:
    group = tmp_path / "media" / "DAY1_17200000"
    group.mkdir(parents=True)
    for user in ("Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure"):
        (group / f"{user}.mp4").write_bytes(b"video")
    return tmp_path / "media"


def test_run_item_uses_identical_prompt_and_only_changes_videos(
    tmp_path: Path,
) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import run_items

    item = _item("SECRET_GOLD_ID", "Which bottle was selected?")
    runner = FakeRunner(
        ["CHOICE: B\nANSWER: two", "CHOICE: A\nANSWER: one"]
    )
    output_dir = tmp_path / "run"
    run_items([item], _complete_media(tmp_path), output_dir, runner)
    assert len(runner.calls) == 2
    assert runner.calls[0]["prompt"] == runner.calls[1]["prompt"]
    assert len(runner.calls[0]["video_paths"]) == 2
    assert len(runner.calls[1]["video_paths"]) == 6
    assert "SECRET_GOLD_ID" not in runner.calls[0]["prompt"]
    assert set(runner.calls[0]) == {
        "prompt",
        "image_paths",
        "video_paths",
        "decoding_mode",
        "call_profile",
    }
    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["condition_id"] for row in rows] == ["minimum_set", "all_six"]
    assert [row["is_correct"] for row in rows] == [True, False]


def test_invalid_parse_is_saved_without_automatic_retry(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import run_items

    runner = FakeRunner(["unclear", "CHOICE: B\nANSWER: two"])
    output_dir = tmp_path / "run"
    run_items(
        [_item("Q1", "Question?")],
        _complete_media(tmp_path),
        output_dir,
        runner,
    )
    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert rows[0]["parse_status"] == "invalid_missing"
    assert rows[0]["is_correct"] is None
    assert rows[0]["attempt"] == 1
    assert len(runner.calls) == 2


def test_resume_skips_valid_rows_and_explicitly_reruns_invalid(
    tmp_path: Path,
) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import run_items

    item = _item("Q1", "Question?")
    media_root = _complete_media(tmp_path)
    output_dir = tmp_path / "run"
    first = FakeRunner(["unclear", "CHOICE: B\nANSWER: two"])
    run_items([item], media_root, output_dir, first)
    second = FakeRunner(["CHOICE: B\nANSWER: two"])
    run_items(
        [item],
        media_root,
        output_dir,
        second,
        rerun_nonvalid=True,
    )
    assert len(second.calls) == 1
    rows = [
        json.loads(line)
        for line in (output_dir / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [
        row["attempt"]
        for row in rows
        if row["condition_id"] == "minimum_set"
    ] == [1, 2]


def test_runner_error_is_saved_and_stops_following_calls(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import (
        ReviewExecutionError,
        run_items,
    )

    runner = FakeRunner([RuntimeError("decoder failed"), "CHOICE: B"])
    output_dir = tmp_path / "run"
    with pytest.raises(ReviewExecutionError, match="decoder failed"):
        run_items(
            [_item("Q1", "Question?")],
            _complete_media(tmp_path),
            output_dir,
            runner,
        )
    assert len(runner.calls) == 1
    row = json.loads(
        (output_dir / "predictions.jsonl").read_text(encoding="utf-8").strip()
    )
    assert row["run_status"] == "error"
    assert row["parse_status"] == "not_parsed"
    assert row["attempt"] == 1


def test_prepare_review_writes_selection_dedup_and_media_reports(
    tmp_path: Path,
) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import prepare_review

    markdown = _item("M1", "Question one?")
    curated = _item("C1", "Question one?", source="curated_trace_v3")
    missing = _item("C2", "Question two?", source="curated_trace_v3")
    output_dir = tmp_path / "run"
    result = prepare_review(
        [markdown, curated, missing],
        tmp_path / "media",
        output_dir,
    )
    assert len(result.items) == 2
    selection = [
        json.loads(line)
        for line in (output_dir / "selection.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [row["qa_id"] for row in selection] == ["C1", "C2"]
    dedup = json.loads(
        (output_dir / "deduplication_report.json").read_text(encoding="utf-8")
    )
    assert dedup["input_count"] == 3
    assert dedup["selected_count"] == 2
    media = json.loads(
        (output_dir / "media_preflight.json").read_text(encoding="utf-8")
    )
    assert media["media_ready_count"] == 0
    assert media["missing_media_count"] == 2


def test_pair_results_use_only_two_valid_conditions() -> None:
    from egolife_two_user_qa.qwen_two_condition_review import (
        build_paired_results,
    )

    items = [
        _item("Q1", "One?"),
        _item("Q2", "Two?"),
        _item("Q3", "Three?"),
    ]
    predictions = [
        {
            "qa_id": "Q1",
            "condition_id": "minimum_set",
            "run_status": "ok",
            "parse_status": "valid",
            "is_correct": True,
            "elapsed_seconds": 1.0,
            "attempt": 1,
        },
        {
            "qa_id": "Q1",
            "condition_id": "all_six",
            "run_status": "ok",
            "parse_status": "valid",
            "is_correct": False,
            "elapsed_seconds": 2.0,
            "attempt": 1,
        },
        {
            "qa_id": "Q2",
            "condition_id": "minimum_set",
            "run_status": "ok",
            "parse_status": "valid",
            "is_correct": False,
            "elapsed_seconds": 1.0,
            "attempt": 1,
        },
        {
            "qa_id": "Q2",
            "condition_id": "all_six",
            "run_status": "ok",
            "parse_status": "valid",
            "is_correct": True,
            "elapsed_seconds": 2.0,
            "attempt": 1,
        },
        {
            "qa_id": "Q3",
            "condition_id": "minimum_set",
            "run_status": "ok",
            "parse_status": "invalid_missing",
            "is_correct": None,
            "elapsed_seconds": 1.0,
            "attempt": 1,
        },
    ]
    paired, summary = build_paired_results(
        items,
        predictions,
        missing_media_qa_ids=set(),
    )
    assert summary["gold_count"] == 3
    assert summary["paired_count"] == 2
    assert summary["accuracy_minimum"] == 0.5
    assert summary["accuracy_all_six"] == 0.5
    assert summary["accuracy_delta"] == 0.0
    assert summary["parse_failures_by_condition"] == {
        "minimum_set": 1,
        "all_six": 0,
    }
    assert summary["pair_categories"] == {
        "both_correct": 0,
        "both_wrong": 0,
        "minimum_only_correct": 1,
        "all_six_only_correct": 1,
    }
    q3 = next(row for row in paired if row["qa_id"] == "Q3")
    assert q3["unpaired_reason"] == "invalid_or_missing_condition"


def test_finalize_review_writes_paired_summary_and_chinese_report(
    tmp_path: Path,
) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import finalize_review

    item = _item("Q1", "Which bottle was selected?")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    predictions = [
        {
            "qa_id": "Q1",
            "condition_id": "minimum_set",
            "run_status": "ok",
            "parse_status": "valid",
            "predicted_choice": "B",
            "correct_choice": "B",
            "is_correct": True,
            "raw_output": "CHOICE: B",
            "input_users": ["Jake", "Lucia"],
            "elapsed_seconds": 1.0,
            "attempt": 1,
        },
        {
            "qa_id": "Q1",
            "condition_id": "all_six",
            "run_status": "ok",
            "parse_status": "valid",
            "predicted_choice": "A",
            "correct_choice": "B",
            "is_correct": False,
            "raw_output": "CHOICE: A",
            "input_users": [
                "Jake",
                "Alice",
                "Tasha",
                "Lucia",
                "Katrina",
                "Shure",
            ],
            "elapsed_seconds": 2.0,
            "attempt": 1,
        },
    ]
    for row in predictions:
        with (output_dir / "predictions.jsonl").open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(json.dumps(row) + "\n")
    finalize_review([item], output_dir, missing_media_qa_ids=set())
    assert (output_dir / "paired_results.jsonl").is_file()
    summary = json.loads(
        (output_dir / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["paired_count"] == 1
    report = (output_dir / "report_cn.md").read_text(encoding="utf-8")
    assert "有效配对数：**1**" in report
    assert "仅 minimum set 正确" in report
    assert "Minimum set 预测：B" in report
    assert "All six 预测：A" in report
    assert "Minimum set 用户：Jake、Lucia" in report
    assert "All six 耗时：2.000 秒" in report


def _write_cli_markdown(path: Path) -> None:
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


def test_cli_prepare_only_does_not_create_runner(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from egolife_two_user_qa.tools import run_qwen_two_condition_review as cli

    markdown = tmp_path / "QA.md"
    _write_cli_markdown(markdown)
    curated = tmp_path / "curated.jsonl"
    curated.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "make_runner",
        lambda *args, **kwargs: pytest.fail("runner must not load"),
    )
    output_dir = tmp_path / "run"
    rc = cli.main(
        [
            "--approved-markdown",
            str(markdown),
            "--curated-jsonl",
            str(curated),
            "--media-root",
            str(tmp_path / "media"),
            "--output-dir",
            str(output_dir),
            "--prepare-only",
        ]
    )
    assert rc == 0
    manifest = json.loads(
        (output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["mode"] == "prepare_only"
    assert manifest["selected_count"] == 1


def test_cli_model_mode_requires_explicit_inference_contract(
    tmp_path: Path,
) -> None:
    from egolife_two_user_qa.tools import run_qwen_two_condition_review as cli

    markdown = tmp_path / "QA.md"
    _write_cli_markdown(markdown)
    curated = tmp_path / "curated.jsonl"
    curated.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="model review requires"):
        cli.main(
            [
                "--approved-markdown",
                str(markdown),
                "--curated-jsonl",
                str(curated),
                "--media-root",
                str(tmp_path / "media"),
                "--output-dir",
                str(tmp_path / "run"),
            ]
        )


def test_cli_rejects_changed_contract_in_existing_model_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from egolife_two_user_qa.tools import run_qwen_two_condition_review as cli

    markdown = tmp_path / "QA.md"
    _write_cli_markdown(markdown)
    curated = tmp_path / "curated.jsonl"
    curated.write_text("", encoding="utf-8")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    existing = {
        "created_at_utc": "2026-09-01T00:00:00+00:00",
        "mode": "model_review",
        "approved_markdown": str(markdown.resolve()),
        "curated_jsonl": str(curated.resolve()),
        "media_root": str((tmp_path / "media").resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_count": 1,
        "media_ready_count": 0,
        "model_id": "Qwen/Qwen3.8-27B",
        "backend": "transformers-local-memory-safe",
        "max_new_tokens": 256,
        "max_image_pixels": 65536,
        "disable_thinking": True,
        "decoding_mode": "greedy",
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(existing),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        cli,
        "make_runner",
        lambda *args, **kwargs: pytest.fail("incompatible run must stop first"),
    )
    with pytest.raises(SystemExit, match="existing run contract differs"):
        cli.main(
            [
                "--approved-markdown",
                str(markdown),
                "--curated-jsonl",
                str(curated),
                "--media-root",
                str(tmp_path / "media"),
                "--output-dir",
                str(output_dir),
                "--max-new-tokens",
                "512",
                "--max-image-pixels",
                "65536",
                "--disable-thinking",
            ]
        )
