# Qwen 双条件视频 QA 评审本地实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 实现一个独立、可恢复、可审计的 Qwen 双条件视频 QA 评审入口，将已批准的 7 条 Markdown 题目和 17 条 trace-review JSONL 题目去重为 21 条 gold，并在 `minimum_set` 与 `all_six` 条件间生成严格配对统计。

**架构：** 新增一个核心模块承载输入归一化、去重、媒体预检、prompt、解析、增量执行、配对统计和中文报告；新增一个薄命令行入口负责创建现有 memory-safe runner。生产代码通过 runner 注入实现零 GPU 测试，不修改 `cli.py`、正式 QA generator 或现有 Slurm 作业。

**技术栈：** Python 3.11+、标准库 `argparse`/`dataclasses`/`json`/`pathlib`、pytest、现有 `qwen3vl_runner.make_runner()`、Qwen Transformers memory-safe backend。

---

## 范围与执行边界

本计划只完成本地评审器及真实输入的零 GPU prepare-only 验证。Torch 媒体定位、同步、`.sbatch`、一次最小 smoke 和正式运行是独立的跨系统阶段；本地实现验收后，必须重新读取 Torch 权威文档、核验共享桥和实时远端状态，并取得用户明确授权后另行执行。

当前 checkout 目录名不是 Python 包名，且 `.venv` 没有 editable install。直接运行 pytest 会得到 `ModuleNotFoundError: egolife_two_user_qa`。已验证的本地测试方式是在临时目录创建指向当前 checkout 的 `egolife_two_user_qa` junction，并将 junction 父目录加入 `PYTHONPATH`。这只是测试启动环境，不修改仓库源码。

## 文件结构

- 新建：`qwen_two_condition_review.py`——全部纯数据合同、runner 调用编排和报告逻辑。
- 新建：`tools/run_qwen_two_condition_review.py`——命令行参数、runner 创建和入口返回码。
- 新建：`tests/test_qwen_two_condition_review.py`——纯 CPU 单元测试和当前真实输入计数回归。
- 读取但不修改：`qwen3vl_runner.py`——复用 `Generator`、`GenerationCallProfile`、`MEMORY_SAFE_BACKEND` 和 `make_runner()`。
- 不修改：`cli.py`、`video_qa_loop.py`、现有 `.sbatch`、人工审核源文件和既有运行产物。

### Task 1：输入归一化、校验与去重

**文件：**

- 创建：`qwen_two_condition_review.py`
- 创建：`tests/test_qwen_two_condition_review.py`

- [ ] **Step 1：写 Markdown、JSONL、校验和去重失败测试**

在 `tests/test_qwen_two_condition_review.py` 写入以下测试骨架：

```python
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
```

- [ ] **Step 2：运行测试并确认因核心模块缺失而失败**

在 PowerShell 中执行：

```powershell
$aliasRoot = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
$aliasPath = Join-Path $aliasRoot 'egolife_two_user_qa'
if (-not (Test-Path -LiteralPath $aliasRoot)) { New-Item -ItemType Directory -Path $aliasRoot | Out-Null }
if (-not (Test-Path -LiteralPath $aliasPath)) { New-Item -ItemType Junction -Path $aliasPath -Target (Get-Location).Path | Out-Null }
$env:PYTHONPATH = $aliasRoot
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q --maxfail=1
```

预期：测试收集失败，原因是 `egolife_two_user_qa.qwen_two_condition_review` 尚不存在。

- [ ] **Step 3：实现最小输入与去重代码**

在 `qwen_two_condition_review.py` 写入以下类型和函数；函数名和字段在后续任务保持不变：

```python
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

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
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).strip()).casefold()


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
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
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
            minimum_required_users=tuple(str(user) for user in value["required_users"]),
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
        group = re.search(r"generation group[：:]\s*(\S+)", body, re.IGNORECASE)
        evidence = re.search(r"evidence id[：:]\s*(\S+)", body, re.IGNORECASE)
        users = re.search(r"minimum required users[：:]\s*([^\n]+)", body, re.IGNORECASE)
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
```

- [ ] **Step 4：运行 Task 1 测试并确认通过**

运行：

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q
```

预期：Task 1 新增测试全部通过；真实文件存在时，最后一个测试确认 `21` 条保留、`3` 条去重。

- [ ] **Step 5：只提交 Task 1 文件**

```powershell
git add -- qwen_two_condition_review.py tests/test_qwen_two_condition_review.py
git diff --cached --check
git commit -m "feat: 归一化并去重双条件评审题目"
```

### Task 2：媒体条件、统一 prompt 与严格解析

**文件：**

- 修改：`qwen_two_condition_review.py`
- 修改：`tests/test_qwen_two_condition_review.py`

- [ ] **Step 1：写媒体、prompt、防泄漏和解析失败测试**

追加导入和测试：

```python
from egolife_two_user_qa.qwen_two_condition_review import (
    ALL_SIX_CONDITION,
    MINIMUM_SET_CONDITION,
    build_condition_specs,
    build_prompt,
    parse_choice,
)


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
    assert specs[1].input_users == ("Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure")
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
def test_parse_choice_rejects_missing_or_conflicting_output(raw: str, status: str) -> None:
    parsed = parse_choice(raw)
    assert parsed.choice is None
    assert parsed.status == status
```

- [ ] **Step 2：运行新增测试并确认缺失符号失败**

运行：

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q --maxfail=1
```

预期：因 `build_condition_specs`、`build_prompt` 或 `parse_choice` 尚未定义而失败。

- [ ] **Step 3：实现媒体、prompt 与解析**

在核心模块追加：

```python
MINIMUM_SET_CONDITION = "minimum_set"
ALL_SIX_CONDITION = "all_six"


@dataclass(frozen=True)
class ConditionSpec:
    condition_id: str
    input_users: tuple[str, ...]
    video_paths: tuple[str, ...]
    missing_paths: tuple[str, ...]


@dataclass(frozen=True)
class ChoiceParse:
    choice: str | None
    status: str


def generation_group_directory(generation_group_id: str) -> str:
    if not re.fullmatch(r"DAY\d+::\d+", generation_group_id):
        raise ValueError("invalid generation_group_id")
    return generation_group_id.replace("::", "_")


def build_condition_specs(item: GoldItem, media_root: str | Path) -> tuple[ConditionSpec, ...]:
    group_root = Path(media_root) / generation_group_directory(item.generation_group_id)
    specs: list[ConditionSpec] = []
    for condition_id, users in (
        (MINIMUM_SET_CONDITION, item.minimum_required_users),
        (ALL_SIX_CONDITION, SIX_USERS),
    ):
        paths = tuple(str(group_root / f"{user}.mp4") for user in users)
        missing = tuple(path for path in paths if not Path(path).is_file())
        specs.append(ConditionSpec(condition_id, tuple(users), paths, missing))
    return tuple(specs)


def build_prompt(question: str, options: Sequence[str]) -> str:
    if len(options) != 5:
        raise ValueError("prompt requires exactly five options")
    option_lines = "\n".join(
        f"{choice}. {option}" for choice, option in zip(CHOICES, options, strict=True)
    )
    return (
        "You are given one or more videos and a multiple-choice question.\n"
        "Answer the question using only the provided videos.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{option_lines}\n\n"
        "Select exactly one option.\n"
        "Output exactly two lines:\n"
        "CHOICE: <A, B, C, D, or E>\n"
        "ANSWER: <brief answer>"
    )


def parse_choice(raw_output: str) -> ChoiceParse:
    declared = re.findall(
        r"(?im)^\s*CHOICE\s*:\s*([A-E])\s*$",
        raw_output,
    )
    distinct = tuple(dict.fromkeys(value.upper() for value in declared))
    if len(distinct) > 1:
        return ChoiceParse(None, "invalid_ambiguous")
    if len(distinct) == 1:
        return ChoiceParse(distinct[0], "valid")
    fallback = re.fullmatch(r"\s*(?:\(([A-E])\)|([A-E])\.?)\s*", raw_output, re.IGNORECASE)
    if fallback is None:
        return ChoiceParse(None, "invalid_missing")
    return ChoiceParse((fallback.group(1) or fallback.group(2)).upper(), "valid")
```

- [ ] **Step 4：运行 Task 2 测试并确认通过**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q
```

预期：Task 1 与 Task 2 测试全部通过。

- [ ] **Step 5：只提交 Task 2 变更**

```powershell
git add -- qwen_two_condition_review.py tests/test_qwen_two_condition_review.py
git diff --cached --check
git commit -m "feat: 构造双条件媒体与统一回答提示"
```

### Task 3：增量执行、错误落盘与显式恢复

**文件：**

- 修改：`qwen_two_condition_review.py`
- 修改：`tests/test_qwen_two_condition_review.py`

- [ ] **Step 1：写 runner 调用、防泄漏、落盘和恢复失败测试**

追加以下测试辅助类和测试：

```python
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


def test_run_item_uses_identical_prompt_and_only_changes_videos(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import run_items

    item = _item("SECRET_GOLD_ID", "Which bottle was selected?")
    runner = FakeRunner(["CHOICE: B\nANSWER: two", "CHOICE: A\nANSWER: one"])
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
    rows = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["condition_id"] for row in rows] == ["minimum_set", "all_six"]
    assert [row["is_correct"] for row in rows] == [True, False]


def test_invalid_parse_is_saved_without_automatic_retry(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import run_items

    runner = FakeRunner(["unclear", "CHOICE: B\nANSWER: two"])
    output_dir = tmp_path / "run"
    run_items([_item("Q1", "Question?")], _complete_media(tmp_path), output_dir, runner)
    rows = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["parse_status"] == "invalid_missing"
    assert rows[0]["is_correct"] is None
    assert rows[0]["attempt"] == 1
    assert len(runner.calls) == 2


def test_resume_skips_valid_rows_and_explicitly_reruns_invalid(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import run_items

    item = _item("Q1", "Question?")
    media_root = _complete_media(tmp_path)
    output_dir = tmp_path / "run"
    first = FakeRunner(["unclear", "CHOICE: B\nANSWER: two"])
    run_items([item], media_root, output_dir, first)
    second = FakeRunner(["CHOICE: B\nANSWER: two"])
    run_items([item], media_root, output_dir, second, rerun_nonvalid=True)
    assert len(second.calls) == 1
    rows = [json.loads(line) for line in (output_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["attempt"] for row in rows if row["condition_id"] == "minimum_set"] == [1, 2]


def test_runner_error_is_saved_and_stops_following_calls(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import ReviewExecutionError, run_items

    runner = FakeRunner([RuntimeError("decoder failed"), "CHOICE: B"])
    output_dir = tmp_path / "run"
    with pytest.raises(ReviewExecutionError, match="decoder failed"):
        run_items([_item("Q1", "Question?")], _complete_media(tmp_path), output_dir, runner)
    assert len(runner.calls) == 1
    row = json.loads((output_dir / "predictions.jsonl").read_text(encoding="utf-8").strip())
    assert row["run_status"] == "error"
    assert row["parse_status"] == "not_parsed"
    assert row["attempt"] == 1
```

- [ ] **Step 2：运行新增测试并观察正确失败**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q --maxfail=1
```

预期：因 `run_items` 和 `ReviewExecutionError` 尚未定义而失败。

- [ ] **Step 3：实现增量执行与恢复**

在核心模块增加 `time` 导入，并追加：

```python
import time


class ReviewExecutionError(RuntimeError):
    pass


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]


def _latest_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row["qa_id"]), str(row["condition_id"]))] = row
    return latest


def _next_attempt(rows: Iterable[dict[str, Any]], qa_id: str, condition_id: str) -> int:
    attempts = [
        int(row.get("attempt", 0))
        for row in rows
        if row.get("qa_id") == qa_id and row.get("condition_id") == condition_id
    ]
    return max(attempts, default=0) + 1


def run_items(
    items: Sequence[GoldItem],
    media_root: str | Path,
    output_dir: str | Path,
    runner: Any,
    *,
    call_profile: Any = None,
    rerun_nonvalid: bool = False,
) -> list[dict[str, Any]]:
    output_root = Path(output_dir)
    predictions_path = output_root / "predictions.jsonl"
    rows = read_prediction_rows(predictions_path)
    latest = _latest_by_key(rows)
    for item in items:
        prompt = build_prompt(item.question, item.options)
        for order, spec in enumerate(build_condition_specs(item, media_root), 1):
            if spec.missing_paths:
                continue
            key = (item.qa_id, spec.condition_id)
            prior = latest.get(key)
            if prior is not None:
                prior_valid = prior.get("run_status") == "ok" and prior.get("parse_status") == "valid"
                if prior_valid or not rerun_nonvalid:
                    continue
            attempt = _next_attempt(rows, item.qa_id, spec.condition_id)
            started = time.perf_counter()
            try:
                raw_output = runner.generate(
                    prompt,
                    image_paths=[],
                    video_paths=list(spec.video_paths),
                    decoding_mode="greedy",
                    call_profile=call_profile,
                )
            except Exception as exc:
                row = {
                    "qa_id": item.qa_id,
                    "condition_id": spec.condition_id,
                    "input_users": list(spec.input_users),
                    "video_paths": list(spec.video_paths),
                    "predicted_choice": None,
                    "correct_choice": item.correct,
                    "is_correct": None,
                    "raw_output": "",
                    "parse_status": "not_parsed",
                    "run_status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "model_id": str(runner.model_id),
                    "elapsed_seconds": time.perf_counter() - started,
                    "condition_order": order,
                    "attempt": attempt,
                }
                _append_jsonl(predictions_path, row)
                raise ReviewExecutionError(str(exc)) from exc
            parsed = parse_choice(str(raw_output))
            row = {
                "qa_id": item.qa_id,
                "condition_id": spec.condition_id,
                "input_users": list(spec.input_users),
                "video_paths": list(spec.video_paths),
                "predicted_choice": parsed.choice,
                "correct_choice": item.correct,
                "is_correct": parsed.choice == item.correct if parsed.status == "valid" else None,
                "raw_output": str(raw_output),
                "parse_status": parsed.status,
                "run_status": "ok",
                "error_type": None,
                "error_message": None,
                "model_id": str(runner.model_id),
                "elapsed_seconds": time.perf_counter() - started,
                "condition_order": order,
                "attempt": attempt,
            }
            _append_jsonl(predictions_path, row)
            rows.append(row)
            latest[key] = row
    return read_prediction_rows(predictions_path)
```

- [ ] **Step 4：运行 Task 3 测试并确认通过**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q
```

- [ ] **Step 5：只提交 Task 3 变更**

```powershell
git add -- qwen_two_condition_review.py tests/test_qwen_two_condition_review.py
git diff --cached --check
git commit -m "feat: 增量执行并恢复双条件评审"
```

### Task 4：准备产物、配对统计与中文报告

**文件：**

- 修改：`qwen_two_condition_review.py`
- 修改：`tests/test_qwen_two_condition_review.py`

- [ ] **Step 1：写 prepare-only、配对统计和报告失败测试**

追加测试：

```python
def test_prepare_review_writes_selection_dedup_and_media_reports(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import prepare_review

    markdown = _item("M1", "Question one?")
    curated = _item("C1", "Question one?", source="curated_trace_v3")
    missing = _item("C2", "Question two?", source="curated_trace_v3")
    output_dir = tmp_path / "run"
    result = prepare_review([markdown, curated, missing], tmp_path / "media", output_dir)
    assert len(result.items) == 2
    selection = [json.loads(line) for line in (output_dir / "selection.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["qa_id"] for row in selection] == ["C1", "C2"]
    dedup = json.loads((output_dir / "deduplication_report.json").read_text(encoding="utf-8"))
    assert dedup["input_count"] == 3
    assert dedup["selected_count"] == 2
    media = json.loads((output_dir / "media_preflight.json").read_text(encoding="utf-8"))
    assert media["media_ready_count"] == 0
    assert media["missing_media_count"] == 2


def test_pair_results_use_only_two_valid_conditions(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import build_paired_results

    items = [_item("Q1", "One?"), _item("Q2", "Two?"), _item("Q3", "Three?")]
    predictions = [
        {"qa_id": "Q1", "condition_id": "minimum_set", "run_status": "ok", "parse_status": "valid", "is_correct": True, "elapsed_seconds": 1.0, "attempt": 1},
        {"qa_id": "Q1", "condition_id": "all_six", "run_status": "ok", "parse_status": "valid", "is_correct": False, "elapsed_seconds": 2.0, "attempt": 1},
        {"qa_id": "Q2", "condition_id": "minimum_set", "run_status": "ok", "parse_status": "valid", "is_correct": False, "elapsed_seconds": 1.0, "attempt": 1},
        {"qa_id": "Q2", "condition_id": "all_six", "run_status": "ok", "parse_status": "valid", "is_correct": True, "elapsed_seconds": 2.0, "attempt": 1},
        {"qa_id": "Q3", "condition_id": "minimum_set", "run_status": "ok", "parse_status": "invalid_missing", "is_correct": None, "elapsed_seconds": 1.0, "attempt": 1},
    ]
    paired, summary = build_paired_results(items, predictions, missing_media_qa_ids=set())
    assert summary["gold_count"] == 3
    assert summary["paired_count"] == 2
    assert summary["accuracy_minimum"] == 0.5
    assert summary["accuracy_all_six"] == 0.5
    assert summary["accuracy_delta"] == 0.0
    assert summary["pair_categories"] == {
        "both_correct": 0,
        "both_wrong": 0,
        "minimum_only_correct": 1,
        "all_six_only_correct": 1,
    }
    assert next(row for row in paired if row["qa_id"] == "Q3")["unpaired_reason"] == "invalid_or_missing_condition"


def test_finalize_review_writes_paired_summary_and_chinese_report(tmp_path: Path) -> None:
    from egolife_two_user_qa.qwen_two_condition_review import finalize_review

    item = _item("Q1", "Which bottle was selected?")
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    predictions = [
        {"qa_id": "Q1", "condition_id": "minimum_set", "run_status": "ok", "parse_status": "valid", "predicted_choice": "B", "correct_choice": "B", "is_correct": True, "raw_output": "CHOICE: B", "elapsed_seconds": 1.0, "attempt": 1},
        {"qa_id": "Q1", "condition_id": "all_six", "run_status": "ok", "parse_status": "valid", "predicted_choice": "A", "correct_choice": "B", "is_correct": False, "raw_output": "CHOICE: A", "elapsed_seconds": 2.0, "attempt": 1},
    ]
    for row in predictions:
        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")
    finalize_review([item], output_dir, missing_media_qa_ids=set())
    assert (output_dir / "paired_results.jsonl").is_file()
    assert json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))["paired_count"] == 1
    report = (output_dir / "report_cn.md").read_text(encoding="utf-8")
    assert "有效配对数：**1**" in report
    assert "仅 minimum set 正确" in report
```

- [ ] **Step 2：运行新增测试并确认缺失函数失败**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q --maxfail=1
```

- [ ] **Step 3：实现 prepare、配对、summary 和中文报告**

在核心模块追加以下函数。`_write_json` 使用 UTF-8、中文不转义，`_write_jsonl` 每次原子重建派生文件；原始 `predictions.jsonl` 仍保持追加写入：

```python
from collections import Counter


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_review(
    items: Sequence[GoldItem],
    media_root: str | Path,
    output_dir: str | Path,
) -> DeduplicationResult:
    output_root = Path(output_dir)
    result = deduplicate_items(items)
    _write_jsonl(output_root / "selection.jsonl", (item_to_dict(item) for item in result.items))
    _write_json(
        output_root / "deduplication_report.json",
        {
            "input_count": len(items),
            "selected_count": len(result.items),
            "removed_count": len(result.removed),
            "removed": list(result.removed),
            "same_group_nonduplicates": list(result.same_group_nonduplicates),
        },
    )
    rows: list[dict[str, Any]] = []
    for item in result.items:
        specs = build_condition_specs(item, media_root)
        missing = sorted({path for spec in specs for path in spec.missing_paths})
        rows.append(
            {
                "qa_id": item.qa_id,
                "generation_group_id": item.generation_group_id,
                "media_ready": not missing,
                "missing_paths": missing,
                "conditions": [asdict(spec) for spec in specs],
            }
        )
    _write_json(
        output_root / "media_preflight.json",
        {
            "gold_count": len(result.items),
            "media_ready_count": sum(bool(row["media_ready"]) for row in rows),
            "missing_media_count": sum(not bool(row["media_ready"]) for row in rows),
            "items": rows,
        },
    )
    return result


def build_paired_results(
    items: Sequence[GoldItem],
    predictions: Sequence[dict[str, Any]],
    *,
    missing_media_qa_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    latest = _latest_by_key(predictions)
    paired_rows: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    minimum_correct = 0
    all_six_correct = 0
    paired_count = 0
    for item in items:
        minimum = latest.get((item.qa_id, MINIMUM_SET_CONDITION))
        all_six = latest.get((item.qa_id, ALL_SIX_CONDITION))
        valid_minimum = minimum is not None and minimum.get("run_status") == "ok" and minimum.get("parse_status") == "valid"
        valid_all_six = all_six is not None and all_six.get("run_status") == "ok" and all_six.get("parse_status") == "valid"
        category = None
        unpaired_reason = None
        if item.qa_id in missing_media_qa_ids:
            unpaired_reason = "missing_media"
        elif not valid_minimum or not valid_all_six:
            unpaired_reason = "invalid_or_missing_condition"
        else:
            paired_count += 1
            minimum_ok = bool(minimum["is_correct"])
            all_six_ok = bool(all_six["is_correct"])
            minimum_correct += int(minimum_ok)
            all_six_correct += int(all_six_ok)
            category = (
                "both_correct" if minimum_ok and all_six_ok
                else "minimum_only_correct" if minimum_ok
                else "all_six_only_correct" if all_six_ok
                else "both_wrong"
            )
            categories[category] += 1
        paired_rows.append(
            {
                "qa_id": item.qa_id,
                "question": item.question,
                "correct_choice": item.correct,
                "minimum_set": minimum,
                "all_six": all_six,
                "paired_valid": unpaired_reason is None,
                "pair_category": category,
                "unpaired_reason": unpaired_reason,
            }
        )
    accuracy_minimum = minimum_correct / paired_count if paired_count else None
    accuracy_all_six = all_six_correct / paired_count if paired_count else None
    summary = {
        "gold_count": len(items),
        "media_ready_count": len(items) - len(missing_media_qa_ids),
        "missing_media_count": len(missing_media_qa_ids),
        "paired_count": paired_count,
        "unpaired_count": len(items) - paired_count,
        "accuracy_minimum": accuracy_minimum,
        "accuracy_all_six": accuracy_all_six,
        "accuracy_delta": (
            accuracy_all_six - accuracy_minimum
            if accuracy_minimum is not None and accuracy_all_six is not None
            else None
        ),
        "pair_categories": {
            name: categories[name]
            for name in ("both_correct", "both_wrong", "minimum_only_correct", "all_six_only_correct")
        },
        "parse_failure_count": sum(
            row.get("run_status") == "ok" and row.get("parse_status") != "valid"
            for row in latest.values()
        ),
        "inference_error_count": sum(row.get("run_status") == "error" for row in latest.values()),
        "elapsed_seconds_total": sum(float(row.get("elapsed_seconds", 0.0)) for row in latest.values()),
    }
    return paired_rows, summary


def render_cn_report(paired_rows: Sequence[dict[str, Any]], summary: dict[str, Any]) -> str:
    def percent(value: float | None) -> str:
        return "不可计算" if value is None else f"{value * 100:.2f}%"

    lines = [
        "# Qwen 双条件视频 QA 配对评审报告",
        "",
        "## 统计摘要",
        "",
        f"- Gold 题数：**{summary['gold_count']}**",
        f"- 媒体完整题数：**{summary['media_ready_count']}**",
        f"- 有效配对数：**{summary['paired_count']}**",
        f"- Minimum set 准确率：**{percent(summary['accuracy_minimum'])}**",
        f"- All six 准确率：**{percent(summary['accuracy_all_six'])}**",
        f"- All six 减 minimum set：**{percent(summary['accuracy_delta'])}**",
        f"- 两个条件都正确：**{summary['pair_categories']['both_correct']}**",
        f"- 两个条件都错误：**{summary['pair_categories']['both_wrong']}**",
        f"- 仅 minimum set 正确：**{summary['pair_categories']['minimum_only_correct']}**",
        f"- 仅 all six 正确：**{summary['pair_categories']['all_six_only_correct']}**",
        f"- 解析失败：**{summary['parse_failure_count']}**",
        f"- 推理异常：**{summary['inference_error_count']}**",
        "",
        "准确率只使用两个条件均成功且解析有效的同一批 QA。一次小样本运行不证明差异稳定或具有统计显著性。",
        "",
        "## 逐题结果",
    ]
    for row in paired_rows:
        lines.extend(
            [
                "",
                f"### {row['qa_id']}",
                "",
                f"- 问题：{row['question']}",
                f"- 正确选项：{row['correct_choice']}",
                f"- 配对有效：{'是' if row['paired_valid'] else '否'}",
                f"- 分类：{row['pair_category'] or row['unpaired_reason']}",
            ]
        )
    return "\n".join(lines) + "\n"


def finalize_review(
    items: Sequence[GoldItem],
    output_dir: str | Path,
    *,
    missing_media_qa_ids: set[str],
) -> dict[str, Any]:
    output_root = Path(output_dir)
    predictions = read_prediction_rows(output_root / "predictions.jsonl")
    paired, summary = build_paired_results(
        items,
        predictions,
        missing_media_qa_ids=missing_media_qa_ids,
    )
    _write_jsonl(output_root / "paired_results.jsonl", paired)
    _write_json(output_root / "summary.json", summary)
    (output_root / "report_cn.md").write_text(
        render_cn_report(paired, summary),
        encoding="utf-8",
    )
    return summary
```

- [ ] **Step 4：运行 Task 4 测试并确认通过**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q
```

- [ ] **Step 5：只提交 Task 4 变更**

```powershell
git add -- qwen_two_condition_review.py tests/test_qwen_two_condition_review.py
git diff --cached --check
git commit -m "feat: 汇总双条件配对评审结果"
```

### Task 5：薄命令行入口与 manifest

**文件：**

- 创建：`tools/run_qwen_two_condition_review.py`
- 修改：`tests/test_qwen_two_condition_review.py`

- [ ] **Step 1：写 prepare-only 不创建模型的失败测试**

追加测试：

```python
def test_cli_prepare_only_does_not_create_runner(tmp_path: Path, monkeypatch) -> None:
    from egolife_two_user_qa.tools import run_qwen_two_condition_review as cli

    markdown = tmp_path / "QA.md"
    markdown.write_text(
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
    curated = tmp_path / "curated.jsonl"
    curated.write_text("", encoding="utf-8")
    monkeypatch.setattr(cli, "make_runner", lambda *args, **kwargs: pytest.fail("runner must not load"))
    output_dir = tmp_path / "run"
    rc = cli.main(
        [
            "--approved-markdown", str(markdown),
            "--curated-jsonl", str(curated),
            "--media-root", str(tmp_path / "media"),
            "--output-dir", str(output_dir),
            "--prepare-only",
        ]
    )
    assert rc == 0
    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["mode"] == "prepare_only"
    assert manifest["selected_count"] == 1


def test_cli_model_mode_requires_explicit_inference_contract(tmp_path: Path) -> None:
    from egolife_two_user_qa.tools import run_qwen_two_condition_review as cli

    markdown = tmp_path / "QA.md"
    markdown.write_text(
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
    curated = tmp_path / "curated.jsonl"
    curated.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit, match="model review requires"):
        cli.main(
            [
                "--approved-markdown", str(markdown),
                "--curated-jsonl", str(curated),
                "--media-root", str(tmp_path / "media"),
                "--output-dir", str(tmp_path / "run"),
            ]
        )
```

- [ ] **Step 2：运行测试并确认工具模块缺失**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q --maxfail=1
```

- [ ] **Step 3：实现工具入口**

创建 `tools/run_qwen_two_condition_review.py`：

```python
"""运行 Qwen minimum-set 与 all-six 成对视频 QA 评审。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from egolife_two_user_qa.qwen3vl_runner import (
    GenerationCallProfile,
    MEMORY_SAFE_BACKEND,
    make_runner,
)
from egolife_two_user_qa.qwen_two_condition_review import (
    finalize_review,
    load_approved_markdown,
    load_curated_jsonl,
    prepare_review,
    run_items,
)

DEFAULT_MODEL_ID = "Qwen/Qwen3.8-27B"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-markdown", required=True)
    parser.add_argument("--curated-jsonl", required=True)
    parser.add_argument("--media-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-image-pixels", type=int)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument("--disable-thinking", dest="disable_thinking", action="store_true")
    thinking.add_argument("--enable-thinking", dest="disable_thinking", action="store_false")
    parser.set_defaults(disable_thinking=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--rerun-nonvalid", action="store_true")
    return parser


def _write_manifest(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    approved = load_approved_markdown(args.approved_markdown)
    curated = load_curated_jsonl(args.curated_jsonl)
    output_dir = Path(args.output_dir)
    prepared = prepare_review(
        [*approved, *curated],
        args.media_root,
        output_dir,
    )
    media_report = json.loads((output_dir / "media_preflight.json").read_text(encoding="utf-8"))
    missing_media_qa_ids = {
        row["qa_id"] for row in media_report["items"] if not row["media_ready"]
    }
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "prepare_only" if args.prepare_only else "model_review",
        "approved_markdown": str(Path(args.approved_markdown).resolve()),
        "curated_jsonl": str(Path(args.curated_jsonl).resolve()),
        "media_root": str(Path(args.media_root).resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_count": len(prepared.items),
        "media_ready_count": media_report["media_ready_count"],
        "model_id": args.model_id,
        "backend": MEMORY_SAFE_BACKEND,
        "max_new_tokens": args.max_new_tokens,
        "max_image_pixels": args.max_image_pixels,
        "disable_thinking": args.disable_thinking,
        "decoding_mode": "greedy",
    }
    if args.prepare_only:
        _write_manifest(output_dir / "run_manifest.json", manifest)
        finalize_review(
            prepared.items,
            output_dir,
            missing_media_qa_ids=missing_media_qa_ids,
        )
        print(f"SELECTED_COUNT={len(prepared.items)}")
        print(f"MEDIA_READY_COUNT={media_report['media_ready_count']}")
        print(f"OUTPUT_DIR={output_dir}")
        return 0
    if args.max_new_tokens is None or args.max_image_pixels is None or args.disable_thinking is None:
        raise SystemExit(
            "model review requires --max-new-tokens, --max-image-pixels, "
            "and exactly one of --enable-thinking/--disable-thinking"
        )
    runner = make_runner(
        MEMORY_SAFE_BACKEND,
        model_id=args.model_id,
        max_new_tokens=args.max_new_tokens,
        max_image_pixels=args.max_image_pixels,
        disable_thinking=args.disable_thinking,
    )
    manifest.update(
        {
            "effective_video_fps": getattr(runner, "video_fps", None),
            "effective_min_video_pixels": getattr(runner, "min_video_pixels", None),
            "effective_max_input_tokens": getattr(runner, "max_input_tokens", None),
        }
    )
    _write_manifest(output_dir / "run_manifest.json", manifest)
    call_profile = GenerationCallProfile(
        max_new_tokens=args.max_new_tokens,
        disable_thinking=args.disable_thinking,
    )
    run_items(
        prepared.items,
        args.media_root,
        output_dir,
        runner,
        call_profile=call_profile,
        rerun_nonvalid=args.rerun_nonvalid,
    )
    summary = finalize_review(
        prepared.items,
        output_dir,
        missing_media_qa_ids=missing_media_qa_ids,
    )
    print(f"PAIRED_COUNT={summary['paired_count']}")
    print(f"OUTPUT_DIR={output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4：运行全部新测试并确认通过**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py -q
```

- [ ] **Step 5：只提交 Task 5 文件**

```powershell
git add -- tools/run_qwen_two_condition_review.py tests/test_qwen_two_condition_review.py
git diff --cached --check
git commit -m "feat: 添加 Qwen 双条件评审入口"
```

### Task 6：真实输入 prepare-only 与本地验收

**文件：**

- 不修改生产文件
- 生成本地审核产物：`C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts\qwen_two_condition_review_preflight_20260901`

- [ ] **Step 1：运行新模块和现有 runner 的聚焦测试**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m pytest tests\test_qwen_two_condition_review.py tests\test_qwen_runner_compat.py -q
```

预期：新测试与现有 16 个 runner compatibility 测试全部通过；不得把此前缺少 `PYTHONPATH` 导致的收集错误算作代码失败。

- [ ] **Step 2：运行语法编译和差异检查**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
.\.venv\Scripts\python.exe -m py_compile qwen_two_condition_review.py tools\run_qwen_two_condition_review.py tests\test_qwen_two_condition_review.py
git diff --check
```

预期：两个命令退出码均为 0。

- [ ] **Step 3：对真实 7+17 输入执行零 GPU prepare-only**

```powershell
$env:PYTHONPATH = 'C:\Users\20661\AppData\Local\Temp\codex_qwen_review_py_alias_20260901'
$outputDir = 'C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts\qwen_two_condition_review_preflight_20260901'
.\.venv\Scripts\python.exe -m egolife_two_user_qa.tools.run_qwen_two_condition_review `
  --approved-markdown 'C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts\six_user_qa_10min_16628910_snapshot_10qa_20260901\QA.md' `
  --curated-jsonl 'C:\Users\20661\Documents\xwechat_files\wxid_i096w25uhusk22_e748\msg\file\2026-09\qa_curated_17_trace_review_v3.jsonl' `
  --media-root 'C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts\six_user_qa_10min_16628910_snapshot_10qa_20260901\stitched' `
  --output-dir $outputDir `
  --prepare-only
```

预期标准输出：

```text
SELECTED_COUNT=21
MEDIA_READY_COUNT=7
OUTPUT_DIR=C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts\qwen_two_condition_review_preflight_20260901
```

- [ ] **Step 4：直接读取产物验证合同，不依赖控制台摘要**

```powershell
$outputDir = 'C:\Users\20661\Desktop\Research\AR\multiuser\review_artifacts\qwen_two_condition_review_preflight_20260901'
$selectionCount = @(Get-Content -LiteralPath (Join-Path $outputDir 'selection.jsonl') -Encoding UTF8 | Where-Object { $_.Trim() }).Count
$dedup = Get-Content -LiteralPath (Join-Path $outputDir 'deduplication_report.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$media = Get-Content -LiteralPath (Join-Path $outputDir 'media_preflight.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$manifest = Get-Content -LiteralPath (Join-Path $outputDir 'run_manifest.json') -Raw -Encoding UTF8 | ConvertFrom-Json
"SELECTION_COUNT=$selectionCount"
"REMOVED_COUNT=$($dedup.removed_count)"
"MEDIA_READY_COUNT=$($media.media_ready_count)"
"MISSING_MEDIA_COUNT=$($media.missing_media_count)"
"MODE=$($manifest.mode)"
```

预期：

```text
SELECTION_COUNT=21
REMOVED_COUNT=3
MEDIA_READY_COUNT=7
MISSING_MEDIA_COUNT=14
MODE=prepare_only
```

- [ ] **Step 5：核对 Git 边界与未提交状态**

```powershell
git status --short
git log -5 --oneline --decorate
```

确认本任务只新增或修改计划列出的三个代码/测试文件；既有 `cli.py`、`video_qa_loop.py`、prompt、schema、renderer、测试和 `.sbatch` dirty changes 保持原状。不得 reset、clean、stash-all、merge 或 push。

## 本地完成标准

只有同时具备以下新鲜证据，才能称本地评审入口完成：

- `tests/test_qwen_two_condition_review.py` 全部通过；
- `tests/test_qwen_runner_compat.py` 的 16 个现有测试通过；
- 三个目标 Python 文件语法编译通过；
- `git diff --check` 通过；
- 真实输入 prepare-only 得到 21 条唯一 gold、3 条去重、7 条媒体完整、14 条缺少媒体；
- 产物中不存在模型预测，且 `run_manifest.json` 明确记录 `mode=prepare_only`；
- 没有连接 Torch、上传文件、提交或取消 Slurm 作业。

本地完成后，远端阶段需要重新核验 7 个缺失视频组的真实媒体映射和当前 Torch 状态，读取权威文档并取得用户明确执行授权。不得根据本地 prepare-only 结果声称 Qwen 已运行、准确率已产生或科学结论成立。
