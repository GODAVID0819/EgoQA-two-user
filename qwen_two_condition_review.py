"""Qwen minimum-set 与 all-six 成对视频 QA 评审核心逻辑。"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

SIX_USERS = ("Jake", "Alice", "Tasha", "Lucia", "Katrina", "Shure")
CHOICES = ("A", "B", "C", "D", "E")
SOURCE_PRIORITY = {"approved_markdown": 1, "curated_trace_v3": 2}
MINIMUM_SET_CONDITION = "minimum_set"
ALL_SIX_CONDITION = "all_six"


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


class ReviewExecutionError(RuntimeError):
    """模型或媒体运行异常已写入预测记录。"""


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


def generation_group_directory(generation_group_id: str) -> str:
    if not re.fullmatch(r"DAY\d+::\d+", generation_group_id):
        raise ValueError("invalid generation_group_id")
    return generation_group_id.replace("::", "_")


def build_condition_specs(
    item: GoldItem,
    media_root: str | Path,
) -> tuple[ConditionSpec, ...]:
    group_root = Path(media_root) / generation_group_directory(
        item.generation_group_id
    )
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
        f"{choice}. {option}"
        for choice, option in zip(CHOICES, options, strict=True)
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
    fallback = re.fullmatch(
        r"\s*(?:\(([A-E])\)|([A-E])\.?)\s*",
        raw_output,
        re.IGNORECASE,
    )
    if fallback is None:
        return ChoiceParse(None, "invalid_missing")
    return ChoiceParse(
        (fallback.group(1) or fallback.group(2)).upper(),
        "valid",
    )


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def read_prediction_rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        return []
    return [
        json.loads(line)
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _latest_by_key(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        latest[(str(row["qa_id"]), str(row["condition_id"]))] = row
    return latest


def _next_attempt(
    rows: Iterable[dict[str, Any]],
    qa_id: str,
    condition_id: str,
) -> int:
    attempts = [
        int(row.get("attempt", 0))
        for row in rows
        if row.get("qa_id") == qa_id
        and row.get("condition_id") == condition_id
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
        for order, spec in enumerate(
            build_condition_specs(item, media_root),
            1,
        ):
            if spec.missing_paths:
                continue
            key = (item.qa_id, spec.condition_id)
            prior = latest.get(key)
            if prior is not None:
                prior_valid = (
                    prior.get("run_status") == "ok"
                    and prior.get("parse_status") == "valid"
                )
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
                "is_correct": (
                    parsed.choice == item.correct
                    if parsed.status == "valid"
                    else None
                ),
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def prepare_review(
    items: Sequence[GoldItem],
    media_root: str | Path,
    output_dir: str | Path,
) -> DeduplicationResult:
    output_root = Path(output_dir)
    result = deduplicate_items(items)
    _write_jsonl(
        output_root / "selection.jsonl",
        (item_to_dict(item) for item in result.items),
    )
    _write_json(
        output_root / "deduplication_report.json",
        {
            "input_count": len(items),
            "selected_count": len(result.items),
            "removed_count": len(result.removed),
            "removed": list(result.removed),
            "same_group_nonduplicates": list(
                result.same_group_nonduplicates
            ),
        },
    )
    rows: list[dict[str, Any]] = []
    for item in result.items:
        specs = build_condition_specs(item, media_root)
        missing = sorted(
            {path for spec in specs for path in spec.missing_paths}
        )
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
            "media_ready_count": sum(
                bool(row["media_ready"]) for row in rows
            ),
            "missing_media_count": sum(
                not bool(row["media_ready"]) for row in rows
            ),
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
        valid_minimum = (
            minimum is not None
            and minimum.get("run_status") == "ok"
            and minimum.get("parse_status") == "valid"
        )
        valid_all_six = (
            all_six is not None
            and all_six.get("run_status") == "ok"
            and all_six.get("parse_status") == "valid"
        )
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
                "both_correct"
                if minimum_ok and all_six_ok
                else "minimum_only_correct"
                if minimum_ok
                else "all_six_only_correct"
                if all_six_ok
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
    accuracy_minimum = (
        minimum_correct / paired_count if paired_count else None
    )
    accuracy_all_six = (
        all_six_correct / paired_count if paired_count else None
    )
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
            if accuracy_minimum is not None
            and accuracy_all_six is not None
            else None
        ),
        "pair_categories": {
            name: categories[name]
            for name in (
                "both_correct",
                "both_wrong",
                "minimum_only_correct",
                "all_six_only_correct",
            )
        },
        "parse_failure_count": sum(
            row.get("run_status") == "ok"
            and row.get("parse_status") != "valid"
            for row in latest.values()
        ),
        "parse_failures_by_condition": {
            condition_id: sum(
                row.get("condition_id") == condition_id
                and row.get("run_status") == "ok"
                and row.get("parse_status") != "valid"
                for row in latest.values()
            )
            for condition_id in (
                MINIMUM_SET_CONDITION,
                ALL_SIX_CONDITION,
            )
        },
        "inference_error_count": sum(
            row.get("run_status") == "error"
            for row in latest.values()
        ),
        "inference_errors_by_condition": {
            condition_id: sum(
                row.get("condition_id") == condition_id
                and row.get("run_status") == "error"
                for row in latest.values()
            )
            for condition_id in (
                MINIMUM_SET_CONDITION,
                ALL_SIX_CONDITION,
            )
        },
        "elapsed_seconds_total": sum(
            float(row.get("elapsed_seconds", 0.0))
            for row in latest.values()
        ),
    }
    return paired_rows, summary


def render_cn_report(
    paired_rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> str:
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
        (
            "- Minimum set 准确率："
            f"**{percent(summary['accuracy_minimum'])}**"
        ),
        (
            "- All six 准确率："
            f"**{percent(summary['accuracy_all_six'])}**"
        ),
        (
            "- All six 减 minimum set："
            f"**{percent(summary['accuracy_delta'])}**"
        ),
        (
            "- 两个条件都正确："
            f"**{summary['pair_categories']['both_correct']}**"
        ),
        (
            "- 两个条件都错误："
            f"**{summary['pair_categories']['both_wrong']}**"
        ),
        (
            "- 仅 minimum set 正确："
            f"**{summary['pair_categories']['minimum_only_correct']}**"
        ),
        (
            "- 仅 all six 正确："
            f"**{summary['pair_categories']['all_six_only_correct']}**"
        ),
        f"- 解析失败：**{summary['parse_failure_count']}**",
        (
            "- Minimum set 解析失败："
            f"**{summary['parse_failures_by_condition']['minimum_set']}**"
        ),
        (
            "- All six 解析失败："
            f"**{summary['parse_failures_by_condition']['all_six']}**"
        ),
        f"- 推理异常：**{summary['inference_error_count']}**",
        "",
        (
            "准确率只使用两个条件均成功且解析有效的同一批 QA。"
            "一次小样本运行不证明差异稳定或具有统计显著性。"
        ),
        "",
        "## 逐题结果",
    ]
    for row in paired_rows:
        minimum = row["minimum_set"] or {}
        all_six = row["all_six"] or {}

        def predicted(value: dict[str, Any]) -> str:
            return str(value.get("predicted_choice") or "未产生")

        def users(value: dict[str, Any]) -> str:
            names = value.get("input_users") or []
            return "、".join(str(name) for name in names) or "未运行"

        def elapsed(value: dict[str, Any]) -> str:
            seconds = value.get("elapsed_seconds")
            return "不可用" if seconds is None else f"{float(seconds):.3f} 秒"

        def parse_status(value: dict[str, Any]) -> str:
            return str(value.get("parse_status") or "未运行")

        lines.extend(
            [
                "",
                f"### {row['qa_id']}",
                "",
                f"- 问题：{row['question']}",
                f"- 正确选项：{row['correct_choice']}",
                f"- 配对有效：{'是' if row['paired_valid'] else '否'}",
                f"- 分类：{row['pair_category'] or row['unpaired_reason']}",
                f"- Minimum set 预测：{predicted(minimum)}",
                f"- Minimum set 解析：{parse_status(minimum)}",
                f"- Minimum set 用户：{users(minimum)}",
                f"- Minimum set 耗时：{elapsed(minimum)}",
                f"- All six 预测：{predicted(all_six)}",
                f"- All six 解析：{parse_status(all_six)}",
                f"- All six 用户：{users(all_six)}",
                f"- All six 耗时：{elapsed(all_six)}",
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
