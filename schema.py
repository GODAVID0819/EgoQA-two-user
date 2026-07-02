"""Schema validation and JSON parsing for generated QA items."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from .io_utils import iter_jsonl


OPTION_LETTERS = ("A", "B", "C", "D", "E")
REQUIRED_QA_FIELDS = {
    "qa_id",
    "question",
    "options",
    "correct",
    "answer",
    "category",
    "required_users",
    "evidence",
    "single_user_answerability",
    "combined_answerability",
    "review",
    "model_id",
    "source_urls",
}

VIDEO_FIRST_REQUIRED_FIELDS = {
    "question_type",
    "generator_rationale",
    "why_two_users_needed",
    "per_user_evidence_claims",
    "attempt_count",
    "video_evidence",
    "referred_timestamps",
    "human_audit",
    "generation_trace",
}
VIDEO_FIRST_JUDGE_CHECKS = {
    "qa_formality",
    "evidence_groundedness",
}


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
        if match:
            cleaned = match.group(1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output")
    return json.loads(cleaned[start : end + 1])


def normalize_correct(value: Any) -> str:
    if isinstance(value, int):
        if 0 <= value < 5:
            return OPTION_LETTERS[value]
    text = str(value).strip().upper()
    if text in OPTION_LETTERS:
        return text
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1)
    raise ValueError(f"Invalid correct option: {value!r}")


def validate_qa_item(item: dict[str, Any], *, strict_review: bool = False) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_QA_FIELDS - set(item))
    if missing:
        errors.append(f"missing fields: {', '.join(missing)}")
    if strict_review:
        missing_video = sorted(VIDEO_FIRST_REQUIRED_FIELDS - set(item))
        if missing_video:
            errors.append(f"missing video-first fields: {', '.join(missing_video)}")

    options = item.get("options")
    if not isinstance(options, list) or len(options) != 5:
        errors.append("options must contain exactly five entries")
    elif any(not isinstance(option, str) or not option.strip() for option in options):
        errors.append("all options must be non-empty strings")

    try:
        correct = normalize_correct(item.get("correct"))
        item["correct"] = correct
        if isinstance(options, list) and len(options) == 5:
            answer = item.get("answer")
            expected_answer = options[OPTION_LETTERS.index(correct)]
            if answer != expected_answer:
                errors.append("answer must equal options[correct]")
    except ValueError as exc:
        errors.append(str(exc))

    required_users = item.get("required_users")
    if not isinstance(required_users, list) or len(required_users) < 2:
        errors.append("required_users must list at least two users")
    elif len(set(required_users)) != len(required_users):
        errors.append("required_users must not contain duplicates")

    single = item.get("single_user_answerability")
    if not isinstance(single, dict):
        errors.append("single_user_answerability must be an object")
    elif required_users:
        missing_single = [user for user in required_users if user not in single]
        if missing_single:
            errors.append(f"single_user_answerability missing required users: {missing_single}")
        asker_user = required_users[0]
        asker_text = str(single.get(asker_user, "")).lower()
        if "insufficient" not in asker_text and "cannot" not in asker_text and "not enough" not in asker_text:
            errors.append(f"asker/speaker user {asker_user} must be marked insufficient")

    combined = str(item.get("combined_answerability", "")).lower()
    if "sufficient" not in combined and "support" not in combined:
        errors.append("combined_answerability must state sufficient support")

    question_type = item.get("question_type")
    if question_type is not None and question_type not in {"commonality", "difference"}:
        errors.append("question_type must be commonality or difference")

    if strict_review:
        video_evidence = item.get("video_evidence")
        if not isinstance(video_evidence, list) or not video_evidence:
            errors.append("video_evidence must be a non-empty list in strict mode")
        generation_trace = item.get("generation_trace")
        if not isinstance(generation_trace, list) or not generation_trace:
            errors.append("generation_trace must be a non-empty list in strict mode")
        human_audit = item.get("human_audit")
        if not isinstance(human_audit, dict):
            errors.append("human_audit must be an object in strict mode")

    review = item.get("review")
    if not isinstance(review, dict):
        errors.append("review must be an object")
    elif strict_review:
        if review.get("review_passed") is not True or review.get("status") != "passed":
            errors.append("review must pass in strict mode")

        judge = review.get("judger")
        if not isinstance(judge, dict):
            errors.append("review.judger must be an object in strict mode")
        else:
            gate = judge.get("gate")
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                errors.append("review.judger.gate.passed must be true in strict mode")
            checks = judge.get("checks")
            if not isinstance(checks, dict):
                errors.append("review.judger.checks must be an object in strict mode")
            else:
                missing_checks = sorted(VIDEO_FIRST_JUDGE_CHECKS - set(checks))
                if missing_checks:
                    errors.append(f"review.judger.checks missing: {', '.join(missing_checks)}")
                for check_name, check_value in checks.items():
                    if not isinstance(check_value, dict):
                        errors.append(f"judge check {check_name} must be an object")
                        continue
                    if str(check_value.get("status", "")).upper() != "PASS":
                        errors.append(f"judge check {check_name} must be PASS in strict mode")

        answerability = review.get("answerability")
        if not isinstance(answerability, dict):
            errors.append("review.answerability must be an object in strict mode")
        else:
            gate = answerability.get("gate")
            evaluations = answerability.get("evaluations")
            if not isinstance(gate, dict) or gate.get("passed") is not True:
                errors.append("review.answerability.gate.passed must be true in strict mode")
            if not isinstance(evaluations, list):
                errors.append("review.answerability.evaluations must be a list in strict mode")

        schema_validation = review.get("schema_validation")
        if not isinstance(schema_validation, dict):
            errors.append("review.schema_validation must be an object in strict mode")
        elif schema_validation.get("passed") is not True:
            errors.append("review.schema_validation.passed must be true in strict mode")

    return errors


def load_and_validate(path: str | Path, *, strict_review: bool = False) -> tuple[int, list[str]]:
    errors = []
    count = 0
    for count, item in enumerate(iter_jsonl(path), 1):
        row_errors = validate_qa_item(item, strict_review=strict_review)
        errors.extend([f"row {count}: {error}" for error in row_errors])
    return count, errors


def write_qa_csv(jsonl_path: str | Path, csv_path: str | Path) -> int:
    rows = list(iter_jsonl(jsonl_path))
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "qa_id",
        "question",
        "correct",
        "answer",
        "category",
        "required_users",
        "combined_answerability",
        "review_passed",
        "question_type",
        "attempt_count",
        "answerability_passed",
        "review",
        "video_evidence",
        "referred_timestamps",
        "human_audit",
        "generation_trace",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            review = row.get("review") if isinstance(row.get("review"), dict) else {}
            writer.writerow(
                {
                    "qa_id": row.get("qa_id", ""),
                    "question": row.get("question", ""),
                    "correct": row.get("correct", ""),
                    "answer": row.get("answer", ""),
                    "category": row.get("category", ""),
                    "required_users": ";".join(row.get("required_users", [])),
                    "combined_answerability": row.get("combined_answerability", ""),
                    "review_passed": review.get("review_passed", review.get("status", "")),
                    "question_type": row.get("question_type", ""),
                    "attempt_count": row.get("attempt_count", ""),
                    "answerability_passed": (
                        review.get("answerability", {}).get("gate", {}).get("passed", "")
                        if isinstance(review.get("answerability"), dict)
                        else ""
                    ),
                    "review": json.dumps(review, ensure_ascii=False),
                    "video_evidence": json.dumps(row.get("video_evidence", []), ensure_ascii=False),
                    "referred_timestamps": json.dumps(row.get("referred_timestamps", []), ensure_ascii=False),
                    "human_audit": json.dumps(row.get("human_audit", {}), ensure_ascii=False),
                    "generation_trace": json.dumps(row.get("generation_trace", []), ensure_ascii=False),
                }
            )
    return len(rows)


def _markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def write_human_review_sheet(jsonl_path: str | Path, sheet_path: str | Path) -> int:
    rows = list(iter_jsonl(jsonl_path))
    sheet_path = Path(sheet_path)
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# EgoLife Human Review Sheet",
        "",
        f"- Source QA: `{jsonl_path}`",
        f"- Accepted rows: {len(rows)}",
        "",
    ]
    for index, row in enumerate(rows, 1):
        review = row.get("review") if isinstance(row.get("review"), dict) else {}
        final = review.get("final_decision") if isinstance(review.get("final_decision"), dict) else {}
        answerability = review.get("answerability") if isinstance(review.get("answerability"), dict) else {}
        judger = review.get("judger") if isinstance(review.get("judger"), dict) else {}
        lines.extend(
            [
                f"## {index}. {row.get('qa_id', '')}",
                "",
                f"- Evidence ID: `{row.get('evidence_id', '')}`",
                f"- Question type: {_markdown_value(row.get('question_type'))}",
                f"- Required users: {', '.join(row.get('required_users', []))}",
                f"- Review status: {_markdown_value(review.get('status'))}",
                f"- Review passed: {_markdown_value(review.get('review_passed'))}",
                f"- Judger gate passed: {_markdown_value((judger.get('gate') or {}).get('passed') if isinstance(judger.get('gate'), dict) else '')}",
                f"- Answerability gate passed: {_markdown_value((answerability.get('gate') or {}).get('passed') if isinstance(answerability.get('gate'), dict) else '')}",
                f"- Answerability warning: {_markdown_value((answerability.get('gate') or {}).get('warning') if isinstance(answerability.get('gate'), dict) else '')}",
                f"- Final reason: {_markdown_value(final.get('reason'))}",
                "",
                "### Question",
                "",
                _markdown_value(row.get("question")),
                "",
                "### Options",
                "",
            ]
        )
        options = row.get("options") if isinstance(row.get("options"), list) else []
        for letter, option in zip(OPTION_LETTERS, options):
            marker = " (correct)" if row.get("correct") == letter else ""
            lines.append(f"- {letter}. {option}{marker}")
        lines.extend(
            [
                "",
                "### Answer",
                "",
                _markdown_value(row.get("answer")),
                "",
                "### Why Two Users Are Needed",
                "",
                _markdown_value(row.get("why_two_users_needed")),
                "",
                "### Combined Answerability",
                "",
                _markdown_value(row.get("combined_answerability")),
                "",
                "### Per-User Evidence",
                "",
            ]
        )
        evidence = row.get("evidence") if isinstance(row.get("evidence"), list) else []
        if evidence:
            for item in evidence:
                user = item.get("user", "") if isinstance(item, dict) else ""
                fact = item.get("needed_fact", item.get("claim", "")) if isinstance(item, dict) else item
                timeframe = item.get("timeframe", "") if isinstance(item, dict) else ""
                lines.append(f"- {user}: {_markdown_value(fact)} ({_markdown_value(timeframe)})")
        else:
            lines.append("- none")
        lines.extend(["", "### Video Evidence", ""])
        video_evidence = row.get("video_evidence") if isinstance(row.get("video_evidence"), list) else []
        if video_evidence:
            for item in video_evidence:
                if not isinstance(item, dict):
                    lines.append(f"- {_markdown_value(item)}")
                    continue
                user = item.get("user", "")
                local_video = item.get("local_video") or item.get("video_path") or ""
                video_url = item.get("video_url", "")
                clip_clock = item.get("clip_clock", "")
                lines.append(
                    f"- {user}: `{local_video}` clock={_markdown_value(clip_clock)} url={_markdown_value(video_url)}"
                )
        else:
            lines.append("- none")
        lines.append("")
    sheet_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(rows)
