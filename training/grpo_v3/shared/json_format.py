"""GRPO v3 completion 的严格 JSON 校验与保守语法修复。"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal


FormatStatus = Literal["raw_valid", "repaired", "unrecoverable"]
MAX_REPAIR_OPERATIONS = 3
REPAIRED_FORMAT_PENALTY = -0.5
UNRECOVERABLE_FORMAT_REWARD = -3.0
FORMAT_REWARD_REVISION = "json_three_tier_v1"


@dataclass(frozen=True)
class FormatValidationResult:
    status: FormatStatus
    value: dict[str, Any] | None
    raw_completion: str
    repaired_completion: str | None
    repair_operations: list[dict[str, Any]]
    semantic_text_changed: bool
    format_penalty: float
    parse_error: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "raw_completion": self.raw_completion,
            "repaired_completion": self.repaired_completion,
            "repair_operations": self.repair_operations,
            "semantic_text_changed": self.semantic_text_changed,
            "format_penalty": self.format_penalty,
            "parse_error": self.parse_error,
        }


def _decode_error(exc: json.JSONDecodeError, *, reason: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": type(exc).__name__,
        "message": exc.msg,
        "lineno": exc.lineno,
        "colno": exc.colno,
        "pos": exc.pos,
    }
    if reason is not None:
        result["reason"] = reason
    return result


def _reason_error(reason: str, *, message: str | None = None) -> dict[str, Any]:
    return {
        "type": "FormatValidationError",
        "message": message or reason,
        "reason": reason,
    }


def _unrecoverable(
    raw: str,
    error: dict[str, Any],
) -> FormatValidationResult:
    return FormatValidationResult(
        status="unrecoverable",
        value=None,
        raw_completion=raw,
        repaired_completion=None,
        repair_operations=[],
        semantic_text_changed=False,
        format_penalty=UNRECOVERABLE_FORMAT_REWARD,
        parse_error=error,
    )


def _strip_bounds(text: str) -> tuple[str, list[int]]:
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    return text[start:end], list(range(start, end))


_FENCE_PATTERN = re.compile(
    r"\A\s*```(?:json)?[ \t]*(?:\r?\n)?(?P<body>.*?)(?:\r?\n)?```\s*\Z",
    re.IGNORECASE | re.DOTALL,
)


def _strip_complete_fence(text: str) -> tuple[str, list[int]] | None:
    match = _FENCE_PATTERN.fullmatch(text)
    if match is None:
        return None
    body = match.group("body")
    body_start = match.start("body")
    leading = len(body) - len(body.lstrip())
    body = body.strip()
    start = body_start + leading
    return body, list(range(start, start + len(body)))


def _scan_string_values(text: str) -> tuple[list[str], bool]:
    values: list[str] = []
    index = 0
    while index < len(text):
        if text[index] != '"':
            index += 1
            continue
        start = index
        index += 1
        escaped = False
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                token = text[start:index + 1]
                try:
                    values.append(json.loads(token))
                except json.JSONDecodeError:
                    return values, False
                index += 1
                break
            index += 1
        else:
            return values, False
    return values, True


def _trailing_comma_positions(text: str) -> list[int]:
    positions: list[int] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        elif char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                positions.append(index)
        index += 1
    return positions


def _container_stack_at(text: str, stop: int) -> list[str] | None:
    stack: list[str] = []
    in_string = False
    escaped = False
    for char in text[:stop]:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]":
            expected = "{" if char == "}" else "["
            if not stack or stack[-1] != expected:
                return None
            stack.pop()
    if in_string:
        return None
    return stack


def _looks_like_object_member_key(text: str, position: int) -> bool:
    index = position
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != '"':
        return False
    try:
        key, consumed = json.JSONDecoder().raw_decode(text[index:])
    except json.JSONDecodeError:
        return False
    if not isinstance(key, str):
        return False
    after = index + consumed
    while after < len(text) and text[after].isspace():
        after += 1
    return after < len(text) and text[after] == ":"


def _missing_member_comma_position(text: str, exc: json.JSONDecodeError) -> int | None:
    if "Expecting ',' delimiter" not in exc.msg:
        return None
    position = exc.pos
    while position < len(text) and text[position].isspace():
        position += 1
    if not _looks_like_object_member_key(text, position):
        return None
    stack = _container_stack_at(text, position)
    if not stack or stack[-1] != "{":
        return None
    previous = position - 1
    while previous >= 0 and text[previous].isspace():
        previous -= 1
    if previous < 0 or text[previous] not in '\"}]0123456789eElL':
        return None
    return position


def validate_completion_json(
    raw_completion: str,
    *,
    max_operations: int = MAX_REPAIR_OPERATIONS,
) -> FormatValidationResult:
    """严格解析 completion；仅用批准的纯语法操作做最多三次修复。"""

    raw = str(raw_completion)
    strict_text = raw.strip()
    try:
        strict_value = json.loads(strict_text)
    except json.JSONDecodeError as strict_exc:
        strict_error = _decode_error(strict_exc)
    else:
        if not isinstance(strict_value, dict):
            return _unrecoverable(raw, _reason_error("top_level_json_must_be_object"))
        return FormatValidationResult(
            status="raw_valid",
            value=strict_value,
            raw_completion=raw,
            repaired_completion=None,
            repair_operations=[],
            semantic_text_changed=False,
            format_penalty=0.0,
            parse_error=None,
        )

    fenced = _strip_complete_fence(raw)
    operations: list[dict[str, Any]] = []
    if fenced is None:
        text, origin_map = _strip_bounds(raw)
    else:
        text, origin_map = fenced
        operations.append({"operation": "strip_markdown_fence", "position": 0})

    original_string_values, strings_closed = _scan_string_values(text)
    if not strings_closed:
        return _unrecoverable(raw, strict_error)

    trailing = _trailing_comma_positions(text)
    if len(operations) + len(trailing) > max_operations:
        return _unrecoverable(raw, _reason_error("repair_operation_limit_exceeded"))
    for position in trailing:
        operations.append({
            "operation": "remove_trailing_comma",
            "position": origin_map[position],
        })
    for position in reversed(trailing):
        text = text[:position] + text[position + 1:]
        del origin_map[position]

    while True:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            insertion = _missing_member_comma_position(text, exc)
            if insertion is None:
                return _unrecoverable(raw, strict_error)
            if len(operations) >= max_operations:
                return _unrecoverable(raw, _reason_error("repair_operation_limit_exceeded"))
            original_position = origin_map[insertion] if insertion < len(origin_map) else len(raw)
            operations.append({
                "operation": "insert_missing_member_comma",
                "position": original_position,
            })
            text = text[:insertion] + "," + text[insertion:]
            origin_map.insert(insertion, original_position)
            continue
        break

    if not isinstance(value, dict):
        return _unrecoverable(raw, _reason_error("top_level_json_must_be_object"))
    if not operations:
        return _unrecoverable(raw, strict_error)
    repaired_string_values, repaired_strings_closed = _scan_string_values(text)
    if not repaired_strings_closed or repaired_string_values != original_string_values:
        return _unrecoverable(raw, _reason_error("semantic_string_tokens_changed"))
    return FormatValidationResult(
        status="repaired",
        value=value,
        raw_completion=raw,
        repaired_completion=text,
        repair_operations=operations,
        semantic_text_changed=False,
        format_penalty=REPAIRED_FORMAT_PENALTY,
        parse_error=strict_error,
    )


def summarize_format_traces(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """按统一口径聚合 reward trace 中的三层格式状态。"""

    counts = Counter({"raw_valid": 0, "repaired": 0, "unrecoverable": 0})
    operation_counts: Counter[str] = Counter()
    for trace in traces:
        record = trace.get("record") if isinstance(trace.get("record"), dict) else {}
        validation = (
            record.get("format_validation")
            if isinstance(record.get("format_validation"), dict)
            else {}
        )
        status = validation.get("status")
        if status in counts:
            counts[str(status)] += 1
        operations = validation.get("repair_operations")
        if isinstance(operations, list):
            for operation in operations:
                if isinstance(operation, dict) and operation.get("operation"):
                    operation_counts[str(operation["operation"])] += 1
    denominator = len(traces)
    rate = lambda count: count / denominator if denominator else 0.0
    return {
        "format_raw_valid_count": counts["raw_valid"],
        "format_repaired_count": counts["repaired"],
        "format_unrecoverable_count": counts["unrecoverable"],
        "format_raw_valid_rate": rate(counts["raw_valid"]),
        "format_repaired_rate": rate(counts["repaired"]),
        "format_unrecoverable_rate": rate(counts["unrecoverable"]),
        "format_repair_operation_counts": dict(sorted(operation_counts.items())),
        "format_repaired_penalty": REPAIRED_FORMAT_PENALTY,
        "format_unrecoverable_reward": UNRECOVERABLE_FORMAT_REWARD,
        "format_reward_revision": FORMAT_REWARD_REVISION,
    }
