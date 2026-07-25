"""从历史 intermediate JSONL 中恢复每一次 generator attempt。"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Iterable

try:
    from egolife_two_user_qa.schema import extract_json_object, validate_qa_item
except ModuleNotFoundError:
    package_name = "_egoqa_repo_v0"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(Path(__file__).resolve().parents[3])]
        package.__package__ = package_name
        sys.modules[package_name] = package
    schema = importlib.import_module(f"{package_name}.schema")
    extract_json_object = schema.extract_json_object
    validate_qa_item = schema.validate_qa_item

from .records import AttemptRecord


def _string_tuple(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(str(item) for item in value if item is not None)


def _parse_generation(generation: dict[str, Any], raw_qa: str) -> dict[str, Any] | None:
    parsed = generation.get("parsed_qa")
    if not isinstance(parsed, dict):
        parsed = generation.get("normalized_qa")
    if isinstance(parsed, dict):
        return parsed
    try:
        return extract_json_object(raw_qa)
    except (ValueError, json.JSONDecodeError):
        return None


def _schema_errors(source: dict[str, Any], parsed_qa: dict[str, Any] | None) -> tuple[str, ...]:
    judge = source.get("judge")
    branch = judge.get("schema_branch") if isinstance(judge, dict) else None
    if isinstance(branch, dict) and str(branch.get("status") or "").upper() in {"PASS", "FAIL"}:
        errors = branch.get("errors")
        if isinstance(errors, list):
            return tuple(str(error) for error in errors)
        return () if str(branch.get("status")).upper() == "PASS" else ("historical schema branch failed",)
    return tuple(validate_qa_item(dict(parsed_qa))) if parsed_qa is not None else ()


def extract_packet_attempts(packet: dict[str, Any]) -> list[AttemptRecord]:
    evidence_id = str(packet.get("evidence_id") or "").strip()
    if not evidence_id:
        raise ValueError("packet missing evidence_id")
    attempts = packet.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"packet {evidence_id} has no attempts")
    trace_rows = packet.get("generation_trace")
    trace_by_index = {
        row.get("attempt"): row
        for row in trace_rows
        if isinstance(row, dict) and isinstance(row.get("attempt"), int)
    } if isinstance(trace_rows, list) else {}

    records: list[AttemptRecord] = []
    for position, attempt in enumerate(attempts, start=1):
        if not isinstance(attempt, dict):
            raise ValueError(f"packet {evidence_id} attempt {position} is not an object")
        attempt_index = attempt.get("attempt", position)
        if not isinstance(attempt_index, int):
            raise ValueError(f"packet {evidence_id} attempt {position} has invalid index")
        source = attempt
        if not isinstance(attempt.get("generation"), dict):
            source = trace_by_index.get(attempt_index, attempt)
        generation = source.get("generation")
        if not isinstance(generation, dict):
            generation = {}
        raw_qa = str(generation.get("raw_output") or "")
        if not raw_qa.strip():
            raise ValueError(f"packet {evidence_id} attempt {attempt_index} has empty raw generation")
        parsed_qa = _parse_generation(generation, raw_qa)
        schema_errors = _schema_errors(source, parsed_qa)
        media = source.get("media") if isinstance(source.get("media"), dict) else {}
        result = source.get("result") if isinstance(source.get("result"), dict) else {}
        judge = source.get("judge")
        answerability = source.get("answerability")
        records.append(
            AttemptRecord(
                attempt_id=f"{evidence_id}::attempt::{attempt_index}",
                evidence_id=evidence_id,
                packet_status=str(packet.get("status") or ""),
                question_type=source.get("question_type") or packet.get("question_type"),
                mode=source.get("generation_mode") or packet.get("generation_mode"),
                attempt_index=attempt_index,
                feedback=str(source.get("feedback_in") or ""),
                generator_prompt=str(generation.get("prompt") or ""),
                generator_image_paths=_string_tuple(media.get("image_paths")),
                generator_video_paths=_string_tuple(media.get("video_paths")),
                evaluator_image_paths=_string_tuple(media.get("full_image_paths")),
                evaluator_video_paths=_string_tuple(media.get("full_video_paths")),
                raw_qa=raw_qa,
                parsed_qa=parsed_qa,
                schema_errors=schema_errors,
                judge=judge if isinstance(judge, dict) else None,
                answerability=answerability if isinstance(answerability, dict) else None,
                accepted=result.get("accepted") is True,
            )
        )
    return records


def iter_attempt_records(path: str | Path) -> Iterable[AttemptRecord]:
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"intermediate JSONL not found: {input_path}")
    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                packet = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            if not isinstance(packet, dict):
                raise ValueError(f"line {line_number} is not a packet object")
            try:
                yield from extract_packet_attempts(packet)
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
