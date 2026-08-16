"""Question-only multiple-choice benchmark with balanced answer-position rotations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .answerability_verification_benchmark import fetch_openrouter_model_metadata
from .io_utils import append_jsonl, iter_jsonl, read_json, write_json, write_jsonl
from .qwen3vl_runner import (
    DEFAULT_OPENROUTER_BASE_URL,
    OPENROUTER_REASONING_EFFORTS,
    OpenRouterRunner,
)
from .schema import OPTION_LETTERS, extract_json_object, normalize_correct


CONFIG_VERSION = 1
LETTERS = tuple(OPTION_LETTERS)
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_TIMEOUT_SECONDS = 600


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolved_path(base_dir: Path, value: Any, *, label: str) -> Path:
    raw = os.path.expandvars(os.path.expanduser(str(value or "").strip()))
    if not raw:
        raise ValueError(f"missing {label}")
    if "$" in raw or "%" in raw:
        raise ValueError(f"{label} contains an unresolved environment variable: {raw!r}")
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(
    config_path: str | Path,
    *,
    annotations_override: str | Path | None = None,
    output_dir_override: str | Path | None = None,
) -> dict[str, Any]:
    """Load and validate a question-only rotation benchmark config."""

    config_path = Path(config_path).resolve()
    raw = read_json(config_path)
    if not isinstance(raw, dict):
        raise ValueError("question-only benchmark config must be a JSON object")
    if int(raw.get("version") or 0) != CONFIG_VERSION:
        raise ValueError(f"config version must be {CONFIG_VERSION}")
    base_dir = config_path.parent
    annotations = _resolved_path(
        base_dir,
        annotations_override or raw.get("annotations"),
        label="annotations",
    )
    accepted_qa = _resolved_path(base_dir, raw.get("accepted_qa"), label="accepted_qa")
    output_dir = _resolved_path(
        base_dir,
        output_dir_override or raw.get("output_dir"),
        label="output_dir",
    )
    if not annotations.is_file():
        raise ValueError(f"annotations file does not exist: {annotations}")
    if not accepted_qa.is_file():
        raise ValueError(f"accepted QA file does not exist: {accepted_qa}")

    expected_qa_count = int(raw.get("expected_qa_count") or 0)
    if expected_qa_count <= 0:
        raise ValueError("expected_qa_count must be positive")
    expected_qa_ids = [str(value or "").strip() for value in raw.get("expected_qa_ids") or []]
    if len(expected_qa_ids) != expected_qa_count:
        raise ValueError("expected_qa_ids length must equal expected_qa_count")
    if any(not value for value in expected_qa_ids):
        raise ValueError("expected_qa_ids contains an empty QA ID")
    if len(set(expected_qa_ids)) != len(expected_qa_ids):
        raise ValueError("expected_qa_ids contains duplicates")

    correct_positions = tuple(
        normalize_correct(value) for value in (raw.get("correct_positions") or LETTERS)
    )
    if correct_positions != LETTERS:
        raise ValueError(
            f"correct_positions must be exactly {list(LETTERS)} in that order; "
            f"received {list(correct_positions)}"
        )
    rotation_method = str(raw.get("rotation_method") or "cyclic_latin_square").strip()
    if rotation_method != "cyclic_latin_square":
        raise ValueError("rotation_method must be cyclic_latin_square")

    model_id = str(raw.get("model_id") or "").strip()
    if not model_id:
        raise ValueError("model_id is required")
    reasoning_effort = str(raw.get("reasoning_effort") or "").strip()
    if reasoning_effort not in OPENROUTER_REASONING_EFFORTS:
        raise ValueError(f"unsupported reasoning_effort={reasoning_effort!r}")
    max_new_tokens = int(raw.get("max_new_tokens") or DEFAULT_MAX_NEW_TOKENS)
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    timeout_seconds = int(raw.get("timeout_seconds") or DEFAULT_TIMEOUT_SECONDS)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    max_retries = int(raw.get("max_retries", 0))
    if max_retries != 0:
        raise ValueError("max_retries must be 0 for this credit-capped experiment")

    return {
        "config_path": config_path,
        "annotations": annotations,
        "accepted_qa": accepted_qa,
        "output_dir": output_dir,
        "base_url": str(raw.get("base_url") or DEFAULT_OPENROUTER_BASE_URL).rstrip("/"),
        "manual_pass_value": str(raw.get("manual_pass_value") or "Pass").strip(),
        "expected_qa_count": expected_qa_count,
        "expected_qa_ids": expected_qa_ids,
        "correct_positions": correct_positions,
        "rotation_method": rotation_method,
        "model_id": model_id,
        "reasoning_effort": reasoning_effort,
        "max_new_tokens": max_new_tokens,
        "timeout_seconds": timeout_seconds,
        "max_retries": max_retries,
        "allow_effort_mapping": raw.get("allow_effort_mapping", False) is True,
    }


def _read_annotations(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"qa_id", "review_status"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"annotation CSV is missing columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    qa_ids = [str(row.get("qa_id") or "").strip() for row in rows]
    if any(not qa_id for qa_id in qa_ids):
        raise ValueError("annotation CSV contains an empty qa_id")
    duplicates = sorted(qa_id for qa_id, count in Counter(qa_ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"annotation CSV contains duplicate QA IDs: {duplicates}")
    return rows


def _read_qas(path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        qa_id = str(row.get("qa_id") or "").strip()
        if not qa_id:
            raise ValueError(f"accepted QA row is missing qa_id: {path}")
        if qa_id in indexed:
            raise ValueError(f"accepted QA file contains duplicate qa_id={qa_id}")
        indexed[qa_id] = row
    if not indexed:
        raise ValueError(f"accepted QA file contains no rows: {path}")
    return indexed


def rotate_options(
    options: list[str],
    *,
    original_correct_letter: str,
    target_correct_letter: str,
) -> dict[str, Any]:
    """Cyclically rotate all options so the correct option lands at the target letter."""

    if len(options) != len(LETTERS):
        raise ValueError("each QA must contain exactly five options")
    original_correct = LETTERS.index(normalize_correct(original_correct_letter))
    target_correct = LETTERS.index(normalize_correct(target_correct_letter))
    shift = (target_correct - original_correct) % len(LETTERS)
    rotated: list[str | None] = [None] * len(LETTERS)
    display_to_canonical: list[int | None] = [None] * len(LETTERS)
    canonical_to_display: list[int | None] = [None] * len(LETTERS)
    for canonical_index, option in enumerate(options):
        display_index = (canonical_index + shift) % len(LETTERS)
        rotated[display_index] = option
        display_to_canonical[display_index] = canonical_index
        canonical_to_display[canonical_index] = display_index
    if any(value is None for value in rotated + display_to_canonical + canonical_to_display):
        raise AssertionError("rotation did not produce a complete permutation")
    return {
        "options": [str(value) for value in rotated],
        "display_to_canonical_index": [int(value) for value in display_to_canonical],
        "canonical_to_display_index": [int(value) for value in canonical_to_display],
        "shift": shift,
    }


def build_question_only_prompt(question: str, options: list[str]) -> str:
    """Build the minimal text-only MCQ prompt used for all rotations."""

    option_lines = "\n".join(f"{letter}. {option}" for letter, option in zip(LETTERS, options))
    return (
        f"Question:\n{question.strip()}\n\n"
        f"Answer choices:\n{option_lines}\n\n"
        'Return exactly one JSON object with one key named "choice". '
        'Its value must be one of "A", "B", "C", "D", or "E". '
        "Do not explain your answer."
    )


def parse_question_only_choice(
    raw_output: str,
    *,
    display_options: list[str],
) -> tuple[str | None, str]:
    """Parse a displayed answer letter without mistaking answer text for a letter."""

    try:
        parsed = extract_json_object(raw_output)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("choice", "answer", "option"):
            if key not in parsed:
                continue
            value = str(parsed[key] or "").strip()
            if len(value) == 1 and value.upper() in LETTERS:
                return value.upper(), f"json_{key}"
            for display_index, option in enumerate(display_options):
                if value.casefold() == option.strip().casefold():
                    return LETTERS[display_index], f"json_{key}_text"
            prefixed = re.fullmatch(r"\s*([A-Ea-e])\s*[.)\]:-]\s*.+", value, re.DOTALL)
            if prefixed:
                return prefixed.group(1).upper(), f"json_{key}_prefixed"

    stripped = raw_output.strip()
    direct = re.fullmatch(r"[\s`*]*([A-Ea-e])(?:[\s.`*)]*)", stripped)
    if direct:
        return direct.group(1).upper(), "direct_letter"
    explicit = re.search(
        r"\b(?:final\s+answer|answer|choice|option)\s*(?:is|=|:)?\s*"
        r"[\[(\"']?([A-E])[\])\"']?.?\b",
        raw_output,
        re.IGNORECASE,
    )
    if explicit:
        return explicit.group(1).upper(), "explicit_answer_letter"
    answering = re.search(
        r"\banswering\s+[\"']?([A-E])[\"']?\b",
        raw_output,
        re.IGNORECASE,
    )
    if answering:
        return answering.group(1).upper(), "explicit_answering_letter"
    return None, "unparsed"


def _reparse_call_result(row: dict[str, Any]) -> dict[str, Any]:
    """Recover a safely stated letter from a stored raw response, if possible."""

    if row.get("selected_letter") in LETTERS:
        return row
    selected_letter, parse_method = parse_question_only_choice(
        str(row.get("raw_output") or ""),
        display_options=list(row.get("display_options") or []),
    )
    if selected_letter is None:
        return row
    repaired = dict(row)
    selected_display_index = LETTERS.index(selected_letter)
    repaired.update(
        {
            "selected_letter": selected_letter,
            "selected_display_index": selected_display_index,
            "selected_canonical_index": row["display_to_canonical_index"][
                selected_display_index
            ],
            "selected_text": row["display_options"][selected_display_index],
            "selected_correct": selected_letter == row["correct_display_letter"],
            "parse_method": f"offline_{parse_method}",
            "reparsed_from_raw": True,
        }
    )
    return repaired


def _validate_qa(qa: dict[str, Any]) -> tuple[list[str], str, int]:
    question = str(qa.get("question") or "").strip()
    if not question:
        raise ValueError(f"{qa.get('qa_id')}: question is empty")
    raw_options = qa.get("options")
    if not isinstance(raw_options, list) or len(raw_options) != len(LETTERS):
        raise ValueError(f"{qa.get('qa_id')}: options must contain exactly five strings")
    options = [str(option or "").strip() for option in raw_options]
    if any(not option for option in options):
        raise ValueError(f"{qa.get('qa_id')}: options contain an empty value")
    if len({option.casefold() for option in options}) != len(options):
        raise ValueError(f"{qa.get('qa_id')}: options must be textually distinct")
    correct = normalize_correct(qa.get("correct"))
    correct_index = LETTERS.index(correct)
    if str(qa.get("answer") or "").strip() != options[correct_index]:
        raise ValueError(f"{qa.get('qa_id')}: answer does not equal options[correct]")
    final_decision = (qa.get("review") or {}).get("final_decision") or {}
    if final_decision.get("accepted") is not True:
        raise ValueError(f"{qa.get('qa_id')}: source QA is not marked accepted")
    return options, correct, correct_index


def prepare_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Validate the cohort and materialize the exact 85-call plan without API use."""

    annotation_rows = _read_annotations(config["annotations"])
    annotations_by_id = {str(row["qa_id"]).strip(): row for row in annotation_rows}
    manual_pass_ids = {
        qa_id
        for qa_id, row in annotations_by_id.items()
        if str(row.get("review_status") or "").strip() == config["manual_pass_value"]
    }
    expected_ids = set(config["expected_qa_ids"])
    if manual_pass_ids != expected_ids:
        missing = sorted(expected_ids - manual_pass_ids)
        unexpected = sorted(manual_pass_ids - expected_ids)
        raise ValueError(
            "manual Pass cohort does not match the frozen 17-QA list; "
            f"missing={missing} unexpected={unexpected}"
        )
    if len(manual_pass_ids) != config["expected_qa_count"]:
        raise ValueError(
            f"expected {config['expected_qa_count']} manual Pass rows, found {len(manual_pass_ids)}"
        )

    qas_by_id = _read_qas(config["accepted_qa"])
    missing_qas = sorted(expected_ids - set(qas_by_id))
    if missing_qas:
        raise ValueError(f"accepted QA bank is missing manual Pass IDs: {missing_qas}")

    selected_rows: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for question_index, qa_id in enumerate(config["expected_qa_ids"], 1):
        qa = qas_by_id[qa_id]
        options, original_correct, correct_index = _validate_qa(qa)
        selected_rows.append(
            {
                "question_index": question_index,
                "qa_id": qa_id,
                "evidence_id": qa.get("evidence_id"),
                "question": str(qa["question"]).strip(),
                "canonical_options": options,
                "original_correct_letter": original_correct,
                "correct_canonical_index": correct_index,
                "correct_answer": options[correct_index],
                "manual_review_status": annotations_by_id[qa_id]["review_status"],
                "reviewer_notes": annotations_by_id[qa_id].get("reviewer_notes", ""),
            }
        )
        for rotation_index, target_letter in enumerate(config["correct_positions"], 1):
            rotation = rotate_options(
                options,
                original_correct_letter=original_correct,
                target_correct_letter=target_letter,
            )
            prompt = build_question_only_prompt(str(qa["question"]), rotation["options"])
            calls.append(
                {
                    "call_key": f"{qa_id}::correct_at_{target_letter}",
                    "question_index": question_index,
                    "rotation_index": rotation_index,
                    "rotation_id": f"correct_at_{target_letter}",
                    "qa_id": qa_id,
                    "evidence_id": qa.get("evidence_id"),
                    "question": str(qa["question"]).strip(),
                    "canonical_options": options,
                    "display_options": rotation["options"],
                    "display_to_canonical_index": rotation["display_to_canonical_index"],
                    "canonical_to_display_index": rotation["canonical_to_display_index"],
                    "original_correct_letter": original_correct,
                    "correct_canonical_index": correct_index,
                    "correct_answer": options[correct_index],
                    "correct_display_letter": target_letter,
                    "cyclic_shift": rotation["shift"],
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "media": {"image_paths": [], "video_paths": []},
                }
            )

    expected_calls = config["expected_qa_count"] * len(config["correct_positions"])
    if len(calls) != expected_calls:
        raise AssertionError(f"expected {expected_calls} calls, materialized {len(calls)}")
    for qa_id in config["expected_qa_ids"]:
        qa_calls = [row for row in calls if row["qa_id"] == qa_id]
        if {row["correct_display_letter"] for row in qa_calls} != set(LETTERS):
            raise AssertionError(f"{qa_id}: correct answer did not occupy A-E exactly once")
        for canonical_index in range(len(LETTERS)):
            occupied = {
                row["canonical_to_display_index"][canonical_index] for row in qa_calls
            }
            if occupied != set(range(len(LETTERS))):
                raise AssertionError(
                    f"{qa_id}: canonical option {canonical_index} did not occupy every letter"
                )

    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "protocol": {
            "name": "question_only_balanced_option_rotation",
            "version": CONFIG_VERSION,
            "media_inputs_per_call": 0,
            "rotations_per_question": len(LETTERS),
            "correct_positions": list(LETTERS),
            "rotation_method": config["rotation_method"],
            "decoding_mode": "greedy",
            "answer_repairs_or_retries": 0,
        },
        "created_at_utc": _utc_now(),
        "config_path": str(config["config_path"]),
        "annotations": str(config["annotations"]),
        "accepted_qa": str(config["accepted_qa"]),
        "output_dir": str(output_dir),
        "manual_pass_value": config["manual_pass_value"],
        "question_count": len(selected_rows),
        "expected_logical_api_call_count": len(calls),
        "expected_media_input_count": 0,
        "model": {
            "model_id": config["model_id"],
            "reasoning_effort": config["reasoning_effort"],
            "max_new_tokens": config["max_new_tokens"],
            "max_retries": config["max_retries"],
        },
        "qa_ids": list(config["expected_qa_ids"]),
    }
    write_json(output_dir / "benchmark_plan.json", plan)
    write_jsonl(output_dir / "selected_questions.jsonl", selected_rows)
    write_jsonl(output_dir / "rotation_plan.jsonl", calls)
    write_jsonl(
        output_dir / "prompts.jsonl",
        [
            {
                "call_key": row["call_key"],
                "qa_id": row["qa_id"],
                "rotation_id": row["rotation_id"],
                "prompt": row["prompt"],
                "media": row["media"],
            }
            for row in calls
        ],
    )
    return {"plan": plan, "questions": selected_rows, "calls": calls}


def _validate_question_only_model(
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    architecture = metadata.get("architecture") or {}
    modalities = list(architecture.get("input_modalities") or [])
    supported_parameters = list(metadata.get("supported_parameters") or [])
    reasoning = metadata.get("reasoning") if isinstance(metadata.get("reasoning"), dict) else {}
    if "text" not in modalities:
        raise ValueError(f"{config['model_id']} does not advertise text input")
    if not {"reasoning", "reasoning_effort"}.intersection(supported_parameters):
        raise ValueError(f"{config['model_id']} does not advertise reasoning control")
    supported_efforts = reasoning.get("supported_efforts")
    if (
        isinstance(supported_efforts, list)
        and config["reasoning_effort"] not in supported_efforts
        and not config["allow_effort_mapping"]
    ):
        raise ValueError(
            f"reasoning effort {config['reasoning_effort']!r} is not an exact supported effort "
            f"{supported_efforts}"
        )
    return {
        "id": metadata.get("id"),
        "canonical_slug": metadata.get("canonical_slug"),
        "name": metadata.get("name"),
        "input_modalities": modalities,
        "context_length": metadata.get("context_length"),
        "supported_parameters": supported_parameters,
        "reasoning": reasoning,
        "pricing": metadata.get("pricing"),
    }


def _existing_calls(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    indexed: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(path):
        key = str(row.get("call_key") or "").strip()
        if not key:
            raise ValueError(f"existing call row is missing call_key: {path}")
        if key in indexed:
            raise ValueError(f"existing calls contain duplicate call_key={key}")
        indexed[key] = row
    return indexed


def _verify_resume_row(
    existing: dict[str, Any],
    planned: dict[str, Any],
    config: dict[str, Any],
) -> None:
    expected = {
        "qa_id": planned["qa_id"],
        "rotation_id": planned["rotation_id"],
        "prompt_sha256": planned["prompt_sha256"],
        "correct_display_letter": planned["correct_display_letter"],
        "model_id": config["model_id"],
        "reasoning_effort": config["reasoning_effort"],
        "media_input_count": 0,
    }
    mismatches = {
        key: {"expected": value, "actual": existing.get(key)}
        for key, value in expected.items()
        if existing.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"resume row {planned['call_key']} does not match this plan: {mismatches}"
        )


def execute_experiment(
    config: dict[str, Any],
    *,
    resume: bool,
    api_key: str | None = None,
    check_model_catalog: bool = True,
    runner: Any | None = None,
) -> dict[str, Any]:
    """Execute missing calls and write a summary; no media arguments are passed."""

    prepared = prepare_experiment(config)
    calls_path = config["output_dir"] / "calls.jsonl"
    existing = _existing_calls(calls_path)
    planned_by_key = {row["call_key"]: row for row in prepared["calls"]}
    unknown = sorted(set(existing) - set(planned_by_key))
    if unknown:
        raise ValueError(f"existing calls are not part of this plan: {unknown}")
    for key, row in existing.items():
        _verify_resume_row(row, planned_by_key[key], config)
    if existing and not resume:
        raise FileExistsError(
            f"{calls_path} already contains {len(existing)} calls; pass --resume to continue"
        )

    effective_api_key = api_key or os.getenv("OPENROUTER_API_KEY")
    if runner is None:
        if not effective_api_key:
            raise RuntimeError("run requires --api-key or OPENROUTER_API_KEY")
        if check_model_catalog:
            metadata_by_id = fetch_openrouter_model_metadata(
                base_url=config["base_url"],
                arms=[{"model_id": config["model_id"]}],
                api_key=effective_api_key,
            )
            snapshot = _validate_question_only_model(
                config, metadata_by_id[config["model_id"]]
            )
            write_json(config["output_dir"] / "openrouter_model_snapshot.json", snapshot)
        runner = OpenRouterRunner(
            model_id=config["model_id"],
            base_url=config["base_url"],
            max_new_tokens=config["max_new_tokens"],
            timeout=config["timeout_seconds"],
            api_key=effective_api_key,
            allow_video_input=False,
            reasoning_effort=config["reasoning_effort"],
            max_retries=0,
        )

    for call_index, planned in enumerate(prepared["calls"], 1):
        if planned["call_key"] in existing:
            print(
                f"question_only_rotation_skip call={call_index}/{len(prepared['calls'])} "
                f"key={planned['call_key']}",
                flush=True,
            )
            continue
        print(
            f"question_only_rotation_start call={call_index}/{len(prepared['calls'])} "
            f"qa_id={planned['qa_id']} rotation={planned['rotation_id']} media=none",
            flush=True,
        )
        started_at = _utc_now()
        start = time.monotonic()
        # Deliberately omit image_paths and video_paths. This is the intervention.
        raw_output = runner.generate(planned["prompt"], decoding_mode="greedy")
        elapsed_seconds = time.monotonic() - start
        selected_letter, parse_method = parse_question_only_choice(
            raw_output,
            display_options=planned["display_options"],
        )
        selected_display_index = LETTERS.index(selected_letter) if selected_letter else None
        selected_canonical_index = (
            planned["display_to_canonical_index"][selected_display_index]
            if selected_display_index is not None
            else None
        )
        selected_text = (
            planned["display_options"][selected_display_index]
            if selected_display_index is not None
            else None
        )
        result = {
            **{key: value for key, value in planned.items() if key not in {"prompt", "media"}},
            "model_id": config["model_id"],
            "reasoning_effort": config["reasoning_effort"],
            "decoding_mode": "greedy",
            "media_input_count": 0,
            "selected_letter": selected_letter,
            "selected_display_index": selected_display_index,
            "selected_canonical_index": selected_canonical_index,
            "selected_text": selected_text,
            "selected_correct": selected_letter == planned["correct_display_letter"],
            "parse_method": parse_method,
            "raw_output": raw_output,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": round(elapsed_seconds, 6),
        }
        append_jsonl(calls_path, result)
        existing[planned["call_key"]] = result
        print(
            f"question_only_rotation_done call={call_index}/{len(prepared['calls'])} "
            f"key={planned['call_key']} choice={selected_letter or 'UNPARSED'} "
            f"correct={result['selected_correct']}",
            flush=True,
        )
    return summarize_experiment(config, prepared=prepared)


def _question_classification(rows: list[dict[str, Any]]) -> str:
    if len(rows) != len(LETTERS) or any(row.get("selected_letter") not in LETTERS for row in rows):
        return "incomplete_or_unparsed"
    correct_count = sum(bool(row.get("selected_correct")) for row in rows)
    letters = [str(row["selected_letter"]) for row in rows]
    canonical = [int(row["selected_canonical_index"]) for row in rows]
    if correct_count == len(LETTERS):
        return "question_only_semantically_solved"
    if len(set(letters)) == 1:
        return "always_A_position_bias" if letters[0] == "A" else "fixed_letter_position_bias"
    if len(set(canonical)) == 1:
        return "stable_wrong_semantic_option"
    return "mixed_or_unstable"


def summarize_experiment(
    config: dict[str, Any],
    *,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize completed calls without issuing any API requests."""

    prepared = prepared or prepare_experiment(config)
    calls_path = config["output_dir"] / "calls.jsonl"
    calls_by_key = _existing_calls(calls_path)
    per_question: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for question in prepared["questions"]:
        planned_rows = [
            row for row in prepared["calls"] if row["qa_id"] == question["qa_id"]
        ]
        rows = [
            _reparse_call_result(calls_by_key[row["call_key"]])
            for row in planned_rows
            if row["call_key"] in calls_by_key
        ]
        all_rows.extend(rows)
        letter_counts = Counter(row.get("selected_letter") or "UNPARSED" for row in rows)
        canonical_counts = Counter(
            str(row["selected_canonical_index"])
            if row.get("selected_canonical_index") is not None
            else "UNPARSED"
            for row in rows
        )
        correct_count = sum(bool(row.get("selected_correct")) for row in rows)
        per_question.append(
            {
                **question,
                "completed_call_count": len(rows),
                "parsed_call_count": sum(row.get("selected_letter") in LETTERS for row in rows),
                "correct_count": correct_count,
                "accuracy": correct_count / len(LETTERS),
                "correct_all_five_positions": correct_count == len(LETTERS) and len(rows) == len(LETTERS),
                "selected_letter_counts": dict(sorted(letter_counts.items())),
                "selected_canonical_index_counts": dict(sorted(canonical_counts.items())),
                "choices_by_correct_position": {
                    row["correct_display_letter"]: row.get("selected_letter") for row in rows
                },
                "classification": _question_classification(rows),
            }
        )

    position_rows: list[dict[str, Any]] = []
    for letter in LETTERS:
        rows = [row for row in all_rows if row.get("correct_display_letter") == letter]
        position_rows.append(
            {
                "correct_position": letter,
                "expected_call_count": config["expected_qa_count"],
                "completed_call_count": len(rows),
                "parsed_call_count": sum(row.get("selected_letter") in LETTERS for row in rows),
                "correct_count": sum(bool(row.get("selected_correct")) for row in rows),
                "accuracy": (
                    sum(bool(row.get("selected_correct")) for row in rows) / len(rows)
                    if rows
                    else None
                ),
                "selected_letter_counts": dict(
                    sorted(Counter(row.get("selected_letter") or "UNPARSED" for row in rows).items())
                ),
            }
        )

    expected_call_count = config["expected_qa_count"] * len(LETTERS)
    correct_total = sum(bool(row.get("selected_correct")) for row in all_rows)
    parsed_total = sum(row.get("selected_letter") in LETTERS for row in all_rows)
    calls_complete = len(all_rows) == expected_call_count
    fully_parsed = parsed_total == expected_call_count
    classifications = Counter(row["classification"] for row in per_question)
    correct_histogram = Counter(row["correct_count"] for row in per_question)
    summary = {
        "protocol": prepared["plan"]["protocol"],
        "summarized_at_utc": _utc_now(),
        "model_id": config["model_id"],
        "reasoning_effort": config["reasoning_effort"],
        "question_count": config["expected_qa_count"],
        "expected_call_count": expected_call_count,
        "completed_call_count": len(all_rows),
        "missing_call_count": expected_call_count - len(all_rows),
        "parsed_call_count": parsed_total,
        "unparsed_call_count": len(all_rows) - parsed_total,
        "unparsed_call_keys": [
            row["call_key"] for row in all_rows if row.get("selected_letter") not in LETTERS
        ],
        "reparsed_from_raw_call_count": sum(
            row.get("reparsed_from_raw") is True for row in all_rows
        ),
        "correct_call_count": correct_total,
        "call_accuracy": correct_total / len(all_rows) if all_rows else None,
        "uniform_random_accuracy_baseline": 1 / len(LETTERS),
        "correct_count_per_question_histogram": {
            str(key): value for key, value in sorted(correct_histogram.items())
        },
        "question_classification_counts": dict(sorted(classifications.items())),
        "question_only_semantically_solved_count": classifications.get(
            "question_only_semantically_solved", 0
        ),
        "always_A_position_bias_count": classifications.get("always_A_position_bias", 0),
        "selected_letter_counts": dict(
            sorted(Counter(row.get("selected_letter") or "UNPARSED" for row in all_rows).items())
        ),
        "correct_by_display_position": position_rows,
        "calls_complete": calls_complete,
        "fully_parsed": fully_parsed,
        # Completion refers to the paid call package. Parsing is reported separately
        # and must not trigger retries or turn a completed run into a failed job.
        "complete": calls_complete,
    }
    write_json(config["output_dir"] / "summary.json", summary)
    write_jsonl(config["output_dir"] / "per_question.jsonl", per_question)
    write_jsonl(
        config["output_dir"] / "non_five_of_five.jsonl",
        [row for row in per_question if not row["correct_all_five_positions"]],
    )
    with (config["output_dir"] / "per_question.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "question_index",
            "qa_id",
            "question",
            "correct_answer",
            "original_correct_letter",
            "completed_call_count",
            "parsed_call_count",
            "correct_count",
            "accuracy",
            "correct_all_five_positions",
            "classification",
            "selected_letter_counts",
            "choices_by_correct_position",
            "reviewer_notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_question:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), dict)
                    else row.get(key)
                    for key in fieldnames
                }
            )
    with (config["output_dir"] / "position_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "correct_position",
            "expected_call_count",
            "completed_call_count",
            "parsed_call_count",
            "correct_count",
            "accuracy",
            "selected_letter_counts",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in position_rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False, sort_keys=True)
                    if isinstance(row.get(key), dict)
                    else row.get(key)
                    for key in fieldnames
                }
            )
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a text-only MCQ intervention where each correct semantic answer occupies "
            "A, B, C, D, and E exactly once."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "run", "summarize"):
        child = subparsers.add_parser(command)
        child.add_argument("--config", required=True)
        child.add_argument("--annotations")
        child.add_argument("--output-dir")
        if command == "run":
            child.add_argument("--api-key")
            child.add_argument("--resume", action="store_true")
            child.add_argument("--skip-model-catalog-check", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(
        args.config,
        annotations_override=args.annotations,
        output_dir_override=args.output_dir,
    )
    if args.command == "validate":
        prepared = prepare_experiment(config)
        print(json.dumps(prepared["plan"], ensure_ascii=False, indent=2))
        return 0
    if args.command == "run":
        summary = execute_experiment(
            config,
            resume=args.resume,
            api_key=args.api_key,
            check_model_catalog=not args.skip_model_catalog_check,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    summary = summarize_experiment(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
